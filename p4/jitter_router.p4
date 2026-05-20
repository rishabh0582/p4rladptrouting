#include <core.p4>
#include <v1model.p4>

/* ---------------------------------------------------------------------------- *
 * Compile-time constants                                                        *
 * ---------------------------------------------------------------------------- */
#ifndef SWITCH_ID
#define SWITCH_ID 1
#endif

#define MAX_HOPS 4
#define INT_TYPE  0x1234   
#define IPV4_TYPE 0x0800

/* ---------------------------------------------------------------------------- *
 * Header definitions                                                            *
 * ---------------------------------------------------------------------------- */

header ethernet_t {
    bit<48> dstAddr;
    bit<48> srcAddr;
    bit<16> etherType;
}

header ipv4_t {
    bit<4>  version;
    bit<4>  ihl;
    bit<8>  diffserv;
    bit<16> totalLen;
    bit<16> identification;
    bit<3>  flags;
    bit<13> fragOffset;
    bit<8>  ttl;
    bit<8>  protocol;
    bit<16> hdrChecksum;
    bit<32> srcAddr;
    bit<32> dstAddr;
}
header arp_t {
    bit<16> htype;
    bit<16> ptype;
    bit<8>  hlen;
    bit<8>  plen;
    bit<16> oper;
    bit<48> sha;
    bit<32> spa;
    bit<48> tha;
    bit<32> tpa;
}

header int_base_t {
    bit<8> hop_count;  
}

header int_hop_t {
    bit<16> switch_id;
    bit<48> ingress_timestamp;
    bit<19> queue_depth;
    bit<5>  pad;
}

/* ---------------------------------------------------------------------------- *
 * Digest structure exported to controller                                       *
 * ---------------------------------------------------------------------------- */
struct telemetry_digest_t {
    bit<16> switch_id;
    bit<48> ingress_timestamp;
    bit<19> queue_depth;
    bit<5>  pad;
    bit<32> src_ip;
    bit<32> dst_ip;
    bit<9>  egress_port;
}

/* ---------------------------------------------------------------------------- *
 * Aggregated header stack                                                       *
 * ---------------------------------------------------------------------------- */
struct headers_t {
    ethernet_t              ethernet;
    ipv4_t                  ipv4;
    int_base_t              int_base;
    int_hop_t[MAX_HOPS]     int_hops;
    arp_t arp;
}

struct metadata_t {
    bit<9>  egress_port_md;   
    bit<1>  do_telemetry;
}

/* ---------------------------------------------------------------------------- *
 * Parser                                                                        *
 * ---------------------------------------------------------------------------- */
parser MyParser(packet_in pkt,
                out headers_t hdr,
                inout metadata_t meta,
                inout standard_metadata_t std_meta) {

    state start {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            0x0800 : parse_ipv4;
            0x0806 : parse_arp;
            default : accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition accept;
    }
    state parse_arp {
        pkt.extract(hdr.arp);
        transition accept;
    }
}

/* ---------------------------------------------------------------------------- *
 * Checksum verification (no-op for experimentation)                             *
 * ---------------------------------------------------------------------------- */
control MyVerifyChecksum(inout headers_t hdr, inout metadata_t meta) {
    apply { }
}

/* ---------------------------------------------------------------------------- *
 * Ingress control                                                               *
 * ---------------------------------------------------------------------------- */
control MyIngress(inout headers_t hdr,
                  inout metadata_t meta,
                  inout standard_metadata_t std_meta) {


    action set_egress_port(bit<9> port, bit<48> dst_mac,
        bit<48> src_mac) {
        std_meta.egress_spec = port;
        hdr.ethernet.dstAddr = dst_mac;
        hdr.ethernet.srcAddr = src_mac;
        meta.egress_port_md  = port;
        meta.do_telemetry    = 1;
    }

    action drop_pkt() {
        mark_to_drop(std_meta);
        meta.do_telemetry = 0;
    }
    
    
    table ipv4_lpm {
        key = {
            hdr.ipv4.dstAddr : lpm;
        }
        actions = {
            set_egress_port;
            drop_pkt;
            NoAction;
        }
        size = 1024;
        default_action = drop_pkt();
    }
    
    apply {
        if (hdr.arp.isValid()) {
            if (std_meta.ingress_port == 1) {
                std_meta.egress_spec = 2;
            } else if (std_meta.ingress_port == 2) {
                std_meta.egress_spec = 1;
            } else if (std_meta.ingress_port == 3) {
                std_meta.egress_spec = 1;
            }
            exit;
        }
        if (!hdr.ipv4.isValid()) {
            exit;
        }

        /* Decrement TTL */
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
        if (hdr.ipv4.ttl == 0) {
            drop_pkt();
            exit;
        }

        ipv4_lpm.apply();

        /* Export queue depth and timestamp to controller */
        if (meta.do_telemetry == 1){
            

            telemetry_digest_t digest_data;
            digest_data.switch_id          = (bit<16>)SWITCH_ID;
            digest_data.ingress_timestamp  = std_meta.ingress_global_timestamp;
            digest_data.queue_depth        = std_meta.enq_qdepth;
            digest_data.pad                = 0;
            digest_data.src_ip             = hdr.ipv4.srcAddr;
            digest_data.dst_ip             = hdr.ipv4.dstAddr;
            digest_data.egress_port        = std_meta.egress_spec;
            digest(1, digest_data);
        }
    }
}

/* ---------------------------------------------------------------------------- *
 * Egress control (pass-through)                                                 *
 * ---------------------------------------------------------------------------- */
control MyEgress(inout headers_t hdr,
                 inout metadata_t meta,
                 inout standard_metadata_t std_meta) {
    apply { }
}

/* ---------------------------------------------------------------------------- *
 * Checksum computation                                                          *
 * ---------------------------------------------------------------------------- */
control MyComputeChecksum(inout headers_t hdr, inout metadata_t meta) {
    apply {
        update_checksum(
            hdr.ipv4.isValid(),
            { hdr.ipv4.version, hdr.ipv4.ihl, hdr.ipv4.diffserv,
              hdr.ipv4.totalLen, hdr.ipv4.identification,
              hdr.ipv4.flags, hdr.ipv4.fragOffset, hdr.ipv4.ttl,
              hdr.ipv4.protocol, hdr.ipv4.srcAddr, hdr.ipv4.dstAddr },
            hdr.ipv4.hdrChecksum,
            HashAlgorithm.csum16
        );
    }
}

/* ---------------------------------------------------------------------------- *
 * Deparser                                                                      *
 * ---------------------------------------------------------------------------- */
control MyDeparser(packet_out pkt, in headers_t hdr) {
    apply {
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.arp);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.int_base);
        pkt.emit(hdr.int_hops[0]);
        pkt.emit(hdr.int_hops[1]);
        pkt.emit(hdr.int_hops[2]);
        pkt.emit(hdr.int_hops[3]);
    }
}

/* ---------------------------------------------------------------------------- *
 * Switch instantiation                                                          *
 * ---------------------------------------------------------------------------- */
V1Switch(
    MyParser(),
    MyVerifyChecksum(),
    MyIngress(),
    MyEgress(),
    MyComputeChecksum(),
    MyDeparser()
) main;
