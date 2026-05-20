#!/usr/bin/env python3

import os
import sys
import subprocess
import time


try:
    from mininet.net import Mininet
    from mininet.topo import Topo
    from mininet.log import setLogLevel, info
    from mininet.cli import CLI
    from mininet.link import TCLink
    MININET_AVAILABLE = True
except ImportError:
    MININET_AVAILABLE = False

try:
    from p4utils.mininetlib.network_API import NetworkAPI
    P4UTILS_AVAILABLE = True
except ImportError:
    P4UTILS_AVAILABLE = False



def run_p4utils_topology(p4_dir: str, log_dir: str):
    net = NetworkAPI()

    # ── Hosts ──────────────────────────────────────────────────────────────
    net.addHost('h1')
    net.addHost('h2')
    

    # ── Switches (BMv2 simple_switch_grpc) ─────────────────────────────────
    for sw_idx, sw_name in enumerate(['s1', 's2', 's3', 's4'], start=1):
        json_path = os.path.join(p4_dir, f'{sw_name}.json')
        p4info_path = os.path.join(p4_dir, 'p4info.txt')
        net.addP4RuntimeSwitch(
            sw_name,
            cli_input=None,
            log_enabled=True,
            log_dir=log_dir,
            pcap_dump=False,
            thrift_port=9090 + sw_idx,   
            grpc_port=50050 + sw_idx,
            device_id=sw_idx - 1,
            switch_exe='simple_switch_grpc',
            p4_src=os.path.join(p4_dir, 'jitter_router.p4'),
            json_path=json_path,
            p4info_path=p4info_path,
        )

    # Host links
    net.addLink('h1', 's1', port2=1)
    net.addLink('h2', 's4', port2=1)
    

    # Path 1: s1 port2 ↔ s2 port1,  s2 port2 ↔ s4 port2
    net.addLink('s1', 's2', port1=2, port2=1)
    net.addLink('s2', 's4', port1=2, port2=2)

    # Path 2: s1 port3 ↔ s3 port1,  s3 port2 ↔ s4 port3
    net.addLink('s1', 's3', port1=3, port2=1)
    net.addLink('s3', 's4', port1=2, port2=3)

    # ── IP assignment ───────────────────────────────────────────────────────
    net.setIntfIp('h1', 's1', '10.0.0.1/24')
    net.setIntfIp('h2', 's4', '10.0.0.2/24')

    net.setIntfMac('h1', 's1', '00:00:00:00:00:01')
    net.setIntfMac('h2', 's4', '00:00:00:00:00:02')

    net.enablePcapDumpAll(log_dir)
    net.disableArpTables()
    net.disableGwArp()

    net.startNetwork()
    # ── Static ARP setup ───────────────────────────────────────────────
    h1 = net.net.get('h1')
    h2 = net.net.get('h2')

    h1.cmd('arp -s 10.0.0.2 00:00:00:00:00:02') 
    h2.cmd('arp -s 10.0.0.1 00:00:00:00:00:01')
    return net



class DualPathTopo(Topo):
    def build(self):
        h1 = self.addHost('h1', ip='10.0.0.1/24')
        h2 = self.addHost('h2', ip='10.0.0.2/24')
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        s3 = self.addSwitch('s3')
        s4 = self.addSwitch('s4')

        self.addLink(h1, s1)
        self.addLink(h2, s4)
        self.addLink(s1, s2)
        self.addLink(s1, s3)
        self.addLink(s2, s4)
        self.addLink(s3, s4)


def run_plain_mininet():
    setLogLevel('info')
    topo = DualPathTopo()
    net = Mininet(topo=topo, link=TCLink)
    net.start()
    info('\n*** Plain Mininet topology started\n')
    CLI(net)
    net.stop()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Launch dual path topology')
    parser.add_argument('--p4dir',  default='p4',   help='Path to compiled P4 artifacts')
    parser.add_argument('--logdir', default='logs', help='Directory for switch logs')
    parser.add_argument('--plain',  action='store_true', help='Use plain Mininet')
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    if args.plain or not P4UTILS_AVAILABLE:
        if not MININET_AVAILABLE:
            print('[ERROR] Neither p4-utils nor plain Mininet is installed.')
            sys.exit(1)
        print('[WARN] p4-utils not found.')
        run_plain_mininet()
    else:
        net = run_p4utils_topology(
            p4_dir=os.path.abspath(args.p4dir),
            log_dir=os.path.abspath(args.logdir),
        )
        CLI(net)
        net.stopNetwork()
