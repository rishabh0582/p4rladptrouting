#!/usr/bin/env python3

import threading
import queue
import time
import logging
import ipaddress
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict

logger = logging.getLogger('controller')

try:
    import grpc
    from p4.v1 import p4runtime_pb2
    from p4.v1 import p4runtime_pb2_grpc
    from p4.config.v1 import p4info_pb2
    import google.protobuf.text_format as text_format
    P4RUNTIME_AVAILABLE = True
except ImportError:
    P4RUNTIME_AVAILABLE = False
    logger.warning('P4Runtime libraries not installed, running in SIMULATION mode.')



@dataclass
class TelemetryRecord:
    switch_id:         int
    ingress_timestamp: int      # microseconds
    queue_depth:       int      # packets
    src_ip:            str
    dst_ip:            str
    egress_port:       int
    received_at:       float = field(default_factory=time.time)


@dataclass
class SwitchConfig:
    name:        str
    grpc_addr:   str
    device_id:   int
    json_path:   str
    p4info_path: str


DEFAULT_SWITCHES = [
    SwitchConfig('s1', 'localhost:50051', 0, 'p4/s1.json', 'p4/p4info.txt'),
    SwitchConfig('s2', 'localhost:50052', 1, 'p4/s2.json', 'p4/p4info.txt'),
    SwitchConfig('s3', 'localhost:50053', 2, 'p4/s3.json', 'p4/p4info.txt'),
    SwitchConfig('s4', 'localhost:50054', 3, 'p4/s4.json', 'p4/p4info.txt'),
]

PATH1_RULES: Dict[str, List] = {

    's1': [
        ('10.0.0.1', 32, 1,
         '00:00:00:00:00:01',
         '00:aa:00:00:01:01'),

        ('10.0.0.2', 32, 2,
         '00:00:00:00:00:02',
         '00:aa:00:00:01:02'),
    ],

    's2': [
        ('10.0.0.1', 32, 1,
         '00:00:00:00:00:01',
         '00:aa:00:00:02:01'),

        ('10.0.0.2', 32, 2,
         '00:00:00:00:00:02',
         '00:aa:00:00:02:02'),
    ],

    's3': [
        ('10.0.0.1', 32, 1,
         '00:00:00:00:00:01',
         '00:aa:00:00:03:01'),

        ('10.0.0.2', 32, 2,
         '00:00:00:00:00:02',
         '00:aa:00:00:03:02'),
    ],

    's4': [
        ('10.0.0.1', 32, 2,
         '00:00:00:00:00:01',
         '00:aa:00:00:04:02'),

        ('10.0.0.2', 32, 1,
         '00:00:00:00:00:02',
         '00:aa:00:00:04:01'),
    ],
}
PATH2_S1_PORT = 3

_HISTORY_LEN = 50



class P4RuntimeClient:
    def __init__(self, cfg: SwitchConfig):
        self.cfg = cfg
        self._stub = None
        self._p4info = None
        self._table_ids: Dict[str, int] = {}
        self._action_ids: Dict[str, int] = {}
        self._param_ids: Dict[str, Dict[str, int]] = {}

    def connect(self):
        channel = grpc.insecure_channel(self.cfg.grpc_addr)
        self._stub = p4runtime_pb2_grpc.P4RuntimeStub(channel)

        self._req_queue = queue.Queue()

        self._stream = self._stub.StreamChannel(
            self._stream_iter()
        )

        arb = p4runtime_pb2.StreamMessageRequest()
        arb.arbitration.device_id = self.cfg.device_id
        arb.arbitration.election_id.high = 0
        arb.arbitration.election_id.low = 1

        self._req_queue.put(arb)


        while True:
            response = next(self._stream)

            if response.HasField("arbitration"):
                break

        self._digest_records = queue.Queue()

        self._stream_thread = threading.Thread(
            target=self._stream_listener,
            daemon=True
        )
        self._stream_thread.start()

        with open(self.cfg.p4info_path, 'r') as f:
            p4info_txt = f.read()

        self._p4info = p4info_pb2.P4Info()
        text_format.Parse(p4info_txt, self._p4info)

        self._build_id_maps()
        self._push_pipeline()

        logger.info('[%s] Connected and pipeline loaded.',
                    self.cfg.name)
        
    def _stream_listener(self):
        try:
            for response in self._stream:

                print(f'[{self.cfg.name}] STREAM MESSAGE')

                if response.HasField("digest"):
                    d = response.digest

                    print(
                        f'[{self.cfg.name}] DIGEST '
                        f'id={d.digest_id} '
                        f'entries={len(d.data)}'
                    )

                    ack = p4runtime_pb2.StreamMessageRequest()
                    ack.digest_ack.digest_id = d.digest_id
                    ack.digest_ack.list_id = d.list_id

                    self._req_queue.put(ack)

                    for entry in d.data:
                        rec = self._parse_digest(entry)

                        if rec:
                            print(
                                f'[{self.cfg.name}] '
                                f'PARSED DIGEST {rec}'
                            )

                            self._digest_records.put(rec)

        except Exception as e:
            print(
                f'[{self.cfg.name}] '
                f'STREAM ERROR: {e}'
            )

    def _build_id_maps(self):
        self._match_field_ids = {}
        for table in self._p4info.tables:
            self._table_ids[table.preamble.name] = table.preamble.id
            self._match_field_ids[table.preamble.name] = {
                mf.name: mf.id for mf in table.match_fields
            }

        for action in self._p4info.actions:
            self._action_ids[action.preamble.name] = action.preamble.id

            self._param_ids[action.preamble.name] = {
                p.name: p.id for p in action.params
            }
        
    def _stream_iter(self):
        while True:
            try:
                req = self._req_queue.get(timeout=1)
                yield req
            except queue.Empty:
                continue
    

    def _push_pipeline(self):
        with open(self.cfg.p4info_path, 'r') as f:
            p4info_content = f.read()
        with open(self.cfg.json_path, 'rb') as f:
            device_config = f.read()
        p4info_obj = p4info_pb2.P4Info()
        text_format.Parse(p4info_content, p4info_obj)
        req = p4runtime_pb2.SetForwardingPipelineConfigRequest()
        req.device_id = self.cfg.device_id
        req.election_id.high = 0
        req.election_id.low = 1
        req.action = p4runtime_pb2.SetForwardingPipelineConfigRequest.VERIFY_AND_COMMIT
        req.config.p4info.CopyFrom(p4info_obj)
        req.config.p4_device_config = device_config
        self._stub.SetForwardingPipelineConfig(req)

    def install_ipv4_lpm(self, ip: str, prefix_len: int, port: int, dst_mac: str, src_mac: str):
        self._write_lpm(ip, prefix_len, port, dst_mac, src_mac, p4runtime_pb2.Update.INSERT)

    def update_ipv4_lpm(self, ip: str, prefix_len: int, port: int, dst_mac: str, src_mac: str):
        self._write_lpm(ip, prefix_len, port, dst_mac, src_mac, p4runtime_pb2.Update.MODIFY)

    def _write_lpm(self, ip, prefix_len, port, dst_mac, src_mac, update_type):
        table_id      = self._table_ids['MyIngress.ipv4_lpm']
        action_id     = self._action_ids['MyIngress.set_egress_port']
        port_param_id = self._param_ids['MyIngress.set_egress_port']['port']
        dst_mac_param_id = self._param_ids['MyIngress.set_egress_port']['dst_mac']
        src_mac_param_id = self._param_ids['MyIngress.set_egress_port']['src_mac']
        ip_bytes      = int(ipaddress.IPv4Address(ip)).to_bytes(4, 'big')

        req = p4runtime_pb2.WriteRequest()
        req.device_id = self.cfg.device_id
        req.election_id.high = 0
        req.election_id.low = 1

        update = req.updates.add()
        update.type = update_type
        entry = update.entity.table_entry
        entry.table_id = table_id
        mf = entry.match.add()
        mf.field_id = self._match_field_ids['MyIngress.ipv4_lpm']['hdr.ipv4.dstAddr']
        mf.lpm.value = ip_bytes
        mf.lpm.prefix_len = prefix_len
        entry.action.action.action_id = action_id
        param = entry.action.action.params.add()
        param.param_id = port_param_id
        param.value = bytes([port])
        param = entry.action.action.params.add()
        param.param_id = dst_mac_param_id
        param.value = bytes.fromhex(dst_mac.replace(':', ''))

        param = entry.action.action.params.add()
        param.param_id = src_mac_param_id
        param.value = bytes.fromhex(src_mac.replace(':', ''))

        try:
            self._stub.Write(req)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.ALREADY_EXISTS and update_type == p4runtime_pb2.Update.INSERT:
                update.type = p4runtime_pb2.Update.MODIFY
                self._stub.Write(req)
            else:
                raise

    def read_digests(self) -> List[TelemetryRecord]:
        records = client.read_digests()

        while True:
            try:
                records.append(
                    self._digest_records.get_nowait()
                )
            except queue.Empty:
                break

        return records

    def _parse_digest(self, digest_data) -> Optional[TelemetryRecord]:
        try:
            fields = digest_data.struct.members
            switch_id   = int.from_bytes(fields[0].bitstring, 'big')
            ts          = int.from_bytes(fields[1].bitstring, 'big')
            qd_raw      = int.from_bytes(fields[2].bitstring, 'big')
            queue_depth = qd_raw >> 5
            src_ip      = str(ipaddress.IPv4Address(fields[4].bitstring))
            dst_ip      = str(ipaddress.IPv4Address(fields[5].bitstring))
            eport       = int.from_bytes(fields[6].bitstring, 'big')
            return TelemetryRecord(switch_id, ts, queue_depth, src_ip, dst_ip, eport)
        except Exception as e:
            logger.debug('Digest parse error: %s', e)
            return None



class SimulatedP4Client:
    def __init__(self, cfg: SwitchConfig, congested: bool = False):
        self.cfg       = cfg
        self.congested = congested
        self._t        = 0

    def connect(self):
        logger.info('[%s] Simulation mode.', self.cfg.name)

    def install_ipv4_lpm(self, *a, **kw): pass
    def update_ipv4_lpm(self, *a, **kw): pass

    def read_digests(self) -> List[TelemetryRecord]:
        import random
        self._t += 1


        cycle = (self._t // 50) % 4
        sw_name = self.cfg.name

        if cycle == 0:                              
            qd           = random.randint(0, 5)
            jitter_noise = random.randint(-30, 30)
        elif cycle == 1:                            
            if sw_name == 's2':
                qd           = random.randint(60, 90)
                jitter_noise = random.randint(-300, 300)
            else:
                qd           = random.randint(0, 5)
                jitter_noise = random.randint(-20, 20)
        elif cycle == 2:                            
            if sw_name == 's3':
                qd           = random.randint(60, 90)
                jitter_noise = random.randint(-300, 300)
            else:
                qd           = random.randint(0, 5)
                jitter_noise = random.randint(-20, 20)
        else:                                       
            qd           = random.randint(20, 40)
            jitter_noise = random.randint(-100, 100)

        base_interval = 1000   
        ts = int(time.time() * 1e6) + self._t * base_interval + jitter_noise
        return [TelemetryRecord(
            switch_id=self.cfg.device_id + 1,
            ingress_timestamp=ts,
            queue_depth=qd,
            src_ip='10.0.0.1', dst_ip='10.0.0.2', egress_port=2,
        )]


# ---------------------------------------------------------------------------
# Main Controller
# ---------------------------------------------------------------------------
class Controller:
    def __init__(self,
                 switch_configs: Optional[List[SwitchConfig]] = None,
                 simulation: bool = False,
                 poll_interval_ms: int = 100):

        self.switch_configs = switch_configs or DEFAULT_SWITCHES
        self.simulation     = simulation or (not P4RUNTIME_AVAILABLE)
        self.poll_interval  = poll_interval_ms / 1000.0

        self._clients:      Dict[str, object] = {}
        self._current_path  = 1
        self._path_lock     = threading.Lock()

        self._telemetry_q   = queue.Queue(maxsize=50000)

        self._running       = False
        self._poll_thread   = None

        self._latest: Dict[int, TelemetryRecord] = {}

        self._sw_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=_HISTORY_LEN))

        self._prev_q_s2 = 0
        self._prev_q_s3 = 0

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #
    def start(self):
        for cfg in self.switch_configs:
            if self.simulation:
                client = SimulatedP4Client(cfg, congested=(cfg.name == 's2'))
            else:
                client = P4RuntimeClient(cfg)
            client.connect()
            self._clients[cfg.name] = client

        if not self.simulation:
            self._install_all_rules()

        self._running    = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name='telemetry-poll')
        self._poll_thread.start()
        logger.info('Controller started (simulation=%s, path=%d)',
                    self.simulation, self._current_path)

    def stop(self):
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=2)

    def set_path(self, path: int):
        self.update_path(path)

    # ------------------------------------------------------------------ #
    # Rule installation                                                    #
    # ------------------------------------------------------------------ #
    def _install_all_rules(self):
        for sw_name, rules in PATH1_RULES.items():
            client = self._clients[sw_name]

            for ip, pfx, port, dst_mac, src_mac in rules:
                
                try:
                    client.install_ipv4_lpm(ip, pfx, port, dst_mac, src_mac)

                    logger.debug(
                        '[%s] Rule %s/%d → port %d',
                        sw_name,
                        ip,
                        pfx,
                        port
                    )

                except Exception as e:
                    logger.error(
                        '[%s] Rule install failed: %s',
                        sw_name,
                        e
                    )

    # ------------------------------------------------------------------ #
    # Path switching                                                       #
    # ------------------------------------------------------------------ #
    def update_path(self, path: int):
        with self._path_lock:
            if path == self._current_path:
                return
            port = 2 if path == 1 else PATH2_S1_PORT
            if not self.simulation:
                try:
                    self._clients['s1'].update_ipv4_lpm('10.0.0.2', 32, port, '00:00:00:00:00:02', '00:aa:00:00:01:02' if path == 1 else '00:aa:00:00:01:03')
                    logger.info('Path switched → %d (s1 port %d)', path, port)
                except Exception as e:
                    logger.error('Path switch failed: %s', e)
                    return
            self._current_path = path
            logger.info('Active path: %d', self._current_path)

    def get_current_path(self) -> int:
        with self._path_lock:
            return self._current_path

    # ------------------------------------------------------------------ #
    # Telemetry polling                                                    #
    # ------------------------------------------------------------------ #
    def _poll_loop(self):
        while self._running:
            for sw_name, client in self._clients.items():
                try:
                    records = []
                    for rec in records:
                        self._latest[rec.switch_id] = rec
                        self._sw_history[rec.switch_id].append(rec)
                        try:
                            self._telemetry_q.put_nowait(rec)
                        except queue.Full:
                            self._telemetry_q.get_nowait()   
                            self._telemetry_q.put_nowait(rec)
                except Exception as e:
                    logger.debug('[%s] Poll error: %s', sw_name, e)
            time.sleep(self.poll_interval)


    def _estimate_jitter_for_switch(self, switch_id: int) -> float:
        
        records    = list(self._sw_history[switch_id])
        timestamps = [r.ingress_timestamp for r in records if r.ingress_timestamp > 0]
        if len(timestamps) < 3:
            return 0.0
        delays = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        delays = [d for d in delays if 0 < d < 1_000_000]   
        if len(delays) < 2:
            return 0.0
        jitter_vals = [abs(delays[i] - delays[i - 1]) for i in range(1, len(delays))]
        return sum(jitter_vals) / len(jitter_vals)

    def _estimate_jitter(self) -> float:
       
        j2 = self._estimate_jitter_for_switch(2)
        j3 = self._estimate_jitter_for_switch(3)
        active = self.get_current_path()
        if active == 1:
            return j2 if j2 > 0 else j3
        return j3 if j3 > 0 else j2


    def _estimate_delay_for_switch(self, switch_id: int) -> float:
        records    = list(self._sw_history[switch_id])
        timestamps = [r.ingress_timestamp for r in records if r.ingress_timestamp > 0]
        if len(timestamps) < 2:
            return 0.0
        delays = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        delays = [d for d in delays if 0 < d < 1_000_000]
        if not delays:
            return 0.0
        return sum(delays) / len(delays)

    def _estimate_delay(self) -> float:
        active = self.get_current_path()
        sw_id  = 2 if active == 1 else 3
        return self._estimate_delay_for_switch(sw_id)


    def get_network_state(self) -> Dict:
        
        q_s2_rec = self._latest.get(1)
        q_s3_rec = self._latest.get(2)
        queue_s2 = q_s2_rec.queue_depth if q_s2_rec else 0
        queue_s3 = q_s3_rec.queue_depth if q_s3_rec else 0

        jitter_s2 = self._estimate_jitter_for_switch(1)
        jitter_s3 = self._estimate_jitter_for_switch(2)
        delay_s2  = self._estimate_delay_for_switch(1)
        delay_s3  = self._estimate_delay_for_switch(2)

        state = {
            'queue_s2':      queue_s2,
            'queue_s3':      queue_s3,
            'prev_queue_s2': self._prev_q_s2,
            'prev_queue_s3': self._prev_q_s3,
            'jitter_s2':     jitter_s2,
            'jitter_s3':     jitter_s3,
            'delay_s2':      delay_s2,
            'delay_s3':      delay_s3,
            'current_path':  self.get_current_path(),
        }

        self._prev_q_s2 = queue_s2
        self._prev_q_s3 = queue_s3
        return state


    def drain_telemetry(self, max_records: int = 1000) -> List[TelemetryRecord]:
        records = []
        for _ in range(max_records):
            try:
                records.append(self._telemetry_q.get_nowait())
            except queue.Empty:
                break
        return records


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s')

    ap = argparse.ArgumentParser(description='P4Runtime Controller')
    ap.add_argument('--sim',      action='store_true')
    ap.add_argument('--path',     type=int, default=1, choices=[1, 2])
    ap.add_argument('--duration', type=int, default=30)
    args = ap.parse_args()

    ctrl = Controller(simulation=args.sim)
    ctrl.start()
    if args.path != 1:
        ctrl.update_path(args.path)

    start = time.time()
    try:
        while True:
            records = ctrl.drain_telemetry(100)
            for r in records:
                logger.info('TEL sw=%d q=%d ts=%d', r.switch_id, r.queue_depth, r.ingress_timestamp)
            state = ctrl.get_network_state()
            logger.info('STATE %s', state)
            time.sleep(1)
            if args.duration > 0 and (time.time() - start) > args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.stop()
