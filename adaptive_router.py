#!/usr/bin/env python3

import os
import sys
import time
import csv
import argparse
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from controller.controller import Controller
from rl.dqn_agent          import DQNAgent
from rl.rl_env             import (JitterRoutingEnv, HYSTERESIS_STEPS,
                                    MAX_JITTER, MAX_QUEUE_DEPTH, MAX_DELAY,
                                    MAX_TREND, MAX_EPISODE_STEPS)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/adaptive_router.log'),
    ]
)
logger = logging.getLogger('adaptive_router')

os.makedirs('logs',    exist_ok=True)
os.makedirs('results', exist_ok=True)

Q_CONFIDENCE_THRESHOLD = 0.05   

JITTER_HIGH_THRESHOLD = 500.0   # above this → aggressive (short hysteresis)
JITTER_LOW_THRESHOLD  = 100.0   # below this → conservative (long hysteresis)
HYSTERESIS_HIGH       = 2       # steps when network is jittery
HYSTERESIS_LOW        = 5       # steps when network is stable


# ---------------------------------------------------------------------------
# AdaptiveRouter
# ---------------------------------------------------------------------------
class AdaptiveRouter:
    def __init__(self,
                 checkpoint:         str,
                 controller:         Controller,
                 decision_interval:  float = 0.5,
                 duration:           float = 60.0,
                 csv_log:            str   = 'results/routing_log.csv'):

        self.controller       = controller
        self.decision_interval = decision_interval
        self.duration         = duration
        self.csv_log          = csv_log

        self.agent = DQNAgent()
        if os.path.exists(checkpoint):
            self.agent.load(checkpoint)
            logger.info('Agent loaded from %s', checkpoint)
        else:
            logger.warning('Checkpoint not found at %s so using untrained agent', checkpoint)

        self._hysteresis         = HYSTERESIS_STEPS
        self._steps_since_switch = self._hysteresis   
        self._prev_action        = controller.get_current_path() - 1
        self._log_rows           = []
        self._running            = False

    def _build_obs(self) -> np.ndarray:
        s   = self.controller.get_network_state()
        ema = getattr(self, '_ema_queue', 0.0)
        ssw = getattr(self, '_steps_since_router_switch', 0)
        switch_mem = min(ssw / MAX_EPISODE_STEPS, 1.0)
        raw = np.array([
            s['queue_s2']      / MAX_QUEUE_DEPTH,
            s['queue_s3']      / MAX_QUEUE_DEPTH,
            s['prev_queue_s2'] / MAX_QUEUE_DEPTH,
            s['prev_queue_s3'] / MAX_QUEUE_DEPTH,
            s['jitter_s2']     / MAX_JITTER,
            s['jitter_s3']     / MAX_JITTER,
            s['delay_s2']      / MAX_DELAY,
            s['delay_s3']      / MAX_DELAY,
            float(s['current_path'] - 1),
            float(np.clip(ema / 50.0, -1.0, 1.0) * 0.5 + 0.5),
            float(switch_mem),
        ], dtype=np.float32)
        return np.clip(raw, 0.0, 1.0)

    def _update_hysteresis(self, jitter: float):
        if jitter > JITTER_HIGH_THRESHOLD:
            self._hysteresis = HYSTERESIS_HIGH
        elif jitter < JITTER_LOW_THRESHOLD:
            self._hysteresis = HYSTERESIS_LOW
        else:
            # Linear interpolation between low and high
            t = (jitter - JITTER_LOW_THRESHOLD) / (JITTER_HIGH_THRESHOLD - JITTER_LOW_THRESHOLD)
            self._hysteresis = int(round(HYSTERESIS_LOW + t * (HYSTERESIS_HIGH - HYSTERESIS_LOW)))

    def run(self):
        self._running = True
        start_time    = time.time()
        step_num      = 0

        logger.info('Adaptive router started | duration=%.0fs interval=%.2fs',
                    self.duration, self.decision_interval)

        with open(self.csv_log, 'w', newline='') as csvf:
            writer = csv.DictWriter(csvf, fieldnames=[
                'timestamp', 'step', 'queue_s2', 'queue_s3',
                'jitter_est', 'delay_est', 'current_path',
                'action', 'switched', 'hysteresis',
                'q_val_path1', 'q_val_path2', 'q_confidence',
            ])
            writer.writeheader()

            try:
                while self._running:
                    elapsed = time.time() - start_time
                    if self.duration > 0 and elapsed >= self.duration:
                        break

                    obs    = self._build_obs()
                    qvals  = self.agent.q_values(obs)
                    state  = self.controller.get_network_state()
                    # track EMA trend for obs (mirrors rl_env B6)
                    act_q  = state['queue_s2'] if state['current_path'] == 1 else state['queue_s3']
                    prev_q = state['prev_queue_s2'] if state['current_path'] == 1 else state['prev_queue_s3']
                    raw_tr = float(act_q - prev_q)
                    self._ema_queue = 0.3 * raw_tr + 0.7 * getattr(self, '_ema_queue', raw_tr)
                    jitter = state['jitter_s2'] if state['current_path'] == 1 else state['jitter_s3']

                    self._steps_since_router_switch = getattr(self, '_steps_since_router_switch', 0) + 1
                    self._update_hysteresis(jitter)

                    q_diff = abs(float(qvals[0]) - float(qvals[1]))
                    if q_diff < Q_CONFIDENCE_THRESHOLD:
                        action = self._prev_action
                        logger.debug('Step %4d  LOW CONFIDENCE (ΔQ=%.3f < %.3f) hold path %d',
                                     step_num, q_diff, Q_CONFIDENCE_THRESHOLD,
                                     self.controller.get_current_path())
                    else:
                        action = int(np.argmax(qvals))

                    # Hysteresis gate
                    switched = False
                    if action != self._prev_action:
                        if self._steps_since_switch >= self._hysteresis:
                            self.controller.update_path(action + 1)
                            switched = True
                            self._steps_since_switch = 0
                            self._steps_since_router_switch = 0  # reset switch memory
                            logger.info(
                                'Step %4d  PATH SWITCH → %d  (ΔQ=%.3f  hyst=%d  j=%.0fµs)',
                                step_num, action + 1, q_diff,
                                self._hysteresis, jitter)
                    self._steps_since_switch += 1
                    self._prev_action = action

                    row = {
                        'timestamp':    elapsed,
                        'step':         step_num,
                        'queue_s2':     state['queue_s2'],
                        'queue_s3':     state['queue_s3'],
                        'jitter_est':   round(jitter, 2),
                        'delay_est':    round(state['delay_s2']  if state['current_path'] == 1  else state['delay_s3'],2),
                        'current_path': state['current_path'],
                        'action':       action,
                        'switched':     int(switched),
                        'hysteresis':   self._hysteresis,
                        'q_val_path1':  round(float(qvals[0]), 4),
                        'q_val_path2':  round(float(qvals[1]), 4),
                        'q_confidence': round(q_diff, 4),
                    }
                    writer.writerow(row)
                    self._log_rows.append(row)

                    step_num += 1
                    time.sleep(self.decision_interval)

            except KeyboardInterrupt:
                logger.info('Interrupted by user.')

        logger.info('Router stopped after %d steps.', step_num)
        self._save_plots()

    def stop(self):
        self._running = False

    def _save_plots(self):
        if not self._log_rows:
            return
        ts     = [r['timestamp']    for r in self._log_rows]
        qd2    = [r['queue_s2']     for r in self._log_rows]
        qd3    = [r['queue_s3']     for r in self._log_rows]
        jits   = [r['jitter_est']   for r in self._log_rows]
        paths  = [r['current_path'] for r in self._log_rows]
        hysts  = [r['hysteresis']   for r in self._log_rows]
        swpts  = [r['timestamp']    for r in self._log_rows if r['switched']]
        confs  = [r['q_confidence'] for r in self._log_rows]

        fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
        fig.suptitle('Online Adaptive Routing — Live Telemetry', fontsize=13)

        axes[0].plot(ts, qd2, label='Queue s2 (Path 1)', color='steelblue')
        axes[0].plot(ts, qd3, label='Queue s3 (Path 2)', color='tomato')
        for sw in swpts:
            axes[0].axvline(sw, color='gray', ls=':', alpha=0.7)
        axes[0].set_ylabel('Queue Depth (pkts)'); axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(ts, jits, color='darkorange')
        for sw in swpts:
            axes[1].axvline(sw, color='gray', ls=':', alpha=0.7)
        axes[1].axhline(JITTER_HIGH_THRESHOLD, color='red',   ls='--', alpha=0.5, label='high threshold')
        axes[1].axhline(JITTER_LOW_THRESHOLD,  color='green', ls='--', alpha=0.5, label='low threshold')
        axes[1].set_ylabel('Jitter (µs)'); axes[1].legend(); axes[1].grid(alpha=0.3)

        axes[2].step(ts, paths, where='post', color='mediumseagreen')
        axes[2].set_ylabel('Active Path')
        axes[2].set_yticks([1, 2]); axes[2].set_yticklabels(['Path 1\n(via s2)', 'Path 2\n(via s3)'])
        axes[2].grid(alpha=0.3)

        ax3b = axes[2].twinx()
        ax3b.step(ts, hysts, where='post', color='purple', alpha=0.4, ls='--')
        ax3b.set_ylabel('Hysteresis (steps)', color='purple')

        axes[3].plot(ts, confs, color='navy', alpha=0.8)
        axes[3].axhline(Q_CONFIDENCE_THRESHOLD, color='red', ls='--', alpha=0.6, label='threshold')
        axes[3].set_ylabel('Q Confidence (|ΔQ|)')
        axes[3].set_xlabel('Time (s)'); axes[3].legend(); axes[3].grid(alpha=0.3)

        plt.tight_layout()
        out = 'results/online_routing.png'
        plt.savefig(out, dpi=150); plt.close(fig)
        logger.info('Online routing plot → %s', out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Online adaptive routing')
    ap.add_argument('--checkpoint', default='rl/checkpoints/best.pt')
    ap.add_argument('--sim',        action='store_true')
    ap.add_argument('--duration',   type=float, default=60.0)
    ap.add_argument('--interval',   type=float, default=0.5)
    ap.add_argument('--csv',        default='results/routing_log.csv')
    args = ap.parse_args()

    ctrl = Controller(simulation=args.sim)
    ctrl.start()

    router = AdaptiveRouter(
        checkpoint=args.checkpoint,
        controller=ctrl,
        decision_interval=args.interval,
        duration=args.duration,
        csv_log=args.csv,
    )
    try:
        router.run()
    finally:
        ctrl.stop()
