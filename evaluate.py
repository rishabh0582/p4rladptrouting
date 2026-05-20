#!/usr/bin/env python3


import os
import sys
import json
import logging
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))

from controller.controller import Controller
from rl.rl_env             import JitterRoutingEnv
from rl.dqn_agent          import DQNAgent

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('evaluate')
os.makedirs('results', exist_ok=True)


# ---------------------------------------------------------------------------
# Policy functions
# ---------------------------------------------------------------------------
def make_rl_policy(agent: DQNAgent):
    def policy(obs, step): return agent.greedy_action(obs)
    return policy

def static_policy(path: int):
    action = path - 1
    def policy(obs, step): return action
    return policy

def round_robin_policy(period: int = 20):
    def policy(obs, step): return (step // period) % 2
    return policy


# ---------------------------------------------------------------------------
# Run one policy for n_episodes
# ---------------------------------------------------------------------------
def run_policy(policy, env: JitterRoutingEnv, n_episodes: int, label: str) -> Dict:
    all_rewards   = []
    all_jitters   = []
    all_q_s2      = []
    all_q_s3      = []
    all_paths     = []    
    all_switches  = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False
        ep_reward = 0.0
        ep_jitters, ep_q2, ep_q3, ep_paths = [], [], [], []
        ep_switches = 0; step = 0

        while not (done or truncated):
            action = policy(obs, step)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward   += reward
            ep_jitters.append(info.get('jitter_global', info['jitter']))
            ep_q2.append(info['queue_s2'])
            ep_q3.append(info['queue_s3'])
            ep_paths.append(info['active_path'])
            ep_switches += info['route_changed']
            step += 1

        all_rewards.append(ep_reward)
        all_jitters.extend(ep_jitters)
        all_q_s2.extend(ep_q2)
        all_q_s3.extend(ep_q3)
        all_paths.extend(ep_paths)
        all_switches.append(ep_switches)
        logger.info('  [%s] ep=%d R=%.2f j=%.1f sw=%d',
                    label, ep, ep_reward,
                    np.mean(ep_jitters) if ep_jitters else 0, ep_switches)

    queue_active = [
        q2 if path == 1 else q3
        for q2, q3, path in zip(all_q_s2, all_q_s3, all_paths)
    ]

    if len(all_jitters) > 1:
        jitter_std = float(np.std(all_jitters))
        if jitter_std < 1e-6:
            logger.warning('[%s] Jitter is essentially constant (std=%.2e) '
                           '— check telemetry pipeline for bugs.', label, jitter_std)

    return {
        'label':                label,
        'mean_reward':          float(np.mean(all_rewards)),
        'std_reward':           float(np.std(all_rewards)),
        'mean_jitter':          float(np.mean(all_jitters)) if all_jitters else 0.0,
        'std_jitter':           float(np.std(all_jitters))  if all_jitters else 0.0,
        'p50_queue':            float(np.percentile(queue_active, 50)),
        'p95_queue':            float(np.percentile(queue_active, 95)),
        'p99_queue':            float(np.percentile(queue_active, 99)),
        'mean_switches_per_ep': float(np.mean(all_switches)),
        'episode_rewards':      all_rewards,
        'all_jitters':          all_jitters,   
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def save_eval_plots(results: List[Dict]):
    labels  = [r['label']       for r in results]
    rewards = [r['mean_reward'] for r in results]
    jitters = [r['mean_jitter'] for r in results]
    p95q    = [r['p95_queue']   for r in results]
    sw_freq = [r['mean_switches_per_ep'] for r in results]

    colors = ['steelblue', 'tomato', 'mediumseagreen', 'darkorange'][:len(results)]
    x      = np.arange(len(labels)); w = 0.6

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Policy Comparison — Jitter-Aware Routing Evaluation', fontsize=13)

    def bar(ax, vals, title, ylabel, fmt='.2f'):
        bars = ax.bar(x, vals, width=w, color=colors, alpha=0.85, edgecolor='white')
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10, ha='right')
        ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(True, axis='y', alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() * 1.01,
                    f'{v:{fmt}}', ha='center', va='bottom', fontsize=9)

    bar(axes[0, 0], rewards, 'Mean Episode Reward ↑', 'Reward')
    bar(axes[0, 1], jitters, 'Mean Jitter (µs) ↓', 'Jitter (µs)')
    bar(axes[1, 0], p95q,    'P95 Active-Path Queue ↓', 'Queue (pkts)', fmt='.1f')
    bar(axes[1, 1], sw_freq, 'Avg Path Switches / Episode', 'Switches',  fmt='.1f')

    # Annotate the jitter chart with % reduction vs best baseline
    rl_j = jitters[0] if jitters else 0
    for i, (lbl, j) in enumerate(zip(labels, jitters)):
        if i == 0 or rl_j == 0:
            continue
        pct = 100.0 * (j - rl_j) / j if j > 0 else 0
        color = 'green' if pct > 0 else 'red'
        axes[0, 1].annotate(f'{pct:+.0f}%', xy=(i, j),
                            xytext=(0, 6), textcoords='offset points',
                            ha='center', fontsize=8, color=color, fontweight='bold')

    plt.tight_layout()
    path = 'results/evaluation_plots.png'
    plt.savefig(path, dpi=150); plt.close(fig)
    logger.info('Evaluation plots → %s', path)

    # Reward curves
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    for r, c in zip(results, colors):
        ax2.plot(r['episode_rewards'], alpha=0.8, color=c, label=r['label'])
    ax2.set_title('Episode Reward Curves')
    ax2.set_xlabel('Episode'); ax2.set_ylabel('Total Reward')
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/reward_curves.png', dpi=150); plt.close(fig2)
    logger.info('Reward curves → results/reward_curves.png')


# ---------------------------------------------------------------------------
# ASCII table
# ---------------------------------------------------------------------------
def ascii_table(results: List[Dict]) -> str:
    headers = ['Policy', 'Mean Reward', 'Mean Jitter (µs)',
               'P95 Queue', 'P99 Queue', 'Avg Switches']
    widths  = [20, 13, 18, 11, 11, 14]
    sep     = '+-' + '-+-'.join('-' * w for w in widths) + '-+'
    fmt     = '| ' + ' | '.join(f'{{:<{w}}}' for w in widths) + ' |'
    lines   = [sep, fmt.format(*headers), sep]
    for r in results:
        row = [r['label'],
               f"{r['mean_reward']:.2f}",
               f"{r['mean_jitter']:.1f}",
               f"{r['p95_queue']:.1f}",
               f"{r['p99_queue']:.1f}",
               f"{r['mean_switches_per_ep']:.1f}"]
        lines.append(fmt.format(*row))
    lines.append(sep)
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    ctrl = Controller(simulation=args.sim)
    ctrl.start()
    env  = JitterRoutingEnv(ctrl, max_steps=args.max_steps)

    agent = DQNAgent()
    if os.path.exists(args.checkpoint):
        agent.load(args.checkpoint)
    else:
        logger.warning('Checkpoint not found — RL policy is untrained.')

    policies = [
        ('RL Adaptive',   make_rl_policy(agent)),
        ('Static Path 1', static_policy(1)),
        ('Static Path 2', static_policy(2)),
        ('Round-Robin',   round_robin_policy(period=args.rr_period)),
    ]

    all_results = []
    for label, policy in policies:
        logger.info('=== Evaluating: %s (%d episodes) ===', label, args.episodes)
        result = run_policy(policy, env, args.episodes, label)
        all_results.append(result)

    static_means = [r['mean_reward'] for r in all_results if 'Static' in r['label']]
    best_static  = max(static_means) if static_means else 0.0
    logger.info('Best static mean reward: %.2f', best_static)

    serialisable = [{k: v for k, v in r.items() if k != 'episode_rewards'}
                    for r in all_results]
    with open('results/evaluation_report.json', 'w') as f:
        json.dump(serialisable, f, indent=2)
    logger.info('Evaluation report → results/evaluation_report.json')

    table = ascii_table(all_results)
    print('\n' + table + '\n')
    with open('results/evaluation_table.txt', 'w') as f:
        f.write(table)

    rl_res = next((r for r in all_results if 'RL' in r['label']), None)
    if rl_res:
        rl_jitter = rl_res['mean_jitter']
        print('\n=== Jitter Reduction Report (GAP 4 — Primary Claim) ===')
        for r in all_results:
            if r['label'] == rl_res['label']:
                continue
            other_jitter = r['mean_jitter']
            if other_jitter > 0:
                pct = 100.0 * (other_jitter - rl_jitter) / other_jitter
                symbol = '↓' if pct > 0 else '↑'
                print(f"  RL vs {r['label']:20s}:  "
                      f"RL={rl_jitter:.1f}µs  baseline={other_jitter:.1f}µs  "
                      f"  {symbol}{abs(pct):.1f}% jitter {'reduction' if pct>0 else 'increase'}")
        print()

    save_eval_plots(all_results)
    ctrl.stop()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Evaluate routing policies')
    ap.add_argument('--checkpoint', default='rl/checkpoints/best.pt')
    ap.add_argument('--sim',        action='store_true')
    ap.add_argument('--episodes',   type=int, default=30)
    ap.add_argument('--max-steps',  type=int, default=400,  
                    help='Episode steps — must match training value')
    ap.add_argument('--rr-period',  type=int, default=20)
    main(ap.parse_args())
