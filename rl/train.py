#!/usr/bin/env python3

import os
import sys
import time
import json
import argparse
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from controller.controller import Controller
from rl.rl_env             import JitterRoutingEnv
from rl.dqn_agent          import DQNAgent, HyperParams

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)-20s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/training.log'),
    ]
)
logger = logging.getLogger('train')

os.makedirs('logs',           exist_ok=True)
os.makedirs('rl/checkpoints', exist_ok=True)
os.makedirs('results',        exist_ok=True)


CONVERGENCE_THRESHOLD = -400.0
CONVERGENCE_WINDOW    = 10    


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def smooth(values: List[float], window: int = 10) -> List[float]:
    out = []
    for i, v in enumerate(values):
        w = values[max(0, i - window + 1): i + 1]
        out.append(sum(w) / len(w))
    return out


def save_training_plots(history: dict, out_dir: str = 'results'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training History : Jitter-Aware RL Router', fontsize=14)
    eps = range(len(history['episode_rewards']))

    ax = axes[0, 0]
    ax.plot(eps, history['episode_rewards'], alpha=0.35, color='steelblue', label='raw')
    ax.plot(eps, smooth(history['episode_rewards']), color='steelblue', lw=2, label='smoothed')
    ax.axhline(CONVERGENCE_THRESHOLD, color='green', ls='--', alpha=0.6, label='convergence')
    ax.set_title('Episode Total Reward'); ax.set_xlabel('Episode'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(eps, history['mean_jitter'], color='tomato')
    ax.set_title('Mean Jitter per Episode (µs)'); ax.set_xlabel('Episode'); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.bar(eps, history['path_switches'], color='mediumseagreen', alpha=0.7)
    ax.set_title('Path Switches per Episode'); ax.set_xlabel('Episode'); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(history['epsilon'], color='orange')
    ax.set_title('Exploration Rate (ε)'); ax.set_xlabel('Episode'); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'training_history.png')
    plt.savefig(path, dpi=150); plt.close(fig)
    logger.info('Training plots → %s', path)


def save_comparison_plot(rl_r, p1_r, p2_r, best_static_r, out_dir='results'):
    fig, ax = plt.subplots(figsize=(12, 5))
    n = min(len(rl_r), len(p1_r), len(p2_r))
    ax.plot(rl_r[:n],          label='RL adaptive',          color='steelblue', lw=2)
    ax.plot(p1_r[:n],          label='Static Path 1',         color='tomato',    lw=2, ls='--')
    ax.plot(p2_r[:n],          label='Static Path 2',         color='gold',      lw=2, ls='--')
    ax.axhline(best_static_r,  label=f'Best static ({best_static_r:.2f})',
               color='gray', ls=':', lw=1.5)
    ax.set_title('RL vs Static Routing : Episode Reward')
    ax.set_xlabel('Evaluation Episode'); ax.set_ylabel('Total Reward')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'rl_vs_static.png')
    plt.savefig(path, dpi=150); plt.close(fig)
    logger.info('Comparison plot → %s', path)


# ---------------------------------------------------------------------------
# Policy runners
# ---------------------------------------------------------------------------
def run_policy(policy_fn, env, n_episodes: int) -> List[float]:
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False; ep_r = 0.0
        while not (done or truncated):
            obs, r, done, truncated, _ = env.step(policy_fn(obs))
            ep_r += r
        rewards.append(ep_r)
    return rewards


def evaluate_agent(agent: DQNAgent, env: JitterRoutingEnv, n: int = 10) -> Dict:
    rewards, jitters, switches = [], [], []
    for _ in range(n):
        obs, _ = env.reset()
        done = trunc = False; ep_r = ep_j = ep_s = 0
        while not (done or trunc):
            action = agent.greedy_action(obs)
            obs, r, done, trunc, info = env.step(action)
            ep_r += r; ep_j += info.get('jitter_global', info['jitter']); ep_s += info['route_changed']
        rewards.append(ep_r); jitters.append(ep_j); switches.append(ep_s)
    return {
        'mean_reward':   float(np.mean(rewards)),
        'std_reward':    float(np.std(rewards)),
        'mean_jitter':   float(np.mean(jitters)),
        'mean_switches': float(np.mean(switches)),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(args):
    ctrl = Controller(simulation=args.sim)
    ctrl.start()

    hp = HyperParams(
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        buffer_capacity=50_000,
        target_update=args.target_update,
        hidden_size=128,
        min_buffer_size=min(500, args.batch_size * 4),
    )

    env   = JitterRoutingEnv(ctrl, max_steps=args.max_steps)
    agent = DQNAgent(hp=hp, log_dir='logs/tb')

    if args.resume:
        agent.load(args.resume)

    history = {
        'episode_rewards': [], 'mean_jitter': [],
        'path_switches': [], 'epsilon': [], 'moving_avg': [],
    }
    best_reward   = float('-inf')
    ckpt_path     = args.checkpoint or 'rl/checkpoints/best.pt'
    converged_for = 0

    logger.info('=== Training | episodes=%d sim=%s ===', args.episodes, args.sim)

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = trunc = False
        ep_r = ep_j = ep_s = 0.0; steps = 0; losses = []

        while not (done or trunc):
            action = agent.select_action(obs)
            nobs, r, done, trunc, info = env.step(action)
            agent.store(obs, action, r, nobs, done or trunc)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            ep_r += r; ep_j += info['jitter']; ep_s += info['route_changed']
            obs = nobs; steps += 1

        history['episode_rewards'].append(ep_r)
        history['mean_jitter'].append(ep_j / max(1, steps))
        history['path_switches'].append(ep_s)
        history['epsilon'].append(agent.epsilon)

        recent = history['episode_rewards'][-CONVERGENCE_WINDOW:]
        moving_avg = float(np.mean(recent))
        history['moving_avg'].append(moving_avg)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        logger.info(
            'Ep %4d | R=%7.2f | mavg=%7.2f | j_global=%.0fµs | sw=%d | ε=%.3f | loss=%.4f',
            ep, ep_r, moving_avg,
            ep_j / max(1, steps), int(ep_s), agent.epsilon, mean_loss,
        )

        if ep_r > best_reward:
            best_reward = ep_r
            agent.save(ckpt_path)
            logger.info('  ↑ New best %.2f , checkpoint saved', best_reward)

        if moving_avg > CONVERGENCE_THRESHOLD and len(recent) == CONVERGENCE_WINDOW:
            converged_for += 1
            if converged_for >= CONVERGENCE_WINDOW:
                logger.info('=== Converged after %d episodes (mavg=%.2f) ===', ep, moving_avg)
                break
        else:
            converged_for = 0

        if ep % args.eval_every == 0:
            stats = evaluate_agent(agent, env, n=5)
            logger.info('  [EVAL] R=%.2f  j=%.0fµs  sw=%.1f',
                        stats['mean_reward'], stats['mean_jitter'], stats['mean_switches'])

    agent.save('rl/checkpoints/final.pt')
    with open('results/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    save_training_plots(history)

    n_eval = min(len(history['episode_rewards']), 30)
    logger.info('Running static baselines for comparison (%d episodes)...', n_eval)
    p1_rewards = run_policy(lambda o: 0, env, n_eval)
    p2_rewards = run_policy(lambda o: 1, env, n_eval)

    best_static_mean = max(np.mean(p1_rewards), np.mean(p2_rewards))
    logger.info('Best static mean reward: %.2f  (path1=%.2f, path2=%.2f)',
                best_static_mean, np.mean(p1_rewards), np.mean(p2_rewards))

    rl_eval_rewards = run_policy(agent.greedy_action, env, n_eval)
    save_comparison_plot(rl_eval_rewards, p1_rewards, p2_rewards, best_static_mean)

    final_eval = evaluate_agent(agent, env, n=1)
    final_eval['best_static_mean_reward'] = round(best_static_mean, 2)
    final_eval['rl_vs_best_static_improvement'] = round(
        final_eval['mean_reward'] - best_static_mean, 2)
    logger.info('=== Final Evaluation ===')
    for k, v in final_eval.items():
        logger.info('  %s: %.4f', k, v)
    with open('results/final_eval.json', 'w') as f:
        json.dump(final_eval, f, indent=2)

    ctrl.stop()
    logger.info('Training complete. Best reward: %.2f', best_reward)
    return agent, history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Train DQN')
    ap.add_argument('--sim',           action='store_true')
    ap.add_argument('--episodes',      type=int,   default=80)
    ap.add_argument('--max-steps',     type=int,   default=400)
    ap.add_argument('--lr',            type=float, default=1e-3)
    ap.add_argument('--gamma',         type=float, default=0.95)
    ap.add_argument('--epsilon-decay', type=int,   default=1200)
    ap.add_argument('--batch-size',    type=int,   default=64)
    ap.add_argument('--target-update', type=int,   default=200)
    ap.add_argument('--eval-every',    type=int,   default=20)
    ap.add_argument('--checkpoint',    type=str,   default=None)
    ap.add_argument('--resume',        type=str,   default=None)
    train(ap.parse_args())
