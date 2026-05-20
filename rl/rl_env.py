#!/usr/bin/env python3

import time
import numpy as np
import logging
from typing import Optional, Dict, Tuple

logger = logging.getLogger('rl_env')

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Normalisation ceilings
# ---------------------------------------------------------------------------
MAX_QUEUE_DEPTH  = 100
MAX_JITTER       = 2000.0   # µs
MAX_DELAY        = 5000.0   # µs
MAX_TREND        = 50.0

PATHS = [1, 2]

HYSTERESIS_STEPS  = 5
HARD_COOLDOWN     = 6      
MAX_EPISODE_STEPS = 10

# ---------------------------------------------------------------------------
# Reward weights  
# ---------------------------------------------------------------------------
ALPHA   = 2.0    
BETA    = 0.3    
GAMMA   = 0.3    
DELTA   = 2.5    
TREND_W = 0.3    

JITTER_SPIKE_THRESHOLD = 0.7
JITTER_SPIKE_PENALTY   = 0.5

REWARD_CLIP_LOW  = -5.0    
REWARD_CLIP_HIGH =  0.0


# ---------------------------------------------------------------------------
# Stub
# ---------------------------------------------------------------------------
class _MinimalEnv:
    metadata = {}
    def reset(self, seed=None, options=None): raise NotImplementedError
    def step(self, action): raise NotImplementedError
    def render(self): pass
    def close(self): pass

_BaseEnv = gym.Env if GYM_AVAILABLE else _MinimalEnv


# ---------------------------------------------------------------------------
# JitterRoutingEnv
# ---------------------------------------------------------------------------
class JitterRoutingEnv(_BaseEnv):
    
    OBS_DIM  = 11   
    metadata = {'render_modes': ['human']}

    def __init__(self,
                 controller,
                 alpha:   float = ALPHA,
                 beta:    float = BETA,
                 gamma:   float = GAMMA,
                 delta:   float = DELTA,
                 trend_w: float = TREND_W,
                 hysteresis_steps: int = HYSTERESIS_STEPS,
                 hard_cooldown:    int = HARD_COOLDOWN,
                 max_steps: int = MAX_EPISODE_STEPS):

        super().__init__()
        self.controller       = controller
        self.alpha            = alpha
        self.beta             = beta
        self.gamma            = gamma
        self.delta            = delta
        self.trend_w          = trend_w
        self.hysteresis_steps = hysteresis_steps
        self.hard_cooldown    = hard_cooldown
        self.max_steps        = max_steps

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self._reset_internal_state()

    # ------------------------------------------------------------------ #
    def _reset_internal_state(self):
        self._step_count          = 0
        self._steps_since_switch  = self.hysteresis_steps
        self._cooldown_remaining  = 0
        self._prev_action         = self.controller.get_current_path() - 1
        self._episode_rewards     = []
        self._ema_queue           = None

    def _get_obs(self) -> np.ndarray:
        s   = self.controller.get_network_state()
        ema = self._ema_queue if self._ema_queue is not None else 0.0
        switch_mem = min(self._steps_since_switch / self.max_steps, 1.0)
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
            float(np.clip(ema / MAX_TREND, -1.0, 1.0) * 0.5 + 0.5),
            float(switch_mem),
        ], dtype=np.float32)
        return np.clip(raw, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        if GYM_AVAILABLE:
            try:
                super().reset(seed=seed)
            except Exception:
                pass
        self._reset_internal_state()
        return self._get_obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action)

        selected_path = PATHS[action]
        prev_path     = self.controller.get_current_path()

        
        if self._cooldown_remaining > 0:
            selected_path = prev_path
            self._cooldown_remaining -= 1

        switched = 0.0
        if selected_path != prev_path:
            if self._steps_since_switch >= self.hysteresis_steps:
                self.controller.set_path(selected_path)
                switched = 1.0
                self._steps_since_switch    = 0
                self._cooldown_remaining    = self.hard_cooldown
        self._steps_since_switch += 1

        time.sleep(0.1)   
        s = self.controller.get_network_state()

        if selected_path == 1:
            q_now  = s['queue_s2'];  q_prev = s['prev_queue_s2']
            j_act  = s['jitter_s2']; d_act  = s['delay_s2']
        else:
            q_now  = s['queue_s3'];  q_prev = s['prev_queue_s3']
            j_act  = s['jitter_s3']; d_act  = s['delay_s3']

        q_active = min(q_now / MAX_QUEUE_DEPTH, 1.0)
        j_active = min(j_act / MAX_JITTER,      1.0)
        d_active = min(d_act / MAX_DELAY,        1.0)

        q_global = min((s['queue_s2'] + s['queue_s3']) / (2 * MAX_QUEUE_DEPTH), 1.0)
        j_global = min((s['jitter_s2'] + s['jitter_s3']) / (2 * MAX_JITTER),    1.0)
        d_global = min((s['delay_s2']  + s['delay_s3'])  / (2 * MAX_DELAY),     1.0)

        queue_norm  = 0.8 * q_active + 0.2 * q_global
        jitter_norm = 0.8 * j_active + 0.2 * j_global
        delay_norm  = 0.8 * d_active + 0.2 * d_global

        raw_trend = float(q_now - q_prev)
        self._ema_queue = (raw_trend if self._ema_queue is None
                           else 0.3 * raw_trend + 0.7 * self._ema_queue)
        trend_norm = float(np.clip(self._ema_queue / MAX_TREND, -1.0, 1.0))

        raw_reward = -(
            self.alpha   * jitter_norm +
            self.beta    * delay_norm  +
            self.gamma   * queue_norm  +
            self.delta   * switched    +   
            self.trend_w * trend_norm
        )
        if jitter_norm > JITTER_SPIKE_THRESHOLD:
            raw_reward -= JITTER_SPIKE_PENALTY

        reward = float(np.clip(raw_reward, REWARD_CLIP_LOW, REWARD_CLIP_HIGH))

        obs = self._get_obs()
        self._episode_rewards.append(reward)
        self._prev_action = action
        self._step_count += 1

        truncated  = self._step_count >= self.max_steps
        terminated = False

        jitter_global = (s['jitter_s2'] + s['jitter_s3']) / 2.0
        info = {
            'queue_s2':      s['queue_s2'],
            'queue_s3':      s['queue_s3'],
            'jitter_s2':     s['jitter_s2'],
            'jitter_s3':     s['jitter_s3'],
            'delay_s2':      s['delay_s2'],
            'delay_s3':      s['delay_s3'],
            'jitter':        j_act,
            'jitter_global': jitter_global,
            'delay':         d_act,
            'active_path':   self.controller.get_current_path(),
            'route_changed': int(switched),
            'cooldown':      self._cooldown_remaining,
            'steps_since_switch': self._steps_since_switch,
            'trend_ema':     self._ema_queue,
            'jitter_spike':  int(jitter_norm > JITTER_SPIKE_THRESHOLD),
        }
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        s = self.controller.get_network_state()
        print(f'[ENV] step={self._step_count:4d}  path={s["current_path"]}'
              f'  q2={s["queue_s2"]:3d}  q3={s["queue_s3"]:3d}'
              f'  j2={s["jitter_s2"]:.0f}µs  j3={s["jitter_s3"]:.0f}µs'
              f'  cd={self._cooldown_remaining}  ssw={self._steps_since_switch}')

    def close(self): pass

    def episode_stats(self):
        if not self._episode_rewards:
            return {}
        return {'total_reward': sum(self._episode_rewards),
                'mean_reward':  sum(self._episode_rewards)/len(self._episode_rewards),
                'steps':        len(self._episode_rewards)}
