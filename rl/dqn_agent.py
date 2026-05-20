#!/usr/bin/env python3

import os
import math
import json
import random
import logging
import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

logger = logging.getLogger('dqn_agent')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning('PyTorch not installed so using numpy Q-table fallback.')

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
@dataclass
class HyperParams:
    lr:               float = 1e-3
    gamma:            float = 0.95
    epsilon_start:    float = 1.0
    epsilon_end:      float = 0.05    
    epsilon_decay:    int   = 4000   
    batch_size:       int   = 64
    buffer_capacity:  int   = 50_000
    target_update:    int   = 200
    hidden_size:      int   = 128
    n_hidden_layers:  int   = 2
    min_buffer_size:  int   = 500
    grad_clip:        float = 10.0


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------
@dataclass
class Transition:
    state:      np.ndarray
    action:     int
    reward:     float
    next_state: np.ndarray
    done:       bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self._buf = deque(maxlen=capacity)

    def push(self, *args):
        self._buf.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self._buf, batch_size)

    def __len__(self):
        return len(self._buf)


# ---------------------------------------------------------------------------
# Q-Network 
# ---------------------------------------------------------------------------
if TORCH_AVAILABLE:
    class QNetwork(nn.Module):
        def __init__(self, state_dim: int, n_actions: int, hp: HyperParams):
            super().__init__()
            layers = [nn.Linear(state_dim, hp.hidden_size), nn.ReLU()]
            for _ in range(hp.n_hidden_layers - 1):
                layers += [nn.Linear(hp.hidden_size, hp.hidden_size), nn.ReLU()]
            layers += [nn.Linear(hp.hidden_size, n_actions)]
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


# ---------------------------------------------------------------------------
# Numpy Q-table fallback
# ---------------------------------------------------------------------------
class NumpyQAgent:
    def __init__(self, n_actions=2, hp: HyperParams = None):
        hp = hp or HyperParams()
        self.hp        = hp
        self.n_actions = n_actions        
        self.q_table   = np.zeros((3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, n_actions))
        self.epsilon   = hp.epsilon_start        
        self.steps_done = 0

    def _discretise(self, state: np.ndarray) -> tuple:        
        cont_bins = np.clip((state[:8] * 2).astype(int), 0, 2).tolist()   # 3 bins
        path_bin  = int(np.clip(int(state[8]), 0, 1))                      # binary
        trend_bin = int(np.clip(int(state[9] * 2), 0, 2))                  # 3 bins
        sw_bin    = int(np.clip(int(state[10] * 2), 0, 2))                 # 3 bins
        return tuple(cont_bins + [path_bin, trend_bin, sw_bin])

    def select_action(self, state: np.ndarray) -> int:
        self._update_epsilon()
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        idx = self._discretise(state)
        return int(np.argmax(self.q_table[idx]))

    def _update_epsilon(self):
        self.epsilon = (self.hp.epsilon_end +
                        (self.hp.epsilon_start - self.hp.epsilon_end) *
                        math.exp(-1.0 * self.steps_done / self.hp.epsilon_decay))
        self.steps_done += 1

    def update(self, state, action, reward, next_state, done):
        idx  = self._discretise(state)
        nidx = self._discretise(next_state)
        target = reward + (0.0 if done else self.hp.gamma * np.max(self.q_table[nidx]))
        self.q_table[idx][action] += 0.01 * (target - self.q_table[idx][action])

    def save(self, path):
        np.save(path + '.npy', self.q_table)
        np.save(path + '_meta.npy', np.array([self.epsilon, self.steps_done]))

    def load(self, path):
        self.q_table    = np.load(path + '.npy')
        meta            = np.load(path + '_meta.npy')
        self.epsilon    = float(meta[0])
        self.steps_done = int(meta[1])


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------
class DQNAgent:    
    STATE_DIM = 11   
    N_ACTIONS = 2

    @property
    def epsilon(self):
        return self._fallback.epsilon if not self._use_torch else self._epsilon_val

    @epsilon.setter
    def epsilon(self, v):
        if not self._use_torch:
            self._fallback.epsilon = v
        else:
            self._epsilon_val = v

    def __init__(self,
                 hp: Optional[HyperParams] = None,
                 device: Optional[str] = None,
                 log_dir: str = 'logs/tb'):

        if not TORCH_AVAILABLE:
            self._fallback  = NumpyQAgent(hp=hp or HyperParams())
            self._use_torch = False
            return

        self._use_torch = True
        self.hp         = hp or HyperParams()
        self.device     = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.online_net = QNetwork(self.STATE_DIM, self.N_ACTIONS, self.hp).to(self.device)
        self.target_net = QNetwork(self.STATE_DIM, self.N_ACTIONS, self.hp).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=self.hp.lr)
        self.loss_fn   = nn.SmoothL1Loss()
        self.buffer    = ReplayBuffer(self.hp.buffer_capacity)

        self._epsilon_val  = self.hp.epsilon_start
        self.steps_done    = 0
        self._update_count = 0

        self._writer = None
        if TB_AVAILABLE and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self._writer = SummaryWriter(log_dir)

        logger.info('DQNAgent on %s | STATE_DIM=%d | hp=%s',
                    self.device, self.STATE_DIM, self.hp)
   
    @staticmethod
    def _sanitise(state: np.ndarray) -> np.ndarray:
        
        return np.clip(np.asarray(state, dtype=np.float32), 0.0, 1.0)

    # ------------------------------------------------------------------ #
    # Action selection                                                     #
    # ------------------------------------------------------------------ #
    def select_action(self, state: np.ndarray) -> int:
        
        if not self._use_torch:
            return self._fallback.select_action(self._sanitise(state))

        self._epsilon_val = (
            self.hp.epsilon_end +
            (self.hp.epsilon_start - self.hp.epsilon_end) *
            math.exp(-1.0 * self.steps_done / self.hp.epsilon_decay)
        )
        self.steps_done += 1

        if random.random() < self._epsilon_val:
            return random.randrange(self.N_ACTIONS)

        s = self._sanitise(state)
        with torch.no_grad():
            t = torch.FloatTensor(s).unsqueeze(0).to(self.device)
            return int(self.online_net(t).argmax(dim=1).item())

    def greedy_action(self, state: np.ndarray) -> int:
        if not self._use_torch:
            idx = self._fallback._discretise(self._sanitise(state))
            return int(np.argmax(self._fallback.q_table[idx]))
        s = self._sanitise(state)
        with torch.no_grad():
            t = torch.FloatTensor(s).unsqueeze(0).to(self.device)
            return int(self.online_net(t).argmax(dim=1).item())

    # ------------------------------------------------------------------ #
    # Experience storage                                                   #
    # ------------------------------------------------------------------ #
    def store(self, state, action, reward, next_state, done):
        if not self._use_torch:
            self._fallback.update(
                self._sanitise(state), action, reward,
                self._sanitise(next_state), done)
            return
        self.buffer.push(
            self._sanitise(state), action, reward,
            self._sanitise(next_state), done)

    # ------------------------------------------------------------------ #
    # Training step                                                        #
    # ------------------------------------------------------------------ #
    def train_step(self) -> Optional[float]:
        if not self._use_torch:
            return None
        if len(self.buffer) < self.hp.min_buffer_size:
            return None

        batch       = self.buffer.sample(self.hp.batch_size)
        states      = torch.FloatTensor([t.state      for t in batch]).to(self.device)
        actions     = torch.LongTensor( [t.action     for t in batch]).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor([t.reward     for t in batch]).to(self.device)
        next_states = torch.FloatTensor([t.next_state for t in batch]).to(self.device)
        dones       = torch.FloatTensor([t.done       for t in batch]).to(self.device)

        q_values = self.online_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q       = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q     = rewards + self.hp.gamma * next_q * (1 - dones)

        loss = self.loss_fn(q_values, target_q)
        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.hp.grad_clip)
        self.optimiser.step()

        self._update_count += 1
        if self._update_count % self.hp.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        loss_val = loss.item()
        if self._writer:
            self._writer.add_scalar('train/loss',    loss_val,         self._update_count)
            self._writer.add_scalar('train/epsilon', self._epsilon_val, self._update_count)
        return loss_val

    # ------------------------------------------------------------------ #
    # Checkpoint I/O                                                       #
    # ------------------------------------------------------------------ #
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        if not self._use_torch:
            self._fallback.save(path)
            return
        torch.save({
            'online_net':    self.online_net.state_dict(),
            'target_net':    self.target_net.state_dict(),
            'optimiser':     self.optimiser.state_dict(),
            'epsilon':       self._epsilon_val,
            'steps_done':    self.steps_done,
            'update_count':  self._update_count,
            'hp':            vars(self.hp),
        }, path)
        logger.info('Checkpoint saved → %s', path)

    def load(self, path: str):
        if not self._use_torch:
            self._fallback.load(path)
            return
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt['online_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimiser.load_state_dict(ckpt['optimiser'])
        self._epsilon_val  = ckpt.get('epsilon',      self.hp.epsilon_end)
        self.steps_done    = ckpt.get('steps_done',   0)
        self._update_count = ckpt.get('update_count', 0)
        logger.info('Checkpoint loaded ← %s  (step=%d ε=%.3f)',
                    path, self.steps_done, self._epsilon_val)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        if not self._use_torch:
            idx = self._fallback._discretise(self._sanitise(state))
            return self._fallback.q_table[idx].copy()
        s = self._sanitise(state)
        with torch.no_grad():
            t = torch.FloatTensor(s).unsqueeze(0).to(self.device)
            return self.online_net(t).cpu().numpy().flatten()
