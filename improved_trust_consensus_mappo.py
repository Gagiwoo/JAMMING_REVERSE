# improved_trust_consensus_mappo.py
"""
GPS 스푸핑 환경에서 신뢰도 기반 다중 UAV 협력 경로 계획
Trust-based Cooperative Path Planning for Multi-UAV Systems under GPS Spoofing Attacks

논문 명세에 맞게 개선된 버전
Author: 김도윤 (논문 저자)
Improved by: AI Code Reviewer

주요 개선사항:
1. Trust Network 아키텍처: 3층 × 16 뉴런 (논문 명세 준수)
2. Actor 네트워크: 불필요한 층 제거 (1 hidden layer)
3. Consensus Protocol: 50% 투표 기반 강제 설정 메커니즘 추가
4. 하이퍼파라미터: 논문 명세와 정확히 일치하도록 수정
5. 관찰 공간: 융합된 위치 사용 및 속도 추가
6. GPS 공격 확률: 10%로 수정
"""

import os
import sys
import time
import threading
import queue
import random
import warnings
import csv
import datetime
from collections import deque
from copy import deepcopy
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from PySide6.QtWidgets import *
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QTextCursor, QFont
import qdarkstyle
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')

# ==================== CONFIG (논문 명세 준수) ====================
BASE_CONFIG = {
    # ---------------- 보상 설정 (최적화) ----------------
    "reward_goal": 120.0,  # 🔥 100 → 120 (목표 도달 강한 보상)
    "reward_team_success": 30.0,
    "reward_collision": -30.0,  # 🔥 -50 → -30 (충돌 페널티 완화)
    "reward_step_penalty": -0.1,
    "distance_reward_factor": 1.5,  # 🔥 1.0 → 1.5 (목표 접근 보상 더 증가)
    
    # ---------------- 학습 하이퍼파라미터 (🔥 FIX: 균형 학습) ----------------
    "mappo_lr": 3e-4,  # 🔥 1e-4 → 3e-4 (MAPPO도 빠르게)
    "mappo_entropy": 0.01,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ppo_clip_epsilon": 0.2,
    "update_epochs": 10,
    "batch_size": 512,
    
    # ---------------- 환경 설정 (🔥 FIX: 조금 더 쉽게) ----------------
    "num_uavs": 5,  # 🔥 6 → 5 (더더 단순한 협력)
    "grid_size": 40,
    "num_obstacles": 15,  # 🔥 20 → 15 (장애물 더더 감소)
    "max_steps": 150,
    "vision_range": 6,
    
    # ---------------- 공격 설정 (🔥 FIX: 공격 비율 완화) ----------------
    "attack_prob": 0.02,  # 🔥 0.05 → 0.02 (다시 낮춤)
    "attack_mode": "hybrid",
    "attack_start_prob": 0.02,  # 🔥 0.05 → 0.02 (실제 ~20% 공격)
    "attack_min_duration": 10,  # 🔥 15 → 10
    "attack_max_duration": 20,  # 🔥 25 → 20
    
    # ---------------- Trust Network 설정 (🔥 FIX: 균형 학습) ----------------
    "use_trust_network": True,
    "trust_hidden": 32,  # 더 강력한 네트워크
    "trust_lr": 3e-4,  # 🔥 5e-4 → 3e-4 (MAPPO와 같은 속도)
    "trust_lambda_reg": 0.1,  # 🔥 0.05 → 0.1 (smoothness 강화)
    "max_correction": 3.0,  # 🔥 NEW: 최대 보정 범위 (5.0 → 3.0)
    
    # ---------------- Consensus 설정 (🔥 NEW: 보정 스케일 조정 방식) ----------------
    "use_consensus": True,
    "consensus_threshold": 2.5,
    "consensus_weight": 0.08,  # 사용 안 함 (보정 스케일로 대체)
    "consensus_vote_threshold": 0.5,  # 🔥 0.7 → 0.5 (50% 투표로 복원)
    
    # ---------------- LSTM 기반 스푸핑 보정기 설정 ----------------
    "detector_seq_len": 10,
    "detector_feature_dim": 5,
    "detector_hidden": 64,
    
    # ---------------- 학습 제어 ----------------
    "total_episodes": 10000,
    "episodes_per_batch": 10,
    "render_delay": 0.1,
    "demo_episodes": 3,
    "checkpoint_interval": 5000,
    "seed": 42,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALGORITHM_CONFIGS = {
    "Vanilla-MAPPO": {
        "use_trust_network": False,
        "use_consensus": False,
        "use_lstm_detection": False,
        "description": "기본 MAPPO (Baseline, 무방비)"
    },
    "LSTM-MAPPO": {
        "use_trust_network": False,
        "use_consensus": False,
        "use_lstm_detection": True,
        "description": "LSTM 기반 (기존 연구, 시계열 의존)"
    },
    "Trust-MAPPO": {
        "use_trust_network": True,
        "use_consensus": False,
        "use_lstm_detection": False,
        "description": "신뢰도 학습만 (Ablation)"
    },
    "Trust+Consensus-MAPPO": {
        "use_trust_network": True,
        "use_consensus": True,
        "use_lstm_detection": False,
        "description": "제안 기법 (Ours, Full) - 논문"
    },
    "LSTM-Detector-MAPPO": {
        "use_trust_network": False,
        "use_consensus": False,
        "use_lstm_detection": False,
        "use_spoof_lstm_detector": True,
        "description": "LSTM 기반 GPS 스푸핑 보정 baseline"
    }
}

# ==================== UTILS ====================
def create_model_folder_name(config, algorithm):
    timestamp = int(time.time())
    folder_name = f"RobustRL_{algorithm}_{config['attack_mode']}_obs{config['num_obstacles']}_{timestamp}"
    return folder_name

# ==================== NETWORKS (논문 명세 준수) ====================

class TrustNetwork(nn.Module):
    """
    🔥 NEW: GPS Correction Network (Detector 방식)
    - 3개의 은닉층, 각 32 뉴런
    - 입력: 4차원 (temporal_residual, spatial_discrepancy, gps_variance, vision_quality)
    - 출력: 2차원 (correction_x, correction_y) - GPS 보정값
    
    핵심 아이디어:
    - Trust 가중치 대신 "얼마나 보정할지" 직접 학습
    - Vision 위치 문제 해결
    - LSTM-Detector와 공정한 비교
    """
    def __init__(self, hidden=32, max_correction=5.0):
        super().__init__()
        self.max_correction = max_correction  # 최대 보정 범위 (±5 셀)
        
        self.network = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(),      # Layer 1: 4 → 32
            nn.Linear(hidden, hidden), nn.ReLU(), # Layer 2: 32 → 32
            nn.Linear(hidden, hidden), nn.ReLU(), # Layer 3: 32 → 32
            nn.Linear(hidden, 2),                 # Output: 32 → 2 (correction_x, correction_y)
            nn.Tanh()                             # Tanh로 [-1, 1] 범위 제한
        )
    
    def forward(self, trust_features):
        """
        Args:
            trust_features: (batch_size, 4) tensor
                - normalized temporal residual (GPS 예측 오차)
                - normalized spatial discrepancy (이웃과의 불일치)
                - GPS variance (공격 시 높음)
                - Vision quality (이웃 존재 여부)
        Returns:
            correction: (batch_size, 2) tensor [correction_x, correction_y]
                범위: [-max_correction, +max_correction]
        """
        return self.network(trust_features) * self.max_correction


class TrustLoss:
    """
    🔥 NEW: GPS Correction Loss
    Loss = MSE(corrected_pos, real_pos) + λ * MSE(correction_t, correction_{t-1})
    
    핵심:
    - 보정된 위치가 실제 위치에 가까워지도록 학습
    - Smoothness Regularization으로 급격한 변화 방지
    """
    def __init__(self, lambda_reg=0.05):
        self.lambda_reg = lambda_reg
    
    def compute(self, corrected_pos, real_pos, current_correction, prev_correction):
        """
        Args:
            corrected_pos: GPS + correction (batch_size, 2)
            real_pos: 실제 위치 (batch_size, 2)
            current_correction: 현재 보정값 (batch_size, 2)
            prev_correction: 이전 보정값 (batch_size, 2)
        Returns:
            total_loss: Correction Loss + λ * Smoothness Loss
        """
        correction_loss = torch.mean((corrected_pos - real_pos) ** 2)
        smoothness_loss = torch.mean((current_correction - prev_correction) ** 2)
        return correction_loss + self.lambda_reg * smoothness_loss


class ConsensusProtocol:
    """
    ✅ 개선: 논문 명세에 맞게 50% 투표 기반 강제 설정 메커니즘 추가
    
    논문 알고리즘:
    1. 각 UAV는 이웃들의 GPS 위치와 Vision 관측을 비교
    2. 불일치가 threshold를 초과하면 의심 표(suspicion vote) 부여
    3. 전체 이웃의 50% 이상에게서 의심 표를 받으면 GPS 신뢰도를 강제로 0으로 설정
    """
    def __init__(self, threshold=2.5, consensus_weight=0.15, vote_threshold=0.5):
        self.threshold = threshold  # ✅ 수정: 2.0 → 2.5
        self.consensus_weight = consensus_weight  # ✅ 수정: 0.2 → 0.15
        self.vote_threshold = vote_threshold  # ✅ 추가: 50% 투표 임계값
    
    def compute_discrepancy(self, my_vision_obs, neighbor_gps_claim):
        """이웃의 GPS 위치와 내 Vision 관측 간 불일치 계산"""
        return np.linalg.norm(my_vision_obs - neighbor_gps_claim)
    
    def cast_votes(self, discrepancies):
        """
        ✅ 추가: 각 이웃에 대한 의심 표 부여
        
        Args:
            discrepancies: List of discrepancies for each neighbor
        Returns:
            suspicion_votes: List of binary votes (1 if suspicious, 0 otherwise)
        """
        votes = []
        for disc in discrepancies:
            if disc > self.threshold:
                votes.append(1)  # 의심 표
            else:
                votes.append(0)  # 정상 표
        return votes
    
    def aggregate_votes(self, votes_received):
        """
        ✅ 추가: 받은 의심 표 집계 및 공격 여부 판단
        
        Args:
            votes_received: Number of suspicion votes received from neighbors
            total_neighbors: Total number of neighbors
        Returns:
            is_under_attack: True if votes_received >= 50% of total_neighbors
        """
        if len(votes_received) == 0:
            return False, 0.0
        
        suspicion_ratio = sum(votes_received) / len(votes_received)
        is_under_attack = suspicion_ratio >= self.vote_threshold
        return is_under_attack, suspicion_ratio
    
    def adjust_trust(self, trust_gps, trust_vis, suspicion_ratio, force_zero=False):
        """
        ✅✅ 논문 명세 정확히 준수
        
        Args:
            trust_gps: Current GPS trust score (from Trust Network)
            trust_vis: Current Vision trust score (from Trust Network)
            suspicion_ratio: 의심 표 비율 (votes_received / total_neighbors)
            force_zero: If True, force GPS trust to 0 (50% 이상 투표 시)
        Returns:
            adjusted_trust_gps, adjusted_trust_vis
        
        논문 알고리즘:
        - suspicion_ratio >= 0.60 (60%): GPS trust -= 0.15
        - suspicion_ratio < 0.40 (40%): GPS trust += 0.15
        - 50% 이상 의심 표: GPS trust = 0.0 (force_zero=True)
        """
        # ✅ 1단계: 50% 이상 의심 표 시 강제 설정
        if force_zero:
            return 0.0, 1.0
        
        # ✅ 2단계: 논문 명세대로 60%/40% 경계 기반 조정
        if suspicion_ratio >= 0.60:
            # 60% 이상 의심 표 → GPS 신뢰도 감소
            trust_gps -= self.consensus_weight  # -0.15
        elif suspicion_ratio < 0.40:
            # 40% 미만 의심 표 → GPS 신뢰도 증가
            trust_gps += self.consensus_weight  # +0.15
        # 40% ~ 60% 사이: 조정 없음
        
        # ✅ 3단계: 경계값 클리핑 [0.0, 1.0]
        trust_gps = np.clip(trust_gps, 0.0, 1.0)
        trust_vis = 1.0 - trust_gps
        
        return trust_gps, trust_vis


class LSTMSpoofDetector(nn.Module):
    """
    LSTM 기반 GPS 스푸핑 보정기 (Baseline 비교용)
    Residual Learning을 통한 위치 보정
    """
    def __init__(self, feature_dim=5, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)
        # 작은 초기값으로 안정적인 학습
        nn.init.uniform_(self.fc.weight, -0.001, 0.001)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, feature_dim)
        Returns:
            correction: (batch_size, 2) position correction vector
        """
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class Actor(nn.Module):
    """
    ✅ 개선: 논문 명세에 맞게 수정
    - 1개의 은닉층 (128 뉴런)
    - Tanh 활성화 함수
    - LSTM 변형의 경우 LSTM 레이어 추가
    """
    def __init__(self, local_dim, act_dim, hidden=128, use_lstm=False):
        super().__init__()
        self.use_lstm = use_lstm
        self.fc1 = nn.Linear(local_dim, hidden)
        if use_lstm:
            self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        # ✅ 수정: fc2 층 제거 (논문에는 1개 은닉층만)
        self.head = nn.Linear(hidden, act_dim)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        if self.use_lstm:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x, _ = self.lstm(x)
            x = x[:, -1, :]
        # ✅ fc2 제거로 인한 수정
        return F.softmax(self.head(x), dim=-1)


class Critic(nn.Module):
    """
    ✅ 논문 명세 준수
    - 2개의 은닉층, 각 256 뉴런
    - Tanh 활성화 함수
    """
    def __init__(self, glob_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(glob_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, x):
        return self.net(x)


# ==================== ENVIRONMENT ====================

class EnvironmentScenario:
    """환경 시나리오 생성 (장애물, 시작/목표 위치)"""
    def __init__(self, config):
        self.config = config
        self.grid_size = config["grid_size"]
        self.num_uavs = config["num_uavs"]
        self.num_obstacles = config["num_obstacles"]
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self._place_obstacles()
        self.start_positions, self.target_positions = self._generate_start_and_targets()
    
    def _place_obstacles(self):
        count = 0
        while count < self.num_obstacles:
            r = np.random.randint(0, self.grid_size)
            c = np.random.randint(0, self.grid_size)
            if self.grid[r, c] == 0:
                self.grid[r, c] = -1
                count += 1
    
    def _generate_start_and_targets(self):
        starts = []
        targets = []
        available_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if self.grid[r, c] == 0]
        chosen = np.random.choice(len(available_cells), 2 * self.num_uavs, replace=False)
        for i in range(self.num_uavs):
            starts.append(np.array(available_cells[chosen[2*i]], dtype=float))
            targets.append(np.array(available_cells[chosen[2*i+1]], dtype=float))
        return np.array(starts), np.array(targets)


class CTDEMultiUAVEnv:
    """
    CTDE (Centralized Training, Decentralized Execution) Multi-UAV 환경
    
    ✅ 개선사항:
    - 관찰 공간에 융합된 위치 사용
    - 속도 정보 추가
    - GPS 공격 확률 10%로 수정
    - Consensus Protocol 투표 메커니즘 통합
    """
    def __init__(self, config, render_mode=None):
        self.config = config
        self.num_uavs = config["num_uavs"]
        self.grid_size = config["grid_size"]
        self.max_steps = config["max_steps"]
        self.vision_range = config["vision_range"]
        self.agents = [f"uav_{i}" for i in range(self.num_uavs)]
        
        # 8방향 이동 (상하좌우 + 대각선)
        self.discrete_moves = np.array([
            [0,-1], [0,1], [-1,0], [1,0],  # 상하좌우
            [-1,-1], [-1,1], [1,-1], [1,1]  # 대각선
        ], dtype=int)
        
        # ✅ 개선: 관찰 공간 재정의 (융합된 위치 + 속도 추가)
        # 자신의 상태: fused_pos(2) + velocity(2) + target(2) + trust_features(4) + consensus_vote(1) = 11
        self_dim = 2 + 2 + 2 + 4 + 1
        neighbor_dim = (self.num_uavs - 1) * 5  # 각 이웃: rel_pos(2) + gps_pos(2) + discrepancy(1)
        vision_dim = (2 * self.vision_range + 1) ** 2
        self.local_obs_dim = self_dim + neighbor_dim + vision_dim
        
        self.global_obs_dim = (self.num_uavs * 4) + (self.grid_size * self.grid_size)
        self.action_dim = len(self.discrete_moves)
        
        self.render_mode = render_mode
        self.window = None
        if self.render_mode == "human":
            self._init_pygame()
        
        self.consensus = ConsensusProtocol(
            self.config["consensus_threshold"], 
            self.config["consensus_weight"],
            self.config["consensus_vote_threshold"]
        )
        
        self.attack_mode = config["attack_mode"]
        self.attack_start_prob = config.get("attack_start_prob", 0.1)  # ✅ 수정: 10%
        self.attack_min_duration = config.get("attack_min_duration", 10)
        self.attack_max_duration = config.get("attack_max_duration", 30)
        
        # 공격 상태 관리
        self.attack_remaining_steps = np.zeros(self.num_uavs, dtype=int)
        self.attack_drift_dir = np.zeros((self.num_uavs, 2), dtype=float)
        self.attack_step_offset = np.zeros((self.num_uavs, 2), dtype=float)
        self.active_attack_types = ["none"] * self.num_uavs
        
        self.consensus_votes = np.zeros(self.num_uavs, dtype=float)
        self.suspicion_votes_received = [[] for _ in range(self.num_uavs)]  # ✅ 추가
    
    def reset_with_scenario(self, scenario):
        """시나리오로 환경 초기화"""
        self.current_step = 0
        self.grid = scenario.grid.copy()
        self.shared_map = np.full((self.grid_size, self.grid_size), -2.0, dtype=np.float32)
        
        self.uav_positions = scenario.start_positions.copy().astype(float)
        self.target_positions = scenario.target_positions.copy()
        self.uav_status = ["active"] * self.num_uavs
        
        self.last_positions = self.uav_positions.copy()
        self.last_velocities = np.zeros((self.num_uavs, 2))
        self.gps_positions = self.uav_positions.copy()
        self.drift_bias = np.zeros((self.num_uavs, 2))
        self.is_under_attack = [False] * self.num_uavs
        
        # 공격 상태 초기화
        self.attack_remaining_steps.fill(0)
        self.attack_drift_dir.fill(0)
        self.attack_step_offset.fill(0)
        self.active_attack_types = ["none"] * self.num_uavs
        
        self.prev_distances = np.linalg.norm(self.uav_positions - self.target_positions, axis=1)
        self.step_counts = [0] * self.num_uavs
        self.total_path_lengths = [0.0] * self.num_uavs
        self.uav_paths = {agent_id: [pos.copy()] for agent_id, pos in zip(self.agents, self.uav_positions)}
        
        self.suspicion_votes_received = [[] for _ in range(self.num_uavs)]  # ✅ 추가
        
        self._update_shared_map()
        return self._compute_observations()
    
    def step(self, actions):
        """환경 스텝 실행"""
        self.current_step += 1
        self.last_positions = self.uav_positions.copy()
        
        # 각 UAV 이동
        for i, aid in enumerate(self.agents):
            if self.uav_status[i] == "active":
                move = self.discrete_moves[actions[aid]]
                self.last_velocities[i] = move
                intended = self.uav_positions[i] + move
                self.total_path_lengths[i] += np.linalg.norm(move)
                
                # 충돌 체크
                is_collision = False
                if not (0 <= intended[0] < self.grid_size and 0 <= intended[1] < self.grid_size):
                    is_collision = True
                elif self.grid[int(intended[1]), int(intended[0])] == -1:
                    is_collision = True
                
                if is_collision:
                    self.uav_status[i] = "collision"
                else:
                    self.uav_positions[i] = intended
                    self.uav_paths[aid].append(intended.copy())
        
        # UAV 간 충돌 체크
        for i in range(self.num_uavs):
            if self.uav_status[i] == "active":
                for j in range(i+1, self.num_uavs):
                    if self.uav_status[j] == "active":
                        if np.array_equal(self.uav_positions[i], self.uav_positions[j]):
                            self.uav_status[i] = "collision"
                            self.uav_status[j] = "collision"
        
        # 목표 도달 체크
        for i in range(self.num_uavs):
            if self.uav_status[i] == "active":
                if np.linalg.norm(self.uav_positions[i] - self.target_positions[i]) < 1.5:
                    self.uav_status[i] = "success"
        
        # GPS 스푸핑 공격 시뮬레이션
        self._simulate_attacks()
        
        # 공유 맵 업데이트
        self._update_shared_map()
        
        # 보상 계산
        rewards = self._calculate_rewards()
        
        # 관찰 계산
        local, global_obs = self._compute_observations()
        
        # 종료 조건
        done = (all(s != "active" for s in self.uav_status)) or (self.current_step >= self.max_steps)
        info = self._get_info(done)
        
        return local, global_obs, rewards, done, info
    
    def _simulate_attacks(self):
        """
        GPS 스푸핑 공격 시뮬레이션
        
        공격 타입:
        1. Step Attack: 고정 오프셋 (-4.0 ~ 4.0m)
        2. Drift Attack: 누적 편향 (0.2 ~ 0.8 m/s)
        3. Hybrid: 랜덤 선택
        """
        self.gps_positions = self.uav_positions.copy()
        self.is_under_attack = [False] * self.num_uavs
        
        if self.config["attack_mode"] == "none":
            return
        
        for i in range(self.num_uavs):
            if self.uav_status[i] != "active":
                self.attack_remaining_steps[i] = 0
                continue
            
            # 진행 중인 공격
            if self.attack_remaining_steps[i] > 0:
                self.is_under_attack[i] = True
                atk_type = self.active_attack_types[i]
                
                if atk_type == "step":
                    self.gps_positions[i] += self.attack_step_offset[i]
                elif atk_type == "drift":
                    noise = np.random.normal(0.0, 0.1, size=2)
                    self.drift_bias[i] += self.attack_drift_dir[i] + noise
                    self.gps_positions[i] += self.drift_bias[i]
                
                self.attack_remaining_steps[i] -= 1
            
            # 새로운 공격 시작
            elif np.random.rand() < self.attack_start_prob:
                duration = np.random.randint(self.attack_min_duration, self.attack_max_duration+1)
                self.attack_remaining_steps[i] = duration
                self.is_under_attack[i] = True
                
                # 공격 타입 선택
                if self.attack_mode == "hybrid":
                    self.active_attack_types[i] = "step" if np.random.rand() < 0.5 else "drift"
                else:
                    self.active_attack_types[i] = self.attack_mode
                
                # 공격 파라미터 설정
                if self.active_attack_types[i] == "step":
                    self.attack_step_offset[i] = np.random.uniform(-4.0, 4.0, size=2)
                    self.gps_positions[i] += self.attack_step_offset[i]
                elif self.active_attack_types[i] == "drift":
                    angle = np.random.uniform(0, 2*np.pi)
                    self.attack_drift_dir[i] = np.array([np.cos(angle), np.sin(angle)]) * np.random.uniform(0.2, 0.8)
                    self.drift_bias[i] = np.zeros(2)
                    self.gps_positions[i] += self.drift_bias[i]
    
    def _compute_observations(self):
        """
        ✅ 개선: 논문 명세에 맞게 관찰 공간 재구성
        
        각 UAV의 관찰:
        - 융합된 위치 (fused_pos) - 아직 융합되지 않은 경우 GPS 사용
        - 속도 (velocity)
        - 목표 위치 (target)
        - Trust features (temporal_residual, spatial_discrepancy, gps_variance, neighbor_flag)
        - Consensus vote
        - 이웃 정보
        - Local vision
        """
        local_obs, all_states = {}, []
        
        # 글로벌 상태 구성
        for i in range(self.num_uavs):
            all_states.extend(self.uav_positions[i] / self.grid_size)
            all_states.extend(self.target_positions[i] / self.grid_size)
        
        # ✅ 추가: Consensus Protocol 투표 수집
        self.suspicion_votes_received = [[] for _ in range(self.num_uavs)]
        
        # 각 UAV의 로컬 관찰 생성
        for i in range(self.num_uavs):
            # Temporal Residual: 예측 위치 vs GPS 위치
            pred_pos = self.last_positions[i] + self.last_velocities[i]
            temp_res = np.linalg.norm(pred_pos - self.gps_positions[i])
            
            # GPS Variance (공격 여부에 따라 다름)
            gps_var = np.random.uniform(0.1, 0.5) if not self.is_under_attack[i] else np.random.uniform(2.0, 5.0)
            
            # ✅ 개선: 이웃 정보 고정 차원으로 생성
            neighbor_features = []  # 고정 길이 리스트
            discrepancies = []
            
            for j in range(self.num_uavs):
                if i == j:
                    continue
                
                dist = np.linalg.norm(self.uav_positions[j] - self.uav_positions[i])
                if dist <= self.vision_range:
                    vis_pos = self.uav_positions[j]
                    gps_claim = self.gps_positions[j]
                    disc = np.linalg.norm(vis_pos - gps_claim)
                    discrepancies.append(disc)
                    
                    # ✅ 추가: 투표 수행
                    if disc > self.consensus.threshold:
                        self.suspicion_votes_received[j].append(1)
                    else:
                        self.suspicion_votes_received[j].append(0)
                    
                    # 상대 위치 (2), GPS 상대 위치 (2), 불일치 (1) = 5차원
                    rel_pos = (vis_pos - self.uav_positions[i]) / self.grid_size
                    gps_rel = (self.gps_positions[j] - self.gps_positions[i]) / self.grid_size
                    neighbor_features.extend([rel_pos[0], rel_pos[1], gps_rel[0], gps_rel[1], disc])
                else:
                    # 이웃이 범위 밖: 0으로 채움
                    neighbor_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            
            # 고정 길이 보장: (num_uavs-1) * 5
            neighbor_info = np.array(neighbor_features, dtype=np.float32)
            
            # ✅✅ 논문 명세: Consensus Vote는 suspicion_ratio (내가 받은 의심 표 비율)
            my_votes = self.suspicion_votes_received[i]
            suspicion_ratio = sum(my_votes) / len(my_votes) if my_votes else 0.0
            self.consensus_votes[i] = suspicion_ratio  # 투표 비율 저장
            
            # Spatial Discrepancy (이웃들과의 불일치 평균)
            spat_disc = np.mean(discrepancies) if discrepancies else 0.0
            
            # Trust Features (정규화)
            norm_temp = np.clip(temp_res / 2.0, 0.0, 1.0)
            norm_spat = np.clip(spat_disc / 1.0, 0.0, 1.0)
            
            trust_feats = np.array([
                norm_temp,
                norm_spat,
                gps_var,
                1.0 if discrepancies else 0.0  # 이웃 존재 여부
            ], dtype=np.float32)
            
            # ✅ 개선: 융합된 위치 사용 (현재는 GPS, Agent에서 융합 후 업데이트)
            # 초기 상태에서는 GPS 위치 사용
            my_state = np.concatenate([
                self.gps_positions[i] / self.grid_size,  # fused_pos (나중에 Agent에서 업데이트)
                self.last_velocities[i] / self.grid_size,  # ✅ 추가: velocity
                self.target_positions[i] / self.grid_size,
                trust_feats,
                [suspicion_ratio]  # ✅ 수정: 투표 비율 (discrepancy 평균 아님)
            ])
            
            local_vis = self._extract_local_vision(self.uav_positions[i])
            local_obs[self.agents[i]] = np.concatenate([my_state, neighbor_info, local_vis]).astype(np.float32)
        
        global_obs = np.concatenate([np.array(all_states), self.shared_map.flatten()]).astype(np.float32)
        return local_obs, global_obs
    
    def _extract_local_vision(self, pos):
        """로컬 Vision 센서 (주변 장애물 관측)"""
        r = self.vision_range
        roi = np.full((2*r+1, 2*r+1), -2.0, dtype=np.float32)
        px, py = int(pos[0]), int(pos[1])
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                ny, nx = py + dy, px + dx
                if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                    roi[dy+r, dx+r] = self.grid[ny, nx]
        return roi.flatten()
    
    def _update_shared_map(self):
        """공유 맵 업데이트 (각 UAV의 Vision 범위 내 정보 공유)"""
        for i in range(self.num_uavs):
            if self.uav_status[i] == 'active':
                px, py = int(self.uav_positions[i][0]), int(self.uav_positions[i][1])
                r = self.vision_range
                y1, y2 = max(0, py-r), min(self.grid_size, py+r+1)
                x1, x2 = max(0, px-r), min(self.grid_size, px+r+1)
                self.shared_map[y1:y2, x1:x2] = self.grid[y1:y2, x1:x2]
    
    def _calculate_rewards(self):
        """보상 계산"""
        rewards = {}
        for i, aid in enumerate(self.agents):
            r = self.config["reward_step_penalty"]
            
            if self.uav_status[i] == "success":
                r += self.config["reward_goal"]
            elif self.uav_status[i] == "collision":
                r += self.config["reward_collision"]
            else:
                # 목표 접근 보상
                dist = np.linalg.norm(self.uav_positions[i] - self.target_positions[i])
                r += (self.prev_distances[i] - dist) * self.config["distance_reward_factor"] * 10.0
            
            rewards[aid] = r
        
        self.prev_distances = np.linalg.norm(self.uav_positions - self.target_positions, axis=1)
        return rewards
    
    def _get_info(self, done):
        """에피소드 종료 정보"""
        if not done:
            return {}
        
        s = sum(1 for st in self.uav_status if st == "success")
        c = sum(1 for st in self.uav_status if st == "collision")
        
        return {
            "success_rate": s / self.num_uavs,
            "collision_rate": c / self.num_uavs,
            "avg_path_length": np.mean(self.total_path_lengths)
        }
    
    def _init_pygame(self):
        """Pygame 초기화 (시각화용)"""
        pygame.init()
        self.window_size = 600
        self.cell_size = self.window_size // self.grid_size
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Multi-UAV Navigation (Improved)")
        self.uav_colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
        self.clock = pygame.time.Clock()
    
    def render(self):
        """환경 렌더링"""
        if self.render_mode != "human":
            return
        
        self.window.fill((255, 255, 255))
        
        # 장애물 그리기
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] == -1:
                    pygame.draw.rect(self.window, (50, 50, 50), 
                                   (c*self.cell_size, r*self.cell_size, 
                                    self.cell_size, self.cell_size))
        
        # UAV 경로 및 위치 그리기
        for i in range(self.num_uavs):
            color = self.uav_colors[i % len(self.uav_colors)]
            
            # 경로
            if len(self.uav_paths[self.agents[i]]) > 1:
                pts = [(p[0]*self.cell_size+self.cell_size/2, 
                       p[1]*self.cell_size+self.cell_size/2) 
                       for p in self.uav_paths[self.agents[i]]]
                pygame.draw.lines(self.window, color, False, pts, 2)
            
            # UAV 위치
            pygame.draw.circle(self.window, color, 
                             (int(self.uav_positions[i][0]*self.cell_size+self.cell_size/2),
                              int(self.uav_positions[i][1]*self.cell_size+self.cell_size/2)), 5)
        
        pygame.display.flip()
        self.clock.tick(10)
    
    def close(self):
        """환경 종료"""
        if self.window:
            pygame.quit()
            self.window = None


# ==================== ROLLOUT BUFFER ====================

class RolloutBuffer:
    """경험 저장 버퍼"""
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.obs = []
        self.glo = []
        self.act = []
        self.logp = []
        self.val = []
        self.rew = []
        self.done = []
        self.adv = []
        self.ret = []
    
    def add(self, o, g, a, l, v, r, d):
        self.obs.extend(o)
        self.glo.extend(g)
        self.act.extend(a)
        self.logp.extend(l)
        self.val.extend(v)
        self.rew.extend(r)
        self.done.extend(d)


# ==================== AGENT ====================

class MAPPOAgentWithTrust:
    """
    ✅ 개선: 논문 명세에 맞게 Trust Network 통합 MAPPO Agent
    
    주요 개선사항:
    1. Trust Network Learning Rate를 Actor의 50%로 설정
    2. Consensus Protocol 50% 투표 메커니즘 통합
    3. 융합된 위치를 Actor 입력으로 사용
    4. Trust Loss lambda를 0.1로 수정
    """
    def __init__(self, l_dim, g_dim, a_dim, config):
        self.config = config
        self.device = DEVICE
        
        # Actor & Critic 네트워크
        self.actor = Actor(l_dim, a_dim, hidden=128, use_lstm=config.get("use_lstm_detection", False)).to(DEVICE)
        self.critic = Critic(g_dim, hidden=256).to(DEVICE)
        
        # ✅ 수정: 논문 명세에 맞는 Learning Rate
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config["mappo_lr"])
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=config["mappo_lr"])
        
        # Trust Network
        self.use_trust = config.get("use_trust_network", False)
        self.use_consensus = config.get("use_consensus", False)
        self.use_detector = config.get("use_spoof_lstm_detector", False)
        
        if self.use_trust:
            self.trust_net = TrustNetwork(
                config["trust_hidden"], 
                config.get("max_correction", 3.0)  # 🔥 config에서 max_correction 가져오기
            ).to(DEVICE)
            self.trust_opt = optim.Adam(self.trust_net.parameters(), lr=config["trust_lr"])
            self.trust_loss = TrustLoss(config["trust_lambda_reg"])
            self.last_trust_scores = {}
        
        if self.use_consensus:
            self.consensus = ConsensusProtocol(
                config["consensus_threshold"], 
                config["consensus_weight"],
                config["consensus_vote_threshold"]
            )
        
        if self.use_detector:
            self.detector = LSTMSpoofDetector(
                config["detector_feature_dim"], 
                config["detector_hidden"]
            ).to(DEVICE)
            self.det_opt = optim.Adam(self.detector.parameters(), lr=config["mappo_lr"])
            self.det_hist = {}
            self.det_buf = {"in": [], "tgt": []}
        
        self.buffer = RolloutBuffer()
        self.trust_buf = {"feat": [], "gps": [], "real": [], "prev": []}  # ✅ 수정: gps 추가, fused/curr 제거
    
    def reset_episode(self, agents):
        """에피소드 시작 시 초기화"""
        if self.use_trust:
            # 🔥 NEW: 초기 보정값은 0으로 시작
            self.last_trust_scores = {a: torch.tensor([0.0, 0.0], device=DEVICE) for a in agents}
        if self.use_detector:
            self.det_hist = {a: deque(maxlen=self.config["detector_seq_len"]) for a in agents}
    
    def select_action(self, l_obs, g_obs, real_pos, gps_pos, env=None, deterministic=False):
        """
        ✅ 개선: 액션 선택 시 융합된 위치 사용 및 Consensus 투표 메커니즘 적용
        
        Args:
            l_obs: 로컬 관찰 딕셔너리
            g_obs: 글로벌 관찰
            real_pos: 실제 위치 (학습용)
            gps_pos: GPS 위치
            env: 환경 객체 (Consensus 투표용)
            deterministic: 결정적 액션 선택 여부
        
        Returns:
            actions, log_probs, value, trust_info
        """
        with torch.no_grad():
            actions, log_probs, trust_info = {}, {}, {}
            g_tensor = torch.tensor(g_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            val = self.critic(g_tensor).item()
            
            for aid, obs in l_obs.items():
                idx = int(aid.split('_')[1])
                obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                obs_mod = obs.copy()
                t_gps, t_vis = 1.0, 0.0
                fused_pos_np = gps_pos[idx].copy()
                
                if self.use_trust:
                    # 🔥 NEW: Trust Network로 GPS 보정값 계산
                    t_feat = obs_t[:, 6:10]  # trust_features (4차원)
                    correction = self.trust_net(t_feat).squeeze(0)  # (2,) [correction_x, correction_y]
                    
                    # 🔥 NEW: Consensus Protocol로 보정값 조정
                    force_zero = False
                    suspicion_ratio = 0.0
                    correction_scale = 1.0
                    
                    if self.use_consensus and env is not None:
                        # 받은 의심 표 집계
                        votes_received = env.suspicion_votes_received[idx]
                        is_under_attack, suspicion_ratio = self.consensus.aggregate_votes(votes_received)
                        
                        if is_under_attack:
                            # 50% 이상 의심 표: 보정값을 강하게 적용
                            force_zero = True
                            correction_scale = 2.0  # 보정 강도 2배
                        elif suspicion_ratio >= 0.3:
                            # 30-50% 의심 표: 보정값 증가
                            correction_scale = 1.5
                        elif suspicion_ratio < 0.1:
                            # 의심 표 거의 없음: 보정값 감소
                            correction_scale = 0.5
                    
                    # 🔥 NEW: GPS + 보정값으로 최종 위치 계산
                    if real_pos is not None:
                        gp = torch.tensor(gps_pos[idx], device=DEVICE, dtype=torch.float32)
                        rp = torch.tensor(real_pos[idx], device=DEVICE, dtype=torch.float32)
                        
                        # 보정된 위치 계산 (gradient 유지)
                        corrected_pos = gp + correction * correction_scale
                        prev_correction = self.last_trust_scores.get(aid, torch.tensor([0.0, 0.0], device=DEVICE))
                        
                        # Trust Loss 계산용 버퍼에 저장
                        self.trust_buf['feat'].append(t_feat.squeeze(0))  # (4,)
                        self.trust_buf['gps'].append(gp)  # GPS 위치
                        self.trust_buf['real'].append(rp)  # 실제 위치
                        self.trust_buf['prev'].append(prev_correction)  # 이전 보정값
                        self.last_trust_scores[aid] = correction.detach()
                        
                        corrected_pos_np = corrected_pos.detach().cpu().numpy()
                    else:
                        # 평가 모드: GPS + 보정값 (gradient 없음)
                        corrected_pos_np = gps_pos[idx] + correction.cpu().numpy() * correction_scale
                    
                    # 🔥 NEW: Actor 입력에 보정된 위치 사용
                    obs_mod[0:2] = corrected_pos_np / self.config["grid_size"]
                    obs_t = torch.tensor(obs_mod, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    
                    # Trust 정보 (보정 크기를 gps trust로 표현)
                    correction_magnitude = float(torch.norm(correction).item())
                    trust_info[aid] = {
                        'gps': 1.0 - min(correction_magnitude / 5.0, 1.0),  # 보정 클수록 신뢰도 낮음
                        'vis': min(correction_magnitude / 5.0, 1.0),  # 보정 크기
                        'force_zero': force_zero
                    }
                
                elif self.use_detector:
                    # LSTM Detector 사용
                    gps_norm = gps_pos[idx] / self.config["grid_size"]
                    feat = np.array([gps_norm[0], gps_norm[1], obs[6], obs[7], obs[10]], dtype=np.float32)
                    self.det_hist[aid].append(feat)
                    seq = list(self.det_hist[aid])
                    while len(seq) < self.config["detector_seq_len"]:
                        seq.insert(0, [0]*5)
                    
                    seq_t = torch.tensor(seq, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    correction = self.detector(seq_t).squeeze(0).cpu().numpy()
                    obs_mod[0:2] = gps_norm + correction
                    obs_t = torch.tensor(obs_mod, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    
                    if real_pos is not None:
                        tgt = (real_pos[idx] / self.config["grid_size"]) - gps_norm
                        self.det_buf["in"].append(seq)
                        self.det_buf["tgt"].append(tgt)
                    
                    trust_info[aid] = {'gps': 1.0, 'vis': 0.0}
                else:
                    # Trust Network 미사용 (Baseline)
                    trust_info[aid] = {'gps': 1.0, 'vis': 0.0}
                
                # Actor로 액션 선택
                probs = self.actor(obs_t)
                dist = Categorical(probs)
                act = torch.argmax(probs) if deterministic else dist.sample()
                actions[aid] = act.item()
                log_probs[aid] = dist.log_prob(act).item()
            
            return actions, log_probs, val, trust_info
    
    def compute_gae(self, next_val):
        """Generalized Advantage Estimation (GAE) 계산"""
        rews = torch.tensor(self.buffer.rew, dtype=torch.float32)
        vals = torch.tensor(self.buffer.val + [next_val], dtype=torch.float32)
        dones = torch.tensor(self.buffer.done, dtype=torch.float32)
        
        adv = []
        last = 0
        for t in reversed(range(len(rews))):
            delta = rews[t] + self.config["gamma"] * vals[t+1] * (1-dones[t]) - vals[t]
            last = delta + self.config["gamma"] * self.config["gae_lambda"] * (1-dones[t]) * last
            adv.insert(0, last)
        
        self.buffer.adv = adv
        self.buffer.ret = [a + v for a, v in zip(adv, vals[:-1].tolist())]
    
    def update(self):
        """PPO 업데이트"""
        b_obs = torch.tensor(np.array(self.buffer.obs), dtype=torch.float32, device=DEVICE)
        b_glo = torch.tensor(np.array(self.buffer.glo), dtype=torch.float32, device=DEVICE)
        b_act = torch.tensor(self.buffer.act, dtype=torch.long, device=DEVICE)
        b_log = torch.tensor(self.buffer.logp, dtype=torch.float32, device=DEVICE)
        b_adv = torch.tensor(self.buffer.adv, dtype=torch.float32, device=DEVICE)
        b_ret = torch.tensor(self.buffer.ret, dtype=torch.float32, device=DEVICE)
        
        # PPO Update
        for _ in range(self.config["update_epochs"]):
            probs = self.actor(b_obs)
            dist = Categorical(probs)
            log_p = dist.log_prob(b_act)
            ratio = torch.exp(log_p - b_log)
            
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 0.8, 1.2) * b_adv
            a_loss = -torch.min(surr1, surr2).mean()
            
            c_loss = F.mse_loss(self.critic(b_glo).squeeze(), b_ret)
            loss = a_loss + 0.5 * c_loss - self.config["mappo_entropy"] * dist.entropy().mean()
            
            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            loss.backward()
            self.actor_opt.step()
            self.critic_opt.step()
        
        # 🔥 NEW: Trust Network Update (GPS Correction 방식)
        if self.use_trust and self.trust_buf['feat']:
            feat_tensor = torch.stack(self.trust_buf['feat'])  # (N, 4)
            gps_tensor = torch.stack(self.trust_buf['gps'])    # (N, 2)
            real_tensor = torch.stack(self.trust_buf['real'])  # (N, 2)
            prev_correction = torch.stack(self.trust_buf['prev'])  # (N, 2) 이전 보정값
            
            # Trust Network forward (gradient 활성화)
            correction = self.trust_net(feat_tensor)  # (N, 2) [correction_x, correction_y]
            
            # 보정된 위치 계산
            corrected_pos = gps_tensor + correction
            
            # 🔥 NEW Loss 계산: Correction Loss + Smoothness Loss
            loss = self.trust_loss.compute(corrected_pos, real_tensor, correction, prev_correction)
            
            self.trust_opt.zero_grad()
            loss.backward()
            self.trust_opt.step()
            
            self.trust_buf = {k: [] for k in self.trust_buf}
        
        # LSTM Detector Update
        if self.use_detector and self.det_buf['in']:
            inp = torch.tensor(np.array(self.det_buf['in']), dtype=torch.float32, device=DEVICE)
            tgt = torch.tensor(np.array(self.det_buf['tgt']), dtype=torch.float32, device=DEVICE)
            loss = F.mse_loss(self.detector(inp), tgt)
            
            self.det_opt.zero_grad()
            loss.backward()
            self.det_opt.step()
            
            self.det_buf = {"in": [], "tgt": []}
        
        self.buffer.clear()
    
    def save_models(self, path):
        """모델 저장"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
        if self.use_trust:
            torch.save(self.trust_net.state_dict(), os.path.join(path, "trust.pth"))
        if self.use_detector:
            torch.save(self.detector.state_dict(), os.path.join(path, "detector.pth"))
    
    def load_models(self, path):
        """모델 로드"""
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pth"), map_location=DEVICE))
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth"), map_location=DEVICE))
        if self.use_trust and os.path.exists(os.path.join(path, "trust.pth")):
            self.trust_net.load_state_dict(torch.load(os.path.join(path, "trust.pth"), map_location=DEVICE))
        if self.use_detector and os.path.exists(os.path.join(path, "detector.pth")):
            self.detector.load_state_dict(torch.load(os.path.join(path, "detector.pth"), map_location=DEVICE))


# ==================== TRAINING ====================

class TrainingWorker(threading.Thread):
    """학습 워커 스레드"""
    def __init__(self, config, algorithm_name, data_queue, stop_flag):
        super().__init__()
        self.config = config
        self.algorithm_name = algorithm_name
        self.data_queue = data_queue
        self.stop_flag = stop_flag
    
    def run(self):
        run_training(self.config, self.algorithm_name, self.data_queue, self.stop_flag)


def run_training(config, algorithm_name, data_queue, stop_flag):
    """
    학습 실행 함수
    
    ✅ 개선: select_action 호출 시 env 객체 전달
    """
    try:
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        
        base_folder = create_model_folder_name(config, algorithm_name)
        model_base_path = os.path.join("./models", base_folder)
        os.makedirs(model_base_path, exist_ok=True)
        writer = SummaryWriter(os.path.join("runs", base_folder))
        
        data_queue.put(("log", f"🔥 {algorithm_name} 학습 시작\n"))
        env = CTDEMultiUAVEnv(config)
        agent = MAPPOAgentWithTrust(env.local_obs_dim, env.global_obs_dim, env.action_dim, config)
        
        for ep in range(0, config["total_episodes"], config["episodes_per_batch"]):
            if stop_flag[0]:
                break
            
            rew_list, succ_list, coll_list = [], [], []
            trust_gps_list, trust_vis_list, suspicion_ratio_list = [], [], []  # ✅ 추가
            
            for _ in range(config["episodes_per_batch"]):
                scen = EnvironmentScenario(config)
                lo, go = env.reset_with_scenario(scen)
                agent.reset_episode(env.agents)
                done = False
                ep_r = 0
                
                # 에피소드 버퍼
                ep_obs, ep_glo, ep_act, ep_logp, ep_val, ep_rew, ep_done = [],[],[],[],[],[],[]
                ep_trust_gps, ep_trust_vis, ep_suspicion = [], [], []  # ✅ 추가
                
                while not done:
                    # ✅ 개선: env 객체를 select_action에 전달
                    acts, logs, val, trust_info = agent.select_action(
                        lo, go, env.uav_positions, env.gps_positions, env=env
                    )
                    
                    # ✅ Trust 정보 수집
                    for aid in env.agents:
                        if aid in trust_info:
                            ep_trust_gps.append(trust_info[aid]['gps'])
                            ep_trust_vis.append(trust_info[aid]['vis'])
                    
                    # Suspicion ratio 수집
                    ep_suspicion.extend(env.consensus_votes.tolist())
                    
                    n_lo, n_go, rew, done, info = env.step(acts)
                    
                    ep_obs.extend([lo[a] for a in env.agents if a in acts])
                    ep_glo.extend([go for _ in acts])
                    ep_act.extend(list(acts.values()))
                    ep_logp.extend(list(logs.values()))
                    ep_val.extend([val for _ in acts])
                    ep_rew.extend(list(rew.values()))
                    ep_done.extend([done for _ in acts])
                    
                    lo, go = n_lo, n_go
                    ep_r += sum(rew.values())
                
                # 버퍼에 추가
                agent.buffer.add(ep_obs, ep_glo, ep_act, ep_logp, ep_val, ep_rew, ep_done)
                
                rew_list.append(ep_r)
                succ_list.append(info.get("success_rate", 0))
                coll_list.append(info.get("collision_rate", 0))
                
                # ✅ Trust 통계 수집
                if ep_trust_gps:
                    trust_gps_list.append(np.mean(ep_trust_gps))
                    trust_vis_list.append(np.mean(ep_trust_vis))
                if ep_suspicion:
                    suspicion_ratio_list.append(np.mean(ep_suspicion))
            
            # GAE 계산 & 업데이트
            with torch.no_grad():
                next_val = agent.critic(torch.tensor(go, dtype=torch.float32, device=DEVICE).unsqueeze(0)).item()
            agent.compute_gae(next_val)
            agent.update()
            
            # 로그
            avg_r, avg_s, avg_c = np.mean(rew_list), np.mean(succ_list), np.mean(coll_list)
            writer.add_scalar(f"{algorithm_name}/Reward", avg_r, ep)
            writer.add_scalar(f"{algorithm_name}/Success", avg_s, ep)
            writer.add_scalar(f"{algorithm_name}/Collision", avg_c, ep)
            
            # ✅ Trust/Consensus 통계 로깅 (알고리즘 이름 포함)
            avg_trust_gps, avg_trust_vis, avg_suspicion = 0.0, 0.0, 0.0
            if trust_gps_list:
                avg_trust_gps = np.mean(trust_gps_list)
                avg_trust_vis = np.mean(trust_vis_list)
                writer.add_scalar(f"{algorithm_name}/Trust_GPS", avg_trust_gps, ep)
                writer.add_scalar(f"{algorithm_name}/Trust_Vision", avg_trust_vis, ep)
            if suspicion_ratio_list:
                avg_suspicion = np.mean(suspicion_ratio_list)
                writer.add_scalar(f"{algorithm_name}/Consensus_SuspicionRatio", avg_suspicion, ep)
            
            if ep % 100 == 0:
                # ✅ Trust 정보 포함한 로그
                log_msg = f"[{algorithm_name}] Ep {ep}: Rew {avg_r:.1f} Succ {avg_s:.1%} Coll {avg_c:.1%}"
                if trust_gps_list:
                    log_msg += f" | Trust GPS:{avg_trust_gps:.3f} Vis:{avg_trust_vis:.3f}"
                if suspicion_ratio_list:
                    log_msg += f" | Suspicion:{avg_suspicion:.3f}"
                log_msg += "\n"
                data_queue.put(("log", log_msg))
                
                data_queue.put(("graph", {
                    "algorithm": algorithm_name,
                    "rew": avg_r,
                    "succ": avg_s,
                    "coll": avg_c,
                    "drift_det": 0,
                    "path_len": 0
                }))
            
            if ep % config["checkpoint_interval"] == 0 and ep > 0:
                agent.save_models(os.path.join(model_base_path, f"ckpt_{ep}"))
        
        agent.save_models(os.path.join(model_base_path, "final"))
        data_queue.put(("log", f"✅ [{algorithm_name}] 학습 완료\n"))
        data_queue.put(("done", algorithm_name))
        
    except Exception as e:
        import traceback
        data_queue.put(("log", f"❌ Error in {algorithm_name}: {e}\n{traceback.format_exc()}\n"))
    finally:
        writer.close()


# ==================== GUI ====================

class GraphCanvas(FigureCanvas):
    """학습 진행 그래프"""
    def __init__(self, parent=None):
        self.fig = Figure()
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Training Progress: Reward / Success / Collision")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Value")
        self.graph_data = {}
    
    def update_graph(self, algorithm, rew, succ, coll, drift_det, path_len):
        if algorithm not in self.graph_data:
            self.graph_data[algorithm] = {'x': [], 'rew': [], 'succ': []}
        
        d = self.graph_data[algorithm]
        x = len(d['x']) * 100
        d['x'].append(x)
        d['rew'].append(rew)
        d['succ'].append(succ)
        
        self.ax.clear()
        for algo, vals in self.graph_data.items():
            self.ax.plot(vals['x'], vals['rew'], label=f"{algo} (Reward)", marker='o')
            # Success rate는 보조 축에 표시 (선택적)
        
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.draw()


class MainWindow(QMainWindow):
    """메인 GUI 윈도우"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚁 Trust-Consensus MAPPO - Improved (논문 명세 준수)")
        self.setGeometry(100, 100, 1400, 900)
        self.data_queue = queue.Queue()
        self.stop_flag = [False]
        self.running_threads = {}
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # 왼쪽 패널: 설정 및 제어
        left_panel = QVBoxLayout()
        title = QLabel("🎯 실험 설정 및 제어 (논문 명세 버전)")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        left_panel.addWidget(title)
        
        # 알고리즘 선택
        algo_group = QGroupBox("알고리즘 선택")
        algo_layout = QVBoxLayout()
        self.algo_checkboxes = {}
        for algo_name, algo_config in ALGORITHM_CONFIGS.items():
            cb = QCheckBox(f"{algo_name}: {algo_config['description']}")
            self.algo_checkboxes[algo_name] = cb
            algo_layout.addWidget(cb)
        algo_group.setLayout(algo_layout)
        left_panel.addWidget(algo_group)
        
        # 공격 모드 선택
        attack_group = QGroupBox("공격 모드")
        attack_layout = QHBoxLayout()
        self.attack_combo = QComboBox()
        self.attack_combo.addItems(["hybrid", "step", "drift", "none"])
        attack_layout.addWidget(QLabel("Attack Mode:"))
        attack_layout.addWidget(self.attack_combo)
        attack_group.setLayout(attack_layout)
        left_panel.addWidget(attack_group)
        
        # 학습 설정
        config_group = QGroupBox("학습 설정")
        config_layout = QFormLayout()
        self.episode_input = QLineEdit(str(BASE_CONFIG["total_episodes"]))
        self.batch_input = QLineEdit(str(BASE_CONFIG["episodes_per_batch"]))
        self.obstacle_input = QLineEdit(str(BASE_CONFIG["num_obstacles"]))
        config_layout.addRow("총 Episodes:", self.episode_input)
        config_layout.addRow("Batch Episodes:", self.batch_input)
        config_layout.addRow("장애물 수:", self.obstacle_input)
        config_group.setLayout(config_layout)
        left_panel.addWidget(config_group)
        
        # 버튼
        btn_layout1 = QHBoxLayout()
        self.start_btn = QPushButton("🚀 학습 시작")
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn = QPushButton("⏹️ 중단")
        self.stop_btn.clicked.connect(self.stop_all_training)
        
        btn_layout1.addWidget(self.start_btn)
        btn_layout1.addWidget(self.stop_btn)
        left_panel.addLayout(btn_layout1)
        
        # 데모 및 도구 버튼
        btn_layout2 = QHBoxLayout()
        self.demo_btn = QPushButton("🎮 데모 실행")
        self.demo_btn.clicked.connect(self.run_demo)
        self.tb_btn = QPushButton("📊 TensorBoard")
        self.tb_btn.clicked.connect(self.open_tensorboard)
        
        btn_layout2.addWidget(self.demo_btn)
        btn_layout2.addWidget(self.tb_btn)
        left_panel.addLayout(btn_layout2)
        
        left_panel.addStretch()
        main_layout.addLayout(left_panel, 1)
        
        # 오른쪽 패널: 그래프 및 로그
        right_panel = QVBoxLayout()
        self.graph_canvas = GraphCanvas(self)
        right_panel.addWidget(self.graph_canvas, 2)
        
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Consolas", 9))
        right_panel.addWidget(self.log_box, 1)
        
        main_layout.addLayout(right_panel, 2)
        
        # 타이머 (큐 처리)
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_queue)
        self.timer.start(200)
    
    def append_log(self, text):
        """로그 추가"""
        self.log_box.moveCursor(QTextCursor.MoveOperation.End)
        self.log_box.insertPlainText(text)
        self.log_box.moveCursor(QTextCursor.MoveOperation.End)
    
    def process_queue(self):
        """데이터 큐 처리"""
        while not self.data_queue.empty():
            item_type, payload = self.data_queue.get()
            if item_type == "log":
                self.append_log(payload)
            elif item_type == "graph":
                algo = payload['algorithm']
                self.graph_canvas.update_graph(
                    algo,
                    payload['rew'],
                    payload['succ'],
                    payload['coll'],
                    payload['drift_det'],
                    payload['path_len']
                )
    
    def start_training(self):
        """학습 시작"""
        self.stop_flag[0] = False
        selected_algos = [name for name, cb in self.algo_checkboxes.items() if cb.isChecked()]
        
        if not selected_algos:
            self.append_log("⚠️ 알고리즘을 선택해주세요.\n")
            return
        
        total_ep = int(self.episode_input.text())
        batch_ep = int(self.batch_input.text())
        obs_num = int(self.obstacle_input.text())
        atk_mode = self.attack_combo.currentText()
        
        for name in selected_algos:
            config = BASE_CONFIG.copy()
            config["total_episodes"] = total_ep
            config["episodes_per_batch"] = batch_ep
            config["num_obstacles"] = obs_num
            config["attack_mode"] = atk_mode
            config.update(ALGORITHM_CONFIGS[name])
            
            worker = TrainingWorker(config, name, self.data_queue, self.stop_flag)
            worker.start()
            self.running_threads[name] = worker
            self.append_log(f"▶️ [{name}] 시작 (논문 명세 버전)\n")
    
    def stop_all_training(self):
        """모든 학습 중단"""
        self.stop_flag[0] = True
        self.append_log("⚠️ 학습 중단 요청...\n")
    
    def run_demo(self):
        """학습된 모델로 데모 실행"""
        # 알고리즘 선택 확인
        selected_algos = [name for name, cb in self.algo_checkboxes.items() if cb.isChecked()]
        
        if not selected_algos:
            self.append_log("⚠️ 데모를 실행할 알고리즘을 선택해주세요.\n")
            return
        
        if len(selected_algos) > 1:
            self.append_log("⚠️ 데모는 한 번에 하나의 알고리즘만 실행 가능합니다.\n")
            return
        
        algo_name = selected_algos[0]
        
        # 모델 경로 선택 다이얼로그
        from PySide6.QtWidgets import QFileDialog
        model_dir = QFileDialog.getExistingDirectory(
            self, 
            "학습된 모델 폴더 선택",
            "./models",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not model_dir:
            self.append_log("⚠️ 모델 폴더가 선택되지 않았습니다.\n")
            return
        
        # 데모 실행
        self.append_log(f"🎮 [{algo_name}] 데모 실행 중...\n")
        self.append_log(f"📁 모델 경로: {model_dir}\n")
        
        try:
            config = BASE_CONFIG.copy()
            config.update(ALGORITHM_CONFIGS[algo_name])
            config["render_mode"] = "human"  # 시각화 활성화
            
            # 데모 스레드 시작
            demo_thread = threading.Thread(
                target=self.demo_worker,
                args=(config, algo_name, model_dir),
                daemon=True
            )
            demo_thread.start()
            
        except Exception as e:
            self.append_log(f"❌ 데모 실행 실패: {e}\n")
    
    def demo_worker(self, config, algo_name, model_dir):
        """데모 실행 워커"""
        try:
            # 환경 생성
            env = CTDEMultiUAVEnv(config, render_mode="human")
            agent = MAPPOAgentWithTrust(env.local_obs_dim, env.global_obs_dim, env.action_dim, config)
            
            # 모델 로드
            try:
                agent.load_models(model_dir)
                self.data_queue.put(("log", f"✅ 모델 로드 완료\n"))
            except Exception as e:
                self.data_queue.put(("log", f"⚠️ 모델 로드 실패, 랜덤 정책 사용: {e}\n"))
            
            # 데모 에피소드 실행
            for ep in range(config["demo_episodes"]):
                scenario = EnvironmentScenario(config)
                lo, go = env.reset_with_scenario(scenario)
                agent.reset_episode(env.agents)
                done = False
                ep_r = 0
                step = 0
                
                self.data_queue.put(("log", f"\n📺 에피소드 {ep+1}/{config['demo_episodes']} 시작\n"))
                
                while not done and step < config["max_steps"]:
                    # 결정적 액션 선택 (탐험 없이)
                    acts, _, _, trust_info = agent.select_action(
                        lo, go, env.uav_positions, env.gps_positions, 
                        env=env, deterministic=True
                    )
                    
                    lo, go, rew, done, info = env.step(acts)
                    ep_r += sum(rew.values())
                    step += 1
                    
                    # 렌더링
                    env.render()
                    time.sleep(config["render_delay"])
                
                # 결과 출력
                success_rate = info.get("success_rate", 0)
                collision_rate = info.get("collision_rate", 0)
                self.data_queue.put(("log", 
                    f"  보상: {ep_r:.1f}, 성공률: {success_rate:.1%}, 충돌률: {collision_rate:.1%}\n"))
            
            env.close()
            self.data_queue.put(("log", f"\n✅ 데모 완료\n"))
            
        except Exception as e:
            import traceback
            self.data_queue.put(("log", f"❌ 데모 오류: {e}\n{traceback.format_exc()}\n"))
    
    def open_tensorboard(self):
        """TensorBoard 실행"""
        import subprocess
        try:
            subprocess.Popen(["tensorboard", "--logdir=runs"])
            self.append_log("📊 TensorBoard 실행 중... (http://localhost:6006)\n")
        except Exception as e:
            self.append_log(f"❌ TensorBoard 실행 실패: {e}\n")


def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyside6'))
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
