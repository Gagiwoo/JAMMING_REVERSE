# 논문 구현 코드 분석 및 개선사항

## 📄 논문 정보
- **제목**: GPS 스푸핑 환경에서 신뢰도 기반 다중 UAV 협력 경로 계획
- **저자**: 김도윤
- **출처**: 한국시뮬레이션학회 논문지 Vol. 26, No. 3 (2017)

---

## 🔍 코드와 논문 간 주요 차이점

### 1. ❌ **하이퍼파라미터 불일치**

| 파라미터 | 논문 명세 | 현재 코드 | 상태 |
|---------|---------|---------|------|
| Actor/Critic Learning Rate | 3×10⁻⁴ | 5×10⁻⁴ | ❌ |
| Trust Network Learning Rate | 1.5×10⁻⁴ (50% of Actor) | 2.5×10⁻⁴ (50% of 5e-4) | ❌ |
| Trust Lambda (정규화 계수) | 0.1 | 0.05 | ❌ |
| Consensus Threshold | 2.5 cells | 2.0 | ❌ |
| Consensus Weight | 0.15 | 0.2 | ❌ |
| Entropy Coefficient | 0.01 | 0.01 | ✅ |
| PPO Clip Epsilon | 0.2 | 0.2 | ✅ |
| Gamma | 0.99 | 0.99 | ✅ |
| GAE Lambda | 0.95 | 0.95 | ✅ |
| Batch Size | 512 | 512 | ✅ |
| Update Epochs | 10 | 10 | ✅ |

### 2. ❌ **Trust Network 아키텍처 불일치**

**논문 명세:**
- 3개의 은닉층
- 각 층 16 뉴런
- 입력: 4차원 (temporal residual, spatial discrepancy, GPS variance, Vision quality)
- 출력: 2차원 (GPS trust, Vision trust) with Softmax

**현재 코드:**
```python
class TrustNetwork(nn.Module):
    def __init__(self, hidden=32):  # ❌ 32 neurons (should be 16)
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(),      # Layer 1
            nn.Linear(hidden, hidden), nn.ReLU(), # Layer 2
            nn.Linear(hidden, 2), nn.Softmax(dim=-1)  # Output (only 2 layers!)
        )
```

**문제점:**
- 은닉 유닛이 32개 (논문: 16개)
- 실제로는 2개의 은닉층만 있음 (논문: 3개)

### 3. ⚠️ **Actor/Critic 네트워크 아키텍처 불일치**

**논문 명세:**
- Actor: 1개 은닉층, 128 뉴런, Tanh 활성화
- Critic: 2개 은닉층, 각 256 뉴런, Tanh 활성화

**현재 코드:**
```python
class Actor(nn.Module):
    def __init__(self, local_dim, act_dim, hidden=128, use_lstm=False):
        self.fc1 = nn.Linear(local_dim, hidden)  # Layer 1 ✅
        if use_lstm: self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.fc2 = nn.Linear(hidden, hidden)  # ❌ Extra layer!
        self.head = nn.Linear(hidden, act_dim)
        
class Critic(nn.Module):
    def __init__(self, glob_dim, hidden=256):
        self.net = nn.Sequential(
            nn.Linear(glob_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )  # ✅ Correct
```

**문제점:**
- Actor에 불필요한 fc2 층이 추가됨 (논문에는 1개 은닉층만)

### 4. ❌ **Consensus Protocol 투표 메커니즘 불완전**

**논문 명세:**
- 각 UAV는 이웃들로부터 받은 **의심 표(suspicion votes)**를 집계
- 전체 이웃 수의 **50% 이상**에게서 의심 표를 받으면 GPS 신뢰도를 **강제로 0**으로 설정

**현재 코드:**
```python
def adjust_trust(self, trust_gps, trust_vis, consensus_vote):
    ratio = np.clip(consensus_vote / self.threshold, 0.0, 2.0)
    
    if ratio > 0.8:  # ❌ 임의의 비율 기반 조정
        delta = (ratio - 0.8) * self.consensus_weight * 1.5
        trust_gps *= (1 - delta)
        trust_vis *= (1 + delta)
    # ... (부드러운 조정만 수행)
```

**문제점:**
- 투표 기반 강제 설정(forced setting) 메커니즘 누락
- 50% 임계값 기반 명확한 결정 대신 부드러운 조정만 수행
- 집단 의사결정 로직 미구현

### 5. ⚠️ **관찰 공간(Observation Space) 구조 문제**

**논문 명세:**
- 융합된 위치 (p_fused)
- 속도 (v_i)
- 목표까지의 거리 (dist_to_goal)
- 이웃들의 상대 위치 (p_j - p_i)
- GPS 신뢰도 점수 (trust_i)

**현재 코드:**
```python
my_state = np.concatenate([
    self.gps_positions[i]/self.grid_size,  # ❌ GPS position (not fused!)
    self.target_positions[i]/self.grid_size,
    trust_feats,  # temporal, spatial, gps_var, neighbor_flag
    [vote]
])
```

**문제점:**
- **GPS 위치**를 사용하지만, 논문에서는 **융합된 위치(fused position)**을 사용해야 함
- 속도(velocity) 정보가 관찰 벡터에 포함되지 않음
- 목표까지의 거리가 벡터로 표현되었지만 스칼라 거리값이 필요

### 6. ⚠️ **GPS 공격 모델 파라미터**

**논문 명세:**
- Attack Probability: 10% per step
- Step Attack Offset: -4.0m ~ 4.0m
- Drift Attack Rate: 0.2 ~ 0.8 m/s
- Attack Duration: 10~30 steps

**현재 코드:**
```python
"attack_start_prob": 0.05,  # ❌ 5% (should be 10%)
"attack_min_duration": 10,  # ✅
"attack_max_duration": 30,  # ✅
# In _simulate_attacks():
self.attack_step_offset[i] = np.random.uniform(-4.0, 4.0, size=2)  # ✅
self.attack_drift_dir[i] = ... * np.random.uniform(0.2, 0.8)  # ✅
```

**문제점:**
- 공격 시작 확률이 5%로 설정 (논문: 10%)

### 7. ⚠️ **Trust Loss 계산 불완전**

**논문 명세:**
```
Loss = MSE(p_fused, p_real) + λ * MSE(trust_t, trust_{t-1})
```

**현재 코드:**
```python
def compute(self, fused_pos, real_pos, current_trust, prev_trust):
    fusion_loss = torch.mean((fused_pos - real_pos) ** 2)
    smoothness_loss = torch.mean((current_trust - prev_trust) ** 2)
    return fusion_loss + self.lambda_reg * smoothness_loss
```

**상태:** ✅ 올바르게 구현됨 (lambda 값만 수정 필요)

---

## 🎯 개선 계획

### Phase 1: 핵심 수정 (Critical)
1. ✅ Trust Network 아키텍처를 3층 × 16 뉴런으로 수정
2. ✅ Actor 네트워크에서 불필요한 fc2 층 제거
3. ✅ Consensus Protocol에 50% 투표 기반 강제 설정 메커니즘 추가
4. ✅ 하이퍼파라미터를 논문 명세에 맞게 수정

### Phase 2: 중요 개선 (Important)
5. ✅ 관찰 공간에 융합된 위치 사용 및 속도 추가
6. ✅ GPS 공격 확률을 10%로 수정
7. ✅ Trust Loss lambda를 0.1로 수정

### Phase 3: 검증 및 테스트
8. 수정된 코드 실행 테스트
9. 논문 결과와 비교 검증
10. 성능 지표 확인 (Success Rate, Collision Rate, Path Length)

---

## 📊 예상 개선 효과

### 개선 전 (현재 코드)
- Trust Network가 과도하게 복잡 (32 neurons)
- Consensus 투표 메커니즘 미흡
- 부정확한 하이퍼파라미터로 인한 학습 불안정

### 개선 후 (논문 명세 준수)
- 논문과 동일한 아키텍처로 재현성 확보
- 강력한 집단 의사결정으로 GPS 스푸핑 탐지 성능 향상
- 안정적인 학습 및 더 높은 성공률 기대

---

## 📝 참고사항

- 현재 코드의 전반적인 구조와 MAPPO 구현은 잘 되어 있음
- 주로 **세부 파라미터와 논리 구현**에서 논문과 차이 발생
- GUI 및 시각화 기능은 논문에 없는 추가 기능으로 유지 가능
- 다중 알고리즘 비교 실험 설정은 유용한 추가 기능

---

## 🔧 수정 적용 순서

1. **config.py**: 하이퍼파라미터 수정
2. **networks.py**: Trust Network, Actor 아키텍처 수정
3. **consensus.py**: 투표 메커니즘 개선
4. **environment.py**: 관찰 공간 및 공격 모델 수정
5. **agent.py**: 융합된 위치 사용 및 Trust Loss 적용
6. **main.py**: 통합 및 테스트

모든 수정사항을 단계별로 적용하여 논문의 구현을 정확히 재현하겠습니다.
