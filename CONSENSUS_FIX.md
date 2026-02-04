# Consensus Protocol 논문 명세 준수 수정

## 📋 수정 개요

논문 "GPS 스푸핑 환경에서 신뢰도 기반 다중 UAV 협력 경로 계획"의 Consensus Protocol을 정확히 구현하기 위한 수정

---

## 🔍 기존 문제점

### 1. **투표 로직이 논문과 다름**
```python
# ❌ 기존 코드 (잘못된 구현)
ratio = np.clip(consensus_vote / self.threshold, 0.0, 2.0)

if ratio > 0.8:
    delta = (ratio - 0.8) * self.consensus_weight * 1.5
    trust_gps *= (1 - delta)
    trust_vis *= (1 + delta)
elif ratio < 0.4:
    delta = (0.4 - ratio) * self.consensus_weight * 0.5
    trust_gps *= (1 + delta)
    trust_vis *= (1 - delta)
```

**문제점:**
- `ratio` 계산이 논문과 다름 (임계값으로 나누는 방식)
- 곱셈 방식으로 조정 (논문은 덧셈/뺄셈)
- 경계값이 60%/40%가 아닌 80%/40%
- 조정량이 고정값이 아닌 비율 기반

### 2. **투표 집계 방식 불일치**
- 이웃의 GPS-Vision 불일치를 **평균**으로 계산
- 논문의 **투표 비율** 개념과 불일치

### 3. **50% 투표 기반 강제 설정 로직 부재**
- 논문: 50% 이상 의심 표 → GPS 신뢰도 0으로 강제 설정
- 기존 코드: 이 로직이 누락되거나 불완전

---

## ✅ 수정 내용

### 1. **ConsensusProtocol.adjust_trust() 논문 명세 준수**

```python
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
```

**핵심 변경:**
- ✅ **덧셈/뺄셈 방식**: `trust_gps += 0.15` 또는 `trust_gps -= 0.15`
- ✅ **60%/40% 경계**: 논문 명세 그대로
- ✅ **고정 조정량**: `consensus_weight = 0.15`
- ✅ **50% 강제 설정**: `force_zero=True` 시 GPS 신뢰도 0

### 2. **투표 비율(suspicion_ratio) 사용**

**환경 클래스 (CTDEMultiUAVEnv):**
```python
# ✅ 수정: consensus_votes를 suspicion_ratio로 변경
self.suspicion_ratio = np.zeros(self.num_uavs, dtype=np.float32)

# 투표 집계
for i in range(self.num_uavs):
    votes_received = self.suspicion_votes_received[i]
    if len(votes_received) > 0:
        self.suspicion_ratio[i] = sum(votes_received) / len(votes_received)
    else:
        self.suspicion_ratio[i] = 0.0
```

**Agent의 select_action():**
```python
# ✅ 수정: suspicion_ratio 사용
suspicion_ratio = obs[8]  # ← consensus_vote 대신 suspicion_ratio

if self.use_consensus and env is not None:
    votes_received = env.suspicion_votes_received[idx]
    is_under_attack, suspicion_ratio = self.consensus.aggregate_votes(votes_received)
    force_zero = is_under_attack
    
    # ✅ suspicion_ratio를 adjust_trust()에 전달
    t_gps, t_vis = self.consensus.adjust_trust(
        t_out[0].item(), 
        t_out[1].item(), 
        suspicion_ratio,  # ← 여기!
        force_zero=force_zero
    )
```

### 3. **TensorBoard에 Trust 통계 로깅**

```python
# ✅ Trust/Consensus 통계 로깅
if trust_gps_list:
    avg_trust_gps = np.mean(trust_gps_list)
    avg_trust_vis = np.mean(trust_vis_list)
    writer.add_scalar("Trust/GPS", avg_trust_gps, ep)
    writer.add_scalar("Trust/Vision", avg_trust_vis, ep)

if suspicion_ratio_list:
    avg_suspicion = np.mean(suspicion_ratio_list)
    writer.add_scalar("Consensus/SuspicionRatio", avg_suspicion, ep)
```

---

## 📊 논문 명세 vs 구현 비교표

| 항목 | 논문 명세 | 기존 코드 | 수정 코드 |
|------|----------|-----------|-----------|
| **조정 방식** | 덧셈/뺄셈 (`±0.15`) | 곱셈 (`*= (1±δ)`) | ✅ 덧셈/뺄셈 |
| **경계값** | 60%/40% | 80%/40% | ✅ 60%/40% |
| **조정량** | 고정 0.15 | 비율 기반 가변 | ✅ 고정 0.15 |
| **50% 강제** | GPS=0 | 불완전 | ✅ force_zero 로직 |
| **투표 방식** | 의심 표 비율 | 불일치 평균 | ✅ 투표 비율 |

---

## 🧪 검증 계획

### 1. **소규모 검증 (1,000 에피소드)**
```bash
# 설정
- Total Episodes: 1000
- Algorithm: Trust+Consensus-MAPPO
- Attack: hybrid
- Obstacles: 20
```

**확인 사항:**
- ✅ Training이 중단 없이 완료되는가?
- ✅ TensorBoard에 Trust/GPS, Trust/Vision, Consensus/SuspicionRatio가 로깅되는가?
- ✅ Success Rate이 Vanilla-MAPPO보다 높은가?

### 2. **TensorBoard 분석**
```bash
tensorboard --logdir runs
```

**관찰 지표:**
1. **Trust/GPS**: 공격 상황에서 감소하는가?
2. **Trust/Vision**: 공격 상황에서 증가하는가?
3. **Consensus/SuspicionRatio**: 공격 시 0.5 이상으로 증가하는가?
4. **Success**: Vanilla-MAPPO 대비 향상되는가?

### 3. **전체 재현 (20,000 에피소드)**
- **목표**: Success Rate 55.1% → **64.7%** (논문 결과 재현)
- **예상 소요 시간**: GPU 최적화 버전 사용 시 2-3시간

---

## 📈 기대 효과

### 1. **논문 재현성 향상**
- 논문 알고리즘을 정확히 구현
- 학술적 검증 가능성 확보

### 2. **성능 개선**
| 메트릭 | Vanilla-MAPPO | 예상 Trust+Consensus-MAPPO |
|--------|---------------|---------------------------|
| Success Rate | 55.1% | **64.7%** ↑ |
| Collision Rate | 8.2% | **4.5%** ↓ |
| GPS Spoofing Robustness | 약함 | **강함** ↑ |

### 3. **디버깅 용이성**
- TensorBoard에서 실시간 Trust 변화 관찰
- 문제 발생 시 빠른 진단 가능

---

## 🚀 다음 단계

1. ✅ **코드 업데이트**
   ```bash
   git pull origin main
   ```

2. ✅ **소규모 검증 실행**
   ```bash
   python improved_trust_consensus_mappo.py
   # GUI에서 Total Episodes: 1000 설정
   # Trust+Consensus-MAPPO 선택 후 학습 시작
   ```

3. ✅ **TensorBoard 모니터링**
   ```bash
   tensorboard --logdir runs
   # http://localhost:6006 접속
   ```

4. ✅ **결과 확인 후 전체 실험**
   - 1,000 에피소드 결과가 정상이면
   - 20,000 에피소드로 확장

---

## 📚 참고 문헌

- **논문**: "GPS 스푸핑 환경에서 신뢰도 기반 다중 UAV 협력 경로 계획" (김도윤, 2017)
- **저널**: 한국시뮬레이션학회 논문지 Vol. 26, No. 3
- **핵심 알고리즘**: Section 3.2 "분산 합의 기반 GPS 스푸핑 탐지"

---

## 🎯 요약

이번 수정으로:
1. ✅ **Consensus Protocol이 논문 명세를 정확히 따름**
2. ✅ **60%/40% 경계, ±0.15 조정, 50% 강제 설정 모두 구현**
3. ✅ **TensorBoard에 Trust 통계 로깅 추가**
4. ✅ **학술적 재현성 확보 및 성능 개선 기대**

**이제 논문을 정확하게 구현한 버전입니다!** 🎉
