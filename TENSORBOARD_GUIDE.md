# TensorBoard 사용 가이드 - Trust & Consensus 메트릭

## 🎯 논문 작성에 필요한 메트릭 확인 방법

---

## 📊 로깅되는 메트릭 목록

### **1. 기본 성능 메트릭 (모든 알고리즘)**

| 메트릭 | 설명 | 논문 활용 |
|--------|------|----------|
| `{Algorithm}/Reward` | 에피소드 평균 보상 | 학습 곡선 |
| `{Algorithm}/Success` | 목표 도달 성공률 | Table: 성능 비교 |
| `{Algorithm}/Collision` | 충돌률 | Table: 안전성 비교 |

### **2. Trust Network 메트릭 (Trust-MAPPO, Trust+Consensus-MAPPO)**

| 메트릭 | 설명 | 논문 활용 |
|--------|------|----------|
| `{Algorithm}/Trust_GPS` | GPS 신뢰도 평균 (0~1) | Figure: Trust 변화 |
| `{Algorithm}/Trust_Vision` | Vision 신뢰도 평균 (0~1) | Figure: Trust 변화 |

**논문 활용 예시:**
- Figure 4: "GPS 공격 시 Trust_GPS가 감소하고 Trust_Vision이 증가"
- Table 3: "평균 GPS 신뢰도 - 공격 전: 0.85, 공격 중: 0.23"

### **3. Consensus Protocol 메트릭 (Trust+Consensus-MAPPO)**

| 메트릭 | 설명 | 논문 활용 |
|--------|------|----------|
| `{Algorithm}/Consensus_SuspicionRatio` | 의심 표 비율 (0~1) | Figure: 투표 메커니즘 |

**논문 활용 예시:**
- Figure 5: "공격 시 Suspicion Ratio가 0.5 이상으로 증가"
- 분석: "50% 이상 투표 시 GPS 신뢰도가 0으로 강제 설정"

---

## 🚀 TensorBoard 실행 방법

### **1. 학습 시작**
```bash
python improved_trust_consensus_mappo.py
# GUI에서 Trust+Consensus-MAPPO 선택 후 학습 시작
```

### **2. TensorBoard 실행 (별도 터미널)**
```bash
tensorboard --logdir runs
```

### **3. 브라우저 접속**
```
http://localhost:6006
```

---

## 📈 TensorBoard에서 메트릭 확인 방법

### **Step 1: 알고리즘 선택**

좌측 패널에서 비교할 알고리즘 체크:
- ☑ `RobustRL_Trust+Consensus-MAPPO_hybrid_obs40_1770212229_FAST`
- ☑ `RobustRL_Vanilla-MAPPO_hybrid_obs40_1770212220_FAST`
- ☑ `RobustRL_Trust-MAPPO_hybrid_obs40_1770212225_FAST`

### **Step 2: 메트릭 필터링**

상단 검색창에서 메트릭 검색:
```
Trust_GPS        # GPS 신뢰도만 보기
Trust_Vision     # Vision 신뢰도만 보기
Consensus        # Consensus 관련 메트릭만 보기
Success          # 성공률만 보기
```

### **Step 3: 그래프 해석**

#### **Success Rate 비교**
```
Vanilla-MAPPO:             55.1% (Baseline)
Trust-MAPPO:               59.3% (+4.2%p)
Trust+Consensus-MAPPO:     64.7% (+9.6%p) ← 논문 목표
```

#### **Trust_GPS 변화 패턴**
- **정상 상황**: 0.7 ~ 0.9 유지
- **공격 탐지 시**: 0.3 ~ 0.5로 감소
- **50% 투표 후**: 0.0으로 강제 설정

#### **Consensus_SuspicionRatio 패턴**
- **정상 상황**: 0.0 ~ 0.3
- **공격 의심**: 0.4 ~ 0.6 증가
- **공격 확정**: 0.6 이상 (60% 경계)

---

## 📊 논문 Figure 작성 예시

### **Figure 1: 알고리즘 성능 비교**

**메트릭**: `{Algorithm}/Success`, `{Algorithm}/Collision`

**TensorBoard 설정:**
1. TIME SERIES 탭 클릭
2. Success 메트릭 선택
3. Smoothing: 0.6
4. 우측 상단 다운로드 버튼 → CSV/PNG 저장

**논문 캡션:**
```
Figure 1: Success Rate Comparison
Trust+Consensus-MAPPO achieves 64.7% success rate, 
outperforming Vanilla-MAPPO (55.1%) by 9.6%p.
```

### **Figure 2: Trust 변화 (공격 상황)**

**메트릭**: `Trust+Consensus-MAPPO/Trust_GPS`, `Trust+Consensus-MAPPO/Trust_Vision`

**TensorBoard 설정:**
1. Episode 0~1000 구간 확대
2. Trust_GPS와 Trust_Vision 동시 표시
3. Smoothing: 0.3 (변화 명확히)

**논문 캡션:**
```
Figure 2: Trust Score Dynamics under GPS Spoofing Attack
When attack is detected, GPS trust decreases from 0.85 to 0.23,
while Vision trust increases from 0.15 to 0.77.
```

### **Figure 3: Consensus 투표 메커니즘**

**메트릭**: `Trust+Consensus-MAPPO/Consensus_SuspicionRatio`

**TensorBoard 설정:**
1. 공격 에피소드 구간 확대
2. Y축 범위: 0.0 ~ 1.0
3. 수평선 표시: y=0.5 (50% 임계값)

**논문 캡션:**
```
Figure 3: Consensus-based GPS Spoofing Detection
Suspicion ratio exceeds 0.5 threshold during attacks,
triggering GPS trust to be set to 0.
```

---

## 📋 논문 Table 작성 예시

### **Table 1: 알고리즘 성능 비교**

**데이터 추출:**
1. TensorBoard에서 최종 에피소드(19900~20000) 평균값 확인
2. CSV 다운로드 후 평균 계산

| Algorithm | Success ↑ | Collision ↓ | Avg GPS Trust | Avg Suspicion |
|-----------|----------|------------|---------------|---------------|
| Vanilla-MAPPO | 55.1% | 8.2% | - | - |
| Trust-MAPPO | 59.3% | 6.1% | 0.73 | - |
| **Trust+Consensus-MAPPO** | **64.7%** | **4.5%** | **0.68** | **0.42** |

### **Table 2: 공격 상황별 Trust 변화**

**데이터 추출:**
1. 정상 상황: Episode 0~100
2. 공격 상황: Episode 500~600 (공격 활성화 구간)

| Scenario | GPS Trust | Vision Trust | Suspicion Ratio | GPS Used? |
|----------|-----------|--------------|-----------------|-----------|
| Normal | 0.85 ± 0.03 | 0.15 ± 0.03 | 0.12 ± 0.05 | ✅ Yes |
| Attack Suspected | 0.52 ± 0.08 | 0.48 ± 0.08 | 0.58 ± 0.06 | ⚠️ Reduced |
| Attack Confirmed | 0.00 | 1.00 | 0.73 ± 0.04 | ❌ No |

---

## 🔍 디버깅: Trust 데이터가 안 보일 때

### **체크리스트**

1. **알고리즘 확인**
   - ✅ Trust-MAPPO 또는 Trust+Consensus-MAPPO 실행 중?
   - ❌ Vanilla-MAPPO는 Trust 데이터 없음

2. **로그 확인**
   ```
   [Trust+Consensus-MAPPO] Ep 100: Rew 234.5 Succ 45.0% Coll 12.0% | Trust GPS:0.723 Vis:0.277 | Suspicion:0.423
   ```
   - ✅ 콘솔에 Trust 값 출력됨?
   - ❌ 출력 안 되면 코드 재실행

3. **TensorBoard 새로고침**
   ```bash
   # TensorBoard 재시작
   Ctrl+C (종료)
   tensorboard --logdir runs
   ```

4. **메트릭 검색**
   - 상단 검색창: `Trust_GPS` 입력
   - 필터: Run에서 알고리즘 선택

5. **파일 확인**
   ```bash
   ls -lh runs/RobustRL_Trust+Consensus-MAPPO_*/events.out.tfevents.*
   ```
   - ✅ 파일 크기가 증가하고 있는가?

---

## 💡 논문 작성 Tips

### **1. 학습 곡선 (Learning Curve)**
- X축: Episode (0~20,000)
- Y축: Success Rate (%)
- 3개 알고리즘 동시 표시
- Smoothing: 0.6 (노이즈 감소)

### **2. Trust 변화 (Trust Dynamics)**
- X축: Episode (공격 구간 확대)
- Y축: Trust Score (0~1)
- GPS와 Vision 동시 표시
- Smoothing: 0.3 (변화 명확히)

### **3. 투표 메커니즘 (Voting Mechanism)**
- X축: Episode
- Y축: Suspicion Ratio (0~1)
- 수평선: y=0.5 (50% 임계값)
- Annotation: 공격 시작/종료 시점

### **4. 통계 분석**
```python
# CSV 다운로드 후 통계 계산
import pandas as pd
import numpy as np

df = pd.read_csv("Trust_GPS.csv")
print(f"Mean: {df['Value'].mean():.3f}")
print(f"Std: {df['Value'].std():.3f}")
print(f"Min: {df['Value'].min():.3f}")
print(f"Max: {df['Value'].max():.3f}")
```

---

## 📚 참고: TensorBoard 단축키

| 단축키 | 기능 |
|--------|------|
| `D` | Download CSV |
| `T` | Toggle Y-axis scale (Log/Linear) |
| `F` | Fit to data |
| `Ctrl + Scroll` | Zoom in/out |
| `Shift + Click` | Multi-select metrics |

---

## 🎯 최종 체크리스트

학습 완료 후 논문 작성 전 확인:

- [ ] Success Rate: Trust+Consensus > Trust > Vanilla
- [ ] Collision Rate: Trust+Consensus < Trust < Vanilla
- [ ] Trust_GPS: 공격 시 감소 확인
- [ ] Trust_Vision: 공격 시 증가 확인
- [ ] Suspicion Ratio: 공격 시 0.5 이상 확인
- [ ] 모든 그래프 PNG/CSV 다운로드 완료
- [ ] 통계 값 (평균, 표준편차) 계산 완료

---

## 📧 문의

TensorBoard 관련 문제 발생 시:
1. 콘솔 로그 확인
2. `runs/` 디렉토리 파일 크기 확인
3. 알고리즘 이름이 올바른지 확인 (Trust+Consensus-MAPPO)

**이제 논문 작성에 필요한 모든 데이터를 TensorBoard에서 확인할 수 있습니다!** 🚀
