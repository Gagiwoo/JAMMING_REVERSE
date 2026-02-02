# Trust-based Cooperative Path Planning for Multi-UAV Systems

GPS 스푸핑 환경에서 신뢰도 기반 다중 UAV 협력 경로 계획 시스템

## 📚 논문 정보

- **제목**: GPS 스푸핑 환경에서 신뢰도 기반 다중 UAV 협력 경로 계획
- **저자**: 김도윤
- **출처**: 한국시뮬레이션학회 논문지 Vol. 26, No. 3 (2017. 9)
- **DOI**: http://doi.org/10.9709/JKSS.2017.26.3.035

## 🎯 프로젝트 개요

본 프로젝트는 GPS 스푸핑 공격 환경에서 다중 UAV의 안전한 협력 경로 계획을 위한 강화학습 기반 시스템입니다.

### 핵심 기술

1. **Trust Network**: 시공간적 신뢰도 학습을 통한 적응적 센서 융합
2. **Consensus Protocol**: 분산 합의 기반 집단 의사결정으로 GPS 스푸핑 탐지
3. **MAPPO**: Multi-Agent Proximal Policy Optimization을 통한 협력 학습
4. **End-to-End Learning**: 신뢰도 네트워크와 경로 계획을 단일 루프에서 학습

## 📁 파일 구조

```
webapp/
├── improved_trust_consensus_mappo.py  # 개선된 메인 코드 (논문 명세 준수)
├── original_code.py                   # 원본 코드 (251213.py)
├── ANALYSIS.md                        # 코드 분석 및 개선사항 문서
├── README.md                          # 본 파일
└── models/                            # 학습된 모델 저장 디렉토리
    └── runs/                          # TensorBoard 로그
```

## 🔧 주요 개선사항

### 1. Trust Network 아키텍처 (✅ 수정 완료)
- **이전**: 2층 × 32 뉴런
- **개선**: 3층 × 16 뉴런 (논문 명세)

### 2. Actor 네트워크 (✅ 수정 완료)
- **이전**: 2개 은닉층 (fc1, fc2)
- **개선**: 1개 은닉층 (fc1) - 불필요한 fc2 제거

### 3. Consensus Protocol (✅ 수정 완료)
- **이전**: 부드러운 조정만 수행
- **개선**: 50% 투표 기반 강제 설정 메커니즘 추가
  ```python
  if suspicion_ratio >= 0.5:  # 50% 이상 의심 표
      trust_gps = 0.0  # GPS 신뢰도 강제 0
      trust_vis = 1.0  # Vision만 사용
  ```

### 4. 하이퍼파라미터 (✅ 수정 완료)

| 파라미터 | 원본 코드 | 논문 명세 | 개선 코드 |
|---------|---------|----------|---------|
| Actor LR | 5×10⁻⁴ | 3×10⁻⁴ | ✅ 3×10⁻⁴ |
| Trust LR | 2.5×10⁻⁴ | 1.5×10⁻⁴ | ✅ 1.5×10⁻⁴ |
| Trust Lambda | 0.05 | 0.1 | ✅ 0.1 |
| Consensus Threshold | 2.0 | 2.5 | ✅ 2.5 |
| Consensus Weight | 0.2 | 0.15 | ✅ 0.15 |
| Attack Probability | 5% | 10% | ✅ 10% |

### 5. 관찰 공간 (✅ 수정 완료)
- **추가**: 속도(velocity) 정보
- **개선**: 융합된 위치(fused_pos) 사용 (GPS 대신)
- **구조**: `[fused_pos(2) + velocity(2) + target(2) + trust_feats(4) + vote(1) + neighbors + vision]`

### 6. GPS 공격 모델 (✅ 수정 완료)
- **공격 확률**: 5% → 10%
- **Step Attack**: -4.0 ~ 4.0m 오프셋
- **Drift Attack**: 0.2 ~ 0.8 m/s 누적 편향
- **지속 시간**: 10~30 스텝

## 🚀 실행 방법

### 1. 의존성 설치

```bash
pip install torch numpy pygame PySide6 qdarkstyle matplotlib tensorboard
```

### 2. 학습 실행

#### GUI 모드 (권장)
```bash
cd /home/user/webapp
python improved_trust_consensus_mappo.py
```

GUI에서:
1. 비교하고 싶은 알고리즘 선택
2. 공격 모드 선택 (hybrid, step, drift, none)
3. 학습 파라미터 설정
4. "학습 시작" 버튼 클릭

#### 커맨드라인 모드
```python
# Python 스크립트에서 직접 실행
from improved_trust_consensus_mappo import *

config = BASE_CONFIG.copy()
config.update(ALGORITHM_CONFIGS["Trust+Consensus-MAPPO"])

env = CTDEMultiUAVEnv(config)
agent = MAPPOAgentWithTrust(env.local_obs_dim, env.global_obs_dim, env.action_dim, config)

# 학습 루프...
```

### 3. TensorBoard 모니터링

```bash
tensorboard --logdir=runs
```

브라우저에서 `http://localhost:6006` 접속

## 📊 비교 실험 알고리즘

1. **Vanilla-MAPPO**: 기본 MAPPO (Baseline, GPS 스푸핑 무방비)
2. **LSTM-MAPPO**: LSTM 기반 시계열 학습
3. **Trust-MAPPO**: Trust Network만 사용 (Ablation Study)
4. **Trust+Consensus-MAPPO**: 제안 기법 (Full, 논문)
5. **LSTM-Detector-MAPPO**: LSTM 기반 GPS 보정 Baseline

## 📈 평가 지표

- **Success Rate**: 목표 도달 성공률
- **Collision Rate**: 충돌 발생률
- **Average Path Length**: 평균 경로 길이
- **Reward**: 누적 보상
- **GPS Trust Score**: GPS 신뢰도 점수 (Trust Network 출력)
- **Suspicion Ratio**: 의심 표 비율 (Consensus Protocol)

## 🔬 실험 설정

### 환경 파라미터
- **Grid Size**: 40 × 40
- **UAV 수**: 10
- **장애물 수**: 40
- **Vision Range**: 5 cells
- **Max Steps**: 200

### 공격 시나리오
- **Hybrid Attack** (기본): Step + Drift 랜덤 혼합
- **Step Attack**: 순간 점프 공격
- **Drift Attack**: 점진적 누적 공격
- **None**: 공격 없음 (정상 환경)

## 💡 사용 팁

### 빠른 테스트
```python
config = BASE_CONFIG.copy()
config["total_episodes"] = 1000
config["episodes_per_batch"] = 5
config["num_obstacles"] = 20
```

### GPU 사용
```python
# 자동으로 CUDA 감지
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 모델 저장/로드
```python
# 저장
agent.save_models("./models/my_model")

# 로드
agent.load_models("./models/my_model")
```

## 🐛 알려진 이슈 및 해결

### 1. PySide6 버전 호환성
```bash
# Qt API 오류 시
pip install PySide6==6.4.0
```

### 2. Matplotlib 폰트 경고
```python
# 이미 코드에 포함됨
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
```

### 3. CUDA Out of Memory
```python
# Batch size 줄이기
config["batch_size"] = 256  # 기본: 512
```

## 📖 참고 자료

- **논문 PDF**: `GPS 스푸핑 환경에서 신뢰도 기반 다중 UAV 협력 경로 계획_김도윤.pdf`
- **분석 문서**: `ANALYSIS.md` (코드 분석 및 개선사항 상세)
- **원본 코드**: `original_code.py` (비교용)

## 🤝 기여

논문 저자: 김도윤
코드 개선: AI Code Reviewer

### 개선 기여 내역
- Trust Network 아키텍처 논문 명세 준수
- Consensus Protocol 50% 투표 메커니즘 구현
- 하이퍼파라미터 정확도 개선
- 관찰 공간 구조 개선 (융합된 위치 + 속도)
- 코드 주석 및 문서화 강화

## 📝 License

논문 및 연구 목적으로 사용 가능합니다.
상업적 사용 시 저자의 허가가 필요합니다.

## 📧 Contact

논문 관련 문의: [논문 저자 연락처]
코드 관련 이슈: [GitHub Issues]

---

**Last Updated**: 2024-01-XX
**Version**: 2.0 (Improved, 논문 명세 준수)
