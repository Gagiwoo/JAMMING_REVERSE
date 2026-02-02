# 🎮 데모/시연 기능 사용 가이드

학습된 모델을 시각화하여 실시간으로 UAV 네비게이션을 관찰할 수 있습니다!

---

## 🚀 데모 실행 방법

### 1. GUI에서 데모 버튼 사용

#### Step 1: 알고리즘 선택
GUI에서 데모를 실행할 알고리즘을 **하나만** 선택합니다.

예: `Trust+Consensus-MAPPO` 체크박스 선택

#### Step 2: 데모 버튼 클릭
`🎮 데모 실행` 버튼을 클릭합니다.

#### Step 3: 모델 디렉토리 선택
파일 선택 다이얼로그가 나타나면 학습된 모델이 저장된 폴더를 선택합니다.

**모델 경로 예시**:
```
./models/RobustRL_Trust+Consensus-MAPPO_hybrid_obs40_1234567890/final/
```

또는 체크포인트:
```
./models/RobustRL_Trust+Consensus-MAPPO_hybrid_obs40_1234567890/ckpt_5000/
```

#### Step 4: 시각화 관찰
Pygame 윈도우가 열리면서 UAV들이 실시간으로 움직이는 모습을 볼 수 있습니다!

---

## 🎬 데모 화면 설명

### 시각화 요소

**Pygame 윈도우 (600x600)**:
- **회색 사각형**: 장애물
- **색깔 원**: UAV (빨강, 초록, 파랑, 노랑, 보라...)
- **색깔 선**: UAV의 이동 경로 (trajectory)
- **실시간 이동**: 각 UAV가 목표를 향해 장애물을 피하며 이동

### 로그 출력

데모 실행 중 GUI의 로그 창에 다음 정보가 표시됩니다:

```
🎮 [Trust+Consensus-MAPPO] 데모 실행 중...
📁 모델 경로: ./models/.../final
✅ 모델 로드 완료

📺 에피소드 1/3 시작
  보상: 234.5, 성공률: 80.0%, 충돌률: 10.0%

📺 에피소드 2/3 시작
  보상: 289.1, 성공률: 90.0%, 충돌률: 0.0%

📺 에피소드 3/3 시작
  보상: 312.7, 성공률: 100.0%, 충돌률: 0.0%

✅ 데모 완료
```

---

## ⚙️ 데모 설정

데모는 다음 설정으로 실행됩니다:

| 설정 | 값 | 설명 |
|-----|-----|------|
| **에피소드 수** | 3 | `demo_episodes` 설정 |
| **렌더 딜레이** | 0.1초 | `render_delay` 설정 |
| **최대 스텝** | 200 | `max_steps` 설정 |
| **정책** | Deterministic | 탐험 없이 학습된 정책만 사용 |
| **시각화** | Pygame | 실시간 렌더링 |

### 설정 변경 방법

`BASE_CONFIG`에서 데모 관련 설정을 변경할 수 있습니다:

```python
BASE_CONFIG = {
    # ...
    "demo_episodes": 3,      # 데모 에피소드 수
    "render_delay": 0.1,     # 렌더링 간격 (초)
    "max_steps": 200,        # 최대 스텝 수
    # ...
}
```

---

## 🔧 트러블슈팅

### 1. "모델 폴더가 선택되지 않았습니다"
**원인**: 모델 디렉토리 선택 다이얼로그에서 취소를 눌렀거나 잘못된 폴더 선택

**해결**:
- 다시 `🎮 데모 실행` 버튼 클릭
- `./models/` 디렉토리 내의 학습 완료된 모델 폴더 선택
- 폴더 내에 `actor.pth`, `critic.pth` 파일이 있는지 확인

### 2. "모델 로드 실패, 랜덤 정책 사용"
**원인**: 선택한 폴더에 모델 파일이 없거나 손상됨

**해결**:
- 학습이 완료된 모델인지 확인
- `final/` 또는 `ckpt_XXXX/` 폴더 선택
- 랜덤 정책으로도 데모는 실행되지만, 성능은 낮음

### 3. "알고리즘을 선택해주세요"
**원인**: 데모 실행 전 알고리즘 체크박스 미선택

**해결**:
- 한 개의 알고리즘 체크박스 선택 후 다시 시도

### 4. "데모는 한 번에 하나의 알고리즘만 실행 가능합니다"
**원인**: 여러 알고리즘 체크박스를 동시에 선택

**해결**:
- 하나의 알고리즘만 선택 (다른 체크박스 해제)

### 5. Pygame 윈도우가 응답하지 않음
**원인**: 데모 실행 중이거나 완료 후 윈도우가 닫히지 않음

**해결**:
- Pygame 윈도우의 X 버튼 클릭하여 수동으로 닫기
- 또는 프로그램 재시작

---

## 💡 활용 팁

### 1. 알고리즘 비교
여러 알고리즘을 순차적으로 데모 실행하여 성능 비교:

```
1. Trust+Consensus-MAPPO 데모 → 성공률 90%
2. Vanilla-MAPPO 데모 → 성공률 50%
3. LSTM-MAPPO 데모 → 성공률 70%
```

### 2. 공격 시나리오별 테스트
다른 공격 모드로 학습된 모델 비교:

```
- hybrid 공격 모델
- step 공격 모델
- drift 공격 모델
- none (정상) 모델
```

### 3. 학습 진행 상황 확인
체크포인트를 순차적으로 데모하여 학습 과정 관찰:

```
ckpt_0     → 랜덤에 가까운 움직임
ckpt_1000  → 기본적인 장애물 회피
ckpt_5000  → 협력 행동 출현
final      → 완성된 협력 네비게이션
```

### 4. GPS 스푸핑 탐지 관찰
Trust+Consensus-MAPPO 데모 중 주의 깊게 관찰:
- GPS 공격 발생 시 UAV 행동 변화
- Consensus Protocol에 의한 협력 탐지
- Vision 센서로의 전환

### 5. 속도 조절
더 빠르게 또는 느리게 보고 싶다면 `render_delay` 조정:

```python
"render_delay": 0.05,   # 빠르게 (2배속)
"render_delay": 0.1,    # 기본
"render_delay": 0.2,    # 느리게 (절반 속도)
```

---

## 📹 데모 영상 녹화 (선택)

Pygame 화면을 녹화하려면:

### Windows
- OBS Studio 사용
- Windows Game Bar (Win + G)

### Mac
- QuickTime Player (화면 녹화)
- OBS Studio

### Linux
- SimpleScreenRecorder
- OBS Studio

---

## 🎯 데모 모범 사례

### 논문 재현용
```
1. Trust+Consensus-MAPPO (제안 기법) 데모
2. Vanilla-MAPPO (Baseline) 데모
3. 성공률, 충돌률 비교
4. 스크린샷/영상 캡처
```

### 발표/시연용
```
1. 정상 환경 (none) 데모 → 기본 성능 확인
2. GPS 공격 환경 (hybrid) 데모 → 공격 대응 능력 확인
3. 로그 출력을 보여주며 설명
```

### 디버깅용
```
1. 체크포인트별 데모 → 학습 진행 확인
2. 다른 obstacle 수 설정으로 테스트
3. 비정상 행동 발견 시 로그 분석
```

---

## 📊 데모 결과 해석

### 좋은 모델의 특징
- ✅ 성공률 80% 이상
- ✅ 충돌률 10% 이하
- ✅ 부드러운 경로 (지그재그 없음)
- ✅ 협력 행동 (UAV들이 서로 회피)
- ✅ GPS 공격 시 빠른 대응

### 개선이 필요한 모델
- ❌ 성공률 50% 미만
- ❌ 충돌률 30% 이상
- ❌ 불규칙한 움직임
- ❌ UAV 간 충돌 빈번
- ❌ GPS 공격 시 혼란

---

## 🔄 코드로 데모 실행 (고급)

GUI 없이 코드로 직접 실행:

```python
from improved_trust_consensus_mappo import *

# 설정
config = BASE_CONFIG.copy()
config.update(ALGORITHM_CONFIGS["Trust+Consensus-MAPPO"])
config["render_mode"] = "human"

# 환경 및 에이전트
env = CTDEMultiUAVEnv(config, render_mode="human")
agent = MAPPOAgentWithTrust(env.local_obs_dim, env.global_obs_dim, env.action_dim, config)

# 모델 로드
agent.load_models("./models/RobustRL_Trust+Consensus-MAPPO_hybrid_obs40_1234567890/final")

# 데모 실행
for ep in range(3):
    scenario = EnvironmentScenario(config)
    lo, go = env.reset_with_scenario(scenario)
    agent.reset_episode(env.agents)
    done = False
    
    while not done:
        acts, _, _, _ = agent.select_action(lo, go, env.uav_positions, env.gps_positions, env=env, deterministic=True)
        lo, go, rew, done, info = env.step(acts)
        env.render()
        time.sleep(0.1)
    
    print(f"Episode {ep+1}: Success={info['success_rate']:.1%}, Collision={info['collision_rate']:.1%}")

env.close()
```

---

## ✅ 체크리스트

데모 실행 전:
- [ ] 모델이 학습 완료되었는가?
- [ ] `./models/` 디렉토리에 모델 폴더가 있는가?
- [ ] Pygame이 설치되어 있는가? (`pip install pygame`)
- [ ] 알고리즘을 하나만 선택했는가?

데모 실행 중:
- [ ] Pygame 윈도우가 정상적으로 열렸는가?
- [ ] UAV들이 움직이는가?
- [ ] 로그 창에 메시지가 출력되는가?

데모 완료 후:
- [ ] 성공률/충돌률을 기록했는가?
- [ ] 영상/스크린샷을 캡처했는가? (필요 시)
- [ ] Pygame 윈도우를 닫았는가?

---

**작성일**: 2024
**버전**: 1.0
**관련 커밋**: `b019002`
