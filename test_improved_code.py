#!/usr/bin/env python3
"""
간단한 테스트 스크립트
개선된 코드가 정상적으로 작동하는지 확인
"""

import sys
import numpy as np
import torch

# 코드 import
try:
    from improved_trust_consensus_mappo import (
        BASE_CONFIG,
        ALGORITHM_CONFIGS,
        TrustNetwork,
        ConsensusProtocol,
        Actor,
        Critic,
        CTDEMultiUAVEnv,
        EnvironmentScenario,
        MAPPOAgentWithTrust
    )
    print("✅ 모든 모듈 import 성공")
except Exception as e:
    print(f"❌ Import 실패: {e}")
    sys.exit(1)

def test_trust_network():
    """Trust Network 테스트"""
    print("\n=== Trust Network 테스트 ===")
    trust_net = TrustNetwork(hidden=16)
    
    # 입력: 4차원 (temporal_res, spatial_disc, gps_var, neighbor_flag)
    test_input = torch.randn(8, 4)  # batch_size=8
    output = trust_net(test_input)
    
    print(f"입력 shape: {test_input.shape}")
    print(f"출력 shape: {output.shape}")
    print(f"출력 합계 (각 행): {output.sum(dim=1)}")  # Should be ~1.0 (Softmax)
    
    # 네트워크 구조 확인
    print(f"\n네트워크 레이어 수: {len([m for m in trust_net.network if isinstance(m, torch.nn.Linear)])}")
    for i, module in enumerate(trust_net.network):
        if isinstance(module, torch.nn.Linear):
            print(f"  Layer {i}: {module.in_features} → {module.out_features}")
    
    assert output.shape == (8, 2), "출력 shape 오류"
    assert torch.allclose(output.sum(dim=1), torch.ones(8), atol=1e-5), "Softmax 합계 오류"
    print("✅ Trust Network 테스트 통과")

def test_consensus_protocol():
    """Consensus Protocol 테스트"""
    print("\n=== Consensus Protocol 테스트 ===")
    consensus = ConsensusProtocol(threshold=2.5, consensus_weight=0.15, vote_threshold=0.5)
    
    # 의심 표 테스트
    discrepancies = [3.0, 2.8, 1.0, 3.5, 2.6]  # 5개 이웃, 4개가 threshold 초과
    votes = consensus.cast_votes(discrepancies)
    print(f"불일치 값: {discrepancies}")
    print(f"투표 결과: {votes}")
    print(f"의심 표 수: {sum(votes)}/{len(votes)}")
    
    is_attack, ratio = consensus.aggregate_votes(votes)
    print(f"공격 여부: {is_attack}, 의심 비율: {ratio:.2%}")
    
    # Trust 조정 테스트
    t_gps, t_vis = consensus.adjust_trust(0.6, 0.4, 2.8, force_zero=False)
    print(f"\n정상 조정: GPS={t_gps:.3f}, Vision={t_vis:.3f}")
    
    t_gps_forced, t_vis_forced = consensus.adjust_trust(0.6, 0.4, 2.8, force_zero=True)
    print(f"강제 설정: GPS={t_gps_forced:.3f}, Vision={t_vis_forced:.3f}")
    
    assert is_attack == True, "50% 이상 의심 표 시 공격으로 판단해야 함"
    assert t_gps_forced == 0.0 and t_vis_forced == 1.0, "강제 설정 오류"
    print("✅ Consensus Protocol 테스트 통과")

def test_actor_network():
    """Actor Network 테스트"""
    print("\n=== Actor Network 테스트 ===")
    local_dim = 100
    act_dim = 8
    actor = Actor(local_dim, act_dim, hidden=128, use_lstm=False)
    
    test_input = torch.randn(4, local_dim)
    output = actor(test_input)
    
    print(f"입력 shape: {test_input.shape}")
    print(f"출력 shape: {output.shape}")
    print(f"출력 합계 (각 행): {output.sum(dim=1)}")  # Should be ~1.0 (Softmax)
    
    # 레이어 수 확인 (fc1 → head, fc2 제거됨)
    fc_layers = [name for name, _ in actor.named_modules() if 'fc' in name or 'head' in name]
    print(f"FC 레이어: {fc_layers}")
    
    assert output.shape == (4, 8), "출력 shape 오류"
    assert torch.allclose(output.sum(dim=1), torch.ones(4), atol=1e-5), "Softmax 합계 오류"
    print("✅ Actor Network 테스트 통과")

def test_environment():
    """환경 테스트"""
    print("\n=== 환경 테스트 ===")
    config = BASE_CONFIG.copy()
    config["num_uavs"] = 3
    config["grid_size"] = 20
    config["num_obstacles"] = 10
    config["max_steps"] = 50
    
    env = CTDEMultiUAVEnv(config)
    scenario = EnvironmentScenario(config)
    
    local_obs, global_obs = env.reset_with_scenario(scenario)
    
    print(f"UAV 수: {env.num_uavs}")
    print(f"Local Obs Dim: {env.local_obs_dim}")
    print(f"Global Obs Dim: {env.global_obs_dim}")
    print(f"Action Dim: {env.action_dim}")
    
    # 한 스텝 실행
    actions = {agent: np.random.randint(0, env.action_dim) for agent in env.agents}
    next_local, next_global, rewards, done, info = env.step(actions)
    
    print(f"\n첫 스텝 실행:")
    print(f"  보상: {list(rewards.values())}")
    print(f"  완료: {done}")
    
    assert len(local_obs) == env.num_uavs, "Local obs 개수 오류"
    assert len(rewards) == env.num_uavs, "보상 개수 오류"
    print("✅ 환경 테스트 통과")

def test_agent():
    """Agent 테스트"""
    print("\n=== Agent 테스트 ===")
    config = BASE_CONFIG.copy()
    config["num_uavs"] = 3
    config["grid_size"] = 20
    config.update(ALGORITHM_CONFIGS["Trust+Consensus-MAPPO"])
    
    env = CTDEMultiUAVEnv(config)
    agent = MAPPOAgentWithTrust(env.local_obs_dim, env.global_obs_dim, env.action_dim, config)
    
    print(f"Agent 구성:")
    print(f"  Trust Network 사용: {agent.use_trust}")
    print(f"  Consensus 사용: {agent.use_consensus}")
    
    scenario = EnvironmentScenario(config)
    local_obs, global_obs = env.reset_with_scenario(scenario)
    agent.reset_episode(env.agents)
    
    # 액션 선택
    actions, log_probs, value, trust_info = agent.select_action(
        local_obs, global_obs, env.uav_positions, env.gps_positions, env=env
    )
    
    print(f"\n액션 선택:")
    print(f"  Actions: {list(actions.values())}")
    print(f"  Value: {value:.3f}")
    print(f"  Trust Info (UAV 0): GPS={trust_info['uav_0']['gps']:.3f}, Vision={trust_info['uav_0']['vis']:.3f}")
    
    assert len(actions) == env.num_uavs, "액션 개수 오류"
    assert len(trust_info) == env.num_uavs, "Trust info 개수 오류"
    print("✅ Agent 테스트 통과")

def test_config():
    """설정 검증"""
    print("\n=== 설정 검증 ===")
    config = BASE_CONFIG.copy()
    
    # 논문 명세 준수 확인
    checks = [
        ("Actor LR", config["mappo_lr"], 3e-4),
        ("Trust LR", config["trust_lr"], 1.5e-4),
        ("Trust Lambda", config["trust_lambda_reg"], 0.1),
        ("Consensus Threshold", config["consensus_threshold"], 2.5),
        ("Consensus Weight", config["consensus_weight"], 0.15),
        ("Attack Start Prob", config["attack_start_prob"], 0.1),
        ("Trust Hidden", config["trust_hidden"], 16),
    ]
    
    all_pass = True
    for name, actual, expected in checks:
        status = "✅" if actual == expected else "❌"
        print(f"{status} {name}: {actual} (기대값: {expected})")
        if actual != expected:
            all_pass = False
    
    if all_pass:
        print("\n✅ 모든 설정이 논문 명세를 준수합니다")
    else:
        print("\n⚠️ 일부 설정이 논문 명세와 다릅니다")
    
    return all_pass

def main():
    """전체 테스트 실행"""
    print("=" * 60)
    print("개선된 Trust-Consensus MAPPO 코드 테스트")
    print("=" * 60)
    
    tests = [
        ("설정 검증", test_config),
        ("Trust Network", test_trust_network),
        ("Consensus Protocol", test_consensus_protocol),
        ("Actor Network", test_actor_network),
        ("환경", test_environment),
        ("Agent", test_agent),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"테스트 결과: {passed}/{len(tests)} 통과, {failed}/{len(tests)} 실패")
    print("=" * 60)
    
    if failed == 0:
        print("✅ 모든 테스트 통과! 코드가 정상적으로 작동합니다.")
        return 0
    else:
        print("❌ 일부 테스트 실패. 코드를 수정해주세요.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
