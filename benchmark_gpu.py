#!/usr/bin/env python3
"""
GPU 성능 벤치마크 스크립트

일반 모드 vs 고속 모드 성능 비교
"""

import time
import torch
import numpy as np
from improved_trust_consensus_mappo import *

def benchmark_gpu():
    """GPU 성능 벤치마크"""
    print("=" * 70)
    print("🔥 GPU 성능 벤치마크")
    print("=" * 70)
    
    # GPU 정보
    if torch.cuda.is_available():
        print(f"✅ CUDA 사용 가능")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("❌ CUDA 사용 불가 - CPU 모드")
        return
    
    print("\n" + "=" * 70)
    print("테스트 1: 네트워크 Forward Pass 속도")
    print("=" * 70)
    
    # Actor 네트워크 테스트
    config = BASE_CONFIG.copy()
    env = CTDEMultiUAVEnv(config)
    actor = Actor(env.local_obs_dim, env.action_dim, hidden=128).to(DEVICE)
    
    # 다양한 배치 크기 테스트
    batch_sizes = [1, 10, 50, 100, 500, 1000]
    
    for batch_size in batch_sizes:
        # 랜덤 입력 생성
        dummy_input = torch.randn(batch_size, env.local_obs_dim, device=DEVICE)
        
        # Warm-up
        for _ in range(10):
            _ = actor(dummy_input)
        
        # 벤치마크
        torch.cuda.synchronize()
        start = time.time()
        
        iterations = 100
        for _ in range(iterations):
            _ = actor(dummy_input)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        throughput = (batch_size * iterations) / elapsed
        
        print(f"  Batch Size {batch_size:4d}: {throughput:8.0f} samples/sec ({elapsed*1000/iterations:.2f} ms/iter)")
    
    print("\n" + "=" * 70)
    print("테스트 2: 메모리 사용량")
    print("=" * 70)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 큰 배치로 메모리 테스트
    large_batch = 2000
    dummy_input = torch.randn(large_batch, env.local_obs_dim, device=DEVICE)
    _ = actor(dummy_input)
    
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2
    peak = torch.cuda.max_memory_allocated(0) / 1024**2
    
    print(f"  현재 할당: {allocated:.1f} MB")
    print(f"  예약됨:     {reserved:.1f} MB")
    print(f"  최대 사용:  {peak:.1f} MB")
    
    print("\n" + "=" * 70)
    print("테스트 3: CPU vs GPU 비교")
    print("=" * 70)
    
    batch_size = 100
    iterations = 100
    
    # CPU 테스트
    actor_cpu = Actor(env.local_obs_dim, env.action_dim, hidden=128).to('cpu')
    dummy_input_cpu = torch.randn(batch_size, env.local_obs_dim)
    
    start = time.time()
    for _ in range(iterations):
        _ = actor_cpu(dummy_input_cpu)
    cpu_time = time.time() - start
    
    # GPU 테스트
    dummy_input_gpu = torch.randn(batch_size, env.local_obs_dim, device=DEVICE)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = actor(dummy_input_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    speedup = cpu_time / gpu_time
    
    print(f"  CPU: {cpu_time:.3f}초 ({batch_size*iterations/cpu_time:.0f} samples/sec)")
    print(f"  GPU: {gpu_time:.3f}초 ({batch_size*iterations/gpu_time:.0f} samples/sec)")
    print(f"  ⚡ 속도 향상: {speedup:.1f}x")
    
    print("\n" + "=" * 70)
    print("테스트 4: 실제 학습 시뮬레이션 (10 스텝)")
    print("=" * 70)
    
    agent = MAPPOAgentWithTrust(env.local_obs_dim, env.global_obs_dim, env.action_dim, config)
    
    # 짧은 학습 시뮬레이션
    scenario = EnvironmentScenario(config)
    lo, go = env.reset_with_scenario(scenario)
    agent.reset_episode(env.agents)
    
    start = time.time()
    
    for step in range(10):
        acts, logs, val, _ = agent.select_action(lo, go, env.uav_positions, env.gps_positions, env=env)
        lo, go, rew, done, info = env.step(acts)
        
        if done:
            break
    
    elapsed = time.time() - start
    
    print(f"  10 스텝 실행 시간: {elapsed:.3f}초")
    print(f"  스텝당 평균: {elapsed/10*1000:.1f} ms")
    print(f"  예상 에피소드 시간 (200 스텝): {elapsed*20:.1f}초")
    
    print("\n" + "=" * 70)
    print("💡 최적화 권장사항")
    print("=" * 70)
    
    # GPU 활용도 추정
    if speedup < 5:
        print("⚠️ GPU 가속이 충분하지 않습니다")
        print("   → 배치 크기를 늘려보세요 (episodes_per_batch: 20)")
        print("   → 환경 병렬화를 고려하세요")
    else:
        print("✅ GPU 가속이 잘 작동하고 있습니다")
    
    if allocated < 500:  # 500MB 미만
        print("⚠️ GPU 메모리 사용량이 적습니다")
        print("   → 배치 크기를 더 늘릴 수 있습니다")
        print("   → 네트워크 크기를 키울 수 있습니다")
    
    # 예상 학습 시간 계산
    steps_per_episode = 150
    episodes_total = 10000
    episodes_per_batch = 10
    
    time_per_episode = elapsed * (steps_per_episode / 10)
    total_time_hours = (time_per_episode * episodes_total) / 3600
    
    print(f"\n📊 예상 전체 학습 시간 (10,000 에피소드):")
    print(f"   일반 모드: {total_time_hours:.1f} 시간")
    
    # 최적화 후 예상 시간
    optimized_time = total_time_hours * 0.4  # 60% 감소 예상
    print(f"   고속 모드: {optimized_time:.1f} 시간 (예상)")
    print(f"   ⚡ 절약 시간: {total_time_hours - optimized_time:.1f} 시간")
    
    print("\n" + "=" * 70)


def monitor_gpu_usage():
    """
    실시간 GPU 사용률 모니터링 (nvidia-smi 대신)
    """
    print("\n실시간 GPU 모니터링 (Ctrl+C로 종료)")
    print("-" * 70)
    print(f"{'Time':<12} {'Memory Used':<15} {'Memory Total':<15} {'Utilization':<15}")
    print("-" * 70)
    
    try:
        while True:
            mem_alloc = torch.cuda.memory_allocated(0) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**2
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            
            util = (mem_alloc / mem_total) * 100
            
            timestamp = time.strftime("%H:%M:%S")
            print(f"{timestamp:<12} {mem_alloc:>6.1f} MB      {mem_total:>6.1f} MB      {util:>5.1f}%", end='\r')
            
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\n모니터링 종료")


def quick_test():
    """빠른 학습 테스트 (1분)"""
    print("\n" + "=" * 70)
    print("🚀 빠른 학습 테스트 (1분)")
    print("=" * 70)
    
    config = BASE_CONFIG.copy()
    config.update(ALGORITHM_CONFIGS["Trust+Consensus-MAPPO"])
    config["total_episodes"] = 10
    config["episodes_per_batch"] = 2
    config["num_obstacles"] = 20
    config["max_steps"] = 50
    
    env = CTDEMultiUAVEnv(config)
    agent = MAPPOAgentWithTrust(env.local_obs_dim, env.global_obs_dim, env.action_dim, config)
    
    print("설정:")
    print(f"  Episodes: {config['total_episodes']}")
    print(f"  UAVs: {config['num_uavs']}")
    print(f"  Max Steps: {config['max_steps']}")
    
    start_time = time.time()
    
    for ep in range(0, config["total_episodes"], config["episodes_per_batch"]):
        print(f"\n배치 {ep//config['episodes_per_batch'] + 1}/{config['total_episodes']//config['episodes_per_batch']}")
        
        for _ in range(config["episodes_per_batch"]):
            scenario = EnvironmentScenario(config)
            lo, go = env.reset_with_scenario(scenario)
            agent.reset_episode(env.agents)
            done = False
            
            ep_obs, ep_glo, ep_act, ep_logp, ep_val, ep_rew, ep_done = [],[],[],[],[],[],[]
            
            while not done:
                acts, logs, val, _ = agent.select_action(lo, go, env.uav_positions, env.gps_positions, env=env)
                n_lo, n_go, rew, done, info = env.step(acts)
                
                ep_obs.extend([lo[a] for a in env.agents if a in acts])
                ep_glo.extend([go for _ in acts])
                ep_act.extend(list(acts.values()))
                ep_logp.extend(list(logs.values()))
                ep_val.extend([val for _ in acts])
                ep_rew.extend(list(rew.values()))
                ep_done.extend([done for _ in acts])
                
                lo, go = n_lo, n_go
            
            agent.buffer.add(ep_obs, ep_glo, ep_act, ep_logp, ep_val, ep_rew, ep_done)
            print(f"  에피소드 완료: Success={info['success_rate']:.0%}")
        
        # 업데이트
        with torch.no_grad():
            next_val = agent.critic(torch.tensor(go, dtype=torch.float32, device=DEVICE).unsqueeze(0)).item()
        agent.compute_gae(next_val)
        agent.update()
        print(f"  네트워크 업데이트 완료")
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(f"✅ 테스트 완료: {elapsed:.1f}초")
    print(f"   에피소드당 평균: {elapsed/config['total_episodes']:.1f}초")
    
    # 전체 학습 시간 예측
    full_episodes = 10000
    predicted_time = (elapsed / config['total_episodes']) * full_episodes / 3600
    
    print(f"\n📊 10,000 에피소드 예상 시간: {predicted_time:.1f} 시간")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "monitor":
            monitor_gpu_usage()
        elif sys.argv[1] == "quick":
            quick_test()
        else:
            print("Usage:")
            print("  python benchmark_gpu.py          - 전체 벤치마크")
            print("  python benchmark_gpu.py monitor  - GPU 모니터링")
            print("  python benchmark_gpu.py quick    - 빠른 학습 테스트")
    else:
        benchmark_gpu()
