"""Quick sanity check: run random-action rollouts to confirm the env is well-formed
before spending any training compute on it."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from robonav_env.dynamic_nav_env import DynamicNavEnv

env = DynamicNavEnv(seed=42)
obs, info = env.reset()
print(f"Observation shape: {obs.shape}  (expected {env.observation_space.shape})")
print(f"Action space: {env.action_space}")
print(f"Sample obs (first 5 dims): {obs[:5]}")

n_episodes = 20
outcomes = {"collided": 0, "reached_goal": 0, "timed_out": 0}
episode_lengths = []

rng = np.random.default_rng(0)
for ep in range(n_episodes):
    obs, _ = env.reset(seed=ep)
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = rng.uniform(-1, 1, size=2)  # random policy
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
    episode_lengths.append(steps)
    if info.get("collided"):
        outcomes["collided"] += 1
    elif info.get("reached_goal"):
        outcomes["reached_goal"] += 1
    else:
        outcomes["timed_out"] += 1

print(f"\nRandom-policy baseline over {n_episodes} episodes:")
print(f"  Collided:     {outcomes['collided']}")
print(f"  Reached goal: {outcomes['reached_goal']}")
print(f"  Timed out:    {outcomes['timed_out']}")
print(f"  Avg episode length: {np.mean(episode_lengths):.1f} steps")
print("\n(Random policy should rarely reach the goal by chance -- this confirms")
print(" the task is non-trivial and gives PPO real room to demonstrate learning.)")
