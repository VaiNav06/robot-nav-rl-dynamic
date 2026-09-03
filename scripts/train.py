"""
Train two PPO policies on DynamicNavEnv for the domain-randomization ablation:

  1. "narrow"  -- trained with obstacle speeds drawn from a NARROW range.
                  This is the low-domain-randomization condition.
  2. "wide"    -- trained with obstacle speeds drawn from a WIDE range.
                  This is the high-domain-randomization condition.

Both policies are then evaluated (see evaluate.py) on a held-out obstacle
speed range that neither has trained on directly, to measure how well each
generalizes.

Run with: python scripts/train.py --condition narrow
          python scripts/train.py --condition wide
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from robonav_env.dynamic_nav_env import DynamicNavEnv

# Obstacle speed ranges (units: arena-lengths per second) used for the ablation.
# "narrow": obstacles are always slow and move in a tight speed band.
# "wide":   obstacles vary widely in speed, forcing a more general policy.
CONDITIONS = {
    "narrow": (0.3, 0.4),
    "wide": (0.1, 0.9),
}

# Held-out evaluation range used by evaluate.py -- deliberately overlaps
# partially with both training ranges rather than being wildly out-of-distribution,
# so the comparison reflects realistic generalization, not an impossible task.
EVAL_SPEED_RANGE = (0.5, 0.8)


def make_env(speed_range, seed):
    def _init():
        env = DynamicNavEnv(obstacle_speed_range=speed_range, seed=seed)
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["narrow", "wide"], required=True)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n_envs", type=int, default=8)
    args = parser.parse_args()

    speed_range = CONDITIONS[args.condition]
    print(f"Training condition: {args.condition}  (obstacle speed range = {speed_range})")

    vec_env = make_vec_env(
        lambda: DynamicNavEnv(obstacle_speed_range=speed_range),
        n_envs=args.n_envs,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        tensorboard_log=f"./results/tb_{args.condition}",
    )
    model.learn(total_timesteps=args.timesteps)

    out_path = f"./results/ppo_{args.condition}.zip"
    model.save(out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
