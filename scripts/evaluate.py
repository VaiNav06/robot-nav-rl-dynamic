"""
Evaluate the "narrow" vs "wide" domain-randomization policies on a held-out
obstacle speed range, and report the generalization gap.

Run with: python scripts/evaluate.py
(after training.py has produced results/ppo_narrow.zip and results/ppo_wide.zip)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from robonav_env.dynamic_nav_env import DynamicNavEnv
from train import EVAL_SPEED_RANGE, CONDITIONS

N_EVAL_EPISODES = 100


def evaluate_policy(model, speed_range, n_episodes=N_EVAL_EPISODES):
    successes, collisions, timeouts = 0, 0, 0
    for ep in range(n_episodes):
        env = DynamicNavEnv(obstacle_speed_range=speed_range, seed=10_000 + ep)
        obs, _ = env.reset(seed=10_000 + ep)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
        if info.get("reached_goal"):
            successes += 1
        elif info.get("collided"):
            collisions += 1
        else:
            timeouts += 1
    return {
        "success_rate": successes / n_episodes,
        "collision_rate": collisions / n_episodes,
        "timeout_rate": timeouts / n_episodes,
    }


def main():
    results = {}
    for condition in ["narrow", "wide"]:
        model_path = f"./results/ppo_{condition}.zip"
        if not os.path.exists(model_path):
            print(f"Missing {model_path} -- run train.py --condition {condition} first.")
            return
        model = PPO.load(model_path)

        # In-distribution: evaluate on the SAME range it trained on (sanity check).
        in_dist = evaluate_policy(model, CONDITIONS[condition])
        # Out-of-distribution: evaluate on the held-out range (the real test).
        ood = evaluate_policy(model, EVAL_SPEED_RANGE)

        results[condition] = {"in_distribution": in_dist, "held_out": ood}
        print(f"\n[{condition}] in-distribution success rate: {in_dist['success_rate']:.1%}")
        print(f"[{condition}] held-out success rate:         {ood['success_rate']:.1%}")

    gap = (results["wide"]["held_out"]["success_rate"]
           - results["narrow"]["held_out"]["success_rate"])
    print(f"\n=== Generalization gap (wide - narrow) on held-out obstacle speeds: "
          f"{gap:+.1%} ===")

    # Plot comparison
    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = ["narrow\n(low randomization)", "wide\n(high randomization)"]
    held_out_rates = [results["narrow"]["held_out"]["success_rate"],
                       results["wide"]["held_out"]["success_rate"]]
    bars = ax.bar(labels, held_out_rates, color=["tab:red", "tab:green"])
    ax.set_ylabel("Success rate on held-out obstacle speeds")
    ax.set_ylim(0, 1)
    ax.set_title("Domain randomization ablation: generalization to unseen dynamics")
    for bar, rate in zip(bars, held_out_rates):
        ax.annotate(f"{rate:.1%}", (bar.get_x() + bar.get_width() / 2, rate + 0.02),
                    ha="center")
    plt.tight_layout()
    plt.savefig("./results/ablation_comparison.png", dpi=130)
    print("Saved plot to results/ablation_comparison.png")


if __name__ == "__main__":
    main()
