"""Render a single frame of the environment (robot, LIDAR fan, obstacles, goal)
so we can visually sanity-check the scene before training."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from robonav_env.dynamic_nav_env import DynamicNavEnv

env = DynamicNavEnv(seed=7)
obs, _ = env.reset(seed=7)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, env.arena_size)
ax.set_ylim(0, env.arena_size)
ax.set_aspect("equal")
ax.set_title("DynamicNavEnv -- initial scene")

# Arena boundary
ax.add_patch(patches.Rectangle((0, 0), env.arena_size, env.arena_size,
                                 fill=False, edgecolor="black", linewidth=1.5))

# LIDAR rays
angles = env.robot_heading + np.linspace(0, 2 * np.pi, env.n_rays, endpoint=False)
lidar = env._simulate_lidar() * env.lidar_range
for ang, dist in zip(angles, lidar):
    end = env.robot_pos + dist * np.array([np.cos(ang), np.sin(ang)])
    ax.plot([env.robot_pos[0], end[0]], [env.robot_pos[1], end[1]],
            color="tab:green", alpha=0.3, linewidth=1)

# Obstacles (moving, randomized)
for pos in env.obstacle_pos:
    ax.add_patch(patches.Circle(pos, env.obstacle_radius, color="tab:red", alpha=0.7))

# Robot
ax.add_patch(patches.Circle(env.robot_pos, env.robot_radius, color="tab:blue"))
heading_end = env.robot_pos + 0.6 * np.array(
    [np.cos(env.robot_heading), np.sin(env.robot_heading)]
)
ax.plot([env.robot_pos[0], heading_end[0]], [env.robot_pos[1], heading_end[1]],
        color="black", linewidth=2)

# Goal
ax.add_patch(patches.Circle(env.goal_pos, 0.4, color="tab:orange", alpha=0.5))
ax.annotate("goal", env.goal_pos, ha="center", va="center", fontsize=9)

ax.legend(handles=[
    patches.Patch(color="tab:blue", label="robot"),
    patches.Patch(color="tab:red", label="moving obstacle"),
    patches.Patch(color="tab:orange", label="goal"),
    patches.Patch(color="tab:green", label="LIDAR rays"),
], loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)

out_path = os.path.join(os.path.dirname(__file__), "..", "results", "env_snapshot.png")
plt.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"Saved: {out_path}")
