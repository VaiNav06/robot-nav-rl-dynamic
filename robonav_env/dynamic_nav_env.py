"""
DynamicNavEnv: a 2D differential-drive robot navigation environment with
moving obstacles, LIDAR-style sensing, and domain randomization.

Design summary
---------------
- Robot: differential-drive (two independently-driven wheels), state =
  (x, y, heading). Action = (left_wheel_speed, right_wheel_speed) in [-1, 1].
- Sensing: a fan of `N_RAYS` simulated LIDAR rays around the robot, each
  returning normalized distance-to-nearest-obstacle-or-wall.
- Obstacles: circular, moving at randomized constant velocity, bouncing off
  the arena walls. Number, radius, and speed range are configurable so we
  can run a controlled train/test split for the domain-randomization
  ablation (see docs/ABLATION.md).
- Reward: dense shaping (progress toward goal, obstacle-proximity penalty,
  time penalty) + sparse terminal bonus (goal reached) / penalty (collision).
- Domain randomization: on every `reset()`, obstacle positions, obstacle
  velocities (drawn from a configurable range), robot mass/friction
  (affecting effective turning response), and LIDAR sensor noise are all
  re-randomized.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DynamicNavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        arena_size: float = 10.0,
        n_obstacles: int = 4,
        obstacle_radius: float = 0.5,
        robot_radius: float = 0.3,
        n_rays: int = 16,
        lidar_range: float = 4.0,
        max_steps: int = 300,
        obstacle_speed_range: tuple = (0.3, 0.8),
        friction_range: tuple = (0.85, 1.0),
        sensor_noise_std: float = 0.02,
        seed: int | None = None,
    ):
        super().__init__()
        self.arena_size = arena_size
        self.n_obstacles = n_obstacles
        self.obstacle_radius = obstacle_radius
        self.robot_radius = robot_radius
        self.n_rays = n_rays
        self.lidar_range = lidar_range
        self.max_steps = max_steps
        self.obstacle_speed_range = obstacle_speed_range
        self.friction_range = friction_range
        self.sensor_noise_std = sensor_noise_std

        # Observation: [n_rays lidar readings] + [dx_to_goal, dy_to_goal, heading, speed_scale]
        obs_dim = n_rays + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Action: (left_wheel, right_wheel) each in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self.wheel_base = 0.4     # distance between wheels
        self.max_wheel_speed = 1.0
        self.dt = 0.1

        self.reset(seed=seed)

    # ------------------------------------------------------------------ #
    # Core Gym API
    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        margin = 1.0
        # Robot spawn: fixed corner-ish start so goal-directed progress is meaningful.
        self.robot_pos = np.array([margin, margin], dtype=np.float32)
        self.robot_heading = self._rng.uniform(-np.pi, np.pi)

        # Goal in the opposite corner region.
        self.goal_pos = np.array(
            [self.arena_size - margin, self.arena_size - margin], dtype=np.float32
        )

        # --- Domain randomization ---
        # Robot friction: scales effective wheel response (simulates mass/surface variation).
        self.friction = self._rng.uniform(*self.friction_range)

        # Randomized moving obstacles: position, velocity direction, and speed.
        self.obstacle_pos = self._rng.uniform(
            margin, self.arena_size - margin, size=(self.n_obstacles, 2)
        ).astype(np.float32)
        speeds = self._rng.uniform(*self.obstacle_speed_range, size=self.n_obstacles)
        angles = self._rng.uniform(0, 2 * np.pi, size=self.n_obstacles)
        self.obstacle_vel = np.stack(
            [speeds * np.cos(angles), speeds * np.sin(angles)], axis=1
        ).astype(np.float32)

        self.steps = 0
        self.prev_dist_to_goal = np.linalg.norm(self.goal_pos - self.robot_pos)

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        left, right = action * self.max_wheel_speed * self.friction

        # Differential-drive kinematics
        v = (left + right) / 2.0
        omega = (right - left) / self.wheel_base

        self.robot_heading += omega * self.dt
        self.robot_pos += v * self.dt * np.array(
            [np.cos(self.robot_heading), np.sin(self.robot_heading)], dtype=np.float32
        )
        self.robot_pos = np.clip(self.robot_pos, 0, self.arena_size)

        # Move obstacles, bounce off walls
        self.obstacle_pos += self.obstacle_vel * self.dt
        for i in range(self.n_obstacles):
            for d in range(2):
                if self.obstacle_pos[i, d] < self.obstacle_radius or \
                   self.obstacle_pos[i, d] > self.arena_size - self.obstacle_radius:
                    self.obstacle_vel[i, d] *= -1
                    self.obstacle_pos[i, d] = np.clip(
                        self.obstacle_pos[i, d], self.obstacle_radius,
                        self.arena_size - self.obstacle_radius
                    )

        self.steps += 1
        dist_to_goal = np.linalg.norm(self.goal_pos - self.robot_pos)
        min_obstacle_dist = np.min(
            np.linalg.norm(self.obstacle_pos - self.robot_pos, axis=1)
        ) - self.obstacle_radius - self.robot_radius

        collided = min_obstacle_dist < 0
        reached_goal = dist_to_goal < 0.4
        timed_out = self.steps >= self.max_steps

        # --- Reward shaping ---
        progress = self.prev_dist_to_goal - dist_to_goal
        reward = 2.0 * progress            # progress toward goal
        reward -= 0.01                     # small time penalty, encourages efficiency
        if min_obstacle_dist < 1.0:
            reward -= (1.0 - min_obstacle_dist) * 0.5   # proximity penalty near obstacles
        if collided:
            reward -= 20.0
        if reached_goal:
            reward += 50.0

        self.prev_dist_to_goal = dist_to_goal
        terminated = bool(collided or reached_goal)
        truncated = bool(timed_out)

        info = {"collided": collided, "reached_goal": reached_goal}
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Sensing
    # ------------------------------------------------------------------ #
    def _get_obs(self):
        lidar = self._simulate_lidar()
        to_goal = self.goal_pos - self.robot_pos
        obs = np.concatenate([
            lidar,
            [to_goal[0] / self.arena_size, to_goal[1] / self.arena_size,
             self.robot_heading / np.pi, self.friction],
        ]).astype(np.float32)
        return obs

    def _simulate_lidar(self):
        angles = self.robot_heading + np.linspace(0, 2 * np.pi, self.n_rays, endpoint=False)
        readings = np.full(self.n_rays, self.lidar_range, dtype=np.float32)

        for i, ang in enumerate(angles):
            ray_dir = np.array([np.cos(ang), np.sin(ang)])
            min_hit = self.lidar_range

            # Distance to obstacles along this ray (circle intersection).
            for obs_p in self.obstacle_pos:
                rel = obs_p - self.robot_pos
                proj = np.dot(rel, ray_dir)
                if proj <= 0:
                    continue
                closest_pt = self.robot_pos + proj * ray_dir
                perp_dist = np.linalg.norm(obs_p - closest_pt)
                if perp_dist <= self.obstacle_radius:
                    chord = np.sqrt(max(self.obstacle_radius ** 2 - perp_dist ** 2, 0))
                    hit_dist = proj - chord
                    if 0 < hit_dist < min_hit:
                        min_hit = hit_dist

            # Distance to arena walls along this ray.
            for axis in range(2):
                if abs(ray_dir[axis]) > 1e-6:
                    if ray_dir[axis] > 0:
                        wall_dist = (self.arena_size - self.robot_pos[axis]) / ray_dir[axis]
                    else:
                        wall_dist = (0 - self.robot_pos[axis]) / ray_dir[axis]
                    if 0 < wall_dist < min_hit:
                        min_hit = wall_dist

            readings[i] = min_hit

        # Sensor noise (domain randomization on perception, not just dynamics).
        readings += self._rng.normal(0, self.sensor_noise_std, size=readings.shape)
        readings = np.clip(readings, 0, self.lidar_range)
        return readings / self.lidar_range  # normalize to [0, 1]
