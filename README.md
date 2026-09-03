# DynamicNav: RL for social/dynamic robot navigation with a domain-randomization ablation

A reinforcement-learning robot that learns to navigate around **moving**
obstacles (not just static ones), trained with PPO, with a controlled
experiment measuring how much domain randomization improves generalization
to obstacle dynamics the policy never saw during training.

## Why dynamic navigation, not static

Most introductory robot-navigation RL projects use static obstacles, which is
a largely solved problem. Navigating around *moving* agents with unknown,
varying speed and direction is the actual open problem in social/dynamic
robot navigation -- the subfield behind work like Stanford's ILIAD lab and
Berkeley's InterACT lab. This project is a small, honest version of that
problem, not a toy.

## The experiment

We train two PPO policies that differ in exactly one thing: how much the
obstacle speed varies during training.

| Condition | Obstacle speed range during training | Hypothesis |
|---|---|---|
| `narrow` | 0.3 - 0.4 (arena-lengths/sec) | Learns a policy tuned to one narrow dynamics regime |
| `wide`   | 0.1 - 0.9 (arena-lengths/sec) | Forced to learn a more general avoidance strategy |

Both are then evaluated on a **held-out** obstacle speed range (0.5 - 0.8)
that neither policy trained on directly. The held-out range deliberately
overlaps partially with both training distributions rather than being wildly
out-of-distribution -- the question is realistic generalization, not an
impossible edge case.

**Metric**: success rate (episodes where the robot reaches the goal without
colliding) on the held-out range. The gap between the `wide` and `narrow`
policies' held-out success rates is the reported generalization gap.

## What's randomized on every episode (domain randomization)

- Obstacle starting positions
- Obstacle velocity direction and speed (drawn from the condition's range)
- Robot friction / effective wheel response (simulates mass and surface variation)
- LIDAR sensor noise

## Environment design

- **Robot**: differential-drive (two independently driven wheels), state = (x, y, heading)
- **Sensing**: 16-ray simulated LIDAR, normalized distance readings, with Gaussian noise
- **Action space**: continuous, `[left_wheel_speed, right_wheel_speed]` in `[-1, 1]`
- **Reward**: dense shaping (progress toward goal, obstacle-proximity penalty,
  small time penalty) plus sparse terminal reward (+50 goal / -20 collision)
- Full implementation: `robonav_env/dynamic_nav_env.py`

A random-action policy essentially never reaches the goal (verified in
`scripts/sanity_check.py`), confirming the task has real headroom for
learning rather than being trivially easy.

## Repo structure

```
robonav_env/
  dynamic_nav_env.py     -- the Gymnasium environment
scripts/
  sanity_check.py        -- verifies the env is well-formed (no training needed)
  visualize_env.py        -- renders a static snapshot of the scene
  train.py                -- trains the narrow/wide PPO policies
  evaluate.py              -- runs the held-out ablation and produces the comparison plot
results/                  -- trained models, plots, snapshots land here
```

## Running it

```bash
pip install -r requirements.txt

# 1. Sanity check + visualize the environment (fast, no training)
python scripts/sanity_check.py
python scripts/visualize_env.py

# 2. Train both conditions (the slow part -- ~15-30 min each on CPU,
#    much faster with a GPU; Colab's free GPU works well)
python scripts/train.py --condition narrow
python scripts/train.py --condition wide

# 3. Run the ablation evaluation and generate the comparison plot
python scripts/evaluate.py
```

## Status

- [x] Environment implemented and verified (random-policy baseline confirms
      task difficulty is well-calibrated)
- [x] Training and evaluation scripts written
- [ ] Both policies trained (run locally/Colab -- see note on PyTorch/CUDA below)
- [ ] Ablation results and write-up

## A note on why training didn't happen in the same sandbox as the environment

PyTorch's default pip install pulls the full CUDA toolkit as a dependency,
which is unnecessary on a CPU-only machine and can be several GB. That's why
this repo separates environment development (numpy/matplotlib only, verified
above) from training (requires `torch` + `stable-baselines3`, best run
locally or on Colab where a GPU is available for free).
