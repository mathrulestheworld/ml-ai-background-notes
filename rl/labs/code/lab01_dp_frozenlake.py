"""Lab 1 reference solution: dynamic programming on Gymnasium's FrozenLake and Taxi.

Run from this folder:  python lab01_dp_frozenlake.py
It prints the results quoted on the lab page. Takes about half a minute on one core.
"""
import time

import gymnasium as gym
import numpy as np


# ---------------------------------------------------------------- part 1: read the model
def tabular_model(env):
    """Convert Gymnasium's env.unwrapped.P into arrays P[s, a, s'], R[s, a] (expected reward), and a
    terminal mask. Terminal transitions go to an absorbing state with no further reward."""
    Pdict = env.unwrapped.P
    S, A = env.observation_space.n, env.action_space.n
    P, R = np.zeros((S, A, S)), np.zeros((S, A))
    done = np.zeros((S, A, S))                               # probability mass that terminates
    for s in range(S):
        for a in range(A):
            for prob, s2, r, terminated in Pdict[s][a]:
                R[s, a] += prob * r
                if terminated:
                    done[s, a, s2] += prob
                else:
                    P[s, a, s2] += prob
    return P, R, done.sum(2)


# ---------------------------------------------------------------- part 2: the algorithms
def policy_evaluation(P, R, pi, gamma):
    S = P.shape[0]
    P_pi, r_pi = P[np.arange(S), pi], R[np.arange(S), pi]
    return np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)


def policy_iteration(P, R, gamma):
    S = P.shape[0]
    pi, n = np.zeros(S, int), 0
    while True:
        n += 1
        v = policy_evaluation(P, R, pi, gamma)
        q = R + gamma * P @ v
        new = np.where(q[np.arange(S), pi] >= q.max(1) - 1e-12, pi, q.argmax(1))   # keep ties: no cycling
        if np.array_equal(new, pi):
            return v, pi, n
        pi = new


def value_iteration(P, R, gamma, tol=1e-10):
    v, n = np.zeros(P.shape[0]), 0
    while True:
        n += 1
        q = R + gamma * P @ v
        v_new = q.max(1)
        if np.abs(v_new - v).max() < tol * (1 - gamma) / (2 * gamma):   # the epsilon-optimal stopping rule
            return v_new, q.argmax(1), n
        v = v_new


def finite_horizon_success(P, R, pi, H):
    """Expected return of a stationary policy within H steps (backward induction with a fixed policy)."""
    S = P.shape[0]
    v = np.zeros(S)
    for _ in range(H):
        v = R[np.arange(S), pi] + P[np.arange(S), pi] @ v
    return v


def rollouts(env, pi, episodes, seed):
    total = 0.0
    for e in range(episodes):
        s, _ = env.reset(seed=seed + e)
        done = False
        while not done:
            s, r, terminated, truncated, _ = env.step(int(pi[s]))
            total += r
            done = terminated or truncated
    return total / episodes


arrows = np.array(["←", "↓", "→", "↑"])                      # FrozenLake actions: 0 left, 1 down, 2 right, 3 up

for name in ["4x4", "8x8"]:
    env = gym.make("FrozenLake-v1", map_name=name, is_slippery=True)
    P, R, _ = tabular_model(env)
    side = int(np.sqrt(P.shape[0]))
    desc = env.unwrapped.desc.astype(str)
    H = env.spec.max_episode_steps                           # Gymnasium truncates episodes after H steps
    print(f"\n=== FrozenLake {name} (slippery): {P.shape[0]} states, time limit {H} steps ===")
    for gamma in [0.9, 0.99, 0.9999]:
        v_pi, pi_pi, n_pi = policy_iteration(P, R, gamma)
        v_vi, pi_vi, n_vi = value_iteration(P, R, gamma)
        same = np.allclose(policy_evaluation(P, R, pi_vi, gamma), v_pi)
        ever = policy_evaluation(P, R, pi_pi, 1.0 - 1e-12)[0]  # undiscounted probability of ever reaching the goal
        in_time = finite_horizon_success(P, R, pi_pi, H)[0]
        print(f"gamma {gamma}: PI {n_pi:2d} evaluations, VI {n_vi:4d} sweeps, same value: {same}; "
              f"P(goal) eventually {ever:.4f}, within {H} steps {in_time:.4f}")
    # The best time-dependent policy for the time limit, by backward induction with gamma = 1
    v = np.zeros(P.shape[0])
    for _ in range(H):
        v = (R + P @ v).max(1)
    print(f"best time-dependent policy for the {H}-step limit (backward induction): P(goal) = {v[0]:.4f}")
    _, pi, _ = policy_iteration(P, R, 0.9999)
    mc = rollouts(env, pi, 2000, seed=0)
    print(f"check by simulation, gamma = 0.9999 policy, 2,000 episodes in Gymnasium: {mc:.4f}")
    grid = np.where(np.isin(desc, ["H", "G"]), desc, arrows[pi.reshape(side, side)])
    print("that policy (H hole, G goal):")
    for row in grid:
        print("  " + " ".join(row))

# ---------------------------------------------------------------- part 3: many random maps
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

rng = np.random.default_rng(0)
succ = []
for k in range(200):
    env = gym.make("FrozenLake-v1", desc=generate_random_map(size=8, p=0.8, seed=int(rng.integers(1 << 30))),
                   is_slippery=True)
    P, R, _ = tabular_model(env)
    _, pi, _ = policy_iteration(P, R, 0.9999)
    succ.append(policy_evaluation(P, R, pi, 1.0 - 1e-12)[0])
succ = np.array(succ)
print(f"\n200 random 8x8 maps: optimal success probability median {np.median(succ):.3f}, "
      f"fraction above 0.99: {np.mean(succ > 0.99):.2f}, below 0.5: {np.mean(succ < 0.5):.2f}")

# ---------------------------------------------------------------- part 4: Taxi
env = gym.make("Taxi-v3")
P, R, _ = tabular_model(env)
t0 = time.perf_counter(); v, pi, n_vi = value_iteration(P, R, 0.99); t_vi = time.perf_counter() - t0
t0 = time.perf_counter(); v2, pi2, n_pi = policy_iteration(P, R, 0.99); t_pi = time.perf_counter() - t0
print(f"\nTaxi-v3: {P.shape[0]} states, {P.shape[1]} actions; VI {n_vi} sweeps, PI {n_pi} evaluations; "
      f"values agree: {np.allclose(v, v2, atol=1e-6)}")
print(f"average undiscounted return of the optimal policy over 1,000 episodes: {rollouts(env, pi, 1000, seed=1):.2f}")
