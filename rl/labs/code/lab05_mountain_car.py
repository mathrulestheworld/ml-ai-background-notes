"""Lab 5 reference solution: tile coding and linear control on Gymnasium's MountainCar-v0.

Run from this folder:  python lab05_mountain_car.py
It prints the results quoted on the lab page. Takes about three minutes on one core.
"""
import time

import gymnasium as gym
import numpy as np

t0 = time.time()
LOW, HIGH = np.array([-1.2, -0.07]), np.array([0.6, 0.07])


# ---------------------------------------------------------------- part 1: a tile coder
class TileCoder:
    """Tile coding of a box in R^k with n tilings of m tiles per dimension, offset asymmetrically by
    (1, 3, 5, ...) / n of a tile, and a hash table that assigns indices to tiles in order of first use."""

    def __init__(self, low, high, n_tilings=8, tiles=8, size=4096):
        self.low, self.scale = np.asarray(low), tiles / (np.asarray(high) - np.asarray(low))
        self.n = n_tilings
        self.offsets = (np.arange(n_tilings)[:, None] * (2 * np.arange(len(low)) + 1)[None, :] % n_tilings) / n_tilings
        self.size, self.table = size, {}

    def __call__(self, s, a=0):
        coords = np.floor((np.asarray(s) - self.low) * self.scale + self.offsets).astype(int)   # one row per tiling
        out = np.empty(self.n, int)
        for t, c in enumerate(coords):
            key = (t, a, *c)
            if key not in self.table:
                if len(self.table) >= self.size:                # full: fall back to hashing, with collisions
                    out[t] = hash(key) % self.size; continue
                self.table[key] = len(self.table)
            out[t] = self.table[key]
        return out


tc = TileCoder(LOW, HIGH)
a1, a2, a3 = tc([-0.5, 0.0]), tc([-0.49, 0.001]), tc([0.3, 0.05])
print("=== Part 1: tile coder (8 tilings of 8 x 8 tiles) ===")
print(f"  shared tiles: nearby states {len(set(a1) & set(a2))} of 8, distant states {len(set(a1) & set(a3))} of 8")


# ---------------------------------------------------------------- part 2: SARSA(lambda) and true online SARSA(lambda)
def sarsa_lambda(lam, alpha, seed, episodes=200, true_online=False, max_steps=5000):
    env = gym.make("MountainCar-v0", max_episode_steps=max_steps)
    rng = np.random.default_rng(seed)
    coder = TileCoder(LOW, HIGH, size=4096)
    w = np.zeros(4096)
    q = lambda s: np.array([w[coder(s, b)].sum() for b in range(3)])
    lengths = []
    s, _ = env.reset(seed=seed)
    for ep in range(episodes):
        if ep > 0:
            s, _ = env.reset()
        qs = q(s); a = int(rng.choice(np.flatnonzero(qs == qs.max())))
        f = coder(s, a); z = np.zeros(4096); q_old, n = 0.0, 0
        while True:
            s2, r, term, trunc, _ = env.step(a); n += 1
            q_sa = w[f].sum()
            if term:
                q_next, f2 = 0.0, None
            else:
                qs2 = q(s2); a2 = int(rng.choice(np.flatnonzero(qs2 == qs2.max())))  # greedy: q = 0 is optimistic
                f2 = coder(s2, a2); q_next = w[f2].sum()
            delta = r + q_next - q_sa                          # gamma = 1
            if true_online:                                    # dutch trace and the true online correction
                zf = z[f].sum()
                z *= lam; z[f] += 1 - alpha * lam * zf
                w += alpha * (delta + q_sa - q_old) * z
                w[f] -= alpha * (q_sa - q_old)
                q_old = q_next
            else:
                z *= lam; z[f] = 1.0                            # replacing traces
                w += alpha * delta * z
            if term or trunc:
                break
            s, a, f = s2, a2, f2
        lengths.append(n)
    return np.array(lengths)


print("\n=== Part 2: SARSA(lambda) with replacing traces, 8 tilings, alpha = 0.5/8, 5 runs of 200 episodes ===")
for lam in [0.0, 0.5, 0.9, 0.99]:
    L = np.array([sarsa_lambda(lam, 0.5 / 8, seed) for seed in range(5)])
    print(f"  lambda = {lam:4.2f}: steps per episode, episodes 1-20 {L[:, :20].mean():6.0f},"
          f" 21-100 {L[:, 20:100].mean():5.0f}, 101-200 {L[:, 100:].mean():5.0f}")

print("\n=== Part 3: replacing traces versus true online SARSA(lambda), lambda = 0.9, episodes 1-100 ===")
for alpha in [0.5, 1.0, 1.5]:
    rep = np.mean([sarsa_lambda(0.9, alpha / 8, seed, episodes=100).mean() for seed in range(5)])
    tol = np.mean([sarsa_lambda(0.9, alpha / 8, seed, episodes=100, true_online=True).mean() for seed in range(5)])
    print(f"  alpha = {alpha}/8: average steps per episode, replacing traces {rep:6.0f}, true online {tol:6.0f}")


# ---------------------------------------------------------------- part 4: LSPI from a batch of random transitions
def lspi(n_samples, seed, iterations=15, gamma=0.99):
    rng = np.random.default_rng(seed)
    env = gym.make("MountainCar-v0").unwrapped
    coder = TileCoder(LOW, HIGH, n_tilings=4, tiles=8, size=4 * 81 * 3)
    d = coder.size
    S = rng.uniform(LOW, [0.5, 0.07], size=(n_samples, 2))     # states drawn uniformly: a "generative model"
    A = rng.integers(3, size=n_samples)
    S2, R, T = np.zeros_like(S), -np.ones(n_samples), np.zeros(n_samples, bool)
    for i in range(n_samples):
        env.reset(); env.state = S[i].copy()
        s2, _, term, _, _ = env.step(int(A[i]))
        S2[i], T[i] = s2, term
    phi = lambda s, a: np.bincount(coder(s, a), minlength=d)
    Phi = np.array([phi(s, a) for s, a in zip(S, A)], float)
    Phi2_all = np.stack([np.array([phi(s, b) for s in S2], float) for b in range(3)])   # next-state features
    w = np.zeros(d)
    for it in range(iterations):
        nxt = (Phi2_all @ w).argmax(0)                          # the current greedy policy at the next states
        Phi2 = Phi2_all[nxt, np.arange(n_samples)] * ~T[:, None]
        A_mat = Phi.T @ (Phi - gamma * Phi2) + 1e-3 * np.eye(d) # LSTDQ
        w_new = np.linalg.solve(A_mat, Phi.T @ R)
        if np.abs(w_new - w).max() < 1e-3:
            w = w_new; break
        w = w_new
    return w, coder, it + 1


def greedy_steps(w, coder, episodes=20, seed=100, cap=1000):
    env = gym.make("MountainCar-v0", max_episode_steps=cap)
    steps = []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep); n = 0
        while True:
            a = int(np.argmax([w[coder(s, b)].sum() for b in range(3)]))
            s, r, term, trunc, _ = env.step(a); n += 1
            if term or trunc:
                break
        steps.append(n if term else np.nan)
    return np.array(steps)


print("\n=== Part 4: LSPI with 4 tilings, from uniformly sampled transitions, gamma = 0.99 ===")
for n in [2_000, 10_000, 50_000]:
    res = []
    for seed in range(3):
        w, coder, its = lspi(n, seed)
        st = greedy_steps(w, coder)
        res.append((np.isfinite(st).mean(), np.nanmean(st) if np.isfinite(st).any() else np.nan, its))
    res = np.array(res)
    print(f"  {n:6,} transitions: episodes reaching the goal {100 * res[:, 0].mean():5.1f}%,"
          f" steps when they do {np.nanmean(res[:, 1]) if np.isfinite(res[:, 1]).any() else np.nan:5.0f},"
          f" LSPI iterations {res[:, 2].mean():4.1f}")
print(f"\ntotal time {time.time() - t0:.0f} s")
