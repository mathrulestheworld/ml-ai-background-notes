"""Lab 2 reference solution: a bandit library, stress tests, a contextual bandit built from a classification
dataset, and off-policy evaluation of logged data.

Run from this folder:  python lab02_bandits.py      (about a minute on one core)
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# ---------------------------------------------------------------- part 1: a small bandit library
class Agent:
    def __init__(self, k, rng):
        self.k, self.rng = k, rng
        self.N, self.S = np.zeros(k), np.zeros(k)

    def select(self, t):
        raise NotImplementedError

    def update(self, a, r):
        self.N[a] += 1
        self.S[a] += r


class EpsGreedy(Agent):
    def __init__(self, k, rng, eps=0.1):
        super().__init__(k, rng); self.eps = eps

    def select(self, t):
        if self.N.min() == 0:
            return int(self.N.argmin())
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.k))
        return int((self.S / self.N).argmax())


class UCB1(Agent):
    def select(self, t):
        if self.N.min() == 0:
            return int(self.N.argmin())
        return int((self.S / self.N + np.sqrt(2 * np.log(t) / self.N)).argmax())


class KLUCB(Agent):
    def select(self, t):
        if self.N.min() == 0:
            return int(self.N.argmin())
        m = self.S / self.N
        lo, hi = m.copy(), np.ones(self.k)
        for _ in range(20):
            mid = (lo + hi) / 2
            p, q = np.clip(m, 1e-12, 1 - 1e-12), np.clip(mid, 1e-12, 1 - 1e-12)
            kl = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
            ok = self.N * kl <= np.log(t)
            lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
        return int(lo.argmax())


class Thompson(Agent):
    def select(self, t):
        return int(self.rng.beta(1 + self.S, 1 + self.N - self.S).argmax())


class Exp3(Agent):
    def __init__(self, k, rng, T=10_000):
        super().__init__(k, rng)
        self.eta, self.L = np.sqrt(2 * np.log(k) / (T * k)), np.zeros(k)

    def select(self, t):
        w = np.exp(-self.eta * (self.L - self.L.min()))
        self.p = w / w.sum()
        return int(self.rng.choice(self.k, p=self.p))

    def update(self, a, r):
        super().update(a, r)
        self.L[a] += (1 - r) / self.p[a]


class SlidingWindowUCB(Agent):
    """UCB on the last `window` pulls only: forgets old rewards in nonstationary problems."""
    def __init__(self, k, rng, window=500):
        super().__init__(k, rng); self.window, self.hist = window, []

    def update(self, a, r):
        self.hist.append((a, r))
        if len(self.hist) > self.window:
            a0, r0 = self.hist.pop(0); self.N[a0] -= 1; self.S[a0] -= r0
        super().update(a, r)

    def select(self, t):
        if self.N.min() == 0:
            return int(self.N.argmin())
        return int((self.S / self.N + np.sqrt(2 * np.log(min(t, self.window)) / self.N)).argmax())


class DiscountedThompson(Agent):
    """Thompson sampling with pseudo-counts that decay by a factor gamma every step."""
    def __init__(self, k, rng, gamma=0.99):
        super().__init__(k, rng); self.g = gamma

    def update(self, a, r):
        self.N *= self.g; self.S *= self.g
        super().update(a, r)

    def select(self, t):
        return int(self.rng.beta(1 + self.S, 1 + self.N - self.S).argmax())


def run(agent_cls, means_fn, T, seed, batch=1, **kw):
    """means_fn(t) gives the arms' success probabilities at step t. Updates are applied every `batch` steps."""
    rng = np.random.default_rng(seed)
    k = len(means_fn(0))
    agent = agent_cls(k, rng, **kw)
    regret, pending = 0.0, []
    for t in range(1, T + 1):
        mu = means_fn(t)
        a = agent.select(t)
        r = float(rng.random() < mu[a])
        regret += mu.max() - mu[a]
        pending.append((a, r))
        if len(pending) == batch:
            for a_, r_ in pending:
                agent.update(a_, r_)
            pending = []
    return regret


def summary(values):
    v = np.array(values)
    return f"{v.mean():7.1f} +- {1.96 * v.std(ddof=1) / np.sqrt(len(v)):5.1f}"


T, seeds = 10_000, range(20)
mu = np.array([0.5, 0.45, 0.4, 0.3, 0.2])
print("=== Part 1: stationary Bernoulli arms, regret after 10,000 pulls (mean +- 95% CI over 20 seeds) ===")
for cls in [EpsGreedy, UCB1, KLUCB, Thompson, Exp3]:
    print(f"  {cls.__name__:18s} {summary([run(cls, lambda t: mu, T, s) for s in seeds])}")

print("\n=== Part 2a: the best arm changes at step 5,000 (arms 0.6/0.4 swap) ===")
switch = lambda t: np.array([0.6, 0.4]) if t <= T // 2 else np.array([0.4, 0.6])
for cls in [UCB1, Thompson, SlidingWindowUCB, DiscountedThompson, Exp3]:
    print(f"  {cls.__name__:18s} {summary([run(cls, switch, T, s) for s in seeds])}")

print("\n=== Part 2b: delayed feedback, updates in batches of 200 pulls ===")
for cls in [UCB1, Thompson]:
    print(f"  {cls.__name__:18s} batch 1: {summary([run(cls, lambda t: mu, T, s) for s in seeds])}   "
          f"batch 200: {summary([run(cls, lambda t: mu, T, s, batch=200) for s in seeds])}")

# ---------------------------------------------------------------- part 3: a contextual bandit from a dataset
X, y = load_digits(return_X_y=True)
X = PCA(20, random_state=0).fit_transform((X - X.mean(0)) / (X.std(0) + 1e-9))
X = np.c_[X / np.abs(X).max(), np.ones(len(X))]             # 21 features including a bias
d, k = X.shape[1], 10
print(f"\n=== Part 3: digits as a 10-action contextual bandit (reward 1 for the right label) ===")
W_full = np.linalg.solve(X.T @ X + np.eye(d), X.T @ np.eye(k)[y])   # the bandits' model, with every label revealed
sup = cross_val_score(LogisticRegression(max_iter=5000, C=100), X, y, cv=5).mean()
print(f"  full-information references: per-action ridge regression on all labels {np.mean((X @ W_full).argmax(1) == y):.3f}, "
      f"logistic regression (5-fold CV) {sup:.3f}")


def contextual(policy, T=5000, seed=0, alpha=0.5, eps=0.05):
    rng = np.random.default_rng(seed)
    A_inv = np.tile(np.eye(d), (k, 1, 1)); b = np.zeros((k, d))
    S, N = np.zeros(k), np.zeros(k)
    rewards = np.zeros(T)
    for t in range(T):
        i = rng.integers(len(X)); x = X[i]
        th = np.einsum("kij,kj->ki", A_inv, b)
        if policy == "LinUCB":
            a = int((th @ x + alpha * np.sqrt(np.einsum("i,kij,j->k", x, A_inv, x))).argmax())
        elif policy == "linear Thompson":
            L = np.linalg.cholesky(A_inv)
            a = int(((th + 0.3 * np.einsum("kij,kj->ki", L, rng.normal(size=(k, d)))) @ x).argmax())
        elif policy == "epsilon-greedy":
            a = int(rng.integers(k)) if rng.random() < eps else int((th @ x).argmax())
        elif policy == "greedy":
            a = int((th @ x + 1e-9 * rng.random(k)).argmax())
        else:                                                # context-free Thompson sampling
            a = int(rng.beta(1 + S, 1 + N - S).argmax())
        r = float(a == y[i])
        rewards[t] = r
        Ax = A_inv[a] @ x
        A_inv[a] -= np.outer(Ax, Ax) / (1 + x @ Ax); b[a] += r * x
        S[a] += r; N[a] += 1
    return rewards


for pol in ["LinUCB", "linear Thompson", "epsilon-greedy", "greedy", "context-free Thompson"]:
    r = contextual(pol)
    print(f"  {pol:22s} accuracy in rounds 1-1,000: {r[:1000].mean():.3f}   rounds 4,001-5,000: {r[4000:].mean():.3f}")

# ---------------------------------------------------------------- part 4: off-policy evaluation
rng = np.random.default_rng(1)
n = 3000
idx = rng.integers(len(X), size=2 * n)
a_log = rng.integers(k, size=2 * n)                          # uniform logging policy: propensity 1/10
r_log = (a_log == y[idx]).astype(float)
# Learn a target policy from the first half of the log: one ridge regression of reward per action.
tr = slice(0, n)
W = np.zeros((k, d))
for a in range(k):
    m = a_log[tr] == a
    Xa = X[idx[tr]][m]
    W[a] = np.linalg.solve(Xa.T @ Xa + 1.0 * np.eye(d), Xa.T @ r_log[tr][m])
target = lambda Z: (Z @ W.T).argmax(1)
true_value = np.mean(target(X) == y)
# Evaluate it on the second half of the log.
te = slice(n, 2 * n)
Xe, ae, re = X[idx[te]], a_log[te], r_log[te]
w = (target(Xe) == ae) / 0.1
q_hat = Xe @ W.T
ips = np.mean(w * re)
snips = np.sum(w * re) / np.sum(w)
dm = np.mean(q_hat[np.arange(n), target(Xe)])
dr = np.mean(q_hat[np.arange(n), target(Xe)] + w * (re - q_hat[np.arange(n), ae]))
print(f"\n=== Part 4: evaluating a policy learned from 3,000 uniformly logged rounds ===")
print(f"  true accuracy of the learned policy on all 1,797 digits: {true_value:.3f}")
print(f"  estimates from 3,000 held-out logged rounds: DM {dm:.3f}, IPS {ips:.3f}, SNIPS {snips:.3f}, DR {dr:.3f}")
