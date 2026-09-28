"""Lab 6 reference solution: policy gradient and actor-critic methods on CartPole.

Run from this folder:  python lab06_policy_gradient.py
It prints the results quoted on the lab page. Takes about two and a half minutes on one core.
"""
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")                # small matrices: one thread is faster
import gymnasium as gym
import numpy as np

t0 = time.time()
GAMMA = 0.99


# ---------------------------------------------------------------- a vectorized CartPole
class VecCartPole:
    """n independent copies of Gymnasium's CartPole-v1 (Euler integration, 500-step limit), stepped together.
    Each copy resets itself when its episode ends."""
    g, m_cart, m_pole, half_len, force, tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    x_max, th_max, limit = 2.4, 12 * 2 * np.pi / 360, 500

    def __init__(self, n, rng):
        self.n, self.rng = n, rng
        self.state = rng.uniform(-0.05, 0.05, (n, 4)); self.t = np.zeros(n, int)

    @classmethod
    def dynamics(cls, state, action):
        x, x_dot, th, th_dot = state.T
        f = np.where(action == 1, cls.force, -cls.force)
        total, pml = cls.m_cart + cls.m_pole, cls.m_pole * cls.half_len
        cos, sin = np.cos(th), np.sin(th)
        temp = (f + pml * th_dot ** 2 * sin) / total
        th_acc = (cls.g * sin - cos * temp) / (cls.half_len * (4 / 3 - cls.m_pole * cos ** 2 / total))
        x_acc = temp - pml * th_acc * cos / total
        return np.stack([x + cls.tau * x_dot, x_dot + cls.tau * x_acc, th + cls.tau * th_dot, th_dot + cls.tau * th_acc], 1)

    @classmethod
    def failed(cls, state):
        return (np.abs(state[:, 0]) > cls.x_max) | (np.abs(state[:, 2]) > cls.th_max)

    def step(self, action):
        """Returns the next states (after any resets), the rewards (1 per step), and two masks: terminated (the pole
        fell or the cart left the track) and truncated (500 steps). self.next_true keeps the true next states."""
        nxt = self.dynamics(self.state, action); self.t += 1
        term = self.failed(nxt)
        trunc = ~term & (self.t >= self.limit)
        done = term | trunc
        self.next_true = nxt.copy()
        nxt[done] = self.rng.uniform(-0.05, 0.05, (done.sum(), 4)); self.t[done] = 0
        self.state = nxt
        return nxt, np.ones(self.n), term, trunc


SCALE = np.array([2.4, 2.0, 0.21, 2.0])                         # rough ranges of the four state variables


def sigmoid(z):
    return 0.5 * (1 + np.tanh(0.5 * z))


def features(s, scale=SCALE):                                   # policy features: scaled state and a constant
    return np.concatenate([s / scale, np.ones(s.shape[:-1] + (1,))], -1)


C = np.array(np.meshgrid(*[np.arange(3)] * 4, indexing="ij")).reshape(4, -1).T      # Fourier basis of order 2: 81 features
C_NORM = np.maximum(np.linalg.norm(C, axis=1), 1)


def fourier(s):
    """Fourier features of the state scaled to [0, 1] from [-2.4, 2.4], [-3, 3], [-0.21, 0.21], [-3.5, 3.5]."""
    u = np.clip((s + [2.4, 3.0, 0.21, 3.5]) / [4.8, 6.0, 0.42, 7.0], 0, 1)
    return np.cos(np.pi * u @ C.T)


def quadratic(s):                                               # batch critics: 1, u_i, u_i u_j with u the scaled state
    u = s / SCALE; i, j = np.triu_indices(4)
    return np.concatenate([np.ones(s.shape[:-1] + (1,)), u, u[..., i] * u[..., j]], -1)


def windows(L, width):
    return "".join(f"{L[:, j:j + width].mean():8.1f}" for j in range(0, L.shape[1], width))


# ---------------------------------------------------------------- part 1: REINFORCE and its estimators
print("=== Part 1: a vectorized CartPole, and REINFORCE with a linear softmax policy ===")
env = gym.make("CartPole-v1"); rng = np.random.default_rng(0); worst = 0.0
for trial in range(20):
    env.reset(seed=trial); s = env.unwrapped.state.copy()
    for t in range(500):
        a = int(rng.integers(2))
        _, _, term, trunc, _ = env.step(a)
        s = VecCartPole.dynamics(s[None], np.array([a]))[0]
        worst = max(worst, np.abs(s - env.unwrapped.state).max())
        if term or trunc:
            break
print(f"  largest state difference from Gymnasium over 20 random episodes: {worst:.1e}")


def reinforce(configs, seeds=10, episodes=1000, alpha_w=0.01):
    """Per-episode REINFORCE; configs are (variant, alpha) pairs, each run with `seeds` independent policies, all
    simulated together. The variants weight the score of each step t by
      'whole':      the discounted return of the whole episode, G_0
      'disc-to-go': gamma^t G_t, the discounted return from step t, discounted back to the start
      'to-go':      G_t, without the factor gamma^t, as most implementations do
      'baseline':   G_t - v(S_t), with v learned by gradient Monte Carlo on the Fourier features.
    The first two estimate the gradient of the discounted value of the start state; the last two estimate the
    direction of exercise 13.7. Returns episode lengths, configs x seeds x episodes."""
    rng = np.random.default_rng(1)
    variant = np.repeat([v for v, _ in configs], seeds); alpha = np.repeat([a for _, a in configs], seeds)
    n = len(alpha); idx = np.arange(n)
    env = VecCartPole(n, rng)
    theta, w = np.zeros((n, 5)), np.zeros((n, len(C)))
    X, A, S = np.zeros((n, 500, 5)), np.zeros((n, 500)), np.zeros((n, 500, 4))
    lengths = [[] for _ in range(n)]
    s = env.state
    while min(len(l) for l in lengths) < episodes:
        x, t = features(s), env.t.copy()
        a = (rng.random(n) < sigmoid((x * theta).sum(1))).astype(int)     # push right with probability sigmoid(theta.x)
        X[idx, t], A[idx, t], S[idx, t] = x, a, s
        s, r, term, trunc = env.step(a)
        for i in np.flatnonzero(term | trunc):
            T = t[i] + 1; lengths[i].append(T)
            if len(lengths[i]) > episodes:
                continue
            disc = GAMMA ** np.arange(T)
            G = np.cumsum(disc[::-1])[::-1] / disc               # G_t for a reward of 1 per step
            score = (A[i, :T] - sigmoid(X[i, :T] @ theta[i]))[:, None] * X[i, :T]   # grad log pi(A_t | S_t)
            if variant[i] == "whole":
                weight = np.full(T, G[0])
            elif variant[i] == "disc-to-go":
                weight = disc * G
            elif variant[i] == "to-go":
                weight = G
            else:
                F = fourier(S[i, :T]); weight = G - F @ w[i]
                w[i] += alpha_w / T * (weight @ F) / C_NORM         # gradient Monte Carlo for the baseline
            theta[i] += alpha[i] * weight @ score
    return np.array([l[:episodes] for l in lengths]).reshape(len(configs), seeds, episodes)


configs1 = [(v, a) for v in ["whole", "disc-to-go", "to-go", "baseline"] for a in [3e-4, 1e-3, 3e-3]]
L1 = reinforce(configs1)
print("  average episode length (the return, at most 500), 10 seeds; solved: last 100 episodes average over 475")
print("  " + " " * 29 + "episodes 1-200 201-400 401-600 601-800 801-1000  solved")
for (v, a), L in zip(configs1, L1):
    print(f"  {v:>10s}, alpha = {a:.0e}      {windows(L, 200)}   {(L[:, -100:].mean(1) > 475).sum():3d} of 10")
print(f"  (time so far {time.time() - t0:.0f} s)")


# ---------------------------------------------------------------- part 2: actor-critic
def actor_critic(configs, seeds=10, episodes=300):
    """Online actor-critic with accumulating traces (lam = 0: one-step), linear softmax actor, Fourier critic.
    configs: (lam, alpha_theta, alpha_w, bootstrap_at_time_limit)."""
    rng = np.random.default_rng(2)
    rep = lambda k: np.repeat([c[k] for c in configs], seeds)
    lam, a_th, a_w, boot = rep(0)[:, None], rep(1)[:, None], rep(2)[:, None] / C_NORM, rep(3).astype(bool)
    n = len(lam); env = VecCartPole(n, rng)
    theta, w = np.zeros((n, 5)), np.zeros((n, len(C))); z_th, z_w = np.zeros_like(theta), np.zeros_like(w)
    lengths = [[] for _ in range(n)]; s = env.state; f = fourier(s)
    while min(len(l) for l in lengths) < episodes:
        x = features(s); p = sigmoid((x * theta).sum(1))
        a = (rng.random(n) < p).astype(int); t = env.t.copy()
        s2, r, term, trunc = env.step(a)
        f2 = fourier(env.next_true)                              # the true next state, before any reset
        end = term | (trunc & ~boot)                             # where the target does not bootstrap
        delta = r + GAMMA * np.where(end, 0, (f2 * w).sum(1)) - (f * w).sum(1)
        z_th = GAMMA * lam * z_th + (a - p)[:, None] * x
        z_w = GAMMA * lam * z_w + f
        theta += a_th * delta[:, None] * z_th
        w += a_w * delta[:, None] * z_w
        done = term | trunc
        for i in np.flatnonzero(done):
            lengths[i].append(t[i] + 1)
        z_th[done] = 0; z_w[done] = 0
        s = s2; f = np.where(done[:, None], fourier(s2), f2)
    return np.array([l[:episodes] for l in lengths]).reshape(len(configs), seeds, episodes)


print("\n=== Part 2: online actor-critic (linear softmax actor, Fourier critic of order 2) ===")
configs2 = [(0, 0.1, 0.03, True), (0, 0.1, 0.03, False), (0.9, 0.03, 0.003, True)]
names = ["one-step actor-critic", "  same, time limit treated as termination", "actor-critic, lambda = 0.9"]
L2 = actor_critic(configs2)
print("  average episode length, 10 seeds" + " " * 13 + "episodes 1-50  51-100 101-150 151-200 201-250 251-300  solved")
for name, L in zip(names, L2):
    print(f"  {name:43s}{windows(L, 50)}   {(L[:, -100:].mean(1) > 475).sum():3d} of 10")
k = configs1.index(("baseline", 1e-3))
print(f"  {'for comparison, REINFORCE with baseline':43s}{windows(L1[k][:, :300], 50)}")
print(f"  (time so far {time.time() - t0:.0f} s)")


# ---------------------------------------------------------------- part 3: the variance of gradient estimators
def rollouts(theta, rng, scale=SCALE, share=1):
    """One episode for each row of theta, all in parallel. With share = k, the initial states and random numbers
    are shared by k consecutive blocks of rows (common random numbers). Returns the states ((T+1) x n x 4), the
    actions, and a mask of the steps taken (both T x n)."""
    n = len(theta); m = n // share
    s = np.tile(rng.uniform(-0.05, 0.05, (m, 4)), (share, 1)); alive = np.ones(n, bool)
    S, A, M = [s], [], []
    for t in range(500):
        p = sigmoid((features(s, scale) * theta).sum(1))
        a = (np.tile(rng.random(m), share) < p).astype(int)
        A.append(a); M.append(alive.copy())
        s = VecCartPole.dynamics(s, a); S.append(s)
        alive &= ~VecCartPole.failed(s)
        if not alive.any():
            break
    return np.array(S), np.array(A), np.array(M)


def returns_to_go(M):
    G = np.zeros(M.shape); run = np.zeros(M.shape[1])
    for t in range(len(M) - 1, -1, -1):
        run = np.where(M[t], 1 + GAMMA * run, 0); G[t] = run
    return G


def gae(V, M, lams):
    """Generalized advantage estimates for each lambda in lams: sum_k (gamma lam)^k delta_{t+k}, with the TD errors
    delta_t = 1 + gamma V(S_{t+1}) - V(S_t) and V = 0 after the last step. Returns len(lams) x T x n."""
    nxt = np.vstack([M[1:], np.zeros((1, M.shape[1]), bool)])
    delta = np.where(M, 1 + GAMMA * np.where(nxt, V[1:], 0) - V[:-1], 0)
    adv = np.zeros((len(lams), M.shape[1])); out = np.zeros((len(lams),) + M.shape)
    for t in range(len(M) - 1, -1, -1):
        adv = np.where(M[t], delta[t] + GAMMA * lams[:, None] * np.where(nxt[t], adv, 0), 0); out[:, t] = adv
    return out


def fit_critic(S, M, G, ridge=1e-3):
    Q = quadratic(S[:-1])[M]
    return np.linalg.solve(Q.T @ Q + ridge * np.eye(Q.shape[1]), Q.T @ G[M])


print("\n=== Part 3: gradient estimators at a fixed policy (average episode length about 110), batches of 10 episodes ===")
rng = np.random.default_rng(4)
theta_fix = np.array([0, 0, 1.5, 1.5, 0.0])                    # push toward the side the pole is falling to
theta_old = np.array([0, 0, 0.75, 0.75, 0.0])                   # a worse policy, for a stale critic
critics = {}
for name, th in [("fitted", theta_fix), ("stale", theta_old)]:
    S, A, M = rollouts(np.tile(th, (2000, 1)), rng); critics[name] = fit_critic(S, M, returns_to_go(M))
critics["zero"] = np.zeros(15)
LAMS = np.array([0, 0.5, 0.9, 0.97, 1.0])
est = {}
for batch in range(10):                                         # 10,000 episodes in batches of 1,000
    S, A, M = rollouts(np.tile(theta_fix, (1000, 1)), rng)
    T = len(M); G = returns_to_go(M); disc = GAMMA ** np.arange(T)[:, None]
    X = features(S[:-1]); score = ((A - sigmoid(X @ theta_fix)) * M)[..., None] * X     # T x n x 5
    per_episode = {"whole": G[0][:, None] * score.sum(0), "disc-to-go": np.einsum("tn,tnd->nd", disc * G, score),
                   "to-go": np.einsum("tn,tnd->nd", G, score)}
    for c, w in critics.items():
        V = quadratic(S) @ w
        if c == "fitted":
            per_episode["disc-baseline"] = np.einsum("tn,tnd->nd", disc * (G - V[:-1]), score)
        for lam, adv in zip(LAMS, gae(V, M, LAMS)):
            per_episode[(c, lam)] = np.einsum("tn,tnd->nd", adv, score)
    for key, e in per_episode.items():
        est.setdefault(key, []).append(e)
est = {k: np.vstack(v) for k, v in est.items()}


def error(key, target, N=10):
    """Squared bias and variance of the average of N per-episode estimates, relative to |mean of target|^2."""
    g = est[target].mean(0); nrm = g @ g
    return ((est[key].mean(0) - g) ** 2).sum() / nrm, est[key].var(0).sum() / N / nrm


print("  relative squared error of a 10-episode estimate = squared bias + variance, both divided by |gradient|^2")
print("  (a) estimates of the gradient of the discounted value of the start state:")
for key, label in [("whole", "whole return G_0"), ("disc-to-go", "gamma^t G_t"),
                   ("disc-baseline", "gamma^t (G_t - v(S_t)), fitted critic")]:
    b, v = error(key, "disc-to-go")
    print(f"    {label:40s} bias^2 {b:6.3f}   variance {v:6.3f}")
print("  (b) estimates of the direction without gamma^t: G_t, and GAE(lambda) with three critics (bias^2 + variance):")
b, v = error("to-go", "to-go")
print(f"    G_t, no critic: {b:5.3f} + {v:5.3f}")
print("    " + " " * 14 + "".join(f"   lambda = {l:<5g}" for l in LAMS))
for c in critics:
    print(f"    {c + ' critic':14s}" + "".join(f"  {b:5.3f} + {v:5.3f}" for b, v in [error((c, lam), "to-go") for lam in LAMS]))
print(f"  (from 10,000 episodes, each squared bias is uncertain by about a thousandth of its variance; time so far {time.time() - t0:.0f} s)")


# ---------------------------------------------------------------- part 4: the natural policy gradient
def batch_pg(configs, seeds=10, N=10, iters=100, lam=0.95):
    """Batch actor-critic: each iteration runs N episodes per policy, estimates the gradient with GAE(lam) and a
    quadratic critic fitted to the previous batch, and takes a step. configs: (method, step, scale), with method
    'vanilla' (theta += step * g) or 'natural' (a step along F^-1 g whose quadratic KL estimate equals step).
    All configurations share initial states and random numbers, seed by seed."""
    rng = np.random.default_rng(5)
    nc = len(configs); R = nc * seeds
    natural = np.repeat([c[0] == "natural" for c in configs], seeds); step = np.repeat([c[1] for c in configs], seeds)
    scale = np.repeat(np.array([c[2] for c in configs]), seeds * N, axis=0)
    theta, w = np.zeros((R, 5)), np.zeros((R, 15)); out = np.zeros((R, iters))
    for it in range(iters):
        S, A, M = rollouts(np.repeat(theta, N, 0), rng, scale, share=nc)
        T = len(M)
        out[:, it] = M.sum(0).reshape(R, N).mean(1)
        X = features(S[:-1], scale)
        score = ((A - sigmoid((X * np.repeat(theta, N, 0)).sum(-1))) * M)[..., None] * X
        V = np.einsum("tnd,nd->tn", quadratic(S), np.repeat(w, N, 0))
        adv = gae(V, M, np.array([lam]))[0]
        g = np.einsum("tn,tnd->nd", adv, score).reshape(R, N, 5).mean(1)          # gradient estimate per policy
        sc = score.transpose(1, 0, 2).reshape(R, N * T, 5)
        F = np.einsum("rkd,rke->rde", sc, sc) / M.sum(0).reshape(R, N).sum(1)[:, None, None]   # Fisher, per step
        F += 1e-3 * np.einsum("rdd->rd", F)[:, :, None] * np.eye(5)                 # damping proportional to diag(F)
        d = np.linalg.solve(F, g[:, :, None])[:, :, 0]
        theta += np.where(natural[:, None], np.sqrt(2 * step / np.maximum((g * d).sum(1), 1e-12))[:, None] * d,
                          step[:, None] * g)
        Q = (quadratic(S[:-1]) * M[..., None]).transpose(1, 0, 2).reshape(R, N * T, 15)   # refit the critic
        G = returns_to_go(M).T.reshape(R, N * T)
        w = np.linalg.solve(np.einsum("rkd,rke->rde", Q, Q) + 1e-3 * np.eye(15),
                            np.einsum("rkd,rk->rd", Q, G)[..., None])[..., 0]
    return out.reshape(nc, seeds, iters)


print("\n=== Part 4: vanilla versus natural policy gradient, 10 episodes per iteration, GAE(0.95) ===")
methods = [("vanilla", 3e-3), ("vanilla", 1e-2), ("vanilla", 3e-2), ("natural", 0.003), ("natural", 0.01), ("natural", 0.03)]
configs4 = [(m, st, sc) for sc in [SCALE, np.ones(4)] for m, st in methods]
L4 = batch_pg(configs4)
print("  average episode length, 10 seeds" + " " * 20 + "iterations 1-20   21-40   41-60   61-80  81-100  solved")
for k, ((m, st, sc), L) in enumerate(zip(configs4, L4)):
    label = f"{m}, {'alpha' if m == 'vanilla' else 'delta'} = {st:g}, {'scaled' if k < len(methods) else 'raw'} features"
    print(f"  {label:48s}{windows(L, 20)}   {(L[:, -10:].mean(1) > 475).sum():3d} of 10")
for k, (m, st) in enumerate(methods):
    if m == "natural":                                          # the same random numbers: compare run by run
        same = L4[k] == L4[k + len(methods)]
        first = [int(np.argmin(r)) if not r.all() else None for r in same]
        differ = sorted(f + 1 for f in first if f is not None)
        print(f"  natural, delta = {st:g}: raw and scaled features give identical runs in {10 - len(differ)} of 10 seeds"
              + (f"; the others first differ at iterations {', '.join(map(str, differ))}" if differ else ""))
print(f"\ntotal time {time.time() - t0:.0f} s")
