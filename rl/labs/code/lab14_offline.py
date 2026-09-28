"""Lab 14 reference solution: imitation (behavioral cloning and DAgger) and offline RL (naive TD3, TD3+BC, IQL)
on the pendulum, from datasets of different quality.

Run from this folder:  python lab14_offline.py
It prints the results quoted on the lab page. Takes about 25 minutes on two cores.
"""
import copy
import multiprocessing as mp
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

t0 = time.time()
GAMMA = 0.99


def mlp(i, o, hidden=256):
    return nn.Sequential(nn.Linear(i, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, o))


class Deterministic(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = mlp(3, 1, hidden)

    def forward(self, s):
        return torch.tanh(self.net(s))


class SquashedGaussian(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = mlp(3, 2, hidden)

    def forward(self, s, deterministic=False):
        mean, log_std = self.net(s).chunk(2, -1)
        log_std = log_std.clamp(-20, 2)
        u = mean if deterministic else mean + log_std.exp() * torch.randn_like(mean)
        logp = (-0.5 * ((u - mean) / log_std.exp()) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return torch.tanh(u), logp


def act_fn(actor, stochastic=False):
    """A function from an observation to an action in [-1, 1]."""
    def f(s):
        with torch.no_grad():
            x = torch.as_tensor(s, dtype=torch.float32)[None]
            a = actor(x, deterministic=not stochastic)[0] if isinstance(actor, SquashedGaussian) else actor(x)
        return a[0].numpy()
    return f


def rollout(policy, episodes, seed, noise=0.0, rng=None, labeler=None):
    """Run a policy (actions in [-1, 1]); return the transitions, the episode returns, and, if a labeler is given,
    its actions at the visited states."""
    env = gym.make("Pendulum-v1")
    S, A, R, S2, L, rets = [], [], [], [], [], []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep); tot = 0.0
        for t in range(200):
            a = policy(s)
            if noise:
                a = np.clip(a + noise * rng.normal(size=1), -1, 1)
            if labeler is not None:
                L.append(labeler(s))
            s2, r, term, trunc, _ = env.step(2.0 * a)
            S.append(s); A.append(a); R.append(r); S2.append(s2); tot += r; s = s2
        rets.append(tot)
    arr = lambda x: np.array(x, dtype=np.float32)
    return dict(s=arr(S), a=arr(A).reshape(-1, 1), r=arr(R), s2=arr(S2), labels=arr(L).reshape(-1, 1)), np.array(rets)


def evaluate(policy, episodes=10, seed=10_000):
    return rollout(policy, episodes, seed)[1].mean()


# ---------------------------------------------------------------- the expert
def train_sac(seed, steps=15000):
    """SAC on Pendulum-v1 (as in Lab 11); returns the final actor, snapshots every 250 steps, and all
    transitions seen (the replay memory)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    env = gym.make("Pendulum-v1")
    actor, critics = SquashedGaussian(), [mlp(4, 1) for _ in range(2)]
    targets = [copy.deepcopy(q) for q in critics]
    log_alpha = torch.zeros(1, requires_grad=True)
    a_opt = torch.optim.Adam(actor.parameters(), 3e-4)
    c_opt = torch.optim.Adam([p for q in critics for p in q.parameters()], 3e-4)
    al_opt = torch.optim.Adam([log_alpha], 3e-4)
    S, A, R, S2 = (np.zeros((steps, k), np.float32) for k in (3, 1, 1, 3))
    s, _ = env.reset(seed=seed); snaps = {}
    for t in range(steps):
        a = rng.uniform(-1, 1, 1) if t < 1000 else act_fn(actor, stochastic=True)(s)
        s2, r, term, trunc, _ = env.step(2.0 * a)
        S[t], A[t], R[t], S2[t] = s, a, r, s2
        s = s2
        if term or trunc:
            s, _ = env.reset()
        if t >= 1000:
            idx = rng.integers(t + 1, size=256)
            s_, a_, r_, s2_ = (torch.as_tensor(x[idx]) for x in (S, A, R, S2))
            with torch.no_grad():
                a2, lp2 = actor(s2_)
                q2 = torch.min(*[q(torch.cat([s2_, a2], -1)) for q in targets])[:, 0]
                y = r_[:, 0] + GAMMA * (q2 - log_alpha.exp() * lp2)
            loss = sum(F.mse_loss(q(torch.cat([s_, a_], -1))[:, 0], y) for q in critics)
            c_opt.zero_grad(); loss.backward(); c_opt.step()
            an, lp = actor(s_)
            a_loss = (log_alpha.exp().detach() * lp - torch.min(*[q(torch.cat([s_, an], -1)) for q in critics])[:, 0]).mean()
            a_opt.zero_grad(); a_loss.backward(); a_opt.step()
            al_loss = -(log_alpha * (lp.detach() - 1.0)).mean()                 # target entropy -1
            al_opt.zero_grad(); al_loss.backward(); al_opt.step()
            with torch.no_grad():
                for q, qt in zip(critics, targets):
                    for p, pt in zip(q.parameters(), qt.parameters()):
                        pt.mul_(0.995).add_(0.005 * p)
        if (t + 1) % 250 == 0:
            snaps[t + 1] = copy.deepcopy(actor)
    replay = dict(s=S, a=A, r=R[:, 0], s2=S2)
    return actor, snaps, replay


# ---------------------------------------------------------------- imitation
def fit_bc(S, A, steps, seed, net=None, hidden=64):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = net or Deterministic(hidden)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    S, A = torch.as_tensor(S), torch.as_tensor(A)
    for _ in range(steps):
        idx = torch.as_tensor(rng.integers(len(S), size=256))
        loss = F.mse_loss(net(S[idx]), A[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def imitation(args):
    """Behavioral cloning from k expert episodes, and DAgger: start from one expert episode, then repeatedly run
    the learner for one episode, have the expert label the states it visited, and retrain on everything."""
    expert, seed, budgets = args
    torch.set_num_threads(1)
    expert_fn = act_fn(expert)
    bc, dagger = [], []
    demo, _ = rollout(expert_fn, max(budgets), seed=100 * seed)
    for k in budgets:                                              # k episodes of 200 labeled states
        net = fit_bc(demo["s"][:200 * k], demo["a"][:200 * k], 2000, seed)
        bc.append(evaluate(act_fn(net)))
    S, A = demo["s"][:200], demo["a"][:200]
    net = fit_bc(S, A, 2000, seed)
    for k in range(2, max(budgets) + 1):
        new, _ = rollout(act_fn(net), 1, seed=100 * seed + 50 + k, labeler=expert_fn)
        S, A = np.concatenate([S, new["s"]]), np.concatenate([A, new["labels"]])
        net = fit_bc(S, A, 500, seed + k, net=net)
        if k in budgets:
            dagger.append(evaluate(act_fn(net)))
    return bc, [bc[0]] + dagger


# ---------------------------------------------------------------- offline RL
def offline(args):
    """Train one offline agent on a dataset: 'bc', 'td3' (naive off-policy TD3), 'td3+bc', or 'iql'. Returns the
    average return of its policy, and the critic's value of the evaluation episodes' first states against the
    discounted return actually obtained from them."""
    algo, data, seed, steps = args
    torch.set_num_threads(1); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    mu, sd = data["s"].mean(0), data["s"].std(0) + 1e-3                  # state normalization (TD3+BC's)
    norm = lambda x: (x - mu) / sd
    S, A, R, S2 = (torch.as_tensor(x) for x in (norm(data["s"]), data["a"], data["r"], norm(data["s2"])))
    actor, critics = Deterministic(), [mlp(4, 1) for _ in range(2)]
    actor_t, targets = copy.deepcopy(actor), [copy.deepcopy(q) for q in critics]
    value = mlp(3, 1)
    a_opt = torch.optim.Adam(actor.parameters(), 3e-4)
    c_opt = torch.optim.Adam([p for q in critics for p in q.parameters()], 3e-4)
    v_opt = torch.optim.Adam(value.parameters(), 3e-4)
    Q = lambda nets, s, a: [q(torch.cat([s, a], -1))[:, 0] for q in nets]
    for it in range(steps):
        idx = torch.as_tensor(rng.integers(len(S), size=256))
        s, a, r, s2 = S[idx], A[idx], R[idx], S2[idx]
        if algo == "bc":
            loss = F.mse_loss(actor(s), a)
            a_opt.zero_grad(); loss.backward(); a_opt.step()
            continue
        if algo == "iql":
            with torch.no_grad():
                q_t = torch.min(*Q(targets, s, a))
            u = q_t - value(s)[:, 0]
            v_loss = (torch.abs(0.7 - (u < 0).float()) * u ** 2).mean()       # expectile 0.7
            v_opt.zero_grad(); v_loss.backward(); v_opt.step()
            with torch.no_grad():
                y = r + GAMMA * value(s2)[:, 0]
            c_loss = sum(F.mse_loss(q, y) for q in Q(critics, s, a))
            c_opt.zero_grad(); c_loss.backward(); c_opt.step()
            with torch.no_grad():
                w = torch.exp(3.0 * (q_t - value(s)[:, 0])).clamp(max=100.0)   # advantage weights, beta = 3
            a_loss = (w * ((actor(s) - a) ** 2)[:, 0]).mean()
            a_opt.zero_grad(); a_loss.backward(); a_opt.step()
        else:                                                              # TD3, with or without the BC term
            with torch.no_grad():
                a2 = (actor_t(s2) + (0.2 * torch.randn_like(a)).clamp(-0.5, 0.5)).clamp(-1, 1)
                y = r + GAMMA * torch.min(*Q(targets, s2, a2))
            c_loss = sum(F.mse_loss(q, y) for q in Q(critics, s, a))
            c_opt.zero_grad(); c_loss.backward(); c_opt.step()
            if it % 2 == 0:
                pi = actor(s); q = Q(critics[:1], s, pi)[0]
                if algo == "td3+bc":
                    lam = 2.5 / q.abs().mean().detach()
                    a_loss = -lam * q.mean() + F.mse_loss(pi, a)
                else:
                    a_loss = -q.mean()
                a_opt.zero_grad(); a_loss.backward(); a_opt.step()
                with torch.no_grad():
                    for net, tgt in [(actor, actor_t)] + list(zip(critics, targets)):
                        for p, pt in zip(net.parameters(), tgt.parameters()):
                            pt.mul_(0.995).add_(0.005 * p)
        if algo == "iql":
            with torch.no_grad():
                for q, qt in zip(critics, targets):
                    for p, pt in zip(q.parameters(), qt.parameters()):
                        pt.mul_(0.995).add_(0.005 * p)
    policy = lambda s_: act_fn(actor)(norm(s_).astype(np.float32))
    ev, rets = rollout(policy, 10, 10_000)
    disc = (ev["r"].reshape(10, 200) * GAMMA ** np.arange(200)).sum(1).mean()
    with torch.no_grad():
        s0 = torch.as_tensor(norm(ev["s"].reshape(10, 200, 3)[:, 0]))
        pred = torch.min(*Q(critics, s0, actor(s0))).mean().item() if algo != "bc" else float("nan")
    return rets.mean(), pred, disc


if __name__ == "__main__":
    torch.set_num_threads(1)
    print("=== Part 1: an expert and four datasets (Pendulum-v1) ===")
    expert, snaps, replay = train_sac(seed=0)
    ev = {k: evaluate(act_fn(a)) for k, a in snaps.items()}
    medium_step = min(ev, key=lambda k: abs(ev[k] + 500))                # a checkpoint scoring about -500
    rng = np.random.default_rng(0)
    datasets = {}
    datasets["expert"], r_exp = rollout(act_fn(expert), 100, 20_000, noise=0.1, rng=rng)
    datasets["medium"], r_med = rollout(act_fn(snaps[medium_step], stochastic=True), 100, 30_000)
    datasets["random"], r_rnd = rollout(lambda s: rng.uniform(-1, 1, 1), 100, 40_000)
    datasets["replay"] = replay
    r_rep = replay["r"].reshape(-1, 200).sum(1)
    print(f"  expert (SAC after 15,000 steps, deterministic): average return {evaluate(act_fn(expert)):.0f}")
    print(f"  medium policy: SAC's actor after {medium_step:,} steps, sampled: {ev[medium_step]:.0f} (deterministic)")
    print("  dataset          transitions   average episode return   (min, max)")
    for name, rets in (("expert", r_exp), ("medium", r_med), ("random", r_rnd), ("replay", r_rep)):
        print(f"  {name:15s}  {len(datasets[name]['r']):11,d}   {rets.mean():22.0f}   ({rets.min():.0f}, {rets.max():.0f})")
    print(f"  (time so far {time.time() - t0:.0f} s)")

    pool = mp.get_context("fork").Pool(2)
    print("\n=== Part 2: behavioral cloning and DAgger, by number of expert-labeled states (mean of 3 seeds) ===")
    budgets = (1, 3, 5, 10, 20)
    res = pool.map(imitation, [(expert, seed, budgets) for seed in range(3)])
    print("  labeled states:  " + "".join(f"{200 * k:8,d}" for k in budgets))
    print("  BC               " + "".join(f"{x:8.0f}" for x in np.mean([r[0] for r in res], 0)))
    print("  DAgger           " + "".join(f"{x:8.0f}" for x in np.mean([r[1] for r in res], 0)))
    print(f"  (time so far {time.time() - t0:.0f} s)")

    print("\n=== Part 3: offline RL, 10,000 gradient steps, 2 seeds: average return of the policy ===")
    algos = ("bc", "td3", "td3+bc", "iql")
    names = ("expert", "medium", "random", "replay")
    jobs = [(algo, datasets[d], seed, 10000) for d in names for algo in algos for seed in range(2)]
    out = pool.map(offline, jobs)
    print("  mean of the seeds, and half the difference between them")
    print("  dataset       data          BC           TD3        TD3+BC           IQL")
    k = 0
    table = {}
    for d, rets in zip(names, (r_exp, r_med, r_rnd, r_rep)):
        row = []
        for algo in algos:
            x1, x2 = out[k][0], out[k + 1][0]
            row.append(f"{(x1 + x2) / 2:8.0f} ±{abs(x1 - x2) / 2:<4.0f}"); table[(d, algo)] = (out[k], out[k + 1]); k += 2
        print(f"  {d:10s}  {rets.mean():6.0f}" + "".join(row))
    print("  the critic's value of the first states against the discounted return obtained (mean of seeds)")
    print("  dataset          TD3 predicted  obtained    TD3+BC predicted  obtained    IQL predicted  obtained")
    for d in names:
        cells = []
        for algo in ("td3", "td3+bc", "iql"):
            (a1, p1, d1), (a2, p2, d2) = table[(d, algo)]
            cells.append(f"{(p1 + p2) / 2:13.0f} {(d1 + d2) / 2:9.0f}")
        print(f"  {d:10s}  " + "    ".join(cells))
    print(f"\ntotal time {time.time() - t0:.0f} s")
    pool.close()
