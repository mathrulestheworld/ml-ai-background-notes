"""Lab 11 reference solution: DDPG, TD3, and SAC on Pendulum and HalfCheetah.

Run from this folder:  python lab11_offpolicy.py
It prints the results quoted on the lab page. Takes about 40 minutes on two cores.
Needs gymnasium[mujoco].
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


def mlp(i, o, hidden=256):
    return nn.Sequential(nn.Linear(i, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, o))


class Replay:
    def __init__(self, n_obs, n_act, size):
        self.s, self.a = np.zeros((size, n_obs), np.float32), np.zeros((size, n_act), np.float32)
        self.r, self.s2, self.d = np.zeros(size, np.float32), np.zeros((size, n_obs), np.float32), np.zeros(size, np.float32)
        self.n, self.i, self.size = 0, 0, size

    def add(self, s, a, r, s2, d):
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i], self.d[self.i] = s, a, r, s2, d
        self.i = (self.i + 1) % self.size; self.n = min(self.n + 1, self.size)

    def sample(self, rng, batch):
        idx = rng.integers(self.n, size=batch)
        return [torch.as_tensor(x[idx]) for x in (self.s, self.a, self.r, self.s2, self.d)]


class SquashedGaussian(nn.Module):
    """SAC's actor: a Gaussian whose mean and log standard deviation come from the network, squashed by tanh."""

    def __init__(self, n_obs, n_act, hidden):
        super().__init__()
        self.net = mlp(n_obs, 2 * n_act, hidden)

    def forward(self, s, deterministic=False):
        mean, log_std = self.net(s).chunk(2, -1)
        log_std = log_std.clamp(-20, 2)
        u = mean if deterministic else mean + log_std.exp() * torch.randn_like(mean)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mean) / log_std.exp()) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)       # the tanh correction, computed stably
        return a, logp


def train(cfg):
    """One run of DDPG, TD3 (optionally without its clipped double Q), or SAC. Actions are scaled to [-1, 1]."""
    c = dict(env="Pendulum-v1", algo="sac", steps=15000, start=1000, batch=256, lr=3e-4, gamma=0.99, tau=0.005,
             noise=0.1, eval_every=1500, eval_episodes=5, seed=0, clipped_double=True, hidden=64)
    c.update(cfg)
    torch.set_num_threads(1); torch.manual_seed(c["seed"]); rng = np.random.default_rng(c["seed"])
    env, eval_env = gym.make(c["env"]), gym.make(c["env"])
    n_obs, n_act = env.observation_space.shape[0], env.action_space.shape[0]
    high = env.action_space.high
    algo = c["algo"]
    if algo == "sac":
        actor = SquashedGaussian(n_obs, n_act, c["hidden"])
        log_alpha = torch.zeros(1, requires_grad=True); alpha_opt = torch.optim.Adam([log_alpha], lr=c["lr"])
    else:
        actor = nn.Sequential(mlp(n_obs, n_act, c["hidden"]), nn.Tanh())
        actor_target = copy.deepcopy(actor)
    critics = [mlp(n_obs + n_act, 1, c["hidden"]) for _ in range(1 if algo == "ddpg" else 2)]
    targets = [copy.deepcopy(q) for q in critics]
    actor_opt = torch.optim.Adam(actor.parameters(), lr=c["lr"])
    critic_opt = torch.optim.Adam([p for q in critics for p in q.parameters()], lr=c["lr"])
    buf = Replay(n_obs, n_act, c["steps"])
    Q = lambda nets, s, a: [q(torch.cat([s, a], -1))[:, 0] for q in nets]

    def act(s, deterministic):
        with torch.no_grad():
            x = torch.as_tensor(s, dtype=torch.float32)[None]
            a = actor(x, deterministic)[0][0].numpy() if algo == "sac" else actor(x)[0].numpy()
        if not deterministic and algo != "sac":
            a = np.clip(a + c["noise"] * rng.normal(size=n_act), -1, 1)
        return a

    def evaluate():
        """Average return of the deterministic policy, and at the first state of each episode the critic's
        predicted value against the discounted return actually obtained."""
        rets, pred, disc = [], [], []
        for ep in range(c["eval_episodes"]):
            s, _ = eval_env.reset(seed=10_000 + ep); total, dsum, k = 0.0, 0.0, 0
            with torch.no_grad():
                x = torch.as_tensor(s, dtype=torch.float32)[None]
                a0 = torch.as_tensor(act(s, True), dtype=torch.float32)[None]
                pred.append(Q(critics, x, a0)[0].item())                   # the critic the actor follows
            while True:
                s, r, term, trunc, _ = eval_env.step(act(s, True) * high)
                total += r; dsum += c["gamma"] ** k * r; k += 1
                if term or trunc:
                    break
            rets.append(total); disc.append(dsum)
        return np.mean(rets), np.mean(pred), np.mean(disc)

    s, _ = env.reset(seed=c["seed"]); curve = []
    for t in range(1, c["steps"] + 1):
        a = rng.uniform(-1, 1, n_act) if t <= c["start"] else act(s, False)
        s2, r, term, trunc, _ = env.step(a * high)
        buf.add(s, a, r, s2, float(term))                               # only true terminations stop bootstrapping
        s = s2
        if term or trunc:
            s, _ = env.reset()
        if t > c["start"]:
            S, A, R, S2, D = buf.sample(rng, c["batch"])
            with torch.no_grad():
                if algo == "sac":
                    a2, logp2 = actor(S2); alpha = log_alpha.exp()
                    q2 = torch.min(*Q(targets, S2, a2)) - alpha * logp2
                elif algo == "td3":
                    noise = (0.2 * torch.randn_like(A)).clamp(-0.5, 0.5)
                    a2 = (actor_target(S2) + noise).clamp(-1, 1)
                    q2s = Q(targets, S2, a2)
                    q2 = torch.min(*q2s) if c["clipped_double"] else q2s[0]
                else:
                    q2 = Q(targets, S2, actor_target(S2))[0]
                y = R + c["gamma"] * (1 - D) * q2
            critic_loss = sum(F.mse_loss(q, y) for q in Q(critics, S, A))
            critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()
            if algo != "td3" or t % 2 == 0:                              # TD3 delays the actor and the targets
                if algo == "sac":
                    a_new, logp = actor(S)
                    actor_loss = (log_alpha.exp().detach() * logp - torch.min(*Q(critics, S, a_new))).mean()
                    alpha_loss = -(log_alpha * (logp.detach() - n_act)).mean()   # target entropy -dim(A)
                    alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()
                else:
                    actor_loss = -Q(critics, S, actor(S))[0].mean()
                actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
                with torch.no_grad():
                    pairs = list(zip(critics, targets)) + ([] if algo == "sac" else [(actor, actor_target)])
                    for net, tgt in pairs:
                        for p, pt in zip(net.parameters(), tgt.parameters()):
                            pt.mul_(1 - c["tau"]).add_(c["tau"] * p)
        if t % c["eval_every"] == 0:
            curve.append(evaluate())
    return np.array(curve)


def report(names, results, ticks):
    for name, res in zip(names, results):
        curves = np.array([r[:, 0] for r in res])
        last = np.array([r[-1] for r in res])
        print(f"  {name:28s}" + "".join(f"{curves.mean(0)[k]:8.0f}" for k in ticks)
              + f"   {last[:, 1].mean():9.1f} {last[:, 2].mean():9.1f}   " + " ".join(f"{x:6.0f}" for x in curves[:, -1]))


if __name__ == "__main__":
    print("=== Part 1: Pendulum-v1, 15,000 steps (the first 1,000 random), networks of 64 units, 3 seeds ===")
    print("  average return of the deterministic policy over 5 episodes (mean of seeds); at the end, the critic's value")
    print("  of the first state against the discounted return actually obtained; the final return of each seed")
    print("  " + " " * 28 + "  3,000   6,000   9,000  12,000  15,000   predicted    actual   per seed")
    variants = [("DDPG", dict(algo="ddpg")), ("TD3", dict(algo="td3")),
                ("TD3 without clipped double Q", dict(algo="td3", clipped_double=False)), ("SAC", dict(algo="sac"))]
    jobs = [dict(cfg, seed=s) for _, cfg in variants for s in range(3)]
    with mp.get_context("fork").Pool(2) as pool:
        res = pool.map(train, jobs)
    report([n for n, _ in variants], [res[3 * k: 3 * k + 3] for k in range(len(variants))], (1, 3, 5, 7, 9))
    print(f"  (time so far {time.time() - t0:.0f} s)")

    print("\n=== Part 2: HalfCheetah-v5, 100,000 steps (the first 5,000 random), networks of 256 units, 2 seeds ===")
    print("  " + " " * 28 + "    20k     40k     60k     80k    100k   predicted    actual   per seed")
    hc = dict(env="HalfCheetah-v5", steps=100_000, start=5000, eval_every=10_000, eval_episodes=3, hidden=256)
    variants = [("TD3", dict(hc, algo="td3")), ("SAC", dict(hc, algo="sac"))]
    jobs = [dict(cfg, seed=s) for _, cfg in variants for s in range(2)]
    with mp.get_context("fork").Pool(2) as pool:
        res = pool.map(train, jobs)
    report([n for n, _ in variants], [res[2 * k: 2 * k + 2] for k in range(len(variants))], (1, 3, 5, 7, 9))
    print(f"\ntotal time {time.time() - t0:.0f} s")
