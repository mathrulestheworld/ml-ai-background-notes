"""Lab 12 reference solution: model-based RL with learned ensembles on the pendulum (PETS and MBPO).

Run from this folder:  python lab12_model_based.py
It prints the results quoted on the lab page. Takes about 15 minutes on two cores.
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
MAX_SPEED, MAX_TORQUE = 8.0, 2.0


def pendulum_reward(obs, u):
    """Pendulum-v1's reward from the observation (cos, sin, speed) and the torque; known to the agent."""
    th = torch.atan2(obs[..., 1], obs[..., 0])
    return -(th ** 2 + 0.1 * obs[..., 2] ** 2 + 0.001 * u ** 2)


class Ensemble(nn.Module):
    """E probabilistic networks trained together: each maps (cos, sin, speed/8, u/2) to a Gaussian over the change
    of (cos, sin, speed). With probabilistic=False, each outputs a mean only (trained by squared error)."""

    def __init__(self, E=5, hidden=64, probabilistic=True):
        super().__init__()
        self.E, self.probabilistic = E, probabilistic
        self.W1 = nn.Parameter(torch.randn(E, 4, hidden) * 0.5); self.b1 = nn.Parameter(torch.zeros(E, 1, hidden))
        self.W2 = nn.Parameter(torch.randn(E, hidden, hidden) / hidden ** 0.5); self.b2 = nn.Parameter(torch.zeros(E, 1, hidden))
        self.W3 = nn.Parameter(torch.randn(E, hidden, 6) / hidden ** 0.5 * 0.1); self.b3 = nn.Parameter(torch.zeros(E, 1, 6))
        self.max_logvar, self.min_logvar = nn.Parameter(torch.full((1, 1, 3), 0.0)), nn.Parameter(torch.full((1, 1, 3), -10.0))

    def forward(self, obs, u):
        """obs: (E, N, 3), u: (E, N) -> mean and log-variance of the next observation, each (E, N, 3)."""
        x = torch.cat([obs[..., :2], obs[..., 2:] / MAX_SPEED, u[..., None] / MAX_TORQUE], -1)
        h = F.silu(torch.baddbmm(self.b1, x, self.W1)); h = F.silu(torch.baddbmm(self.b2, h, self.W2))
        out = torch.baddbmm(self.b3, h, self.W3)
        mean, logvar = obs + out[..., :3] * torch.tensor([1.0, 1.0, MAX_SPEED]), out[..., 3:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def fit(self, S, U, S2, rng, steps=400, batch=256):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        n = len(S)
        S, U, S2 = map(lambda a: torch.as_tensor(a, dtype=torch.float32), (S, U, S2))
        for _ in range(steps):
            idx = torch.as_tensor(rng.integers(n, size=(self.E, batch)))       # a bootstrap sample per member
            mean, logvar = self(S[idx], U[idx])
            if self.probabilistic:
                loss = (((mean - S2[idx]) ** 2) * torch.exp(-logvar) + logvar).mean() \
                       + 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
            else:
                loss = ((mean - S2[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    def step(self, obs, u, member=None, sample=True):
        """Predict the next observations for a batch spread over the members: obs (E, N, 3)."""
        mean, logvar = self(obs, u)
        if self.probabilistic and sample:
            mean = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return mean


def pets_episode(env, model, rng, plan, H=15, pop=200, elites=20, iters=4):
    """One episode of model predictive control with the cross-entropy method; each candidate is evaluated with one
    particle per ensemble member (trajectory sampling with a fixed member per particle)."""
    obs, _ = env.reset(seed=int(rng.integers(1 << 30))); total, data = 0.0, []
    mean_plan = np.zeros(H)
    for t in range(200):
        if plan:
            mu, sigma = mean_plan.copy(), np.full(H, 1.5)
            with torch.no_grad():
                for it in range(iters):
                    seqs = np.clip(mu + sigma * rng.normal(size=(pop, H)), -MAX_TORQUE, MAX_TORQUE)
                    U = torch.as_tensor(seqs, dtype=torch.float32)
                    o = torch.as_tensor(obs, dtype=torch.float32).expand(model.E, pop, 3).clone()
                    ret = torch.zeros(model.E, pop)
                    for h in range(H):
                        u = U[:, h].expand(model.E, pop)
                        o = model.step(o, u); ret += pendulum_reward(o, u)
                    score = ret.mean(0).numpy()                                 # average over the members
                    elite = seqs[np.argsort(-score)[:elites]]
                    mu, sigma = elite.mean(0), elite.std(0) + 0.05
            u = float(mu[0]); mean_plan = np.append(mu[1:], 0.0)
        else:
            u = float(rng.uniform(-MAX_TORQUE, MAX_TORQUE))
        obs2, r, term, trunc, _ = env.step(np.array([u]))
        data.append((obs, u, obs2)); total += r; obs = obs2
        if term or trunc:
            break
    return total, data


def pets(cfg):
    """PETS-style learning: one random episode, then MPC with the model retrained after every episode."""
    seed, probabilistic, E = cfg["seed"], cfg["probabilistic"], cfg["E"]
    torch.set_num_threads(1); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    env = gym.make("Pendulum-v1")
    model = Ensemble(E=E, probabilistic=probabilistic)
    returns, S, U, S2 = [], [], [], []
    for ep in range(cfg["episodes"]):
        total, data = pets_episode(env, model, rng, plan=ep > 0)
        returns.append(total)
        S += [d[0] for d in data]; U += [d[1] for d in data]; S2 += [d[2] for d in data]
        model.fit(np.array(S), np.array(U), np.array(S2), rng)
    return returns


# ---------------------------------------------------------------- SAC and MBPO
def mlp(i, o, hidden=64):
    return nn.Sequential(nn.Linear(i, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, o))


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(3, 2)

    def forward(self, s, deterministic=False):
        mean, log_std = self.net(s).chunk(2, -1)
        log_std = log_std.clamp(-20, 2)
        u = mean if deterministic else mean + log_std.exp() * torch.randn_like(mean)
        logp = (-0.5 * ((u - mean) / log_std.exp()) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return torch.tanh(u), logp


def sac_mbpo(cfg):
    """SAC on Pendulum-v1 with `utd` updates per environment step. With mbpo=True, 95% of each minibatch comes
    from branched model rollouts: every 250 steps the ensemble is refit, and every step 50 rollouts of length k
    start from real states, with k growing from 1 to 5 over the first 3,000 steps."""
    seed = cfg["seed"]
    torch.set_num_threads(1); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    env, eval_env = gym.make("Pendulum-v1"), gym.make("Pendulum-v1")
    actor, critics = Actor(), [mlp(4, 1) for _ in range(2)]
    targets = [copy.deepcopy(q) for q in critics]
    log_alpha = torch.zeros(1, requires_grad=True)
    a_opt = torch.optim.Adam(actor.parameters(), 3e-4); c_opt = torch.optim.Adam([p for q in critics for p in q.parameters()], 3e-4)
    al_opt = torch.optim.Adam([log_alpha], 3e-4)
    class Buffer:
        def __init__(self, size):
            self.d = {k: np.zeros((size, n), np.float32) for k, n in zip("sar2", (3, 1, 1, 3))}
            self.n, self.i, self.size = 0, 0, size

        def add(self, s_, a_, r_, s2_):
            m = len(s_)
            idx = (self.i + np.arange(m)) % self.size
            for k, v in zip("sar2", (s_, a_, r_, s2_)):
                self.d[k][idx] = np.asarray(v, dtype=np.float32).reshape(m, -1)
            self.i = (self.i + m) % self.size; self.n = min(self.n + m, self.size)

    real, fake = Buffer(cfg["steps"]), Buffer(50_000)
    model = Ensemble(E=5) if cfg["mbpo"] else None
    Q = lambda nets, s, a: [q(torch.cat([s, a], -1))[:, 0] for q in nets]

    def evaluate():
        tot = 0.0
        for ep in range(5):
            s, _ = eval_env.reset(seed=1000 + ep)
            for t in range(200):
                with torch.no_grad():
                    a = actor(torch.as_tensor(s, dtype=torch.float32)[None], True)[0][0].numpy()
                s, r, term, trunc, _ = eval_env.step(a * MAX_TORQUE); tot += r
        return tot / 5

    def batch_from(buf, n):
        idx = rng.integers(buf.n, size=n)
        S_, A_, R_, S2_ = (torch.as_tensor(buf.d[k][idx]) for k in "sar2")
        return S_, A_, R_[:, 0], S2_

    s, _ = env.reset(seed=seed); curve = []
    for t in range(1, cfg["steps"] + 1):
        if t <= 500:
            a = rng.uniform(-1, 1, 1)
        else:
            with torch.no_grad():
                a = actor(torch.as_tensor(s, dtype=torch.float32)[None])[0][0].numpy()
        s2, r, term, trunc, _ = env.step(a * MAX_TORQUE)
        real.add([s], [a], [r], [s2])
        s = s2
        if term or trunc:
            s, _ = env.reset()
        if cfg["mbpo"] and t >= 500:
            if t % 250 == 0 or t == 500:
                model.fit(real.d["s"][:real.n], real.d["a"][:real.n, 0] * MAX_TORQUE, real.d["2"][:real.n], rng, steps=300)
            k = 1 + int(4 * min(1.0, t / 3000))
            S0 = batch_from(real, 50)[0]
            with torch.no_grad():
                o = S0
                for h in range(k):
                    a_m = actor(o)[0]
                    member = torch.as_tensor(rng.integers(5, size=len(o)))
                    nxt = model.step(o.expand(5, -1, -1), (a_m[:, 0] * MAX_TORQUE).expand(5, -1))[member, torch.arange(len(o))]
                    r_m = pendulum_reward(nxt, a_m[:, 0] * MAX_TORQUE)
                    fake.add(o.numpy(), a_m.numpy(), r_m.numpy(), nxt.numpy())
                    o = nxt
        if t > 500:
            for u_ in range(cfg["utd"]):
                if cfg["mbpo"]:
                    parts = [batch_from(fake, 243), batch_from(real, 13)]
                    S, A, R, S2 = [torch.cat([p[i] for p in parts]) for i in range(4)]
                else:
                    S, A, R, S2 = batch_from(real, 256)
                with torch.no_grad():
                    a2, lp2 = actor(S2); alpha = log_alpha.exp()
                    y = R + 0.99 * (torch.min(*Q(targets, S2, a2)) - alpha * lp2)
                loss = sum(F.mse_loss(q, y) for q in Q(critics, S, A))
                c_opt.zero_grad(); loss.backward(); c_opt.step()
                an, lp = actor(S)
                a_loss = (log_alpha.exp().detach() * lp - torch.min(*Q(critics, S, an))).mean()
                a_opt.zero_grad(); a_loss.backward(); a_opt.step()
                al_loss = -(log_alpha * (lp.detach() - 1.0)).mean()          # target entropy -dim(A) = -1
                al_opt.zero_grad(); al_loss.backward(); al_opt.step()
                with torch.no_grad():
                    for q, qt in zip(critics, targets):
                        for p, pt in zip(q.parameters(), qt.parameters()):
                            pt.mul_(0.995).add_(0.005 * p)
        if t % 1000 == 0:
            curve.append(evaluate())
    return curve


if __name__ == "__main__":
    print("=== Part 1: learning the pendulum with a learned model and MPC (15 episodes of 200 steps, 3 seeds) ===")
    print("  return of each episode (the first with random torques), mean of seeds; a return above about -200 means")
    print("  the pendulum is swung up and held")
    variants = [("probabilistic ensemble of 5 (PETS)", dict(probabilistic=True, E=5)),
                ("one deterministic network", dict(probabilistic=False, E=1))]
    jobs = [dict(cfg, seed=s, episodes=15) for _, cfg in variants for s in range(3)]
    with mp.get_context("fork").Pool(2) as pool:
        res = pool.map(pets, jobs)
    print("  " + " " * 36 + "episode:  1       2       3       5       8      10      15")
    for k, (name, _) in enumerate(variants):
        r = np.array(res[3 * k: 3 * k + 3])
        print(f"  {name:36s}   " + "".join(f"{r[:, e - 1].mean():8.0f}" for e in (1, 2, 3, 5, 8, 10, 15))
              + "   seeds at episode 15: " + " ".join(f"{x:5.0f}" for x in r[:, -1]))
    print(f"  (time so far {time.time() - t0:.0f} s)")

    print("\n=== Part 2: SAC, SAC with 10 updates per step, and MBPO, 4,000 environment steps, 3 seeds ===")
    print("  average return of the deterministic policy over 5 episodes, mean of seeds")
    variants = [("SAC, 1 update per step", dict(utd=1, mbpo=False)), ("SAC, 10 updates per step", dict(utd=10, mbpo=False)),
                ("MBPO (SAC, 10 updates, model data)", dict(utd=10, mbpo=True))]
    jobs = [dict(cfg, seed=s, steps=4000) for _, cfg in variants for s in range(3)]
    with mp.get_context("fork").Pool(2) as pool:
        res = pool.map(sac_mbpo, jobs)
    print("  " + " " * 36 + "  steps: 1,000   2,000   3,000   4,000   seeds at 4,000")
    for k, (name, _) in enumerate(variants):
        r = np.array(res[3 * k: 3 * k + 3])
        print(f"  {name:36s}       " + "".join(f"{x:8.0f}" for x in r.mean(0)) + "   " + " ".join(f"{x:5.0f}" for x in r[:, -1]))
    print(f"\ntotal time {time.time() - t0:.0f} s")
