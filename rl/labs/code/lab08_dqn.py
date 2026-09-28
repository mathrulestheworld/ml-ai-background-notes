"""Lab 8 reference solution: deep Q-networks on CartPole and MinAtar.

Run from this folder:  python lab08_dqn.py
It prints the results quoted on the lab page. Takes about 25 minutes on two cores.
"""
import multiprocessing as mp
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import torch

torch.set_num_threads(1)
t0 = time.time()
GAMMA = 0.99


# ---------------------------------------------------------------- part 1: many DQN agents on CartPole at once
class VecCartPole:
    """n independent copies of Gymnasium's CartPole-v1 (checked against Gymnasium in Lab 6), each resetting itself."""
    g, m_cart, m_pole, half_len, force, tau, x_max, th_max = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02, 2.4, 12 * 2 * np.pi / 360

    def __init__(self, n, rng):
        self.n, self.rng = n, rng
        self.s = rng.uniform(-0.05, 0.05, (n, 4)); self.t = np.zeros(n, int)

    def step(self, a):
        x, xd, th, thd = self.s.T
        f = np.where(a == 1, self.force, -self.force); total, pml = self.m_cart + self.m_pole, self.m_pole * self.half_len
        temp = (f + pml * thd ** 2 * np.sin(th)) / total
        tha = (self.g * np.sin(th) - np.cos(th) * temp) / (self.half_len * (4 / 3 - self.m_pole * np.cos(th) ** 2 / total))
        xa = temp - pml * tha * np.cos(th) / total
        s2 = np.stack([x + self.tau * xd, xd + self.tau * xa, th + self.tau * thd, thd + self.tau * tha], 1)
        self.t += 1
        term = (np.abs(s2[:, 0]) > self.x_max) | (np.abs(s2[:, 2]) > self.th_max)
        done = term | (self.t >= 500)
        true_next = s2.copy()
        s2[done] = self.rng.uniform(-0.05, 0.05, (done.sum(), 4)); self.t[done] = 0; self.s = s2
        return true_next, term, done


def init_params(G, sizes, gen):
    """One MLP per agent, stored with a leading agent dimension, initialized like torch.nn.Linear."""
    params = []
    for i, o in zip(sizes[:-1], sizes[1:]):
        bound = 1 / np.sqrt(i)
        params += [(torch.rand(G, i, o, generator=gen) * 2 - 1) * bound, (torch.rand(G, 1, o, generator=gen) * 2 - 1) * bound]
    return [p.requires_grad_() for p in params]


def q_values(params, x, dueling):
    """Forward pass of all agents' networks: x is (agents, batch, 4); the last layer has 3 outputs, used either as
    two action values (the third ignored) or as a state value and two advantages (dueling agents)."""
    for k in range(0, len(params), 2):
        x = torch.baddbmm(params[k + 1], x, params[k])
        if k < len(params) - 2:
            x = torch.relu(x)
    duel = x[..., :1] + x[..., 1:] - x[..., 1:].mean(-1, keepdim=True)
    return torch.where(dueling[:, None, None], duel, x[..., :2])


def batched_dqn(configs, seeds=4, steps=40000, hidden=64, batch=64, buffer=50000, lr=1e-3, start=1000, eps_steps=10000):
    G = len(configs) * seeds
    per = lambda k, d: torch.tensor([c.get(k, d) for c in configs for _ in range(seeds)])
    period, window, double, dueling = per("period", 100), per("replay", buffer), per("double", False), per("dueling", False)
    tau, lrs = per("tau", 0.0), per("lr", lr)
    gen = torch.Generator().manual_seed(0); rng = np.random.default_rng(0)
    params = init_params(G, [4, hidden, hidden, 3], gen)
    target = [p.detach().clone() for p in params]
    m, v = [torch.zeros_like(p) for p in params], [torch.zeros_like(p) for p in params]   # Adam, written out so that
    lr_g = lrs.float()[:, None, None]                                                      # each agent has its own step size
    env = VecCartPole(G, rng)
    S, S2 = torch.zeros(G, buffer, 4), torch.zeros(G, buffer, 4)
    A, Dn = torch.zeros(G, buffer, dtype=torch.long), torch.zeros(G, buffer)
    ar = torch.arange(G)[:, None]
    ep_len, lengths, curve, values = np.zeros(G), [[] for _ in range(G)], [], []
    starts = torch.as_tensor(rng.uniform(-0.05, 0.05, (G, 256, 4)), dtype=torch.float32)   # for value estimates
    obs = torch.as_tensor(env.s, dtype=torch.float32)
    for t in range(steps):
        eps = max(0.05, 1 - 0.95 * t / eps_steps)
        with torch.no_grad():
            greedy = q_values(params, obs[:, None], dueling)[:, 0].argmax(1).numpy()
        a = np.where(rng.random(G) < eps, rng.integers(2, size=G), greedy)
        nxt, term, done = env.step(a)
        i = t % buffer
        S[:, i], A[:, i] = obs, torch.as_tensor(a)
        S2[:, i], Dn[:, i] = torch.as_tensor(nxt, dtype=torch.float32), torch.as_tensor(term, dtype=torch.float32)
        ep_len += 1
        for g in np.flatnonzero(done):
            lengths[g].append(ep_len[g]); ep_len[g] = 0
        obs = torch.as_tensor(env.s, dtype=torch.float32)
        if t >= start:
            recent = torch.minimum(window, torch.tensor(min(t + 1, buffer)))       # sample among the most recent
            idx = (i - (torch.rand(G, batch, generator=gen) * recent[:, None]).long()) % buffer
            s, a_, s2, d = S[ar, idx], A[ar, idx], S2[ar, idx], Dn[ar, idx]
            with torch.no_grad():
                q_next = q_values(target, s2, dueling)
                choice = torch.where(double[:, None], q_values(params, s2, dueling).argmax(2), q_next.argmax(2))
                y = 1 + GAMMA * (1 - d) * q_next.gather(2, choice[..., None])[..., 0]
            q = q_values(params, s, dueling).gather(2, a_[..., None])[..., 0]
            loss = (torch.nn.functional.smooth_l1_loss(q, y, reduction="none").mean(1)).sum()
            grads = torch.autograd.grad(loss, params)
            with torch.no_grad():
                k = t - start + 1
                for p, g_, m_, v_ in zip(params, grads, m, v):
                    m_.mul_(0.9).add_(0.1 * g_); v_.mul_(0.999).add_(0.001 * g_ * g_)
                    p.sub_(lr_g * (m_ / (1 - 0.9 ** k)) / ((v_ / (1 - 0.999 ** k)).sqrt() + 1e-8))
                sync = ((t + 1) % period == 0)[:, None, None]
                soft = (tau > 0)[:, None, None]; tau_ = tau[:, None, None].float()
                for p, pt in zip(params, target):
                    pt.copy_(torch.where(soft, (1 - tau_) * pt + tau_ * p, torch.where(sync, p, pt)))
        if (t + 1) % 1000 == 0:
            curve.append([np.mean(l[-10:]) if l else 0.0 for l in lengths])
            with torch.no_grad():                                  # the predicted value of start states
                values.append(q_values(params, starts, dueling).max(2).values.mean(1).numpy())
    shape = (len(configs), seeds, -1)                                       # configs x seeds x checkpoints
    return np.array(curve).T.reshape(shape), np.array(values).T.reshape(shape)


# ---------------------------------------------------------------- part 2: DQN on MinAtar
def minatar_dqn(args):
    """One DQN or double DQN agent on a MinAtar game; returns training curve and a greedy evaluation."""
    game, double, seed, steps = args
    from minatar import Environment
    torch.set_num_threads(1); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    env = Environment(game, sticky_action_prob=0.1); env.seed(seed)
    actions = env.minimal_action_set(); nA, C = len(actions), env.state_shape()[2]
    net = lambda: torch.nn.Sequential(torch.nn.Conv2d(C, 16, 3), torch.nn.ReLU(), torch.nn.Flatten(),
                                      torch.nn.Linear(16 * 8 * 8, 128), torch.nn.ReLU(), torch.nn.Linear(128, nA))
    q, tgt = net(), net(); tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=2.5e-4)
    N = 100000
    S, S2 = np.zeros((N, C, 10, 10), np.bool_), np.zeros((N, C, 10, 10), np.bool_)
    A, R, D = np.zeros(N, np.int64), np.zeros(N, np.float32), np.zeros(N, np.float32)
    obs = lambda: torch.as_tensor(env.state().transpose(2, 0, 1)[None], dtype=torch.float32)
    env.reset(); s = env.state().transpose(2, 0, 1).copy(); ret, rets, curve = 0.0, [], []
    for t in range(steps):
        if rng.random() < max(0.1, 1 - 0.9 * t / 100000):
            a = int(rng.integers(nA))
        else:
            with torch.no_grad():
                a = q(torch.as_tensor(s[None], dtype=torch.float32)).argmax().item()
        r, term = env.act(actions[a]); s2 = env.state().transpose(2, 0, 1).copy()
        i = t % N; S[i], A[i], R[i], S2[i], D[i] = s, a, r, s2, term
        ret += r; s = s2
        if term:
            rets.append(ret); ret = 0.0; env.reset(); s = env.state().transpose(2, 0, 1).copy()
        if t >= 5000:
            idx = rng.integers(min(t + 1, N), size=32)
            x2 = torch.as_tensor(S2[idx], dtype=torch.float32)
            with torch.no_grad():
                q_next = tgt(x2)
                choice = q(x2).argmax(1, keepdim=True) if double else q_next.argmax(1, keepdim=True)
                y = torch.as_tensor(R[idx]) + GAMMA * (1 - torch.as_tensor(D[idx])) * q_next.gather(1, choice).squeeze(1)
            qa = q(torch.as_tensor(S[idx], dtype=torch.float32)).gather(1, torch.as_tensor(A[idx])[:, None]).squeeze(1)
            loss = torch.nn.functional.smooth_l1_loss(qa, y); opt.zero_grad(); loss.backward(); opt.step()
        if (t + 1) % 1000 == 0:
            tgt.load_state_dict(q.state_dict())
        if (t + 1) % 25000 == 0:
            curve.append(np.mean(rets[-50:]))
    predicted, discounted, totals = [], [], []                  # greedy evaluation, with epsilon = 0.01
    for ep in range(20):
        env.reset(); total, disc, k = 0.0, 0.0, 0
        with torch.no_grad():
            predicted.append(q(obs()).max().item())
        while True:
            with torch.no_grad():
                a = int(rng.integers(nA)) if rng.random() < 0.01 else q(obs()).argmax().item()
            r, term = env.act(actions[a]); total += r; disc += GAMMA ** k * r; k += 1
            if term or k >= 10000:
                break
        discounted.append(disc); totals.append(total)
    return curve, np.mean(totals), np.mean(predicted), np.mean(discounted)


if __name__ == "__main__":
    print("=== Part 1: DQN variants on CartPole, 4 seeds each, trained together for 100,000 steps ===")
    base = dict(lr=3e-4, period=250)
    configs = [base, {**base, "period": 1}, {**base, "replay": 64}, {**base, "replay": 5000},
               {**base, "double": True}, {**base, "dueling": True}]
    names = ["DQN (target copied every 250 steps)", "  no target network", "  replay of the last 64 transitions only",
             "  replay of the last 5,000 transitions only", "double DQN", "dueling DQN"]
    curves, values = batched_dqn(configs, steps=100000)
    print("  average length of the last 10 training episodes (at most 500), mean over seeds, and the fraction of the")
    print("  last 50 checkpoints (one per 1,000 steps) at which it was at least 475, for each seed")
    print("  " + " " * 44 + "steps: 25,000  50,000  75,000 100,000   at 475 or more")
    for name, c in zip(names, curves):
        print(f"  {name:46s}  " + "".join(f"{c[:, k].mean():8.1f}" for k in (24, 49, 74, 99))
              + "   " + " ".join(f"{(row[50:] >= 475).mean():.2f}" for row in c))
    print("\n  predicted value of the start states, max_a q(s, a), mean over 256 start states, for each seed")
    print("  (a reward of 1 per step and gamma = 0.99 bound every true value by 100)")
    print("  " + " " * 44 + "steps:  25,000             100,000     largest over training")
    for name, v in zip(names, values):
        print(f"  {name:46s}  " + " ".join(f"{x:5.0f}" if x < 1e4 else f"{x:5.0e}" for x in v[:, 24]) + "   "
              + " ".join(f"{x:5.0f}" if x < 1e4 else f"{x:5.0e}" for x in v[:, 99]) + "   "
              + " ".join(f"{x:5.0f}" if x < 1e4 else f"{x:5.0e}" for x in v.max(1)))
    print(f"  (time so far {time.time() - t0:.0f} s)")

    print("\n=== Part 2: DQN and double DQN on MinAtar Space Invaders, 2 seeds each, 100,000 steps ===")
    jobs = [("space_invaders", dbl, seed, 100000) for dbl in (False, True) for seed in (0, 1)]
    with mp.get_context("fork").Pool(2) as pool:
        results = pool.map(minatar_dqn, jobs)
    for k, name in enumerate(["DQN", "double DQN"]):
        res = results[2 * k: 2 * k + 2]
        curve = np.mean([r[0] for r in res], 0)
        print(f"  {name:10s} training return (last 50 episodes, mean of seeds) at 25k, 50k, 75k, 100k steps: "
              + " ".join(f"{x:5.1f}" for x in curve))
        for seed, (_, total, pred, disc) in enumerate(res):
            print(f"  {'':10s} seed {seed}: greedy return {total:5.1f}; value of the start state: predicted {pred:5.2f},"
                  f" actual discounted return {disc:5.2f}, overestimate {pred - disc:+5.2f}")
    print(f"\ntotal time {time.time() - t0:.0f} s")
