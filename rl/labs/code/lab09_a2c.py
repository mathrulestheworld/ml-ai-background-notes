"""Lab 9 reference solution: advantage actor-critic with parallel and stale actors on the cart-pole.

Run from this folder:  python lab09_a2c.py
It prints the results quoted on the lab page. Takes about four minutes on one core.
"""
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import torch

torch.set_num_threads(1)
t0 = time.time()


class VecCartPole:
    """n independent copies of Gymnasium's CartPole-v1 (checked against Gymnasium in Lab 6), each resetting itself.
    step returns the true next states (before any reset) and the termination and truncation flags."""
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
        trunc = (self.t >= 500) & ~term
        done = term | trunc
        true_next = s2.copy()
        s2[done] = self.rng.uniform(-0.05, 0.05, (done.sum(), 4)); self.t[done] = 0; self.s = s2
        return true_next, term, trunc


def init_mlp(G, sizes, gen, last_scale):
    """One MLP per agent, weights of shape (agents, inputs, outputs); the last layer scaled down."""
    params = []
    for k, (i, o) in enumerate(zip(sizes[:-1], sizes[1:])):
        bound = 1 / np.sqrt(i) * (last_scale if k == len(sizes) - 2 else 1.0)
        params += [(torch.rand(G, i, o, generator=gen) * 2 - 1) * bound, torch.zeros(G, 1, o)]
    return [p.requires_grad_() for p in params]


def mlp(params, x):
    for k in range(0, len(params), 2):
        x = torch.baddbmm(params[k + 1], x, params[k])
        if k < len(params) - 2:
            x = torch.tanh(x)
    return x


SCALE = torch.tensor([2.4, 2.0, 0.21, 2.0])                    # rough ranges of the state's components


def a2c(configs, seeds, steps=384000, envs=16, T=16, gamma=0.99, checkpoint=6400):
    """Batched A2C: each (config, seed) pair is an agent with its own actor and critic networks (two tanh layers of
    64 units), its own `envs` cart-poles, and its own Adam statistics; all agents are trained in lockstep.
    Config keys and defaults: lam=0.95 (GAE), ent=0.01 (entropy coefficient), lr=1e-3, vscale=10 (the critic
    predicts vscale times its network's output), lag=0 (the actors use the actor's parameters from `lag` updates
    earlier), correction='none' | 'vtrace' (rho_bar = c_bar = 1) | 'clip' (PPO's clipped objective, epsilon = 0.2).
    Returns the average length of each agent's last 10 episodes at every checkpoint: configs x seeds x checkpoints."""
    G = len(configs) * seeds
    get = lambda k, d: [c.get(k, d) for c in configs for _ in range(seeds)]
    lam = torch.tensor(get("lam", 0.95))[:, None, None]
    ent, lr = torch.tensor(get("ent", 0.01)), torch.tensor(get("lr", 1e-3))[:, None, None]
    vscale = torch.tensor(get("vscale", 10.0))[:, None, None]
    vtrace = torch.tensor([c == "vtrace" for c in get("correction", "none")])[:, None, None]
    clip = torch.tensor([c == "clip" for c in get("correction", "none")])[:, None, None]
    lags = get("lag", 0)
    gen = torch.Generator().manual_seed(0); rng = np.random.default_rng(0)
    actor, critic = init_mlp(G, [4, 64, 64, 2], gen, 0.01), init_mlp(G, [4, 64, 64, 1], gen, 0.01)
    params = actor + critic
    m, v = [torch.zeros_like(p) for p in params], [torch.zeros_like(p) for p in params]
    history = [[p.detach().clone() for p in actor]]                  # the actor's recent parameters, newest last
    env = VecCartPole(G * envs, rng)
    ep_len, lengths, curve = np.zeros(G * envs), [[] for _ in range(G)], []
    for it in range(steps // (envs * T)):
        # acting: agent g's actors use its parameters from lags[g] updates ago
        behavior = [torch.stack([history[max(0, len(history) - 1 - lags[g])][k][g] for g in range(G)])
                    for k in range(len(actor))]
        S, A, S2, TERM, TRUNC, LOGMU = [], [], [], [], [], []
        for t in range(T):
            s = torch.as_tensor(env.s, dtype=torch.float32).view(G, envs, 4) / SCALE
            with torch.no_grad():
                logits = mlp(behavior, s)
                a = (torch.rand(G, envs, generator=gen) < torch.softmax(logits, -1)[..., 1]).long()
                LOGMU.append(torch.log_softmax(logits, -1).gather(-1, a[..., None])[..., 0])
            nxt, term, trunc = env.step(a.view(-1).numpy())
            ep_len += 1
            for i in np.flatnonzero(term | trunc):
                lengths[i // envs].append(ep_len[i]); ep_len[i] = 0
            S.append(s); A.append(a)
            S2.append(torch.as_tensor(nxt, dtype=torch.float32).view(G, envs, 4) / SCALE)
            TERM.append(torch.as_tensor(term, dtype=torch.float32).view(G, envs))
            TRUNC.append(torch.as_tensor(trunc, dtype=torch.float32).view(G, envs))
        S, S2, A, LOGMU = torch.stack(S, 2), torch.stack(S2, 2), torch.stack(A, 2), torch.stack(LOGMU, 2)
        TERM, TRUNC = torch.stack(TERM, 2), torch.stack(TRUNC, 2)             # all (agents, envs, T)
        # learning
        logp_all = torch.log_softmax(mlp(actor, S.reshape(G, -1, 4)).view(G, envs, T, 2), -1)
        logpi = logp_all.gather(-1, A[..., None])[..., 0]
        entropy = -(logp_all.exp() * logp_all).sum(-1)
        values = vscale * mlp(critic, S.reshape(G, -1, 4)).view(G, envs, T)
        with torch.no_grad():
            v_next = vscale * mlp(critic, S2.reshape(G, -1, 4)).view(G, envs, T) * (1 - TERM)   # 0 at terminations
            v_old = values.detach()
            ratio = torch.exp(logpi.detach() - LOGMU)
            rho = torch.where(vtrace, ratio.clamp(max=1.0), torch.ones_like(ratio))
            c = lam * rho                                                  # GAE when rho = 1, V-trace otherwise
            end = TERM + TRUNC
            delta = rho * (1.0 + gamma * v_next - v_old)                   # the reward is 1 at every step
            acc, gap = torch.zeros(G, envs), torch.zeros(G, envs, T)
            for t in reversed(range(T)):
                acc = delta[..., t] + gamma * c[..., t] * (1 - end[..., t]) * acc
                gap[..., t] = acc
            targets = v_old + gap                                          # TD(lambda) or V-trace targets
            later = torch.cat([targets[..., 1:], torch.zeros(G, envs, 1)], 2)
            boot = torch.where(end.bool() | (torch.arange(T) == T - 1), v_next, later)
            adv = torch.where(vtrace, rho * (1.0 + gamma * boot - v_old), gap)
        ratio_live = torch.exp(logpi - LOGMU)
        surrogate = torch.minimum(ratio_live * adv, ratio_live.clamp(0.8, 1.2) * adv)
        pg = -torch.where(clip, surrogate, logpi * adv).mean((1, 2))
        loss = (pg - ent * entropy.mean((1, 2)) + 0.5 * ((values - targets) ** 2).mean((1, 2))).sum()
        grads = torch.autograd.grad(loss, params)
        with torch.no_grad():                                              # Adam, one step size per agent
            for p, g_, m_, v_ in zip(params, grads, m, v):
                m_.mul_(0.9).add_(0.1 * g_); v_.mul_(0.999).add_(0.001 * g_ * g_)
                p.sub_(lr * (m_ / (1 - 0.9 ** (it + 1))) / ((v_ / (1 - 0.999 ** (it + 1))).sqrt() + 1e-8))
        history.append([p.detach().clone() for p in actor]); history = history[-(max(lags) + 1):]
        if (it + 1) * envs * T // checkpoint > it * envs * T // checkpoint:   # crossed a multiple of `checkpoint`
            curve.append([np.mean(l[-10:]) if l else 0.0 for l in lengths])
    return np.array(curve).T.reshape(len(configs), seeds, -1)


def report(names, curves, at=(64000, 128000, 256000, 384000), checkpoint=6400):
    print("  " + " " * 40 + "steps: " + "".join(f"{s // 1000:6d}k" for s in at) + "   solved   final length of each seed")
    for name, c in zip(names, curves):
        final = c[:, -5:].mean(1)                                    # the last 32,000 steps
        print(f"  {name:44s}" + "".join(f"{c[:, s // checkpoint - 1].mean():7.0f}" for s in at)
              + f"   {int((final >= 475).sum())} of {len(final)}   " + " ".join(f"{x:3.0f}" for x in final))


if __name__ == "__main__":
    print("=== Part 1: A2C on the cart-pole, 16 environments, rollouts of 16 steps, 6 seeds ===")
    print("  average length of the last 10 episodes (at most 500), mean over seeds; 'solved' counts the seeds whose")
    print("  average over the last 32,000 steps is at least 475")
    base = dict(lam=0.95, ent=0.01, vscale=10.0)
    configs = [base, {**base, "vscale": 1.0}, {**base, "vscale": 100.0}, {**base, "lam": 0.0}, {**base, "lam": 0.8},
               {**base, "lam": 1.0}, {**base, "ent": 0.0}, {**base, "ent": 0.1}]
    names = ["A2C (lambda 0.95, entropy 0.01, critic x10)", "  critic x1", "  critic x100", "  lambda = 0",
             "  lambda = 0.8", "  lambda = 1", "  entropy coefficient 0", "  entropy coefficient 0.1"]
    report(names, a2c(configs, seeds=6))
    print(f"  (time so far {time.time() - t0:.0f} s)")

    print("\n=== Part 2: stale actors, which act with the parameters of `lag` updates earlier ===")
    configs = [dict(lag=L, correction=c) for L in (0, 8, 32) for c in ("none", "vtrace", "clip")]
    names = [f"lag {L:2d}, " + {"none": "no correction", "vtrace": "V-trace", "clip": "clipped objective"}[c]
             for L in (0, 8, 32) for c in ("none", "vtrace", "clip")]
    report(names, a2c(configs, seeds=6))
    print(f"  (time so far {time.time() - t0:.0f} s)")

    print("\n=== Part 3: how many environments? The same 384,000 steps, rollouts of 16 steps, 6 seeds ===")
    for envs in (4, 16, 64):
        t1 = time.time()
        curves = a2c([dict(), dict(lr=4e-3)], seeds=6, envs=envs)
        print(f"  {envs} environments, {384000 // (envs * 16):,} updates  ({time.time() - t1:.0f} s)")
        report(["    step size 1e-3", "    step size 4e-3"], curves)
    print(f"\ntotal time {time.time() - t0:.0f} s")
