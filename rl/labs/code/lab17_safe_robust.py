"""Lab 17 reference solution: constrained, shielded, and robust reinforcement learning.

Part 1: PPO on a navigation task with a hazard, with fixed penalties and Lagrangian multipliers (budget in expectation).
Part 2: a hard constraint: Lagrangian learning with a zero budget versus a model-based safety filter (shield).
Part 3: sim-to-real on a pendulum: nominal training, domain randomization, history, and RMA-style adaptation.

Run from this folder:  python lab17_safe_robust.py
It prints the results quoted on the lab page. Takes about 16 minutes on two cores.
"""
import multiprocessing as mp
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import torch
import torch.nn as nn

t0 = time.time()


# ============================================================ PPO shared by all parts
class ActorCritic(nn.Module):
    """Gaussian policy (state-independent standard deviation) with a reward critic and, optionally, a cost critic."""

    def __init__(self, obs_dim, act_dim, log_std=-0.5):
        super().__init__()
        mlp = lambda o: nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, o))
        self.pi, self.vr, self.vc = mlp(act_dim), mlp(1), mlp(1)
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std))
        with torch.no_grad():                               # start with actions near zero
            self.pi[-1].weight.mul_(0.01); self.pi[-1].bias.zero_()

    def dist(self, o):
        return torch.distributions.Normal(self.pi(o), self.log_std.exp())


def gae(r, v, v_next, done, gamma, lam):
    """Generalized advantage estimates over a (T, n) batch; v_next is 0 after a terminal step."""
    adv, last = np.zeros_like(r), 0.0
    for t in range(len(r) - 1, -1, -1):
        last = r[t] + gamma * v_next[t] - v[t] + gamma * lam * (1 - done[t]) * last
        adv[t] = last
    return adv


def ppo_update(ac, opt, O, A, LP, adv, ret_r, ret_c=None, epochs=10, mb=512):
    N = len(O)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    for _ in range(epochs):
        perm = torch.randperm(N)
        for k in range(0, N, mb):
            i = perm[k:k + mb]
            d = ac.dist(O[i]); ratio = (d.log_prob(A[i]).sum(-1) - LP[i]).exp()
            loss = -torch.min(ratio * adv[i], ratio.clamp(0.8, 1.2) * adv[i]).mean()
            loss = loss + 0.5 * ((ac.vr(O[i])[:, 0] - ret_r[i]) ** 2).mean()
            if ret_c is not None:
                loss = loss + 0.5 * ((ac.vc(O[i])[:, 0] - ret_c[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(ac.parameters(), 0.5); opt.step()


# ============================================================ Parts 1 and 2: navigation with a hazard
START, GOAL, HAZARD_R, NAV_T = np.array([-1.0, 0.0]), np.array([1.0, 0.0]), 0.8, 60


class Nav:
    """n point robots: v <- 0.8 v + 0.2 a, p <- p + 0.1 v, |a_i| <= 1. Start near (-1, 0), goal (1, 0) (reached within
    0.1), a circular hazard of radius 0.8 at the origin. Reward: 10 x progress toward the goal - 0.5 per step + 10 on
    arrival. Cost: 1 per step inside the hazard. Episodes end on arrival or after 60 steps."""

    def __init__(self, n, seed):
        self.n, self.rng = n, np.random.default_rng(seed)
        self.p, self.v, self.t = np.zeros((n, 2)), np.zeros((n, 2)), np.zeros(n, int)
        self.reset(np.arange(n))

    def reset(self, idx):
        self.p[idx] = START + self.rng.uniform(-0.1, 0.1, (len(idx), 2)); self.v[idx] = 0; self.t[idx] = 0

    def obs(self):
        return np.concatenate([self.p, self.v, (self.t / NAV_T)[:, None]], 1).astype(np.float32)

    def step(self, a, shield=None):
        a = np.clip(a, -1, 1)
        if shield is not None:
            a = shield(self.p, self.v, a)
        d0 = np.linalg.norm(self.p - GOAL, axis=1)
        self.v = 0.8 * self.v + 0.2 * a; self.p = self.p + 0.1 * self.v; self.t += 1
        d1 = np.linalg.norm(self.p - GOAL, axis=1)
        reached = d1 < 0.1
        r = 10 * (d0 - d1) - 0.5 + 10 * reached
        c = (np.linalg.norm(self.p, axis=1) < HAZARD_R).astype(float)
        done = reached | (self.t >= NAV_T)
        self.reset(np.flatnonzero(done))
        return r, c, done, reached


def make_shield(radius, damping=0.8, horizon=6):
    """A safety filter built on a model of the dynamics (with the given damping) and of the hazard (with the given
    radius): keep the proposed action if, after it, braking for `horizon` steps keeps the robot outside the hazard in
    the model; otherwise brake."""

    def brake(v):
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        return -v / np.maximum(speed, 1e-6) * np.minimum(1.0, speed / 0.2)

    def shield(p, v, a):
        vv = damping * v + 0.2 * a; pp = p + 0.1 * vv; ok = np.linalg.norm(pp, axis=1) >= radius
        for _ in range(horizon):
            vv = damping * vv + 0.2 * brake(vv); pp = pp + 0.1 * vv; ok &= np.linalg.norm(pp, axis=1) >= radius
        return np.where(ok[:, None], a, brake(v))

    return shield


def nav_run(cfg):
    """PPO (64 robots x 60 steps per iteration, 200 iterations) with a multiplier rule: 'none', 'penalty', 'gradient'
    (lambda += eta (J_c - budget)), or 'pid' (lambda = kp (J_c - budget) + integral of ki (J_c - budget))."""
    torch.set_num_threads(1)
    rule, seed, budget = cfg["rule"], cfg["seed"], cfg.get("budget", 5.0)
    shield = make_shield(*cfg["shield"]) if "shield" in cfg else None
    torch.manual_seed(seed); env = Nav(64, seed); ac = ActorCritic(5, 2)
    opt = torch.optim.Adam(ac.parameters(), lr=3e-4)
    lam = cfg.get("lam", 0.0); integral = 0.0; hazard_steps = 0.0; overshoot = 0.0
    ep_len, ep_cost, history = np.zeros(64), np.zeros(64), []
    for it in range(200):
        buf = {k: [] for k in ("o", "a", "lp", "r", "c", "d", "vr", "vc")}
        lens, costs = [], []
        for t in range(NAV_T):
            o = env.obs()
            with torch.no_grad():
                ot = torch.as_tensor(o); d = ac.dist(ot); a = d.sample()
                lp, vr, vc = d.log_prob(a).sum(-1), ac.vr(ot)[:, 0], ac.vc(ot)[:, 0]
            r, c, done, reached = env.step(a.numpy(), shield)
            hazard_steps += c.sum(); ep_len += 1; ep_cost += c
            lens += list(ep_len[done]); costs += list(ep_cost[done]); ep_len[done] = 0; ep_cost[done] = 0
            for k, x in zip(buf, (o, a.numpy(), lp.numpy(), r, c, done, vr.numpy(), vc.numpy())):
                buf[k].append(x)
        with torch.no_grad():
            ot = torch.as_tensor(env.obs()); last_r, last_c = ac.vr(ot)[:, 0].numpy(), ac.vc(ot)[:, 0].numpy()
        R, C, D, VR, VC = (np.array(buf[k]) for k in ("r", "c", "d", "vr", "vc"))
        adv_r = gae(R, VR, np.concatenate([VR[1:], last_r[None]]) * (1 - D), D, 0.99, 0.95)
        adv_c = gae(C, VC, np.concatenate([VC[1:], last_c[None]]) * (1 - D), D, 0.99, 0.95)
        J_c = float(np.mean(costs))                          # average hazard steps per finished episode
        err = J_c - budget
        if rule == "gradient":
            lam = max(0.0, lam + cfg["eta"] * err)
        elif rule == "pid":
            integral = max(0.0, integral + cfg["ki"] * err); lam = max(0.0, cfg["kp"] * err + integral)
        overshoot += max(err, 0.0)
        adv = (adv_r - lam * adv_c) / (1 + lam)
        flat = lambda x: torch.as_tensor(np.array(x).reshape(NAV_T * 64, -1), dtype=torch.float32)
        ppo_update(ac, opt, flat(buf["o"]), flat(buf["a"]), flat(buf["lp"])[:, 0], flat(adv)[:, 0],
                   flat(adv_r + VR)[:, 0], flat(adv_c + VC)[:, 0])
        history.append((np.mean(lens), J_c, lam))
    h = np.array(history)
    return dict(length=h[-20:, 0].mean(), cost=h[-20:, 1].mean(), lam=h[-1, 2], hazard_steps=hazard_steps,
                overshoot=overshoot)


# ============================================================ Part 3: pendulum, simulation to reality
DT, PEND_T, K = 0.05, 200, 6
M_LO, M_HI, D_MAX = 0.6, 2.4, 3                              # randomization: mass on [0.6, 2.4], delay 0..3 steps


class Pendulum:
    """Vectorized Pendulum-v1 (swing up and balance, torque |u| <= 2) with per-robot mass m, which scales the effect of
    the torque, and a delay of 0 to 7 steps (50 ms each) before the torque is applied. With a sampler, each episode
    draws new parameters. Keeps the last K observations and applied commands for policies that use history."""

    def __init__(self, n, seed, m=1.0, delay=0, sampler=None):
        self.n, self.rng, self.sampler = n, np.random.default_rng(seed), sampler
        self.m, self.delay = np.full(n, m, float), np.full(n, delay, int)
        self.th, self.om, self.t = np.zeros(n), np.zeros(n), np.zeros(n, int)
        self.u_past, self.hist = np.zeros((n, 8)), np.zeros((n, K, 4), np.float32)
        self.reset(np.arange(n))

    def reset(self, idx):
        if self.sampler is not None:
            self.m[idx], self.delay[idx] = self.sampler(self.rng, len(idx))
        self.th[idx] = self.rng.uniform(-np.pi, np.pi, len(idx)); self.om[idx] = self.rng.uniform(-1, 1, len(idx))
        self.t[idx] = 0; self.u_past[idx] = 0; self.hist[idx] = 0

    def obs(self):
        return np.stack([np.cos(self.th), np.sin(self.th), self.om / 8], 1).astype(np.float32)

    def params(self):
        """The true parameters, scaled to [-1, 1] over the randomization range (privileged information)."""
        z = np.stack([(np.log(self.m) - np.log(M_LO)) / (np.log(M_HI) - np.log(M_LO)), self.delay / D_MAX], 1)
        return (2 * z - 1).astype(np.float32)

    def features(self, kind):
        o = self.obs()
        if kind == "history":
            return np.concatenate([o, self.hist.reshape(self.n, -1)], 1)
        return o

    def step(self, u):
        u = np.clip(u, -2, 2)
        o = self.obs()
        self.u_past = np.roll(self.u_past, 1, 1); self.u_past[:, 0] = u
        applied = self.u_past[np.arange(self.n), self.delay]
        angle = (self.th + np.pi) % (2 * np.pi) - np.pi
        r = -(angle ** 2 + 0.1 * self.om ** 2 + 0.001 * u ** 2)
        self.om = np.clip(self.om + (15 * np.sin(self.th) + 3 / self.m * applied) * DT, -8, 8)
        self.th = self.th + self.om * DT; self.t += 1
        self.hist = np.roll(self.hist, 1, 1); self.hist[:, 0, :3] = o; self.hist[:, 0, 3] = u / 2
        done = self.t >= PEND_T
        self.reset(np.flatnonzero(done))
        return r, done


def randomized(rng, k):
    return np.exp(rng.uniform(np.log(M_LO), np.log(M_HI), k)), rng.integers(0, D_MAX + 1, k)


def pend_train(kind, seed, iters, m=1.0, delay=0, sampler=None):
    """PPO on the pendulum, 32 robots x 200 steps per iteration. kind: 'plain' (angle features), 'history' (plus the last
    6 observations and commands), or 'privileged' (plus the true parameters: the teacher of RMA)."""
    torch.manual_seed(seed); env = Pendulum(32, seed, m, delay, sampler)
    feat = lambda: (np.concatenate([env.obs(), env.params()], 1) if kind == "privileged" else env.features(kind))
    ac = ActorCritic(feat().shape[1], 1, log_std=0.0)
    opt = torch.optim.Adam(ac.parameters(), lr=1e-3)
    for it in range(iters):
        O, A, LP, R, V, D = [], [], [], [], [], []
        for t in range(PEND_T):
            o = feat()
            with torch.no_grad():
                ot = torch.as_tensor(o); d = ac.dist(ot); a = d.sample()
                lp, v = d.log_prob(a).sum(-1), ac.vr(ot)[:, 0]
            r, done = env.step(2 * a[:, 0].numpy())
            for lst, x in zip((O, A, LP, R, V, D), (o, a.numpy(), lp.numpy(), r / 10, v.numpy(), done)):
                lst.append(x)
        with torch.no_grad():
            last = ac.vr(torch.as_tensor(feat()))[:, 0].numpy()
        R, V, D = np.array(R), np.array(V), np.array(D)
        # episodes end only by the time limit: bootstrap from the value of the last state instead of a terminal zero
        v_next = np.where(D, V, np.concatenate([V[1:], last[None]]))
        adv = gae(R, V, v_next, D, 0.99, 0.95)
        flat = lambda x: torch.as_tensor(np.array(x).reshape(PEND_T * 32, -1), dtype=torch.float32)
        ppo_update(ac, opt, flat(O), flat(A), flat(LP)[:, 0], flat(adv)[:, 0], flat(adv + V)[:, 0], mb=400)
    return ac


def train_adapter(teacher, seed, rounds=20):
    """RMA's adaptation module: a network that estimates the privileged parameters from the last 6 observations and
    commands, trained by regression on rollouts in which the teacher acts on the module's own estimates."""
    torch.manual_seed(seed)
    phi = nn.Sequential(nn.Linear(4 * K, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2))
    opt = torch.optim.Adam(phi.parameters(), lr=1e-3)
    env = Pendulum(64, seed + 1, sampler=randomized)
    X, Y = [], []
    for rd in range(rounds):
        for t in range(PEND_T):
            h = env.hist.reshape(64, -1).copy()
            with torch.no_grad():
                z = phi(torch.as_tensor(h)) if rd > 0 else torch.zeros(64, 2)
                a = teacher.pi(torch.cat([torch.as_tensor(env.obs()), z], 1))[:, 0].numpy()
            X.append(h); Y.append(env.params())
            env.step(2 * a)
        Xt, Yt = torch.as_tensor(np.concatenate(X)), torch.as_tensor(np.concatenate(Y))
        for _ in range(300):
            i = torch.randint(0, len(Xt), (512,))
            loss = ((phi(Xt[i]) - Yt[i]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return phi


def pend_eval(policy, kind, m, delay, n=100, seed=123):
    """Average return of the deterministic policy over 100 episodes on a fixed 'real' system."""
    env = Pendulum(n, seed, m, delay); total = np.zeros(n)
    for t in range(PEND_T):
        with torch.no_grad():
            o = torch.as_tensor(env.obs())
            if kind == "rma":
                ac, phi = policy; x = torch.cat([o, phi(torch.as_tensor(env.hist.reshape(n, -1)))], 1)
            elif kind == "privileged":
                x = torch.cat([o, torch.as_tensor(env.params())], 1); ac = policy
            else:
                x = torch.as_tensor(env.features(kind)); ac = policy
            u = 2 * ac.pi(x)[:, 0].numpy()
        r, _ = env.step(u); total += r
    return total.mean()


TESTS = [(1.0, 0), (1.0, 3), (0.6, 3), (2.4, 0), (2.4, 3), (1.0, 5), (3.2, 1)]


def pend_run(seed):
    torch.set_num_threads(1)
    nominal = pend_train("plain", seed, 40)
    rand = pend_train("plain", seed, 150, sampler=randomized)
    hist = pend_train("history", seed, 150, sampler=randomized)
    teacher = pend_train("privileged", seed, 150, sampler=randomized)
    phi = train_adapter(teacher, seed)
    rows = []
    for m, d in TESTS:
        oracle = pend_train("plain", seed, 40, m=m, delay=d)
        rows.append([pend_eval(nominal, "plain", m, d), pend_eval(rand, "plain", m, d), pend_eval(hist, "history", m, d),
                     pend_eval(teacher, "privileged", m, d), pend_eval((teacher, phi), "rma", m, d),
                     pend_eval(oracle, "plain", m, d)])
    return np.array(rows)


def dispatch(job):
    kind, cfg = job
    return nav_run(cfg) if kind == "nav" else pend_run(cfg)


if __name__ == "__main__":
    part1 = [("unconstrained", dict(rule="none")), ("fixed penalty 0.3", dict(rule="penalty", lam=0.3)),
             ("fixed penalty 1", dict(rule="penalty", lam=1.0)), ("Lagrangian, step 0.05", dict(rule="gradient", eta=0.05)),
             ("Lagrangian, step 0.5", dict(rule="gradient", eta=0.5)), ("PID, kp 0.3, ki 0.05", dict(rule="pid", kp=0.3, ki=0.05))]
    part2 = [("Lagrangian, budget 0", dict(rule="gradient", eta=0.05, budget=0.0)),
             ("shield, exact model", dict(rule="none", shield=(0.8, 0.8))),
             ("shield, radius 0.7 (true 0.8)", dict(rule="none", shield=(0.7, 0.8))),
             ("shield, damping 0.7 (true 0.8)", dict(rule="none", shield=(0.8, 0.7)))]
    jobs = [("pend", 0), ("pend", 1)]
    jobs += [("nav", dict(cfg, seed=s)) for _, cfg in part1 + part2 for s in (0, 1)]
    with mp.get_context("fork").Pool(2) as pool:
        out = pool.map(dispatch, jobs, chunksize=1)
    pend, nav = out[:2], out[2:]
    mean = lambda rs, k: np.mean([r[k] for r in rs])

    print("=== Part 1: a budget of 5 hazard steps per episode, in expectation (PPO, 200 iterations, 2 seeds) ===")
    print("  final behavior: average of the last 20 iterations; overshoot: sum over iterations of (cost - 5) when positive")
    print("                               steps to goal   hazard steps   multiplier   overshoot in training")
    for i, (name, _) in enumerate(part1):
        rs = nav[2 * i:2 * i + 2]
        lam = "-" if name == "unconstrained" else f"{mean(rs, 'lam'):.2f}"
        print(f"  {name:28s}{mean(rs, 'length'):11.1f}{mean(rs, 'cost'):15.2f}{lam:>13s}{mean(rs, 'overshoot'):18.0f}")

    print("\n=== Part 2: a hard constraint, never enter the hazard (2 seeds) ===")
    print("                                    steps to goal   hazard steps (final)   hazard steps during training")
    for i, (name, _) in enumerate(part2):
        rs = nav[2 * len(part1) + 2 * i:2 * len(part1) + 2 * i + 2]
        print(f"  {name:32s}{mean(rs, 'length'):10.1f}{mean(rs, 'cost'):20.2f}{mean(rs, 'hazard_steps'):27,.0f}")

    print("\n=== Part 3: a pendulum trained in simulation, tested on 'real' systems (return, higher is better, 2 seeds) ===")
    print("  randomization range: mass 0.6 to 2.4 (the torque's effect scales as 1/m), delay 0 to 3 steps (0 to 150 ms)")
    print("  real system          nominal   randomized   rand. + history   teacher (true params)   RMA   oracle")
    table = np.mean(pend, 0)
    for (m, d), row in zip(TESTS, table):
        inside = "in range " if (M_LO <= m <= M_HI and d <= D_MAX) else "outside  "
        print(f"  m {m:3.1f}, {50 * d:3d} ms {inside}" + f"{row[0]:6.0f}{row[1]:11.0f}{row[2]:15.0f}{row[3]:20.0f}{row[4]:10.0f}{row[5]:8.0f}")
    print(f"\n(total time {time.time() - t0:.0f} s)")
