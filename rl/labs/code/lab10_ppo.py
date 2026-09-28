"""Lab 10 reference solution: proximal policy optimization from scratch, on LunarLander and HalfCheetah.

Run from this folder:  python lab10_ppo.py
It prints the results quoted on the lab page. Takes about 45 minutes on two cores.
Needs gymnasium[box2d] and gymnasium[mujoco].
"""
import multiprocessing as mp
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

t0 = time.time()


class RunningMeanStd:
    """Running mean and variance of a stream of vectors (Chan et al.'s parallel update)."""

    def __init__(self, shape):
        self.mean, self.var, self.count = np.zeros(shape), np.ones(shape), 1e-4

    def update(self, x):
        b_mean, b_var, b = x.mean(0), x.var(0), x.shape[0]
        delta, total = b_mean - self.mean, self.count + b
        self.mean = self.mean + delta * b / total
        self.var = (self.var * self.count + b_var * b + delta ** 2 * self.count * b / total) / total
        self.count = total


def layer(i, o, std):
    lin = nn.Linear(i, o)
    nn.init.orthogonal_(lin.weight, std); nn.init.zeros_(lin.bias)
    return lin


class Agent(nn.Module):
    """Separate actor and critic, two tanh layers of 64 units each, orthogonal initialization; the actor's last
    layer is 100 times smaller than the others. Discrete actions: a softmax. Continuous: a Gaussian whose mean is
    the network's output and whose log standard deviation is a free parameter."""

    def __init__(self, n_obs, n_act, continuous):
        super().__init__()
        self.continuous = continuous
        self.critic = nn.Sequential(layer(n_obs, 64, 2 ** 0.5), nn.Tanh(), layer(64, 64, 2 ** 0.5), nn.Tanh(), layer(64, 1, 1.0))
        self.actor = nn.Sequential(layer(n_obs, 64, 2 ** 0.5), nn.Tanh(), layer(64, 64, 2 ** 0.5), nn.Tanh(), layer(64, n_act, 0.01))
        if continuous:
            self.log_std = nn.Parameter(torch.zeros(n_act))

    def dist(self, x):
        if self.continuous:
            mean = self.actor(x)
            return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))
        return torch.distributions.Categorical(logits=self.actor(x))

    def log_prob(self, d, a):
        return d.log_prob(a).sum(-1) if self.continuous else d.log_prob(a)

    def entropy(self, d):
        return d.entropy().sum(-1) if self.continuous else d.entropy()


def ppo(cfg):
    """One PPO run; returns the average return of the last 20 finished episodes at every 1/10 of the steps."""
    c = dict(env="LunarLander-v3", steps=500_000, envs=8, T=256, epochs=10, minibatches=32, lr=3e-4, gamma=0.999,
             lam=0.98, clip=0.2, ent=0.01, vf=0.5, max_grad=0.5, adv_norm=True, clip_value=False, norm_obs=False,
             norm_reward=False, seed=0)
    c.update(cfg)
    torch.set_num_threads(1); torch.manual_seed(c["seed"]); np.random.seed(c["seed"])
    env = gym.make_vec(c["env"], num_envs=c["envs"], vectorization_mode="sync",
                       vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})
    continuous = isinstance(env.single_action_space, gym.spaces.Box)
    n_obs = env.single_observation_space.shape[0]
    n_act = env.single_action_space.shape[0] if continuous else env.single_action_space.n
    agent = Agent(n_obs, n_act, continuous)
    opt = torch.optim.Adam(agent.parameters(), lr=c["lr"], eps=1e-5)
    obs_rms, ret_rms, ret_acc = RunningMeanStd(n_obs), RunningMeanStd(()), np.zeros(c["envs"])
    norm = lambda o: np.clip((o - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10) if c["norm_obs"] else o
    obs, _ = env.reset(seed=c["seed"])
    if c["norm_obs"]:
        obs_rms.update(obs)
    N, T = c["envs"], c["T"]
    iters = c["steps"] // (N * T)
    ep_ret, finished, curve, kls, clipfracs = np.zeros(N), [], [], [], []
    for it in range(iters):
        for g in opt.param_groups:                                      # step size annealed linearly to 0
            g["lr"] = c["lr"] * (1 - it / iters)
        O, A, LOGP, R, V, VNEXT, TERM, END = [], [], [], [], [], [], [], []
        for t in range(T):
            o = torch.as_tensor(norm(obs), dtype=torch.float32)
            with torch.no_grad():
                d = agent.dist(o); a = d.sample(); logp = agent.log_prob(d, a); v = agent.critic(o)[:, 0]
            a_env = a.numpy()
            if continuous:
                a_env = np.clip(a_env, env.single_action_space.low, env.single_action_space.high)
            obs2, r, term, trunc, info = env.step(a_env)
            ep_ret += r
            done = term | trunc
            for i in np.flatnonzero(done):
                finished.append(ep_ret[i]); ep_ret[i] = 0.0
            if c["norm_reward"]:                                        # scale rewards by the std of the return
                ret_acc = ret_acc * c["gamma"] + r; ret_rms.update(ret_acc); ret_acc[done] = 0.0
                r = r / np.sqrt(ret_rms.var + 1e-8)
            next_obs = obs2.copy()                                      # the true next states, before resets
            if done.any():
                for i in np.flatnonzero(info["_final_obs"]):
                    next_obs[i] = info["final_obs"][i]
            if c["norm_obs"]:
                obs_rms.update(obs2)
            with torch.no_grad():
                v_next = agent.critic(torch.as_tensor(norm(next_obs), dtype=torch.float32))[:, 0]
            O.append(o); A.append(a); LOGP.append(logp); V.append(v); VNEXT.append(v_next)
            R.append(torch.as_tensor(r, dtype=torch.float32)); TERM.append(torch.as_tensor(term, dtype=torch.float32))
            END.append(torch.as_tensor(done, dtype=torch.float32))
            obs = obs2
        V, VNEXT, R, TERM, END = map(torch.stack, (V, VNEXT, R, TERM, END))   # (T, N)
        adv, last = torch.zeros(T, N), torch.zeros(N)
        for t in reversed(range(T)):                                    # GAE, reset at every episode end
            delta = R[t] + c["gamma"] * (1 - TERM[t]) * VNEXT[t] - V[t]
            last = delta + c["gamma"] * c["lam"] * (1 - END[t]) * last
            adv[t] = last
        ret = adv + V
        O, A, LOGP = torch.cat(O), torch.cat(A), torch.cat(LOGP)
        adv, ret, V = adv.reshape(-1), ret.reshape(-1), V.reshape(-1)
        batch = N * T; mb = batch // c["minibatches"]
        for epoch in range(c["epochs"]):
            for idx in torch.randperm(batch).split(mb):
                d = agent.dist(O[idx]); logp = agent.log_prob(d, A[idx])
                ratio = torch.exp(logp - LOGP[idx])
                a_mb = adv[idx]
                if c["adv_norm"]:
                    a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg = -torch.minimum(ratio * a_mb, ratio.clamp(1 - c["clip"], 1 + c["clip"]) * a_mb).mean()
                v = agent.critic(O[idx])[:, 0]
                if c["clip_value"]:
                    v_clip = V[idx] + (v - V[idx]).clamp(-c["clip"], c["clip"])
                    vloss = 0.5 * torch.maximum((v - ret[idx]) ** 2, (v_clip - ret[idx]) ** 2).mean()
                else:
                    vloss = 0.5 * ((v - ret[idx]) ** 2).mean()
                loss = pg - c["ent"] * agent.entropy(d).mean() + c["vf"] * vloss
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), c["max_grad"]); opt.step()
        with torch.no_grad():                                           # how far the last epoch moved the policy
            logr = agent.log_prob(agent.dist(O), A) - LOGP
            kls.append(((logr.exp() - 1) - logr).mean().item())         # an unbiased estimate of KL(old || new)
            clipfracs.append(((logr.exp() - 1).abs() > c["clip"]).float().mean().item())
        if (it + 1) % (iters // 10) == 0:
            curve.append(np.mean(finished[-20:]) if finished else np.nan)
    return curve, float(np.mean(kls[-(iters // 2):])), float(np.mean(clipfracs[-(iters // 2):]))


def show(name, results):
    curves = np.array([r[0] for r in results])
    final = curves[:, -2:].mean(1)
    print(f"  {name:34s}" + "".join(f"{x:7.0f}" for x in curves.mean(0)[1::2])
          + f"   {np.mean([r[1] for r in results]):6.3f} {np.mean([r[2] for r in results]):5.2f}   "
          + " ".join(f"{x:5.0f}" for x in final))


if __name__ == "__main__":
    seeds = (0, 1)
    print("=== Parts 1 and 2: PPO on LunarLander-v3, 8 environments x 256 steps, 500,000 steps, 2 seeds ===")
    print("  average return of the last 20 episodes, mean of seeds; KL and fraction clipped after the last epoch,")
    print("  averaged over the second half of training; the final return of each seed (last 20% of training)")
    print("  " + " " * 34 + "   100k   200k   300k   400k   500k       KL  clip   per seed")
    variants = [("PPO (10 epochs, clipping 0.2)", {}), ("  4 epochs", {"epochs": 4}), ("  1 epoch", {"epochs": 1}),
                ("  no clipping", {"clip": 1e6}), ("  no advantage normalization", {"adv_norm": False}),
                ("  value clipping", {"clip_value": True}), ("  discount 0.99, lambda 0.95", {"gamma": 0.99, "lam": 0.95})]
    jobs = [dict(cfg, seed=s) for _, cfg in variants for s in seeds]
    with mp.get_context("fork").Pool(2) as pool:
        res = pool.map(ppo, jobs)
    for k, (name, _) in enumerate(variants):
        show(name, res[2 * k: 2 * k + 2])
    print(f"  (time so far {time.time() - t0:.0f} s)")

    print("\n=== Part 3: PPO on HalfCheetah-v5, 8 environments x 256 steps, 1,000,000 steps, 2 seeds ===")
    print("  " + " " * 34 + "   200k   400k   600k   800k  1000k       KL  clip   per seed")
    mujoco = dict(env="HalfCheetah-v5", steps=1_000_000, gamma=0.99, lam=0.95, ent=0.0)
    variants = [("normalized observations and rewards", dict(mujoco, norm_obs=True, norm_reward=True)),
                ("no normalization", mujoco)]
    jobs = [dict(cfg, seed=s) for _, cfg in variants for s in (0, 1)]
    with mp.get_context("fork").Pool(2) as pool:
        res = pool.map(ppo, jobs)
    for k, (name, _) in enumerate(variants):
        show(name, res[2 * k: 2 * k + 2])
    print(f"\ntotal time {time.time() - t0:.0f} s")
