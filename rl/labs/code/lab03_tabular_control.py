"""Lab 3 reference solution: tabular control with Monte Carlo and TD methods on Gymnasium environments.

Run from this folder:  python lab03_tabular_control.py
It prints the results quoted on the lab page. Takes about four minutes on one core.
"""
import time

import gymnasium as gym
import numpy as np


# ---------------------------------------------------------------- part 1: a tabular control library
class Agent:
    """Tabular action values with epsilon-greedy behavior. `alpha` is a constant step size, or a function of
    the number of updates n of the pair being updated, such as lambda n: n ** -0.8."""

    def __init__(self, nS, nA, alpha=0.5, gamma=1.0, eps=0.1, seed=0):
        self.Q = np.zeros((nS, nA))
        self.N = np.zeros((nS, nA))
        self.nA, self.gamma, self.eps = nA, gamma, eps
        self.alpha = alpha if callable(alpha) else (lambda n, a=alpha: a)
        self.rng = np.random.default_rng(seed)

    def values(self, s):                                     # the estimates that define the policy
        return self.Q[s]

    def act(self, s, eps=None):
        eps = self.eps if eps is None else eps
        if self.rng.random() < eps:
            return int(self.rng.integers(self.nA))
        q = self.values(s)
        return int(self.rng.choice(np.flatnonzero(q == q.max())))   # random tie-breaking

    def step_size(self, s, a):
        self.N[s, a] += 1
        return self.alpha(self.N[s, a])

    def update(self, s, a, r, s2, a2, terminated, truncated):
        target = r if terminated else r + self.gamma * self.next_value(s2, a2)
        self.Q[s, a] += self.step_size(s, a) * (target - self.Q[s, a])


class Sarsa(Agent):
    def next_value(self, s2, a2):
        return self.Q[s2, a2]


class ExpectedSarsa(Agent):
    def next_value(self, s2, a2):
        q = self.Q[s2]
        greedy = q == q.max()
        p = self.eps / self.nA + (1 - self.eps) * greedy / greedy.sum()   # the epsilon-greedy policy, ties split
        return p @ q


class QLearning(Agent):
    def next_value(self, s2, a2):
        return self.Q[s2].max()


class DoubleQLearning(Agent):
    def __init__(self, nS, nA, **kw):
        super().__init__(nS, nA, **kw)
        self.Q2 = np.zeros((nS, nA))
        self.N2 = np.zeros((nS, nA))

    def values(self, s):
        return self.Q[s] + self.Q2[s]

    def update(self, s, a, r, s2, a2, terminated, truncated):
        if self.rng.random() < 0.5:                          # update one copy, chosen by a coin flip
            Qa, Qb, N = self.Q, self.Q2, self.N
        else:
            Qa, Qb, N = self.Q2, self.Q, self.N2
        N[s, a] += 1
        target = r if terminated else r + self.gamma * Qb[s2, Qa[s2].argmax()]
        Qa[s, a] += self.alpha(N[s, a]) * (target - Qa[s, a])


class MonteCarlo(Agent):
    """On-policy first-visit Monte Carlo control; the step size 1/n makes each estimate a sample average."""

    def __init__(self, nS, nA, **kw):
        super().__init__(nS, nA, alpha=lambda n: 1 / n, **kw)
        self.episode = []

    def update(self, s, a, r, s2, a2, terminated, truncated):
        self.episode.append((s, a, r))
        if not (terminated or truncated):
            return
        G, first = 0.0, {}
        for t, (s_, a_, _) in enumerate(self.episode):
            first.setdefault((s_, a_), t)
        for t in reversed(range(len(self.episode))):         # returns computed backward
            s_, a_, r_ = self.episode[t]
            G = r_ + self.gamma * G
            if first[(s_, a_)] == t:
                self.Q[s_, a_] += self.step_size(s_, a_) * (G - self.Q[s_, a_])
        self.episode = []


def train(env, agent, episodes, seed, encode=int, eval_every=None, evaluate_fn=None):
    """Run `episodes` episodes; return the online returns (and periodic evaluations if requested).
    Only `terminated` ends the bootstrap; `truncated` just ends the episode."""
    returns, evals = [], []
    obs, _ = env.reset(seed=seed)
    for ep in range(episodes):
        if ep > 0:
            obs, _ = env.reset()
        s = encode(obs)
        a, G, done = agent.act(s), 0.0, False
        while not done:
            obs2, r, terminated, truncated, _ = env.step(a)
            s2 = encode(obs2)
            a2 = agent.act(s2)
            agent.update(s, a, r, s2, a2, terminated, truncated)
            s, a, G, done = s2, a2, G + r, terminated or truncated
        returns.append(G)
        if eval_every and (ep + 1) % eval_every == 0:
            evals.append(evaluate_fn(agent))
    return np.array(returns), np.array(evals)


def evaluate(env, policy, episodes, seed, encode=int, max_steps=None):
    """Average return of a deterministic policy (a function of the encoded state) over fixed seeds."""
    total = 0.0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, n = False, 0
        while not done:
            obs, r, terminated, truncated, _ = env.step(policy(encode(obs)))
            total, n = total + r, n + 1
            done = terminated or truncated or (max_steps is not None and n >= max_steps)
    return total / episodes


def greedy(agent):
    return lambda s: int(agent.values(s).argmax())


def value_iteration(env, gamma, eps=0.0, sweeps=10_000, tol=1e-10):
    """Optimal action values from Gymnasium's model env.unwrapped.P (as in Lab 1). With eps > 0, the backup
    values the next state under the epsilon-greedy policy, which gives the best epsilon-greedy policy's values."""
    P = env.unwrapped.P
    nS, nA = len(P), len(P[0])
    Q = np.zeros((nS, nA))
    for _ in range(sweeps):
        V = (1 - eps) * Q.max(1) + eps * Q.mean(1)
        Qn = np.array([[sum(p * (r + (0.0 if d else gamma * V[s2])) for p, s2, r, d in P[s][a])
                        for a in range(nA)] for s in range(nS)])
        if np.abs(Qn - Q).max() < tol:
            return Qn
        Q = Qn
    return Q


METHODS = {"SARSA": Sarsa, "Expected SARSA": ExpectedSarsa, "Q-learning": QLearning,
           "double Q-learning": DoubleQLearning}
t0 = time.time()

# ---------------------------------------------------------------- part 2: blackjack
print("=== Part 2: blackjack (Sutton and Barto's rules), 300,000 training episodes ===")
bj = gym.make("Blackjack-v1", sab=True)
enc = lambda o: (o[0] * 11 + o[1]) * 2 + o[2]                  # (player sum 0-31, dealer card 1-10, usable ace)
stick_from = {0: [17, 13, 13, 12, 12, 12, 17, 17, 17, 17],     # the optimal policy of chapter 5, exercise 5.6
              1: [19, 18, 18, 18, 18, 18, 18, 18, 19, 19]}     # indexed by usable ace, then dealer card 1-10
decision_states = [(p, d, u) for p in range(12, 22) for d in range(1, 11) for u in (0, 1)]


def optimal_bj(s):
    u, rest = s % 2, s // 2
    p, d = divmod(rest, 11)
    return 0 if p >= 12 and p >= stick_from[u][d - 1] else 1   # 0 = stick, 1 = hit


policies = {"optimal (chapter 5)": optimal_bj}
bj_agents = {"Monte Carlo control": MonteCarlo(32 * 11 * 2, 2, eps=0.1, seed=1),
             "Q-learning": QLearning(32 * 11 * 2, 2, alpha=lambda n: n ** -0.8, eps=0.1, seed=1)}
for name, agent in bj_agents.items():
    train(bj, agent, 300_000, seed=1, encode=enc)
    policies[name] = greedy(agent)
for name, pol in policies.items():
    wrong = [s for s in decision_states if pol(enc(s)) != optimal_bj(enc(s))]
    ret = evaluate(bj, pol, 200_000, seed=10**6, encode=enc)
    print(f"  {name:20s} return per hand {ret:+.4f}; decisions differing from optimal: {len(wrong):3d} of 200")
    if 0 < len(wrong) <= 16:
        print("    at (sum, dealer, usable ace):", ", ".join(f"({p},{d},{u})" for p, d, u in wrong))
print(f"  (standard error of each return: about {np.sqrt(0.9 / 200_000):.4f})")

# ---------------------------------------------------------------- part 3: cliff walking
print("\n=== Part 3a: CliffWalking-v1, epsilon = 0.1, alpha = 0.5, 500 episodes, 10 runs ===")
cliff = gym.make("CliffWalking-v1")


def route(agent, env, cap=100):
    s, _ = env.reset(seed=0)
    path = [s]
    for _ in range(cap):
        s, _, terminated, _, _ = env.step(int(agent.values(s).argmax()))
        path.append(s)
        if terminated:
            break
    return len(path) - 1, min(p // 12 for p in path)            # steps, highest row (0 is the top)


for name, cls in METHODS.items():
    online, lengths, rows = [], [], []
    for run in range(10):
        agent = cls(48, 4, alpha=0.5, eps=0.1, seed=run)
        rets, _ = train(cliff, agent, 500, seed=run)
        online.append(rets[400:].mean())
        n, row = route(agent, cliff); lengths.append(n); rows.append(row)
    print(f"  {name:18s} online return, episodes 401-500: {np.mean(online):6.1f};"
          f" greedy route {np.median(lengths):.0f} steps, highest row {np.median(rows):.0f}")

print("\n=== Part 3b: CliffWalkingSlippery-v1 (1/3 intended, 1/3 each perpendicular), alpha = 0.1, 3,000 episodes ===")
slip = gym.make("CliffWalkingSlippery-v1")
Qs = value_iteration(slip, gamma=1.0)
print(f"  optimal (value iteration)  expected return from the start: {Qs[36].max():7.1f};"
      f" simulated: {evaluate(slip, lambda s: int(Qs[s].argmax()), 1000, seed=5000, max_steps=1000):7.1f}")
Qe = value_iteration(slip, gamma=1.0, eps=0.1)
print(f"  best epsilon-greedy policy (eps 0.1): its greedy part's return:"
      f" {evaluate(slip, lambda s: int(Qe[s].argmax()), 1000, seed=5000, max_steps=1000):7.1f}")
arrows = "^>v<"
for name, Q in [("optimal", Qs), ("best eps-greedy", Qe)]:
    rows = ["".join(arrows[int(Q[i * 12 + j].argmax())] for j in range(12)) for i in range(3)]
    print(f"  {name + ' policy, rows 0-2:':34s}" + "  ".join(rows) + f"   start: {arrows[int(Q[36].argmax())]}")
for name, cls in METHODS.items():
    agent = cls(48, 4, alpha=0.1, eps=0.1, seed=0)
    rets, _ = train(slip, agent, 3000, seed=0)
    ret = evaluate(slip, greedy(agent), 1000, seed=5000, max_steps=1000)
    print(f"  {name:26s} greedy return: {ret:7.1f}; online return in the last 500 episodes:"
          f" {rets[-500:].mean():7.1f}")

# ---------------------------------------------------------------- part 4: taxi and frozen lake
print("\n=== Part 4a: Taxi-v3, epsilon = 0.1, alpha = 0.5, return of the greedy policy every 250 episodes (3 runs) ===")
taxi = gym.make("Taxi-v3")
Qt = value_iteration(taxi, gamma=0.99)
opt = evaluate(taxi, lambda s: int(Qt[s].argmax()), 100, seed=10_000)
print(f"  optimal policy on the 100 test episodes: {opt:.2f}")
for name, cls in METHODS.items():
    curves = []
    for run in range(3):
        agent = cls(500, 6, alpha=0.5, gamma=0.99, eps=0.1, seed=run)
        _, ev = train(taxi, agent, 2000, seed=run, eval_every=250,
                      evaluate_fn=lambda ag: evaluate(taxi, greedy(ag), 100, seed=10_000))
        curves.append(ev)
    print(f"  {name:18s}" + "".join(f"{v:8.1f}" for v in np.mean(curves, 0)))
print(f"  {'(episodes)':18s}" + "".join(f"{e:8d}" for e in range(250, 2001, 250)))

print("\n=== Part 4b: FrozenLake 4x4 (slippery), gamma = 0.99, Q-learning step sizes, 10,000 episodes (3 runs) ===")
fl = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
Qf = value_iteration(fl, gamma=0.99)
print(f"  optimal stationary policy (gamma 0.99): success rate within 100 steps"
      f" {evaluate(fl, lambda s: int(Qf[s].argmax()), 5000, seed=20_000):.3f};"
      f" value of the start state {Qf[0].max():.3f}")
schedules = {"alpha = 0.1": 0.1, "alpha = 0.01": 0.01, "alpha = 1/n": lambda n: 1 / n,
             "alpha = 1/n^0.6": lambda n: n ** -0.6}
for name, alpha in schedules.items():
    succ, err = [], []
    for run in range(3):
        agent = QLearning(16, 4, alpha=alpha, gamma=0.99, eps=0.1, seed=run)
        train(fl, agent, 10_000, seed=run)
        succ.append(evaluate(fl, greedy(agent), 2000, seed=20_000))
        err.append(np.abs(agent.Q[0].max() - Qf[0].max()))
    print(f"  {name:16s} greedy success rate {np.mean(succ):.3f}; error in the start state's value {np.mean(err):.3f}")
print(f"\ntotal time {time.time() - t0:.0f} s")
