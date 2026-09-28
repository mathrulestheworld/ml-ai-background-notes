"""Lab 4 reference solution: planning with learned and given models on Gymnasium environments.

Run from this folder:  python lab04_planning.py
It prints the results quoted on the lab page. Takes about a minute and a half on one core.
"""
import heapq
import math
import time

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

t0 = time.time()


# ---------------------------------------------------------------- part 1: Dyna and prioritized sweeping on Taxi
taxi_model = gym.make("Taxi-v3").unwrapped
t_next = np.array([[taxi_model.P[s][a][0][1] for a in range(6)] for s in range(500)])
t_rew = np.array([[taxi_model.P[s][a][0][2] for a in range(6)] for s in range(500)], float)
t_done = np.array([[taxi_model.P[s][a][0][3] for a in range(6)] for s in range(500)])
starts = np.flatnonzero(taxi_model.initial_state_distrib)      # the 300 possible initial states


def greedy_return(Q):
    """Average return of the greedy policy over all 300 initial states, from the known deterministic model,
    with Gymnasium's 200-step limit."""
    s, total, alive = starts.copy(), np.zeros(len(starts)), np.ones(len(starts), bool)
    for _ in range(200):
        a = Q[s].argmax(1)
        total += alive * t_rew[s, a]
        alive &= ~t_done[s, a]; s = t_next[s, a]
    return total.mean()


Qt = np.zeros((500, 6))                                         # optimal action values, for the target
for _ in range(1000):
    Qt = t_rew + 0.99 * np.where(t_done, 0, Qt.max(1)[t_next])
TARGET = greedy_return(Qt)


def taxi_agent(method, seed, n=10, alpha=1.0, gamma=0.99, eps=0.1, checkpoints=(2_000, 5_000, 10_000, 20_000, 40_000)):
    """Q-learning plus n planning updates per real step from a table model (Taxi is deterministic), with random
    (Dyna-Q) or prioritized (prioritized sweeping) choice of the pairs to update; alpha = 1 since the world is
    deterministic. Returns the greedy policy's average return over all initial states at each checkpoint of real
    steps, and the total number of updates."""
    env = gym.make("Taxi-v3")
    rng = np.random.default_rng(seed)
    Q = np.zeros((500, 6))
    model, keys, preds, pq = {}, [], {}, []
    steps = updates = 0
    curve = []

    def td(s, a, r, s2, term):
        return r + (0.0 if term else gamma * Q[s2].max()) - Q[s, a]

    s, _ = env.reset(seed=seed)
    while steps < checkpoints[-1]:
        a = int(rng.integers(6)) if rng.random() < eps else int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))
        s2, r, term, trunc, _ = env.step(a)
        steps += 1
        if method != "prioritized sweeping":
            Q[s, a] += alpha * td(s, a, r, s2, term); updates += 1
        if (s, a) not in model:
            keys.append((s, a))
        model[(s, a)] = (r, s2, term); preds.setdefault(s2, set()).add((s, a))
        if method == "Dyna-Q":
            for i in rng.integers(len(keys), size=n):
                ps, pa = keys[i]; pr, ps2, pt = model[(ps, pa)]
                Q[ps, pa] += alpha * td(ps, pa, pr, ps2, pt); updates += 1
        elif method == "prioritized sweeping":
            p = abs(td(s, a, r, s2, term))
            if p > 1e-4:
                heapq.heappush(pq, (-p, s, a))
            done_updates = 0
            while pq and done_updates < n:
                _, ps, pa = heapq.heappop(pq); pr, ps2, pt = model[(ps, pa)]
                if abs(td(ps, pa, pr, ps2, pt)) <= 1e-4:
                    continue                                    # a stale entry, already updated
                Q[ps, pa] += alpha * td(ps, pa, pr, ps2, pt); updates += 1; done_updates += 1
                for qs, qa in preds.get(ps, ()):
                    qr, _, qt = model[(qs, qa)]
                    p = abs(td(qs, qa, qr, ps, qt))
                    if p > 1e-4:
                        heapq.heappush(pq, (-p, qs, qa))
        s = s2
        if term or trunc:
            s, _ = env.reset()
        if steps in checkpoints:
            curve.append(greedy_return(Q))
    return curve, updates


print(f"=== Part 1: Taxi-v3, average return of the greedy policy over all 300 initial states (optimal {TARGET:.2f}) ===")
print(f"  {'real steps:':30s}" + "".join(f"{c:>8,}" for c in (2_000, 5_000, 10_000, 20_000, 40_000)) + "   updates   time")
for method, n in [("Q-learning", 0), ("Dyna-Q", 10), ("Dyna-Q", 50), ("prioritized sweeping", 10)]:
    t1 = time.time()
    out = [taxi_agent(method, seed, n=n) for seed in range(5)]
    curve = np.mean([o[0] for o in out], 0)
    label = method if method == "Q-learning" else f"{method}, n = {n}"
    print(f"  {label:30s}" + "".join(f"{v:8.1f}" for v in curve)
          + f"  {np.mean([o[1] for o in out]):9,.0f}  {(time.time() - t1) / 5:4.1f} s")


# ---------------------------------------------------------------- part 2: a learned stochastic model
fl = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True).unwrapped
nS, nA, gamma = 64, 4, 0.99
term = np.isin(fl.desc.ravel(), [b"H", b"G"])
T = np.zeros((nS, nA, nS)); Rw = np.zeros((nS, nA))
for s in range(nS):
    for a in range(nA):
        for p, s2, r, d in fl.P[s][a]:
            T[s, a, s2] += p; Rw[s, a] += p * r
cumT = T.cumsum(2)


def start_value(Q):                                             # exact value of the greedy policy at the start state
    pi = Q.argmax(1)
    return np.linalg.solve(np.eye(nS) - gamma * T[np.arange(nS), pi] * ~term[None, :], Rw[np.arange(nS), pi])[0]


Qstar = np.zeros((nS, nA))
for _ in range(3000):
    Qstar = Rw + gamma * T @ np.where(term, 0, Qstar.max(1))


def frozen_agent(method, seed, steps=20_000, n=10, alpha=0.1, eps=0.1):
    rng = np.random.default_rng(seed)
    Q = np.zeros((nS, nA)); counts = np.zeros((nS, nA, nS)); seen = []
    s, curve = 0, []
    for t in range(1, steps + 1):
        a = int(rng.integers(nA)) if rng.random() < eps else int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))
        s2 = int((cumT[s, a] > rng.random()).argmax()); r = float(s2 == nS - 1)
        Q[s, a] += alpha * (r + gamma * (0 if term[s2] else Q[s2].max()) - Q[s, a])
        if counts[s, a].sum() == 0:
            seen.append((s, a))
        counts[s, a, s2] += 1
        if method == "Dyna, expected updates":
            V = np.where(term, 0, Q.max(1))
            for k in rng.integers(len(seen), size=n):
                ps, pa = seen[k]; p = counts[ps, pa] / counts[ps, pa].sum()
                Q[ps, pa] = p @ ((np.arange(nS) == nS - 1) + gamma * V)   # alpha = 1: the model already averages
                V[ps] = 0 if term[ps] else Q[ps].max()
        s = 0 if term[s2] else s2
        if t % 5000 == 0:
            curve.append(start_value(Q))
    return curve


print(f"\n=== Part 2: FrozenLake 8x8 (slippery), gamma = 0.99: value of the start state under the greedy policy ===")
print(f"  optimal: {start_value(Qstar):.3f}")
for method in ["Q-learning", "Dyna, expected updates"]:
    c = np.mean([frozen_agent(method, seed) for seed in range(5)], 0)
    print(f"  {method:24s} after 5,000 / 10,000 / 15,000 / 20,000 real steps: " + " / ".join(f"{v:.3f}" for v in c))


# ---------------------------------------------------------------- part 3: decision-time planning in blackjack
cards = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])
rng = np.random.default_rng(0)


def add(total, ace, c):                                         # add one card to a hand (sum, usable ace)
    total += c
    if c == 1 and total + 10 <= 21:
        total, ace = total + 10, True
    if total > 21 and ace:
        total, ace = total - 10, False
    return total, ace


def dealer_outcome(player, showing):
    d, dace = add(0, False, showing)
    d, dace = add(d, dace, int(cards[rng.integers(13)]))
    while d < 17:
        d, dace = add(d, dace, int(cards[rng.integers(13)]))
    return 1.0 if d > 21 or player > d else (0.0 if player == d else -1.0)


def uct_decision(p, ace, showing, budget, c=1.0):
    """UCT over the player's hands (sum, usable ace) with the dealer's card fixed; chance nodes are card draws."""
    N, Nsa, Wsa = {}, {}, {}
    for _ in range(budget):
        path, (hp, ha), r = [], (p, ace), None
        while r is None:
            node = (hp, ha)
            N.setdefault(node, 0)
            ucb = [Wsa.get((node, a), 0.0) / Nsa[(node, a)] + c * math.sqrt(math.log(N[node] + 1) / Nsa[(node, a)])
                   if Nsa.get((node, a), 0) > 0 else math.inf for a in (0, 1)]
            a = int(np.argmax(ucb)); path.append((node, a))
            if a == 0:
                r = dealer_outcome(hp, showing)
            else:
                hp, ha = add(hp, ha, int(cards[rng.integers(13)]))
                if hp > 21:
                    r = -1.0
        for node, a in path:
            N[node] += 1; Nsa[(node, a)] = Nsa.get((node, a), 0) + 1; Wsa[(node, a)] = Wsa.get((node, a), 0.0) + r
    root = (p, ace)
    return int(max((0, 1), key=lambda a: Nsa.get((root, a), 0)))


def rollout_decision(p, ace, showing, m):                       # base policy: stick on 20 or 21
    def value(action):
        total = 0.0
        for _ in range(m):
            hp, ha = p, ace
            if action == 1:
                hp, ha = add(hp, ha, int(cards[rng.integers(13)]))
                while hp < 20:
                    hp, ha = add(hp, ha, int(cards[rng.integers(13)]))
            total += -1.0 if hp > 21 else dealer_outcome(hp, showing)
        return total / m
    return int(value(1) > value(0))


stick_from = {0: [17, 13, 13, 12, 12, 12, 17, 17, 17, 17], 1: [19, 18, 18, 18, 18, 18, 18, 18, 19, 19]}
states = [(p, u, d) for p in range(12, 22) for u in (0, 1) for d in range(1, 11)]
optimal = {s: int(s[0] < stick_from[s[1]][s[2] - 1]) for s in states}


def play(table, hands=20_000):
    env, total = gym.make("Blackjack-v1", sab=True), 0.0
    for h in range(hands):
        (p, d, u), _ = env.reset(seed=h)
        done = False
        while not done:
            (p, d, u), r, done, _, _ = env.step(1 if p < 12 else table[(p, u, d)])
        total += r
    return total / hands


print("\n=== Part 3: blackjack, one search per decision state, returns on the same 20,000 deals ===")
print(f"  base policy (stick on 20 or 21):   {play({s: int(s[0] < 20) for s in states}):+.4f}")
for name, decide in [("rollout, m = 100", lambda s: rollout_decision(s[0], bool(s[1]), s[2], 100)),
                     ("UCT, 100 simulations", lambda s: uct_decision(s[0], bool(s[1]), s[2], 100)),
                     ("UCT, 1,000 simulations", lambda s: uct_decision(s[0], bool(s[1]), s[2], 1000)),
                     ("UCT, 10,000 simulations", lambda s: uct_decision(s[0], bool(s[1]), s[2], 10_000))]:
    table = {s: decide(s) for s in states}
    wrong = sum(table[s] != optimal[s] for s in states)
    print(f"  {name:32s}  {play(table):+.4f}; decisions differing from optimal: {wrong:3d} of 200")
print(f"  optimal policy:                    {play(optimal):+.4f}")


# ---------------------------------------------------------------- part 4: RTDP on a large FrozenLake
desc = [list(row) for row in generate_random_map(size=40, p=0.9, seed=3)]
desc[39][39], desc[12][12] = "F", "G"                           # move the goal to (12, 12)
big = gym.make("FrozenLake-v1", desc=["".join(row) for row in desc], is_slippery=True).unwrapped
nB, g = 40 * 40, 0.99
bterm = np.isin(big.desc.ravel(), [b"H", b"G"]); goal = int(np.flatnonzero(big.desc.ravel() == b"G")[0])
nxt = np.zeros((nB, 4, 3), int)
for s in range(nB):
    for a in range(4):
        outs = big.P[s][a] if len(big.P[s][a]) == 3 else big.P[s][a] * 3
        nxt[s, a] = [s2 for _, s2, _, _ in outs]


def backup_value(V, s):                                         # expected value of each action; reward 1 at the goal
    return ((nxt[s] == goal) + g * np.where(bterm[nxt[s]], 0, V[nxt[s]])).mean(1)


V = np.zeros(nB)                                                # value iteration from zero, to convergence
while True:
    delta = 0.0
    for s in range(nB):
        if not bterm[s]:
            v = backup_value(V, s).max(); delta = max(delta, abs(v - V[s])); V[s] = v
    if delta < 1e-8:
        break
v_star = V[0]
gi, gj = divmod(goal, 40)                                       # optimistic bound: the goal is at least d steps away
h = np.array([g ** (abs(i - gi) + abs(j - gj) - 1) for i in range(40) for j in range(40)]); h[bterm] = 0
Vh, vi_sweeps = h.copy(), 0
while Vh[0] > 1.01 * v_star:
    for s in range(nB):
        if not bterm[s]:
            Vh[s] = backup_value(Vh, s).max()
    vi_sweeps += 1
Vr, touched, updates, trials = h.copy(), set(), 0, 0            # RTDP from the same bound
rng = np.random.default_rng(0)
while Vr[0] > 1.01 * v_star:
    s, trials = 0, trials + 1
    for _ in range(2000):
        q = backup_value(Vr, s); a = int(np.argmax(q)); Vr[s] = q[a]
        touched.add(s); updates += 1
        s = int(nxt[s, a, rng.integers(3)])
        if bterm[s]:
            break
n_free = int((~bterm).sum())
print(f"\n=== Part 4: a random 40x40 FrozenLake with the goal at (12, 12) ({n_free} non-terminal states), gamma = 0.99 ===")
print(f"  optimal value of the start state {v_star:.4f}; from the bound gamma^(distance - 1), start value within 1%:")
print(f"    value iteration: {vi_sweeps} sweeps, {vi_sweeps * n_free:,} updates")
print(f"    RTDP:            {trials:,} trials, {updates:,} updates, {len(touched)} states updated"
      f" ({100 * len(touched) / n_free:.0f}%)")
print(f"\ntotal time {time.time() - t0:.0f} s")
