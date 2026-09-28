"""Lab 15 reference solution: regret minimization and self-play in Leduc hold'em.

Run from this folder:  python lab15_poker.py
It prints the results quoted on the lab page. Takes about 3 minutes on one core; needs numpy and scipy.
"""
import time

import numpy as np
from scipy.optimize import linprog

t0 = time.time()

# ---------------------------------------------------------------- the game
# Leduc hold'em: a deck of two jacks, two queens, and two kings (ranks 0, 1, 2); each player antes 1 chip and gets
# one private card; a betting round; one public card; a second betting round; showdown. In each round player 0 acts
# first, bets and raises are 2 chips in the first round and 4 in the second, and at most two raises (a bet and a
# raise) are allowed per round. Actions: 'f' fold, 'c' check or call, 'r' bet or raise. A pair with the public card
# wins at showdown, otherwise the higher private card, and equal ranks split the pot. Suits never matter, so cards
# are represented by their ranks and the deck's composition enters through the chance probabilities.
DEALS = []
for c0 in range(3):
    for c1 in range(3):
        for pub in range(3):
            p = (2 / 6) * ((2 - (c1 == c0)) / 5) * ((2 - (pub == c0) - (pub == c1)) / 4)
            if p > 0:
                DEALS.append(((c0, c1, pub), p))


def round_over(s):
    return s.endswith("f") or s == "cc" or (len(s) >= 2 and s[-1] == "c" and "r" in s)


def legal(s):
    raises = s.count("r")
    if s.endswith("r"):
        return "fcr" if raises < 2 else "fc"
    return "cr"


def contributions(h1, h2):
    """Chips put in by each player after the betting histories of the two rounds."""
    put = [1.0, 1.0]
    for s, size in ((h1, 2.0), (h2, 4.0)):
        level, mine = 0.0, [0.0, 0.0]
        for k, a in enumerate(s):
            p = k % 2
            if a == "c":
                mine[p] = level
            elif a == "r":
                level += size; mine[p] = level
        put[0] += mine[0]; put[1] += mine[1]
    return put


def payoff(cards, h1, h2):
    """Payoff to player 0 at a terminal history."""
    put = contributions(h1, h2)
    last = h2 if h2 else h1
    if last.endswith("f"):
        folder = (len(last) - 1) % 2
        return -put[0] if folder == 0 else put[1]
    c0, c1, pub = cards
    s0, s1 = (c0 == pub) * 10 + c0, (c1 == pub) * 10 + c1
    return put[1] if s0 > s1 else -put[0] if s1 > s0 else 0.0


class Node:
    __slots__ = ("key", "player", "actions", "children", "terminal", "value", "depth")


def build(cards, h1="", h2=""):
    """The betting tree for one deal. Information set keys: (player, private card, public card or -1, h1, h2)."""
    node = Node()
    in_round2 = round_over(h1) and not h1.endswith("f")
    s = h2 if in_round2 else h1
    node.depth = len(h1) + len(h2)
    if h1.endswith("f") or (in_round2 and round_over(h2)):
        node.terminal, node.value = True, payoff(cards, h1, h2)
        return node
    node.terminal = False
    node.player = len(s) % 2
    node.key = (node.player, cards[node.player], cards[2] if in_round2 else -1, h1, h2)
    node.actions = legal(s)
    node.children = [build(cards, h1 + a, h2) if not in_round2 else build(cards, h1, h2 + a) for a in node.actions]
    return node


TREES = [(build(cards), p) for cards, p in DEALS]
INFOSETS = {}


def collect(node):
    if not node.terminal:
        INFOSETS[node.key] = len(node.actions)
        for ch in node.children:
            collect(ch)


for tree, _ in TREES:
    collect(tree)


def uniform():
    return {k: np.full(n, 1.0 / n) for k, n in INFOSETS.items()}


# ---------------------------------------------------------------- evaluation and best responses
def expected_value(sigma, node=None):
    """Player 0's expected payoff when both players follow sigma."""
    if node is None:
        return sum(p * expected_value(sigma, t) for t, p in TREES)
    if node.terminal:
        return node.value
    return sum(pa * expected_value(sigma, ch) for pa, ch in zip(sigma[node.key], node.children) if pa > 0)


def best_response(sigma, i):
    """A best response of player i to sigma, found information set by information set, deepest first; returns
    the pure strategy and its expected payoff to player i."""
    groups = {}                                     # information set -> list of (node, opponent-and-chance reach)

    def gather(node, w):
        if node.terminal or w == 0:
            return
        if node.player == i:
            groups.setdefault(node.key, []).append((node, w))
            for ch in node.children:
                gather(ch, w)
        else:
            for pa, ch in zip(sigma[node.key], node.children):
                gather(ch, w * pa)

    for tree, p in TREES:
        gather(tree, p)
    br = {}
    sign = 1.0 if i == 0 else -1.0

    def value(node):                                 # player i's payoff below node, under br and sigma_-i
        if node.terminal:
            return sign * node.value
        if node.player == i:
            return value(node.children[br[node.key]])
        return sum(pa * value(ch) for pa, ch in zip(sigma[node.key], node.children) if pa > 0)

    for key in sorted(groups, key=lambda k: -len(k[3]) - len(k[4])):
        nodes = groups[key]
        q = [sum(w * value(n.children[a]) for n, w in nodes) for a in range(len(nodes[0][0].actions))]
        br[key] = int(np.argmax(q))
    pure = {k: np.eye(n)[br[k]] if k in br else np.full(n, 1.0 / n) for k, n in INFOSETS.items() if k[0] == i}
    return pure, sum(p * value(t) for t, p in TREES)


def exploitability(sigma):
    """NashConv: what the two players gain, in chips per hand, by best-responding to sigma."""
    return best_response(sigma, 0)[1] + best_response(sigma, 1)[1]


def merge(s0, s1):
    """A profile whose player-0 strategy is taken from s0 and player-1 strategy from s1."""
    return {k: (s0 if k[0] == 0 else s1)[k] for k in INFOSETS}


# ---------------------------------------------------------------- CFR, CFR+, and Monte Carlo CFR
def matched(r):
    pos = np.maximum(r, 0)
    return pos / pos.sum() if pos.sum() > 0 else np.full(len(r), 1.0 / len(r))


COUNT = [0]                                          # nodes touched, to compare the cost of the algorithms


def cfr_pass(node, w_opp, sig, dR, i):
    """One full traversal of a deal for player i, with the strategies sig fixed for the whole iteration: returns
    player i's expected payoff below node, and accumulates regrets, weighted by chance's and the opponent's reach,
    in dR."""
    COUNT[0] += 1
    if node.terminal:
        return node.value if i == 0 else -node.value
    sigma = sig[node.key]
    if node.player != i:
        return sum(sigma[a] * cfr_pass(ch, w_opp * sigma[a], sig, dR, i)
                   for a, ch in enumerate(node.children) if sigma[a] > 0)
    u = np.array([cfr_pass(ch, w_opp, sig, dR, i) for ch in node.children])
    v = sigma @ u
    dR[node.key] += w_opp * (u - v)
    return v


def es_pass(node, R, S, i, rng):
    """External-sampling MCCFR: the opponent's actions are sampled, player i's are all explored; the opponent's
    average strategy is accumulated where it acts."""
    COUNT[0] += 1
    if node.terminal:
        return node.value if i == 0 else -node.value
    sigma = matched(R[node.key])
    if node.player != i:
        S[node.key] += sigma
        a = rng.choice(len(sigma), p=sigma)
        return es_pass(node.children[a], R, S, i, rng)
    u = np.array([es_pass(ch, R, S, i, rng) for ch in node.children])
    v = sigma @ u
    R[node.key] += u - v
    return v


def average(S):
    return {k: s / s.sum() if s.sum() > 0 else np.full(len(s), 1.0 / len(s)) for k, s in S.items()}


def run_cfr(kind, checkpoints, seed=0):
    rng = np.random.default_rng(seed)
    R = {k: np.zeros(n) for k, n in INFOSETS.items()}
    S = {k: np.zeros(n) for k, n in INFOSETS.items()}
    COUNT[0] = 0
    out, probs = [], np.array([p for _, p in TREES])
    for t in range(1, max(checkpoints) + 1):
        for i in (0, 1):
            if kind == "mccfr":
                tree = TREES[rng.choice(len(TREES), p=probs)][0]
                es_pass(tree, R, S, i, rng)
            else:                                    # strategies fixed during the pass; regrets applied after it
                sig = {k: matched(r) for k, r in R.items()}
                dR = {k: np.zeros(n) for k, n in INFOSETS.items() if k[0] == i}
                for tree, p in TREES:
                    cfr_pass(tree, p, sig, dR, i)
                for k, d in dR.items():
                    R[k] = np.maximum(R[k] + d, 0) if kind == "cfr+" else R[k] + d
                for k, reach in own_reach(sig, i).items():     # the average, weighted by i's own reach
                    S[k] += (t if kind == "cfr+" else 1.0) * reach * sig[k]
        if t in checkpoints:
            out.append((t, COUNT[0], exploitability(average(S)), exploitability({k: matched(r) for k, r in R.items()})))
    return out, average(S)


# ---------------------------------------------------------------- self-play with exact values
def q_values(sigma, i):
    """Player i's action values at each of its information sets: expected payoffs after each action, given that
    play reaches the set, with the histories weighted by chance's and the opponent's reach."""
    num = {k: np.zeros(n) for k, n in INFOSETS.items() if k[0] == i}
    den = {k: 0.0 for k in num}

    def walk(node, w):
        if node.terminal:
            return node.value if i == 0 else -node.value
        if node.player != i:
            return sum(pa * walk(ch, w * pa) for pa, ch in zip(sigma[node.key], node.children) if pa > 0)
        u = np.array([walk(ch, w) for ch in node.children])
        num[node.key] += w * u; den[node.key] += w
        return sigma[node.key] @ u

    for tree, p in TREES:
        walk(tree, p)
    return {k: num[k] / den[k] if den[k] > 0 else np.zeros(len(num[k])) for k in num}


def self_play(method, T, checkpoints, eta=0.05, alpha=0.2, reset=500):
    """Both players update their current strategies from exact action values. 'best response': jump to a best
    response to the other's current strategy. 'mirror descent': add eta times the values to the logits at every
    information set. 'magnet' and 'moving magnet': the regularized step toward a magnet (uniform, or reset to the
    current strategy every `reset` steps)."""
    logits = {k: np.zeros(n) for k, n in INFOSETS.items()}
    magnet = {k: np.zeros(n) for k, n in INFOSETS.items()}                  # log of the uniform magnet, up to a constant
    sigma, out = uniform(), []
    for t in range(1, T + 1):
        if method == "best response":
            sigma = merge(best_response(sigma, 0)[0], best_response(sigma, 1)[0])
        else:
            q = {**q_values(sigma, 0), **q_values(sigma, 1)}
            a = 0.0 if method == "mirror descent" else alpha
            for k in logits:
                z = (logits[k] + eta * a * magnet[k] + eta * q[k]) / (1 + eta * a)
                logits[k] = z - z.max()
            sigma = {k: np.exp(z) / np.exp(z).sum() for k, z in logits.items()}
            if method == "moving magnet" and t % reset == 0:
                magnet = {k: z.copy() for k, z in logits.items()}
        if t in checkpoints:
            out.append(exploitability(sigma))
    return out


# ---------------------------------------------------------------- PSRO
def own_reach(sigma, i):
    """Player i's own probability of reaching each of its information sets (the same for every history in it)."""
    reach = {}

    def walk(node, r):
        if node.terminal:
            return
        if node.player == i:
            reach[node.key] = r
            for pa, ch in zip(sigma[node.key], node.children):
                walk(ch, r * pa)
        else:
            for ch in node.children:
                walk(ch, r)

    for tree, _ in TREES:
        walk(tree, 1.0)
    return reach


def mixture(pop, weights, i):
    """The behavioral strategy equivalent to playing member k of the population with probability weights[k]."""
    reaches = [own_reach(s, i) for s in pop]
    mix = {}
    for k, n in INFOSETS.items():
        if k[0] != i:
            continue
        num = sum(w * r.get(k, 0.0) * s[k] for w, r, s in zip(weights, reaches, pop))
        den = sum(w * r.get(k, 0.0) for w, r in zip(weights, reaches))
        mix[k] = num / den if den > 0 else np.full(n, 1.0 / n)
    return mix


def zero_sum_nash(M):
    """Maximin mixed strategies of the row and column players of the matrix game M (payoffs to the row player)."""
    m, n = M.shape
    res = linprog(np.r_[np.zeros(m), -1.0], A_ub=np.c_[-M.T, np.ones(n)], b_ub=np.zeros(n),
                  A_eq=np.r_[np.ones(m), 0.0][None], b_eq=[1.0], bounds=[(0, None)] * m + [(None, None)])
    res2 = linprog(np.r_[np.zeros(n), 1.0], A_ub=np.c_[M, -np.ones(m)], b_ub=np.zeros(m),
                   A_eq=np.r_[np.ones(n), 0.0][None], b_eq=[1.0], bounds=[(0, None)] * n + [(None, None)])
    x, y = np.maximum(res.x[:m], 0), np.maximum(res2.x[:n], 0)
    return x / x.sum(), y / y.sum()


def psro(meta, iters, checkpoints):
    """Populations start with the uniform strategy; each iteration adds a best response of each player to the
    other's meta-strategy: a Nash equilibrium of the empirical game ('nash', the double oracle), the uniform mixture
    of the population ('uniform', fictitious play), or its latest member ('last', iterated best response)."""
    u = uniform()
    pop0, pop1 = [{k: v for k, v in u.items() if k[0] == 0}], [{k: v for k, v in u.items() if k[0] == 1}]
    M = np.array([[expected_value(merge(pop0[0], pop1[0]))]])
    out = []
    for it in range(1, iters + 1):
        if meta == "nash":
            x, y = zero_sum_nash(M)
        elif meta == "uniform":
            x, y = np.full(len(pop0), 1 / len(pop0)), np.full(len(pop1), 1 / len(pop1))
        else:
            x, y = np.eye(len(pop0))[-1], np.eye(len(pop1))[-1]
        mix0, mix1 = mixture(pop0, x, 0), mixture(pop1, y, 1)
        if it in checkpoints:
            out.append(exploitability(merge(mix0, mix1)))
        br0 = best_response(merge(mix0, mix1), 0)[0]; br1 = best_response(merge(mix0, mix1), 1)[0]
        pop0.append(br0); pop1.append(br1)
        row = [expected_value(merge(br0, s1)) for s1 in pop1]
        col = [expected_value(merge(s0, br1)) for s0 in pop0[:-1]]
        M = np.block([[M, np.array(col)[:, None]], [np.array(row[:-1])[None], np.array([[row[-1]]])]])
    return out


if __name__ == "__main__":
    n0 = sum(1 for k in INFOSETS if k[0] == 0)
    print(f"Leduc hold'em: {len(DEALS)} deals of ranks, {len(INFOSETS)} information sets ({n0} per player)")

    print("\n=== Part 1: CFR and CFR+ (alternating updates), exploitability in chips per hand ===")
    cps = (10, 30, 100, 300, 1000)
    res = {kind: run_cfr(kind, cps) for kind in ("cfr", "cfr+")}
    print("  iterations       CFR average   current      CFR+ average   current")
    for k, t in enumerate(cps):
        (_, _, a1, c1), (_, _, a2, c2) = res["cfr"][0][k], res["cfr+"][0][k]
        print(f"  {t:10d}   {a1:14.4f}  {c1:8.3f}   {a2:14.4f}  {c2:8.3f}")
    avg = res["cfr+"][1]
    print(f"  value of the game to player 0 (CFR+ average): {expected_value(avg):.4f}")
    print("  player 0's first action with each card (probabilities of check, bet):  " +
          "   ".join(f"{'JQK'[c]} {avg[(0, c, -1, '', '')][0]:.2f} {avg[(0, c, -1, '', '')][1]:.2f}" for c in range(3)))
    print(f"  (time so far {time.time() - t0:.0f} s)")

    print("\n=== Part 2: external-sampling Monte Carlo CFR against CFR, by nodes touched ===")
    mc, _ = run_cfr("mccfr", (1000, 3000, 10000, 30000, 100000))
    print("  MCCFR iterations   nodes touched   exploitability")
    for t, nodes, a, _ in mc:
        print(f"  {t:16,d}   {nodes:13,d}   {a:14.4f}")
    for kind in ("cfr", "cfr+"):
        print(f"  {kind.upper():5s} for comparison: " + ", ".join(f"{n:,d} nodes {a:.4f}" for _, n, a, _ in res[kind][0][:4]))
    print(f"  (time so far {time.time() - t0:.0f} s)")

    print("\n=== Part 3: self-play with exact action values: exploitability of the current strategies ===")
    print("  step size 0.05; regularization 0.2 toward a uniform magnet, or toward the current strategy every 500 steps")
    cps = (10, 100, 300, 1000, 3000)
    print("  steps              " + "".join(f"{t:9d}" for t in cps))
    for m in ("best response", "mirror descent", "magnet", "moving magnet"):
        print(f"  {m:17s}" + "".join(f"{x:9.3f}" for x in self_play(m, max(cps), cps)))
    print(f"  (time so far {time.time() - t0:.0f} s)")

    print("\n=== Part 4: PSRO with exact best responses: exploitability of the meta-strategy ===")
    cps = (1, 5, 10, 20, 40, 80, 150)
    print("  iterations         " + "".join(f"{t:9d}" for t in cps))
    for meta, name in (("last", "iterated BR"), ("uniform", "fictitious play"), ("nash", "PSRO (Nash)")):
        print(f"  {name:17s}" + "".join(f"{x:9.3f}" for x in psro(meta, max(cps), cps)))
    print(f"\ntotal time {time.time() - t0:.0f} s")
