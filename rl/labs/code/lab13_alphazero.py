"""Lab 13 reference solution: AlphaZero and Gumbel search from scratch, on tic-tac-toe and Connect Four.

Run from this folder:  python lab13_alphazero.py
It prints the results quoted on the lab page. Takes about 17 minutes on two cores.
"""
import math
import multiprocessing as mp
import os
import time
from functools import lru_cache

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

t0 = time.time()


# ---------------------------------------------------------------- games
# A state is (me, opp, n): bitboards of the stones of the player to move and of the opponent, and the number of
# stones on the board. play() returns the next state, from the other player's point of view, and the outcome for
# the player to move in it: -1 if the move just played won, 0 for a draw, None if the game goes on.
class TicTacToe:
    A, name, cells, shape = 9, "tic-tac-toe", 9, (3, 3)
    LINES = [sum(1 << i for i in line) for line in
             [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]]
    BITS = np.array([1 << i for i in range(9)], dtype=np.int64)

    @staticmethod
    def initial():
        return (0, 0, 0)

    @staticmethod
    def legal(s):
        full = s[0] | s[1]
        return [a for a in range(9) if not (full >> a) & 1]

    @classmethod
    def play(cls, s, a):
        me = s[0] | (1 << a)
        if any(me & m == m for m in cls.LINES):
            return (s[1], me, s[2] + 1), -1
        return (s[1], me, s[2] + 1), (0 if s[2] + 1 == 9 else None)

    @classmethod
    def encode(cls, s):
        return ((np.array(s[:2], dtype=np.int64)[:, None] & cls.BITS) != 0).astype(np.float32).ravel()


class ConnectFour:
    """6 rows and 7 columns; bit 7c + r is row r (from the bottom) of column c, and bit 7c + 6 is always empty."""
    A, name, cells, shape = 7, "Connect Four", 42, (7, 6)
    BITS = np.array([1 << (7 * c + r) for c in range(7) for r in range(6)], dtype=np.int64)
    BOTTOM = [1 << (7 * c) for c in range(7)]
    TOP = [1 << (7 * c + 5) for c in range(7)]
    COL = [0b111111 << (7 * c) for c in range(7)]

    @staticmethod
    def initial():
        return (0, 0, 0)

    @classmethod
    def legal(cls, s):
        full = s[0] | s[1]
        return [c for c in range(7) if not full & cls.TOP[c]]

    @staticmethod
    def won(b):
        for k in (1, 7, 6, 8):
            m = b & (b >> k)
            if m & (m >> (2 * k)):
                return True
        return False

    @classmethod
    def play(cls, s, a):
        full = s[0] | s[1]
        me = s[0] | ((full + cls.BOTTOM[a]) & cls.COL[a])
        if cls.won(me):
            return (s[1], me, s[2] + 1), -1
        return (s[1], me, s[2] + 1), (0 if s[2] + 1 == 42 else None)

    @classmethod
    def encode(cls, s):
        return ((np.array(s[:2], dtype=np.int64)[:, None] & cls.BITS) != 0).astype(np.float32).ravel()


# ---------------------------------------------------------------- network
class Net(nn.Module):
    """A policy-value network: the stones of the player to move and of the opponent, through two hidden layers
    (after two 3x3 convolutions with `channels` > 0), to the logits of a policy over all actions and a value in
    [-1, 1] for the player to move."""

    def __init__(self, game, hidden, channels=0):
        super().__init__()
        self.shape, self.channels = game.shape, channels
        if channels:
            self.conv = nn.Sequential(nn.Conv2d(2, channels, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(), nn.Flatten())
            n_in = channels * game.cells
        else:
            self.conv, n_in = nn.Identity(), 2 * game.cells
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.pi, self.v = nn.Linear(hidden, game.A), nn.Linear(hidden, 1)

    def forward(self, x):
        if self.channels:
            x = x.view(-1, 2, *self.shape)
        h = self.body(self.conv(x))
        return self.pi(h), torch.tanh(self.v(h))[:, 0]

    def numpy_weights(self):
        w = []
        for p in self.parameters():
            p = p.detach().numpy()
            w.append(p.reshape(p.shape[0], -1).T.copy() if p.ndim == 4 else p.T.copy() if p.ndim == 2 else p.copy())
        return self.shape, self.channels, w


def patch_index(C, H, W):
    """Indices that gather the 3x3 neighborhoods of every cell from a zero-padded (C, H+2, W+2) array."""
    return np.array([[ch * (H + 2) * (W + 2) + (r + dr) * (W + 2) + (c + dc)
                      for ch in range(C) for dr in range(3) for dc in range(3)] for r in range(H) for c in range(W)])


class NumpyNet:
    """The same network in numpy, for fast evaluation of one position at a time during search."""

    def __init__(self, weights):
        (self.H, self.W), self.C, self.w = weights
        if self.C:
            self.idx1, self.idx2 = patch_index(2, self.H, self.W), patch_index(self.C, self.H, self.W)
            self.pad1 = np.zeros((2, self.H + 2, self.W + 2), np.float32)
            self.pad2 = np.zeros((self.C, self.H + 2, self.W + 2), np.float32)

    def __call__(self, x):
        w = self.w
        if self.C:
            H, W = self.H, self.W
            self.pad1[:, 1:H + 1, 1:W + 1] = x.reshape(2, H, W)
            h = np.maximum(self.pad1.ravel()[self.idx1] @ w[0] + w[1], 0)          # (H W, C)
            self.pad2[:, 1:H + 1, 1:W + 1] = h.T.reshape(self.C, H, W)
            h = np.maximum(self.pad2.ravel()[self.idx2] @ w[2] + w[3], 0)
            x, w = h.T.ravel(), w[4:]
        W1, b1, W2, b2, Wp, bp, Wv, bv = w
        h = np.maximum(x @ W1 + b1, 0); h = np.maximum(h @ W2 + b2, 0)
        return h @ Wp + bp, math.tanh(float(h @ Wv[:, 0] + bv[0]))


# ---------------------------------------------------------------- search
class Node:
    __slots__ = ("state", "logits", "P", "N", "W", "children", "legal", "value", "terminal")

    def __init__(self, state, terminal=None):
        self.state, self.terminal, self.children = state, terminal, {}

    def expand(self, game, net):
        self.legal = np.full(game.A, False); self.legal[game.legal(self.state)] = True
        logits, self.value = net(game.encode(self.state))
        self.logits = np.where(self.legal, logits - logits[self.legal].max(), -np.inf)
        self.P = np.exp(self.logits); self.P /= self.P.sum()
        self.N, self.W = np.zeros(game.A), np.zeros(game.A)
        return self.value


def child(game, node, a, net):
    """The child of node by action a, created and evaluated if new; returns it and the value of its position for
    the player to move there (exact if the game is over)."""
    c = node.children.get(a)
    if c is None:
        s2, outcome = game.play(node.state, a)
        c = node.children[a] = Node(s2, outcome)
        return c, (outcome if outcome is not None else c.expand(game, net))
    return c, (c.terminal if c.terminal is not None else None)


def backup(path, v):
    """v is the value of the leaf for the player to move there; each parent sees it with the opposite sign."""
    for node, a in reversed(path):
        v = -v
        node.N[a] += 1; node.W[a] += v


def puct_search(game, state, net, n, rng, c_puct=1.25, noise=False, dir_alpha=1.0, eps=0.25):
    """AlphaZero's search: n simulations, each descending by PUCT to a new position that the network evaluates.
    Returns the root (whose visit counts define the move and the training target)."""
    root = Node(state); root.expand(game, net)
    P_root = root.P
    if noise:
        legal = np.flatnonzero(root.legal)
        eta = np.zeros(game.A); eta[legal] = rng.dirichlet([dir_alpha] * len(legal))
        P_root = (1 - eps) * root.P + eps * eta
    for _ in range(n):
        node, path = root, []
        while True:
            P = P_root if node is root else node.P
            Q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), 0.0)
            score = np.where(node.legal, Q + c_puct * P * math.sqrt(node.N.sum() + 1) / (1 + node.N), -np.inf)
            a = int(np.argmax(score)); path.append((node, a))
            node, v = child(game, node, a, net)
            if v is not None:                      # a new position, or the end of the game
                break
        backup(path, v)
    return root


def completed_sigma(node, c_visit=50.0, c_scale=0.1):
    """Gumbel MuZero's sigma(completed q): visited actions keep their mean value, unvisited ones get the mixed
    value estimate; the values are rescaled to [0, 1] over the legal actions and multiplied by (c_visit + max N)
    c_scale (the defaults of DeepMind's mctx library)."""
    visited = node.N > 0
    Q = np.where(visited, node.W / np.maximum(node.N, 1), 0.0)
    if visited.any():
        p = node.P[visited]
        v_mix = (node.value + node.N.sum() * (p @ Q[visited]) / p.sum()) / (1 + node.N.sum())
    else:
        v_mix = node.value
    q = np.where(visited, Q, v_mix)[node.legal]
    lo, hi = q.min(), q.max()
    q = (q - lo) / max(hi - lo, 1e-8)
    out = np.full(len(node.N), -np.inf); out[node.legal] = (c_visit + node.N.max()) * c_scale * q
    return out


def considered_visits(m, n):
    """The visit count that sequential halving assigns to the next simulation, for m considered actions and n
    simulations (as in mctx): the actions whose visit count equals it are candidates for that simulation."""
    if m <= 1:
        return list(range(n))
    log2m, seq, visits, k = math.ceil(math.log2(m)), [], [0] * m, m
    while len(seq) < n:
        for _ in range(max(1, int(n / (log2m * k)))):
            seq.extend(visits[:k])
            for i in range(k):
                visits[i] += 1
        k = max(2, k // 2)
    return seq[:n]


def gumbel_search(game, state, net, n, rng, noise=True, m_max=16):
    """Gumbel MuZero's search: at the root, the Gumbel-top-k trick and sequential halving; below the root, the
    deterministic selection argmax pi'(a) - N(a) / (1 + sum N), with pi' = softmax(logits + sigma(completed q)).
    Returns the root, the chosen action, and the improved policy used as the training target."""
    root = Node(state); root.expand(game, net)
    g = np.where(root.legal, rng.gumbel(size=game.A) if noise else 0.0, -np.inf)
    seq = considered_visits(min(m_max, int(root.legal.sum())), n)
    for k in range(n):
        node, path = root, []
        while True:
            if node is root:
                score = g + root.logits + completed_sigma(root)
                a = int(np.argmax(np.where(root.N == seq[k], score, -np.inf)))
            else:
                z = node.logits + completed_sigma(node)
                pi = np.exp(z - z.max()); pi /= pi.sum()
                a = int(np.argmax(np.where(node.legal, pi - node.N / (1 + node.N.sum()), -np.inf)))
            path.append((node, a))
            node, v = child(game, node, a, net)
            if v is not None:
                break
        backup(path, v)
    score = g + root.logits + completed_sigma(root)
    action = int(np.argmax(np.where(root.N == root.N.max(), score, -np.inf))) if n > 0 else int(np.argmax(score))
    z = root.logits + completed_sigma(root)
    target = np.exp(z - z.max()); target /= target.sum()
    return root, action, target


def choose(game, state, net, algo, n, rng, train=False, temp_moves=0):
    """The move of an agent, and in training the policy target."""
    if algo == "policy":                           # the raw network, greedy
        root = Node(state); root.expand(game, net)
        return int(np.argmax(root.logits)), None
    if algo == "gumbel":
        _, a, target = gumbel_search(game, state, net, n, rng, noise=train)
        return a, target
    root = puct_search(game, state, net, n, rng, noise=train)
    target = root.N / root.N.sum()
    if train and state[2] < temp_moves:
        return int(rng.choice(game.A, p=target)), target
    return int(np.argmax(root.N + 1e-6 * root.P)), target


# ---------------------------------------------------------------- self-play and training
def self_play(args):
    """Play games against itself; return (inputs, policy targets, outcomes) for every position."""
    game, weights, algo, n, games, seed, temp_moves = args
    rng, net = np.random.default_rng(seed), NumpyNet(weights)
    X, PI, Z = [], [], []
    for _ in range(games):
        s, hist, outcome = game.initial(), [], None
        while outcome is None:
            a, target = choose(game, s, net, algo, n, rng, train=True, temp_moves=temp_moves)
            hist.append((game.encode(s), target))
            s, outcome = game.play(s, a)
        z = outcome                                # for the player to move at the end; alternate backward
        for x, target in reversed(hist):
            z = -z
            X.append(x); PI.append(target); Z.append(z)
    return np.array(X), np.array(PI, dtype=np.float32), np.array(Z, dtype=np.float32)


def mirror_c4(X, PI):
    """Connect Four is symmetric under reflection of the columns."""
    Xm = X.reshape(-1, 2, 7, 6)[:, :, ::-1, :].reshape(len(X), -1)
    return Xm, PI[:, ::-1]


def train_agent(game, algo, n, iters, games_per_iter, hidden, seed, pool, temp_moves, window=20, steps=100,
                snapshots=(), augment=False, channels=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Net(game, hidden, channels)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    data, snaps = [], {0: net.numpy_weights()}
    for it in range(1, iters + 1):
        w = net.numpy_weights()
        jobs = [(game, w, algo, n, games_per_iter // 2, int(rng.integers(1 << 30)), temp_moves) for _ in range(2)]
        parts = pool.map(self_play, jobs) if pool else list(map(self_play, jobs))
        X = np.concatenate([p[0] for p in parts]); PI = np.concatenate([p[1] for p in parts])
        Z = np.concatenate([p[2] for p in parts])
        if augment:
            Xm, PIm = mirror_c4(X, PI)
            X, PI, Z = np.concatenate([X, Xm]), np.concatenate([PI, PIm]), np.concatenate([Z, Z])
        data = (data + [(X, PI, Z)])[-window:]
        X, PI, Z = (torch.as_tensor(np.concatenate([d[k] for d in data])) for k in range(3))
        for _ in range(steps):
            idx = torch.as_tensor(rng.integers(len(X), size=256))
            logits, v = net(X[idx])
            loss = F.mse_loss(v, Z[idx]) - (PI[idx] * F.log_softmax(logits, -1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if it in snapshots:
            snaps[it] = net.numpy_weights()
    snaps[iters] = net.numpy_weights()
    return snaps


# ---------------------------------------------------------------- tic-tac-toe: exact evaluation
@lru_cache(maxsize=None)
def solve(s):
    """Exact value of a tic-tac-toe position for the player to move, and the values of its moves."""
    q = {}
    for a in TicTacToe.legal(s):
        s2, outcome = TicTacToe.play(s, a)
        q[a] = -(outcome if outcome is not None else solve(s2)[0])
    return max(q.values()), tuple(sorted(q.items()))


def all_positions():
    seen, stack = set(), [TicTacToe.initial()]
    while stack:
        s = stack.pop()
        if s in seen:
            continue
        seen.add(s)
        for a in TicTacToe.legal(s):
            s2, outcome = TicTacToe.play(s, a)
            if outcome is None:
                stack.append(s2)
    return sorted(seen)


POSITIONS = None


def ttt_eval(args):
    """Fraction of all positions in which the agent picks an optimal move, and its losses in 100 games against a
    perfect player who picks uniformly among optimal moves (50 as the first player, 50 as the second)."""
    weights, algo, n, seed = args
    rng, net = np.random.default_rng(seed), NumpyNet(weights)
    good = 0
    for s in POSITIONS:
        v, q = solve(s)
        a, _ = choose(TicTacToe, s, net, algo, n, rng)
        good += dict(q)[a] == v
    losses = 0
    for g in range(100):
        s, outcome, agent_turn = TicTacToe.initial(), None, g % 2 == 0
        while outcome is None:
            if agent_turn:
                a, _ = choose(TicTacToe, s, net, algo, n, rng)
            else:
                v, q = solve(s)
                a = int(rng.choice([b for b, x in q if x == v]))
            s, outcome = TicTacToe.play(s, a)
            if outcome == -1 and agent_turn is False:
                losses += 1
            agent_turn = not agent_turn
    return good / len(POSITIONS), losses


# ---------------------------------------------------------------- Connect Four: opponents and matches
def tactical_move(s, rng):
    """Win if possible; otherwise block the opponent's immediate win; otherwise avoid moves that let the opponent
    win at once; otherwise a random move, preferring the center columns."""
    legal = ConnectFour.legal(s)
    for a in legal:
        if ConnectFour.play(s, a)[1] == -1:
            return a
    opp = (s[1], s[0], s[2])
    for a in legal:
        if ConnectFour.play(opp, a)[1] == -1:
            return a
    safe = [a for a in legal
            if not any(ConnectFour.play(ConnectFour.play(s, a)[0], b)[1] == -1 for b in ConnectFour.legal(ConnectFour.play(s, a)[0]))]
    cand = safe or legal
    w = np.array([4 - abs(3 - a) for a in cand], dtype=float)
    return int(rng.choice(cand, p=w / w.sum()))


def rollout_mcts_move(s, rng, n=400, c=1.4):
    """Plain UCT with uniformly random rollouts, the classic baseline."""
    class U:
        __slots__ = ("s", "kids", "untried", "N", "W", "outcome")

        def __init__(self, s, outcome):
            self.s, self.outcome, self.kids, self.N, self.W = s, outcome, {}, 0, 0.0
            self.untried = [] if outcome is not None else ConnectFour.legal(s)

    root = U(s, None)
    for _ in range(n):
        node, path = root, [root]
        while not node.untried and node.outcome is None:
            node = max(node.kids.values(), key=lambda k: k.W / k.N + c * math.sqrt(math.log(node.N) / k.N))
            path.append(node)
        if node.untried and node.outcome is None:
            a = node.untried.pop(int(rng.integers(len(node.untried))))
            s2, out = ConnectFour.play(node.s, a)
            node.kids[a] = U(s2, out); node = node.kids[a]; path.append(node)
        # value for the player to move at node
        if node.outcome is not None:
            v = node.outcome
        else:
            st, out, sign = node.s, None, 1
            while out is None:
                lg = ConnectFour.legal(st)
                st, out = ConnectFour.play(st, lg[int(rng.integers(len(lg)))])
                sign = -sign
            v = out * sign                          # out is for the player to move at st; flip back to node
        for nd in reversed(path):
            nd.N += 1; nd.W += -v                   # W from the view of the player who moved into nd
            v = -v
    return max(root.kids, key=lambda a: root.kids[a].N)


def c4_match(args):
    """Games of an agent against an opponent, half with each color, from 20 random two-move openings played both
    ways when the opponent is deterministic. Returns the agent's score (win 1, draw 0.5)."""
    weights, algo, n, opponent, games, seed, opp_args = args
    rng, net = np.random.default_rng(seed), NumpyNet(weights)
    opp_net = NumpyNet(opp_args[0]) if opponent in ("puct", "gumbel", "policy") else None
    score = 0.0
    for g in range(games):
        s, outcome = ConnectFour.initial(), None
        orng = np.random.default_rng(seed + g // 2)          # the same opening for both colors
        for _ in range(2):
            s, outcome = ConnectFour.play(s, int(orng.integers(7)))
        agent_turn = g % 2 == 0
        while outcome is None:
            if agent_turn:
                a, _ = choose(ConnectFour, s, net, algo, n, rng)
            elif opponent == "random":
                a = int(rng.choice(ConnectFour.legal(s)))
            elif opponent == "tactical":
                a = tactical_move(s, rng)
            elif opponent == "rollout":
                a = rollout_mcts_move(s, rng, n=opp_args[1])
            else:
                a, _ = choose(ConnectFour, s, opp_net, opponent, opp_args[1], rng)
            s, outcome = ConnectFour.play(s, a)
            if outcome == -1:
                score += 1.0 if agent_turn else 0.0
            elif outcome == 0:
                score += 0.5
            agent_turn = not agent_turn
    return score


def c4_score(pool, weights, algo, n, opponent, games, seed, opp_args=(None, 0)):
    parts = pool.map(c4_match, [(weights, algo, n, opponent, games // 2, seed + 1000 * k, opp_args) for k in range(2)])
    return sum(parts) / games


if __name__ == "__main__":
    POSITIONS = all_positions()
    pool = mp.get_context("fork").Pool(2)

    print("=== Part 1: AlphaZero on tic-tac-toe (self-play with 100 simulations per move, 100 games per iteration) ===")
    print("  fraction of all 4,520 positions in which the move is optimal, and losses in 100 games against a perfect")
    print("  player; the raw policy network, and PUCT search with 100 simulations")
    snaps = train_agent(TicTacToe, "puct", 100, iters=40, games_per_iter=100, hidden=128, seed=0, pool=pool,
                        temp_moves=4, steps=200, snapshots=(5, 10, 20))
    its = (0, 5, 10, 20, 40)
    res = pool.map(ttt_eval, [(snaps[it], algo, n, 1) for it in its for algo, n in (("policy", 0), ("puct", 100))])
    print("    iteration    policy: optimal   losses    search: optimal   losses")
    for k, it in enumerate(its):
        (p_opt, p_loss), (s_opt, s_loss) = res[2 * k], res[2 * k + 1]
        print(f"  {it:11d}   {p_opt:15.3f}   {p_loss:6d}   {s_opt:15.3f}   {s_loss:6d}")
    print(f"  (time so far {time.time() - t0:.0f} s)")

    print("\n=== Part 2: AlphaZero on Connect Four (50 simulations per move, 60 iterations of 100 games) ===")
    c4 = train_agent(ConnectFour, "puct", 50, iters=60, games_per_iter=100, hidden=256, seed=0, pool=pool,
                     temp_moves=8, steps=400, snapshots=(10, 20, 40), augment=True)
    print(f"  (training took {time.time() - t0:.0f} s so far)")
    print("  score (win 1, draw 1/2) from random two-move openings, both colors: the raw policy against the tactical")
    print("  player, and search with 50 simulations against random, tactical, and UCT with 400 random rollouts")
    print("    iteration   policy v tactical   search v random   search v tactical   search v UCT-400")
    for it in (0, 10, 20, 40, 60):
        w = c4[it]
        r = (c4_score(pool, w, "policy", 0, "tactical", 100, 10), c4_score(pool, w, "puct", 50, "random", 50, 20),
             c4_score(pool, w, "puct", 50, "tactical", 100, 30), c4_score(pool, w, "puct", 50, "rollout", 50, 40, (None, 400)))
        print(f"  {it:11d}" + "".join(f"{x:18.2f}" for x in r))
    print("  the final network with more search, against UCT with 400 random rollouts (50 games each)")
    for n in (0, 50, 200, 800):
        sc = c4_score(pool, c4[60], "puct" if n else "policy", n, "rollout", 50, 50, (None, 400))
        print(f"    {n:4d} simulations: {sc:.2f}")
    print(f"  (time so far {time.time() - t0:.0f} s)")

    print("\n=== Part 3a: training tic-tac-toe with 2 simulations per move: PUCT against Gumbel search, 3 seeds ===")
    print("  after 40 iterations of 100 games: optimal-move fraction and losses against the perfect player")
    print("                    policy: optimal   losses    search: optimal   losses")
    for algo in ("puct", "gumbel"):
        rows = []
        for seed in range(3):
            w = train_agent(TicTacToe, algo, 2, iters=40, games_per_iter=100, hidden=128, seed=seed, pool=pool,
                            temp_moves=4, steps=200)[40]
            rows.append(pool.map(ttt_eval, [(w, "policy", 0, 1), (w, algo, 2, 1)]))
        for seed, ((p_opt, p_loss), (s_opt, s_loss)) in enumerate(rows):
            print(f"  {algo:6s} seed {seed}   {p_opt:15.3f}   {p_loss:6d}   {s_opt:15.3f}   {s_loss:6d}")
    print(f"  (time so far {time.time() - t0:.0f} s)")

    print("\n=== Part 3b: Gumbel against PUCT search with the same Connect Four network and budget (100 games) ===")
    for n in (2, 4, 8, 16, 32, 64):
        sc = c4_score(pool, c4[60], "gumbel", n, "puct", 100, 60, (c4[60], n))
        print(f"  {n:3d} simulations each: Gumbel scores {sc:.2f}")
    print(f"\ntotal time {time.time() - t0:.0f} s")
    pool.close()
