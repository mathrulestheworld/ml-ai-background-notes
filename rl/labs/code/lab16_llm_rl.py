"""Lab 16 reference solution: reinforcement learning with a verifier for a small language model.

Run from this folder:  python lab16_llm_rl.py
It prints the results quoted on the lab page. Takes about 25 minutes on two cores.
"""
import copy
import math
import multiprocessing as mp
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

t0 = time.time()

# ---------------------------------------------------------------- the task
# A prompt is a list of n digits (2 <= n <= 8); the answer is the final state of s <- (3 s + d) mod 10, starting
# from s = 0. The model may answer directly, "> a", or first write the intermediate states, ": s1 s2 ... sn > a".
# Each intermediate step is a small lookup, but the direct answer is a weighted sum of all digits modulo 10.
PAD, EQ, THINK, ANS, EOS = 10, 11, 12, 13, 14
VOCAB, NMAX = 15, 8
PLEN = NMAX + 1                                            # prompts are left-padded to 9 tokens
RMAX = NMAX + 4                                            # longest response: ':', n states, '>', answer, end


def states(digits):
    s, out = 0, []
    for d in digits:
        s = (3 * s + d) % 10
        out.append(s)
    return out


def make_prompt(digits):
    return [PAD] * (NMAX - len(digits)) + list(digits) + [EQ]


def make_response(digits, think):
    st = states(digits)
    return ([THINK] + st if think else []) + [ANS, st[-1], EOS]


def reward(digits, resp):
    """1 if the response ends with '> a <end>' and a is the right answer; the reasoning is not checked."""
    if EOS not in resp:
        return 0.0
    body = resp[:resp.index(EOS)]
    return float(len(body) >= 2 and body[-2] == ANS and body[-1] == states(digits)[-1])


# ---------------------------------------------------------------- the model
class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.heads = heads

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, d=96, layers=3, heads=4):
        super().__init__()
        self.tok, self.pos = nn.Embedding(VOCAB, d), nn.Embedding(PLEN + RMAX, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, VOCAB)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln(h))


@torch.no_grad()
def sample(model, prompts, temperature=1.0):
    """Sample responses for a batch of prompts (a LongTensor of shape (B, 9)); returns the response tokens,
    padded after the end token with the end token."""
    seq, done = prompts, torch.zeros(len(prompts), dtype=torch.bool)
    for _ in range(RMAX):
        logits = model(seq)[:, -1] / temperature
        nxt = torch.multinomial(F.softmax(logits, -1), 1)[:, 0]
        nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
        seq = torch.cat([seq, nxt[:, None]], 1)
        done |= nxt == EOS
        if done.all():
            break
    return seq[:, PLEN:]


def token_logprobs(model, prompts, resp):
    """Log-probabilities of the response tokens, and a mask of the tokens up to and including the end token."""
    seq = torch.cat([prompts, resp], 1)
    logp = F.log_softmax(model(seq[:, :-1])[:, PLEN - 1:], -1).gather(-1, resp[..., None])[..., 0]
    ended = torch.cumsum((resp == EOS).long(), 1)
    mask = (ended == 0) | ((ended == 1) & (resp == EOS))
    return logp, mask.float()


def random_problems(rng, B, lengths=range(2, NMAX + 1)):
    ns = rng.choice(list(lengths), size=B)
    return [list(rng.integers(10, size=n)) for n in ns]


# ---------------------------------------------------------------- pretraining (supervised fine-tuning)
def pretrain(seed, steps=4000, p_think=0.1):
    """Train on correct responses, 90% of them direct answers and 10% with the intermediate states written out."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = GPT()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
    for step in range(steps):
        probs = random_problems(rng, 256)
        resp = [make_response(p, rng.random() < p_think) for p in probs]
        resp = [r + [EOS] * (RMAX - len(r)) for r in resp]
        prompts = torch.tensor([make_prompt(p) for p in probs]); resp = torch.tensor(resp)
        logp, mask = token_logprobs(model, prompts, resp)
        loss = -(logp * mask).sum() / mask.sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model


def evaluate(model, rng, per_length=200, temperature=1.0):
    """Accuracy, rate of writing the intermediate states, and mean response length, for each problem length."""
    out = {}
    for n in range(2, NMAX + 1):
        probs = [list(rng.integers(10, size=n)) for _ in range(per_length)]
        resp = sample(model, torch.tensor([make_prompt(p) for p in probs]), temperature)
        r = [reward(p, x.tolist()) for p, x in zip(probs, resp)]
        think = (resp[:, 0] == THINK).float().mean().item()
        length = ((torch.cumsum((resp == EOS).long(), 1) == 0).sum(1) + 1).float().mean().item()
        out[n] = (np.mean(r), think, length)
    return out


# ---------------------------------------------------------------- RL with a verifier
def rl(args):
    """Policy-gradient fine-tuning with G samples per prompt and one gradient step per batch (so no importance
    ratios are needed). Advantages and token weights:
      REINFORCE: r - batch mean, each token weighted 1 / (B G RMAX);
      RLOO:      r - mean of the other G - 1 samples of the prompt, same weights;
      GRPO:      (r - group mean) / group std, each token weighted 1 / (B G |response|);
      Dr. GRPO:  r - group mean, each token weighted 1 / (B G RMAX);
    and Dr. GRPO with a cost of 0.02 per response token subtracted from the reward."""
    method, base_state, seed, steps = args
    torch.set_num_threads(1); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = GPT(); model.load_state_dict(base_state)
    ref = GPT(); ref.load_state_dict(base_state)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    B, G = 32, 8
    curve = []
    for step in range(steps + 1):
        probs = random_problems(rng, B)
        prompts = torch.tensor([make_prompt(p) for p in probs]).repeat_interleave(G, 0)
        resp = sample(model, prompts)
        correct = torch.tensor([reward(probs[i // G], x.tolist()) for i, x in enumerate(resp)]).view(B, G)
        length = ((torch.cumsum((resp == EOS).long(), 1) == 0).sum(1) + 1).float().view(B, G)
        r = correct - (0.02 * length if method.endswith("length cost") else 0.0)
        if step % 25 == 0:
            think = (resp[:, 0] == THINK).float().mean().item()
            with torch.no_grad():
                lp, mask = token_logprobs(model, prompts, resp); lr_, _ = token_logprobs(ref, prompts, resp)
                kl = ((lp - lr_) * mask).sum(1).mean().item()
            curve.append((step, correct.mean().item(), think, kl))
        if step == steps:
            break
        if method == "REINFORCE":
            adv = r - r.mean()
        elif method == "RLOO":
            adv = (r - (r.sum(1, keepdim=True) - r) / (G - 1))
        elif method == "GRPO":
            adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)
        else:                                                 # Dr. GRPO, with or without the length cost
            adv = r - r.mean(1, keepdim=True)
        lp, mask = token_logprobs(model, prompts, resp)
        if method == "GRPO":
            w = mask / mask.sum(1, keepdim=True)
        else:
            w = mask / RMAX
        loss = -(adv.view(-1, 1) * w * lp).sum() / (B * G)
        opt.zero_grad(); loss.backward(); opt.step()
    return curve, {k: v.clone() for k, v in model.state_dict().items()}


def pass_at_k(model, rng, n_problems=100, n_samples=64, ks=(1, 8, 64), lengths=(4, 6, 8)):
    """Unbiased pass@k from n_samples per problem (Chen et al., 2021), for problems of each length."""
    out = {}
    for n in lengths:
        vals = np.zeros(len(ks))
        for _ in range(n_problems):
            p = list(rng.integers(10, size=n))
            resp = sample(model, torch.tensor([make_prompt(p)] * n_samples))
            c = sum(reward(p, x.tolist()) for x in resp)
            vals += [1 - math.comb(n_samples - int(c), k) / math.comb(n_samples, k) for k in ks]
        out[n] = vals / n_problems
    return out


if __name__ == "__main__":
    torch.set_num_threads(2)
    print("=== Part 1: a small pretrained model (3 layers, width 96; 4,000 steps on 90% direct answers) ===")
    base = pretrain(seed=0)
    rng = np.random.default_rng(1)
    ev = evaluate(base, rng)
    print("  sampled at temperature 1: accuracy, rate of writing the intermediate states, and mean length")
    print("  digits      " + "".join(f"{n:7d}" for n in range(2, NMAX + 1)))
    for k, name in enumerate(("accuracy", "writes states", "length")):
        print(f"  {name:12s}" + "".join(f"{ev[n][k]:7.2f}" for n in range(2, NMAX + 1)))
    print(f"  (time so far {time.time() - t0:.0f} s)")

    print("\n=== Part 2: RL with a verifier, 200 steps of 32 prompts x 8 samples, 2 seeds ===")
    torch.set_num_threads(1)
    state = {k: v.clone() for k, v in base.state_dict().items()}
    methods = ("REINFORCE", "RLOO", "GRPO", "Dr. GRPO", "Dr. GRPO, length cost")
    with mp.get_context("fork").Pool(2) as pool:
        res = pool.map(rl, [(m, state, seed, 200) for m in methods for seed in range(2)])
    print("  accuracy / rate of writing the states / KL from the pretrained model (nats per response), mean of seeds")
    print("  step " + "".join(f"{m:>24s}" for m in ("REINFORCE", "RLOO", "GRPO", "Dr. GRPO", "Dr. GRPO + cost")))
    for k in range(len(res[0][0])):
        row = f"  {res[0][0][k][0]:4d}"
        for j in range(len(methods)):
            c = np.mean([res[2 * j + s][0][k][1:] for s in range(2)], 0)
            row += f"      {c[0]:.2f} / {c[1]:.2f} / {c[2]:4.1f}"
        print(row)
    print(f"  (time so far {time.time() - t0:.0f} s)")

    print("\n=== Part 3: what RL changed (seed 0), by number of digits ===")
    models = {"pretrained": base}
    for name, j in (("Dr. GRPO", 3), ("+ length cost", 4)):
        models[name] = GPT(); models[name].load_state_dict(res[2 * j][1])
    evs = {name: evaluate(m, np.random.default_rng(1)) for name, m in models.items()}
    print("  digits                       " + "".join(f"{n:7d}" for n in range(2, NMAX + 1)))
    for k, what in enumerate(("accuracy", "writes states", "length")):
        for name in models:
            print(f"  {what if name == 'pretrained' else '':14s} {name:14s}" + "".join(f"{evs[name][n][k]:7.2f}" for n in range(2, NMAX + 1)))
    print("  pass@k from 64 samples per problem, 100 problems per length")
    pk0, pk1 = pass_at_k(base, np.random.default_rng(2)), pass_at_k(models["Dr. GRPO"], np.random.default_rng(2))
    print("  digits      model        pass@1    pass@8   pass@64")
    for n in pk0:
        print(f"  {n:6d}      pretrained " + "".join(f"{x:9.3f}" for x in pk0[n]))
        print(f"  {'':6s}      Dr. GRPO   " + "".join(f"{x:9.3f}" for x in pk1[n]))
    print(f"\ntotal time {time.time() - t0:.0f} s")
