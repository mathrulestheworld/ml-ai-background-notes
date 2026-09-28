[Background Notes](../../README.md) › [Reinforcement Learning](../README.md)

# Lab 16. Reinforcement Learning for a Small Language Model

[← Lab 15. Regret Minimization and Self-Play in Poker](lab-15-regret-minimization-and-self-play-in-poker.md) · [Lab 17. Safe and Robust Reinforcement Learning →](lab-17-safe-and-robust-reinforcement-learning.md)

## <a id="overview"></a>Overview

This lab puts chapter 28 to work on a language model small enough to train on a laptop. You will pretrain a three-layer transformer on a sequential arithmetic task in which it mostly answers directly and only occasionally writes out its intermediate steps, then fine-tune it with reinforcement learning against a verifier that checks only the final answer. What happens is a miniature of what happened with reasoning models: the model discovers that writing out its steps pays, its responses grow longer, and its accuracy jumps. You will compare four policy-gradient estimators, add a cost for length, and measure what the training did to pass@$`k`$.

- **Task:** a prompt is a list of $`n`$ digits, $`2\le n\le8`$, and the answer is the final state of $`s\leftarrow(3s+d)\bmod10`$, starting from $`s=0`$ and applying each digit $`d`$ in turn. A response is either a direct answer, `> a <end>`, or a worked one, `: s1 s2 ... sn > a <end>`, which writes out every intermediate state. Each step of the worked form is a small table lookup, while the direct answer requires a weighted sum of all the digits modulo 10, which a small transformer learns only for short lists. The reward is 1 if the response ends with the correct answer, whatever came before.
- **Prerequisites:** chapter 28, DL chapter 9 for the transformer; PyTorch.
- **Reference solution:** [lab16_llm_rl.py](code/lab16_llm_rl.py), about 25 minutes on two cores: 7 for pretraining, 17 for reinforcement learning. Try each part yourself before reading it.

## <a id="part-1-a-small-pretrained-model"></a>Part 1 — A small pretrained model

1. **Tokens and format.** Use 15 tokens: the ten digits, padding, `=`, `:` (start working), `>` (answer), and an end token. Left-pad every prompt to 8 digits followed by `=`, so that all prompts have 9 tokens, and allow responses of up to 12 tokens.
2. **The model.** A decoder-only transformer with learned position embeddings, 3 layers, width 96, 4 heads, pre-layer normalization, and causal attention.
3. **Pretraining.** Train with the next-token loss on the response tokens only (up to and including the end token), on random problems whose responses are direct answers 90% of the time and worked ones 10% of the time: 4,000 steps of batches of 256, AdamW with a one-cycle schedule peaking at $`10^{-3}`$. The model learns both formats and uses them in the proportions it saw.
4. **Evaluation.** For each list length, sample 200 responses at temperature 1 and measure the accuracy, the fraction of responses that write out the states, and the mean response length.

## <a id="part-2-reinforcement-learning-with-a-verifier"></a>Part 2 — Reinforcement learning with a verifier

1. Each step, sample 32 random problems and 8 responses to each from the current policy, and score them with the verifier. Take one gradient step per batch, so that no importance ratios are needed, with Adam at step size $`3\times10^{-4}`$, for 200 steps.
2. Implement four estimators, differing only in the advantage and in how tokens are weighted: **REINFORCE** with the batch mean as baseline; **RLOO**, with the mean of the other 7 responses to the same prompt as baseline; **GRPO**, with advantages normalized by the group's standard deviation and each response's loss averaged over its own tokens; and **Dr. GRPO**, with the group mean as baseline and every token weighted by the same constant. Add a fifth run, Dr. GRPO with a cost of 0.02 per response token subtracted from the reward.
3. Every 25 steps, record the accuracy, the fraction of responses that write out the states, and the KL divergence of the policy from the pretrained model on the sampled responses. Use 2 seeds.

## <a id="part-3-what-reinforcement-learning-changed"></a>Part 3 — What reinforcement learning changed

1. Evaluate the Dr. GRPO policies, with and without the length cost, by list length, as in part 1.
2. Estimate pass@$`k`$ for $`k=1,8,64`$ with the unbiased estimator of exercise 28.7, from 64 samples on each of 100 problems of lengths 4, 6, and 8, for the pretrained model and the Dr. GRPO policy.

## <a id="expected-results"></a>Expected results

```text
=== Part 1: a small pretrained model (3 layers, width 96; 4,000 steps on 90% direct answers) ===
  sampled at temperature 1: accuracy, rate of writing the intermediate states, and mean length
  digits            2      3      4      5      6      7      8
  accuracy       0.99   0.41   0.28   0.29   0.28   0.20   0.23
  writes states   0.08   0.11   0.12   0.14   0.09   0.08   0.08
  length         3.24   3.44   3.60   3.81   3.60   3.64   3.67

=== Part 2: RL with a verifier, 200 steps of 32 prompts x 8 samples, 2 seeds ===
  accuracy / rate of writing the states / KL from the pretrained model (nats per response), mean of seeds
  step                REINFORCE                    RLOO                    GRPO                Dr. GRPO         Dr. GRPO + cost
     0      0.42 / 0.10 /  0.0      0.42 / 0.10 /  0.0      0.42 / 0.10 /  0.0      0.42 / 0.10 /  0.0      0.42 / 0.10 /  0.0
    25      0.88 / 0.95 /  2.6      0.90 / 0.96 /  2.7      0.86 / 0.93 /  2.7      0.89 / 0.95 /  2.7      0.88 / 0.95 /  2.5
    50      0.93 / 0.99 /  2.4      0.94 / 0.99 /  2.4      0.93 / 0.99 /  2.5      0.94 / 0.99 /  2.4      0.92 / 0.99 /  2.4
    75      0.93 / 0.99 /  2.5      0.95 / 1.00 /  2.3      0.94 / 1.00 /  2.3      0.92 / 0.99 /  2.4      0.95 / 0.99 /  2.3
   100      0.96 / 1.00 /  2.4      0.97 / 1.00 /  2.4      0.96 / 1.00 /  2.5      0.97 / 1.00 /  2.4      0.96 / 1.00 /  2.4
   125      0.96 / 1.00 /  2.4      0.98 / 1.00 /  2.3      0.96 / 1.00 /  2.5      0.98 / 1.00 /  2.4      0.97 / 1.00 /  2.3
   150      0.95 / 1.00 /  2.5      0.96 / 1.00 /  2.5      0.98 / 1.00 /  2.4      0.97 / 1.00 /  2.5      0.99 / 1.00 /  2.4
   175      0.95 / 1.00 /  2.6      0.98 / 1.00 /  2.4      0.96 / 1.00 /  2.4      0.98 / 1.00 /  2.4      0.97 / 1.00 /  2.5
   200      0.99 / 1.00 /  2.4      0.97 / 1.00 /  2.5      0.98 / 1.00 /  2.5      0.99 / 1.00 /  2.5      0.96 / 1.00 /  2.5

=== Part 3: what RL changed (seed 0), by number of digits ===
  digits                             2      3      4      5      6      7      8
  accuracy       pretrained       1.00   0.40   0.28   0.32   0.27   0.27   0.23
                 Dr. GRPO         0.99   1.00   0.99   0.97   0.98   0.96   0.96
                 + length cost    1.00   0.98   0.99   0.98   0.93   0.94   0.96
  writes states  pretrained       0.08   0.10   0.11   0.12   0.06   0.10   0.07
                 Dr. GRPO         1.00   1.00   1.00   1.00   1.00   1.00   1.00
                 + length cost    1.00   1.00   0.99   1.00   1.00   1.00   1.00
  length         pretrained       3.22   3.42   3.55   3.75   3.45   3.84   3.63
                 Dr. GRPO         6.03   7.03   8.01   8.98  10.02  11.01  12.00
                 + length cost    6.03   6.99   7.95   9.01   9.99  10.97  11.96
  pass@k from 64 samples per problem, 100 problems per length
  digits      model        pass@1    pass@8   pass@64
       4      pretrained     0.283    0.921    1.000
              Dr. GRPO       0.989    1.000    1.000
       6      pretrained     0.281    0.928    1.000
              Dr. GRPO       0.985    1.000    1.000
       8      pretrained     0.263    0.904    1.000
              Dr. GRPO       0.965    1.000    1.000
```

Things to notice:

- **Part 1.** The pretrained model answers two-digit problems perfectly but gets only 20% to 41% of longer ones right (chance is 10%), although it can do much better: it writes out its steps only about 10% of the time, the rate at which it saw worked answers, and part 2 shows that the worked form, once used, is almost always right. Its competence is in the model, but its default behavior does not use it.
- **Part 2, learning to show its work.** Within 25 steps, 200 batches of problems, every estimator raises the rate of writing out the states from 10% to 93–96% and the accuracy from 42% to 86–90%; after 200 steps, the model works out every problem and is right 96–99% of the time. Nothing rewarded the intermediate steps; the verifier checks only the answer, and working the problem out is simply the behavior that makes the answer right. This is the mechanism by which RL with verifiable rewards lengthens the responses of reasoning models, here in its simplest form.
- **Part 2, the estimators.** On this task the five runs are indistinguishable: the signal is strong, since a worked answer is right more than 90% of the time and a direct answer to a list of three or more digits only 20% to 40%, and the lengths are fixed by the problems, so GRPO's length bias has nothing to act on. The KL divergence from the pretrained model rises to about 2.5 nats per response within 25 steps and stays there: most of the change is the one decision to start working, $`-\ln0.1\approx2.3`$ nats.
- **Part 3, overthinking.** The trained model writes out its steps even for two-digit problems, which it answered correctly and in half the tokens before. With the length cost, the direct answer is better on those problems by $`0.02\times3=0.06`$ reward, yet the model does not go back to it: once it almost always works problems out, it almost never samples a direct answer to a two-digit problem, so it never learns that the shortcut is now better. The collapse of exploration of chapter 28 locks in a habit, and the small cost is not enough to overcome it within 200 steps.
- **Part 3, pass@$`k`$.** RL raised pass@1 from about 0.28 to 0.97–0.99, but pass@64 was already 1.000: the pretrained model, sampled 64 times, nearly always produced a worked answer somewhere among the samples. RL made the model use reliably what it could already do occasionally, the pattern of Yue et al.; here nothing was lost at large $`k`$, since every problem remained solvable in one sample.

## <a id="going-further"></a>Going further

1. **Rarer demonstrations.** Pretrain with worked answers only 1% or 0.1% of the time. How does the speed of RL depend on this rate, and do the estimators start to differ once the right behavior is rarely sampled?
2. **Breaking the habit.** Raise the length cost, add an entropy bonus, or sample at a higher temperature during training. Which lets the model return to direct answers on two-digit problems while keeping worked answers on longer ones?
3. **KL regularization.** Add a KL penalty toward the pretrained model, in the reward with $`k_1`$ and in the loss with $`k_3`$ as in chapter 28's first code, at several strengths. Which prevents the switch to worked answers, and at what cost in accuracy?
4. **Reward hacking.** Replace the verifier by a weaker one that checks only that the response has the right format, or that rewards the answer digit appearing anywhere in the response. What does the policy learn?
5. **Length bias.** Make the worked form variable in length, for instance by allowing the model to repeat a state, and compare GRPO with Dr. GRPO. Do correct and incorrect responses drift as in chapter 28's second code?

---

[← Lab 15. Regret Minimization and Self-Play in Poker](lab-15-regret-minimization-and-self-play-in-poker.md) · [Lab 17. Safe and Robust Reinforcement Learning →](lab-17-safe-and-robust-reinforcement-learning.md)
