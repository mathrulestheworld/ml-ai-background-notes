[Background Notes](../../README.md) › [Reinforcement Learning](../README.md)

# Lab 2. Bandit Algorithms in Practice

[← Lab 1. Dynamic Programming on FrozenLake and Taxi](lab-01-dynamic-programming-on-frozenlake-and-taxi.md) · [Lab 3. Tabular Control with Monte Carlo and TD Methods →](lab-03-tabular-control-with-monte-carlo-and-td-methods.md)

## <a id="overview"></a>Overview

This lab turns [chapter 3](../03-multi-armed-bandits.md) and [chapter 4](../04-contextual-bayesian-and-adversarial-bandits.md) into working code: a small library of bandit algorithms with a common interface, stress tests that break the stochastic assumptions, a contextual bandit built from a real dataset, and off-policy evaluation of logged decisions.

- **Data:** Bernoulli arms generated in the code, and the $`8\times8`$ handwritten digits bundled with scikit-learn, used as a 10-action contextual bandit.
- **Prerequisites:** chapters 3–4; NumPy and scikit-learn.
- **Reference solution:** [lab02_bandits.py](code/lab02_bandits.py), about a minute and a half on one core. Try each part yourself before reading it.

## <a id="part-1-a-bandit-library"></a>Part 1 — A bandit library

1. Write a base class with `select(t)` and `update(a, r)` and implement ε-greedy, UCB1, KL-UCB, Thompson sampling with Beta posteriors, and Exp3 (loss-based, $`\eta=\sqrt{2\ln k/(Tk)}`$).
2. Write a runner that takes a function giving the arms' means at each step, so that the same runner can simulate stationary and changing problems, and that measures the pseudo-regret $`\sum_t(\mu^*_t-\mu_t(A_t))`$.
3. On the five Bernoulli arms of chapter 3 (means 0.5, 0.45, 0.4, 0.3, 0.2), report the regret after 10,000 pulls as a mean with a 95% confidence interval over 20 seeds. Which differences between algorithms are statistically clear, and which are not?

## <a id="part-2-stress-tests"></a>Part 2 — Stress tests

1. **A change point.** Two arms with means 0.6 and 0.4 swap at step 5,000. Add two forgetting algorithms, sliding-window UCB (statistics over the last 500 pulls) and discounted Thompson sampling (pseudo-counts multiplied by 0.99 every step), and compare all of them.
2. **Delayed feedback.** Apply the updates only in batches of 200 pulls, as a system that retrains once an hour would. Which of UCB1 and Thompson sampling suffers, and why?

## <a id="part-3-a-contextual-bandit-from-a-classification-dataset"></a>Part 3 — A contextual bandit from a classification dataset

Any classification dataset becomes a contextual bandit by revealing the input as the context, letting the actions be the labels, and giving reward 1 only when the chosen label is right, so the learner never sees the true label of a wrong guess.

1. Reduce the digits to 20 principal components, add a bias feature, and sample contexts with replacement for 5,000 rounds.
2. Implement the disjoint LinUCB of chapter 4 ($`\alpha=0.5`$), linear Thompson sampling, ε-greedy ($`\varepsilon=0.05`$), and greedy play on the ridge estimates, with Sherman–Morrison updates of each action's inverse Gram matrix. Add a context-free Thompson sampler as a baseline.
3. Compare their accuracy early and late with the accuracy of the same linear model trained with every label revealed.

## <a id="part-4-off-policy-evaluation"></a>Part 4 — Off-policy evaluation

1. Log 6,000 rounds with a uniformly random policy (propensity 0.1 for every action).
2. Learn a policy from the first half: one ridge regression of the reward per action on the rounds where that action was logged, and act greedily.
3. Estimate the learned policy's accuracy from the second half with the direct method, IPS, self-normalized IPS, and the doubly robust estimator, and compare with its true accuracy on the whole dataset.

## <a id="expected-results"></a>Expected results

```text
=== Part 1: stationary Bernoulli arms, regret after 10,000 pulls (mean +- 95% CI over 20 seeds) ===
  EpsGreedy            177.1 +-  32.2
  UCB1                 268.0 +-  12.6
  KLUCB                104.4 +-  11.8
  Thompson              95.9 +-  48.9
  Exp3                 260.1 +-  23.1

=== Part 2a: the best arm changes at step 5,000 (arms 0.6/0.4 swap) ===
  UCB1                 128.6 +-  15.3
  Thompson             668.7 +- 105.4
  SlidingWindowUCB     360.7 +-  10.0
  DiscountedThompson   242.8 +-   7.8
  Exp3                 822.0 +-  86.7

=== Part 2b: delayed feedback, updates in batches of 200 pulls ===
  UCB1               batch 1:   268.0 +-  12.6   batch 200:   325.1 +-  13.0
  Thompson           batch 1:    95.9 +-  48.9   batch 200:    76.3 +-  12.1

=== Part 3: digits as a 10-action contextual bandit (reward 1 for the right label) ===
  full-information references: per-action ridge regression on all labels 0.907, logistic regression (5-fold CV) 0.905
  LinUCB                 accuracy in rounds 1-1,000: 0.342   rounds 4,001-5,000: 0.836
  linear Thompson        accuracy in rounds 1-1,000: 0.392   rounds 4,001-5,000: 0.907
  epsilon-greedy         accuracy in rounds 1-1,000: 0.250   rounds 4,001-5,000: 0.731
  greedy                 accuracy in rounds 1-1,000: 0.245   rounds 4,001-5,000: 0.570
  context-free Thompson  accuracy in rounds 1-1,000: 0.114   rounds 4,001-5,000: 0.102

=== Part 4: evaluating a policy learned from 3,000 uniformly logged rounds ===
  true accuracy of the learned policy on all 1,797 digits: 0.881
  estimates from 3,000 held-out logged rounds: DM 0.386, IPS 0.943, SNIPS 0.901, DR 0.936
```

Things to notice:

- **Part 1.** KL-UCB and Thompson sampling have the lowest regret, as in chapter 3, but Thompson sampling's interval is wide: in a few of the 20 runs it spends a long time on the second-best arm, whose mean is only 0.05 lower. UCB1's conservative bonus costs it more than a well-chosen ε costs ε-greedy at this horizon; at longer horizons ε-greedy's linear regret would overtake it. Twenty seeds are enough to separate UCB1 from KL-UCB but not KL-UCB from Thompson sampling; reporting intervals is what makes that visible.
- **Part 2a.** UCB1 adapts to a single swap surprisingly well: the arm that was bad has been pulled only about 250 times against 4,750 for the other, so after the change its bonus soon lets it be tried again, while the old best arm's average, built from thousands of pulls, drops slowly but steadily below it. Thompson sampling, whose posterior for the losing arm has become very confident, and Exp3, whose cumulative loss estimates must first be overturned, adapt slowly. Of the forgetting algorithms, discounted Thompson sampling is best; the sliding window pays for forgetting during the stationary halves. These rankings depend on the scenario: with changes every few hundred steps, forgetting becomes essential.
- **Part 2b.** Batching hurts UCB1, which, being deterministic, pulls the same arm for all 200 steps of a batch, and does not hurt Thompson sampling, whose randomization keeps exploring within a batch, as Chapelle and Li observed.
- **Part 3.** Linear Thompson sampling learns to recognize digits from right-or-wrong feedback alone and reaches 0.907, the accuracy of the same linear model trained with every label. Greedy play fails here, unlike in exercise 4.5: with rare rewards and all estimates starting at zero, each action learns only from the contexts it happened to win early, and many actions are never tried on the digits they should claim. The context-free baseline is at chance.
- **Part 4.** The direct method is far too low, 0.39 against a true 0.88, because a linear regression of a rare binary reward gives shrunken predictions that are poor probabilities. IPS is unbiased but noisy: with weights of 0 or 10, its standard error on 3,000 rounds is about 0.05. Self-normalized IPS is closest, and the doubly robust estimator removes most of the direct method's bias.

## <a id="going-further"></a>Going further

1. **KL-UCB near the boundary.** Rerun part 1 with means 0.1, 0.05, 0.03, 0.02, 0.01, as in click-through rates. How does the gap between UCB1 and KL-UCB change, and why? (Compare the Hoeffding and KL confidence radii for small means.)
2. **Frequent changes.** Make the two arms swap every 500 steps and tune the window and the discount factor. Plot the regret against the window length and explain the trade-off.
3. **Replay evaluation.** Evaluate the whole LinUCB algorithm offline on uniformly logged data by the replay method of Li et al. (2011): step through the log, and whenever LinUCB's choice matches the logged action, feed it the reward; otherwise skip the round. Compare the accuracy it reaches with its online accuracy in part 3.
4. **A neural-linear bandit.** Replace the PCA features by the last hidden layer of a small network trained on the rounds seen so far, retrained every 500 rounds, and keep linear Thompson sampling on top. Does it beat the linear model?
5. **Gittins in practice.** For two Bernoulli arms with uniform priors and $`\gamma=0.95`$, compute Gittins indices with the chapter 4 code and compare the discounted reward of the index policy with Thompson sampling and with greedy play, averaged over the prior.

---

[← Lab 1. Dynamic Programming on FrozenLake and Taxi](lab-01-dynamic-programming-on-frozenlake-and-taxi.md) · [Lab 3. Tabular Control with Monte Carlo and TD Methods →](lab-03-tabular-control-with-monte-carlo-and-td-methods.md)
