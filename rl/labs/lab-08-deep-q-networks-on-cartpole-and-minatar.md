[ML Mastery Notes](../../README.md) › [Reinforcement Learning](../README.md)

# Lab 8. Deep Q-Networks on CartPole and MinAtar

[← Lab 7. System Identification, LQR, and Model Predictive Control](lab-07-system-identification-lqr-and-model-predictive-control.md) · [Lab 9. Advantage Actor-Critic with Parallel and Stale Actors →](lab-09-advantage-actor-critic-with-parallel-and-stale-actors.md)

## <a id="overview"></a>Overview

This lab puts chapter 16 to work. You will write DQN from scratch in PyTorch, train two dozen cart-pole agents at once by batching their networks, remove its components one at a time, and measure how far its value estimates drift above what any policy can achieve. Then you will train DQN and double DQN with a convolutional network on a miniature Atari game and compare their value estimates with the returns their policies actually obtain.

- **Environments:** the cart-pole of Lab 6, simulated in NumPy with the dynamics of Gymnasium's `CartPole-v1` for many copies at once: reward 1 per step, termination when the pole tilts more than 12° or the cart leaves $[-2.4,2.4]$, and truncation at 500 steps. And Space Invaders from [MinAtar](https://github.com/kenjyoung/MinAtar) ([Young and Tian, 2019](https://arxiv.org/abs/1903.03176)), a 10 × 10 version of the Atari game whose observation is a stack of 6 binary channels (the cannon, the aliens, their direction, and the bullets) and whose minimal action set has 4 actions (no-op, left, right, fire), with sticky actions: with probability 0.1 the previous action is repeated.
- **Prerequisites:** chapters 12 and 16; PyTorch, NumPy, and MinAtar (`pip install minatar`).
- **Reference solution:** [lab08_dqn.py](code/lab08_dqn.py), about 25 minutes on two cores, most of it in part 2. Try each part yourself before reading it.

## <a id="part-1-many-dqn-agents-on-the-cart-pole"></a>Part 1 — Many DQN agents on the cart-pole

1. Write a network for many agents at once: store each layer's weights with a leading agent dimension, shape (agents, inputs, outputs), and compute all agents' forward passes with one batched matrix product (`torch.baddbmm`). Use two hidden layers of 64 ReLU units and three outputs, read either as two action values or, for a **dueling** agent, as a state value and two advantages combined as $v+a-\bar a$.
2. Give each agent its own replay memory of 50,000 transitions, its own target network, and its own copy of the cart-pole. Train with minibatches of 64, the Huber loss, and Adam with step size $3\times10^{-4}$, written out by hand so that each agent could have its own step size. Start learning after 1,000 steps, anneal ε from 1 to 0.05 over the first 10,000 steps, use $\gamma=0.99$, and copy the network to the target every 250 steps. Bootstrap at truncations: only a real termination ends the return.
3. Train six variants with four seeds each, all 24 agents together, for 100,000 steps: DQN; DQN without a target network (a copy after every update); DQN that samples only from the most recent 64 or 5,000 transitions; double DQN; and dueling DQN.
4. Report the average length of each agent's last 10 episodes every 1,000 steps, and, as a measure of stability, the fraction of the last 50 checkpoints at which it was at least 475.
5. Track the agents' values: every 1,000 steps, compute $\max_a\hat q(s,a)$ on 256 fixed start states. With a reward of 1 per step and $\gamma=0.99$, no policy's value exceeds $1/(1-\gamma)=100$, so any prediction above 100 is an overestimate.

## <a id="part-2-dqn-and-double-dqn-on-minatar"></a>Part 2 — DQN and double DQN on MinAtar

1. Use the MinAtar network of Young and Tian: a convolution with 16 filters of size 3 × 3, a ReLU, a fully connected layer of 128 units, and one output per action.
2. Train DQN with a replay memory of 100,000 transitions, one update of 32 transitions per step, the Huber loss, Adam with step size $2.5\times10^{-4}$, a target copied every 1,000 steps, ε annealed from 1 to 0.1 over 100,000 steps, and learning from step 5,000. Train double DQN with the same settings. Run two seeds of each for 100,000 steps, in parallel.
3. Report the average return of the last 50 training episodes every 25,000 steps.
4. Evaluate each final network over 20 episodes with ε = 0.01: its average return, its predicted value of the start state $\max_a\hat q(s_0,a)$, and the actual discounted return of its episodes.

## <a id="expected-results"></a>Expected results

```text
=== Part 1: DQN variants on CartPole, 4 seeds each, trained together for 100,000 steps ===
  average length of the last 10 training episodes (at most 500), mean over seeds, and the fraction of the
  last 50 checkpoints (one per 1,000 steps) at which it was at least 475, for each seed
                                              steps: 25,000  50,000  75,000 100,000   at 475 or more
  DQN (target copied every 250 steps)                239.6   254.9   263.3   326.1   0.26 0.48 0.00 0.48
    no target network                                  9.4     9.7     9.8     9.8   0.00 0.00 0.00 0.00
    replay of the last 64 transitions only           105.3   117.8   282.3   456.0   0.24 0.00 0.08 0.30
    replay of the last 5,000 transitions only        189.4   372.1   383.8   412.2   0.74 0.18 0.52 0.80
  double DQN                                         160.9   312.9   367.5   209.7   0.30 0.22 0.16 0.24
  dueling DQN                                        199.2   307.6   229.5   120.8   0.20 0.22 0.20 0.12

  predicted value of the start states, max_a q(s, a), mean over 256 start states, for each seed
  (a reward of 1 per step and gamma = 0.99 bound every true value by 100)
                                              steps:  25,000             100,000     largest over training
  DQN (target copied every 250 steps)                65    66    65    66     112   107   156   132     119   107   159   132
    no target network                             2e+06 2e+06 2e+06 2e+06   1e+08 1e+08 8e+07 1e+08   1e+08 1e+08 8e+07 1e+08
    replay of the last 64 transitions only           67    64    64    63     102   104   102    99     104   104   102    99
    replay of the last 5,000 transitions only        63    63    63    63      99   102   100    99      99   103   100    99
  double DQN                                         66    66    68    66     113   107   145   129     113   107   145   129
  dueling DQN                                        65    66    66    66     133   113   113   146     133   115   113   146

=== Part 2: DQN and double DQN on MinAtar Space Invaders, 2 seeds each, 100,000 steps ===
  DQN        training return (last 50 episodes, mean of seeds) at 25k, 50k, 75k, 100k steps:   6.7  11.2  13.4  21.2
             seed 0: greedy return  23.4; value of the start state: predicted 13.74, actual discounted return 11.21, overestimate +2.53
             seed 1: greedy return  23.4; value of the start state: predicted 14.02, actual discounted return 10.55, overestimate +3.46
  double DQN training return (last 50 episodes, mean of seeds) at 25k, 50k, 75k, 100k steps:   6.0  10.2  15.8  20.4
             seed 0: greedy return  31.5; value of the start state: predicted 13.32, actual discounted return 12.29, overestimate +1.03
             seed 1: greedy return  25.3; value of the start state: predicted 12.67, actual discounted return 10.70, overestimate +1.98
```

Things to notice:

- **Part 1, the target network.** Without it, the value estimates reach two million after 25,000 steps and a hundred million by the end, a million times more than any policy can earn, and every agent's episodes last 9 or 10 steps: the network's outputs are dominated by the divergence, and it pushes the cart the same way at every step. This is the deadly triad of chapter 12 at full strength: each update moves the bootstrap target along with the prediction, and the network chases its own output. Copying the target every 250 steps turns the regression toward a moving target into a sequence of ordinary regressions, and every other agent learns.
- **Part 1, instability.** No agent keeps balancing the pole. Every one that learns alternates between stretches at 500 steps and collapses to much shorter episodes, and the fraction of time at 475 or more ranges from 0 to 0.8. This is the forgetting of chapter 16: once the memory fills with successful episodes, it holds few examples of failure, and the network's values for dangerous states drift until the agent visits them again.
- **Part 1, comparing variants.** With four seeds, the spread across the seeds of one variant is as large as the differences between variants: plain DQN's seeds range from 0 to 0.48, double DQN's from 0.16 to 0.30. The only unambiguous conclusion is that the target network is necessary. Replaying only the most recent 5,000 transitions did best in this run, and even a memory of 64 transitions, almost online Q-learning, learned the task; the age of the replayed data is one of the factors that [Fedus et al. (2020)](https://arxiv.org/abs/2007.06700) found to matter. Changing the random seeds changes such rankings, which is why chapter 18 insists on many runs and confidence intervals.
- **Part 1, overestimation.** Every agent that samples from the full memory ends with values above 100: 107 to 156 for DQN, 107 to 145 for double DQN, 113 to 146 for dueling DQN. Double DQN did not remove the overestimation here, presumably because its target still evaluates the online network's choice with a network trained on the same data, and with $\gamma=0.99$ a small bias in each target accumulates over an effective horizon of about 100 steps. The agents that sample only recent data kept their values between 99 and 104, as a nearly perfect policy's should be.
- **Part 2.** DQN learns Space Invaders from about 7 points per episode to about 21 during training, with at least 10% random actions, and its greedy policies score between 23 and 32. In both seeds, DQN's predicted value of the start state exceeds the discounted return its policy actually obtains, by 2.5 and 3.5, while double DQN's exceeds it by 1.0 and 2.0: the pattern that chapter 16 predicts, an upward bias that the double estimator reduces without removing. Double DQN's policies also score higher here, but two seeds cannot establish that.

## <a id="going-further"></a>Going further

1. **Enough seeds to decide.** Rerun part 1 with 16 seeds per variant (the batched agents make this cheap) and compare the variants by the interquartile mean of the time spent at 475 or more, with bootstrap confidence intervals ([Agarwal et al., 2021](https://arxiv.org/abs/2108.13264)). Which differences survive?
2. **Polyak averaging.** The script also supports a soft target update, `tau`. Replace the copy every 250 steps by $\tau=0.005$ and compare the stability and the value estimates (exercise 16.5).
3. **Where the overestimation comes from.** For a full-memory agent, compare its values with Monte Carlo returns of its greedy policy from the same states, separately for states that the current policy visits and states that only old policies visited. Is the bias concentrated where the data are old?
4. **Rainbow's components.** Add 3-step targets and proportional prioritized replay with a sum tree (chapter 18) to the MinAtar agent, and compare their effects at 100,000 steps on Space Invaders and Breakout.
5. **A distributional agent.** Replace the output by 51 categorical atoms or 51 quantiles (chapter 17) and compare the returns and the calibration of the predicted values with part 2.

---

[← Lab 7. System Identification, LQR, and Model Predictive Control](lab-07-system-identification-lqr-and-model-predictive-control.md) · [Lab 9. Advantage Actor-Critic with Parallel and Stale Actors →](lab-09-advantage-actor-critic-with-parallel-and-stale-actors.md)
