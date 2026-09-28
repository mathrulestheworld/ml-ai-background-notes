[ML Mastery Notes](../../README.md) › [Reinforcement Learning](../README.md)

# Lab 14. Imitation and Offline Reinforcement Learning

[← Lab 13. AlphaZero and Gumbel Search from Scratch](lab-13-alphazero-and-gumbel-search-from-scratch.md) · [Lab 15. Regret Minimization and Self-Play in Poker →](lab-15-regret-minimization-and-self-play-in-poker.md)

## <a id="overview"></a>Overview

This lab puts chapter 25 and chapter 26 to work on one small problem. You will train an expert, collect datasets of different quality from it and from weaker policies, and then learn from them without further reward: first by imitating the expert, with behavioral cloning and with DAgger, and then from fixed datasets with rewards, with behavioral cloning, a naive off-policy agent, TD3+BC, and implicit Q-learning. The point is to see each method's characteristic failure, and to see that which method wins depends on the data.

- **Environment:** `Pendulum-v1`, as in Lab 11: swing a pendulum up and hold it with a torque in $`[-2,2]`$; 200-step episodes, and a return above about $`-200`$ means the pendulum is swung up and held. The environment never terminates, so every transition bootstraps.
- **Prerequisites:** chapters 25 and 26, and the SAC and TD3 of Lab 11; PyTorch and Gymnasium.
- **Reference solution:** [lab14_offline.py](code/lab14_offline.py), about 25 minutes on two cores, most of it in part 3. Try each part yourself before reading it.

## <a id="part-1-an-expert-and-four-datasets"></a>Part 1 — An expert and four datasets

1. Train SAC as in Lab 11, with networks of two layers of 256 units, for 15,000 steps (the first 1,000 random), keep a copy of the actor every 250 steps, and keep every transition it experienced. Its deterministic policy is the **expert**.
2. Evaluate the saved actors and pick the one whose deterministic policy scores closest to $`-500`$ as the **medium** policy.
3. Collect four datasets: **expert**, 100 episodes of the expert with Gaussian noise of standard deviation 0.1 added to its actions (in $`[-1,1]`$ units); **medium**, 100 episodes of the medium policy, sampling its actions; **random**, 100 episodes of uniformly random actions; and **replay**, the 15,000 transitions SAC saw while learning, from random play to expert play. Report each dataset's average episode return.

## <a id="part-2-behavioral-cloning-and-dagger"></a>Part 2 — Behavioral cloning and DAgger

1. **Behavioral cloning.** Fit a deterministic policy, an MLP with two layers of 64 units and a tanh output, to the expert's actions by squared error (2,000 Adam steps, step size $`10^{-3}`$, minibatches of 256), using $`k=1,3,5,10,20`$ episodes of the noise-free expert, that is, 200 to 4,000 labeled states.
2. **DAgger.** Start from the policy cloned from one expert episode. Then repeatedly run the current learner for one episode, ask the expert for the action it would have taken in every state the learner visited, add these labeled states to the data, and continue training the learner on everything (500 more steps). Evaluate after the same numbers of labeled states as behavioral cloning.
3. Evaluate every policy over the same 10 episodes and average over 3 seeds.

## <a id="part-3-offline-rl-from-datasets-of-different-quality"></a>Part 3 — Offline RL from datasets of different quality

Train four agents on each dataset for 10,000 gradient steps, with networks of two layers of 256 units, Adam with step size $`3\times10^{-4}`$, minibatches of 256, $`\gamma=0.99`$, and states normalized by the dataset's mean and standard deviation:

1. **BC:** behavioral cloning of the dataset's actions.
2. **TD3:** plain TD3 on the fixed data, with twin critics, target smoothing, and delayed actor updates, but no special treatment of actions outside the data.
3. **TD3+BC:** TD3 whose actor maximizes $`\lambda Q(s,\pi(s))-(\pi(s)-a)^2`$, with $`\lambda=2.5/\operatorname{mean}|Q|`$ computed on each minibatch.
4. **IQL:** a value network fitted to the target critics' values by expectile regression with $`\tau=0.7`$, critics trained toward $`r+\gamma V(s')`$, and a policy extracted by advantage-weighted regression with weights $`\exp(3(Q-V))`$ clipped at 100.

Evaluate each agent's deterministic policy over 10 episodes, with 2 seeds. For the three agents with critics, also compare the critic's value $`Q(s_0,\pi(s_0))`$ of each evaluation episode's first state with the discounted return actually obtained from it.

## <a id="expected-results"></a>Expected results

```text
=== Part 1: an expert and four datasets (Pendulum-v1) ===
  expert (SAC after 15,000 steps, deterministic): average return -110
  medium policy: SAC's actor after 3,500 steps, sampled: -597 (deterministic)
  dataset          transitions   average episode return   (min, max)
  expert                20,000                     -150   (-365, -1)
  medium                20,000                     -790   (-1068, -386)
  random                20,000                    -1219   (-1844, -628)
  replay                15,000                     -419   (-1800, -1)

=== Part 2: behavioral cloning and DAgger, by number of expert-labeled states (mean of 3 seeds) ===
  labeled states:       200     600   1,000   2,000   4,000
  BC                  -1259    -510    -421    -135    -131
  DAgger              -1259   -1042    -122    -111    -111

=== Part 3: offline RL, 10,000 gradient steps, 2 seeds: average return of the policy ===
  mean of the seeds, and half the difference between them
  dataset       data          BC           TD3        TD3+BC           IQL
  expert        -150    -123 ±0      -1611 ±2       -306 ±195     -117 ±6
  medium        -790    -755 ±30     -1617 ±1       -216 ±21      -697 ±96
  random       -1219   -1126 ±28      -109 ±0      -1020 ±23      -238 ±5
  replay        -419    -372 ±49      -111 ±1       -175 ±0       -473 ±205
  the critic's value of the first states against the discounted return obtained (mean of seeds)
  dataset          TD3 predicted  obtained    TD3+BC predicted  obtained    IQL predicted  obtained
  expert                -78      -660              -81      -166              -90      -100
  medium                -83      -669              -92      -158             -142      -294
  random                -82       -95             -115      -447             -198      -152
  replay                -84       -96              -93      -140             -132      -226
```

Things to notice:

- **Part 2, compounding errors.** Cloned from one or two episodes, the policy fails completely, and it takes about 2,000 expert-labeled states, ten episodes, before it reliably swings the pendulum up; even with 4,000 it stays slightly below the expert. DAgger reaches the expert's $`-110`$ with 1,000 labels and stays there: the labels it collects are in the states the learner actually visits, the states where behavioral cloning's errors compound. With 600 labels, however, DAgger is *worse* than cloning. Its first learner, trained on one episode, spins the pendulum, and the states it visits, though labeled correctly, are far from the swing-up the expert performs, so at that point it has seen one episode of the states that matter where cloning has seen three. DAgger's guarantee is about the long run, and its early data can be spent on states a better learner will never visit.
- **Part 3, extrapolation error.** Plain TD3 fails badly on the expert and medium data, below $`-1,600`$, worse than random play, while its critic predicts a return near $`-80`$ where the policy obtains about $`-665`$. These datasets cover a narrow band of actions in each state, and the actor climbs to actions whose values the critic invents, exactly the failure of chapter 26. On the random and replay data, which cover the actions broadly, the same algorithm is the best of all, $`-109`$ and $`-111`$, and its critic's predictions are within about 13 of the returns obtained. Coverage, not the quality of the behavior, decides whether off-policy RL works offline.
- **Part 3, the price of staying close.** Behavioral cloning does a little better than its data, since averaging removes the noise, but never better than the behavior's mean action. TD3+BC improves greatly on the medium data, to $`-216`$ from the data's $`-790`$, and is good on the replay data, but on the random data its cloning term holds it near random behavior ($`-1,020`$), where plain TD3 reached $`-109`$, and on the expert data one of its two seeds failed. A single weight $`\alpha=2.5`$ is too much constraint for some datasets and too little for others.
- **Part 3, in-sample learning.** IQL, which never evaluates actions outside the data, is the best on the expert data ($`-117`$, with the most accurate critic) and learns a good policy from random data ($`-238`$), but it is mediocre on the medium and replay data, with large differences between seeds. No method wins everywhere, and two seeds are not enough to rank the close cases: offline RL results must be read per dataset, over several seeds, and with the hyperparameter sensitivity in mind, which is why chapter 26 stressed that choosing among offline policies without running them is itself a hard problem.

## <a id="going-further"></a>Going further

1. **The right amount of conservatism.** Sweep TD3+BC's $`\alpha`$ over $`\{0.5,1,2.5,10\}`$ and IQL's $`\tau`$ over $`\{0.5,0.7,0.9\}`$ on each dataset. Is there a setting that works everywhere, and can the critic's own predictions tell you which setting to choose?
2. **CQL.** Add conservative Q-learning to part 3, with the log-sum-exp over 10 actions sampled uniformly and 10 from the policy, as in chapter 26's appendix. Where does it land between TD3 and IQL?
3. **Noisy and mixed demonstrations.** Clone from demonstrations mixing the expert and the random policy in different proportions, and compare with filtering the demonstrations by return, and with IQL on the same data.
4. **Offline to online.** Take the TD3+BC and IQL agents trained on the medium data and continue training them online for 5,000 steps, keeping or dropping the behavior-regularization term. Do you see the initial dip of chapter 26?
5. **A harder body.** Repeat part 3 on `Hopper-v5`, where episodes terminate when the robot falls, with datasets collected by your SAC from Lab 11. Termination adds absorbing states; how does it change the failure of plain TD3?

---

[← Lab 13. AlphaZero and Gumbel Search from Scratch](lab-13-alphazero-and-gumbel-search-from-scratch.md) · [Lab 15. Regret Minimization and Self-Play in Poker →](lab-15-regret-minimization-and-self-play-in-poker.md)
