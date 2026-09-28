[ML Mastery Notes](../../README.md) › [Reinforcement Learning](../README.md)

# Lab 10. Proximal Policy Optimization from Scratch

[← Lab 9. Advantage Actor-Critic with Parallel and Stale Actors](lab-09-advantage-actor-critic-with-parallel-and-stale-actors.md) · [Lab 11. Off-Policy Actor-Critics for Continuous Control →](lab-11-off-policy-actor-critics-for-continuous-control.md)

## <a id="overview"></a>Overview

This lab puts chapter 20 to work. You will implement PPO from scratch with Gymnasium's vectorized environments, train it to land a spacecraft, remove its ingredients one at a time to see which matter, and then train a simulated cheetah to run with a Gaussian policy.

- **Environments:** `LunarLander-v3` (Box2D): 8 observations, 4 discrete actions (nothing, left engine, main engine, right engine); rewards for moving toward the pad and landing, $`-100`$ for crashing, $`+100`$ for coming to rest, and small costs for firing the engines; a score of 200 counts as solved. `HalfCheetah-v5` (MuJoCo): 17 observations, 6 torques in $`[-1,1]`$, a reward for forward speed minus a small control cost, 1,000-step episodes.
- **Prerequisites:** chapters 19 and 20; PyTorch and Gymnasium with the Box2D and MuJoCo extras (`pip install "gymnasium[box2d,mujoco]"`).
- **Reference solution:** [lab10_ppo.py](code/lab10_ppo.py), about 45 minutes on two cores, most of it in the 10-epoch runs. Try each part yourself before reading it.

## <a id="part-1-ppo-from-scratch"></a>Part 1 — PPO from scratch

1. Create 8 environments with `gym.make_vec(..., vectorization_mode="sync", vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})`. In this mode a finished sub-environment is reset within the same call to `step`, and its true final observation is returned in `info["final_obs"]`; use it to bootstrap at truncations (chapter 19, appendix A).
2. Build separate actor and critic networks, each with two tanh layers of 64 units, orthogonal initialization with gain $`\sqrt2`$, and output layers with gains 0.01 (actor) and 1 (critic).
3. Each iteration, collect 256 steps from each environment, compute GAE advantages and value targets, and run 10 epochs of minibatch updates (32 minibatches of 64) on the loss of chapter 20: the clipped objective with $`\epsilon=0.2`$, the value loss with coefficient 0.5, and an entropy bonus of 0.01. Normalize the advantages in each minibatch, clip the gradient's norm to 0.5, and anneal Adam's step size linearly from $`3\times10^{-4}`$ to 0. Use $`\gamma=0.999`$ and $`\lambda=0.98`$.
4. After the last epoch of each iteration, measure how far the policy moved: the mean KL divergence from the policy that collected the batch, estimated by $`\mathbb E[(r-1)-\ln r]`$ with $`r`$ the probability ratio, and the fraction of samples whose ratio is outside $`[1-\epsilon,1+\epsilon]`$.
5. Train for 500,000 steps with 2 seeds, and report the average return of the last 20 episodes as training proceeds.

## <a id="part-2-which-ingredients-matter"></a>Part 2 — Which ingredients matter

Rerun part 1 with one change at a time: 4 epochs and 1 epoch instead of 10; no clipping (a clipping range so large it never binds); no advantage normalization; PPO's clipped value loss (with the same $`\epsilon`$, in the units of the rewards); and $`\gamma=0.99`$, $`\lambda=0.95`$, the usual defaults.

## <a id="part-3-continuous-control"></a>Part 3 — Continuous control

1. For continuous actions, let the actor output the mean of a Gaussian and keep its log standard deviation as a free parameter, initialized to 0; sum the log-probabilities and entropies over the action dimensions, and clip the sampled actions to the bounds only when sending them to the environment.
2. Add running normalization of the observations (clipped to $`[-10,10]`$) and scale the rewards by a running estimate of the standard deviation of the discounted return.
3. Train on `HalfCheetah-v5` for 1,000,000 steps with $`\gamma=0.99`$, $`\lambda=0.95`$ and no entropy bonus, with and without the normalizations, 2 seeds each.

## <a id="expected-results"></a>Expected results

```text
=== Parts 1 and 2: PPO on LunarLander-v3, 8 environments x 256 steps, 500,000 steps, 2 seeds ===
  average return of the last 20 episodes, mean of seeds; KL and fraction clipped after the last epoch,
  averaged over the second half of training; the final return of each seed (last 20% of training)
                                       100k   200k   300k   400k   500k       KL  clip   per seed
  PPO (10 epochs, clipping 0.2)          60    165    215    246    247    0.003  0.03     250   243
    4 epochs                             12     73    105    127    169    0.002  0.01     143   168
    1 epoch                            -107    -26    -23    -25     21    0.000  0.00      24     2
    no clipping                          46     84    174    234    238    0.046  0.00     254   200
    no advantage normalization          155    185    225    215    206    0.004  0.04     188   201
    value clipping                        8     96    102    106    112    0.005  0.05     106   120
    discount 0.99, lambda 0.95          -31     -5     20     54    117    0.004  0.04      36   165

=== Part 3: PPO on HalfCheetah-v5, 8 environments x 256 steps, 1,000,000 steps, 2 seeds ===
                                       200k   400k   600k   800k  1000k       KL  clip   per seed
  normalized observations and rewards    757   1359   1226   1621   1594    0.016  0.23    1446  1667
  no normalization                      296    830   1364   1891   2048    0.018  0.25    1443  2653
```

Things to notice:

- **Part 1.** PPO solves the task in both seeds, with an average near 250 by 500,000 steps, and the policy moves little in each iteration: a mean KL of about 0.003 from the policy that collected the batch, with 3% of the samples outside the clipping range after the last epoch. The small step is by design; what the clipping buys is the ability to take ten epochs of steps on each batch without the policy running away.
- **Part 2, epochs.** Reuse is where PPO's sample efficiency comes from. With one epoch, the agent is essentially A2C with minibatches and barely learns in 500,000 steps; with 4 epochs it reaches 169; with 10, 247. The measured KL grows with the number of epochs, as each batch pushes the policy further.
- **Part 2, clipping.** Without clipping, the agent still learns about as well here, while its policy moves 15 times further per iteration (a KL of 0.046). With this small step size and a well-scaled advantage, ten epochs are not enough to reach the instabilities that the clipping prevents. Lab 9 showed a setting where they are: stale data. Going further 1 asks you to find the point where the unclipped agent breaks.
- **Part 2, the details.** Clipping the value loss is clearly harmful: LunarLander's returns span hundreds of points, and a critic that may move only 0.2 per update cannot follow them, so its advantages are poor; this is the finding of Andrychowicz et al. (chapter 20). The usual discount of 0.99 is also much worse, and inconsistent across seeds: the landing bonus and the crash penalty come at the end of episodes of hundreds of steps, beyond the effective horizon of about 100 steps that $`\gamma=0.99`$ gives, and the agent settles for hovering. Without advantage normalization, both seeds end slightly lower; with two seeds, this is at most a small effect, as Andrychowicz et al. found.
- **Part 3.** PPO learns to run, reaching 1,400 to 2,700 after a million steps, typical of PPO on this task. The normalizations did not help here: HalfCheetah's observations and rewards are already of moderate scale, and the difference between the two variants is smaller than the difference between the seeds of one of them. The policy moves much more per iteration than on LunarLander, a KL near 0.017 with a quarter of the samples clipped: with a continuous action space and a learned standard deviation, ten epochs push hard against the trust region. The off-policy methods of chapter 21 reach several thousand on the same task with a fraction of the environment steps (Lab 11).

## <a id="going-further"></a>Going further

1. **Breaking the unclipped agent.** Rerun "no clipping" with 30 epochs, then with a step size of $`10^{-3}`$, and compare with the clipped agent under the same settings. Then add early stopping of the epochs when the measured KL exceeds 0.015. Which of the three controls, clipping, fewer epochs, or early stopping, is most robust?
2. **A squashed Gaussian.** Replace the clipping of actions in part 3 by a tanh-squashed Gaussian with the log-density correction of chapter 21, and compare.
3. **A shared network.** Share the first layer of the actor and the critic, and tune the value coefficient. Does it help on LunarLander?
4. **Phasic policy gradient.** Add PPG's auxiliary phase to part 1: every 16 iterations, train the critic for 6 more epochs on all the stored data and distill it into an auxiliary value head of the actor with a KL term.
5. **Enough seeds.** Run part 3 with 8 seeds per variant and compare the interquartile means with bootstrap confidence intervals. Does normalization matter on HalfCheetah? On Hopper?

---

[← Lab 9. Advantage Actor-Critic with Parallel and Stale Actors](lab-09-advantage-actor-critic-with-parallel-and-stale-actors.md) · [Lab 11. Off-Policy Actor-Critics for Continuous Control →](lab-11-off-policy-actor-critics-for-continuous-control.md)
