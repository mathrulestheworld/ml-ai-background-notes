[ML Mastery Notes](../../README.md) › [Reinforcement Learning](../README.md)

# Lab 9. Advantage Actor-Critic with Parallel and Stale Actors

[← Lab 8. Deep Q-Networks on CartPole and MinAtar](lab-08-deep-q-networks-on-cartpole-and-minatar.md) · [Lab 10. Proximal Policy Optimization from Scratch →](lab-10-proximal-policy-optimization-from-scratch.md)

## <a id="overview"></a>Overview

This lab puts chapter 19 to work. You will write a synchronous advantage actor–critic, train dozens of agents at once on the cart-pole by batching their networks, and measure which of its ingredients matter. Then you will make the actors stale, as they are in a distributed system, and compare three responses: ignoring the lag, correcting for it with V-trace, and bounding the policy's change with PPO's clipped objective. Finally, you will trade the number of parallel environments against the number of updates.

- **Environment:** the cart-pole of Lab 6, simulated in NumPy for all agents' environments at once, with the dynamics of Gymnasium's `CartPole-v1`: reward 1 per step, termination when the pole tilts more than 12° or the cart leaves $`[-2.4,2.4]`$, truncation at 500 steps. The simulator must report terminations and truncations separately and return the true final state of each episode before resetting it.
- **Prerequisites:** chapters 13 and 19, and the batched networks of Lab 8; PyTorch and NumPy.
- **Reference solution:** [lab09_a2c.py](code/lab09_a2c.py), about four minutes on one core. Try each part yourself before reading it.

## <a id="part-1-a2c-and-its-ingredients"></a>Part 1 — A2C and its ingredients

1. Give each agent an actor and a critic, separate networks with two tanh layers of 64 units, stored with a leading agent dimension as in Lab 8. Scale the observations by rough ranges of their components, $`(2.4, 2, 0.21, 2)`$, and initialize the last layer of each network with weights 100 times smaller than usual, so that the initial policy is nearly uniform and the initial values nearly zero.
2. Let the critic predict $`k`$ times its network's output, for a fixed **critic scale** $`k`$. With a reward of 1 per step and $`\gamma=0.99`$, the values approach 100, and $`k`$ decides how far the network's own outputs must move to reach them.
3. Each iteration, run 16 environments per agent for 16 steps, compute GAE(λ) advantages and TD(λ) targets with the rules of chapter 19 for terminations, truncations, and the end of the rollout, and take one step of Adam (step size $`10^{-3}`$, written out per agent) on the actor–critic loss with value coefficient 0.5 and entropy coefficient β.
4. Train for 384,000 steps per agent, with 6 seeds of each configuration: the default (λ = 0.95, β = 0.01, $`k=10`$), critic scales 1 and 100, λ of 0, 0.8, and 1, and β of 0 and 0.1. Report the average length of each agent's last 10 episodes, and count a seed as solved if that average is at least 475 over its last 32,000 steps.

## <a id="part-2-stale-actors"></a>Part 2 — Stale actors

1. Keep a history of the actor's recent parameters, and let each agent's actors act with the parameters of `lag` updates earlier, as distributed actors do. Store the behavior policy's log-probabilities of the actions taken.
2. Implement three learners for the same stale data. **No correction:** the A2C update of part 1, as if the data were on-policy. **V-trace:** with $`\rho_t=\min(1,\pi/\mu)`$ and $`c_t=\lambda\min(1,\pi/\mu)`$, compute V-trace targets for the critic and use $`\rho_s(r_s+\gamma v_{s+1}-V(x_s))`$ as the actor's advantage. **Clipped objective:** GAE advantages and targets as in part 1, but PPO's clipped objective for the actor, $`\min\bigl(r\hat A,\operatorname{clip}(r,0.8,1.2)\hat A\bigr)`$, with $`r=\pi/\mu`$.
3. Train each learner with lags of 0, 8, and 32 updates, 6 seeds each.

## <a id="part-3-how-many-environments"></a>Part 3 — How many environments?

1. With the same budget of 384,000 steps and rollouts of 16 steps, train with 4, 16, and 64 environments per agent, that is, with 6,000, 1,500, and 375 updates, at step sizes $`10^{-3}`$ and $`4\times10^{-3}`$.
2. Record the wall-clock time of each setting as well as its learning curve.

## <a id="expected-results"></a>Expected results

```text
=== Part 1: A2C on the cart-pole, 16 environments, rollouts of 16 steps, 6 seeds ===
  average length of the last 10 episodes (at most 500), mean over seeds; 'solved' counts the seeds whose
  average over the last 32,000 steps is at least 475
                                          steps:     64k   128k   256k   384k   solved   final length of each seed
  A2C (lambda 0.95, entropy 0.01, critic x10)     112    199    454    500   6 of 6   500 500 500 500 500 500
    critic x1                                     255    239    244    231   1 of 6   491 356 153 158 149 161
    critic x100                                   193    290    295    432   4 of 6   219 499 300 500 500 500
    lambda = 0                                    200    280    423    498   5 of 6   466 500 500 500 500 500
    lambda = 0.8                                  131    168    402    420   4 of 6   500 500 273 500 500 263
    lambda = 1                                    382    245    467    473   5 of 6   222 500 500 500 500 500
    entropy coefficient 0                         187    374    411    500   6 of 6   500 500 500 500 500 500
    entropy coefficient 0.1                       161    447    447    500   6 of 6   500 500 489 500 500 500

=== Part 2: stale actors, which act with the parameters of `lag` updates earlier ===
                                          steps:     64k   128k   256k   384k   solved   final length of each seed
  lag  0, no correction                           161    276    462    500   6 of 6   500 500 500 500 500 500
  lag  0, V-trace                                 175    392    437    500   6 of 6   500 500 500 500 500 500
  lag  0, clipped objective                       149    264    470    500   6 of 6   500 500 500 500 500 500
  lag  8, no correction                           187     33    120     10   0 of 6     9   9   9   9   9   9
  lag  8, V-trace                                 186    237    115    173   1 of 6   500   9  10 289   9   9
  lag  8, clipped objective                       154    354    439    500   6 of 6   500 500 500 500 500 500
  lag 32, no correction                           500    361    181    202   2 of 6   500 500   9  51  56  74
  lag 32, V-trace                                 235    129    191    213   2 of 6    48 209  10 500   9 500
  lag 32, clipped objective                       195    443    500    500   6 of 6   500 500 500 500 500 500

=== Part 3: how many environments? The same 384,000 steps, rollouts of 16 steps, 6 seeds ===
  4 environments, 6,000 updates  (53 s)
                                          steps:     64k   128k   256k   384k   solved   final length of each seed
      step size 1e-3                              402    500    500    500   6 of 6   500 500 500 494 500 500
      step size 4e-3                              439    500    500    500   6 of 6   500 500 500 500 500 500
  16 environments, 1,500 updates  (20 s)
                                          steps:     64k   128k   256k   384k   solved   final length of each seed
      step size 1e-3                              176    239    393    500   5 of 6   467 500 500 500 500 500
      step size 4e-3                              225    400    445    467   5 of 6   500 500 500 500 261 500
  64 environments, 375 updates  (13 s)
                                          steps:     64k   128k   256k   384k   solved   final length of each seed
      step size 1e-3                              165     98    102    141   0 of 6   103 120 146 107 236  99
      step size 4e-3                               68    111    242    233   1 of 6   383 500  97 137 113 194
```

The times depend on the machine.

Things to notice:

- **Part 1, the critic's scale.** It is the most important choice here. With $`k=1`$ only one seed of six solves the task, and the others stall between 150 and 360 steps: the critic starts near zero and needs thousands of updates to raise its outputs to values near 100, and meanwhile its advantages are nearly all positive, which makes it a poor baseline, the situation of REINFORCE without a baseline in Lab 6. With $`k=10`$ every seed solves it. With $`k=100`$ four do: every update of the network now moves the values a hundred times more, and the critic becomes noisy. The principled fix is to normalize the targets, as PopArt does, or the rewards, as PPO implementations do (chapter 19).
- **Part 1, λ and the entropy bonus.** Neither has a clear effect on this task: every λ solves the task in most seeds, and the differences are within the spread of the seeds. The cart-pole's rewards are dense and its critic becomes accurate quickly, so the bias of small λ matters little; Lab 6 showed that λ matters when the critic is poor. The entropy bonus is also weak here: the advantages are typically of order 1 or more, so a coefficient of 0.01 barely competes with them (exercise 19.1), and even 0.1 neither helps nor hurts.
- **Part 2.** Without lag, all three learners are the same algorithm up to details, and all solve the task. With a lag of 8 updates, the uncorrected learner collapses in every seed to a policy that pushes the cart the same way at every step, and V-trace saves only one seed of six; with a lag of 32, each saves two. The clipped objective solves the task in every seed at both lags. V-trace corrects the targets for the difference between the behavior and the current policy, but nothing stops the current policy from moving further in the direction of the stale data at every update, and once it has collapsed, neither the true gradient nor V-trace's gives the abandoned action any weight (exercise 19.7). Clipping stops each sample's contribution once the policy has moved 20% from the behavior in the direction the sample favors, which bounds the drift. This is the experiment of chapter 19's figure, and the reason for the trust regions of chapter 20. The lags here are extreme; with lags of a few updates and small steps, V-trace works well, as in IMPALA.
- **Part 3.** Per environment step, fewer environments learn faster: with 4 environments, every seed solves the task within 128,000 steps, while with 64 environments and the same step size no seed does within 384,000. The total amount of data is the same; what differs is the number of updates, 6,000 against 375, and a larger batch makes each update less noisy but not larger. A four times larger step size helps the large batch only slightly. Per second, the balance reverses: the batched computation makes a step of 64 environments barely more expensive than a step of 4, so the large batch processes the same data four times faster. Beyond a **critical batch size**, larger batches stop reducing the number of updates needed, and the gradient noise scale predicts it, in reinforcement learning as in supervised learning ([McCandlish, Kaplan, Amodei, and the OpenAI Dota Team, 2018](https://arxiv.org/abs/1812.06162)). Distributed agents collect data in huge batches because data are cheap for them; PPO's several epochs per batch are another way of getting more updates out of each sample.

## <a id="going-further"></a>Going further

1. **A shared torso.** Replace the two networks by one with a policy head and a value head, and tune the value coefficient $`c_v`$ in $`\{0.1, 0.5, 2\}`$. How does the critic scale interact with $`c_v`$?
2. **PopArt.** Replace the fixed critic scale by PopArt normalization of the value targets (exercise 19.8), and compare with the scales of part 1.
3. **Small lags and small steps.** Repeat part 2 with lags of 1, 2, and 4 updates and a step size of $`3\times10^{-4}`$, and measure the average KL between the behavior and the current policy. Where does V-trace start to matter, and where does it stop being enough?
4. **A2C on MinAtar.** Port the agent to MinAtar's Breakout with the convolutional network of Lab 8, a shared torso, and 16 environments, and compare it with DQN at equal numbers of frames and at equal wall-clock time.
5. **Real asynchrony.** Move the actors to a separate process that receives the learner's parameters through a queue every few updates, an IMPALA in miniature, and measure the distribution of the lag and the throughput as you vary the number of actors.

---

[← Lab 8. Deep Q-Networks on CartPole and MinAtar](lab-08-deep-q-networks-on-cartpole-and-minatar.md) · [Lab 10. Proximal Policy Optimization from Scratch →](lab-10-proximal-policy-optimization-from-scratch.md)
