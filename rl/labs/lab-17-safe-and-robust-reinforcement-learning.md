[ML Mastery Notes](../../README.md) › [Reinforcement Learning](../README.md)

# Lab 17. Safe and Robust Reinforcement Learning

[← Lab 16. Reinforcement Learning for a Small Language Model](lab-16-reinforcement-learning-for-a-small-language-model.md)

## <a id="overview"></a>Overview

This lab puts chapter 31 to work with deep RL. You will train PPO agents under a safety constraint, first a budget that must hold on average and then a rule that must never be broken, and compare a learned multiplier with a model-based shield. Then you will train pendulum controllers in a simulator and test them on "real" systems whose actuator strength and control delay differ from the simulator's, comparing nominal training, domain randomization, a policy with memory, and the teacher–student adaptation of RMA. The chapter's codes showed each idea on a problem small enough to solve exactly; here the policies are neural networks trained from samples, and the results are noisier, slower, and sometimes different.

- **Environments:** two small environments written in NumPy and simulated in batches. *Hazard navigation:* a point robot with momentum, $v\leftarrow0.8v+0.2a$ and $p\leftarrow p+0.1v$ with $a\in[-1,1]^2$, starts near $(-1,0)$ and must reach $(1,0)$; a circular hazard of radius 0.8 lies between them. The reward is 10 times the progress toward the goal, $-0.5$ per step, and 10 on arrival; the cost is 1 for each step inside the hazard; episodes end on arrival or after 60 steps. Going straight takes about 23 steps with 17 of them in the hazard, and going around adds several steps. *Pendulum:* the swing-up task of `Pendulum-v1` (reward $-(\theta^2+0.1\dot\theta^2+0.001u^2)$, torque $|u|\le2$, 200 steps of 50 ms), with two parameters that the real system may change: a mass $m$ by which the torque's effect is divided, and a delay of whole steps before a commanded torque is applied.
- **Prerequisites:** chapter 31; PPO from Lab 10 and chapter 20; PyTorch and NumPy.
- **Reference solution:** [lab17_safe_robust.py](code/lab17_safe_robust.py), about 16 minutes on two cores, split about evenly between the navigation parts and the pendulum. Try each part yourself before reading it.

## <a id="part-1-a-budget-in-expectation-penalties-and-multipliers"></a>Part 1 — A budget in expectation: penalties and multipliers

1. **PPO with two critics.** Write PPO for a Gaussian policy with a state-independent standard deviation, starting at $e^{-0.5}$, and networks of two hidden layers of 64 tanh units, with the policy's output layer scaled down so that initial actions are near zero. Give it two value heads, one for the reward and one for the cost, each trained to its own GAE targets ($\gamma=0.99$, $\lambda=0.95$). Collect 60 steps from 64 robots per iteration, and take 10 epochs of minibatches of 512 with clipping at 0.2 and Adam at $3\times10^{-4}$. Train for 200 iterations. The observation is the position, the velocity, and the fraction of the episode elapsed.
2. **The penalized advantage.** Use the advantage $(A_r-\lambda A_c)/(1+\lambda)$, normalized over the batch. The division by $1+\lambda$ keeps its scale from growing with the multiplier, as in the Safety Gym baselines.
3. **Four rules for $\lambda$.** The budget is $d=5$ hazard steps per episode, and $J_c$ is the average cost of the episodes that finished in the iteration. Compare: no constraint; fixed penalties $\lambda=0.3$ and $\lambda=1$; gradient ascent $\lambda\leftarrow\max(0,\lambda+\eta(J_c-d))$ with $\eta=0.05$ and $0.5$; and a PI controller, $\lambda=\max(0,K_P(J_c-d)+I)$ with $I\leftarrow\max(0,I+K_I(J_c-d))$, $K_P=0.3$ and $K_I=0.05$.
4. **Measure.** Report the steps to the goal and the hazard steps per episode averaged over the last 20 iterations, the final multiplier, and the **overshoot**: the sum over iterations of $J_c-d$ when positive, a measure of how much the constraint was violated during training. Use 2 seeds.

## <a id="part-2-a-hard-constraint-learning-to-avoid-versus-a-shield"></a>Part 2 — A hard constraint: learning to avoid versus a shield

1. **Lagrangian with a zero budget.** Repeat part 1's gradient-ascent rule with $d=0$, and count every step any robot spends in the hazard during training.
2. **A shield.** Write a safety filter that uses a model of the dynamics and of the hazard: it accepts the proposed action if, after it, a braking maneuver, accelerating against the velocity for 6 steps, keeps the predicted position outside the hazard; otherwise it brakes. The environment applies the filtered action; the policy is trained on its own proposed actions, treating the shield as part of the environment. Run PPO without any cost term inside the shield.
3. **A wrong shield.** Repeat with a shield whose model is wrong in two ways, one at a time: it believes the hazard's radius is 0.7 instead of 0.8, or it believes the velocity decays by a factor of 0.7 per step instead of 0.8.
4. **Measure.** Report the steps to the goal and the hazard steps per episode at the end, and the total hazard steps during training, with 2 seeds.

## <a id="part-3-from-simulation-to-reality"></a>Part 3 — From simulation to reality

1. **Nominal training.** Train PPO on the nominal pendulum ($m=1$, no delay) for 40 iterations of 32 robots $\times$ 200 steps, with Adam at $10^{-3}$ and minibatches of 400. The episodes end only by the time limit, so bootstrap from the value of the last state rather than from zero. The policy sees $(\cos\theta,\sin\theta,\dot\theta/8)$.
2. **Domain randomization.** Train for 150 iterations on simulators whose parameters are drawn anew in every episode: $m$ log-uniform on $[0.6,2.4]$, and a delay of 0 to 3 steps (0 to 150 ms).
3. **Memory.** Train the same way with a policy that also sees the last 6 observations and commanded torques, which contain what it needs to infer the parameters.
4. **RMA.** Train a *teacher* the same way, with the true parameters, scaled to $[-1,1]$ over the randomization range, as extra inputs. Then train an *adaptation module*, a network that estimates those parameters from the last 6 observations and commands, by regression on data collected by the teacher acting on the module's own estimates (20 rounds of 64 episodes, 300 Adam steps after each). Deploy the teacher with the module's estimates.
5. **Test.** Evaluate each deterministic policy for 100 episodes on seven "real" systems: five inside the randomization range, at its center and corners, and two outside it. For reference, train an **oracle** on each real system itself, as in step 1. Use 2 seeds.

## <a id="expected-results"></a>Expected results

```text
=== Part 1: a budget of 5 hazard steps per episode, in expectation (PPO, 200 iterations, 2 seeds) ===
  final behavior: average of the last 20 iterations; overshoot: sum over iterations of (cost - 5) when positive
                               steps to goal   hazard steps   multiplier   overshoot in training
  unconstrained                      23.5          17.46            -              2690
  fixed penalty 0.3                  23.5          17.25         0.30              2665
  fixed penalty 1                    31.0           0.13         1.00                20
  Lagrangian, step 0.05              26.5           4.96         0.18                89
  Lagrangian, step 0.5               26.7           4.90         0.49                89
  PID, kp 0.3, ki 0.05               26.6           5.00         0.28                81

=== Part 2: a hard constraint, never enter the hazard (2 seeds) ===
                                    steps to goal   hazard steps (final)   hazard steps during training
  Lagrangian, budget 0                  34.3                0.07                      5,604
  shield, exact model                   27.8                0.00                          0
  shield, radius 0.7 (true 0.8)         25.2               14.60                    353,424
  shield, damping 0.7 (true 0.8)        27.8                0.00                      4,998

=== Part 3: a pendulum trained in simulation, tested on 'real' systems (return, higher is better, 2 seeds) ===
  randomization range: mass 0.6 to 2.4 (the torque's effect scales as 1/m), delay 0 to 3 steps (0 to 150 ms)
  real system          nominal   randomized   rand. + history   teacher (true params)   RMA   oracle
  m 1.0,   0 ms in range   -138       -139           -144                -140      -139    -138
  m 1.0, 150 ms in range   -967       -194           -211                -166      -169    -157
  m 0.6, 150 ms in range   -910       -182           -235                -142      -156    -116
  m 2.4,   0 ms in range   -549       -346           -435                -379      -339    -300
  m 2.4, 150 ms in range   -906       -493           -483                -584      -683    -342
  m 1.0, 250 ms outside   -1185       -869           -779                -662      -836    -457
  m 3.2,  50 ms outside    -843       -680           -691                -686      -689    -379
```

Things to notice:

- **Part 1, penalties must be tuned; multipliers tune themselves.** A penalty of 0.3 is too small to matter, and the agent drives through the hazard as if unconstrained. A penalty of 1 is too large: the agent avoids the hazard almost entirely, at a cost of 7.5 extra steps, when the budget allowed 5 hazard steps. Every Lagrangian rule lands on the budget, at about 26.5 steps: meeting it costs 3 steps over the unconstrained route, and ruling out the hazard costs 4.5 more. The price of a constraint is itself a useful thing to measure, and a Lagrangian method measures it for any budget without retuning.
- **Part 1, no oscillation here.** Unlike the switching policies of the chapter's third code, the step size of the multiplier hardly mattered, and the PI controller only slightly reduced the overshoot. The route bends continuously as $\lambda$ changes, so the cost responds smoothly, and 64 robots give a fairly precise estimate of $J_c$ in every iteration. Notice also that the three rules end with different multipliers, 0.18 to 0.49, for the same behavior: near the solution, the behavior is insensitive to $\lambda$ over a range, which is why the multiplier is a poor thing to read off, and the cost a good one (going further 1).
- **Part 2, a model makes safety during learning possible.** Learning to avoid the hazard costs over 5,000 hazard steps before the multiplier grows large enough, and the final agent keeps a wide margin, taking 34 steps. The exact shield allows no violation at all, ever, and its agent is *faster*, 28 steps, because it can skim the edge of the hazard knowing the shield will catch it, while the Lagrangian agent pays for every mistake and learns to stay away.
- **Part 2, a shield enforces its model, not the truth.** With the hazard's radius underestimated, the shield permits the band between 0.7 and 0.8, and the agent learns to drive through it: 14.6 hazard steps per episode, nearly as many as the unconstrained agent, and 350,000 during training. The agent optimized against the shield exactly as against a flawed reward. With the dynamics misestimated, the shield brakes too late when the robot moves fast, and about 5,000 violations occur during training, nearly as many as the Lagrangian agent's, before the policy learns to approach slowly enough for the braking to work.
- **Part 3, the reality gap is a delay.** The nominal policy is as good as the oracle on the nominal system and collapses with a 150 ms delay, returning $-967$ and $-910$ where the oracles return $-157$ and $-116$. Randomization recovers most of the gap everywhere in its range, at no cost on the nominal system.
- **Part 3, adaptation helps unevenly.** Knowing the parameters helps where the best behavior depends on them and they can be used: with a 150 ms delay and a nominal or strong actuator, the teacher and RMA beat the randomized policy ($-166$ and $-169$ against $-194$; $-142$ and $-156$ against $-182$). With a weak actuator, the privileged teacher is *worse* than the randomized policy, since learning a family of behaviors in the same budget is harder than learning one robust behavior (RMA, whose estimates are imperfect, happens to match the randomized policy there), and with a weak actuator and a long delay, where the history says least about the parameters, RMA's estimates are poor and it does worst of the policies trained on the range ($-683$). The policy with memory did no better than the randomized one inside the range, except marginally at the hardest corner: inferring the parameters implicitly, from a short window and in 150 iterations, was harder than ignoring them.
- **Part 3, outside the range everything degrades.** With a 250 ms delay or a very weak actuator, all policies lose far more than the oracle. The teacher, fed a delay parameter beyond its training range, degrades least with the long delay, a hint that explicit parameters extrapolate better than implicit inference, but no method substitutes for a randomization range that covers the real system.

## <a id="going-further"></a>Going further

1. **Noisy costs and oscillation.** Reduce part 1 to 8 robots per iteration, so that $J_c$ is estimated from few episodes, and compare gradient ascent with $\eta=0.5$ and the PI controller. Plot $J_c$ and $\lambda$ over iterations. When does the multiplier oscillate, and does a derivative term, or averaging $J_c$ over several iterations, help?
2. **Budgets and the price of safety.** Run the Lagrangian agent with budgets of 0, 2, 5, 10, and 15 hazard steps, and plot the steps to the goal against the budget. Is the curve convex, as the linear-programming view of chapter 31's appendix B suggests for tabular problems?
3. **A robust shield.** Make the shield plan against a set of models, velocity decay factors from 0.7 to 0.9, and brake if any of them predicts entering the hazard. Does it remove the violations of the misestimated shield, and what does its caution cost in steps? Then add a small penalty for each intervention, as Alshiekh et al. did, and measure how often the shield still has to act at the end of training.
4. **Better memory.** Replace the history window of part 3 with a GRU that sees the previous torque and the observation, and train it for longer. Does implicit adaptation catch up with RMA, or with the oracle, inside the range?
5. **Automatic domain randomization.** Start with a narrow range around the nominal system and widen each parameter's range whenever the policy's return at the edge of the range exceeds a threshold, as in OpenAI's Rubik's cube work. How wide do the ranges grow, and how do the resulting policies do on the two systems outside the fixed range of part 3?

---

[← Lab 16. Reinforcement Learning for a Small Language Model](lab-16-reinforcement-learning-for-a-small-language-model.md)
