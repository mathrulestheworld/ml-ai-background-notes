[Background Notes](../../README.md) › [Reinforcement Learning](../README.md)

# Lab 6. Policy Gradient and Actor-Critic Methods on CartPole

[← Lab 5. Tile Coding and Linear Control on Mountain Car](lab-05-tile-coding-and-linear-control-on-mountain-car.md) · [Lab 7. System Identification, LQR, and Model Predictive Control →](lab-07-system-identification-lqr-and-model-predictive-control.md)

## <a id="overview"></a>Overview

This lab puts chapter 13 to work on the cart-pole. You will write a vectorized simulator and check it against Gymnasium, compare the gradient estimators of REINFORCE, build an online actor–critic, measure the bias and variance of generalized advantage estimates, and compare the vanilla and natural policy gradients.

- **Environment:** `CartPole-v1`. The state is the cart's position and velocity and the pole's angle and angular velocity; the actions push the cart left or right; the reward is 1 per step until the pole tilts more than 12° or the cart leaves $`[-2.4,2.4]`$, and Gymnasium cuts episodes off at 500 steps. To run many episodes at once, you will reimplement the dynamics in NumPy.
- **Policy:** throughout, a linear softmax policy over the two actions, $`\pi(\text{right}\mid s)=\sigma(\boldsymbol\theta^\top\mathbf x(s))`$, with $`\mathbf x(s)=(s_1/2.4,\ s_2/2,\ s_3/0.21,\ s_4/2,\ 1)`$, the state scaled by rough ranges of its components. Its score is $`\nabla\ln\pi(a\mid s)=(a-\sigma(\boldsymbol\theta^\top\mathbf x(s)))\,\mathbf x(s)`$ with $`a=1`$ for right. The discount is $`\gamma=0.99`$.
- **Prerequisites:** chapters 11 (the Fourier basis, semi-gradient TD) and 13; NumPy and Gymnasium.
- **Reference solution:** [lab06_policy_gradient.py](code/lab06_policy_gradient.py), about two and a half minutes on one core. Try each part yourself before reading it.

## <a id="part-1-reinforce-and-its-estimators"></a>Part 1 — REINFORCE and its estimators

1. Write the cart-pole dynamics for $`n`$ copies at once, with the constants and the Euler integration of Gymnasium's source, and automatic resets. Check that it reproduces `env.unwrapped.state` exactly over random episodes.
2. Implement per-episode REINFORCE with four weightings of the score $`\nabla\ln\pi(A_t\mid S_t)`$: the whole discounted return $`G_0`$ at every step; $`\gamma^tG_t`$, the return from step $`t`$ discounted back to the start; $`G_t`$, without the factor $`\gamma^t`$, as most implementations do; and $`G_t-\hat v(S_t)`$, with $`\hat v`$ learned by gradient Monte Carlo on a Fourier basis of order 2 (81 features, per-feature step sizes $`0.01/\|\mathbf c\|`$, averaged over the steps of each episode).
3. For step sizes $`3\times10^{-4}`$, $`10^{-3}`$, and $`3\times10^{-3}`$, run 10 seeds of 1,000 episodes. Report the average return in windows of 200 episodes, and count the seeds whose last 100 episodes average more than 475.

## <a id="part-2-online-actorcritic"></a>Part 2 — Online actor–critic

1. Implement the one-step actor–critic of chapter 13 with a TD(0) critic on the same Fourier features, $`\alpha^{\boldsymbol\theta}=0.1`$ and $`\alpha^{\mathbf w}=0.03/\|\mathbf c\|`$, updating at every step.
2. At the 500-step limit the episode is cut off, not over: bootstrap from the true next state. Then run the agent again treating the time limit as termination.
3. Add accumulating eligibility traces with $`\lambda=0.9`$ to the actor and the critic, with $`\alpha^{\boldsymbol\theta}=0.03`$ and $`\alpha^{\mathbf w}=0.003/\|\mathbf c\|`$.
4. Compare the three agents over 300 episodes, 10 seeds, with REINFORCE with baseline.

## <a id="part-3-the-bias-and-variance-of-gradient-estimators"></a>Part 3 — The bias and variance of gradient estimators

1. Fix the policy $`\boldsymbol\theta=(0,0,1.5,1.5,0)`$, which pushes toward the side the pole is falling to and lasts about 110 steps on average.
2. Fit three critics: least squares on quadratic features of the scaled state ($`1`$, $`u_i`$, $`u_iu_j`$; 15 features) to the returns of 2,000 episodes of this policy ("fitted"); the same for the worse policy $`(0,0,0.75,0.75,0)`$ ("stale"); and $`\hat v=0`$ ("zero").
3. Over 10,000 episodes, compute each episode's gradient estimate with the whole return, $`\gamma^tG_t`$, $`\gamma^t(G_t-\hat v(S_t))`$, $`G_t`$, and GAE($`\lambda`$) for $`\lambda=0`$, 0.5, 0.9, 0.97, and 1 with each critic.
4. For the average of 10 episodes, compute the squared bias and the variance, relative to the squared norm of the target: the mean of the $`\gamma^tG_t`$ estimates for the first group, and of the $`G_t`$ estimates for the others, which estimate a different direction (exercise 13.7).

## <a id="part-4-the-natural-policy-gradient"></a>Part 4 — The natural policy gradient

1. Write a batch actor–critic: each iteration runs 10 episodes, computes GAE(0.95) with a quadratic critic fitted by least squares to the previous batch's returns, and averages the estimates into $`\hat{\mathbf g}`$.
2. The vanilla update is $`\boldsymbol\theta\leftarrow\boldsymbol\theta+\alpha\hat{\mathbf g}`$. For the natural update, estimate the Fisher matrix $`\hat{\mathbf F}`$ as the average over all steps of the batch of $`\nabla\ln\pi\,\nabla\ln\pi^\top`$, add $`10^{-3}`$ of its diagonal as damping, and take the step $`\Delta=\sqrt{2\delta/\hat{\mathbf g}^\top\hat{\mathbf F}^{-1}\hat{\mathbf g}}\;\hat{\mathbf F}^{-1}\hat{\mathbf g}`$, whose quadratic estimate of the KL divergence, $`\tfrac12\Delta^\top\hat{\mathbf F}\Delta`$, equals $`\delta`$. The matrix is only $`5\times5`$, so solve with it directly.
3. Run $`\alpha\in\{0.003,0.01,0.03\}`$ and $`\delta\in\{0.003,0.01,0.03\}`$ for 100 iterations, 10 seeds, with the scaled features and with the raw state, $`\mathbf x(s)=(s,1)`$. Give all configurations the same initial states and random numbers, seed by seed, and compare the natural gradient runs with raw and scaled features episode by episode.

## <a id="expected-results"></a>Expected results

```text
=== Part 1: a vectorized CartPole, and REINFORCE with a linear softmax policy ===
  largest state difference from Gymnasium over 20 random episodes: 0.0e+00
  average episode length (the return, at most 500), 10 seeds; solved: last 100 episodes average over 475
                               episodes 1-200 201-400 401-600 601-800 801-1000  solved
       whole, alpha = 3e-04          24.2    37.1    81.0   105.6   117.8     0 of 10
       whole, alpha = 1e-03          37.2   101.1   100.3   102.0   104.1     0 of 10
       whole, alpha = 3e-03          29.9    41.2    42.1    49.8    38.7     0 of 10
  disc-to-go, alpha = 3e-04          25.1    43.5   138.4   238.6   282.5     0 of 10
  disc-to-go, alpha = 1e-03          66.5   194.9   245.2   274.1   269.4     0 of 10
  disc-to-go, alpha = 3e-03          89.6   152.3   147.4   173.2   237.1     0 of 10
       to-go, alpha = 3e-04          25.4    77.7   212.8   266.8   286.6     1 of 10
       to-go, alpha = 1e-03          83.3   148.5   186.9   189.1   198.5     0 of 10
       to-go, alpha = 3e-03          63.9   122.2   119.3   179.6   178.1     0 of 10
    baseline, alpha = 3e-04          26.3   110.8   385.6   465.1   478.5     9 of 10
    baseline, alpha = 1e-03         143.3   436.6   452.3   467.5   468.5     6 of 10
    baseline, alpha = 3e-03         235.3   322.5   418.0   400.1   382.5     3 of 10

=== Part 2: online actor-critic (linear softmax actor, Fourier critic of order 2) ===
  average episode length, 10 seeds             episodes 1-50  51-100 101-150 151-200 201-250 251-300  solved
  one-step actor-critic                         424.7   500.0   500.0   500.0   500.0   500.0    10 of 10
    same, time limit treated as termination     364.6   426.5   463.4   432.4   474.4   465.4     4 of 10
  actor-critic, lambda = 0.9                    382.5   500.0   500.0   500.0   500.0   500.0    10 of 10
  for comparison, REINFORCE with baseline        24.1    45.0   174.4   329.7   404.7   444.9

=== Part 3: gradient estimators at a fixed policy (average episode length about 110), batches of 10 episodes ===
  relative squared error of a 10-episode estimate = squared bias + variance, both divided by |gradient|^2
  (a) estimates of the gradient of the discounted value of the start state:
    whole return G_0                         bias^2  0.004   variance 16.778
    gamma^t G_t                              bias^2  0.000   variance  1.972
    gamma^t (G_t - v(S_t)), fitted critic    bias^2  0.001   variance  0.859
  (b) estimates of the direction without gamma^t: G_t, and GAE(lambda) with three critics (bias^2 + variance):
    G_t, no critic: 0.000 + 1.527
                     lambda = 0       lambda = 0.5     lambda = 0.9     lambda = 0.97    lambda = 1    
    fitted critic   0.002 + 0.136  0.002 + 0.153  0.002 + 0.286  0.001 + 0.441  0.001 + 0.652
    stale critic    0.160 + 0.045  0.155 + 0.046  0.153 + 0.060  0.093 + 0.134  0.000 + 0.535
    zero critic     0.999 + 0.001  0.995 + 0.003  0.817 + 0.051  0.424 + 0.265  0.000 + 1.527
  (from 10,000 episodes, each squared bias is uncertain by about a thousandth of its variance)

=== Part 4: vanilla versus natural policy gradient, 10 episodes per iteration, GAE(0.95) ===
  average episode length, 10 seeds                    iterations 1-20   21-40   41-60   61-80  81-100  solved
  vanilla, alpha = 0.003, scaled features             25.1    45.1   172.4   348.5   402.7     1 of 10
  vanilla, alpha = 0.01, scaled features              53.7   223.8   277.4   311.8   297.0     2 of 10
  vanilla, alpha = 0.03, scaled features              65.7   136.4   134.4   146.9   139.8     0 of 10
  natural, delta = 0.003, scaled features             99.6   323.8   373.7   388.2   401.1     5 of 10
  natural, delta = 0.01, scaled features             178.9   338.6   380.2   395.1   392.2     3 of 10
  natural, delta = 0.03, scaled features             176.3   331.7   359.2   340.9   356.6     1 of 10
  vanilla, alpha = 0.003, raw features                27.9    42.9    57.5    93.8   168.9     0 of 10
  vanilla, alpha = 0.01, raw features                 39.5    72.2   123.3   128.4   166.9     0 of 10
  vanilla, alpha = 0.03, raw features                 48.9    64.3    50.5    54.3    80.4     0 of 10
  natural, delta = 0.003, raw features                99.6   323.8   373.7   388.2   401.1     5 of 10
  natural, delta = 0.01, raw features                178.9   338.6   380.2   392.7   390.1     2 of 10
  natural, delta = 0.03, raw features                176.3   332.2   364.7   352.8   335.0     0 of 10
  natural, delta = 0.003: raw and scaled features give identical runs in 10 of 10 seeds
  natural, delta = 0.01: raw and scaled features give identical runs in 6 of 10 seeds; the others first differ at iterations 70, 89, 90, 95
  natural, delta = 0.03: raw and scaled features give identical runs in 0 of 10 seeds; the others first differ at iterations 36, 36, 37, 37, 37, 37, 43, 47, 58, 58
```

Things to notice:

- **Part 1.** The simulator matches Gymnasium to the last bit, so every result below is about `CartPole-v1` itself. The whole-return estimator is by far the worst: it multiplies each score by rewards that arrived before the action, which add noise and no signal. Using the return from each step, $`\gamma^tG_t`$, more than doubles the final performance; dropping $`\gamma^t`$ changes little here. The baseline changes everything: 9 of 10 seeds solve the task with it, against at most one without. In the cart-pole every reward is $`+1`$, so every return is large and positive; without a baseline every action taken is reinforced, and the signal is the small difference between large, noisy increases. This is the opposite of the short corridor of chapter 13, where the returns vary about as much as they average and the baseline helps less. Note also the step sizes: with the baseline, the largest, $`3\times10^{-3}`$, is fastest over the first 200 episodes and the smallest best over the last 200, so a constant step size is a compromise.
- **Part 2.** The one-step actor–critic balances the pole for the full 500 steps in essentially every episode after the first 50, in every seed, while REINFORCE with baseline ($`\alpha=10^{-3}`$) averages 45 steps per episode over episodes 51–100. The actor–critic learns at every step from a one-step TD error; REINFORCE waits for the end of an episode and then uses a noisy return. Traces do not help here: with $`\lambda=0.9`$ the agent is slightly slower over the first 50 episodes and as good afterward. Treating the time limit as termination makes the agent unstable, with only 4 of 10 seeds solved at the end: at step 500 the critic is told that a state like any other is worth nothing, since its features do not include the time, and the TD errors this produces punish the actions that kept the pole up ([Pardo, Tavakoli, Levdik, and Kormushev, 2018](https://arxiv.org/abs/1712.00378)).
- **Part 3.** A relative variance above 1 means the noise of a 10-episode estimate is larger than the gradient itself. The whole-return estimator's is 17; the return from each step brings it to 2.0, and the baseline to 0.86, all without bias. For the estimates of the direction without $`\gamma^t`$, the critic's quality decides the best λ. With the fitted critic, the bias is negligible at every λ and the variance falls as λ decreases, so $`\lambda=0`$ is best: an accurate critic should be trusted. With the stale critic, small λ is biased, and $`\lambda`$ between 0 and 0.9 gives errors near 0.2, against 0.54 at $`\lambda=1`$. With no critic at all, $`\lambda<1`$ only shortens the horizon, and $`\lambda=0.97`$ is best. In deep RL, where the critic lags behind a changing policy, values around 0.95 are the usual compromise (chapter 19).
- **Part 4.** The natural gradient learns about three times faster in the first 20 iterations than the best vanilla step size. The vanilla gradient depends on the features: with the raw state, the pole's angle is a number about five times smaller than in the scaled features, so its parameter must become five times larger while its gradient is five times smaller: at the same step size, progress along it is about 25 times slower, and the step size cannot grow to compensate, because the other components are now larger. No vanilla step size averages more than 170 in any window. The natural gradient is invariant: with $`\delta=0.003`$, raw and scaled features give identical runs, episode for episode, in all 10 seeds. With larger steps, rounding differences eventually flip an action and the runs part, but their averages stay close. Measuring each step by the KL divergence keeps the policy moving by the same amount even near the optimum, so the natural gradient levels off around 400; decreasing δ, or checking each step on the objective as TRPO does (chapter 20), fixes that.

## <a id="going-further"></a>Going further

1. **Time as a feature.** Add $`t/500`$ to the critic's features in part 2 and treat the time limit as termination. Does the agent recover the performance of the bootstrapping version? Which is the better fix when the time limit is part of the task, and which when it is only a convenience of the simulation?
2. **A constant baseline.** In part 1, subtract a constant $`b`$ from $`G_t`$ for several values of $`b`$, including the average return. The expected gradient does not change; how does the performance change, and how does the best $`b`$ compare with the optimal baseline of exercise 13.3?
3. **Entropy regularization.** Add $`\tau\nabla\mathcal H(\pi(\cdot\mid S_t))`$ to the actor's update in part 4, as in exercise 13.8. Does it help the natural gradient past its plateau, or does it make the plateau lower?
4. **A neural network policy.** Replace the linear policy of part 2 by a network with one hidden layer of 64 units in PyTorch, with a network critic, and update from batches of steps collected from 8 parallel copies of the environment. This is A2C (chapter 19). How much faster or slower is it than the linear agent, and how sensitive to its step sizes?
5. **Natural gradients without the matrix.** For a network, $`\mathbf F`$ is too large to form. Compute $`\mathbf F^{-1}\hat{\mathbf g}`$ by conjugate gradient, using only products $`\mathbf F\mathbf v`$ computed as averages of $`\nabla\ln\pi\,(\nabla\ln\pi^\top\mathbf v)`$, and add a backtracking line search that accepts a step only if the empirical KL divergence is below δ and a surrogate estimate of the improvement is positive. Check that it matches part 4 on the linear policy: this is TRPO.

---

[← Lab 5. Tile Coding and Linear Control on Mountain Car](lab-05-tile-coding-and-linear-control-on-mountain-car.md) · [Lab 7. System Identification, LQR, and Model Predictive Control →](lab-07-system-identification-lqr-and-model-predictive-control.md)
