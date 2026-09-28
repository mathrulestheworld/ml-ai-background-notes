[Background Notes](../../README.md) › [Reinforcement Learning](../README.md)

# Lab 12. Model-Based RL with Learned Ensembles

[← Lab 11. Off-Policy Actor-Critics for Continuous Control](lab-11-off-policy-actor-critics-for-continuous-control.md) · [Lab 13. AlphaZero and Gumbel Search from Scratch →](lab-13-alphazero-and-gumbel-search-from-scratch.md)

## <a id="overview"></a>Overview

This lab puts [chapter 23](../23-model-based-rl-and-world-models.md) to work. You will learn a model of a pendulum's dynamics from a few hundred transitions, plan through it with the cross-entropy method as PETS does, and then use the same kind of model to generate short rollouts for a soft actor–critic as MBPO does. Along the way you will test two of the chapter's claims against small experiments: that modeling uncertainty keeps a planner out of trouble, and that a model's rollouts, rather than many updates per real sample, are what make MBPO data-efficient.

- **Environment:** `Pendulum-v1`: swing a pendulum up and hold it upright with a torque limited to $`[-2,2]`$. The observation is $`(\cos\theta,\sin\theta,\dot\theta)`$, the reward is $`-(\theta^2+0.1\,\dot\theta^2+0.001\,u^2)`$ with $`\theta`$ the angle from upright, and episodes last 200 steps, so a return above about $`-200`$ means the pendulum is swung up and held. The agent is given the reward function and learns only the dynamics.
- **Prerequisites:** chapter 23, the cross-entropy method of [chapter 15](../15-optimal-control-and-trajectory-optimization.md#sampling-based-optimization), and the soft actor–critic of [Lab 11](lab-11-off-policy-actor-critics-for-continuous-control.md); PyTorch and Gymnasium.
- **Reference solution:** [lab12_model_based.py](code/lab12_model_based.py), about 15 minutes on two cores, a third of it in part 1. Try each part yourself before reading it.

## <a id="part-1-probabilistic-ensembles-and-model-predictive-control"></a>Part 1 — Probabilistic ensembles and model predictive control

1. **The model.** Write an ensemble of $`E`$ networks trained together (batched matrix multiplications make this as cheap as one wider network). Each member maps $`(\cos\theta,\sin\theta,\dot\theta/8,u/2)`$ through two hidden layers of 64 SiLU units to the mean and log-variance of a Gaussian over the *change* in the observation, with the speed component scaled back up by 8. Bound the log-variance softly between learned limits, as PETS does: $`\log\sigma^2\leftarrow v_{\max}-\operatorname{softplus}(v_{\max}-\log\sigma^2)`$ and $`\log\sigma^2\leftarrow v_{\min}+\operatorname{softplus}(\log\sigma^2-v_{\min})`$, starting from $`v_{\max}=0`$ and $`v_{\min}=-10`$.
2. **Training.** Minimize the Gaussian negative log-likelihood $`(\mu-s')^2e^{-\log\sigma^2}+\log\sigma^2`$, averaged over components, plus $`0.01(v_{\max}-v_{\min})`$ to keep the limits tight. Give each member its own bootstrap minibatch of 256 drawn with replacement from all the data, and after every episode take 400 Adam steps with step size $`10^{-3}`$. For comparison, write a deterministic variant: one network that outputs a mean only, trained by squared error.
3. **Planning.** At every step, choose the action by the cross-entropy method over sequences of $`H=15`$ torques: sample 200 sequences from a Gaussian, clip them to the torque limits, score each, refit the Gaussian to the best 20, and repeat 4 times, then execute the first action of the final mean. Start from the previous step's plan shifted by one step, with standard deviation 1.5, and keep at least 0.05. Score a sequence by propagating one particle per member, each sampling its next observation from its own member's Gaussian at every step (the "TS∞" scheme of exercise 23.4), and averaging the particles' predicted returns.
4. **Learning loop.** Run one episode of uniformly random torques, then 14 episodes of planning, refitting the model to all the data after each. Record each episode's return, for the ensemble of 5 and for the deterministic network, with 3 seeds.

## <a id="part-2-branched-model-rollouts-for-a-model-free-learner"></a>Part 2 — Branched model rollouts for a model-free learner

1. Start from your SAC of Lab 11, with networks of two layers of 64 units, Adam with step size $`3\times10^{-4}`$, minibatches of 256, $`\gamma=0.99`$, $`\tau=0.005`$, a target entropy of $`-1`$, and 500 initial random steps. Run it for 4,000 environment steps with one update per step and with 10.
2. **MBPO.** From step 500, fit a probabilistic ensemble of 5 to the real data every 250 steps (300 Adam steps). At every environment step, start 50 rollouts from states drawn from the real replay memory, and run each for $`k`$ steps under the current stochastic policy, choosing a random member for each rollout at each step and computing the rewards with the known reward function. Grow $`k`$ linearly from 1 to 5 over the first 3,000 steps, and store the model transitions in a separate memory of 50,000.
3. Train SAC with 10 updates per step on minibatches of 243 model and 13 real transitions, 95% model data as in MBPO.
4. Every 1,000 steps, evaluate the deterministic policy over 5 episodes. Use 3 seeds for each of the three agents.

## <a id="expected-results"></a>Expected results

```text
=== Part 1: learning the pendulum with a learned model and MPC (15 episodes of 200 steps, 3 seeds) ===
  return of each episode (the first with random torques), mean of seeds; a return above about -200 means
  the pendulum is swung up and held
                                      episode:  1       2       3       5       8      10      15
  probabilistic ensemble of 5 (PETS)        -1346    -721    -810    -255    -168    -284    -163   seeds at episode 15:  -124  -124  -240
  one deterministic network                 -1346    -669    -128    -166    -124    -123    -123   seeds at episode 15:  -129  -115  -125

=== Part 2: SAC, SAC with 10 updates per step, and MBPO, 4,000 environment steps, 3 seeds ===
  average return of the deterministic policy over 5 episodes, mean of seeds
                                        steps: 1,000   2,000   3,000   4,000   seeds at 4,000
  SAC, 1 update per step                        -1570   -1659    -867   -1300   -1287 -1339 -1273
  SAC, 10 updates per step                       -904    -142    -150    -143    -142  -144  -142
  MBPO (SAC, 10 updates, model data)            -1489    -303    -165    -141    -141  -140  -141
```

Things to notice:

- **Part 1, data efficiency.** With the deterministic network, the planner swings the pendulum up and holds it in the third episode, after learning from 400 transitions, 200 of them random, and in the episodes printed after that its mean return stays between $`-166`$ and $`-123`$. The ensemble gets there by the fifth to eighth episode. SAC in Lab 11 needed about 9,000 transitions for the same result, and in part 2 it has not learned it after 4,000. This is the case in which models shine: low-dimensional, smooth, deterministic dynamics that a small network fits from little data, and a reward the agent knows.
- **Part 1, the ensemble did not pay.** The probabilistic ensemble, with five times the computation per candidate, learned more slowly than a single deterministic network and was less reliable: its mean return dips again at episode 10, where at least one seed failed to hold the pendulum, and at episode 15 one seed returned $`-240`$, where the three deterministic seeds returned between $`-115`$ and $`-129`$. The pendulum has no aleatoric noise to model, and a few hundred transitions cover the small state space well enough that the members soon agree. What remains of the modeled variance most likely just adds noise to the particles' returns, and averaging over members makes the planner cautious about swings that some members predict poorly. The benefit of uncertainty that the chapter describes, keeping the planner out of regions where the model is wrong, needs regions where the model *is* wrong and a planner that can exploit them: larger state spaces, stochastic dynamics, or scarcer data (going further 1 and 2).
- **Part 2, updates per sample.** SAC with one update per step is still far from balancing after 4,000 steps. With ten updates per step it balances by 2,000 steps in every seed, and MBPO, which also takes ten updates per step but trains mostly on model data, gets there by 3,000. MBPO's head start in its paper was measured against SAC with one update per step; with the replay ratio equalized, it disappears here, which is what REDQ and DroQ found at larger scale ([chapter 21](../21-continuous-control-and-maximum-entropy-rl.md)). MBPO also starts more slowly, because its early minibatches are 95% rollouts of a model fit to a few hundred random transitions.
- **Part 2, when the model data should help.** A high replay ratio on real data alone is limited by overfitting: the critic sees the same few thousand transitions hundreds of times. Model rollouts from those states add new actions and new next states near the data, which helps when the real data are scarce relative to the difficulty of the task, as in MBPO's MuJoCo experiments. On the pendulum, the real data already suffice.

## <a id="going-further"></a>Going further

1. **Uncertainty where it matters.** Make the pendulum stochastic by adding Gaussian noise to the applied torque, or by drawing the pendulum's mass anew in each episode, and repeat part 1. Does the probabilistic ensemble now beat the deterministic network, and does each of its two ingredients, the variance head and the ensemble, matter on its own?
2. **Watching a planner exploit its model.** Train a deterministic network on the random episode only, then plan with a longer horizon ($`H=50`$) and more iterations of the cross-entropy method. Compare the planner's predicted returns with the returns obtained, and repeat with the ensemble. Relate the gap to exercise 23.1.
3. **Rollout length and the real-data fraction.** Run MBPO with $`k`$ fixed at 1, 5, and 15, and with a 50% fraction of real data. Which settings hurt, and how does this match the bound of exercise 23.3?
4. **A harder body.** Compare MBPO with SAC at 10 updates per step on `HalfCheetah-v5` for 30,000 steps, with networks of 200 units and an ensemble of 7. Does the model help once the dynamics are harder to fit and the real data scarcer?
5. **A value at the end of the horizon.** Add SAC's learned value of the final predicted state to the cross-entropy method's score, as TD-MPC does, and shorten the horizon to 5. How short can the horizon become before planning stops helping?

---

[← Lab 11. Off-Policy Actor-Critics for Continuous Control](lab-11-off-policy-actor-critics-for-continuous-control.md) · [Lab 13. AlphaZero and Gumbel Search from Scratch →](lab-13-alphazero-and-gumbel-search-from-scratch.md)
