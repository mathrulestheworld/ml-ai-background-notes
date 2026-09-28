[ML Mastery Notes](../../README.md) › [Reinforcement Learning](../README.md)

# Lab 11. Off-Policy Actor-Critics for Continuous Control

[← Lab 10. Proximal Policy Optimization from Scratch](lab-10-proximal-policy-optimization-from-scratch.md) · [Lab 12. Model-Based RL with Learned Ensembles →](lab-12-model-based-rl-with-learned-ensembles.md)

## <a id="overview"></a>Overview

This lab puts chapter 21 to work. You will implement DDPG, TD3, and soft actor–critic in one short program that shares a replay memory and networks, compare them on a pendulum, look at what their critics believe, and train two of them to run a simulated cheetah with a tenth of the data PPO needed in Lab 10.

- **Environments:** `Pendulum-v1`: swing a pendulum up and hold it upright with a torque limited to $[-2,2]$; 200-step episodes, rewards between about $-16$ and 0 per step, so a return near $-150$ or better means the pendulum is swung up and held. `HalfCheetah-v5` (MuJoCo): 6 torques in $[-1,1]$, a reward for forward speed minus a small control cost, 1,000-step episodes.
- **Prerequisites:** chapter 21; PyTorch and Gymnasium with the MuJoCo extra.
- **Reference solution:** [lab11_offpolicy.py](code/lab11_offpolicy.py), about 40 minutes on two cores, most of it in part 2. Try each part yourself before reading it.

## <a id="part-1-three-off-policy-actorcritics"></a>Part 1 — Three off-policy actor–critics

1. Write a replay memory of transitions $(s,a,r,s',d)$ with $d=1$ only for true terminations, and scale all actions to $[-1,1]$, multiplying by the action bound only when stepping the environment.
2. **DDPG:** a deterministic actor with a tanh output, one critic, Gaussian exploration noise of standard deviation 0.1, target networks for both with $\tau=0.005$, and one update of each per environment step.
3. **TD3:** add a second critic and use the minimum of the two target critics in the target, smooth the target action with clipped noise ($\tilde\sigma=0.2$, $c=0.5$), and update the actor and the target networks every second step. Also run a variant without the clipped double Q, whose targets use the first target critic only.
4. **SAC:** a squashed Gaussian actor whose network outputs the mean and log standard deviation, with the tanh correction of the log-density computed stably; twin critics with the soft target; the reparameterized actor loss; and automatic tuning of the temperature toward an entropy of $-\dim\mathcal A$.
5. On `Pendulum-v1`, with networks of two layers of 64 units, Adam with step size $3\times10^{-4}$, minibatches of 256, and 1,000 initial random steps, train each algorithm for 15,000 steps with 3 seeds. Every 1,500 steps, evaluate the deterministic policy (the mean action, for SAC) over 5 episodes. At the end, compare the first critic's value of the initial state of each evaluation episode with the discounted return actually obtained.

## <a id="part-2-running"></a>Part 2 — Running

Train TD3 and SAC on `HalfCheetah-v5` for 100,000 steps with networks of two layers of 256 units and 5,000 initial random steps, 2 seeds each, evaluating over 3 episodes every 10,000 steps.

## <a id="expected-results"></a>Expected results

```text
=== Part 1: Pendulum-v1, 15,000 steps (the first 1,000 random), networks of 64 units, 3 seeds ===
  average return of the deterministic policy over 5 episodes (mean of seeds); at the end, the critic's value
  of the first state against the discounted return actually obtained; the final return of each seed
                                3,000   6,000   9,000  12,000  15,000   predicted    actual   per seed
  DDPG                            -980    -235    -108     -98     -98       -89.6     -87.5      -99    -97    -98
  TD3                            -1425    -387    -124    -101    -280      -107.2    -157.5     -102   -639   -100
  TD3 without clipped double Q   -1420    -326    -131    -103    -561       -95.4    -255.4     -761   -820   -102
  SAC                            -1245    -249    -107     -97     -97      -134.6     -87.4      -96    -96    -98

=== Part 2: HalfCheetah-v5, 100,000 steps (the first 5,000 random), networks of 256 units, 2 seeds ===
                                  20k     40k     60k     80k    100k   predicted    actual   per seed
  TD3                              -51    1066    2404    3262    4233       125.5     324.3     3942   4524
  SAC                             -195    1997    3554    4424    5066       200.7     395.4     5280   4853
```

Things to notice:

- **Part 1, learning speed.** All four agents swing the pendulum up within about 9,000 steps, 45 episodes, and DDPG and SAC stay there in every seed. On this small task DDPG is not the brittle algorithm of the MuJoCo benchmarks, and its critic is accurate: it predicts $-89.6$ for the initial states, where its policy obtains $-87.5$.
- **Part 1, instability and overestimation.** The two TD3 variants were balancing the pendulum at 12,000 steps, and some of their seeds collapsed in the last 3,000: one of three for TD3, two of three without the clipped double Q. The collapsed variant without the clipped double Q is also where the critic's beliefs and the outcomes diverge most: its critic predicts $-95$ for the initial states while its policy obtains $-255$, an overestimate of 160, against 50 for TD3 with it. With three seeds and one final evaluation, this is suggestive rather than conclusive: the late collapses show that off-policy actor–critics can lose a good policy, and that evaluation curves must be read over many checkpoints and seeds.
- **Part 1, SAC's critic.** SAC's critic predicts $-135$ where its policy obtains $-87$. This is not an error: the critic estimates *soft* values, which include the entropy bonus $-\alpha\ln\pi$ at future steps, and the log-density of a narrow squashed Gaussian is positive, so the bonus is negative once the policy has become nearly deterministic.
- **Part 2.** SAC and TD3 run at 3,900 to 5,300 after 100,000 steps, far beyond what PPO reached in Lab 10 after a million steps, 1,400 to 2,700. Every transition is used for many updates, and the critics learn from all past data rather than only the latest batch. SAC is ahead in both seeds, as it usually is on this task. Both critics now *under*estimate the initial states' values, predicting only 40 to 50% of the discounted return obtained: the policy is improving quickly, and the critics' bootstrapped targets lag behind the returns of the current policy.

## <a id="going-further"></a>Going further

1. **More seeds, more checkpoints.** Rerun part 1 with 10 seeds and evaluations every 500 steps, and plot the median and interquartile range. Which differences survive?
2. **Overestimation in isolation.** Freeze the actor of a trained DDPG agent and train a fresh critic on its replay memory with and without the clipped double Q, comparing the predictions with Monte Carlo returns from many states.
3. **High update-to-data ratios.** Give SAC 5 or 20 updates per step on `HalfCheetah-v5` for 30,000 steps, then add layer normalization to the critics, as DroQ does. When does the extra computation pay?
4. **A fixed temperature.** Replace SAC's automatic temperature by fixed values of $\alpha$ from 0.01 to 1, and relate the results to exercise 21.4.
5. **Another body.** Train SAC on `Hopper-v5` or `Walker2d-v5`, where episodes end when the robot falls. How does termination change the role of the entropy bonus?

---

[← Lab 10. Proximal Policy Optimization from Scratch](lab-10-proximal-policy-optimization-from-scratch.md) · [Lab 12. Model-Based RL with Learned Ensembles →](lab-12-model-based-rl-with-learned-ensembles.md)
