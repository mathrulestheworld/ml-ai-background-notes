[Background Notes](../../README.md) › [Reinforcement Learning](../README.md)

# Lab 7. System Identification, LQR, and Model Predictive Control

[← Lab 6. Policy Gradient and Actor-Critic Methods on CartPole](lab-06-policy-gradient-and-actor-critic-methods-on-cartpole.md) · [Lab 8. Deep Q-Networks on CartPole and MinAtar →](lab-08-deep-q-networks-on-cartpole-and-minatar.md)

## <a id="overview"></a>Overview

This lab puts chapter 15 to work, and takes a first step into model-based reinforcement learning. You will identify a linear model of the cart-pole from data and control it with the LQR of the estimated model, swing up Gymnasium's pendulum with model predictive control, and replace the pendulum's known dynamics with models learned from random transitions.

- **Environments:** the cart-pole of Lab 6 with a continuous force, simulated with the dynamics of Gymnasium's `CartPole-v1`; and `Pendulum-v1`, whose state is the angle $`\theta`$ (0 upright) and angular velocity, whose torque is limited to $`[-2,2]`$, too little to lift the pendulum directly, and whose reward is $`-(\theta^2+0.1\dot\theta^2+0.001u^2)`$ per step, with $`\theta`$ wrapped to $`[-\pi,\pi]`$, over 200-step episodes from random states.
- **Prerequisites:** chapters 11 and 15; NumPy, SciPy (`solve_discrete_are`, `solve_discrete_lyapunov`), and Gymnasium.
- **Reference solution:** [lab07_optimal_control.py](code/lab07_optimal_control.py), under a minute on one core. Try each part yourself before reading it.

## <a id="part-1-identification-and-certainty-equivalence"></a>Part 1 — Identification and certainty equivalence

1. Write the cart-pole dynamics with a continuous force, and compute the linearization $`(A,B)`$ at the upright equilibrium by finite differences. Compute the LQR gain for $`Q=\operatorname{diag}(1,0.1,10,0.1)`$ and $`R=0.01`$, and the optimal cost $`\operatorname{tr}P`$ for initial states with covariance $`I`$.
2. Collect $`N`$ transitions from random states near upright (each component with standard deviation 0.05) under random forces (standard deviation 1), with a little measurement noise, and estimate $`[A\;B]`$ by least squares, regressing $`\mathbf x_{t+1}`$ on $`(\mathbf x_t,u_t)`$.
3. Compute the LQR gain of the estimated model, the **certainty-equivalent** controller. Evaluate it on the true linearization, by the cost of its closed loop from the Lyapunov equation, and on the nonlinear cart-pole with the force limited to 10 N, from a tilt of 15°.
4. Repeat for $`N`$ from 10 to 10,000, with 20 data sets each, and report medians.

## <a id="part-2-model-predictive-control-of-the-pendulum"></a>Part 2 — Model predictive control of the pendulum

1. Write `Pendulum-v1`'s dynamics, including its clipping of the angular velocity to $`[-8,8]`$, and check them against `env.unwrapped.state`.
2. Implement MPC with MPPI: at each step, perturb the current $`H`$-step torque sequence with Gaussian noise (standard deviation 1), roll out $`n`$ sequences in the model in parallel, weight them by $`e^{-J/\lambda}`$ with $`\lambda=0.3`$, update the sequence, repeat once, apply the first torque, and shift the sequence for the next step.
3. Evaluate the average return over 10 episodes from Gymnasium's random initial states, for horizons of 10, 20, 30, and 50 steps with 200 samples, and 30 steps with 50 samples, and compare with random and zero torques.

## <a id="part-3-mpc-with-a-learned-model"></a>Part 3 — MPC with a learned model

1. Collect 1,000 transitions with uniformly random torques from `Pendulum-v1`, resetting every 200 steps.
2. Fit models of the angular acceleration, $`(\dot\theta_{t+1}-\dot\theta_t)/\Delta t`$, by least squares on four sets of features: $`(\sin\theta,u)`$, the true form; $`(\theta,u)`$, a linearization; $`u`$ alone, which ignores gravity; and all quadratic monomials in $`(\cos\theta,\sin\theta,\dot\theta,u)`$, a generic model that knows nothing about pendulums. Integrate each as the true dynamics do.
3. Use each model in the MPC of part 2 with $`H=30`$ and $`n=200`$, and compare the returns.

## <a id="expected-results"></a>Expected results

```text
=== Part 1: least-squares identification of the cart-pole near upright, and certainty-equivalent LQR ===
  N transitions   error in [A B]   error in the gain   relative excess cost on the linearization   balances from 15 degrees
             10          2.0e-02             2.5e-01                                    9.0e-01                17 of 20
             20          1.1e-02             1.6e-01                                    1.1e-01                20 of 20
             50          5.7e-03             8.8e-02                                    5.9e-02                20 of 20
            100          4.3e-03             8.0e-02                                    2.8e-02                20 of 20
          1,000          1.2e-03             2.1e-02                                    2.2e-03                20 of 20
         10,000          4.9e-04             4.6e-03                                    1.3e-04                20 of 20
  (medians over 20 data sets)

=== Part 2: MPC with MPPI on Gymnasium's Pendulum-v1 ===
  largest state difference from Gymnasium over 5 random episodes: 0.0e+00
  average return over 10 episodes (standard deviation):
    random torques:                        -1314.0 (257.9)
    no torque:                             -1285.5 (335.8)
    MPC, horizon 10, 200 samples:            -416.4 (388.0)   0.6 ms per step
    MPC, horizon 20, 200 samples:            -168.2 ( 74.7)   1.3 ms per step
    MPC, horizon 30, 200 samples:            -205.6 ( 71.2)   2.0 ms per step
    MPC, horizon 30,  50 samples:            -204.6 ( 69.9)   1.5 ms per step
    MPC, horizon 50, 200 samples:            -220.4 ( 95.0)   3.4 ms per step

=== Part 3: MPC with models learned from 1,000 random transitions ===
  sin(theta), torque (the true form)           [14.94  2.89]   fit error  0.91   MPC return   -209.7 ( 75.4)
  theta, torque (linearized)                   [5.39 2.79]     fit error  6.67   MPC return   -948.9 (281.6)
  torque only (no gravity)                     [3.2]           fit error 10.74   MPC return  -1259.5 (208.1)
  all quadratics in (cos, sin, speed, torque)  15 weights      fit error  0.82   MPC return   -209.8 ( 75.6)
  (the true weights are 3g/(2l) = 15 and 3/(ml^2) = 3)
```

The times per step depend on the machine.

Things to notice:

- **Part 1.** Each transition gives one equation for each of the four rows of $`[A\;B]`$, each row with five unknowns, so even ten transitions determine a model; but with so few, the noise and the nonlinearity leave the gain 25% off, and three of the twenty controllers fail. Twenty transitions already give a controller that balances the real, nonlinear cart-pole from 15° in every data set, with a cost 11% above optimal on the linearization. The excess cost falls much faster than the errors: from 1,000 to 10,000 transitions the gain's error falls by a factor of 4.6 and the excess cost by 17, roughly the square. This is the certainty-equivalence result of [Mania, Tu, and Recht (2019)](https://arxiv.org/abs/1902.07826): because the LQR gain is optimal, the cost is flat to first order around it, and errors in the gain cost only quadratically. The model's error does not fall to zero at the rate of pure noise, since the data come from a nonlinear system and the best linear fit over states of size 0.05 differs slightly from the linearization at the equilibrium. Compare the numbers of transitions with the thousands of episodes that the model-free methods of Lab 6 needed on the same system.
- **Part 2.** MPC swings the pendulum up and balances it with no learning at all, for an average return between $`-170`$ and $`-220`$ with horizons of 20 to 50 steps, against about $`-1,300`$ for random or zero torques. The horizon matters most. Ten steps (half a second) is too short to see that swinging away from the top can lead to it, and from some states the controller never swings up, hence the large spread. Longer horizons than 20 do slightly worse with the same number of samples, since a longer sequence is harder to optimize by random perturbation, and they cost more computation per step. Four times fewer samples at 30 steps change little, because the plan is warm-started from the previous step and improves over many steps.
- **Part 3.** A model with the right form, fitted from 1,000 random transitions, is as good as the true model for control: its weights are within a few percent of the true values, and its remaining fit error comes entirely from the 2.6% of transitions that hit the velocity limit, which the features cannot represent. The generic quadratic model, which contains the true form among its 15 features, does as well. A linear model of gravity, accurate only near the top, fits badly and controls badly, and a model without gravity is no better than doing nothing. Model-based reinforcement learning is this recipe with richer models, more careful data collection, and uncertainty about the model (chapter 23).

## <a id="going-further"></a>Going further

1. **Robust LQR from little data.** In part 1 with $`N=10`$, bootstrap the data set to estimate the uncertainty in $`(A,B)`$, and choose the gain that minimizes the worst cost over the bootstrap models. Does it fail less often than the certainty-equivalent gain?
2. **iLQR as the MPC optimizer.** Replace MPPI in part 2 by a few iterations of iLQR per step, warm-started from the shifted previous solution, and compare the returns, the computation per step, and the sensitivity to the initial torque sequence. Where does iLQR get stuck?
3. **A terminal value.** Add to the MPC objective a terminal cost equal to the value function of an LQR around the upright position, applied only when the pendulum is near the top. How much shorter can the horizon be?
4. **Data collection with the controller.** In part 3, collect the data with the MPC itself, starting from a model learned on 100 random transitions and refitting after each episode. How few transitions are needed, and what goes wrong with the misspecified models when the data come from the controller's own behavior?
5. **Energy-shaping swing-up.** Write the classical hand-designed controller: pump energy with $`u=k\,\dot\theta\,(E_{\text{top}}-E)`$ until the pendulum is near the top, then switch to LQR. Compare its return and robustness to model errors with MPC.

---

[← Lab 6. Policy Gradient and Actor-Critic Methods on CartPole](lab-06-policy-gradient-and-actor-critic-methods-on-cartpole.md) · [Lab 8. Deep Q-Networks on CartPole and MinAtar →](lab-08-deep-q-networks-on-cartpole-and-minatar.md)
