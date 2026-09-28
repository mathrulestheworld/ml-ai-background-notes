[ML Mastery Notes](../../README.md) › [Reinforcement Learning](../README.md)

# Lab 5. Tile Coding and Linear Control on Mountain Car

[← Lab 4. Planning with Learned and Given Models](lab-04-planning-with-learned-and-given-models.md) · [Lab 6. Policy Gradient and Actor-Critic Methods on CartPole →](lab-06-policy-gradient-and-actor-critic-methods-on-cartpole.md)

## <a id="overview"></a>Overview

This lab turns chapter 11 and the eligibility traces of chapter 8 into a working linear control agent for Gymnasium's mountain car. You will write a hashed tile coder, compare SARSA(λ) across values of λ, test true online SARSA(λ) against replacing traces, and solve the task offline from a batch of transitions with least-squares policy iteration.

- **Environment:** `MountainCar-v0`. The state is the car's position and velocity, the actions push left, not at all, or right, and every step costs $`-1`$ until the car reaches the flag at position 0.5. Gymnasium cuts episodes off after 200 steps; this lab raises the limit to 5,000 with `gym.make("MountainCar-v0", max_episode_steps=5000)`, so that early episodes can finish, and bootstraps correctly at truncation.
- **Prerequisites:** chapters 8 and 11; NumPy and Gymnasium.
- **Reference solution:** [lab05_mountain_car.py](code/lab05_mountain_car.py), about three minutes on one core. Try each part yourself before reading it.

## <a id="part-1-a-tile-coder"></a>Part 1 — A tile coder

1. Write a tile coder for a box in $`\mathbb R^k`$ with $`n`$ tilings of $`m`$ tiles per dimension. Offset tiling $`t`$ by $`(1,3,5,\dots)\cdot t/n`$ of a tile in the successive dimensions, the asymmetric offsets of chapter 11.
2. Map each tile, identified by (tiling, action, integer coordinates), to a feature index through a hash table that assigns indices in order of first use, and falls back to a hash with collisions when the table is full.
3. Check the generalization: how many active tiles do two nearby states share, and two distant ones?

## <a id="part-2-sarsa"></a>Part 2 — SARSA(λ)

1. Implement SARSA(λ) with replacing traces over the tile-coded features, one set of tiles per action, with 8 tilings of $`8\times8`$ tiles, $`\alpha=0.5/8`$, $`\gamma=1`$, and a greedy policy: initial weights of 0 are optimistic, since every true value is negative.
2. Run 5 runs of 200 episodes for $`\lambda=0`$, 0.5, 0.9, and 0.99, and report the average number of steps per episode in episodes 1–20, 21–100, and 101–200.

## <a id="part-3-true-online-sarsa"></a>Part 3 — True online SARSA(λ)

1. Implement true online SARSA(λ): the dutch trace $`\mathbf z\leftarrow\gamma\lambda\mathbf z+(1-\alpha\gamma\lambda\,\mathbf z^\top\mathbf x)\mathbf x`$ and the corrected update of chapter 8, with $`\hat q`$ in place of $`\hat v`$.
2. Compare it with SARSA(λ) with replacing traces for $`\lambda=0.9`$ and $`\alpha=0.5/8`$, $`1/8`$, and $`1.5/8`$, over 100 episodes.

## <a id="part-4-least-squares-policy-iteration"></a>Part 4 — Least-squares policy iteration

1. Collect a batch of transitions from states drawn uniformly over the state space, with uniformly random actions, by setting `env.unwrapped.state` before each step. This is a generative model, not the agent's own experience.
2. Implement LSPI: with 4 tilings of $`8\times8`$ tiles per action and $`\gamma=0.99`$, evaluate the current greedy policy's action values with LSTDQ on the batch, $`\mathbf w=\bigl(\sum_i\boldsymbol\phi_i(\boldsymbol\phi_i-\gamma\boldsymbol\phi'_i)^\top+\varepsilon\mathbf I\bigr)^{-1}\sum_i r_i\boldsymbol\phi_i`$, where $`\boldsymbol\phi'_i`$ are the features of the next state and the action the current policy chooses there (zero if the step ended the episode); make the policy greedy; and repeat until the weights stop changing.
3. With 2,000, 10,000, and 50,000 transitions, evaluate the greedy policy on 20 episodes from Gymnasium's usual starts. How often does it reach the goal, and how fast?

## <a id="expected-results"></a>Expected results

```text
=== Part 1: tile coder (8 tilings of 8 x 8 tiles) ===
  shared tiles: nearby states 7 of 8, distant states 0 of 8

=== Part 2: SARSA(lambda) with replacing traces, 8 tilings, alpha = 0.5/8, 5 runs of 200 episodes ===
  lambda = 0.00: steps per episode, episodes 1-20    420, 21-100   159, 101-200   136
  lambda = 0.50: steps per episode, episodes 1-20    395, 21-100   147, 101-200   126
  lambda = 0.90: steps per episode, episodes 1-20    327, 21-100   121, 101-200   108
  lambda = 0.99: steps per episode, episodes 1-20    297, 21-100   147, 101-200   140

=== Part 3: replacing traces versus true online SARSA(lambda), lambda = 0.9, episodes 1-100 ===
  alpha = 0.5/8: average steps per episode, replacing traces    163, true online    156
  alpha = 1.0/8: average steps per episode, replacing traces    153, true online    144
  alpha = 1.5/8: average steps per episode, replacing traces    160, true online    157

=== Part 4: LSPI with 4 tilings, from uniformly sampled transitions, gamma = 0.99 ===
   2,000 transitions: episodes reaching the goal   0.0%, steps when they do   nan, LSPI iterations 15.0
  10,000 transitions: episodes reaching the goal 100.0%, steps when they do   162, LSPI iterations  7.3
  50,000 transitions: episodes reaching the goal  98.3%, steps when they do   124, LSPI iterations  6.0
```

Things to notice:

- **Part 1.** Two states 0.01 apart in position and 0.001 in velocity share 7 of their 8 tiles, so an update at one moves the other's value almost as much; states far apart share none. The tile width sets how far generalization reaches, and the number of tilings sets how finely values can vary.
- **Part 2.** Traces help: $`\lambda=0.9`$ learns fastest and ends best, about 108 steps per episode against 136 for one-step SARSA. The first episodes are long for every $`\lambda`$, because the car must discover the goal by optimism alone, and traces shorten them by pushing down the values of the whole path of a failed attempt, not only its last step. With $`\lambda=0.99`$, the returns are nearly Monte Carlo, and the updates become noisy enough to slow learning after the first episodes, the same U-shape in $`\lambda`$ as on the random walk of chapter 8.
- **Part 3.** True online SARSA(λ) is slightly better than replacing traces at every step size tried here, and both stay stable up to $`\alpha=1.5/8`$. On mountain car, with binary features and few revisits of the same tiles within a short window, replacing traces are already close to the online λ-return; the true online algorithm matters more with non-binary features, where replacing traces are not defined, and at large step sizes.
- **Part 4.** LSPI solves the task offline, without ever acting in the environment, in six or seven iterations of policy evaluation and improvement, provided the batch covers the state space: with 10,000 uniformly sampled transitions every test episode reaches the goal, and with 50,000 the policy is faster, about 124 steps, comparable to what SARSA(λ) reaches after 200 episodes of online learning. With 2,000 transitions, about two per feature, the least-squares solution is too poorly determined and the greedy policy never reaches the goal. The data were drawn uniformly over the states, which a real agent could not do; learning from data the agent collected itself, with poor coverage of the states that matter, is the problem of offline reinforcement learning (chapter 26).

## <a id="going-further"></a>Going further

1. **Feature resolution.** Rerun part 2 with 4, 8, and 16 tilings, scaling $`\alpha`$ by the number of tilings, and with $`4\times4`$, $`8\times8`$, and $`16\times16`$ tiles. Which matters more, the number of tilings or the tile size?
2. **The 200-step limit.** Use Gymnasium's default limit of 200 steps. Does the agent still learn, and what happens to the early episodes? Then treat truncation as termination, and measure the damage.
3. **Fourier features.** Replace the tile coder by a Fourier basis of order 3 to 7 over the scaled state, with per-feature step sizes $`\alpha/\|\mathbf c\|`$, and compare with the tile coder at its best step size.
4. **LSPI from the agent's own data.** Collect the batch by running the greedy policy of a partly trained SARSA(λ) agent with ε-greedy exploration instead of sampling states uniformly. How many transitions does LSPI need now, and why?
5. **Continuous actions.** Try `MountainCarContinuous-v0`, which rewards reaching the goal and penalizes the squared force. Discretize the force into 3, 5, and 9 levels and compare. The policy-gradient methods of chapter 13 handle continuous actions directly.

---

[← Lab 4. Planning with Learned and Given Models](lab-04-planning-with-learned-and-given-models.md) · [Lab 6. Policy Gradient and Actor-Critic Methods on CartPole →](lab-06-policy-gradient-and-actor-critic-methods-on-cartpole.md)
