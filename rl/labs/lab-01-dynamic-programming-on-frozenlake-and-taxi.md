[ML Mastery Notes](../../README.md) › [Reinforcement Learning](../README.md)

# Lab 1. Dynamic Programming on FrozenLake and Taxi

[Lab 2. Bandit Algorithms in Practice →](lab-02-bandit-algorithms-in-practice.md)

## <a id="overview"></a>Overview

This lab puts chapter 1 and chapter 2 to work on real Gymnasium environments. You will extract the exact model of an environment, solve it with policy iteration and value iteration, check the answer by simulation, and discover that the discount factor, the time limit, and the objective you actually care about can disagree.

- **Environments:** `FrozenLake-v1` (4×4 and 8×8, slippery), `Taxi-v3`.
- **Prerequisites:** chapters 1–2; NumPy.
- **Reference solution:** [lab01_dp_frozenlake.py](code/lab01_dp_frozenlake.py), which prints the results below in a few seconds. Try each part yourself before reading it.

In FrozenLake the agent crosses a frozen lake from the start S to the goal G without falling into a hole H. The ice is slippery: the agent moves in the intended direction with probability 1/3 and in each of the two perpendicular directions with probability 1/3. Reaching the goal gives reward 1; every other reward is 0, and holes and the goal end the episode. Gymnasium also cuts every episode off after 100 steps.

## <a id="part-1-read-the-model"></a>Part 1 — Read the model

Gymnasium's toy-text environments expose their dynamics: `env.unwrapped.P[s][a]` is a list of tuples `(probability, next_state, reward, terminated)`.

1. Convert it into arrays `P[s, a, s']` and `R[s, a]` (the expected reward), sending terminal transitions to an absorbing state with value zero. The simplest way is to leave their probability out of `P`, so that the rows of `P` for actions that may terminate sum to less than one.
2. Check your arrays: every row of `P` plus its terminal mass must sum to one, and `R[s, a]` must be the probability of stepping onto G.
3. Print `env.unwrapped.P[0][0]` and explain the three entries in terms of the slippery dynamics.

## <a id="part-2-policy-iteration-and-value-iteration"></a>Part 2 — Policy iteration and value iteration

1. Implement exact policy evaluation by a linear solve, policy iteration with greedy improvement that keeps the current action on ties, and value iteration with the stopping rule of chapter 2.
2. Solve both maps for $`\gamma=0.9`$, $`0.99`$, and $`0.9999`$. Confirm that the two algorithms give policies with the same value, and compare the number of policy-iteration evaluations with the number of value-iteration sweeps.
3. Print the policy on the map. Explain the famous first move on the 4×4 map: why does the optimal agent start by pushing **left**, into the wall?
4. Compute the probability of ever reaching the goal under each policy, by evaluating it with $`\gamma\to1`$. Why is $`v(\text{start})`$ with $`\gamma=0.9`$ so much smaller than this probability?

## <a id="part-3-discounting-versus-time-limits"></a>Part 3 — Discounting versus time limits

The quantity a user of the environment sees is the success rate within Gymnasium's 100-step limit.

1. Compute it exactly for each of your policies by running the policy's Bellman backup for 100 steps from zero (backward induction with a fixed policy), and check one value by simulating 2,000 episodes in the real environment.
2. Compute the best **time-dependent** policy for the 100-step limit by backward induction with the optimality backup and $`\gamma=1`$.
3. On the 8×8 map, which discount factor gives the best stationary policy for the time-limited task? Explain why the policy that is optimal as $`\gamma\to1`$ is the worst of the three here, and relate this to finite horizons and time limits.

## <a id="part-4-many-maps-and-taxi"></a>Part 4 — Many maps, and Taxi

1. Generate 200 random 8×8 maps with `generate_random_map(size=8, p=0.8)` (each tile frozen with probability 0.8; the generator guarantees a path) and record the optimal probability of ever reaching the goal. How often is success nearly certain, and how often is it below one half?
2. Solve `Taxi-v3` (500 states, 6 actions, reward $`-1`$ per step, $`+20`$ for a correct drop-off, $`-10`$ for an illegal pickup or drop-off) with both algorithms and $`\gamma=0.99`$, and measure the average undiscounted return of the optimal policy over 1,000 episodes. Why do both algorithms need so few iterations here, compared with FrozenLake?

## <a id="expected-results"></a>Expected results

```text
=== FrozenLake 4x4 (slippery): 16 states, time limit 100 steps ===
gamma 0.9: PI  6 evaluations, VI  167 sweeps, same value: True; P(goal) eventually 0.7805, within 100 steps 0.7298
gamma 0.99: PI  7 evaluations, VI  724 sweeps, same value: True; P(goal) eventually 0.8235, within 100 steps 0.7402
gamma 0.9999: PI  8 evaluations, VI 1204 sweeps, same value: True; P(goal) eventually 0.8235, within 100 steps 0.7402
best time-dependent policy for the 100-step limit (backward induction): P(goal) = 0.7442
check by simulation, gamma = 0.9999 policy, 2,000 episodes in Gymnasium: 0.7410
that policy (H hole, G goal):
  ← ↑ ↑ ↑
  ← H ← H
  ↑ ↓ ← H
  H → ↓ G

=== FrozenLake 8x8 (slippery): 64 states, time limit 100 steps ===
gamma 0.9: PI 10 evaluations, VI  180 sweeps, same value: True; P(goal) eventually 0.7488, within 100 steps 0.6011
gamma 0.99: PI 11 evaluations, VI  830 sweeps, same value: True; P(goal) eventually 0.8938, within 100 steps 0.6317
gamma 0.9999: PI 12 evaluations, VI 2085 sweeps, same value: True; P(goal) eventually 1.0000, within 100 steps 0.5143
best time-dependent policy for the 100-step limit (backward induction): P(goal) = 0.6407
check by simulation, gamma = 0.9999 policy, 2,000 episodes in Gymnasium: 0.5030
that policy (H hole, G goal):
  ↑ → → → → → → →
  ↑ ↑ ↑ ↑ ↑ ↑ ↑ →
  ← ← ← H → ↑ → →
  ← ← ← ↓ ← H → →
  ← ↑ ← H → ↓ ↑ →
  ← H H ↓ ↑ ← H →
  ← H ↓ ← H ← H →
  ← ↓ ← H ↓ → ↓ G

200 random 8x8 maps: optimal success probability median 0.878, fraction above 0.99: 0.44, below 0.5: 0.34

Taxi-v3: 500 states, 6 actions; VI 19 sweeps, PI 17 evaluations; values agree: True
average undiscounted return of the optimal policy over 1,000 episodes: 7.87
```

Things to notice:

- **The cautious first move.** Pushing left into the wall means the agent either stays put or slips up or down, never toward the hole on its right. With enough patience, slow and safe beats fast and risky, and on the 8×8 map the $`\gamma\to1`$ policy reaches the goal with probability 1.0000 to four decimals: it has a way to avoid the holes almost surely, at the cost of many steps.
- **Discounting changes the objective.** With $`\gamma=0.9`$ the value of the start state is only 0.069 on the 4×4 map, because a success after 25 steps is worth $`0.9^{24}\approx0.08`$. The policy trades a little success probability for speed.
- **The time limit changes it again.** Within 100 steps, the patient $`\gamma=0.9999`$ policy succeeds only 51% of the time on the 8×8 map, while the $`\gamma=0.99`$ policy succeeds 63% of the time and the best time-dependent policy 64%. The discount factor is acting as a knob that trades patience against the deadline the environment imposes but the state does not show.
- **Policy iteration needs few iterations**, 6 to 12 evaluations, while value iteration's sweeps grow with the effective horizon, as in the figure of chapter 2. Taxi is deterministic with short episodes, so value information propagates in about as many sweeps as the longest optimal route.
- **Taxi's optimal return**, about 7.9, is the benchmark a learning agent should approach in chapter 7.

## <a id="going-further"></a>Going further

1. **Add the time to the state.** Build an MDP whose state is (cell, steps remaining) for the 100-step limit, solve it with $`\gamma=1`$, and confirm that its value at the start equals the backward-induction result. How many states does it have, and what does its policy do differently near the deadline?
2. **Modified policy iteration.** Implement it with $`m`$ evaluation sweeps and plot the total number of backups to reach a fixed accuracy against $`m`$ on the 8×8 map with $`\gamma=0.999`$.
3. **Asynchronous updates.** Implement Gauss–Seidel value iteration, and then a version that updates states in order of decreasing Bellman error (a priority queue). Count backups to convergence. This is prioritized sweeping without learning (chapter 10).
4. **The linear program.** Solve the 8×8 map with `scipy.optimize.linprog` in primal and dual form and read the optimal occupancy measure from the dual. Which cells does the optimal agent spend most of its discounted time in?
5. **A harder map.** Make the ice more slippery by editing the model: the intended move with probability 0.2 and each other direction with probability 0.8/3. How does the optimal success probability change, and does the cautious strategy survive?

---

[Lab 2. Bandit Algorithms in Practice →](lab-02-bandit-algorithms-in-practice.md)
