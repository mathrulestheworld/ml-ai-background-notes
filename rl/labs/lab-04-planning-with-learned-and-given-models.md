[Background Notes](../../README.md) › [Reinforcement Learning](../README.md)

# Lab 4. Planning with Learned and Given Models

[← Lab 3. Tabular Control with Monte Carlo and TD Methods](lab-03-tabular-control-with-monte-carlo-and-td-methods.md) · [Lab 5. Tile Coding and Linear Control on Mountain Car →](lab-05-tile-coding-and-linear-control-on-mountain-car.md)

## <a id="overview"></a>Overview

This lab puts [chapter 10](../10-planning-and-learning-with-tabular-models.md) to work. You will add planning to a learning agent and measure how much real experience it saves and how much computation it costs, plan with a model learned in a stochastic world, use a simulator for decision-time planning with rollouts and Monte Carlo tree search, and run real-time dynamic programming on a map too large to sweep comfortably.

- **Environments:** `Taxi-v3`; the slippery $`8\times8`$ `FrozenLake-v1`; `Blackjack-v1` with Sutton and Barto's rules; and a random $`40\times40`$ FrozenLake made with Gymnasium's `generate_random_map`.
- **Prerequisites:** chapters 7 and 10, and [Lab 1](lab-01-dynamic-programming-on-frozenlake-and-taxi.md) for reading the model `env.unwrapped.P`; NumPy and Gymnasium.
- **Reference solution:** [lab04_planning.py](code/lab04_planning.py), about a minute and a half on one core. Try each part yourself before reading it.

## <a id="part-1-dyna-and-prioritized-sweeping-on-taxi"></a>Part 1 — Dyna and prioritized sweeping on Taxi

Taxi is deterministic, so a table that stores the last outcome of each state–action pair is an exact model, and a step size of 1 is safe.

1. Write one agent with three planning options: none (Q-learning), Dyna-Q with $`n`$ planning updates per real step on pairs drawn uniformly from the model, and prioritized sweeping with at most $`n`$ updates per step from a priority queue of pairs ordered by the size of their pending update, with their predecessors queued after each update. Use ε-greedy behavior with $`\varepsilon=0.1`$, random tie-breaking, and $`\gamma=0.99`$.
2. Measure performance without noise: from the deterministic model, compute the return of the greedy policy from each of the 300 possible initial states of Taxi, with Gymnasium's 200-step limit, and average. Compute the same for the optimal policy from value iteration.
3. Report this average after 2,000, 5,000, 10,000, 20,000, and 40,000 real steps for Q-learning, Dyna-Q with $`n=10`$ and 50, and prioritized sweeping with $`n=10`$, together with the total number of updates and the running time. Which method is most economical in real experience, and which in computation?

## <a id="part-2-planning-with-a-learned-stochastic-model"></a>Part 2 — Planning with a learned stochastic model

On the slippery $`8\times8`$ lake, the last outcome of a pair is not a model of it.

1. Keep counts of the observed outcomes of each pair. After each real step, make 10 planning updates on pairs drawn from those seen, each an expected update under the empirical distribution of outcomes; since the model already averages, use a step size of 1 in planning.
2. Compare with Q-learning ($`\alpha=0.1`$) over 20,000 real steps, measuring the true value of the start state under each greedy policy, computed exactly from the true model, every 5,000 steps.

## <a id="part-3-decision-time-planning-in-blackjack"></a>Part 3 — Decision-time planning in blackjack

1. Implement the rollout policy of chapter 10 with the base policy that sticks on 20 or 21, and $`m=100`$ simulated hands per action.
2. Implement UCT for one decision: the tree's nodes are the player's hands (sum, usable ace) with the dealer's card fixed, the actions are stick and hit, sticking ends the simulation with a sampled dealer outcome, and hitting leads to a chance node, a random card. Recommend the most visited action at the root.
3. The decision depends only on the state (sum, usable ace, dealer card), so run each search once for each of the 200 decision states and reuse its answer; this is the same as fixing the search's random seed per state, and it makes the evaluation cheap. Compare the base policy, rollout, and UCT with 100, 1,000, and 10,000 simulations on the same 20,000 deals, and count each policy's decisions that differ from the optimal policy of chapter 5.

## <a id="part-4-rtdp-on-a-large-lake"></a>Part 4 — RTDP on a large lake

1. Generate a $`40\times40`$ lake with `generate_random_map(size=40, p=0.9, seed=3)`, and move the goal to row 12, column 12, so that much of the map lies beyond it.
2. With $`\gamma=0.99`$ and reward 1 at the goal, use $`\gamma^{d(s)-1}`$ as an optimistic bound on each state's value, where $`d(s)`$ is the Manhattan distance to the goal: no policy can reach the goal in fewer than $`d(s)`$ steps.
3. Starting from this bound, count the updates that value iteration and RTDP need before the start state's value is within 1% of optimal, and the number of states RTDP ever updates.

## <a id="expected-results"></a>Expected results

```text
=== Part 1: Taxi-v3, average return of the greedy policy over all 300 initial states (optimal 7.93) ===
  real steps:                      2,000   5,000  10,000  20,000  40,000   updates   time
  Q-learning                      -286.0  -251.3  -197.2   -74.4     7.4     40,000   0.9 s
  Dyna-Q, n = 10                  -291.5   -84.9     7.9     7.9     7.9    440,000   2.2 s
  Dyna-Q, n = 50                  -239.1     1.5     6.0     7.7     7.9  2,040,000   5.0 s
  prioritized sweeping, n = 10    -362.0     2.7     6.5     7.8     7.8     11,956   1.1 s

=== Part 2: FrozenLake 8x8 (slippery), gamma = 0.99: value of the start state under the greedy policy ===
  optimal: 0.415
  Q-learning               after 5,000 / 10,000 / 15,000 / 20,000 real steps: 0.000 / 0.000 / 0.003 / 0.003
  Dyna, expected updates   after 5,000 / 10,000 / 15,000 / 20,000 real steps: 0.000 / 0.056 / 0.242 / 0.360

=== Part 3: blackjack, one search per decision state, returns on the same 20,000 deals ===
  base policy (stick on 20 or 21):   -0.3471
  rollout, m = 100                  -0.0702; decisions differing from optimal:  49 of 200
  UCT, 100 simulations              -0.0645; decisions differing from optimal:  28 of 200
  UCT, 1,000 simulations            -0.0563; decisions differing from optimal:  11 of 200
  UCT, 10,000 simulations           -0.0478; decisions differing from optimal:   2 of 200
  optimal policy:                    -0.0484

=== Part 4: a random 40x40 FrozenLake with the goal at (12, 12) (1439 non-terminal states), gamma = 0.99 ===
  optimal value of the start state 0.4005; from the bound gamma^(distance - 1), start value within 1%:
    value iteration: 91 sweeps, 130,949 updates
    RTDP:            497 trials, 67,122 updates, 1287 states updated (89%)
```

Things to notice:

- **Part 1.** Planning multiplies the value of experience. Dyna-Q with 10 planning steps reaches the optimal policy's average return by 10,000 real steps, while Q-learning needs about 40,000 to come close. Prioritized sweeping is the best of all at 5,000 steps, with a tiny fraction of the computation, 12,000 updates in all against 440,000 and 2 million: most of Dyna-Q's random updates change nothing. More planning is not uniformly better: with 50 planning steps, the greedy policy is better at 5,000 steps but worse at 10,000 than with 10, and it varies more between runs. Planning propagates what the model knows; it cannot supply experience of states the agent has never visited, and a greedy policy evaluated from all 300 initial states is penalized for any route that passes through such a state, where all the values are still 0. Prioritized sweeping also levels off slightly below the optimum in some runs (exercise 1 below). Note the running times: the most economical method in experience is not the cheapest in computation.
- **Part 2.** On the stochastic lake, where the only reward is at the far corner, Q-learning with a small step size barely starts to learn in 20,000 steps: a success is rare, and each one moves one value by 10% of its error. Planning with expected updates under the learned distribution propagates each success through the whole model and reaches 0.36 against an optimal 0.415. The model is right because it keeps counts; a last-outcome model would have been wrong in exactly the way exercise 10.3 of the chapter shows.
- **Part 3.** Rollout turns a bad base policy into a good one, but it stops at one step of policy improvement: all 49 of its decisions that differ from the optimal ones are sticks on hands where the optimal play is to hit, because it evaluates hitting by continuing with the base policy. UCT improves its continuation within the tree as it searches, and with 10,000 simulations per decision it agrees with the optimal policy in 198 of the 200 states. Its return, $`-0.0478`$, is not better than optimal; the difference from $`-0.0484`$ is within the noise of 20,000 hands, even with the deals shared.
- **Part 4.** RTDP reaches the same accuracy at the start state with half the updates of value iteration, but it still updates 89% of the non-terminal states, unlike the gridworld of chapter 10, where it touched 9%. On a slippery lake every move goes sideways a third of the time, so even the best trajectories from the start spread across the map, and trials that end in holes wander before they do. RTDP's savings depend on how concentrated the relevant states are, which is a property of the problem, not of the algorithm.

## <a id="going-further"></a>Going further

1. **Where prioritized sweeping falls short.** Find the initial states from which prioritized sweeping's greedy policy in part 1 is not optimal after 40,000 steps. Are their routes through pairs it never tried, or through values it never updated? What changes if every real transition is also applied as a direct update?
2. **A changing Taxi.** Wrap Taxi so that after 20,000 steps one of the four pick-up locations moves. Compare Dyna-Q and a Dyna-Q+ agent with an exploration bonus in planning, as in the blocking and shortcut mazes.
3. **Sample-based planning in part 2.** Replace the expected updates with sample updates from the counts, at the same number of computations: $`b=3`$ possible outcomes per pair means three sample updates cost about as much as one expected update. Which is better per unit of computation here, and why is the answer different from the large-$`b`$ regime of the chapter?
4. **UCT without the state cache.** Run UCT afresh at every decision on 2,000 deals with 1,000 simulations, and compare its return with the cached version. How much does the randomness of the search itself cost?
5. **Labeled RTDP.** Add a convergence test to RTDP that marks a state as solved when its residual and those of all states reachable under its greedy action are below a tolerance, and stop trials at solved states. How many updates does it save in part 4?

---

[← Lab 3. Tabular Control with Monte Carlo and TD Methods](lab-03-tabular-control-with-monte-carlo-and-td-methods.md) · [Lab 5. Tile Coding and Linear Control on Mountain Car →](lab-05-tile-coding-and-linear-control-on-mountain-car.md)
