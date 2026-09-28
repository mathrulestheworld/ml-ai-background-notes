[Background Notes](../../README.md) › [Reinforcement Learning](../README.md)

# Lab 3. Tabular Control with Monte Carlo and TD Methods

[← Lab 2. Bandit Algorithms in Practice](lab-02-bandit-algorithms-in-practice.md) · [Lab 4. Planning with Learned and Given Models →](lab-04-planning-with-learned-and-given-models.md)

## <a id="overview"></a>Overview

This lab turns chapter 5, chapter 6, and chapter 7 into a small library of tabular control agents and runs it on four Gymnasium environments whose models are known, so that every learned policy can be checked against the optimal one. Along the way you will see the cliff-walking trade-off reappear in a stochastic environment, double Q-learning underestimate, a SARSA policy that gets worse with more training, and a step size that finds a good policy while getting the values badly wrong.

- **Environments:** `Blackjack-v1` with Sutton and Barto's rules (`sab=True`); `CliffWalking-v1` and its stochastic variant `CliffWalkingSlippery-v1`; `Taxi-v3`; and the slippery $`4\times4`$ `FrozenLake-v1`.
- **Prerequisites:** chapters 5–7, and Lab 1, whose value iteration on `env.unwrapped.P` supplies the optimal policies; NumPy and Gymnasium.
- **Reference solution:** [lab03_tabular_control.py](code/lab03_tabular_control.py), about four minutes on one core. Try each part yourself before reading it.

## <a id="part-1-a-tabular-control-library"></a>Part 1 — A tabular control library

1. Write an agent class that holds a table $`Q[s,a]`$, acts ε-greedily with random tie-breaking, and takes a step size that is either a constant or a function of the number of updates of the pair being updated, such as $`\alpha_n=n^{-0.8}`$.
2. Implement SARSA, Expected SARSA, Q-learning, and double Q-learning as subclasses that differ only in the value of the next state used in the target; double Q-learning also needs its own update, with a second table and a coin flip. In Expected SARSA, split the greedy probability among tied actions, so that the expectation matches the behavior.
3. Implement on-policy first-visit Monte Carlo control with ε-greedy behavior and sample-average step sizes, $`\alpha_n=1/n`$, updating at the end of each episode.
4. Write a training loop that bootstraps unless the episode `terminated` and uses `truncated` only to end the episode, as in chapter 7. Write an evaluation function that runs a deterministic policy on a fixed list of seeds, so that different policies are compared on the same episodes.

## <a id="part-2-blackjack"></a>Part 2 — Blackjack

Gymnasium's blackjack observation is the triple (player's sum, dealer's showing card, usable ace), and the actions are stick (0) and hit (1). Unlike the version in chapter 5, the player also decides with sums below 12, where hitting is always right, since it cannot bust.

1. Train Monte Carlo control ($`\varepsilon=0.1`$) and Q-learning ($`\varepsilon=0.1`$, $`\alpha_n=n^{-0.8}`$, $`\gamma=1`$) for 300,000 episodes each.
2. Write the optimal policy found by dynamic programming in exercise 5.6 as a function: hit below 12, and otherwise stick from the thresholds printed there, which depend on the dealer's card and on whether the ace is usable.
3. For each learned greedy policy, list the decision states (sums 12–21, ten dealer cards, usable ace or not: 200 states) where it differs from the optimal policy, and estimate its return per hand on 200,000 hands. Which kinds of states do the two methods get wrong, and why?

## <a id="part-3-cliff-walking-deterministic-and-slippery"></a>Part 3 — Cliff walking, deterministic and slippery

1. Reproduce chapter 7's comparison on `CliffWalking-v1` with all four TD methods ($`\varepsilon=0.1`$, $`\alpha=0.5`$, 500 episodes, 10 runs): the online return in episodes 401–500 and the greedy route. Where does double Q-learning's route go, and why?
2. In `CliffWalkingSlippery-v1`, each move goes in the intended direction or in one of the two perpendicular directions, with probability 1/3 each, so a step along the edge falls off the cliff one time in three. Compute the optimal action values by value iteration on `env.unwrapped.P` with $`\gamma=1`$; this is a stochastic shortest-path problem, since every step costs and the goal can be reached from everywhere. Also compute the values of the best ε-greedy policy, by replacing the maximum in the backup with the value of the ε-greedy policy, $`(1-\varepsilon)\max_aQ(s',a)+\varepsilon\,\mathrm{mean}_aQ(s',a)`$.
3. Train the four methods for 3,000 episodes with $`\alpha=0.1`$. Compare each greedy policy's return with the optimal policy's, and the online returns with each other.

## <a id="part-4-learning-curves-and-step-sizes"></a>Part 4 — Learning curves and step sizes

1. On `Taxi-v3` ($`\gamma=0.99`$, $`\varepsilon=0.1`$, $`\alpha=0.5`$), evaluate each method's greedy policy on the same 100 episodes every 250 training episodes, for 2,000 episodes and three runs, and compare with the optimal policy from value iteration.
2. On the slippery $`4\times4`$ FrozenLake with $`\gamma=0.99`$, train Q-learning for 10,000 episodes with step sizes 0.1, 0.01, $`1/n`$, and $`1/n^{0.6}`$, where $`n`$ counts the updates of each pair. Compare the greedy policies' success rates with the optimal policy's, and the learned value of the start state with the true one.

## <a id="expected-results"></a>Expected results

```text
=== Part 2: blackjack (Sutton and Barto's rules), 300,000 training episodes ===
  optimal (chapter 5)  return per hand -0.0440; decisions differing from optimal:   0 of 200
  Monte Carlo control  return per hand -0.0499; decisions differing from optimal:  14 of 200
    at (sum, dealer, usable ace): (12,2,0), (12,3,0), (12,3,1), (13,4,1), (13,5,0), (14,5,1), (15,3,1), (15,4,1), (16,5,1), (16,8,0), (16,9,0), (18,1,1), (18,8,1), (18,10,1)
  Q-learning           return per hand -0.0449; decisions differing from optimal:   6 of 200
    at (sum, dealer, usable ace): (12,4,0), (12,5,0), (13,3,0), (18,2,1), (18,4,1), (18,5,1)
  (standard error of each return: about 0.0021)

=== Part 3a: CliffWalking-v1, epsilon = 0.1, alpha = 0.5, 500 episodes, 10 runs ===
  SARSA              online return, episodes 401-500:  -26.2; greedy route 17 steps, highest row 0
  Expected SARSA     online return, episodes 401-500:  -21.0; greedy route 15 steps, highest row 1
  Q-learning         online return, episodes 401-500:  -49.5; greedy route 13 steps, highest row 2
  double Q-learning  online return, episodes 401-500:  -24.8; greedy route 17 steps, highest row 0

=== Part 3b: CliffWalkingSlippery-v1 (1/3 intended, 1/3 each perpendicular), alpha = 0.1, 3,000 episodes ===
  optimal (value iteration)  expected return from the start:   -64.7; simulated:   -65.6
  best epsilon-greedy policy (eps 0.1): its greedy part's return:   -77.9
  optimal policy, rows 0-2:         ^>>>>>>>>>>>  ^>>>>>>>>>>>  ^^^^^^^^^^^>   start: <
  best eps-greedy policy, rows 0-2: ^>>>>>>>>>>>  ^^^^^^^^^^>>  ^^^^^^^^^^^>   start: <
  SARSA                      greedy return:   -78.8; online return in the last 500 episodes:  -122.6
  Expected SARSA             greedy return:   -78.8; online return in the last 500 episodes:  -117.7
  Q-learning                 greedy return:   -65.6; online return in the last 500 episodes:  -146.8
  double Q-learning          greedy return:   -95.3; online return in the last 500 episodes:  -161.2

=== Part 4a: Taxi-v3, epsilon = 0.1, alpha = 0.5, return of the greedy policy every 250 episodes (3 runs) ===
  optimal policy on the 100 test episodes: 8.05
  SARSA               -211.5  -115.9   -87.7   -43.3   -39.8   -34.7    -9.2   -56.5
  Expected SARSA      -214.1   -59.4   -36.5   -22.4   -19.6     3.9     5.3     6.0
  Q-learning          -183.7   -58.8   -20.5   -11.6     6.6     8.0     8.0     8.0
  double Q-learning   -576.8  -260.8  -141.5   -44.0   -13.7    -7.5    -0.2     1.2
  (episodes)             250     500     750    1000    1250    1500    1750    2000

=== Part 4b: FrozenLake 4x4 (slippery), gamma = 0.99, Q-learning step sizes, 10,000 episodes (3 runs) ===
  optimal stationary policy (gamma 0.99): success rate within 100 steps 0.739; value of the start state 0.542
  alpha = 0.1      greedy success rate 0.740; error in the start state's value 0.015
  alpha = 0.01     greedy success rate 0.129; error in the start state's value 0.440
  alpha = 1/n      greedy success rate 0.724; error in the start state's value 0.491
  alpha = 1/n^0.6  greedy success rate 0.740; error in the start state's value 0.010
```

Things to notice:

- **Part 2.** Both learned policies are close to optimal, and the decisions they get wrong cost little, because they are mostly decisions where hitting and sticking have nearly equal values. Q-learning's six errors are all borderline: a hard 12 or 13 against a small card, and a soft 18 against a small card. Monte Carlo control's fourteen errors show two further effects. Twelve of them stick where the optimal policy hits: on-policy Monte Carlo learns the values of the ε-greedy policy, which charges every later decision for its 10% of random actions, so hitting, which leads to more decisions, looks worse than it is. And nine of them are soft hands, which are rarer, so their sample averages are noisier. Every policy is evaluated on the same 200,000 deals, so most of the noise in the three returns is shared, and the differences between them are measured more precisely than the standard error of each suggests.
- **Part 3a.** Gymnasium's cliff reproduces chapter 7. Double Q-learning, perhaps surprisingly, takes SARSA's safe route rather than Q-learning's optimal one. Early on, the two tables have experienced different falls. A table that has never seen the fall from an edge cell still values the step down at 0, its highest estimate there, since every action it has tried cost $`-1`$; so it selects the fall as the best action, and the other table, which has seen the fall, evaluates it at about $`-100`$. This is the underestimation of the double estimator from Appendix B of chapter 7, made worse by the optimistic initial values. The underestimates make the edge look dangerous, the greedy route moves away from it, and exploration then visits the edge too rarely to correct them.
- **Part 3b.** On slippery ice, even the optimal policy keeps away from the cliff: it walks along the second row, and at the start it pushes left into the wall, which never falls and slips upward one time in three, the same trick as the first move on FrozenLake in Lab 1. Q-learning's greedy policy reaches the optimal return. SARSA's and Expected SARSA's greedy policies do worse, about $`-78`$ against $`-66`$, because they learn the best ε-greedy policy, which walks along the top row: when 10% of its moves are random on top of the slipping, it pays to stay further away. While exploring, the ranking reverses, and SARSA and Expected SARSA do better online than Q-learning, as in the deterministic cliff. Double Q-learning is the worst on both counts: half the updates per table and underestimation along the edge.
- **Part 4a.** All the greedy policies begin by looping until the 200-step limit, and the worst early returns, far below $`-200`$, come from policies that repeat an illegal pick-up or drop-off, at $`-10`$ a step. Q-learning reaches the optimal return by 1,500 episodes. Expected SARSA is slower but steady. SARSA's greedy policy gets worse between 1,750 and 2,000 episodes. With $`\alpha=0.5`$, every update moves an estimate halfway toward a target that depends on the randomly chosen next action, and an exploratory illegal action at the next step moves it by 5 or more; the greedy policy built on these jumpy estimates occasionally loops. In Taxi, which is deterministic, Expected SARSA's and Q-learning's targets have no such noise. Double Q-learning starts slowest, since each table receives half the updates.
- **Part 4b.** This is the step-size lesson of chapter 7 on a real environment. With $`\alpha=0.1`$ the greedy policy is optimal and the value of the start state is accurate to 0.015. With $`\alpha=0.01`$ learning is too slow. With $`1/n`$ the greedy policy is nearly optimal, but the value estimate is still far too low after 10,000 episodes: the policy depends only on the ranking of the actions in each state, which can be right long before the values are. The polynomial step size $`1/n^{0.6}`$ gets both right.

## <a id="going-further"></a>Going further

1. **Stabilizing SARSA.** Rerun part 4a with SARSA at $`\alpha=0.1`$ and with Expected SARSA and Q-learning at $`\alpha=1`$. Which settings give a greedy policy that improves steadily, and why is $`\alpha=1`$ safe for two of the methods but not for SARSA?
2. **Exploring starts.** Gymnasium's blackjack keeps the hands in `env.unwrapped.player` and `env.unwrapped.dealer`. Overwrite them after `reset` to start episodes from random states, implement Monte Carlo ES from chapter 5, and compare its policy with ε-greedy Monte Carlo control's.
3. **The truncation bug.** Treat `truncated` as `terminated` in the training loop and rerun part 4b with a time limit of 20 steps, `gym.make(..., max_episode_steps=20)`. How do the learned values and the greedy policy change? Which states are affected most?
4. **GLIE on slippery ice.** Decay ε as $`\varepsilon_k=0.1\cdot100/(100+k)`$ in part 3b and train for 20,000 episodes. Does SARSA's greedy policy approach the optimal one? Compare with exercise 7.7 of chapter 7.
5. **Maximization bias as an environment.** Write chapter 7's maximization-bias example as a custom Gymnasium environment (a subclass of `gym.Env` with `Discrete` spaces), run your Q-learning and double Q-learning on it, and reproduce the figure of chapter 7.

---

[← Lab 2. Bandit Algorithms in Practice](lab-02-bandit-algorithms-in-practice.md) · [Lab 4. Planning with Learned and Given Models →](lab-04-planning-with-learned-and-given-models.md)
