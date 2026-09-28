[Background Notes](../../README.md) › [Reinforcement Learning](../README.md)

# Lab 15. Regret Minimization and Self-Play in Poker

> [!WARNING]
> Work in progress: this part of the notes is still being revised.

[← Lab 14. Imitation and Offline Reinforcement Learning](lab-14-imitation-and-offline-reinforcement-learning.md) · [Lab 16. Reinforcement Learning for a Small Language Model →](lab-16-reinforcement-learning-for-a-small-language-model.md)

## <a id="overview"></a>Overview

This lab puts [chapter 27](../27-multi-agent-rl-and-self-play.md) to work on a poker game small enough to solve exactly and large enough to be interesting. You will write the game, compute exact best responses and hence exact exploitability, and then compare the ways of finding an equilibrium that the chapter describes: counterfactual regret minimization in its full, improved, and sampled forms; self-play dynamics that update the players' current strategies, with and without regularization; and population methods that grow a set of strategies with a best-response oracle.

- **Game:** **Leduc hold'em** ([Southey et al., 2005](https://arxiv.org/abs/1207.1411)), a standard research game. The deck has two jacks, two queens, and two kings; each player antes 1 chip and receives one private card; a betting round follows; one public card is revealed; a second betting round follows; and at the showdown a player whose card pairs the public card wins, and otherwise the higher card wins. In each round player 0 acts first, bets and raises are 2 chips in the first round and 4 in the second, and at most two raises (a bet and a raise) are allowed per round. Since suits never matter, cards can be represented by their ranks, with the composition of the deck entering through the chance probabilities: 24 deals of ranks with positive probability, and 144 information sets per player.
- **Prerequisites:** chapter 27; NumPy and SciPy.
- **Reference solution:** [lab15_poker.py](code/lab15_poker.py), about 3 minutes on one core. Try each part yourself before reading it.

## <a id="part-1-exact-evaluation-cfr-and-cfr"></a>Part 1 — Exact evaluation, CFR, and CFR+

1. **The game tree.** For every deal, build the tree of betting histories, with the actions fold, check or call, and bet or raise, and key each decision node by its information set: the acting player, its private card, the public card if revealed, and the betting histories of both rounds. Store the payoff to player 0 at every terminal node.
2. **Best responses.** For player $`i`$ against a fixed strategy of the other player, collect the histories of each of $`i`$'s information sets with their weights, the probability that chance and the opponent lead there; then, processing the information sets from the deepest to the shallowest, choose at each the action with the highest weighted value, given the choices already made below it. The best response's value against a strategy profile gives the exploitability, the sum of what the two players gain by best-responding. As a check, the uniformly random strategy has an exploitability of 4.747 chips per hand.
3. **CFR and CFR+.** Traverse all deals once per player per iteration, with alternating updates: compute the regret-matching strategies at the start of the pass and keep them fixed throughout it, accumulate each information set's counterfactual regrets, weighted by chance's and the opponent's reach, and apply them at the end of the pass. Accumulate the average strategy weighted by the player's own probability of reaching each information set, computed once per information set, not once per history. For CFR+, floor the cumulative regrets at zero after each pass and weight iteration $`t`$ by $`t`$ in the average. Run 1,000 iterations, measuring the exploitability of the average and of the current strategies.

## <a id="part-2-monte-carlo-cfr"></a>Part 2 — Monte Carlo CFR

Implement **external-sampling MCCFR**: in each iteration, for each player in turn, sample a deal, explore all of that player's actions, sample one action at each of the opponent's decisions from its current strategy, and update regrets with the sampled values; accumulate the opponent's average strategy at its sampled decisions. Run 100,000 iterations, counting the nodes touched, and compare its exploitability with CFR's and CFR+'s at equal numbers of nodes.

## <a id="part-3-self-play-dynamics-without-averaging"></a>Part 3 — Self-play dynamics without averaging

Compute each player's exact action values at its information sets against the current profile: the expected payoff after each action, given that play reaches the information set, with the histories weighted by chance's and the opponent's reach. Then run four dynamics for 3,000 steps, both players updating simultaneously, and measure the exploitability of the *current* strategies:

1. **Iterated best response**, naive self-play: each player switches to a best response to the other's current strategy.
2. **Mirror descent**: add $`\eta=0.05`$ times the action values to the logits at every information set.
3. **Regularized dynamics** with a uniform magnet: the step $`z\leftarrow(z+\eta\alpha\ln\rho+\eta q)/(1+\eta\alpha)`$ of chapter 27's first code, with $`\alpha=0.2`$.
4. **Moving magnet**: the same, with the magnet reset to the current strategy every 500 steps.

## <a id="part-4-populations-with-a-best-response-oracle"></a>Part 4 — Populations with a best-response oracle

Start each player's population with the uniform strategy and compute the empirical game, the matrix of expected payoffs between the members of the two populations. At each iteration, choose a meta-strategy for each player, a distribution over its population; add to each population a best response to the other player's meta-strategy, played as a single behavioral strategy (weight each member's action probabilities by the meta-probability times the member's own reach of the information set); and extend the empirical game. Compare three meta-solvers over 150 iterations: the latest member (iterated best response again), the uniform distribution (fictitious play), and a Nash equilibrium of the empirical game, found by linear programming (PSRO, here the double oracle). Measure the exploitability of the meta-strategies.

## <a id="expected-results"></a>Expected results

```text
Leduc hold'em: 24 deals of ranks, 288 information sets (144 per player)

=== Part 1: CFR and CFR+ (alternating updates), exploitability in chips per hand ===
  iterations       CFR average   current      CFR+ average   current
          10           1.7772     1.395           1.2209     0.920
          30           0.6218     2.029           0.1635     0.290
         100           0.1914     1.863           0.0268     0.096
         300           0.0710     1.276           0.0047     0.062
        1000           0.0236     1.577           0.0005     0.019
  value of the game to player 0 (CFR+ average): -0.0856
  player 0's first action with each card (probabilities of check, bet):  J 0.93 0.07   Q 0.26 0.74   K 0.25 0.75

=== Part 2: external-sampling Monte Carlo CFR against CFR, by nodes touched ===
  MCCFR iterations   nodes touched   exploitability
             1,000          38,182           1.4146
             3,000         107,485           0.5381
            10,000         364,674           0.3258
            30,000       1,072,038           0.1704
           100,000       3,612,485           0.0682
  CFR   for comparison: 18,613 nodes 1.7772, 56,541 nodes 0.6218, 191,217 nodes 0.1914, 564,398 nodes 0.0710
  CFR+  for comparison: 24,076 nodes 1.2209, 75,520 nodes 0.1635, 263,072 nodes 0.0268, 808,354 nodes 0.0047

=== Part 3: self-play with exact action values: exploitability of the current strategies ===
  step size 0.05; regularization 0.2 toward a uniform magnet, or toward the current strategy every 500 steps
  steps                     10      100      300     1000     3000
  best response        8.067    4.789    5.392    5.392    7.608
  mirror descent       1.871    1.349    2.549    2.692    2.924
  magnet               1.912    0.900    0.249    0.169    0.168
  moving magnet        1.912    0.900    0.249    0.101    0.130

=== Part 4: PSRO with exact best responses: exploitability of the meta-strategy ===
  iterations                 1        5       10       20       40       80      150
  iterated BR          4.747    6.761    5.800    5.739    6.833    5.507    5.507
  fictitious play      4.747    3.970    2.547    1.601    0.892    0.585    0.351
  PSRO (Nash)          4.747    4.836    3.041    1.908    0.799    0.278    0.104
```

Things to notice:

- **Part 1.** Both algorithms solve the game, CFR+ much faster: after 1,000 iterations its average strategy is exploitable by 0.0005 chips per hand, against 0.024 for CFR, and it reaches CFR's 1,000-iteration level in about 100 iterations. The current strategies behave as the chapter predicts: CFR's never settle, staying above 1 chip per hand, while CFR+'s converge too, though more slowly than its average. The value of the game to player 0 is $`-0.0856`$ chips per hand; acting first is a disadvantage, as in Kuhn poker. The solution bets the Q and the K about three times in four as its first action, and the J, the worst card, rarely.
- **Part 1, two easy bugs.** The first version of the reference solution updated the regrets of an information set during the sweep over deals, so that later deals in the same iteration saw a partly updated strategy, and it accumulated the average strategy once per history rather than once per information set, which weights information sets by how many histories happen to be visited. CFR+ still converged, slowly, and plain CFR stalled near 0.5 chips per hand. Both mistakes are natural when an implementation walks one deal at a time, and exact exploitability is what exposes them.
- **Part 2.** In this small game, sampling does not pay: after 3.6 million nodes, external-sampling MCCFR is exploitable by 0.068, while CFR+ reaches 0.005 with 0.8 million and CFR 0.071 with 0.56 million. A full traversal of Leduc hold'em costs only about 1,200 nodes per player, so exact counterfactual values are cheap; sampling wins when a full traversal is impossible, as in the poker games of chapter 27, where each iteration can then cost a single path.
- **Part 3.** Naive self-play jumps between best responses and never becomes less exploitable than 4.8 chips per hand, worse than random play. Unregularized mirror descent drifts away from equilibrium, from 1.35 after 100 steps to 2.9 after 3,000. With a fixed magnet, the current strategies converge, to the regularized equilibrium, exploitable by 0.168: stable but biased. Moving the magnet reduces the exploitability to 0.10 after 1,000 steps, but with these step sizes it does not keep improving, rising to 0.13 at 3,000 steps (and further in longer runs). This simple version applies the one-state update of chapter 27 at every information set with conditional action values; the convergence guarantees of R-NaD and magnetic mirror descent need more care, and the best current strategy here is still 200 times more exploitable than CFR+'s average.
- **Part 4.** Iterated best response cycles, as in chapter 27. Fictitious play improves steadily, to 0.35 after 150 iterations. PSRO with a Nash meta-solver is slower at first, since the equilibrium of a small empirical game puts its weight on a few members, whose best responses are narrow, but it overtakes fictitious play after about 40 iterations and reaches 0.10 after 150, with 150 strategies per player. Each iteration adds one pure strategy, and Leduc hold'em has more than $`10^{40}`$, so the double oracle's finite convergence guarantee is a long way off; what makes PSRO useful in large games is that a good meta-solver needs far fewer oracle calls than the game has strategies.

## <a id="going-further"></a>Going further

1. **Discounting.** Implement linear CFR and discounted CFR ([Brown and Sandholm, 2019](https://arxiv.org/abs/1809.04040)), which discount early regrets and averages, and compare them with CFR+.
2. **Where sampling pays.** Implement outcome-sampling MCCFR, and make the game larger, with more ranks or more raises per round, until a full traversal becomes expensive. Where does MCCFR overtake CFR+ in wall-clock time?
3. **A moving magnet that converges.** Move the magnet only when the regularized dynamics have converged, for instance when the strategies change by less than a threshold, or decrease the step size after each move. Can you drive the current strategies' exploitability toward CFR+'s?
4. **Learned oracles.** Replace PSRO's exact best response with a learner that only plays the game, such as tabular Q-learning or a policy-gradient agent trained against the meta-strategy for a fixed number of hands. How does the error of the oracle change the convergence?
5. **Approximate averages.** Train a small network to imitate CFR+'s average strategy from sampled information sets, as Deep CFR and NFSP do, and measure how much exploitability the approximation costs.

---

[← Lab 14. Imitation and Offline Reinforcement Learning](lab-14-imitation-and-offline-reinforcement-learning.md) · [Lab 16. Reinforcement Learning for a Small Language Model →](lab-16-reinforcement-learning-for-a-small-language-model.md)
