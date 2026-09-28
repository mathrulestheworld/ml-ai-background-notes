[Background Notes](../../README.md) › [Reinforcement Learning](../README.md)

# Lab 13. AlphaZero and Gumbel Search from Scratch

> [!WARNING]
> Work in progress: this part of the notes is still being revised.

[← Lab 12. Model-Based RL with Learned Ensembles](lab-12-model-based-rl-with-learned-ensembles.md) · [Lab 14. Imitation and Offline Reinforcement Learning →](lab-14-imitation-and-offline-reinforcement-learning.md)

## <a id="overview"></a>Overview

This lab puts [chapter 24](../24-planning-with-learned-models.md) to work. You will write AlphaZero from scratch: a policy–value network, the PUCT search it guides, self-play that generates its training data, and the training loop that closes the circle. You will check it against the exact solution of tic-tac-toe, scale it to Connect Four against classical opponents, and then replace PUCT by Gumbel search to test the chapter's claim that Gumbel search keeps improving the policy when the budget is only a couple of simulations per move.

- **Games:** tic-tac-toe, small enough to solve exactly: 4,520 positions with a move to make, whose minimax values a memoized search computes in a second. Connect Four on the standard board of 6 rows and 7 columns, where four in a row horizontally, vertically, or diagonally wins; it is solved, a first-player win, but far too large to solve in a lab.
- **Prerequisites:** chapter 24 and the Monte Carlo tree search of [AI chapter 4](../../ai/04-adversarial-search-and-games.md#monte-carlo-tree-search); PyTorch and NumPy.
- **Reference solution:** [lab13_alphazero.py](code/lab13_alphazero.py), about 17 minutes on two cores, two thirds of it in part 2. Try each part yourself before reading it.

## <a id="part-1-alphazero-on-tic-tac-toe"></a>Part 1 — AlphaZero on tic-tac-toe

1. **The game.** Represent a position from the point of view of the player to move, as two bitboards, the mover's stones and the opponent's, and the number of stones placed. Write `play(state, action)` to return the next position, seen from the other player, and the outcome for the player to move there: $`-1`$ if the move just played completed a line, 0 for a full board, and nothing otherwise. Working from the mover's point of view means that one network and one search serve both players.
2. **The network.** An MLP with two hidden layers of 128 ReLU units maps the two bitboards, as 18 inputs, to 9 policy logits and a value in $`[-1,1]`$ through a tanh. Illegal moves are masked out of the policy when it is used. Evaluating one position at a time with PyTorch costs tens of microseconds of overhead, so copy the weights after every training round into a NumPy version of the network for the search.
3. **PUCT search.** Expand the root with the network, then run $`n`$ simulations. Each descends from the root by $`\arg\max_a Q(s,a)+c\,P(s,a)\sqrt{\sum_bN(s,b)+1}/(1+N(s,a))`$ with $`c=1.25`$ and $`Q=0`$ for unvisited actions, until it reaches a new position, which the network evaluates, or the end of the game, whose outcome is exact, and backs the value up the path with alternating signs. (The $`+1`$ under the square root makes the first simulation follow the prior's most likely move.)
4. **Self-play.** Play games in which both sides search with 100 simulations. At the root, mix Dirichlet noise into the prior, $`P=0.75\,\mathbf p+0.25\,\boldsymbol\eta`$ with $`\boldsymbol\eta\sim\operatorname{Dir}(1)`$ over the legal moves; for the first 4 moves of a game, sample the move in proportion to the visit counts, and afterward play the most visited move. Store each position with the visit distribution and, once the game ends, its outcome $`z`$ from the point of view of the player who moved there.
5. **Training.** After every iteration of 100 games, take 200 Adam steps (step size $`10^{-3}`$, weight decay $`10^{-4}`$, minibatches of 256) on the positions of the last 20 iterations, minimizing $`(z-v)^2-\boldsymbol\pi^\top\ln\mathbf p`$. Train for 40 iterations.
6. **Evaluation.** At iterations 0, 5, 10, 20, and 40, measure two things, for the raw network, which plays its policy's most likely move, and for search with 100 simulations and no noise: the fraction of all 4,520 positions in which the chosen move is optimal, and the number of losses in 100 games against a perfect player, half with each color, who chooses uniformly among its optimal moves.

## <a id="part-2-connect-four"></a>Part 2 — Connect Four

1. Use bitboards with 7 bits per column, the top one always empty, so that a line of four in any direction is found with four shifts and masks, by 1 (vertical), 7 (horizontal), 6, and 8 (diagonals).
2. Train an MLP with two hidden layers of 256 units for 60 iterations of 100 self-play games with 50 simulations per move, sampling moves from the visit counts for the first 8 moves, and 400 Adam steps per iteration. Double every iteration's data with the left–right mirror image of each position and policy, as AlphaGo Zero did with the symmetries of Go.
3. Write three opponents: a **random** player; a **tactical** player that wins at once if it can, otherwise blocks the opponent's immediate win, otherwise avoids moves that let the opponent win at once, and otherwise plays a random move weighted toward the center; and plain **UCT** with 400 simulations, each finished by a uniformly random rollout.
4. Play matches from random two-move openings, each opening once with each color, and score a win 1 and a draw 1/2. For the networks at iterations 0, 10, 20, 40, and 60, measure the raw policy against the tactical player (100 games) and search with 50 simulations against all three opponents (100 games against the tactical player, 50 against the others). Finally, play the last network with 0, 50, 200, and 800 simulations against UCT (50 games each).

## <a id="part-3-gumbel-search-with-few-simulations"></a>Part 3 — Gumbel search with few simulations

1. **Gumbel search.** At the root, draw Gumbel noise $`g(a)`$ for the legal moves and consider the $`m=\min(16,\text{legal moves})`$ moves with the largest $`g(a)+\text{logits}(a)`$. Allocate the simulations by sequential halving, as DeepMind's mctx library does: a precomputed schedule gives, for every simulation, the visit count that the next move to simulate must have, and among the considered moves with that count, simulate the one with the highest $`g(a)+\text{logits}(a)+\sigma(\hat q(a))`$. Here $`\hat q`$ are the **completed** values, the mean values of visited moves and, for unvisited ones, the mixed estimate of the state's value from the network's value and the visited moves' values weighted by the prior; they are rescaled to $`[0,1]`$ over the legal moves, and $`\sigma(\hat q)=(50+\max_bN(b))\cdot0.1\cdot\hat q`$. Below the root, select deterministically $`\arg\max_a\bigl(\pi'(a)-N(a)/(1+\sum_bN(b))\bigr)`$ with $`\pi'=\operatorname{softmax}(\text{logits}+\sigma(\hat q))`$. Act with the best of the most visited considered moves, and train the policy on $`\operatorname{softmax}(\text{logits}+\sigma(\hat q))`$ at the root. The Gumbel noise does the exploring: use no Dirichlet noise and no temperature, and no noise at evaluation.
2. **Training with 2 simulations.** Repeat part 1 with 2 simulations per move, once with PUCT and once with Gumbel search, with 3 seeds each, and evaluate the raw network and its search at the same budget.
3. **Playing with few simulations.** With the final Connect Four network of part 2, play Gumbel search against PUCT, both with the same budget of 2 to 64 simulations, for 100 games per budget.

## <a id="expected-results"></a>Expected results

```text
=== Part 1: AlphaZero on tic-tac-toe (self-play with 100 simulations per move, 100 games per iteration) ===
  fraction of all 4,520 positions in which the move is optimal, and losses in 100 games against a perfect
  player; the raw policy network, and PUCT search with 100 simulations
    iteration    policy: optimal   losses    search: optimal   losses
            0             0.574       76             0.982       42
            5             0.885        3             0.988        5
           10             0.889        3             0.989        0
           20             0.910        0             0.988       10
           40             0.917        7             0.987        8

=== Part 2: AlphaZero on Connect Four (50 simulations per move, 60 iterations of 100 games) ===
  score (win 1, draw 1/2) from random two-move openings, both colors: the raw policy against the tactical
  player, and search with 50 simulations against random, tactical, and UCT with 400 random rollouts
    iteration   policy v tactical   search v random   search v tactical   search v UCT-400
            0              0.00              0.90              0.28              0.00
           10              0.17              1.00              0.54              0.14
           20              0.20              1.00              0.69              0.14
           40              0.37              0.98              0.82              0.47
           60              0.41              1.00              0.84              0.43
  the final network with more search, against UCT with 400 random rollouts (50 games each)
       0 simulations: 0.12
      50 simulations: 0.43
     200 simulations: 0.59
     800 simulations: 0.70

=== Part 3a: training tic-tac-toe with 2 simulations per move: PUCT against Gumbel search, 3 seeds ===
  after 40 iterations of 100 games: optimal-move fraction and losses against the perfect player
                    policy: optimal   losses    search: optimal   losses
  puct   seed 0             0.622      100             0.622      100
  puct   seed 1             0.646       60             0.646       60
  puct   seed 2             0.628       59             0.628       59
  gumbel seed 0             0.814       52             0.865       36
  gumbel seed 1             0.869       26             0.907       30
  gumbel seed 2             0.831       51             0.893       32

=== Part 3b: Gumbel against PUCT search with the same Connect Four network and budget (100 games) ===
    2 simulations each: Gumbel scores 0.54
    4 simulations each: Gumbel scores 0.55
    8 simulations each: Gumbel scores 0.49
   16 simulations each: Gumbel scores 0.49
   32 simulations each: Gumbel scores 0.45
   64 simulations each: Gumbel scores 0.43
```

Things to notice:

- **Part 1, search and learning.** Before any training, search with 100 simulations guided by an untrained network already picks an optimal move in 98% of positions: tic-tac-toe is shallow enough that 100 simulations, with exact outcomes at the end of the game, see most of its tactics. Yet it loses 42 of 100 games to the perfect player, who needs only one mistake. Training raises the raw policy from 57% to 92% optimal moves, and by iteration 5 both the network alone and the search lose only a handful of games. The two measures answer different questions: the fraction of optimal moves weighs all positions equally, including absurd ones, while the games test the positions that good play actually reaches.
- **Part 1, a self-play blind spot.** The losses do not reach zero and stay there. At iteration 40, every loss of the policy and of the search comes from one position: the perfect player, as the first player, opens on the middle of an edge, and the network answers in a far corner, which loses to correct play, while 100 simulations do not find the refutation. Self-play openings favor the center and the corners, so the network has seen few edge openings, and the perfect player, choosing among all its optimal moves, finds them. This is the tic-tac-toe version of the adversarial policies that beat KataGo ([chapter 27](../27-multi-agent-rl-and-self-play.md)). The count of losses is noisy for the same reason: it depends on how often the opponent happens to choose that opening, which is why it rises from 0 at iteration 10 to 10 at iteration 20.
- **Part 2, learning Connect Four.** Six thousand self-play games, about 9 minutes, take the search with 50 simulations from 0.28 to 0.84 against the tactical player and from 0 to 0.43, nearly even, against UCT with 400 rollouts. The raw network is still weak, scoring 0.41 against the tactical player and 0.12 against UCT, and search does much of the work: with the same network, more simulations raise the score against UCT from 0.43 at 50 to 0.70 at 800. This is far from perfect play, which AlphaZero-style programs approach only with residual networks and far more self-play; the curve is what matters here, and it is still rising at iteration 60.
- **Part 3a, training with 2 simulations.** With PUCT, the two simulations always visit the prior's favorite move first, and the most visited move is always the prior's favorite, so the search never changes the move, and the visit-count targets, with half or all of their weight on the prior's favorite, carry almost no information about the values: the three networks end at 62% to 65% optimal moves, barely above the 57% of the untrained network of part 1, and their search adds nothing. With Gumbel search, the second simulation compares two sampled moves by their values, and the training target moves the policy toward the better one, so the networks learn, reaching 81% to 87% optimal moves, and their search improves on them further. The Gumbel agents are still far from the 100-simulation AlphaZero of part 1, and they lose many games, but they learn from a budget at which AlphaZero does not, as Danihelka et al. found in 9×9 Go and Atari.
- **Part 3b, playing with few simulations.** With a well-trained network and no learning involved, the two searches are about even: Gumbel scores 0.54 and 0.55 at 2 and 4 simulations and 0.43 to 0.45 at 32 and 64, differences of the size of the noise of 100 games, about $`\pm0.05`$ per score. Gumbel search's advantage is in the policy improvement it guarantees for training, not in playing strength with a good network.

## <a id="going-further"></a>Going further

1. **Curing the blind spot.** Start a fraction of the self-play games from a random opening move, or from random positions, and check whether the edge opening stops losing. How does this trade off against the quality of play in common positions?
2. **What noise and temperature do.** Retrain part 1 without Dirichlet noise, and with the temperature applied only to the first move. How quickly does the network learn, and how many blind spots does it keep?
3. **A convolutional network.** The reference solution includes a network with two $`3\times3`$ convolutions (`channels=32`). Train it on Connect Four and compare its strength per game and per second of computation with the MLP. Which matters more on two CPU cores?
4. **Gumbel search in the larger game.** Train Connect Four with 4 or 8 simulations per move, with PUCT and with Gumbel search. Does part 3a's advantage survive in a game with longer episodes and a more useful network value?
5. **Measuring against the truth.** Connect Four has been solved, and public solvers return the exact value of any position in milliseconds. Use one to measure the fraction of optimal moves along your agent's games, and compare it with the score against UCT.

---

[← Lab 12. Model-Based RL with Learned Ensembles](lab-12-model-based-rl-with-learned-ensembles.md) · [Lab 14. Imitation and Offline Reinforcement Learning →](lab-14-imitation-and-offline-reinforcement-learning.md)
