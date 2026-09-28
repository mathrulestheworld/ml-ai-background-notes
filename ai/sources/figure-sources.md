[Background Notes](../../README.md) › [Artificial Intelligence](../README.md)

# Figure sources

> [!WARNING]
> Work in progress: this part of the notes is still being revised.

# <a id="figure-sources"></a>Figure sources

Every figure in the AI chapters is an original plot generated for these notes on 24 September 2026 with NumPy, SciPy, Matplotlib, and NetworkX; no external artwork or course slides are reproduced. Each chapter's figures come from one script in `Sources/Figure code`, which is the complete record of the construction. The shared module `aifig.py` sets the palette, fonts, and export (opaque white background, 200 dpi PNG). The tables below summarize the problems, seeds, and settings of each figure; the [computing setup](computing-setup.md) gives the environment and commands.

Seeds refer to `numpy.random.default_rng` ("NumPy seed") or Python's `random.Random` ("Python seed"). Every script accepts figure numbers as command-line arguments (for example `python ch11_temporal.py 2`), numbered in the order of the figures in the script, and regenerates all its figures without arguments. According to their docstrings, `ch03_csp.py` takes about two and three minutes for its two figures, `ch04_games.py` about three minutes for figure 2, `ch05_sat.py` a few minutes for each figure, `ch07_planning.py` about ten minutes, and `ch10_approximate.py` about two and three minutes; the others take seconds. Results from random problems are averages over the stated number of instances; they illustrate the claims in the notes and are not benchmarks.

## <a id="1-agents-and-uninformed-search"></a>1. Agents and uninformed search

Script: `ch01_search.py`.

| Local figure | Data and construction |
| --- | --- |
| [ai-search-grid.png](images/ai-search-grid.png) | A $`21\times33`$ 4-connected grid with border walls, two vertical walls with gaps, and two horizontal walls; a "swamp" block (rows 4–16, columns 24–29) where entering a cell costs 6 instead of 1. Start $`(10,3)`$, goal $`(10,29)`$. Breadth-first graph search, depth-first graph search trying right, down, left, up, and uniform-cost search with a binary heap and lazy deletion; cells shaded by expansion order (viridis colormap, lightened), path in red. |
| [ai-search-8puzzle.png](images/ai-search-8puzzle.png) | Left: breadth-first search backward from the 8-puzzle goal $`(1,\dots,8,\text{blank})`$ over all 181,440 reachable states, counted by distance. Right: five random states at each even distance from 4 to 20 (NumPy seed 0); breadth-first graph search with the goal test on generation (nodes generated and largest frontier) and iterative deepening that avoids undoing the previous move (nodes generated). |

## <a id="2-heuristic-search"></a>2. Heuristic search

Script: `ch02_heuristic.py`.

| Local figure | Data and construction |
| --- | --- |
| [ai-heuristic-grid.png](images/ai-heuristic-grid.png) | A $`25\times41`$ grid with unit costs, a cup-shaped wall open toward the start, and a short wall near the start; start $`(12,4)`$, goal $`(12,36)`$. Best-first graph search with $`f=g`$, $`f=h`$, $`f=g+h`$, and $`f=g+2h`$ for the Manhattan distance $`h`$, ties broken toward larger $`g`$ and then first in, first out. |
| [ai-heuristic-8puzzle.png](images/ai-heuristic-8puzzle.png) | Left: A\* graph search on ten random 8-puzzle states at each even true distance from 4 to 30 (NumPy seed 0), with $`h=0`$, misplaced tiles, Manhattan distance, and an additive pattern database of tiles $`\{1,2,3,4\}`$ and $`\{5,6,7,8\}`$ built by 0–1 breadth-first search that counts only pattern-tile moves. Right: the mean of each heuristic over all 181,440 reachable states at each true distance. |

## <a id="3-constraint-satisfaction-and-local-search"></a>3. Constraint satisfaction and local search

Script: `ch03_csp.py`.

| Local figure | Data and construction |
| --- | --- |
| [ai-csp-queens.png](images/ai-csp-queens.png) | Left: $`n`$-queens for $`n=4,\dots,40`$ with columns as variables and rows tried in increasing order: chronological backtracking, backtracking with forward checking, and forward checking with MRV (ties to the lowest column); assignments counted up to a budget of two million. Right: min-conflicts with $`O(n)`$ conflict counters, median over five runs (NumPy seed 0) for $`n`$ from 10 to 1,000 with a random initial assignment and from 10 to 10,000 with a greedy one (each queen on a least-conflicted row, ties at random). |
| [ai-csp-phase.png](images/ai-csp-phase.png) | 40 random graphs with 100 vertices and $`m=\mathrm{round}(cn/2)`$ distinct random edges for each mean degree $`c`$ from 2 to 8 in steps of 0.25 (NumPy seed 1); 3-coloring by backtracking with forward checking and MRV, ties broken by the number of uncolored neighbors. Median and 90th percentile of assignments, and the fraction colorable; the dotted line marks 4.69. |

## <a id="4-adversarial-search-and-games"></a>4. Adversarial search and games

Script: `ch04_games.py`.

| Local figure | Data and construction |
| --- | --- |
| [ai-games-alphabeta.png](images/ai-games-alphabeta.png) | Uniform trees with branching factor 5 and depths 1–9, leaf values independent uniform on $`[0,1]`$ (NumPy seed 0); ten trees per depth for alpha–beta with children in index order, and the first of them with children sorted best-first by their exact minimax values, which the script checks against the Knuth–Moore count. |
| [ai-games-mcts.png](images/ai-games-mcts.png) | Tic-tac-toe; UCT with $`C=1.4`$, random playouts, and rewards 1, 0.5, 0 against a player that picks uniformly among minimax-optimal moves; 100 games for each budget from 3 to 3,000 iterations per move and each side (Python seed 0); the move with the most visits is played. |

## <a id="5-propositional-logic-and-satisfiability"></a>5. Propositional logic and satisfiability

Script: `ch05_sat.py`.

| Local figure | Data and construction |
| --- | --- |
| [ai-sat-phase.png](images/ai-sat-phase.png) | Random 3-SAT with three distinct variables per clause and random signs; 40 formulas for each $`n\in\{50,75,100\}`$ and each ratio from 2 to 7 in steps of 0.25 (NumPy seed 0). DPLL with unit propagation, branching on the literal most frequent in the shortest open clauses; fraction satisfiable and median number of branching decisions. |
| [ai-sat-growth.png](images/ai-sat-growth.png) | The same DPLL on 30 random formulas for each $`n`$ from 20 to 140 in steps of 20 at ratios 3, 4.27, and 6 (NumPy seed 1); median decisions. The printed rate is a least-squares fit of $`\log_2`$ decisions for $`n\ge60`$ at the threshold. |

## <a id="6-first-order-logic-and-knowledge-representation"></a>6. First-order logic and knowledge representation

Script: `ch06_logic.py`.

| Local figure | Data and construction |
| --- | --- |
| [ai-fol-datalog.png](images/ai-fol-datalog.png) | Left: the ten parent facts of the chapter's kinship example, drawn by generation, with the derived cousin pairs as dashed arcs. Right: the recursive ancestor rule on a chain of $`n`$ parent facts for $`n=10,\dots,320`$; rule firings of naive evaluation (all rules on all facts every round) and semi-naive evaluation (at least one fact new in the previous round), with reference curves $`n^3/3`$ and $`n^2/2`$. |

## <a id="7-automated-planning"></a>7. Automated planning

Script: `ch07_planning.py`.

| Local figure | Data and construction |
| --- | --- |
| [ai-plan-heuristics.png](images/ai-plan-heuristics.png) | Blocks world with move actions between blocks and to and from the table. Ten tasks for each number of blocks from 3 to 14 (NumPy seed 0): random initial towers (each block ends a tower with probability 0.35) and a goal of all "on" atoms of another random configuration, resampled until it does not already hold. Breadth-first search (up to 7 blocks), A\* with $`h^{\max}`$ (up to 6), A\* with $`h^{\mathrm{add}}`$ (up to 10), and greedy best-first search with $`h^{\mathrm{FF}}`$ (up to 14), all as graph search with unit costs; relaxation costs are computed by fixed-point iteration over the ground actions. Right: plan length divided by the breadth-first optimum. |

## <a id="8-bayesian-networks-and-markov-networks"></a>8. Bayesian networks and Markov networks

Script: `ch08_bayesnets.py`.

| Local figure | Data and construction |
| --- | --- |
| [ai-bn-structures.png](images/ai-bn-structures.png) | Diagrams of the chain, fork, and collider with the middle node unobserved and observed (shaded). |
| [ai-bn-explaining-away.png](images/ai-bn-explaining-away.png) | Left: exact posteriors of burglary and earthquake in the burglary network (CPTs as in the chapter) by enumeration, for five evidence sets. Right: 2,000 pairs of independent standard normal "talent" and "looks" (NumPy seed 0), selected when their sum exceeds 1.5; correlations in the whole sample, the selected, and the unselected. |

## <a id="9-exact-inference"></a>9. Exact inference

Script: `ch09_inference.py`.

| Local figure | Data and construction |
| --- | --- |
| [ai-inference-width.png](images/ai-inference-width.png) | Left: induced width of elimination orders on $`n\times n`$ grids ($`n=2,\dots,15`$, NetworkX `grid_2d_graph`): row by row, min-degree and min-fill (ties by node name), and random (mean of five, NumPy seed 0). Right: five random Bayesian networks for each size from 10 to 320 nodes, each node with 0–3 parents chosen uniformly among earlier nodes, moralized and eliminated in min-fill order; median of $`\sum2^{|\text{factor}|}`$ over the product factors created, for binary variables, against $`n2^n`$. |

## <a id="10-approximate-inference"></a>10. Approximate inference

Script: `ch10_approximate.py`.

| Local figure | Data and construction |
| --- | --- |
| [ai-approx-sampling.png](images/ai-approx-sampling.png) | $`P(B\mid j,m)`$ in the burglary network (exact value 0.284171835). Rejection sampling and likelihood weighting with 100 runs, and Gibbs sampling over $`(B,E,A)`$ with a burn-in of 100 sweeps and 20 runs, for each sample size from 100 to 100,000 (30,000 for Gibbs) (NumPy seed 0); mean absolute error, counting 0.284 when rejection keeps no sample. |
| [ai-approx-ising.png](images/ai-approx-ising.png) | Left: five $`4\times4`$ Ising models per coupling $`J`$ with fields $`h_i\sim\mathcal N(0,0.3^2)`$ (NumPy seed 1); exact marginals by enumeration of 65,536 states; mean field by 500 coordinate sweeps; loopy belief propagation with damping 0.5 until the messages change by less than $`10^{-10}`$ or 500 iterations; Gibbs sampling with 200 burn-in and 2,000 recorded sweeps. Right: Gibbs sampling on a $`24\times24`$ grid without field, started with all spins up, random-order sweeps, $`J=0.3`$ and $`J=0.6`$, 1,500 sweeps. |

## <a id="11-temporal-probabilistic-models"></a>11. Temporal probabilistic models

Script: `ch11_temporal.py`.

| Local figure | Data and construction |
| --- | --- |
| [ai-temporal-casino.png](images/ai-temporal-casino.png) | 300 rolls of the occasionally dishonest casino (switch probabilities 0.05 and 0.1; loaded die shows six with probability 0.5; NumPy seed 4), started from the stationary distribution $`(2/3,1/3)`$. Normalized forward and backward messages, smoothed posteriors, and Viterbi in log space with the true parameters. |
| [ai-temporal-tracking.png](images/ai-temporal-tracking.png) | A constant-velocity model in two dimensions with $`\Delta t=1`$, random accelerations with variance 0.1 per coordinate, position noise with variance 1, initial state $`(0,0,1,0.3)`$, 60 steps (NumPy seed 0). Kalman filter with initial covariance $`I`$; bootstrap particle filter with multinomial resampling every step, particles initialized around the true initial state with unit noise. Left: one track, drawn with the second coordinate horizontal, and Kalman two-standard-deviation ellipses every eight steps. Right: position RMSE averaged over 30 tracks for 10 to 3,000 particles. |

## <a id="12-decision-theory-and-the-value-of-information"></a>12. Decision theory and the value of information

Script: `ch12_decisions.py`.

| Local figure | Data and construction |
| --- | --- |
| [ai-decision-utility.png](images/ai-decision-utility.png) | Left: $`U(w)=\log w`$ and the lottery $`[0.5,20;\,0.5,180]`$ in thousands. Right: $`k`$ alternatives with true values $`\mathcal N(0,1)`$ and estimates with added $`\mathcal N(0,\sigma^2)`$ noise, $`\sigma\in\{0.5,1,2\}`$; 20,000 trials per point (NumPy seed 0); mean of estimate minus true value for the alternative with the highest estimate. |
| [ai-decision-vpi.png](images/ai-decision-vpi.png) | A treat-or-wait decision with utilities 0.9 (treat, either state), 0.2 (wait, disease), 1.0 (wait, healthy); expected utilities and the value of perfect information for priors on a grid of 999 points, and the value of tests with (sensitivity, specificity) of (1, 1), (0.95, 0.9), (0.8, 0.7), and (0.6, 0.55). |

## <a id="13-causal-inference"></a>13. Causal inference

Script: `ch13_causal.py`.

| Local figure | Data and construction |
| --- | --- |
| [ai-causal-simpson.png](images/ai-causal-simpson.png) | Left: success counts from Charig et al. (1986) as reproduced in the causal-inference literature: A 81/87 and 192/263, B 234/270 and 55/80. Right: 500 datasets of 2,000 units (NumPy seed 0) with $`Z\sim\mathcal N(0,1)`$, $`P(X=1\mid Z)=\sigma(1.5Z)`$, $`Y=X+2Z+\varepsilon`$, and a collider $`C=X+Y+\varepsilon'`$; difference of means, least-squares adjustment for $`Z`$, inverse propensity weighting with a logistic propensity fitted by 25 Newton steps, and least-squares adjustment for $`Z`$ and $`C`$. |
| [ai-causal-iv.png](images/ai-causal-iv.png) | $`X=aZ+\gamma U+\varepsilon_X`$, $`Y=X+\gamma U+\varepsilon_Y`$ with independent standard normal $`Z`$, $`U`$, and noises; 1,000 units and 400 repetitions per setting (NumPy seed 1). Left: $`a=0.8`$ and $`\gamma`$ from 0 to 2, least-squares slope versus the Wald estimate $`\widehat{\mathrm{Cov}}(Z,Y)/\widehat{\mathrm{Cov}}(Z,X)`$. Right: $`\gamma=1`$ and $`a`$ from 0.03 to 0.8; median and 10th–90th percentiles of the Wald estimate. |

## <a id="14-learning-graphical-models"></a>14. Learning graphical models

Script: `ch14_learning.py`.

| Local figure | Data and construction |
| --- | --- |
| [ai-learn-trees.png](images/ai-learn-trees.png) | Random trees over 20 variables with 4 values (each node's parent uniform among earlier nodes); CPT rows drawn from a Dirichlet with concentration 1 plus $`4s`$ on the parent's value, $`s=0.4`$ (left) or $`s=2`$ (right) (NumPy seed 0). Left: Chow–Liu with plug-in mutual information and NetworkX's maximum spanning tree, 20 models per sample size. Right: one model, 20,000 test samples, CPTs estimated from 20 training sets per size with Dirichlet pseudocounts 0.1, 1, and 10; the fraction of test samples given probability zero by maximum likelihood is printed. |

## <a id="15-game-theory-and-multiagent-systems"></a>15. Game theory and multiagent systems

Script: `ch15_gametheory.py`.

| Local figure | Data and construction |
| --- | --- |
| [ai-gt-regret.png](images/ai-gt-regret.png) | Left: Hedge self-play on rock-paper-scissors with step 0.1 for 3,000 rounds, the row player's initial weights $`(1,0,0)`$. Right: ten random $`10\times10`$ zero-sum games with standard normal payoffs (NumPy seed 0), Hedge self-play with step $`2\sqrt{8\ln10/10^4}`$ for 10,000 rounds; duality gap of the running averages at 25 logarithmically spaced times; game values from SciPy's `linprog`. |
