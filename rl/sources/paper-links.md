[Background Notes](../../README.md) › [Reinforcement Learning](../README.md)

# Paper links

> [!WARNING]
> Work in progress: this part of the notes is still being revised.

# <a id="papers"></a>Papers

The notes cite the original papers where their ideas arise. This index collects the principal ones by topic; the chapters give further references inline.

## <a id="markov-decision-processes-and-dynamic-programming"></a>Markov decision processes and dynamic programming

| Work | Idea developed in the module |
| --- | --- |
| Bellman, [*Dynamic Programming*](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) (1957) | Bellman's principle of optimality and the dynamic-programming framework from which Markov decision processes grew (chapter 1). |
| Blackwell, [*Discrete Dynamic Programming*](https://doi.org/10.1214/aoms/1177704593) (1962) | Some deterministic stationary policy is optimal for all discount factors close enough to one, and such a Blackwell-optimal policy also maximizes the average reward (chapter 2). |
| Puterman and Shin, [*Modified Policy Iteration Algorithms for Discounted Markov Decision Problems*](https://doi.org/10.1287/mnsc.24.11.1127) (1978) | Modified policy iteration, which evaluates each policy with $`m`$ sweeps: value iteration for $`m=1`$, policy iteration for $`m=\infty`$, and usually cheaper than both in between (chapter 2). |
| de Farias and Van Roy, [*The Linear Programming Approach to Approximate Dynamic Programming*](https://doi.org/10.1287/opre.51.6.850.24925) (2003) | Approximate linear programming: the linear program for the optimal values, with the values restricted to a linear combination of features (chapter 2). |
| Ye, [*The Simplex and Policy-Iteration Methods Are Strongly Polynomial for the Markov Decision Problem with a Fixed Discount Rate*](https://doi.org/10.1287/moor.1110.0516) (2011) | For a fixed discount factor, policy iteration needs a number of iterations polynomial in the numbers of states and actions and in $`1/(1-\gamma)`$, whatever the rewards and transitions (chapter 2). |
| Silver et al., [*Reward Is Enough*](https://doi.org/10.1016/j.artint.2021.103535) (2021) | The argument that maximizing reward in a rich enough environment could by itself give rise to perception, language, and social intelligence (chapter 1). |
| Bowling et al., [*Settling the Reward Hypothesis*](https://arxiv.org/abs/2212.10420) (2023) | The reward hypothesis holds exactly when the designer's preferences over distributions of trajectories satisfy the von Neumann–Morgenstern axioms and one more axiom (chapter 1). |
| Pardo et al., [*Time Limits in Reinforcement Learning*](https://arxiv.org/abs/1712.00378) (2018) | At a time limit the episode is truncated, not terminated, so the correct target bootstraps from the value of the last state (chapters 1 and 19). |

## <a id="bandits"></a>Bandits

| Work | Idea developed in the module |
| --- | --- |
| Thompson, [*On the Likelihood That One Unknown Probability Exceeds Another in View of the Evidence of Two Samples*](https://doi.org/10.2307/2332286) (1933) | Adaptive allocation of patients to treatments, the original motivation for bandits and the source of Thompson sampling (chapter 3). |
| Lai and Robbins, [*Asymptotically Efficient Adaptive Allocation Rules*](https://doi.org/10.1016/0196-8858(85)90002-8) (1985) | Any consistent strategy pulls each suboptimal arm at least logarithmically often, with a constant set by the relative entropy between the arms' distributions (chapter 3). |
| Auer, Cesa-Bianchi, and Fischer, [*Finite-Time Analysis of the Multiarmed Bandit Problem*](https://doi.org/10.1023/A:1013689704352) (2002) | UCB1 and its logarithmic regret in finite time (chapter 3). |
| Agrawal and Goyal, [*Analysis of Thompson Sampling for the Multi-Armed Bandit Problem*](https://proceedings.mlr.press/v23/agrawal12.html) (2012) | The first proof that Thompson sampling has logarithmic regret (chapter 3). |
| Gittins, [*Bandit Processes and Dynamic Allocation Indices*](https://doi.org/10.1111/j.2517-6161.1979.tb01068.x) (1979) | The Bayes-optimal policy has an index form: pull the arm whose index, computed from its own posterior alone, is largest (chapter 4). |
| Auer et al., [*The Nonstochastic Multiarmed Bandit Problem*](https://doi.org/10.1137/S0097539701398375) (2002) | Exp3: exponential weights run on importance-weighted estimates of rewards chosen by an adversary (chapter 4). |
| Li et al., [*A Contextual-Bandit Approach to Personalized News Article Recommendation*](https://arxiv.org/abs/1003.0146) (2010), and Abbasi-Yadkori, Pál, and Szepesvári, [*Improved Algorithms for Linear Stochastic Bandits*](https://papers.nips.cc/paper_files/paper/2011/hash/e1d5be1c7f2f456670de3d53c7b54f4a-Abstract.html) (2011) | LinUCB for news recommendation, and the confidence ellipsoid around the least-squares estimate, whose axes are short in the directions the data have explored (chapter 4). |
| Foster and Rakhlin, [*Beyond UCB: Optimal and Efficient Contextual Bandits with Regression Oracles*](https://arxiv.org/abs/2002.04926) (2020) | SquareCB, which reduces the contextual bandit to a sequence of regression problems solved by an oracle for the class (chapter 4). |
| Dudík, Langford, and Li, [*Doubly Robust Policy Evaluation and Learning*](https://arxiv.org/abs/1103.4601) (2011) | The doubly robust estimator of a policy's value from logged data, which uses a reward model as a control variate for inverse propensity weighting (chapter 4). |

## <a id="monte-carlo-and-temporal-difference-learning"></a>Monte Carlo and temporal-difference learning

| Work | Idea developed in the module |
| --- | --- |
| Singh and Sutton, [*Reinforcement Learning with Replacing Eligibility Traces*](https://doi.org/10.1007/BF00114726) (1996) | First-visit and every-visit Monte Carlo compared in mean squared error, and replacing traces, which reset a visited state's trace to 1 (chapters 5 and 8). |
| Sutton, [*Learning to Predict by the Methods of Temporal Differences*](https://doi.org/10.1007/BF00115009) (1988) | TD(λ), which implements the λ-return incrementally with eligibility traces (chapter 8). |
| Jaakkola, Jordan, and Singh, [*On the Convergence of Stochastic Iterative Dynamic Programming Algorithms*](https://doi.org/10.1162/neco.1994.6.6.1185) (1994), and Tsitsiklis, [*Asynchronous Stochastic Approximation and Q-Learning*](https://doi.org/10.1007/BF00993306) (1994) | Tabular TD(0) and Q-learning converge with probability one, as asynchronous stochastic approximation of contraction mappings (chapters 6, 7, and 8). |
| Tesauro, [*Temporal Difference Learning and TD-Gammon*](https://doi.org/10.1145/203330.203343) (1995) | TD-Gammon: a network trained by TD(λ) from self-play that played backgammon close to the level of the world's best players (chapters 1 and 6). |
| Schultz, Dayan, and Montague, [*A Neural Substrate of Prediction and Reward*](https://doi.org/10.1126/science.275.5306.1593) (1997) | The phasic responses of dopamine neurons behave like a TD error: positive when things turn out better than predicted, zero when as predicted, negative when worse (chapter 6). |
| Bhandari, Russo, and Singal, [*A Finite Time Analysis of Temporal Difference Learning with Linear Function Approximation*](https://arxiv.org/abs/1806.02450) (2018) | Finite-time error bounds for TD learning with linear function approximation (chapter 6). |

## <a id="control-multi-step-and-off-policy-learning"></a>Control, multi-step, and off-policy learning

| Work | Idea developed in the module |
| --- | --- |
| Watkins, [*Learning from Delayed Rewards*](https://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf) (1989), and Watkins and Dayan, [*Q-Learning*](https://doi.org/10.1007/BF00992698) (1992) | Q-learning, whose target maximizes over the next actions, the λ-return and Watkins's Q(λ), and the proof that tabular Q-learning converges to $`q_*`$ (chapters 7 and 8). |
| Singh et al., [*Convergence Results for Single-Step On-Policy Reinforcement-Learning Algorithms*](https://doi.org/10.1023/A:1007678930559) (2000) | Policies that are greedy in the limit with infinite exploration (GLIE), under which tabular SARSA converges to the optimal action values (chapters 5 and 7). |
| van Hasselt, [*Double Q-Learning*](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) (2010) | The overestimation caused by maximizing over noisy estimates, removed by learning two independent estimates, one to choose the action and the other to evaluate it (chapters 7 and 16). |
| van Seijen and Sutton, [*True Online TD(λ)*](https://proceedings.mlr.press/v32/seijen14.html) (2014) | True online TD(λ), a backward view that reproduces the online λ-return algorithm exactly for linear function approximation at $`O(d)`$ cost per step (chapter 8). |
| Peng and Williams, [*Incremental Multi-Step Q-Learning*](https://doi.org/10.1007/BF00114731) (1996) | Peng's Q(λ), which never cuts its traces and learns faster, at the price of converging to a mixture of the values of the behavior and greedy policies (chapter 8). |
| Precup, Sutton, and Singh, [*Eligibility Traces for Off-Policy Policy Evaluation*](https://scholarworks.umass.edu/cs_faculty_pubs/80/) (2000) | Per-decision importance sampling, in which each reward is weighted only by the ratios of the actions before it (chapter 9). |
| Munos et al., [*Safe and Efficient Off-Policy Reinforcement Learning*](https://arxiv.org/abs/1606.02647) (2016) | Retrace: traces made of importance ratios truncated at one, convergent for any behavior policy and efficient when the two policies are close (chapter 9). |
| De Asis et al., [*Multi-Step Reinforcement Learning: A Unifying Algorithm*](https://arxiv.org/abs/1703.01327) (2018) | Q(σ), which chooses at each step between sampling the action with an importance ratio and taking the expectation over actions, as tree backup does (chapter 9). |
| Jiang and Li, [*Doubly Robust Off-Policy Value Evaluation for Reinforcement Learning*](https://arxiv.org/abs/1511.03722) (2016) | The doubly robust estimator for sequential decisions, with an approximate model of the action values as a control variate (chapter 9). |

## <a id="planning-with-tabular-models"></a>Planning with tabular models

| Work | Idea developed in the module |
| --- | --- |
| Sutton, [*Dyna, an Integrated Architecture for Learning, Planning, and Reacting*](https://doi.org/10.1145/122344.122377) (1991) | Dyna: one agent that learns a model, plans with simulated experience from it, and learns directly from real experience, all at once (chapter 10). |
| Moore and Atkeson, [*Prioritized Sweeping: Reinforcement Learning with Less Data and Less Time*](https://doi.org/10.1007/BF00993104) (1993) | Prioritized sweeping: a priority queue of state–action pairs ordered by the size of their pending update, so that changes propagate backward to predecessors (chapter 10). |
| Barto, Bradtke, and Singh, [*Learning to Act Using Real-Time Dynamic Programming*](https://doi.org/10.1016/0004-3702%2894%2900011-O) (1995) | Real-time dynamic programming, the on-policy trajectory-sampling version of value iteration (chapter 10). |
| Tesauro and Galperin, [*On-Line Policy Improvement Using Monte-Carlo Search*](https://papers.nips.cc/paper_files/paper/1996/hash/996009f2374006606f4c0b0fda878af1-Abstract.html) (1996) | Rollout algorithms, which improve a base policy at decision time by simulating each action followed by the base policy (chapter 10). |
| Kearns, Mansour, and Ng, [*A Sparse Sampling Algorithm for Near-Optimal Planning in Large Markov Decision Processes*](https://doi.org/10.1023/A:1017932429737) (2002) | Sparse sampling: near-optimal decision-time planning with a computation that does not depend on the number of states (chapter 10). |
| Kocsis and Szepesvári, [*Bandit Based Monte-Carlo Planning*](https://doi.org/10.1007/11871842_29) (2006) | UCT, which applies UCB1 to the choice among each node's children in Monte Carlo tree search (chapter 10). |
| van Hasselt, Hessel, and Aslanides, [*When to Use Parametric Models in Reinforcement Learning?*](https://arxiv.org/abs/1906.05243) (2019) | Replay as a nonparametric model, as good as a parametric one for updates from observed states, and data-efficient Rainbow (chapters 10 and 18). |

## <a id="function-approximation-and-the-deadly-triad"></a>Function approximation and the deadly triad

| Work | Idea developed in the module |
| --- | --- |
| Sutton, [*Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding*](https://proceedings.neurips.cc/paper/1995/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html) (1996) | Tile coding, with several tilings offset from one another, applied successfully to control, and the name Sarsa (chapters 7 and 11). |
| Tsitsiklis and Van Roy, [*An Analysis of Temporal-Difference Learning with Function Approximation*](https://doi.org/10.1109/9.580874) (1997) | Linear TD converges under the on-policy distribution, to a fixed point whose error is at most $`(1-\gamma\lambda)/(1-\gamma)`$ times the best achievable (chapters 8 and 11). |
| Bradtke and Barto, [*Linear Least-Squares Algorithms for Temporal Difference Learning*](https://doi.org/10.1007/BF00114723) (1996) | LSTD, which computes the TD fixed point directly from estimates of $`\mathbf A`$ and $`\mathbf b`$ built from all the data (chapter 11). |
| Baird, [*Residual Algorithms: Reinforcement Learning with Function Approximation*](https://doi.org/10.1016/B978-1-55860-377-6.50013-X) (1995) | Baird's counterexample, on which off-policy semi-gradient TD diverges even with exact expected updates (chapter 12). |
| Sutton et al., [*Fast Gradient-Descent Methods for Temporal-Difference Learning with Linear Function Approximation*](https://doi.org/10.1145/1553374.1553501) (2009) | Gradient-TD methods, which descend the projected Bellman error with an auxiliary weight vector learned on a faster time scale and converge under off-policy sampling (chapter 12). |
| Sutton, Mahmood, and White, [*An Emphatic Approach to the Problem of Off-Policy Temporal-Difference Learning*](https://jmlr.org/papers/v17/14-488.html) (2016) | Emphatic TD, which reweights each update by a followon trace so that the expected update is stable with linear function approximation (chapters 9 and 12). |
| Ernst, Geurts, and Wehenkel, [*Tree-Based Batch Mode Reinforcement Learning*](https://jmlr.org/papers/v6/ernst05a.html) (2005), and Riedmiller, [*Neural Fitted Q Iteration – First Experiences with a Data Efficient Neural Reinforcement Learning Method*](https://doi.org/10.1007/11564096_32) (2005) | Fitted Q-iteration, repeated regression onto bootstrapped targets from a fixed batch of transitions, with trees and then with neural networks (chapters 12 and 16). |
| Munos and Szepesvári, [*Finite-Time Bounds for Fitted Value Iteration*](https://jmlr.org/papers/v9/munos08a.html) (2008) | The loss of fitted value iteration bounded by the regression errors made along the way and by concentrability coefficients of the data (chapters 12, 16, and 30). |
| van Hasselt et al., [*Deep Reinforcement Learning and the Deadly Triad*](https://arxiv.org/abs/1812.02648) (2018) | The deadly triad measured in 336 variants of deep Q-learning: unbounded divergence was rare, but soft divergence was common (chapters 12 and 16). |

## <a id="policy-gradients-and-actorcritic-methods"></a>Policy gradients and actor–critic methods

| Work | Idea developed in the module |
| --- | --- |
| Barto, Sutton, and Anderson, [*Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems*](https://doi.org/10.1109/TSMC.1983.6313077) (1983) | The actor–critic architecture: a policy, the actor, learned together with a value function, the critic (chapter 13). |
| Williams, [*Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*](https://doi.org/10.1007/BF00992696) (1992) | REINFORCE, the Monte Carlo policy gradient, with a baseline to reduce its variance (chapters 13 and 28). |
| Sutton et al., [*Policy Gradient Methods for Reinforcement Learning with Function Approximation*](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) (1999) | The policy gradient theorem, and compatible features, with which an approximate critic still gives the exact gradient (chapter 13). |
| Konda and Tsitsiklis, [*Actor-Critic Algorithms*](https://papers.nips.cc/paper_files/paper/1999/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) (1999) | Convergence of actor–critic methods on two time scales, with the critic learning faster than the actor (chapter 13). |
| Greensmith, Bartlett, and Baxter, [*Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning*](https://jmlr.org/papers/v5/greensmith04a.html) (2004) | Baselines as control variates, and the variance-minimizing baseline, which weights the returns by the squared score (chapter 13). |
| Kakade, [*A Natural Policy Gradient*](https://papers.nips.cc/paper_files/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html) (2001) | The natural policy gradient $`\mathbf F^{-1}\nabla J`$, steepest ascent per unit of KL divergence between policies (chapters 13 and 20). |
| Peters and Schaal, [*Natural Actor-Critic*](https://doi.org/10.1016/j.neucom.2007.11.026) (2008) | The natural actor–critic, which learns a compatible critic and moves the policy parameters along its weights (chapter 13). |
| Silver et al., [*Deterministic Policy Gradient Algorithms*](https://proceedings.mlr.press/v32/silver14.html) (2014) | The deterministic policy gradient theorem, the basis of DDPG and TD3 (chapters 13 and 21). |
| Schulman et al., [*High-Dimensional Continuous Control Using Generalized Advantage Estimation*](https://arxiv.org/abs/1506.02438) (2016) | Generalized advantage estimation: the λ-return applied to advantages, an exponentially weighted average of $`n`$-step estimates (chapters 8, 13, and 19). |
| Mei et al., [*On the Global Convergence Rates of Softmax Policy Gradient Methods*](https://arxiv.org/abs/2005.06392) (2020) | Exact softmax policy gradient converges at the rate $`O(1/k)`$, with constants that depend on the problem and the initialization, and plateaus near deterministic policies (chapters 13, 19, and 20). |

## <a id="partial-observability"></a>Partial observability

| Work | Idea developed in the module |
| --- | --- |
| Åström, [*Optimal Control of Markov Processes with Incomplete State Information*](https://doi.org/10.1016/0022-247X%2865%2990154-X) (1965) | The belief as a sufficient statistic for the history: the optimal action depends on the history only through it (chapter 14). |
| Smallwood and Sondik, [*The Optimal Control of Partially Observable Markov Processes over a Finite Horizon*](https://doi.org/10.1287/opre.21.5.1071) (1973) | The optimal value is piecewise linear and convex in the belief for every finite horizon, a maximum over a finite set of alpha-vectors (chapters 1 and 14). |
| Kaelbling, Littman, and Cassandra, [*Planning and Acting in Partially Observable Stochastic Domains*](https://doi.org/10.1016/S0004-3702%2898%2900023-X) (1998) | POMDPs for planning, the witness algorithm, and the tiger problem (chapters 1 and 14). |
| Pineau, Gordon, and Thrun, [*Point-Based Value Iteration: An Anytime Algorithm for POMDPs*](https://www.ijcai.org/Proceedings/03/Papers/147.pdf) (2003) | Point-based value iteration, which backs up one alpha-vector per belief in a finite set of reachable beliefs, against the curse of history (chapter 14). |
| Spaan and Vlassis, [*Perseus: Randomized Point-Based Value Iteration for POMDPs*](https://doi.org/10.1613/jair.1659) (2005) | Randomized point-based backups over a large set of beliefs, which often cover the whole set with a few vectors (chapter 14). |
| Silver and Veness, [*Monte-Carlo Planning in Large POMDPs*](https://papers.nips.cc/paper_files/paper/2010/hash/edfbe1afcf9246bb0d40eb4d8027d90f-Abstract.html) (2010) | POMCP: Monte Carlo tree search over histories, with each simulation starting from a state sampled from the belief (chapter 14). |
| Ye et al., [*DESPOT: Online POMDP Planning with Regularization*](https://doi.org/10.1613/jair.5328) (2017) | A sparse search tree over a fixed set of sampled scenarios, regularized against overfitting them, with performance guarantees (chapter 14). |
| Hausknecht and Stone, [*Deep Recurrent Q-Learning for Partially Observable MDPs*](https://arxiv.org/abs/1507.06527) (2015) | DRQN, which replaced DQN's frame stack by an LSTM (chapter 14). |
| Ni, Eysenbach, and Salakhutdinov, [*Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs*](https://arxiv.org/abs/2110.05038) (2022) | Carefully tuned recurrent model-free agents as strong baselines across partially observable benchmarks (chapter 14). |
| Littman, Sutton, and Singh, [*Predictive Representations of State*](https://papers.nips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html) (2001) | Predictive state representations: the state as the probabilities of future tests, observable quantities that can be estimated from data (chapter 14). |

## <a id="optimal-control-and-trajectory-optimization"></a>Optimal control and trajectory optimization

| Work | Idea developed in the module |
| --- | --- |
| Bradtke, Ydstie, and Barto, [*Adaptive Linear Quadratic Control Using Policy Iteration*](https://doi.org/10.1109/ACC.1994.735224) (1994) | Model-free policy iteration for the LQR, with a quadratic Q-function estimated by least squares (chapter 15). |
| Fazel et al., [*Global Convergence of Policy Gradient Methods for the Linear Quadratic Regulator*](https://arxiv.org/abs/1801.05039) (2018) | Policy gradient converges to the globally optimal LQR controller despite the nonconvexity of the cost in the gain (chapters 15 and 30). |
| Recht, [*A Tour of Reinforcement Learning: The View from Continuous Control*](https://arxiv.org/abs/1806.09460) (2019) | The LQR as the standard test bed for reinforcement learning with continuous states and actions (chapter 15). |
| Mayne, [*A Second-Order Gradient Method for Determining Optimal Trajectories of Non-Linear Discrete-Time Systems*](https://doi.org/10.1080/00207176608921369) (1966), and Li and Todorov, [*Iterative Linear Quadratic Regulator Design for Nonlinear Biological Movement Systems*](https://doi.org/10.5220/0001143902220229) (2004) | Differential dynamic programming, a Newton-like shooting method, and iLQR, its Gauss–Newton version without the second derivatives of the dynamics (chapter 15). |
| Kelly, [*An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation*](https://doi.org/10.1137/16M1062569) (2017) | Direct collocation, which optimizes states and controls together with the dynamics as constraints (chapter 15). |
| de Boer et al., [*A Tutorial on the Cross-Entropy Method*](https://doi.org/10.1007/s10479-005-5724-z) (2005) | The cross-entropy method: sample control sequences from a Gaussian, keep the elites, and refit the Gaussian to them (chapter 15). |
| Williams, Aldrich, and Theodorou, [*Model Predictive Path Integral Control: From Theory to Parallel Computation*](https://doi.org/10.2514/1.G001921) (2017) | MPPI, which averages all sampled control sequences with weights $`\propto e^{-J/\lambda}`$, the solution of a KL-regularized problem (chapter 15). |
| Mayne et al., [*Constrained Model Predictive Control: Stability and Optimality*](https://doi.org/10.1016/S0005-1098%2899%2900214-9) (2000) | Model predictive control with constraints on states and controls, and the terminal costs and constraints that make it stable (chapter 15). |
| Levine and Koltun, [*Guided Policy Search*](https://proceedings.mlr.press/v28/levine13.html) (2013) | Guided policy search: a global neural network policy trained to imitate many local trajectory optimizers (chapter 15). |

## <a id="deep-q-learning-and-distributional-rl"></a>Deep Q-learning and distributional RL

| Work | Idea developed in the module |
| --- | --- |
| Mnih et al., [*Playing Atari with Deep Reinforcement Learning*](https://arxiv.org/abs/1312.5602) (2013), and [*Human-Level Control through Deep Reinforcement Learning*](https://doi.org/10.1038/nature14236) (2015) | DQN: Atari games learned from pixels and the score with one architecture and one set of hyperparameters, stabilized by replay and a target network (chapter 16). |
| Bellemare et al., [*The Arcade Learning Environment: An Evaluation Platform for General Agents*](https://doi.org/10.1613/jair.3912) (2013) | The Arcade Learning Environment, the standard benchmark of deep reinforcement learning for a decade (chapter 16). |
| Lin, [*Self-Improving Reactive Agents Based on Reinforcement Learning, Planning and Teaching*](https://doi.org/10.1007/BF00992699) (1992) | Experience replay, updating on transitions stored in a memory (chapter 16). |
| van Hasselt, Guez, and Silver, [*Deep Reinforcement Learning with Double Q-Learning*](https://arxiv.org/abs/1509.06461) (2016) | Double DQN: the online network chooses the next action and the target network evaluates it (chapters 7 and 16). |
| Wang et al., [*Dueling Network Architectures for Deep Reinforcement Learning*](https://arxiv.org/abs/1511.06581) (2016) | The dueling architecture, with separate streams for the state value and the advantages (chapter 16). |
| Machado et al., [*Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents*](https://doi.org/10.1613/jair.5699) (2018) | Evaluation protocols for Atari, with sticky actions that repeat the previous action with probability 0.25 against memorized action sequences (chapter 16). |
| Agarwal et al., [*Deep Reinforcement Learning at the Edge of the Statistical Precipice*](https://arxiv.org/abs/2108.13264) (2021) | Interval estimates of robust aggregates, such as the interquartile mean with bootstrap confidence intervals, in place of point estimates from a few seeds (chapters 16 and 18). |
| Bellemare, Dabney, and Munos, [*A Distributional Perspective on Reinforcement Learning*](https://arxiv.org/abs/1707.06887) (2017) | The distributional Bellman operator and C51, whose categorical return distributions made deep Q-learning agents much better even when only their means are used (chapter 17). |
| Dabney et al., [*Distributional Reinforcement Learning with Quantile Regression*](https://arxiv.org/abs/1710.10044) (2018), and [*Implicit Quantile Networks for Distributional Reinforcement Learning*](https://arxiv.org/abs/1806.06923) (2018) | Quantile regression for the locations of atoms with fixed probabilities, and implicit quantile networks, which take the quantile level as an input (chapter 17). |
| Dabney et al., [*A Distributional Code for Value in Dopamine-Based Reinforcement Learning*](https://doi.org/10.1038/s41586-019-1924-6) (2020) | Different dopamine neurons seem to encode different quantiles or expectiles of the reward distribution, as a distributional TD algorithm would (chapters 6 and 17). |

## <a id="data-efficient-and-distributed-agents"></a>Data-efficient and distributed agents

| Work | Idea developed in the module |
| --- | --- |
| Schaul et al., [*Prioritized Experience Replay*](https://arxiv.org/abs/1511.05952) (2016) | Replay in proportion to the magnitude of each transition's last TD error, with importance weights to correct the bias (chapter 18). |
| Hessel et al., [*Rainbow: Combining Improvements in Deep Reinforcement Learning*](https://arxiv.org/abs/1710.02298) (2018) | Six extensions of DQN combined in one agent, and an ablation of each (chapters 8 and 18). |
| Horgan et al., [*Distributed Prioritized Experience Replay*](https://arxiv.org/abs/1803.00933) (2018), and Kapturowski et al., [*Recurrent Experience Replay in Distributed Reinforcement Learning*](https://openreview.net/forum?id=r1lyTjAqYX) (2019) | Ape-X: hundreds of actors with different exploration rates feeding one prioritized replay memory, and R2D2, which trains recurrent agents on replayed sequences with stored states and a burn-in prefix (chapters 8, 14, and 18). |
| Badia et al., [*Agent57: Outperforming the Atari Human Benchmark*](https://arxiv.org/abs/2003.13350) (2020) | A bandit meta-controller that chooses among a family of exploration policies, the first agent to exceed the human benchmark on all 57 Atari games (chapter 18). |
| Kaiser et al., [*Model-Based Reinforcement Learning for Atari*](https://arxiv.org/abs/1903.00374) (2020) | The Atari 100k benchmark, 100,000 agent steps on 26 games, and the model-based agent SimPLe (chapter 18). |
| Schwarzer et al., [*Bigger, Better, Faster: Human-Level Atari with Human-Level Efficiency*](https://arxiv.org/abs/2305.19452) (2023) | BBF: a much larger network, a high replay ratio with periodic resets, self-predictive representations, and annealing of the discount and the multi-step horizon (chapter 18). |
| Nikishin et al., [*The Primacy Bias in Deep Reinforcement Learning*](https://arxiv.org/abs/2205.07802) (2022) | Overfitting to early experience, the primacy bias, countered by periodically resetting some of the network's layers (chapters 16 and 18). |
| Gallici et al., [*Simplifying Deep Temporal Difference Learning*](https://arxiv.org/abs/2407.04811) (2025) | Layer normalization keeps TD learning stable without replay or target networks, and PQN, Q-learning with many vectorized environments and λ-returns (chapters 12 and 18). |
| Mnih et al., [*Asynchronous Methods for Deep Reinforcement Learning*](https://arxiv.org/abs/1602.01783) (2016) | A3C: parallel actors on CPU threads, each computing gradients on short rollouts and applying them without locks to shared parameters (chapters 8 and 19). |
| Espeholt et al., [*IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures*](https://arxiv.org/abs/1802.01561) (2018) | Actors decoupled from a central learner, and V-trace, the truncated off-policy correction for the lag of the actors' policies (chapters 9 and 19). |
| Rudin et al., [*Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning*](https://arxiv.org/abs/2109.11978) (2021) | Thousands of robots simulated in parallel on one GPU, with which a quadruped learned to walk in minutes (chapters 19 and 31). |

## <a id="trust-regions-ppo-continuous-control-and-maximum-entropy-rl"></a>Trust regions, PPO, continuous control, and maximum-entropy RL

| Work | Idea developed in the module |
| --- | --- |
| Kakade and Langford, [*Approximately Optimal Approximate Reinforcement Learning*](https://dl.acm.org/doi/10.5555/645531.656005) (2002) | The performance difference lemma, and conservative policy iteration with its lower bound on the improvement (chapters 1 and 20). |
| Schulman et al., [*Trust Region Policy Optimization*](https://arxiv.org/abs/1502.05477) (2015), and [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347) (2017) | TRPO, a surrogate maximized under a KL constraint justified by a bound on the improvement, and PPO, which replaces the constraint by clipping the probability ratio (chapter 20). |
| Abdolmaleki et al., [*Maximum a Posteriori Policy Optimisation*](https://arxiv.org/abs/1806.06920) (2018) | MPO: policy improvement as expectation maximization, with KL trust regions and a critic trained with Retrace (chapters 9 and 20). |
| Andrychowicz et al., [*What Matters in On-Policy Reinforcement Learning? A Large-Scale Empirical Study*](https://arxiv.org/abs/2006.05990) (2021) | More than 50 implementation choices of deep actor–critics measured in more than 250,000 training runs (chapters 19 and 20). |
| Huang et al., [*The 37 Implementation Details of Proximal Policy Optimization*](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) (2022) | The details of the reference PPO implementation that decide whether it learns (chapter 20). |
| Lillicrap et al., [*Continuous Control with Deep Reinforcement Learning*](https://arxiv.org/abs/1509.02971) (2016) | DDPG, DQN for continuous actions with a deterministic actor, and Polyak-averaged target networks (chapters 16 and 21). |
| Fujimoto, van Hoof, and Meger, [*Addressing Function Approximation Error in Actor-Critic Methods*](https://arxiv.org/abs/1802.09477) (2018) | TD3: clipped double Q-learning with the minimum of two critics, target policy smoothing, and delayed policy updates (chapters 7 and 21). |
| Haarnoja et al., [*Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*](https://arxiv.org/abs/1801.01290) (2018) | Soft actor–critic, an off-policy actor–critic for the maximum-entropy objective, whose temperature sets the price of randomness (chapter 21). |
| Levine, [*Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review*](https://arxiv.org/abs/1805.00909) (2018) | The maximum-entropy objective derived by casting control as probabilistic inference (chapter 21). |
| Chen et al., [*Randomized Ensembled Double Q-Learning: Learning Fast without a Model*](https://arxiv.org/abs/2101.05982) (2021) | REDQ: high update-to-data ratios made stable by an ensemble of critics, each target using the minimum over a random pair (chapter 21). |

## <a id="exploration"></a>Exploration

| Work | Idea developed in the module |
| --- | --- |
| Bellemare et al., [*Unifying Count-Based Exploration and Intrinsic Motivation*](https://arxiv.org/abs/1606.01868) (2016) | Pseudo-counts derived from a density model, which extend count-based bonuses to large state spaces (chapter 22). |
| Pathak et al., [*Curiosity-Driven Exploration by Self-Supervised Prediction*](https://arxiv.org/abs/1705.05363) (2017) | The intrinsic curiosity module, which rewards errors in predicting the next state's features, learned by an inverse dynamics model so that they ignore what the agent cannot affect (chapter 22). |
| Burda et al., [*Exploration by Random Network Distillation*](https://arxiv.org/abs/1810.12894) (2019) | Random network distillation: the bonus is the error of a network trained to predict the output of a fixed random network (chapter 22). |
| Badia et al., [*Never Give Up: Learning Directed Exploration Strategies*](https://arxiv.org/abs/2002.06038) (2020) | An episodic novelty bonus from nearest neighbors in a learned embedding, modulated by a lifelong RND bonus (chapters 18 and 22). |
| Osband, Russo, and Van Roy, [*(More) Efficient Reinforcement Learning via Posterior Sampling*](https://arxiv.org/abs/1306.0940) (2013) | Posterior sampling for RL: sample an MDP from the posterior at the start of each episode and follow its optimal policy, with a regret bound (chapters 22 and 30). |
| Osband et al., [*Deep Exploration via Randomized Value Functions*](https://arxiv.org/abs/1703.07608) (2019) | Randomized least-squares value iteration, which samples a value function by fitting it to perturbed rewards with a random prior (chapter 22). |
| Osband et al., [*Deep Exploration via Bootstrapped DQN*](https://arxiv.org/abs/1602.04621) (2016) | An ensemble of Q-heads on a shared torso, with one head sampled per episode to act greedily (chapter 22). |
| Ecoffet et al., [*First Return, Then Explore*](https://www.nature.com/articles/s41586-020-03157-9) (2021) | Go-Explore, which remembers promising states, returns to them, and explores from there (chapter 22). |
| Eysenbach et al., [*Diversity Is All You Need: Learning Skills without a Reward Function*](https://arxiv.org/abs/1802.06070) (2019) | Skills learned without reward by maximizing the mutual information between a skill variable and the states it reaches (chapter 22). |

## <a id="model-based-rl-and-world-models"></a>Model-based RL and world models

| Work | Idea developed in the module |
| --- | --- |
| Deisenroth and Rasmussen, [*PILCO: A Model-Based and Data-Efficient Approach to Policy Search*](https://dl.acm.org/doi/10.5555/3104482.3104541) (2011) | Gaussian-process dynamics with analytic propagation of uncertainty, which swung up a cart-pole from a few trials (chapter 23). |
| Chua et al., [*Deep Reinforcement Learning in a Handful of Trials Using Probabilistic Dynamics Models*](https://arxiv.org/abs/1805.12114) (2018) | PETS: an ensemble of probabilistic networks and planning by the cross-entropy method at every step (chapters 15 and 23). |
| Janner et al., [*When to Trust Your Model: Model-Based Policy Optimization*](https://arxiv.org/abs/1906.08253) (2019) | MBPO: short model rollouts branched from real states, feeding an off-policy learner (chapter 23). |
| Ha and Schmidhuber, [*World Models*](https://arxiv.org/abs/1803.10122) (2018) | A variational autoencoder and a recurrent model of the frames, with a small controller trained on their features (chapter 23). |
| Hafner et al., [*Learning Latent Dynamics for Planning from Pixels*](https://arxiv.org/abs/1811.04551) (2019), and [*Dream to Control: Learning Behaviors by Latent Imagination*](https://arxiv.org/abs/1912.01603) (2020) | The recurrent state-space model with planning in its latent space, and Dreamer's actor and critic trained on trajectories imagined in it (chapters 8, 14, 15, and 23). |
| Hafner et al., [*Mastering Diverse Control Tasks through World Models*](https://doi.org/10.1038/s41586-025-08744-2) (2025) | DreamerV3: one set of hyperparameters across domains, with symlog predictions, two-hot targets, and returns normalized by their percentiles (chapters 23 and 29). |
| Hansen, Wang, and Su, [*Temporal Difference Learning for Model Predictive Control*](https://arxiv.org/abs/2203.04955) (2022) | TD-MPC: planning in a value-equivalent latent model, with a learned value as the terminal cost (chapters 15 and 23). |
| Micheli, Alonso, and Fleuret, [*Transformers Are Sample-Efficient World Models*](https://arxiv.org/abs/2209.00588) (2023) | IRIS: frames tokenized by a discrete autoencoder, and the next tokens, reward, and termination predicted by an autoregressive transformer (chapter 23). |
| Bruce et al., [*Genie: Generative Interactive Environments*](https://arxiv.org/abs/2402.15391) (2024) | An interactive environment learned from unlabeled Internet videos, with a latent action model (chapters 23 and 29). |
| Hafner, Yan, and Lillicrap, [*Training Agents inside of Scalable World Models*](https://arxiv.org/abs/2509.24527) (2025) | Dreamer 4: an agent trained purely inside a real-time transformer world model, from offline data, that obtained diamonds in Minecraft (chapters 23 and 29). |

## <a id="planning-with-learned-models"></a>Planning with learned models

| Work | Idea developed in the module |
| --- | --- |
| Silver et al., [*Mastering the Game of Go with Deep Neural Networks and Tree Search*](https://doi.org/10.1038/nature16961) (2016) | AlphaGo: policy and value networks combined with Monte Carlo tree search, the first program to defeat a professional Go player on a full board (chapter 24). |
| Anthony, Tian, and Barber, [*Thinking Fast and Slow with Deep Learning and Tree Search*](https://arxiv.org/abs/1705.08439) (2017) | Expert iteration: the search is an expert, the network an apprentice that imitates it and in turn makes the expert stronger (chapter 24). |
| Silver et al., [*Mastering the Game of Go without Human Knowledge*](https://doi.org/10.1038/nature24270) (2017), and [*A General Reinforcement Learning Algorithm That Masters Chess, Shogi, and Go through Self-Play*](https://doi.org/10.1126/science.aar6404) (2018) | AlphaGo Zero, trained by self-play alone, without human data or rollouts, and AlphaZero, the same algorithm for chess, shogi, and Go (chapter 24). |
| Schrittwieser et al., [*Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model*](https://www.nature.com/articles/s41586-020-03051-4) (2020) | MuZero: search with a learned model that predicts only what the search needs, rewards, values, and policies (chapters 8 and 24). |
| Ye et al., [*Mastering Atari Games with Limited Data*](https://arxiv.org/abs/2111.00210) (2021) | EfficientZero: MuZero made data-efficient with a self-supervised consistency loss on its hidden states (chapter 24). |
| Grill et al., [*Monte-Carlo Tree Search as Regularized Policy Optimization*](https://arxiv.org/abs/2007.12509) (2020) | The visit distribution of PUCT approximately solves a regularized policy optimization problem at the root (chapter 24). |
| Danihelka et al., [*Policy Improvement by Planning with Gumbel*](https://openreview.net/forum?id=bERaNdoegnO) (2022) | Gumbel MuZero: a search redesigned at the root so that it improves the policy even with very few simulations (chapter 24). |
| Fawzi et al., [*Discovering Faster Matrix Multiplication Algorithms with Reinforcement Learning*](https://www.nature.com/articles/s41586-022-05172-4) (2022), and Mankowitz et al., [*Faster Sorting Algorithms Discovered Using Deep Reinforcement Learning*](https://www.nature.com/articles/s41586-023-06004-9) (2023) | Algorithm discovery as a single-player game, for matrix multiplication and for sorting routines merged into the LLVM C++ library (chapters 24 and 31). |
| Hubert et al., [*Olympiad-Level Formal Mathematical Reasoning with Reinforcement Learning*](https://www.nature.com/articles/s41586-025-09833-y) (2025) | AlphaProof: proof search in Lean, trained on millions of formalized problems and by RL on variants of each new problem at test time (chapters 24 and 28). |

## <a id="imitation-learning-and-inverse-rl"></a>Imitation learning and inverse RL

| Work | Idea developed in the module |
| --- | --- |
| Pomerleau, [*ALVINN: An Autonomous Land Vehicle in a Neural Network*](https://papers.nips.cc/paper_files/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html) (1988) | Behavior cloning of a driver's steering from camera images (chapter 25). |
| Ross, Gordon, and Bagnell, [*A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*](https://arxiv.org/abs/1011.0686) (2011) | DAgger, which counters the compounding errors of behavior cloning by aggregating the expert's labels on the states the learner itself visits (chapter 25). |
| Ng and Russell, [*Algorithms for Inverse Reinforcement Learning*](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf) (2000), and Abbeel and Ng, [*Apprenticeship Learning via Inverse Reinforcement Learning*](https://doi.org/10.1145/1015330.1015430) (2004) | Inverse RL and its ill-posedness, and apprenticeship learning, which finds a policy that matches the expert's feature expectations (chapter 25). |
| Ziebart et al., [*Maximum Entropy Inverse Reinforcement Learning*](https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf) (2008) | The distribution of maximum entropy among those that match the expert's feature counts (chapter 25). |
| Ho and Ermon, [*Generative Adversarial Imitation Learning*](https://arxiv.org/abs/1606.03476) (2016) | GAIL: the expert's behavior matched directly by a discriminator and a policy trained against it, without recovering a reward (chapter 25). |
| Fu, Luo, and Levine, [*Learning Robust Rewards with Adversarial Inverse Reinforcement Learning*](https://arxiv.org/abs/1710.11248) (2018) | AIRL, whose discriminator separates a reward term from a shaping term (chapter 25). |
| Reddy, Dragan, and Levine, [*SQIL: Imitation Learning via Reinforcement Learning with Sparse Rewards*](https://arxiv.org/abs/1905.11108) (2020) | Soft Q-learning with a fixed reward, 1 for every demonstration transition and 0 for the agent's own (chapter 25). |
| Chi et al., [*Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*](https://arxiv.org/abs/2303.04137) (2023), and Zhao et al., [*Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*](https://arxiv.org/abs/2304.13705) (2023) | Actions generated by a conditional diffusion model, and ACT, chunks of future actions predicted by a transformer trained as a conditional VAE (chapter 25). |
| Baker et al., [*Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos*](https://arxiv.org/abs/2206.11795) (2022) | An inverse dynamics model that labels 70,000 hours of Minecraft video for behavior cloning, fine-tuned by RL to craft diamond tools (chapters 25 and 29). |

## <a id="offline-rl"></a>Offline RL

| Work | Idea developed in the module |
| --- | --- |
| Levine et al., [*Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems*](https://arxiv.org/abs/2005.01643) (2020) | Learning a policy from a fixed data set collected by a behavior policy, without further interaction (chapter 26). |
| Fujimoto, Meger, and Precup, [*Off-Policy Deep Reinforcement Learning without Exploration*](https://arxiv.org/abs/1812.02900) (2019), and Fujimoto and Gu, [*A Minimalist Approach to Offline Reinforcement Learning*](https://arxiv.org/abs/2106.06860) (2021) | The failure of off-policy agents on fixed data, even expert data, and TD3+BC, a behavior cloning term added to the actor of TD3 (chapter 26). |
| Kumar et al., [*Conservative Q-Learning for Offline Reinforcement Learning*](https://arxiv.org/abs/2006.04779) (2020) | CQL: a Q-function made pessimistic where the data are silent (chapter 26). |
| Kostrikov, Nair, and Levine, [*Offline Reinforcement Learning with Implicit Q-Learning*](https://arxiv.org/abs/2110.06169) (2022) | IQL: expectile regression that never evaluates actions outside the data, with the policy extracted by advantage-weighted regression (chapter 26). |
| Yu et al., [*MOPO: Model-Based Offline Policy Optimization*](https://arxiv.org/abs/2005.13239) (2020) | Model rewards penalized by the ensemble's uncertainty, and a proof that the return under the penalized model lower-bounds the true return (chapter 26). |
| Jin, Yang, and Wang, [*Is Pessimism Provably Efficient for Offline RL?*](https://arxiv.org/abs/2012.15085) (2021) | Pessimism bounds the suboptimality by the uncertainty along the trajectories of the optimal policy only (chapters 26 and 30). |
| Chen et al., [*Decision Transformer: Reinforcement Learning via Sequence Modeling*](https://arxiv.org/abs/2106.01345) (2021) | A transformer that predicts actions from states, actions, and returns-to-go, conditioned on a high return at test time (chapter 26). |
| Brandfonbrener et al., [*When Does Return-Conditioned Supervised Learning Work for Offline Reinforcement Learning?*](https://arxiv.org/abs/2206.01079) (2022) | Return-conditioned supervised learning avoids bootstrapping but cannot stitch trajectories, and fails where high returns reflect luck (chapter 26). |
| Ball et al., [*Efficient Online Reinforcement Learning with Offline Data*](https://arxiv.org/abs/2302.02948) (2023) | RLPD: online SAC with layer-normalized critic ensembles, a high update-to-data ratio, and half of each minibatch drawn from the offline data (chapter 26). |
| Park, Li, and Levine, [*Flow Q-Learning*](https://arxiv.org/abs/2502.02538) (2025) | A flow-matching policy distilled into a one-step policy that maximizes the Q-function while staying close to the flow (chapter 26). |
| Fu et al., [*D4RL: Datasets for Deep Data-Driven Reinforcement Learning*](https://arxiv.org/abs/2004.07219) (2020) | Offline data sets of varying quality, including navigation mazes that require stitching (chapter 26). |

## <a id="multi-agent-rl-and-self-play"></a>Multi-agent RL and self-play

| Work | Idea developed in the module |
| --- | --- |
| Shapley, [*Stochastic Games*](https://doi.org/10.1073/pnas.39.10.1095) (1953), and Littman, [*Markov Games as a Framework for Multi-Agent Reinforcement Learning*](https://doi.org/10.1016/B978-1-55860-335-6.50027-1) (1994) | The stochastic game, or Markov game, and minimax-Q, whose target is the value of the matrix game of Q-values at the next state (chapter 27). |
| Zinkevich et al., [*Regret Minimization in Games with Incomplete Information*](https://proceedings.neurips.cc/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html) (2007) | Counterfactual regret minimization, a small regret minimizer at every information set (chapter 27). |
| Heinrich and Silver, [*Deep Reinforcement Learning from Self-Play in Imperfect-Information Games*](https://arxiv.org/abs/1603.01121) (2016) | Neural fictitious self-play: a DQN best response to the others' average policies, and a network that imitates the agent's own past best responses (chapter 27). |
| Moravčík et al., [*DeepStack: Expert-Level Artificial Intelligence in Heads-Up No-Limit Poker*](https://doi.org/10.1126/science.aam6960) (2017) | Depth-limited re-solving at every decision with networks for counterfactual values, which beat professionals at heads-up no-limit hold'em (chapter 27). |
| Brown and Sandholm, [*Superhuman AI for Heads-Up No-Limit Poker: Libratus Beats Top Professionals*](https://doi.org/10.1126/science.aao1733) (2018), and [*Superhuman AI for Multiplayer Poker*](https://doi.org/10.1126/science.aay2400) (2019) | Libratus's blueprint strategy refined by safe subgame solving, and Pluribus at six-player no-limit hold'em (chapter 27). |
| Brown et al., [*Combining Deep Reinforcement Learning and Search for Imperfect-Information Games*](https://arxiv.org/abs/2007.13544) (2020) | ReBeL: self-play and search over public belief states, the recipe of AlphaZero extended to imperfect information (chapter 27). |
| Perolat et al., [*Mastering the Game of Stratego with Model-Free Multiagent Reinforcement Learning*](https://doi.org/10.1126/science.add4679) (2022) | DeepNash: Stratego learned by model-free self-play with regularized Nash dynamics (chapter 27). |
| Lanctot et al., [*A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning*](https://arxiv.org/abs/1711.00832) (2017) | Joint-policy correlation, the overfitting of independent learners to each other, and policy-space response oracles (chapter 27). |
| Vinyals et al., [*Grandmaster Level in StarCraft II Using Multi-Agent Reinforcement Learning*](https://doi.org/10.1038/s41586-019-1724-z) (2019) | AlphaStar: an IMPALA-style learner with V-trace and a league of self-play agents, at grandmaster level (chapters 19 and 27). |
| Lowe et al., [*Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments*](https://arxiv.org/abs/1706.02275) (2017) | MADDPG: decentralized actors with centralized critics that see the state and all actions (chapter 27). |
| Rashid et al., [*QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning*](https://arxiv.org/abs/1803.11485) (2018) | The team's action value as a monotonic mixing of per-agent utilities, by a network whose weights are generated from the state (chapter 27). |
| Meta FAIR Diplomacy Team et al., [*Human-Level Play in the Game of Diplomacy by Combining Language Models with Strategic Reasoning*](https://doi.org/10.1126/science.ade9097) (2022) | Cicero: a strategic planner regularized toward human play, and a language model that negotiates (chapter 27). |

## <a id="rl-for-language-models-and-reasoning"></a>RL for language models and reasoning

| Work | Idea developed in the module |
| --- | --- |
| Christiano et al., [*Deep Reinforcement Learning from Human Preferences*](https://arxiv.org/abs/1706.03741) (2017) | A reward model fitted to human choices between pairs of clips with the Bradley–Terry model, and optimized by RL (chapter 25). |
| Stiennon et al., [*Learning to Summarize from Human Feedback*](https://arxiv.org/abs/2009.01325) (2020) | One of the first RLHF systems, which fine-tuned a language model with PPO against a reward model of human preferences (chapter 28). |
| Ouyang et al., [*Training Language Models to Follow Instructions with Human Feedback*](https://arxiv.org/abs/2203.02155) (2022) | RLHF with PPO and a KL penalty toward the supervised model, for instruction following (chapters 20 and 28). |
| Gao, Schulman, and Hilton, [*Scaling Laws for Reward Model Overoptimization*](https://arxiv.org/abs/2210.10760) (2023) | Overoptimization of a learned reward, measured against a large gold reward model (chapter 28). |
| Ahmadian et al., [*Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs*](https://arxiv.org/abs/2402.14740) (2024) | RLOO: REINFORCE with a leave-one-out baseline, which matches or beats PPO for RLHF at a fraction of its complexity (chapters 13 and 28). |
| Shao et al., [*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*](https://arxiv.org/abs/2402.03300) (2024) | GRPO: PPO's clipped surrogate with the critic replaced by statistics of a group of responses to the same prompt (chapter 28). |
| Liu et al., [*Understanding R1-Zero-Like Training: A Critical Perspective*](https://arxiv.org/abs/2503.20783) (2025), and Yu et al., [*DAPO: An Open-Source LLM Reinforcement Learning System at Scale*](https://arxiv.org/abs/2503.14476) (2025) | The length bias of GRPO's per-response normalization, removed in Dr. GRPO, and DAPO's token-level loss with overlong reward shaping (chapter 28). |
| Cui et al., [*The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models*](https://arxiv.org/abs/2505.22617) (2025) | The entropy of the policy changes with the covariance between an action's log-probability and its advantage, which explains why the entropy collapses (chapter 28). |
| Yue et al., [*Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?*](https://arxiv.org/abs/2504.13837) (2025) | RL with verifiable rewards beats the base model at pass@1, but for large $`k`$ the base model solves as many problems or more (chapter 28). |
| Lightman et al., [*Let's Verify Step by Step*](https://arxiv.org/abs/2305.20050) (2024) | A process reward model trained on 800,000 human labels of reasoning steps, well above an outcome reward model for choosing among sampled solutions (chapter 28). |
| DeepSeek-AI, [*DeepSeek-R1 Incentivizes Reasoning in LLMs through Reinforcement Learning*](https://doi.org/10.1038/s41586-025-09422-z) (2025) | Reasoning trained by large-scale RL with outcome rewards, after a process reward model had been exploited by the policy (chapter 28). |

## <a id="generalist-agents-meta-rl-and-open-endedness"></a>Generalist agents, meta-RL, and open-endedness

| Work | Idea developed in the module |
| --- | --- |
| Duan et al., [*RL²: Fast Reinforcement Learning via Slow Reinforcement Learning*](https://arxiv.org/abs/1611.02779) (2016), and Wang et al., [*Learning to Reinforcement Learn*](https://arxiv.org/abs/1611.05763) (2016) | A recurrent policy trained across a distribution of tasks, whose activations implement a fast learning algorithm (chapter 29). |
| Finn, Abbeel, and Levine, [*Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks*](https://arxiv.org/abs/1703.03400) (2017) | MAML: an initialization from which a few policy-gradient steps on a new task give a good policy (chapter 29). |
| Rakelly et al., [*Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables*](https://arxiv.org/abs/1903.08254) (2019) | PEARL: a posterior over a latent task variable, inferred from the transitions so far and sampled to condition a soft actor–critic (chapter 29). |
| Laskin et al., [*In-Context Reinforcement Learning with Algorithm Distillation*](https://arxiv.org/abs/2210.14215) (2023) | A transformer trained on the learning histories of an RL algorithm, which improves in context on new tasks (chapter 29). |
| Schaul et al., [*Universal Value Function Approximators*](https://proceedings.mlr.press/v37/schaul15.html) (2015) | Values and policies conditioned on a goal, so that one network generalizes across goals (chapter 29). |
| Reed et al., [*A Generalist Agent*](https://arxiv.org/abs/2205.06175) (2022) | Gato: one transformer trained by supervised learning on 604 tasks (chapter 29). |
| Brohan et al., [*RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*](https://arxiv.org/abs/2307.15818) (2023) | A vision–language model fine-tuned to output actions as tokens, which transfers knowledge from the web to robot control (chapters 25 and 29). |
| Black et al., [*π0: A Vision-Language-Action Flow Model for General Robot Control*](https://arxiv.org/abs/2410.24164) (2025) | A flow-matching action expert added to a pretrained vision–language model (chapters 25 and 29). |
| Wang et al., [*Paired Open-Ended Trailblazer (POET): Endlessly Generating Increasingly Complex and Diverse Learning Environments and Their Solutions*](https://arxiv.org/abs/1901.01753) (2019) | A population of environments coevolved with a population of agents, each new environment neither too easy nor too hard (chapter 29). |
| Dennis et al., [*Emergent Complexity and Zero-Shot Transfer via Unsupervised Environment Design*](https://arxiv.org/abs/2012.02096) (2020), and Jiang, Grefenstette, and Rocktäschel, [*Prioritized Level Replay*](https://arxiv.org/abs/2010.03934) (2021) | Levels designed by an adversary that maximizes the agent's regret, and levels replayed in proportion to their value-prediction error (chapter 29). |
| Hughes et al., [*Open-Endedness Is Essential for Artificial Superhuman Intelligence*](https://arxiv.org/abs/2406.04268) (2024) | Open-endedness, the study of processes that keep producing new and more complex things with no fixed objective (chapter 29). |

## <a id="theory"></a>Theory

| Work | Idea developed in the module |
| --- | --- |
| Kearns and Singh, [*Near-Optimal Reinforcement Learning in Polynomial Time*](https://doi.org/10.1023/A:1017984413808) (2002), and Brafman and Tennenholtz, [*R-MAX – A General Polynomial Time Algorithm for Near-Optimal Reinforcement Learning*](https://www.jmlr.org/papers/v3/brafman02a.html) (2002) | Polynomial sample complexity, by setting apart the pairs not yet visited enough and either exploring them explicitly or treating them as maximally rewarding (chapters 22 and 30). |
| Jaksch, Ortner, and Auer, [*Near-Optimal Regret Bounds for Reinforcement Learning*](https://jmlr.org/papers/v11/jaksch10a.html) (2010), and Azar, Osband, and Munos, [*Minimax Regret Bounds for Reinforcement Learning*](https://arxiv.org/abs/1703.05449) (2017) | UCRL2, which plans in the most optimistic MDP within confidence sets, and UCBVI, value iteration with a bonus added to the rewards (chapter 30). |
| Azar, Munos, and Kappen, [*Minimax PAC Bounds on the Sample Complexity of Reinforcement Learning with a Generative Model*](https://doi.org/10.1007/s10994-013-5368-1) (2013) | With a generative model, $`\tilde O\bigl(SA/((1-\gamma)^3\varepsilon^2)\bigr)`$ samples suffice to estimate $`Q^*`$, and this many are necessary (chapter 30). |
| Li et al., [*Is Q-Learning Minimax Optimal? A Tight Sample Complexity Analysis*](https://arxiv.org/abs/2102.06548) (2024) | Q-learning with its usual step sizes needs a factor of the horizon more samples than the minimax rate (chapters 7 and 30). |
| Jin et al., [*Is Q-Learning Provably Efficient?*](https://arxiv.org/abs/1807.03765) (2018) | Q-learning with UCB bonuses and the step size $`(H+1)/(H+n)`$, within a factor $`\sqrt H`$ of the lower bound on regret (chapters 7 and 30). |
| Jin et al., [*Provably Efficient Reinforcement Learning with Linear Function Approximation*](https://arxiv.org/abs/1907.05388) (2020) | The linear MDP, in which transitions and rewards are linear in known features, and least-squares value iteration with the elliptical bonus of LinUCB (chapter 30). |
| Du et al., [*Is a Good Representation Sufficient for Sample Efficient Reinforcement Learning?*](https://arxiv.org/abs/1910.03016) (2020), and Weisz, Amortila, and Szepesvári, [*Exponential Lower Bounds for Planning in MDPs with Linearly-Realizable Optimal Action-Value Functions*](https://arxiv.org/abs/2010.01374) (2021) | Exponential lower bounds when the action values are only approximately linear, and even when $`Q^*`$ is exactly linear, with a generative model (chapter 30). |
| Jin, Liu, and Miryoosefi, [*Bellman Eluder Dimension: New Rich Classes of RL Problems, and Sample-Efficient Algorithms*](https://arxiv.org/abs/2102.00815) (2021) | The Bellman eluder dimension, which combines the eluder dimension with Bellman rank (chapter 30). |
| Foster et al., [*The Statistical Complexity of Interactive Decision Making*](https://arxiv.org/abs/2112.13487) (2021) | The decision–estimation coefficient, which lower-bounds the regret of any algorithm for any model class and upper-bounds that of one algorithm (chapter 30). |
| Agarwal et al., [*On the Theory of Policy Gradient Methods: Optimality, Approximation, and Distribution Shift*](https://arxiv.org/abs/1908.00261) (2021) | Global convergence of softmax policy gradient in tabular problems, and the natural policy gradient's rate $`O(1/k)`$, independent of the numbers of states and actions (chapters 13, 20, and 30). |
| Lan, [*Policy Mirror Descent for Reinforcement Learning: Linear Convergence, New Sampling Complexity, and Generalized Problem Classes*](https://arxiv.org/abs/2102.00135) (2023) | Policy mirror descent, which converges linearly with strongly convex regularizers (chapters 20 and 30). |

## <a id="rl-in-the-real-world"></a>RL in the real world

| Work | Idea developed in the module |
| --- | --- |
| Ng, Harada, and Russell, [*Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping*](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) (1999) | Potential-based shaping, which leaves the optimal policy unchanged while it can make learning much faster (chapters 1 and 31). |
| Skalse et al., [*Defining and Characterizing Reward Hacking*](https://arxiv.org/abs/2209.13085) (2022) | Unhackable proxy rewards, and the result that over all stochastic policies two rewards are unhackable with respect to each other only if one of them is constant (chapter 31). |
| Andrychowicz et al., [*Hindsight Experience Replay*](https://arxiv.org/abs/1707.01495) (2017) | Hindsight experience replay, which relabels trajectories with the goals they actually reached (chapter 31). |
| Sutton, Precup, and Singh, [*Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning*](https://doi.org/10.1016/S0004-3702(99)00052-1) (1999) | Options, temporally extended actions with their own policies and termination conditions (chapter 31). |
| Achiam et al., [*Constrained Policy Optimization*](https://arxiv.org/abs/1705.10528) (2017) | CPO: a linearized cost constraint added to each trust-region step (chapter 31). |
| Tobin et al., [*Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*](https://arxiv.org/abs/1703.06907) (2017), and Peng et al., [*Sim-to-Real Transfer of Robotic Control with Dynamics Randomization*](https://arxiv.org/abs/1710.06537) (2018) | Simulator properties sampled anew in every episode: textures, lighting, and camera positions for perception, and 95 dynamical parameters for control (chapter 31). |
| OpenAI et al., [*Solving Rubik's Cube with a Robot Hand*](https://arxiv.org/abs/1910.07113) (2019) | Automatic domain randomization, which widens the range of each parameter whenever the policy performs well at its edge (chapter 31). |
| Kumar et al., [*RMA: Rapid Motor Adaptation for Legged Robots*](https://arxiv.org/abs/2107.04034) (2021) | Rapid motor adaptation: a policy that uses privileged environment parameters in simulation, and a module that infers them from recent history (chapter 31). |
| Kalashnikov et al., [*QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation*](https://arxiv.org/abs/1806.10293) (2018) | Vision-based grasping learned by distributed Q-learning, maximizing over actions with the cross-entropy method, from 580,000 real grasp attempts (chapters 16 and 31). |
| Bellemare et al., [*Autonomous Navigation of Stratospheric Balloons Using Reinforcement Learning*](https://doi.org/10.1038/s41586-020-2939-8) (2020) | Station-keeping of Loon's balloons by a distributional Q-learning agent trained in a simulator built from historical wind data (chapters 17 and 31). |
| Degrave et al., [*Magnetic Control of Tokamak Plasmas through Deep Reinforcement Learning*](https://doi.org/10.1038/s41586-021-04301-9) (2022) | Control of all magnetic coils of a tokamak by MPO, with a large critic used only in training and a small actor fast enough for real time (chapter 31). |
| Wurman et al., [*Outracing Champion Gran Turismo Drivers with Deep Reinforcement Learning*](https://doi.org/10.1038/s41586-021-04357-7) (2022) | GT Sophy: a distributional soft actor–critic with rewards that encode racing etiquette, which won head-to-head races against champion drivers (chapters 17, 27, and 31). |
