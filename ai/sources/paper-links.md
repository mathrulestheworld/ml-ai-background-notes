[ML Mastery Notes](../../README.md) › [Artificial Intelligence](../README.md)

# Paper links

# <a id="papers"></a>Papers

The notes cite the original papers where their ideas arise. This index collects the principal ones by topic; the chapters give further references inline.

## <a id="search-constraint-satisfaction-and-games"></a>Search, constraint satisfaction, and games

| Work | Idea developed in the module |
| --- | --- |
| Hart, Nilsson, and Raphael, [*A Formal Basis for the Heuristic Determination of Minimum Cost Paths*](https://doi.org/10.1109/TSSC.1968.300136) (1968) | A\* and its optimality with admissible heuristics (chapter 2). |
| Dechter and Pearl, [*Generalized Best-First Search Strategies and the Optimality of A\**](https://doi.org/10.1145/3828.3830) (1985) | The sense in which A\* is optimally efficient (chapter 2). |
| Korf, [*Depth-First Iterative-Deepening: An Optimal Admissible Tree Search*](https://www.sciencedirect.com/science/article/pii/0004370285900840) (1985) | Iterative deepening and IDA\* (chapters 1–2). |
| Culberson and Schaeffer, [*Pattern Databases*](https://doi.org/10.1111/0824-7935.00065) (1998), and Felner, Korf, and Hanan, [*Additive Pattern Database Heuristics*](https://doi.org/10.1613/jair.1480) (2004) | Abstraction heuristics stored in tables, and when they can be added (chapter 2). |
| Likhachev, Gordon, and Thrun, [*ARA\*: Anytime A\* with Provable Bounds on Sub-Optimality*](https://proceedings.neurips.cc/paper/2003/hash/ee8fe9093fbbb687bef15a38facc44d2-Abstract.html) (2003) | Weighted A\* without reopening and anytime search (chapter 2). |
| Agostinelli, McAleer, Shmakov, and Baldi, [*Solving the Rubik's Cube with Deep Reinforcement Learning and Search*](https://doi.org/10.1038/s42256-019-0070-z) (2019) | A learned cost-to-go heuristic inside weighted A\* (chapter 2). |
| Minton, Johnston, Philips, and Laird, [*Minimizing Conflicts: A Heuristic Repair Method for Constraint Satisfaction and Scheduling Problems*](https://www.sciencedirect.com/science/article/pii/000437029290007K) (1992) | The min-conflicts heuristic (chapter 3). |
| Kirkpatrick, Gelatt, and Vecchi, [*Optimization by Simulated Annealing*](https://doi.org/10.1126/science.220.4598.671) (1983), and Hajek, [*Cooling Schedules for Optimal Annealing*](https://doi.org/10.1287/moor.13.2.311) (1988) | Simulated annealing and its logarithmic schedule (chapter 3). |
| Cheeseman, Kanefsky, and Taylor, [*Where the Really Hard Problems Are*](https://www.ijcai.org/Proceedings/91-1/Papers/052.pdf) (1991) | Hard instances cluster at phase transitions (chapters 3 and 5). |
| Knuth and Moore, [*An Analysis of Alpha-Beta Pruning*](https://www.sciencedirect.com/science/article/pii/0004370275900193) (1975) | Correctness and the best case of alpha–beta (chapter 4). |
| Campbell, Hoane, and Hsu, [*Deep Blue*](https://www.sciencedirect.com/science/article/pii/S0004370201001291) (2002), and Schaeffer et al., [*Checkers Is Solved*](https://doi.org/10.1126/science.1144079) (2007) | Game-tree search at scale (chapter 4). |
| Auer, Cesa-Bianchi, and Fischer, [*Finite-Time Analysis of the Multiarmed Bandit Problem*](https://doi.org/10.1023/A:1013689704352) (2002), and Kocsis and Szepesvári, [*Bandit Based Monte-Carlo Planning*](https://doi.org/10.1007/11871842_29) (2006) | UCB1 and UCT (chapter 4). |
| Silver et al., [*Mastering the Game of Go with Deep Neural Networks and Tree Search*](https://doi.org/10.1038/nature16961) (2016), and [*A General Reinforcement Learning Algorithm That Masters Chess, Shogi, and Go through Self-Play*](https://doi.org/10.1126/science.aar6404) (2018) | AlphaGo and AlphaZero (chapter 4). |
| Brown and Sandholm, [*Superhuman AI for Heads-Up No-Limit Poker: Libratus Beats Top Professionals*](https://doi.org/10.1126/science.aao1733) (2018), and Zinkevich et al., [*Regret Minimization in Games with Incomplete Information*](https://proceedings.neurips.cc/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html) (2007) | Counterfactual regret minimization and imperfect-information games (chapters 4 and 15). |

## <a id="logic-and-planning"></a>Logic and planning

| Work | Idea developed in the module |
| --- | --- |
| Cook, [*The Complexity of Theorem-Proving Procedures*](https://doi.org/10.1145/800157.805047) (1971) | NP-completeness of satisfiability (chapter 5). |
| Davis, Logemann, and Loveland, [*A Machine Program for Theorem-Proving*](https://doi.org/10.1145/368273.368557) (1962) | The DPLL procedure (chapter 5). |
| Marques-Silva and Sakallah, [*GRASP: A Search Algorithm for Propositional Satisfiability*](https://doi.org/10.1109/12.769433) (1999), and Moskewicz et al., [*Chaff: Engineering an Efficient SAT Solver*](https://doi.org/10.1145/378239.379017) (2001) | Conflict-driven clause learning, watched literals, and VSIDS (chapter 5). |
| Selman, Kautz, and Cohen, [*Noise Strategies for Improving Local Search*](https://cdn.aaai.org/AAAI/1994/AAAI94-051.pdf) (1994) | WalkSAT (chapter 5). |
| Mitchell, Selman, and Levesque, [*Hard and Easy Distributions of SAT Problems*](https://cdn.aaai.org/AAAI/1992/AAAI92-071.pdf) (1992), and Mézard, Parisi, and Zecchina, [*Analytic and Algorithmic Solution of Random Satisfiability Problems*](https://doi.org/10.1126/science.1073287) (2002) | The random 3-SAT threshold (chapter 5). |
| Haken, [*The Intractability of Resolution*](https://www.sciencedirect.com/science/article/pii/0304397585901446) (1985) | Exponential lower bounds for resolution proofs (chapter 5). |
| Robinson, [*A Machine-Oriented Logic Based on the Resolution Principle*](https://doi.org/10.1145/321250.321253) (1965) | Resolution with unification (chapter 6). |
| Church, [*A Note on the Entscheidungsproblem*](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/9461BEAD94BB16D56EC78933D7D67DEF/S0022481200038664a.pdf/a-note-on-the-entscheidungsproblem.pdf) (1936), and Turing, [*On Computable Numbers*](https://doi.org/10.1112/plms/s2-42.1.230) (1936) | Undecidability of first-order validity (chapter 6). |
| McCarthy and Hayes, [*Some Philosophical Problems from the Standpoint of Artificial Intelligence*](https://www-formal.stanford.edu/jmc/mcchay69.pdf) (1969) | The situation calculus and the frame problem (chapters 5–7). |
| McCarthy, [*Circumscription*](https://www.sciencedirect.com/science/article/pii/0004370280900119) (1980), and Reiter, [*A Logic for Default Reasoning*](https://www.sciencedirect.com/science/article/pii/0004370280900144) (1980) | Nonmonotonic reasoning (chapter 6). |
| Fikes and Nilsson, [*STRIPS*](https://www.sciencedirect.com/science/article/pii/0004370271900105) (1971), and Bylander, [*The Computational Complexity of Propositional STRIPS Planning*](https://www.sciencedirect.com/science/article/pii/0004370294900817) (1994) | The STRIPS formalism and its PSPACE-completeness (chapter 7). |
| Blum and Furst, [*Fast Planning through Planning Graph Analysis*](https://www.sciencedirect.com/science/article/pii/S0004370296000471) (1997), and Kautz and Selman, [*Pushing the Envelope*](https://aaai.org/papers/177-aaai96-177-pushing-the-envelope-planning-propositional-logic-and-stochastic-search/) (1996) | Graphplan and planning as satisfiability (chapter 7). |
| Bonet and Geffner, [*Planning as Heuristic Search*](https://www.sciencedirect.com/science/article/pii/S0004370201001084) (2001), and Hoffmann and Nebel, [*The FF Planning System*](https://doi.org/10.1613/jair.855) (2001) | Delete-relaxation heuristics and relaxed plans (chapter 7). |
| Richter and Westphal, [*The LAMA Planner*](https://doi.org/10.1613/jair.2972) (2010) | Landmarks in satisficing planning (chapter 7). |
| Nau et al., [*SHOP2: An HTN Planning System*](https://doi.org/10.1613/jair.1141) (2003) | Hierarchical task network planning (chapter 7). |
| Valmeekam et al., [*On the Planning Abilities of Large Language Models*](https://arxiv.org/abs/2305.15771) (2023) | What language models do and do not do on planning benchmarks (chapter 7). |

## <a id="graphical-models-and-inference"></a>Graphical models and inference

| Work | Idea developed in the module |
| --- | --- |
| Pearl, [*Probabilistic Reasoning in Intelligent Systems*](https://www.sciencedirect.com/book/monograph/9780080514895/probabilistic-reasoning-in-intelligent-systems) (1988) | Bayesian networks, d-separation, and belief propagation (chapters 8–9). |
| Geiger, Verma, and Pearl, [*Identifying Independence in Bayesian Networks*](https://doi.org/10.1002/net.3230200504) (1990), and Lauritzen et al., [*Independence Properties of Directed Markov Fields*](https://doi.org/10.1002/net.3230200503) (1990) | Soundness and completeness of d-separation, and the moral-graph criterion (chapter 8). |
| Verma and Pearl, [*Equivalence and Synthesis of Causal Models*](https://arxiv.org/abs/1304.1108) (1990) | Markov equivalence: same skeleton and v-structures (chapters 8 and 13). |
| Lauritzen and Spiegelhalter, [*Local Computations with Probabilities on Graphical Structures*](https://doi.org/10.1111/j.2517-6161.1988.tb01721.x) (1988) | The junction tree algorithm and the Asia network (chapters 8–9). |
| Cooper, [*The Computational Complexity of Probabilistic Inference Using Bayesian Belief Networks*](https://www.sciencedirect.com/science/article/pii/000437029090060D) (1990), and Dagum and Luby, [*Approximating Probabilistic Inference in Bayesian Belief Networks Is NP-Hard*](https://www.sciencedirect.com/science/article/pii/000437029390036B) (1993) | Hardness of exact and approximate inference (chapter 9). |
| Kschischang, Frey, and Loeliger, [*Factor Graphs and the Sum-Product Algorithm*](https://doi.org/10.1109/18.910572) (2001) | Message passing on factor graphs (chapters 8–9). |
| Darwiche, [*A Differential Approach to Inference in Bayesian Networks*](https://doi.org/10.1145/765568.765570) (2003) | Arithmetic circuits and knowledge compilation (chapter 9). |
| Metropolis et al., [*Equation of State Calculations by Fast Computing Machines*](https://doi.org/10.1063/1.1699114) (1953), and Hastings, [*Monte Carlo Sampling Methods Using Markov Chains and Their Applications*](https://doi.org/10.1093/biomet/57.1.97) (1970) | The Metropolis–Hastings algorithm (chapters 3 and 10). |
| Geman and Geman, [*Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images*](https://doi.org/10.1109/TPAMI.1984.4767596) (1984) | Gibbs sampling (chapter 10). |
| Neal, [*MCMC Using Hamiltonian Dynamics*](https://arxiv.org/abs/1206.1901) (2011) | Hamiltonian Monte Carlo (chapter 10). |
| Murphy, Weiss, and Jordan, [*Loopy Belief Propagation for Approximate Inference: An Empirical Study*](https://arxiv.org/abs/1301.6725) (1999), and Yedidia, Freeman, and Weiss, [*Constructing Free-Energy Approximations and Generalized Belief Propagation Algorithms*](https://doi.org/10.1109/TIT.2005.850085) (2005) | Loopy belief propagation and the Bethe free energy (chapter 10). |
| Blei, Kucukelbir, and McAuliffe, [*Variational Inference: A Review for Statisticians*](https://doi.org/10.1080/01621459.2017.1285773) (2017) | Mean-field and stochastic variational inference (chapter 10). |
| Kalman, [*A New Approach to Linear Filtering and Prediction Problems*](https://doi.org/10.1115/1.3662552) (1960) | The Kalman filter (chapter 11). |
| Viterbi, [*Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding Algorithm*](https://doi.org/10.1109/TIT.1967.1054010) (1967), and Baum et al., [*A Maximization Technique Occurring in the Statistical Analysis of Probabilistic Functions of Markov Chains*](https://doi.org/10.1214/aoms/1177697196) (1970) | Viterbi decoding and Baum–Welch learning (chapter 11). |
| Gordon, Salmond, and Smith, [*Novel Approach to Nonlinear/Non-Gaussian Bayesian State Estimation*](https://doi.org/10.1049/ip-f-2.1993.0015) (1993) | The bootstrap particle filter (chapter 11). |
| Besag, [*Statistical Analysis of Non-Lattice Data*](https://doi.org/10.2307/2987782) (1975) | Pseudolikelihood (chapter 14). |
| Lafferty, McCallum, and Pereira, [*Conditional Random Fields*](https://repository.upenn.edu/entities/publication/c9aea099-b5c8-4fdd-901c-15b6f889e4a7) (2001) | Conditional random fields for sequence labeling (chapters 8 and 14). |
| Chow and Liu, [*Approximating Discrete Probability Distributions with Dependence Trees*](https://doi.org/10.1109/TIT.1968.1054142) (1968) | Optimal tree-structured models (chapter 14). |

## <a id="decisions-causality-and-multiagent-systems"></a>Decisions, causality, and multiagent systems

| Work | Idea developed in the module |
| --- | --- |
| Howard, [*Information Value Theory*](https://doi.org/10.1109/TSSC.1966.300074) (1966), and Howard and Matheson, [*Influence Diagrams*](https://doi.org/10.1287/deca.1050.0020) (1984, reprinted 2005) | The value of information and decision networks (chapter 12). |
| Kahneman and Tversky, [*Prospect Theory*](https://doi.org/10.2307/1914185) (1979) | Systematic departures of human choice from expected utility (chapter 12). |
| Smith and Winkler, [*The Optimizer's Curse*](https://doi.org/10.1287/mnsc.1050.0451) (2006) | Post-decision disappointment from choosing by noisy estimates (chapter 12). |
| Rubin, [*Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies*](https://doi.org/10.1037/h0037350) (1974) | Potential outcomes (chapter 13). |
| Pearl, [*Causal Diagrams for Empirical Research*](https://doi.org/10.1093/biomet/82.4.669) (1995), and Shpitser and Pearl, [*Identification of Joint Interventional Distributions*](https://escholarship.org/uc/item/9598x714) (2006) | The do-calculus and its completeness (chapter 13). |
| Rosenbaum and Rubin, [*The Central Role of the Propensity Score*](https://doi.org/10.1093/biomet/70.1.41) (1983), and Robins, Rotnitzky, and Zhao, [*Estimation of Regression Coefficients When Some Regressors Are Not Always Observed*](https://doi.org/10.1080/01621459.1994.10476818) (1994) | Propensity scores and doubly robust estimation (chapter 13). |
| Angrist, Imbens, and Rubin, [*Identification of Causal Effects Using Instrumental Variables*](https://doi.org/10.1080/01621459.1996.10476902) (1996) | Instruments and the local average treatment effect (chapter 13). |
| Shimizu et al., [*A Linear Non-Gaussian Acyclic Model for Causal Discovery*](https://www.jmlr.org/papers/v7/shimizu06a.html) (2006), and Peters, Bühlmann, and Meinshausen, [*Causal Inference by Using Invariant Prediction*](https://doi.org/10.1111/rssb.12167) (2016) | Identifiability beyond Markov equivalence and invariance across environments (chapter 13). |
| Schölkopf et al., [*Toward Causal Representation Learning*](https://doi.org/10.1109/JPROC.2021.3058954) (2021) | Causality and representation learning (chapter 13). |
| Nash, [*Equilibrium Points in n-Person Games*](https://doi.org/10.1073/pnas.36.1.48) (1950) | Existence of mixed equilibria (chapter 15). |
| Aumann, [*Subjectivity and Correlation in Randomized Strategies*](https://www.sciencedirect.com/science/article/pii/0304406874900378) (1974) | Correlated equilibrium (chapter 15). |
| Daskalakis, Goldberg, and Papadimitriou, [*The Complexity of Computing a Nash Equilibrium*](https://doi.org/10.1137/070699652) (2009), and Chen, Deng, and Teng, [*Settling the Complexity of Computing Two-Player Nash Equilibria*](https://doi.org/10.1145/1516512.1516516) (2009) | PPAD-completeness of equilibrium computation (chapter 15). |
| Freund and Schapire, [*Adaptive Game Playing Using Multiplicative Weights*](https://doi.org/10.1006/game.1999.0738) (1999), and Hart and Mas-Colell, [*A Simple Adaptive Procedure Leading to Correlated Equilibrium*](https://doi.org/10.1111/1468-0262.00153) (2000) | No-regret learning and convergence to equilibria (chapter 15). |
| Roughgarden and Tardos, [*How Bad Is Selfish Routing?*](https://doi.org/10.1145/506147.506153) (2002) | The price of anarchy (chapter 15). |
| Vickrey, [*Counterspeculation, Auctions, and Competitive Sealed Tenders*](https://doi.org/10.1111/j.1540-6261.1961.tb02789.x) (1961) | The second-price auction and truthful mechanisms (chapter 15). |
