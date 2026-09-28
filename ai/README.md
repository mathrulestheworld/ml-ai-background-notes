[Background Notes](../README.md)

# Artificial Intelligence

Artificial Intelligence develops the methods by which an agent decides what to do when the answer is not learned end to end: searching through possibilities, reasoning with logic, representing and updating uncertain beliefs, weighing outcomes by their utility, reasoning about causes and interventions, and interacting with other agents. It builds on the probability, statistics, and information theory of Foundations, on the generative models and EM of Machine Learning, and, where heuristics and evaluation functions are learned, on the networks of Deep Learning. Thirteen core chapters follow the main topics of the reading plan, and two optional chapters cover its extensions.

## <a id="chapters"></a>Chapters

| Chapter | Main content | Reading plan |
| --- | --- | --- |
| 1. Agents and Uninformed Search | Agents, rationality, and kinds of environments; search problems and state spaces; breadth-first, uniform-cost, depth-first, iterative-deepening, and bidirectional search; repeated states and memory. | Topic 1 |
| 2. Heuristic Search | Greedy search and A\*; admissibility, consistency, and optimality; dominance and the effective branching factor; relaxed problems and pattern databases; IDA\* and memory-bounded search; weighted A\*. | Topic 2 |
| 3. Constraint Satisfaction and Local Search | Variables, domains, and constraints; backtracking with arc consistency, forward checking, and ordering; conflict-directed backjumping; tree-structured problems and treewidth; min-conflicts, hill climbing, and simulated annealing; phase transitions. | Topic 3 |
| 4. Adversarial Search and Games | Minimax; alpha–beta pruning and move ordering; transposition tables; evaluation functions and the horizon effect; expectiminimax; Monte Carlo tree search and UCT; from AlphaGo to imperfect-information games. | Topic 4 |
| 5. Propositional Logic and Satisfiability | Knowledge bases and entailment; model checking; resolution and its completeness; Horn clauses and chaining; DPLL, conflict-driven clause learning, and WalkSAT; the satisfiability threshold; logical agents. | Topic 5 |
| 6. First-Order Logic and Knowledge Representation | Objects, relations, and quantifiers; unification; forward chaining and Datalog; backward chaining and logic programming; resolution and semidecidability; ontologies, description logics, knowledge graphs, and defaults. | Topic 6 |
| 7. Automated Planning | STRIPS and PDDL; progression and regression; the delete relaxation and the heuristics $`h^{\max}`$, $`h^{\mathrm{add}}`$, and $`h^{\mathrm{FF}}`$; landmarks; planning as satisfiability; partial-order and hierarchical planning. | Topic 7 |
| 8. Bayesian Networks and Markov Networks | Conditional independence; Bayesian networks and their compactness; d-separation and explaining away; I-maps and Markov equivalence; Markov networks and the Hammersley–Clifford theorem; factor graphs and log-linear models. | Topic 8 |
| 9. Exact Inference | Inference tasks and their hardness; variable elimination; elimination orders and treewidth; sum-product and max-product on trees; the junction tree algorithm. | Topic 9 |
| 10. Approximate Inference | Rejection sampling, likelihood weighting, and importance sampling; Gibbs sampling and Metropolis–Hastings; mixing; the evidence lower bound and mean field; loopy belief propagation. | Topic 10 |
| 11. Temporal Probabilistic Models | Markov chains; hidden Markov models: filtering, smoothing, Viterbi decoding, and Baum–Welch; Kalman filters; dynamic Bayesian networks and particle filtering. | Topic 11 |
| 12. Decision Theory and the Value of Information | The axioms of utility and the expected-utility theorem; the utility of money and risk attitudes; departures of human judgment; the optimizer's curse; multiattribute utility; decision networks; the value of information. | Topic 12 |
| 13. Causal Inference | Simpson's paradox; potential outcomes; structural causal models and interventions; the backdoor and frontdoor criteria and do-calculus; adjustment, propensity weighting, and instruments; counterfactuals; causal discovery. | Topic 13 |
| 14. Learning Graphical Models *(optional)* | Maximum likelihood and Dirichlet priors for Bayesian networks; moment matching and pseudolikelihood for log-linear models; conditional random fields; EM with hidden variables; BIC, Chow–Liu trees, and structure search. | Topic 14 |
| 15. Game Theory and Multiagent Systems *(optional)* | Normal-form games, dominance, and Nash equilibrium; zero-sum games and the minimax theorem; correlated equilibria and no-regret learning; the price of anarchy; extensive-form and imperfect-information games; auctions, VCG, and social choice. | Topic 15 |

Chapters 1–4 treat problem solving as search: through state spaces with and without heuristics, through assignments to constrained variables, and through game trees against an opponent. Chapters 5–7 represent knowledge in logic, propositional and first-order, and use it to infer and to plan. Chapters 8–11 represent uncertainty with probabilistic graphical models and develop exact, approximate, and temporal inference. Chapters 12–13 turn beliefs into decisions, through utilities and the value of information, and into interventions, through causal models. The optional chapters learn graphical models from data and extend single-agent decisions to interacting agents. Proofs and longer derivations appear in collapsed appendices at the end of each chapter.

## <a id="shared-conventions"></a>Shared conventions

- A search problem has states $`s`$, actions $`a`$, a transition model $`\mathrm{Result}(s,a)`$, action costs $`c(s,a,s')`$, path cost $`g`$, and heuristic $`h`$; $`b`$ is the branching factor, $`d`$ the depth of the shallowest goal, and $`C^*`$ the optimal cost.
- Random variables are capitalized ($`X`$) and their values lowercase ($`x`$); sets of variables are written the same way. $`P(X\mid e)`$ is a distribution over $`X`$ given evidence $`e`$, $`\alpha`$ a normalizing constant, and $`P(y\mid\mathrm{do}(x))`$ an interventional distribution. Graphs are directed acyclic for Bayesian networks and undirected for Markov networks; $`\mathrm{Pa}_i`$ are the parents of $`X_i`$.
- Utilities are to be maximized and costs minimized; in games, MAX's utility is the value. Logarithms are natural unless written $`\log_2`$.
- Each code block runs on its own on a CPU with NumPy, SciPy, and the Python standard library, and implements its algorithm from scratch. Seeds are fixed, and the comment lines at the end of a block record what it printed in the environment described in the computing setup. The problems are puzzles, games, small networks, and synthetic data, so that each block finishes in seconds; results illustrate the chapters' claims and are not benchmarks.

## <a id="examples-and-supporting-resources"></a>Examples and supporting resources

The chapter text contains the definitions, derivations, and worked examples, and every figure is generated by a script in `Sources/Figure code`. The computing setup records the Python environment and how to regenerate the figures, and figure sources records the problems and construction behind each figure.

The reading plan lists the lectures and readings for each topic. Course links collects the courses it draws on, video links the lecture recordings by chapter, books and documentation the reference texts, libraries, and tools for larger problems, and papers the principal research behind each chapter. The [CS188 projects](https://inst.eecs.berkeley.edu/~cs188/sp24/projects/) implement search, adversarial search, logic, and probabilistic tracking for a Pacman agent with autograders, and fit chapters 1–6 and 8–11 directly.

## <a id="connections-to-later-modules"></a>Connections to later modules

**NLP and large language models** replace hand-built knowledge with learned representations. Decoding a language model is a search over sequences with beam search (chapter 2); hidden Markov models and conditional random fields (chapter 11) are the classical sequence models that neural ones superseded; and whether language models can reason and plan is measured against the logic and planning of chapters 5–7.

**Generative AI** builds on the variational inference of chapter 10 for variational autoencoders and on Markov chain Monte Carlo and energy-based models for sampling, and its diffusion models are Markov chains run in reverse.

**Reinforcement learning** extends the decisions of chapter 12 to sequences of actions with uncertain effects, the Markov decision processes this module leaves out; it learns the value functions that chapters 2 and 4 use as heuristics and evaluation functions, and its multiagent form continues chapter 15.

**Safety and frontier** research returns to this module's foundations: rational agency and the specification of utilities (chapters 1 and 12), the optimizer's curse and Goodhart's law, causal reasoning about what a model has learned (chapter 13), and game-theoretic analyses of interacting agents and of oversight (chapter 15).

## Reading plan and sources

- [Reading plan](reading-plan.md)
- [Book and documentation links](sources/book-and-documentation-links.md)
- [Computing setup](sources/computing-setup.md)
- [Course links](sources/course-links.md)
- [Figure sources](sources/figure-sources.md)
- [Paper links](sources/paper-links.md)
- [Video links](sources/video-links.md)
