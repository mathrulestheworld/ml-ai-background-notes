[ML Mastery Notes](../README.md) › [Artificial Intelligence](README.md)

# Reading plan

## <a id="courses"></a>Courses

- **Main course — UC Berkeley CS188, Spring 2024: *Introduction to Artificial Intelligence*.** [Course homepage](https://inst.eecs.berkeley.edu/~cs188/sp24/) · [Online textbook](https://inst.eecs.berkeley.edu/~cs188/textbook/) · [Lecture playlist](https://www.youtube.com/playlist?list=PLp8QV47qJEg67UTShQ4er4RYQ3rOeDKxv). Supplies the main lecture sequence: agents and search, games, propositional and first-order logic, Bayesian networks, hidden Markov models, and decisions under uncertainty. The online textbook collects the course notes by topic. Spring 2024 has no lectures on constraint satisfaction, so block 3 uses the Spring 2023 slides and notes.
- **Supplement — CMU 10-708, Spring 2020: Eric Xing, *Probabilistic Graphical Models*.** [Lectures with slides, scribe notes, and videos](https://www.cs.cmu.edu/~epxing/Class/10708-20/lectures.html). Adds Markov random fields, exact inference beyond elimination, variational inference, Markov chain Monte Carlo, learning, and causality at graduate depth.
- **Supplement — Stanford CS228: Stefano Ermon, *Probabilistic Graphical Models* course notes.** [Notes](https://ermongroup.github.io/cs228-notes/). Short written chapters on representation, inference, and learning that match blocks 8–10 and 14.

The order below follows the chapters of this module. Each block lists the material to cover and its primary lectures; further reading and exercises provide additional depth. Blocks 1–13 form the main plan; 14–15 are optional extensions. Timing is flexible.

CS188 slide links are original PDFs and the notes are the course's own; the videos are the Spring 2024 recordings. Markov decision processes, bandits, and reinforcement learning, which CS188 teaches between games and probability, belong to the RL module; its machine-learning lectures repeat material from ML and DL. Foundations covered probability, Bayes' rule, and Monte Carlo integration, and ML covered naive Bayes, Gaussian mixtures, and EM; review those chapters as needed in place of CS188's probability review.

## <a id="topic-list"></a>Topic list

- [ ] [1. Agents and uninformed search](#1-agents-and-uninformed-search)
- [ ] [2. Heuristic search](#2-heuristic-search)
- [ ] [3. Constraint satisfaction and local search](#3-constraint-satisfaction-and-local-search)
- [ ] [4. Adversarial search and games](#4-adversarial-search-and-games)
- [ ] [5. Propositional logic and satisfiability](#5-propositional-logic-and-satisfiability)
- [ ] [6. First-order logic and knowledge representation](#6-first-order-logic-and-knowledge-representation)
- [ ] [7. Automated planning](#7-automated-planning)
- [ ] [8. Bayesian networks and Markov networks](#8-bayesian-networks-and-markov-networks)
- [ ] [9. Exact inference](#9-exact-inference)
- [ ] [10. Approximate inference](#10-approximate-inference)
- [ ] [11. Temporal probabilistic models](#11-temporal-probabilistic-models)
- [ ] [12. Decision theory and the value of information](#12-decision-theory-and-the-value-of-information)
- [ ] [13. Causal inference](#13-causal-inference)
- [ ] [14. Learning graphical models — optional](#14-learning-graphical-models-optional)
- [ ] [15. Game theory and multiagent systems — optional](#15-game-theory-and-multiagent-systems-optional)

### <a id="reference-books-used-below"></a>Reference books used below

- **AIMA:** Russell and Norvig, [Artificial Intelligence: A Modern Approach](https://aima.cs.berkeley.edu/), fourth edition (Pearson, 2020). The standard textbook for blocks 1–12 and 15; CS188 cites its chapters. Not freely available; the book page links code and exercises.
- **AIFCA:** Poole and Mackworth, [Artificial Intelligence: Foundations of Computational Agents](https://artint.info/3e/html/ArtInt3e.html), third edition (Cambridge, 2023). Free HTML; a second full treatment with a stronger emphasis on representation, causality, and relational models.
- **PGM:** Koller and Friedman, *Probabilistic Graphical Models: Principles and Techniques* (MIT Press, 2009). The reference for blocks 8–10 and 14; not freely available. The CS228 notes above follow it closely.
- **ECI:** Peters, Janzing, and Schölkopf, [Elements of Causal Inference](https://library.oapen.org/bitstream/id/056a11be-ce3a-44b9-8987-a6c68fce8d9b/11283.pdf) (MIT Press, 2017). Open access; the reference for block 13 together with Brady Neal's course book.

## <a id="1-agents-and-uninformed-search"></a>1. Agents and uninformed search

**Topics:** Agents, environments, and rationality; performance measures; properties of environments; search problems as states, actions, transitions, goals, and costs; the search tree and the state graph; breadth-first, depth-first, uniform-cost, and iterative-deepening search; completeness, optimality, and time and space complexity; tree search versus graph search.

- **CS188 lecture 1: Intro to AI, rational agents.** [Slides · PDF](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/lectures/cs188-sp24-lec01.pdf) · [Video](https://www.youtube.com/watch?v=NG3MHjcvk2A)
- **CS188 lecture 2: State spaces, uninformed search.** [Slides · PDF](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/lectures/cs188-sp24-lec02.pdf) · [Video](https://www.youtube.com/watch?v=PU6VQbV49_g)
- **CS188 textbook 1.1–1.3.** [Agents](https://inst.eecs.berkeley.edu/~cs188/textbook/search/agents.html) · [State spaces and search problems](https://inst.eecs.berkeley.edu/~cs188/textbook/search/state.html) · [Uninformed search](https://inst.eecs.berkeley.edu/~cs188/textbook/search/uninformed.html)

**Further reading:** AIMA chapters 2 and 3.1–3.4; AIFCA chapters 1 and 3.1–3.5. Foundations' terminology chapter places AI among the neighboring disciplines.

## <a id="2-heuristic-search"></a>2. Heuristic search

**Topics:** Best-first search; greedy search and A*; admissible and consistent heuristics; optimality of A* for tree and graph search; dominance and the effective branching factor; heuristics from relaxed problems and pattern databases; memory-bounded search with IDA*; weighted A* and bounded suboptimality.

- **CS188 lecture 3: Informed search, A\* and heuristics.** [Slides · PDF](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/lectures/cs188-sp24-lec03.pdf) · [Video](https://www.youtube.com/watch?v=zRFZwAUQT8U)
- **CS188 textbook 1.4.** [Informed search](https://inst.eecs.berkeley.edu/~cs188/textbook/search/informed.html)
- **Basel *Planning and Optimization* E6: Pattern databases.** [Slides · PDF](https://ai.dmi.unibas.ch/_files/teaching/hs23/po/slides/po-e06.pdf) Abstraction heuristics defined precisely.

**Further reading:** AIMA 3.5–3.6; AIFCA 3.6–3.8. Amit Patel's [notes on A* for games](https://theory.stanford.edu/~amitp/GameProgramming/) cover heuristics, tie-breaking, and implementation details for grid maps.

## <a id="3-constraint-satisfaction-and-local-search"></a>3. Constraint satisfaction and local search

**Topics:** Variables, domains, and constraints; constraint graphs and factor graphs; backtracking search; forward checking and arc consistency (AC-3); variable and value ordering; tree-structured CSPs and cutset conditioning; local search with min-conflicts, hill climbing, simulated annealing, and beam search; the phase transition in random problems.

- **CS188 Spring 2023 lectures 7–8: Constraint satisfaction problems I and II.** [Slides I · PDF](https://inst.eecs.berkeley.edu/~cs188/sp23/assets/lectures/cs188-sp23-lec07.pdf) · [Slides II · PDF](https://inst.eecs.berkeley.edu/~cs188/sp23/assets/lectures/cs188-sp23-lec08.pdf) · [Note 7](https://inst.eecs.berkeley.edu/~cs188/sp23/assets/notes/cs188-sp23-note07.pdf) · [Note 8](https://inst.eecs.berkeley.edu/~cs188/sp23/assets/notes/cs188-sp23-note08.pdf)
- **CS188 lecture 4: Local search.** [Slides · PDF](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/lectures/cs188-sp24-lec04.pdf) · [Video](https://www.youtube.com/watch?v=Wqmd_i0coLc)
- **Stanford CS221, Autumn 2019: Factor graphs 1, constraint satisfaction problems.** [Video](https://www.youtube.com/watch?v=Yo-xat4cn8M) CSPs as factor graphs, the view that connects this block to blocks 8–9.
- **CS188 textbook chapter 2 and 1.5.** [Constraint satisfaction problems](https://inst.eecs.berkeley.edu/~cs188/textbook/csp/csps.html) · [Local search](https://inst.eecs.berkeley.edu/~cs188/textbook/search/local.html)

**Further reading:** AIMA chapter 6 and 4.1–4.2; AIFCA chapter 4.

## <a id="4-adversarial-search-and-games"></a>4. Adversarial search and games

**Topics:** Games as search problems; minimax and its optimality against an optimal opponent; alpha–beta pruning and move ordering; depth-limited search and evaluation functions; expectimax for chance and for imperfect opponents; multiplayer games; Monte Carlo tree search and the UCT rule.

- **CS188 lecture 5: Games, trees, minimax, pruning.** [Slides · PDF](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/lectures/cs188-sp24-lec05.pdf) · [Video](https://www.youtube.com/watch?v=h8Tz4knj-vM)
- **CS188 lecture 6: Games, expectimax, Monte Carlo tree search.** [Slides · PDF](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/lectures/cs188-sp24-lec06.pdf) · [Video](https://www.youtube.com/watch?v=_-xH1CAfoxM)
- **CS188 textbook chapter 3.** [Games](https://inst.eecs.berkeley.edu/~cs188/textbook/games/games.html) · [Monte Carlo tree search](https://inst.eecs.berkeley.edu/~cs188/textbook/games/monte-carlo.html)

**Further reading:** AIMA chapter 5. The survey by [Browne et al. (2012)](https://doi.org/10.1109/TCIAIG.2012.2186810) catalogues Monte Carlo tree search variants.

## <a id="5-propositional-logic-and-satisfiability"></a>5. Propositional logic and satisfiability

**Topics:** Knowledge-based agents; syntax and semantics of propositional logic; models, entailment, validity, and satisfiability; inference by model checking; proof rules and resolution; conjunctive normal form; Horn clauses with forward and backward chaining; DPLL, conflict-driven clause learning, and WalkSAT; hardness and the satisfiability threshold.

- **CS188 lecture 7: Propositional logic and planning.** [Slides · PDF](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/lectures/cs188-sp24-lec07.pdf) · [Video](https://www.youtube.com/watch?v=WVaBk3ldQIo)
- **CS188 lecture 8: Logical inference, theorem proving, Boolean satisfiability, DPLL.** [Slides · PDF](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/lectures/cs188-sp24-lec08.pdf) · [Video](https://www.youtube.com/watch?v=09RkMHtqUP0)
- **Stanford CS221, Autumn 2019: Logic 1, propositional logic.** [Video](https://www.youtube.com/watch?v=xL0kNw5TudI) Syntax, semantics, and inference rules with a careful treatment of soundness and completeness.
- **CS188 textbook 10.1–10.6.** [Knowledge-based agents](https://inst.eecs.berkeley.edu/~cs188/textbook/logic/knowledge.html) · [Propositional inference](https://inst.eecs.berkeley.edu/~cs188/textbook/logic/inference.html)

**Further reading:** AIMA chapter 7; AIFCA chapter 5.

## <a id="6-first-order-logic-and-knowledge-representation"></a>6. First-order logic and knowledge representation

**Topics:** Objects, relations, and functions; terms, quantifiers, and models; translating knowledge into first-order sentences; propositionalization; unification and generalized modus ponens; forward chaining and Datalog; backward chaining and logic programming; resolution and its completeness; semidecidability; ontologies, categories, and description logics; the closed-world assumption and negation as failure.

- **CS188 lecture 9: First-order logic.** [Slides · PDF](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/lectures/cs188-sp24-lec09.pdf) · [Video](https://www.youtube.com/watch?v=fiE-oT3FPms)
- **Stanford CS221, Autumn 2019: Logic 2, first-order logic.** [Video](https://www.youtube.com/watch?v=_Iz83hfkFds)
- **CS188 textbook 10.7–10.9.** [First-order logic](https://inst.eecs.berkeley.edu/~cs188/textbook/logic/first-order-logic.html) · [First-order inference](https://inst.eecs.berkeley.edu/~cs188/textbook/logic/first-order-logical-inference.html) · [Logical agents](https://inst.eecs.berkeley.edu/~cs188/textbook/logic/logical-agents.html)

**Further reading:** AIMA chapters 8–10; AIFCA chapters 15–16 on individuals, relations, knowledge graphs, and ontologies.

## <a id="7-automated-planning"></a>7. Automated planning

**Topics:** Planning tasks in STRIPS and PDDL; forward (progression) and backward (regression) search; the complexity of planning; planning as satisfiability; the delete relaxation and the heuristics $`h^{\max}`$, $`h^{\text{add}}`$, and $`h^{\text{FF}}`$; landmarks; partial-order and hierarchical planning; where planning under uncertainty begins.

- **CS188 lecture 7: Propositional logic and planning.** [Slides · PDF](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/lectures/cs188-sp24-lec07.pdf) · [Video](https://www.youtube.com/watch?v=WVaBk3ldQIo) Planning as satisfiability with successor-state axioms.
- **Basel *Planning and Optimization*, Fall 2023: Malte Helmert and Gabriele Röger.** [Course page](https://dmi.unibas.ch/en/studies/computer-science/courses-in-fall-semester-2023/lecture-planning-and-optimization/). Slide sets [A2 What is planning?](https://ai.dmi.unibas.ch/_files/teaching/hs23/po/slides/po-a02.pdf), [B5 STRIPS](https://ai.dmi.unibas.ch/_files/teaching/hs23/po/slides/po-b05.pdf), [B6 Complexity](https://ai.dmi.unibas.ch/_files/teaching/hs23/po/slides/po-b06.pdf), [C2 Progression and regression](https://ai.dmi.unibas.ch/_files/teaching/hs23/po/slides/po-c02.pdf), [C4 SAT planning](https://ai.dmi.unibas.ch/_files/teaching/hs23/po/slides/po-c04.pdf), [D1 Relaxed planning tasks](https://ai.dmi.unibas.ch/_files/teaching/hs23/po/slides/po-d01.pdf), [D6 $`h^{\max}`$ and $`h^{\text{add}}`$](https://ai.dmi.unibas.ch/_files/teaching/hs23/po/slides/po-d06.pdf), [D8 $`h^{\text{FF}}`$](https://ai.dmi.unibas.ch/_files/teaching/hs23/po/slides/po-d08.pdf), and [G3 Landmarks](https://ai.dmi.unibas.ch/_files/teaching/hs23/po/slides/po-g03.pdf).

**Further reading:** AIMA chapter 11; AIFCA chapter 6.

## <a id="8-bayesian-networks-and-markov-networks"></a>8. Bayesian networks and Markov networks

**Topics:** Joint distributions and the cost of tables; directed graphical models and their factorization; conditional independence and d-separation; the Markov blanket; I-maps and perfect maps; undirected models, potentials, and the Hammersley–Clifford theorem; factor graphs; converting between representations.

- **CS188 lecture 10: Intro to probability** (review as needed). [Slides · PDF](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/lectures/cs188-sp24-lec10.pdf) · [Video](https://www.youtube.com/watch?v=rzcZqPAeWVs)
- **CS188 lecture 11: Bayes nets.** [Slides · PDF](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/lectures/cs188-sp24-lec11.pdf) · [Video](https://www.youtube.com/watch?v=lJ7nHubyahw)
- **CMU 10-708 lecture 2: Undirected graphical models.** [Slides · PDF](https://www.cs.cmu.edu/~epxing/Class/10708-20/lectures/lecture02-MRFrepresentation.pdf) · [Video](https://www.youtube.com/watch?v=aJviHlCgy7U)
- **CMU 10-708 lecture 3: Directed graphical models.** [Slides · PDF](https://www.cs.cmu.edu/~epxing/Class/10708-20/lectures/lecture03-BNrepresentation.pdf) · [Video](https://www.youtube.com/watch?v=cyXQkRwszE8)
- **Stanford CS221, Autumn 2019: Factor graphs 2, conditional independence.** [Video](https://www.youtube.com/watch?v=o0mKSvbMunA)
- **CS228 notes.** [Bayesian networks](https://ermongroup.github.io/cs228-notes/representation/directed/) · [Markov random fields](https://ermongroup.github.io/cs228-notes/representation/undirected/)

**Further reading:** AIMA 13.1–13.2; PGM chapters 3–4; CS188 textbook [6.3–6.5](https://inst.eecs.berkeley.edu/~cs188/textbook/bayes-nets/representation.html).

## <a id="9-exact-inference"></a>9. Exact inference

**Topics:** Inference by enumeration; variable elimination and factor operations; elimination orderings, induced width, and treewidth; the hardness of inference; sum-product message passing on trees; the junction tree algorithm; MAP inference with max-product; exact inference as a baseline for approximation.

- **CS188 lecture 12: Bayes nets, inference.** [Slides · PDF](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/lectures/cs188-sp24-lec12.pdf) · [Video](https://www.youtube.com/watch?v=FW9XbSGcR4Q)
- **CMU 10-708 lecture 4: Exact inference.** [Slides · PDF](https://www.cs.cmu.edu/~epxing/Class/10708-20/lectures/lecture04-ExactInference.pdf) · [Video](https://www.youtube.com/watch?v=LEBeACzpYvY)
- **CS228 notes.** [Variable elimination](https://ermongroup.github.io/cs228-notes/inference/ve/) · [Belief propagation](https://ermongroup.github.io/cs228-notes/inference/jt/) · [MAP inference](https://ermongroup.github.io/cs228-notes/inference/map/)

**Further reading:** AIMA 13.3; PGM chapters 9–10 and 13.

## <a id="10-approximate-inference"></a>10. Approximate inference

**Topics:** Prior sampling, rejection sampling, and likelihood weighting; importance sampling and effective sample size; Markov chains and their stationary distributions; Gibbs sampling and Metropolis–Hastings; mixing and diagnostics; variational inference as optimization, the evidence lower bound, and mean field; loopy belief propagation and the Bethe approximation.

- **CS188 lecture 13: Bayes nets, sampling.** [Slides · PDF](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/lectures/cs188-sp24-lec13.pdf) · [Video](https://www.youtube.com/watch?v=DkgxBR_43AI)
- **CMU 10-708 lectures 9–10: Sampling 1 and 2.** [Slides 1 · PDF](https://www.cs.cmu.edu/~epxing/Class/10708-20/lectures/lecture09-MC.pdf) · [Video 1](https://www.youtube.com/watch?v=iE9PPB4bHiw) · [Slides 2 · PDF](https://www.cs.cmu.edu/~epxing/Class/10708-20/lectures/lecture10-MCMC-opt.pdf) · [Video 2](https://www.youtube.com/watch?v=gc1u1Ds7SnI)
- **CMU 10-708 lectures 7–8: Variational inference 1 and 2.** [Slides 1 · PDF](https://www.cs.cmu.edu/~epxing/Class/10708-20/lectures/lecture07-VI1.pdf) · [Video 1](https://www.youtube.com/watch?v=_fMDYpdwSOw) · [Slides 2 · PDF](https://www.cs.cmu.edu/~epxing/Class/10708-20/lectures/lecture08-VI2.pdf) · [Video 2](https://www.youtube.com/watch?v=UOtJBjZy3b4)
- **CS228 notes.** [Sampling-based inference](https://ermongroup.github.io/cs228-notes/inference/sampling/) · [Variational inference](https://ermongroup.github.io/cs228-notes/inference/variational/)

**Further reading:** AIMA 13.4; PGM chapters 11–12. Blei, Kucukelbir, and McAuliffe, [*Variational Inference: A Review for Statisticians*](https://arxiv.org/abs/1601.00670), is the standard introduction to modern variational methods; Wainwright and Jordan's monograph (listed with the books) develops the theory.

## <a id="11-temporal-probabilistic-models"></a>11. Temporal probabilistic models

**Topics:** Markov chains and stationarity; hidden Markov models; filtering with the forward algorithm, prediction, and smoothing with forward–backward; the most likely sequence and the Viterbi algorithm; learning with Baum–Welch; linear-Gaussian models and the Kalman filter; dynamic Bayesian networks; particle filtering.

- **CS188 lecture 14: Markov chains, HMMs.** [Slides · PDF](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/lectures/cs188-sp24-lec14.pdf) · [Video](https://www.youtube.com/watch?v=Hw8uDHUPhWE)
- **CS188 lecture 15: Forward and Viterbi algorithms, dynamic Bayes nets, particle filtering.** [Slides · PDF](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/lectures/cs188-sp24-lec15.pdf) · [Video](https://www.youtube.com/watch?v=U5eAXW7LsMI)
- **CMU 10-708 lecture 6: Case studies, HMM and CRF.** [Slides · PDF](https://www.cs.cmu.edu/~epxing/Class/10708-20/lectures/lecture06-HMMCRF.pdf) · [Video](https://www.youtube.com/watch?v=sZN6S4q1p_Q)
- **CS188 textbook chapter 8.** [Markov models](https://inst.eecs.berkeley.edu/~cs188/textbook/hmms/markov.html) · [Particle filtering](https://inst.eecs.berkeley.edu/~cs188/textbook/hmms/particle-filtering.html)

**Further reading:** AIMA chapter 14; Rabiner's [tutorial on hidden Markov models](https://doi.org/10.1109/5.18626); Thrun, Burgard, and Fox, *Probabilistic Robotics*, chapters 2–4, for Bayes filters in robotics.

## <a id="12-decision-theory-and-the-value-of-information"></a>12. Decision theory and the value of information

**Topics:** Preferences, lotteries, and the axioms of rationality; the von Neumann–Morgenstern utility theorem; utility of money and risk attitudes; multiattribute utility; decision networks and their evaluation; the value of perfect information and its properties; limits of the expected-utility model.

- **CS188 lecture 16: Utility theory, rationality, decision networks, VPI.** [Slides · PDF](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/lectures/cs188-sp24-lec16.pdf) · [Video](https://www.youtube.com/watch?v=hY3JEKiCcOU)
- **CS188 textbook chapter 7.** [Utilities](https://inst.eecs.berkeley.edu/~cs188/textbook/vpis/utilities.html) · [Decision networks](https://inst.eecs.berkeley.edu/~cs188/textbook/vpis/decision-networks.html) · [Value of perfect information](https://inst.eecs.berkeley.edu/~cs188/textbook/vpis/vpi.html)

**Further reading:** AIMA chapter 16; AIFCA 12.1–12.3. Foundations' statistics chapter treats losses, risk, and Bayes decisions for estimation; this block adds where utilities come from and how to decide what to observe.

## <a id="13-causal-inference"></a>13. Causal inference

**Topics:** Association and causation; potential outcomes; structural causal models and interventions; the do-operator and the truncated factorization; confounding, the backdoor and frontdoor criteria; the rules of do-calculus; estimation by adjustment and inverse propensity weighting; instrumental variables; counterfactuals; causal discovery from independence tests.

- **Brady Neal, *Introduction to Causal Inference*.** [Course page](https://www.bradyneal.com/causal-inference-course) · [Course book · PDF](https://www.bradyneal.com/Introduction_to_Causal_Inference-Dec17_2020-Neal.pdf). Lectures [1 Introduction](https://www.youtube.com/watch?v=CfzO4IEMVUk), [2 Potential outcomes](https://www.youtube.com/watch?v=q8x9aetyok0), [3 The flow of association and causation in graphs](https://www.youtube.com/watch?v=Go4EkHN_PcA), [4 Causal models](https://www.youtube.com/watch?v=dB8r4Afmobo), [5 Identification](https://www.youtube.com/watch?v=z91LnTDyhtI), [6 Estimation](https://www.youtube.com/watch?v=YzcOYU-s2t4), [8 Instrumental variables](https://www.youtube.com/watch?v=Mco16tUSA-U), [10 Causal discovery from observational data](https://www.youtube.com/watch?v=lVE-4deFe7c), and [13 Counterfactuals and mediation](https://www.youtube.com/watch?v=f8PEpthLlN4).
- **CMU 10-708 lectures 17–18: Causality 1 and 2.** [Slides 1 · PDF](https://www.cs.cmu.edu/~epxing/Class/10708-20/lectures/lecture17-Causality1.pdf) · [Video 1](https://www.youtube.com/watch?v=Cw887D_sE04) · [Slides 2 · PDF](https://www.cs.cmu.edu/~epxing/Class/10708-20/lectures/lecture18-Causality2.pdf) · [Video 2](https://www.youtube.com/watch?v=R4JGk4JyrHw)

**Further reading:** ECI chapters 1–7; AIFCA chapter 11; Hernán and Robins, [*Causal Inference: What If*](https://miguelhernan.org/whatifbook), for the potential-outcomes view used in epidemiology.

## <a id="14-learning-graphical-models-optional"></a>14. Learning graphical models — optional

**Topics:** Maximum likelihood for Bayesian networks with complete data; Dirichlet priors and Bayesian estimation; log-linear models and the moment-matching gradient; pseudolikelihood; missing data and EM in Bayesian networks; structure learning by score (BIC, the Chow–Liu tree) and by independence tests; conditional random fields.

- **CMU 10-708 lecture 5: Parameter estimation.** [Slides · PDF](https://www.cs.cmu.edu/~epxing/Class/10708-20/lectures/lecture05-ParameterEst.pdf) · [Video](https://www.youtube.com/watch?v=stSWElRm6mw)
- **CMU 10-708 lecture 16: Structure learning.** [Slides · PDF](https://www.cs.cmu.edu/~epxing/Class/10708-20/lectures/lecture16-NetworkLearning.pdf) · [Video](https://www.youtube.com/watch?v=aQXIdBsa-hU)
- **CS228 notes.** [Learning in directed models](https://ermongroup.github.io/cs228-notes/learning/directed/) · [Learning in undirected models](https://ermongroup.github.io/cs228-notes/learning/undirected/) · [Structure learning](https://ermongroup.github.io/cs228-notes/learning/structure/)

**Further reading:** PGM chapters 17–20; AIMA chapter 20.

## <a id="15-game-theory-and-multiagent-systems-optional"></a>15. Game theory and multiagent systems — optional

**Topics:** Normal-form games; dominance and best responses; pure and mixed Nash equilibria; zero-sum games, the minimax theorem, and linear programming; correlated and coarse correlated equilibria; no-regret learning and its convergence to equilibrium; extensive-form games and subgame perfection; mechanism design, auctions, and the VCG mechanism; the price of anarchy.

- **Stanford CS364A, Fall 2013: Tim Roughgarden, *Algorithmic Game Theory*.** [Course page with lecture notes](https://timroughgarden.org/f13/f13.html). Lectures [2 Mechanism design basics](https://www.youtube.com/watch?v=z1QZqYuiGa8), [7 VCG mechanism](https://www.youtube.com/watch?v=TLl3FVXPVIY), [11 Selfish routing and the price of anarchy](https://www.youtube.com/watch?v=jnLEEr3pc4Y), [13 Hierarchy of equilibrium concepts](https://www.youtube.com/watch?v=aV16MDoRZoc), [17 No-regret dynamics](https://www.youtube.com/watch?v=ssAEgJKRe9o), [18 Swap regret and minimax](https://www.youtube.com/watch?v=XQ32p6clz9k), and [20 Mixed Nash equilibria and PPAD-completeness](https://www.youtube.com/watch?v=P8adJn_KQO0).

**Further reading:** Shoham and Leyton-Brown, [*Multiagent Systems*](https://www.masfoundations.org/download.html) (free PDF), chapters 3–5 and 10–11; AIMA chapter 18; AIFCA chapter 14.

## <a id="exercises-and-local-references"></a>Exercises and local references

- The [CS188 projects](https://inst.eecs.berkeley.edu/~cs188/sp24/projects/) implement search, adversarial search, logic, and Bayesian-network and HMM tracking in Python for a Pacman world, with autograders; the machine-learning project repeats ML and DL material, and the reinforcement-learning project belongs with RL. Past exams with solutions are linked from the course site and cover every block through 12.
- The [AIMA Python repository](https://github.com/aimacode/aima-python) implements most algorithms of the textbook, with notebooks.
- Foundations supplies probability, Monte Carlo, and information theory; ML supplies naive Bayes, Gaussian mixtures, and EM; DL supplies the networks used as learned heuristics and evaluation functions.

## <a id="connections-to-later-modules"></a>Connections to later modules

Markov decision processes, dynamic programming, bandits, and reinforcement learning belong in RL, which extends the search, games, and decision theory of this module to sequential decisions with learned values. Variational autoencoders and diffusion models in Generative AI build on the variational inference of block 10. Language models in NLP and LLMs replace the hand-built knowledge of blocks 5–6 with learned representations, and their decoding uses the search of blocks 1–2. Safety and Frontier returns to rational agency, utility, and causal reasoning. Search, logic, planning, graphical models, decisions, and causality remain in this module.
