[Background Notes](../../README.md) › [Artificial Intelligence](../README.md)

# Book and documentation links

# <a id="books-and-documentation"></a>Books and documentation

The chapter notes develop the module's main material. These books and documentation pages provide alternative explanations, fuller proofs, and the tools used for larger problems. The [reading plan](../reading-plan.md) lists the lecture slides and readings for each topic.

## <a id="main-references"></a>Main references

| Source | Relevant material |
| --- | --- |
| Russell and Norvig, [*Artificial Intelligence: A Modern Approach*](https://aima.cs.berkeley.edu/), fourth edition (Pearson, 2020) | The standard textbook: chapters 2–3 (chapter 1), 3.5–4 (chapters 2–3), 5 (chapter 4), 6 (chapter 3), 7 (chapter 5), 8–10 (chapter 6), 11 (chapter 7), 12–13 (chapters 8–10), 14 (chapter 11), 16 (chapter 12), 18 (chapter 15), and 20 (chapter 14). Not freely available; the book page links code and exercises. |
| Poole and Mackworth, [*Artificial Intelligence: Foundations of Computational Agents*](https://artint.info/3e/html/ArtInt3e.html), third edition (Cambridge, 2023) | Free HTML: chapters 1 and 3 (chapters 1–2), 4 (chapter 3), 5 (chapter 5), 6 (chapter 7), 9–10 (chapters 8–11 and 14), 11 (chapter 13), 12 (chapter 12), 14 (chapter 15), and 15–16 (chapter 6). |
| [CS188 online textbook](https://inst.eecs.berkeley.edu/~cs188/textbook/) | The main course's notes: search, CSPs, games, Bayesian networks, decision networks, HMMs, and logic (chapters 1–6 and 8–12). |
| Koller and Friedman, *Probabilistic Graphical Models: Principles and Techniques* (MIT Press, 2009) | The comprehensive reference for chapters 8–11 and 14: representation (3–6), exact inference (9–10, 13), approximate inference (11–12), temporal models (6, 15), and learning (16–20). Not freely available. |

## <a id="specialized-books-and-monographs"></a>Specialized books and monographs

| Source | Relevant material |
| --- | --- |
| Wainwright and Jordan, [*Graphical Models, Exponential Families, and Variational Inference*](https://doi.org/10.1561/2200000001) (2008) | The variational view of inference: mean field, Bethe and loopy belief propagation, and convex relaxations (chapters 9–10 and 14). |
| Murphy, [*Probabilistic Machine Learning: Advanced Topics*](https://probml.github.io/pml-book/book2.html) (MIT Press, 2023) | Free PDF; graphical models, inference algorithms, Monte Carlo, variational inference, state-space models, and causality (chapters 8–11, 13–14). |
| Ghallab, Nau, and Traverso, [*Automated Planning and Acting*](https://doi.org/10.1017/CBO9781139583923) (Cambridge, 2016) | Planning with deterministic and nondeterministic models, hierarchical refinement, and acting (chapter 7). |
| Thrun, Burgard, and Fox, [*Probabilistic Robotics*](https://mitpress.mit.edu/9780262201629/probabilistic-robotics/) (MIT Press, 2005) | Bayes filters, Kalman and particle filters, and localization (chapter 11). |
| Pearl, [*Causality*](https://doi.org/10.1017/CBO9780511803161), second edition (Cambridge, 2009) | Structural causal models, do-calculus, and counterfactuals (chapter 13). |
| Peters, Janzing, and Schölkopf, [*Elements of Causal Inference*](https://library.oapen.org/bitstream/id/056a11be-ce3a-44b9-8987-a6c68fce8d9b/11283.pdf) (MIT Press, 2017) | Open access; causal models, identifiability from observational data, and learning (chapter 13). |
| Neal, [*Introduction to Causal Inference*](https://www.bradyneal.com/Introduction_to_Causal_Inference-Dec17_2020-Neal.pdf) (course book, 2020) | Free PDF of the causal-inference course used in chapter 13. |
| Hernán and Robins, [*Causal Inference: What If*](https://miguelhernan.org/whatifbook) (2020, revised) | The potential-outcomes view with epidemiological examples: confounding, selection bias, and estimation (chapter 13). |
| Shoham and Leyton-Brown, [*Multiagent Systems*](https://www.masfoundations.org/download.html) (Cambridge, 2009) | Free PDF; normal-form and extensive-form games, computing equilibria, learning in games, mechanism design, and social choice (chapter 15). |
| Nisan, Roughgarden, Tardos, and Vazirani, eds., [*Algorithmic Game Theory*](https://doi.org/10.1017/CBO9780511800481) (Cambridge, 2007) | Complexity of equilibria, no-regret learning, the price of anarchy, and auctions (chapter 15). |

## <a id="library-documentation"></a>Library documentation

| Source | Relevant material |
| --- | --- |
| [NumPy documentation](https://numpy.org/doc/stable/) | Arrays, `einsum` for factor products, and random generation; used in every chapter. |
| [SciPy `linprog`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html) | Linear programming for zero-sum games (chapter 15). |
| [NetworkX documentation](https://networkx.org/documentation/stable/) | Grid graphs and maximum spanning trees (chapters 9 and 14). |

## <a id="tools-for-larger-problems"></a>Tools for larger problems

The notes implement every algorithm from scratch on small examples. For real problems, mature implementations exist:

| Tool | Use |
| --- | --- |
| [PySAT](https://pysathq.github.io/) | Python bindings to modern CDCL SAT solvers (chapter 5). |
| [Fast Downward](https://www.fast-downward.org/) | A classical planning system implementing the heuristics of chapter 7 on PDDL tasks. |
| [pgmpy](https://pgmpy.org/) | Bayesian and Markov networks: exact and approximate inference, parameter and structure learning (chapters 8–10, 14). |
| [Stan](https://mc-stan.org/) and [PyMC](https://www.pymc.io/) | Probabilistic programming with Hamiltonian Monte Carlo and variational inference (chapter 10). |
| [DoWhy](https://www.pywhy.org/dowhy/) | Causal effect identification, estimation, and refutation tests (chapter 13). |
| [OpenSpiel](https://github.com/google-deepmind/open_spiel) | Games and algorithms, including MCTS, CFR, and regret matching (chapters 4 and 15). |
