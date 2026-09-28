[Background Notes](../README.md)

# Machine Learning

Machine Learning develops the classical methods for learning from data: how a prediction rule is fitted, how its error is estimated and controlled, and how structure is found without labels. It builds on the linear algebra, probability, optimization, and learning theory of [Foundations](../foundations/README.md) and covers the supervised and unsupervised core that later modules assume. Fifteen core chapters follow the main topics of the [reading plan](reading-plan.md), and two optional chapters cover its extensions.

## <a id="chapters"></a>Chapters

| Chapter | Main content | Reading plan |
| --- | --- | --- |
| [1. Learning Problems and Nearest Neighbors](01-learning-problems-and-nearest-neighbors.md) | Population and empirical risk; the Bayes predictor; the $`k`$-nearest-neighbor rule and the choice of $`k`$; distances and scaling; the Cover–Hart theorem and universal consistency; the curse of dimensionality. | [Topics 1](reading-plan.md#1-learning-problems-and-prediction) and [2](reading-plan.md#2-nearest-neighbors-and-dimensionality) |
| [2. The Perceptron and Linear Separation](02-the-perceptron-and-linear-separation.md) | Linear classifiers and margins; the perceptron update; Novikoff's mistake bound; online-to-batch conversion; nonseparable data and averaging; the multiclass perceptron. | [Topic 3](reading-plan.md#3-perceptron-and-linear-separation) |
| [3. Linear Regression and Regularization](03-linear-regression-and-regularization.md) | Least squares as a learning method; ridge regression as shrinkage and MAP estimation; effective degrees of freedom; the lasso, coordinate descent, and regularization paths; the elastic net. | [Topic 4](reading-plan.md#4-linear-regression-and-regularization) |
| [4. Generative Classifiers](04-generative-classifiers.md) | Classification through Bayes' rule; naive Bayes event models and smoothing; quadratic, linear, and regularized discriminant analysis; Fisher's discriminant and reduced-rank LDA. | [Topic 5](reading-plan.md#5-probabilistic-estimation-and-generative-classifiers) |
| [5. Logistic Regression and Probabilistic Prediction](05-logistic-regression-and-probabilistic-prediction.md) | Conditional likelihood and log loss; Newton's method as IRLS; separation; regularization; the softmax model; generative versus discriminative estimation; decisions from probabilities; calibration and proper scoring rules. | [Topic 6](reading-plan.md#6-logistic-regression-and-probabilistic-prediction) |
| [6. Losses, Model Selection, and Evaluation](06-losses-model-selection-and-evaluation.md) | Surrogate losses and what they estimate; the bias–variance decomposition and double descent; analytic criteria and cross-validation; nested cross-validation and early stopping; leakage; ROC and precision–recall curves; comparing classifiers. | [Topic 7](reading-plan.md#7-losses-model-selection-and-biasvariance) |
| [7. Statistical Learning Theory](07-statistical-learning-theory.md) | PAC learning of rectangles and conjunctions; computational hardness; growth functions and Sauer's lemma; VC-dimension proofs; the fundamental theorem; structural risk minimization; mistake bounds and the Littlestone dimension. | [Topic 8](reading-plan.md#8-statistical-learning-theory) |
| [8. Support Vector Machines and Kernels](08-support-vector-machines-and-kernels.md) | Hard and soft margins; the hinge loss; duality and support vectors; positive semidefinite kernels and the representer theorem; the kernel perceptron and kernel ridge regression; Nyström and random features; choosing kernel parameters. | [Topic 9](reading-plan.md#9-margins-support-vector-machines-and-kernels) |
| [9. Decision and Regression Trees](09-decision-and-regression-trees.md) | Recursive partitioning; regression splits and impurity measures for classification; categorical features and missing values; cost-complexity pruning; instability. | [Topic 10](reading-plan.md#10-decision-and-regression-trees) |
| [10. Bagging and Random Forests](10-bagging-and-random-forests.md) | The variance of an average; bootstrap aggregation and out-of-bag error; random forests, strength, and correlation; extremely randomized trees; impurity and permutation importance. | [Topic 11](reading-plan.md#11-bagging-and-random-forests) |
| [11. Boosting](11-boosting.md) | Weak and strong learning; AdaBoost and its training-error bound; the exponential loss and stagewise additive modeling; margins; gradient boosting with Newton steps and regularized trees. | [Topic 12](reading-plan.md#12-adaboost-and-gradient-boosting) |
| [12. Principal Components and Dimensionality Reduction](12-principal-components-and-dimensionality-reduction.md) | PCA as a fitted transformation; choosing the number of components and the eigenvalues of noise; probabilistic PCA; principal components regression; kernel PCA; multidimensional scaling. | [Topic 13](reading-plan.md#13-principal-components-and-dimensionality-reduction) |
| [13. Clustering](13-clustering.md) | Clustering objectives and an impossibility theorem; $`k`$-means, Lloyd's algorithm, and $`k`$-means++; agglomerative clustering and linkage; choosing the number of clusters; evaluating and comparing partitions. | [Topic 14](reading-plan.md#14-clustering) |
| [14. Gaussian Mixtures and Expectation Maximization](14-gaussian-mixtures-and-expectation-maximization.md) | Latent labels and responsibilities; EM for Gaussian mixtures; the evidence lower bound and why EM increases the likelihood; initialization, degeneracies, and BIC; the relation to $`k`$-means. | [Topic 15](reading-plan.md#15-gaussian-mixtures-and-expectation-maximization) |
| [15. Gaussian Processes](15-gaussian-processes.md) | Bayesian linear regression; distributions over functions; kernels as prior assumptions; posterior prediction and kernel ridge regression; the marginal likelihood; Bayesian optimization. | [Topic 16](reading-plan.md#16-gaussian-processes) |
| [16. Semi-Supervised and Active Learning](16-semi-supervised-and-active-learning.md) *(optional)* | When unlabeled data can help; EM, self-training, and co-training; graph-based label propagation; uncertainty sampling; label complexity and sampling bias. | [Topic 17](reading-plan.md#17-semi-supervised-and-active-learning-optional) |
| [17. Smoothing, Density Estimation, and Basis Expansions](17-smoothing-density-estimation-and-basis-expansions.md) *(optional)* | Kernel density estimation and bandwidth selection; local linear regression; regression and smoothing splines; additive models and backfitting; $`k`$-d trees and approximate nearest-neighbor search. | [Topic 18](reading-plan.md#18-further-classical-methods-optional) |

Chapters 1–5 introduce the basic supervised methods, from local averaging to linear and probabilistic classifiers. Chapters 6–7 turn to how error is estimated and why fitting a sample can generalize. Chapters 8–11 develop kernels, trees, and ensembles, the strongest classical methods for fixed-length feature vectors. Chapters 12–15 treat unsupervised learning and probabilistic models: dimensionality reduction, clustering, latent-variable models, and Gaussian processes. Proofs and longer derivations appear in collapsed appendices at the end of each chapter.

## <a id="shared-conventions"></a>Shared conventions

- A dataset is $`D=\{(x_i,y_i)\}_{i=1}^n`$, with pairs drawn independently from a law $`P`$. There are $`n`$ observations and $`d`$ features. The design matrix $`X\in\mathbb R^{n\times d}`$ stores the inputs as rows, as in [Foundations](../foundations/README.md#shared-conventions). A classification problem has $`K`$ classes.
- A loss is written with the target first, $`\ell(y,\hat y)`$. The population risk $`R(f)`$ and empirical risk $`\widehat R_n(f)`$ are the quantities written $`R_*`$ and $`\widehat R_n`$ in [Information and Learning Theory](../foundations/05-information-and-learning-theory.md). $`R^*`$ is the Bayes risk.
- Binary labels are $`\{0,1\}`$ for probability models, with $`\eta(x)=P(Y=1\mid X=x)`$, and $`\{-1,+1\}`$ for margin-based methods: the perceptron, support vector machines, AdaBoost, and margin losses. Each chapter states which encoding it uses; the two are related by $`\tilde y=2y-1`$.
- Logarithms are natural. Vectors are columns, and $`\|\cdot\|`$ without a subscript is the Euclidean norm.
- Each code block runs on its own. Random seeds are fixed, and the comment lines at the end of a block record what it printed in the environment described in the [computing setup](sources/computing-setup.md).

## <a id="examples-and-supporting-resources"></a>Examples and supporting resources

The chapter text contains the definitions, derivations, and worked examples, and every figure is generated by a script in `Sources/Figure code`. The [computing setup](sources/computing-setup.md) records the Python environment and how to regenerate the figures, and [figure sources](sources/figure-sources.md) records the data and construction behind each figure.

The [reading plan](reading-plan.md) lists the lectures, readings, and scikit-learn examples for each topic. [Course links](sources/course-links.md) collects the two courses it draws on, [video links](sources/video-links.md) their lecture recordings by chapter, [books and documentation](sources/book-and-documentation-links.md) the reference texts and library guides, and [papers](sources/paper-links.md) the original and supplementary research cited in the chapters. The [CMU 10-601 homework archive](https://www.cs.cmu.edu/~ninamf/courses/601sp15/homeworks.shtml) supplies exercises; its theory and derivation questions fit the chapters directly.

## <a id="connections-to-later-modules"></a>Connections to later modules

**Deep learning** replaces fixed features with learned representations. The perceptron, logistic and softmax regression, and the losses of chapter 6 are the output layers and objectives of neural networks; regularization, early stopping, calibration, and double descent reappear at scale. The [DL overview](../dl/README.md) lists its chapters.

**Artificial intelligence** develops general graphical models and inference. The generative classifiers and Gaussian mixtures here are small graphical models, and EM is the standard way to fit such models when some variables are unobserved; the AI module applies it to hidden Markov models and general Bayesian networks. The [AI overview](../ai/README.md) lists its chapters.

**[NLP and large language models](../nlp-llms/README.md)** build on the softmax model, naive Bayes text classification, and learned embeddings, and the EM algorithm of chapter 14 learns the word alignments of statistical translation. Approximate nearest-neighbor search, introduced in chapter 17, powers retrieval over those embeddings.

**[Generative AI](../generative-ai/README.md)** extends Gaussian mixtures, probabilistic PCA, and the evidence lower bound of chapter 14 to deep latent-variable models, and replaces kernel density estimates with learned densities.

**Reinforcement learning** takes up sequential decisions. Bayesian optimization and active learning, in chapters 15 and 16, are the first places in this module where a learner chooses its own data, the problem that bandits and Markov decision processes treat in general.

## Reading plan and sources

- [Reading plan](reading-plan.md)
- [Book and documentation links](sources/book-and-documentation-links.md)
- [Computing setup](sources/computing-setup.md)
- [Course links](sources/course-links.md)
- [Figure sources](sources/figure-sources.md)
- [Paper links](sources/paper-links.md)
- [Video links](sources/video-links.md)
