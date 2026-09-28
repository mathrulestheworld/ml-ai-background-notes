[Background Notes](../../README.md) › [Machine Learning](../README.md)

# Book and documentation links

# <a id="books-and-documentation"></a>Books and documentation

The chapter notes develop the module's main material. These books and official documentation provide alternative explanations, fuller proofs, and the details of the library interfaces used in the code. The [reading plan](../reading-plan.md) lists the lecture slides, course notes, and scikit-learn examples for each topic.

## <a id="main-references"></a>Main references

| Source | Relevant material |
| --- | --- |
| Hastie, Tibshirani, and Friedman, [*The Elements of Statistical Learning*](https://hastie.su.domains/ElemStatLearn/), 2nd edition | The statistical reference for most chapters: §§2.3–2.9 (chapters 1 and 6), 3 (chapter 3), 4 (chapters 4–5), 5–6 and §9.1 (chapter 17), 7 (chapter 6), §8.5 (chapter 14), §8.7 and 15 (chapter 10), §9.2 (chapter 9), 10 (chapter 11), 12 (chapter 8), §13.3 (chapter 1), §§14.3 and 14.5 (chapters 12–13). The authors' site hosts the PDF. |
| Shalev-Shwartz and Ben-David, [*Understanding Machine Learning*](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) | Chapters 2–6 for PAC learning, uniform convergence, and VC dimension; chapter 7 for structural risk minimization; chapter 21 for online learning; chapter 26 for Rademacher complexity. The proof reference for chapter 7. If the authors' server is unavailable, the [publisher's page](https://www.cambridge.org/core/books/understanding-machine-learning/3059695661405D25673058E43C8BE2A6) has the full text, which may require library access. |
| Bishop, [*Pattern Recognition and Machine Learning*](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) | Chapter 4 for linear classifiers, chapter 6 for kernels and Gaussian processes, chapter 9 for mixtures and EM, and chapter 12 for probabilistic PCA. |
| Murphy, [*Probabilistic Machine Learning: An Introduction*](https://probml.github.io/pml-book/book1.html) | A probabilistic account of the same supervised and unsupervised methods, with current notation; also listed in the Foundations references. |

## <a id="specialized-books"></a>Specialized books

| Source | Relevant material |
| --- | --- |
| Devroye, Györfi, and Lugosi, [*A Probabilistic Theory of Pattern Recognition*](https://doi.org/10.1007/978-1-4612-0711-5) | Consistency of nearest-neighbor and partitioning rules, Stone's theorem, and distribution-free lower bounds (chapter 1). |
| Györfi, Kohler, Krzyżak, and Walk, [*A Distribution-Free Theory of Nonparametric Regression*](https://doi.org/10.1007/b97848) | Rates of convergence for local averaging, nearest-neighbor, and partitioning estimates (chapters 1 and 9). |
| Kearns and Vazirani, [*An Introduction to Computational Learning Theory*](https://mitpress.mit.edu/9780262111935/an-introduction-to-computational-learning-theory/) | PAC learning of concrete classes, Occam's razor, and computational hardness of learning (chapter 7). |
| Minsky and Papert, [*Perceptrons*](https://direct.mit.edu/books/monograph/3132/PerceptronsAn-Introduction-to-Computational) | What single-layer perceptrons cannot represent (chapter 2). |
| Schölkopf and Smola, [*Learning with Kernels*](https://mitpress.mit.edu/9780262536578/learning-with-kernels/) | Kernels, reproducing kernel Hilbert spaces, support vector machines, and kernel PCA (chapters 8 and 12). |
| Breiman, Friedman, Olshen, and Stone, [*Classification and Regression Trees*](https://www.routledge.com/Classification-and-Regression-Trees/Breiman-Friedman-Stone-Olshen/p/book/9780412048418) | The CART algorithm, surrogate splits, and cost-complexity pruning (chapter 9). |
| Quinlan, [*C4.5: Programs for Machine Learning*](https://dl.acm.org/doi/10.5555/583200) | Information-gain trees, gain ratio, and rule extraction (chapter 9). |
| Schapire and Freund, [*Boosting: Foundations and Algorithms*](https://direct.mit.edu/books/oa-monograph/5342/BoostingFoundations-and-Algorithms) | AdaBoost, its training and generalization analyses, margins, and game-theoretic views; open access (chapter 11). |
| Rasmussen and Williams, [*Gaussian Processes for Machine Learning*](https://gaussianprocess.org/gpml/chapters/) | Chapter 2 for regression, chapter 4 for covariance functions, chapter 5 for the marginal likelihood, and chapter 6 for the relation to kernel methods and splines; free chapters (chapter 15). |
| Chapelle, Schölkopf, and Zien (eds.), [*Semi-Supervised Learning*](https://mitpress.mit.edu/9780262514125/semi-supervised-learning/) | Assumptions, generative, low-density, and graph-based methods, and benchmark comparisons (chapter 16). |
| Silverman, [*Density Estimation for Statistics and Data Analysis*](https://www.routledge.com/Density-Estimation-for-Statistics-and-Data-Analysis/Silverman/p/book/9780412246203) | Kernel density estimation and bandwidth selection (chapter 17). |
| Neal, [*Bayesian Learning for Neural Networks*](https://link.springer.com/book/10.1007/978-1-4612-0745-0) | The limit in which a wide Bayesian neural network becomes a Gaussian process (chapter 15). |

## <a id="notes-and-guides"></a>Notes and guides

| Source | Relevant material |
| --- | --- |
| Mitchell, [*Generative and Discriminative Classifiers: Naive Bayes and Logistic Regression*](https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf) | A book-chapter draft comparing the two estimators (chapters 4–5). |
| Welling, [*Kernel Ridge Regression*](https://web2.qatar.cmu.edu/~gdicaro/10315-Fall19/additional/welling-notes-on-kernel-ridge.pdf) | A short derivation of the dual solution (chapter 8). |
| Hsu, Chang, and Lin, [*A Practical Guide to Support Vector Classification*](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf) | Scaling, kernel choice, and grid search over $`(C,\gamma)`$ (chapter 8). |

## <a id="library-documentation"></a>Library documentation

The links point to the current stable documentation. The code in the notes was tested with scikit-learn 1.4.2, whose documentation is archived at [scikit-learn.org/1.4](https://scikit-learn.org/1.4/); defaults and parameter names occasionally change between versions.

| Documentation | Relevant material |
| --- | --- |
| [scikit-learn nearest neighbors](https://scikit-learn.org/stable/modules/neighbors.html) and [density estimation](https://scikit-learn.org/stable/modules/density.html) | $`k`$-NN classifiers and regressors, $`k`$-d trees and ball trees, and `KernelDensity` (chapters 1 and 17). |
| [scikit-learn linear models](https://scikit-learn.org/stable/modules/linear_model.html) | Least squares, [`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), [`Lasso`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html), [`ElasticNet`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html), and [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), including their penalty parameterizations (chapters 3 and 5). |
| [scikit-learn naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html) and [discriminant analysis](https://scikit-learn.org/stable/modules/lda_qda.html) | Event models, smoothing, LDA, QDA, and shrinkage covariance estimates (chapter 4). |
| [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html) | Reliability diagrams, Platt scaling, and isotonic regression (chapter 5). |
| [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html), [model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html), and [common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) | Splitters, scoring rules, metrics, pipelines, and data leakage (chapter 6). |
| [scikit-learn support vector machines](https://scikit-learn.org/stable/modules/svm.html) and [kernel approximation](https://scikit-learn.org/stable/modules/kernel_approximation.html) | `SVC`, `LinearSVC`, kernel parameters, multiclass strategies, Nyström and random Fourier features (chapter 8). |
| [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) | The solver behind scikit-learn's `SVC` (chapter 8). |
| [scikit-learn decision trees](https://scikit-learn.org/stable/modules/tree.html) | Splitting criteria, missing values, and cost-complexity pruning (chapter 9). |
| [scikit-learn ensembles](https://scikit-learn.org/stable/modules/ensemble.html) | Bagging, random forests, extremely randomized trees, AdaBoost, and gradient boosting, including histogram-based boosting (chapters 10–11). |
| [scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html) and [partial dependence](https://scikit-learn.org/stable/modules/partial_dependence.html) | Model inspection for forests and boosted trees (chapters 10–11). |
| [scikit-learn matrix decompositions](https://scikit-learn.org/stable/modules/decomposition.html) | PCA, probabilistic PCA scores, and kernel PCA (chapter 12). |
| [scikit-learn clustering](https://scikit-learn.org/stable/modules/clustering.html) | $`k`$-means, agglomerative clustering, DBSCAN, spectral clustering, and clustering metrics (chapter 13). |
| [scikit-learn Gaussian mixtures](https://scikit-learn.org/stable/modules/mixture.html) | EM for mixtures, covariance types, initialization, and model selection (chapter 14). |
| [scikit-learn Gaussian processes](https://scikit-learn.org/stable/modules/gaussian_process.html) | Kernels, hyperparameter fitting, and `GaussianProcessRegressor` (chapter 15). |
| [scikit-learn semi-supervised learning](https://scikit-learn.org/stable/modules/semi_supervised.html) | Self-training, label propagation, and label spreading (chapter 16). |
| [SciPy `make_smoothing_spline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_smoothing_spline.html) and [scikit-learn `SplineTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html) | Smoothing splines with generalized cross-validation, and B-spline features for linear models (chapter 17). |
