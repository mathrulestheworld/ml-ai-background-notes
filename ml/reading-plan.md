[Background Notes](../README.md) › [Machine Learning](README.md)

# Reading plan

## <a id="courses"></a>Courses

- **Main course — CMU 10-601, Spring 2015: Tom Mitchell and Maria-Florina Balcan.** [Course and lecture schedule](https://www.cs.cmu.edu/~ninamf/courses/601sp15/lectures.shtml). Supplies the main mathematical development, formal learning theory, and unsupervised learning.
- **Supplement — Cornell CS4780/5780, Fall 2018: Kilian Weinberger.** [Course homepage](https://www.cs.cornell.edu/courses/cs4780/2018fa/) · [Topic notes](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/) · [Course-linked video playlist](https://www.youtube.com/playlist?list=PLl8OlHZGYOQ7bkVbuRthEsaLr7bONzbXS). Adds the introductory nearest-neighbor/perceptron sequence, classical supervised methods, ensembles, and Gaussian processes.

The order below combines the courses by topic. Each block contains the material to cover and its primary lectures; further reading and Python examples provide additional depth. Blocks 1–16 form the main plan; 17–18 are optional extensions. Timing is flexible.

CMU slide links are original PDFs. Cornell’s Fall 2018 topic notes are HTML and use Spring 2017 recordings. Topic numbers differ from recording numbers; the video labels below follow the recordings. Python links lead to official scikit-learn examples with downloadable notebooks.

## <a id="topic-list"></a>Topic list

- [ ] [1. Learning problems and prediction](#1-learning-problems-and-prediction)
- [ ] [2. Nearest neighbors and dimensionality](#2-nearest-neighbors-and-dimensionality)
- [ ] [3. Perceptron and linear separation](#3-perceptron-and-linear-separation)
- [ ] [4. Linear regression and regularization](#4-linear-regression-and-regularization)
- [ ] [5. Probabilistic estimation and generative classifiers](#5-probabilistic-estimation-and-generative-classifiers)
- [ ] [6. Logistic regression and probabilistic prediction](#6-logistic-regression-and-probabilistic-prediction)
- [ ] [7. Losses, model selection and bias–variance](#7-losses-model-selection-and-biasvariance)
- [ ] [8. Statistical learning theory](#8-statistical-learning-theory)
- [ ] [9. Margins, support vector machines and kernels](#9-margins-support-vector-machines-and-kernels)
- [ ] [10. Decision and regression trees](#10-decision-and-regression-trees)
- [ ] [11. Bagging and random forests](#11-bagging-and-random-forests)
- [ ] [12. AdaBoost and gradient boosting](#12-adaboost-and-gradient-boosting)
- [ ] [13. Principal components and dimensionality reduction](#13-principal-components-and-dimensionality-reduction)
- [ ] [14. Clustering](#14-clustering)
- [ ] [15. Gaussian mixtures and expectation maximization](#15-gaussian-mixtures-and-expectation-maximization)
- [ ] [16. Gaussian processes](#16-gaussian-processes)
- [ ] [17. Semi-supervised and active learning — optional](#17-semi-supervised-and-active-learning-optional)
- [ ] [18. Further classical methods — optional](#18-further-classical-methods-optional)

### <a id="reference-books-used-below"></a>Reference books used below

- **ESL:** Hastie, Tibshirani and Friedman, [The Elements of Statistical Learning, second edition — PDF, UCLA copy](https://faculty.stat.ucla.edu/ywu/research/documents/BOOKS/ElementsLearningII.pdf). Read the specified sections alongside the lectures.
- **UML:** Shalev-Shwartz and Ben-David, [Understanding Machine Learning: From Theory to Algorithms — publisher’s book page](https://www.cambridge.org/core/books/understanding-machine-learning/3059695661405D25673058E43C8BE2A6). An extended proof reference for topic 8. The author’s free-PDF server was unavailable when checked; the publisher’s full text may require library access. The CMU lectures and generalization notes linked below are freely accessible.

## <a id="1-learning-problems-and-prediction"></a>1. Learning problems and prediction

**Topics:** Inputs and targets; regression and classification; models versus fitting algorithms; hypothesis classes; loss, population risk and empirical risk; Bayes prediction; training, validation and test data; simple tree rules as an introductory model.

- **CMU, Jan 12: Introduction and decision trees.** [Slides · PDF](https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/01_DTreesAndOverfitting-1-12-2015.pdf) · [Video](https://www.youtube.com/watch?v=m4NlfvrRCdg) The first lecture introduces the course and a concrete learning algorithm.
- **Cornell topic 1: Machine learning setup.** [Notes · HTML](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote01_MLsetup.html) · [Video 1](https://www.youtube.com/watch?v=MrLPzBxG95I) · [Video 2](https://www.youtube.com/watch?v=zj-5nkNKAow&t=248)

**Further reading:** ESL §§2.1–2.5 supplies a statistical account of prediction and decision rules. The Foundations terminology chapter is the local terminology reference.

## <a id="2-nearest-neighbors-and-dimensionality"></a>2. Nearest neighbors and dimensionality

**Topics:** Nearest-neighbor classification and regression; the role of k; distance and feature scaling; local averaging; Bayes error and the 1-NN result; the curse of dimensionality. Search data structures are an optional extension.

- **Cornell topic 2: Nearest neighbors and the curse of dimensionality.** [Notes · HTML](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html) · [Video 3](https://www.youtube.com/watch?v=oymtGlGdT-k) · [Video 4](https://www.youtube.com/watch?v=BbYV8UfMJSA)

**Further reading:** ESL §2.3 compares least squares and nearest neighbors; §13.3 develops nearest-neighbor classifiers.

- **Python example:** [Nearest Neighbors Classification](https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html).

## <a id="3-perceptron-and-linear-separation"></a>3. Perceptron and linear separation

**Topics:** Hyperplanes and intercepts; signed labels; the perceptron update; separability and geometric margin; the mistake-bound proof; what fails without separability. Keep this elementary algorithm in ML before studying multilayer networks in DL.

- **Cornell topic 3: Perceptron.** [Notes · HTML](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote03.html) · [Video 5](https://www.youtube.com/watch?v=wl7gVvI-HuY) · [Video 6](https://www.youtube.com/watch?v=kObhWlqIeD8)
- **CMU handout:** [Perceptron algorithm and mistake-bound proof · PDF](https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/perceptron-notes.pdf). Pages 1–2 are the initial reading; the later PAC and margin refinements fit after topic 8.

**Further reading:** The margin-dependent mistake bound is a useful first complete algorithm analysis; distinguish it from an i.i.d. generalization guarantee.

## <a id="4-linear-regression-and-regularization"></a>4. Linear regression and regularization

**Topics:** Least squares; normal equations and projection geometry; Gaussian-noise likelihood; gradient-based fitting; ridge as shrinkage and MAP; lasso and elastic-net formulations; feature scaling and choosing regularization strength. Detailed lasso algorithms and regularization paths are supplementary reading.

- **CMU, Feb 4: Linear regression and the generative–discriminative comparison.** [Slides · PDF](https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/07_GenDiscr2_2-4-2015.pdf) · [Video](https://www.youtube.com/watch?v=wA5h9pD1qrU)
- **Cornell topic 8: Linear regression.** [Notes · HTML](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote08.html) · [Video 13](https://www.youtube.com/watch?v=_21o_ylL0q4)

**Further reading:** ESL §§3.2 and 3.4 cover least squares and shrinkage methods. Cornell’s ERM note in topic 7 collects ridge, lasso and elastic-net objectives. Use the Foundations linear-algebra chapter for the projection and stable-solve background.

- **Python example:** [Plot Ridge coefficients as a function of the regularization](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_path.html).

## <a id="5-probabilistic-estimation-and-generative-classifiers"></a>5. Probabilistic estimation and generative classifiers

**Topics:** Likelihood, MLE and MAP in learning problems; Bayes classification; conditional independence; categorical, multinomial and Gaussian naive Bayes; smoothing; Gaussian discriminant analysis and LDA/QDA; the assumptions behind decision boundaries. Full-covariance LDA/QDA comes from the additional reading, beyond CMU’s Gaussian naive Bayes.

- **CMU, Jan 21: Probability and estimation.** [Slides · PDF](https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/03_MLE_MAP_NBayes-1-21-2015.pdf) · [Video](https://www.youtube.com/watch?v=8dForUUki7s) Review selectively: estimation is already introduced in Foundations.
- **CMU, Jan 26: Naive Bayes.** [Slides · PDF](https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/04_NBayes-1-26-2015.pptx.pdf) · [Video](https://www.youtube.com/watch?v=ngKU5466i4o)
- **CMU, Jan 28: Gaussian naive Bayes.** [Slides · PDF](https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/05_GNB_1-28-2015.pdf) · [Video](https://www.youtube.com/watch?v=XObok-E2DTM)
- **Prose notes:** [Mitchell, Estimating Probabilities · PDF](https://www.cs.cmu.edu/~tom/mlbook/Joint_MLE_MAP.pdf); [Mitchell, Naive Bayes and Logistic Regression · PDF](https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf).

**Further reading:** ESL §4.3 supplies LDA/QDA. In Mitchell’s classifier chapter, read the Bayes/naive-Bayes portions here and retain the logistic-regression portions for topic 6.

## <a id="6-logistic-regression-and-probabilistic-prediction"></a>6. Logistic regression and probabilistic prediction

**Topics:** Conditional likelihood; logits, log-loss and gradients; regularized logistic regression; multiclass/softmax extension; generative versus discriminative estimation; predicted probabilities versus decisions and thresholds. Calibration is an explicit reference addition.

- **CMU, Feb 2: Logistic regression.** [Slides · PDF](https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/06_GenDiscr_LR_2-2-2015.pdf) · [Video](https://www.youtube.com/watch?v=z_xPu9KrgCY)
- **Cornell topic 6: Logistic regression, MLE and MAP.** [Notes · HTML](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote06.html) · [Video 11](https://www.youtube.com/watch?v=GnkDzIOxfzI)

**Further reading:** ESL §4.4 develops logistic regression, including multiclass modeling. Mitchell’s classifier chapter from topic 5 compares naive Bayes and logistic regression. For optimization, revisit the Foundations calculus chapter as needed.

- **Python example:** [Probability Calibration curves](https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html).

## <a id="7-losses-model-selection-and-biasvariance"></a>7. Losses, model selection and bias–variance

**Topics:** ERM and regularized ERM; squared, absolute, logistic, hinge and exponential losses; bias–variance decomposition; underfitting and overfitting; cross-validation; early stopping; model and hyperparameter selection. Extend the course treatment to nested validation, preprocessing leakage, imbalanced metrics and calibration.

- **Cornell topic 10: Empirical risk minimization.** [Notes · HTML](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote10.html) · [Video 15](https://www.youtube.com/watch?v=FwYNPomeBBg&t=1660s) · [Video 16](https://www.youtube.com/watch?v=AkmPv2WEsHw)
- **Cornell topic 12: Bias–variance tradeoff.** [Notes · HTML](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote12.html) · [Video 19](https://www.youtube.com/watch?v=zUJbRO0Wavo&t=81s) · [Video 20](https://www.youtube.com/watch?v=65UJPA10dW8)
- **Cornell topic 11: Model selection.** [Notes · HTML](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote11.html) · [Video 21](https://www.youtube.com/watch?v=a7cofmFgwIk)

**Further reading:** ESL §2.9 and chapter 7 provide the fuller assessment framework. [Scikit-learn: common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html) connects evaluation rules to actual pipelines.

- **Python example:** [Nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

## <a id="8-statistical-learning-theory"></a>8. Statistical learning theory

**Topics:** Realizable and agnostic PAC learning; finite-class sample complexity; concentration and uniform convergence; shattering, growth functions, VC dimension and Sauer’s lemma; Rademacher complexity; regularization and capacity control; approximation, estimation and optimization error. Work through representative proofs and compute dimensions/bounds for concrete classes.

- **CMU, Feb 9: Learning theory I.** [Slides · PDF](https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/08_Theory_2-9-2015.pdf) · [Video](https://www.youtube.com/watch?v=kyhYHMOEIk4)
- **CMU, Feb 11: Learning theory II.** [Slides · PDF](https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/09_Theory2_2-11-2015.pdf) · [Video](https://www.youtube.com/watch?v=3qbLJdjE3Nw)
- **CMU, Feb 16: Learning theory III.** [Slides · PDF](https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/10_Theory3_2-16-2015.pdf) · [Video](https://www.youtube.com/watch?v=N6oylcKNgds)
- **CMU prose notes:** [Generalization guarantees · PDF](https://www.cs.cmu.edu/~ninamf/courses/601sp15/sc-2015.pdf).

**Further reading:** UML chapters 2–6 give a sustained treatment of PAC learning, uniform convergence and VC theory; chapter 26 extends the complexity viewpoint to Rademacher bounds. These supply proofs and details beyond the three lectures. Foundations’ learning-theory chapter remains the compact local reference.

## <a id="9-margins-support-vector-machines-and-kernels"></a>9. Margins, support vector machines and kernels

**Topics:** Hard and soft margins; hinge loss; constrained and regularized formulations; Lagrange duality and support vectors; positive semidefinite kernels and feature maps; kernelized perceptron and SVMs; kernel width and regularization. Kernel ridge regression is a short extension.

- **CMU, Mar 23: Margins and kernels.** [Slides · PDF](https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/17_margins-kernels_03-23-2015.pdf) · [Video](https://www.youtube.com/watch?v=T6AksiAVJ0k)
- **CMU, Mar 25: Support vector machines.** [Slides · PDF](https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/18_svm-ssl_03-25-2015.pdf) · [Video](https://www.youtube.com/watch?v=JoJhXsdTWxM) The semi-supervised portion can wait until topic 17.
- **Cornell topic 13: Kernels.** [Notes · HTML](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote13.html) · [Video 22](https://www.youtube.com/watch?v=FgTQG2IozlM&t=31s) · [Video 23](https://www.youtube.com/watch?v=erqL3y2es1I)
- **Cornell topic 14: Kernels continued.** [Notes · HTML](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote14.html) · [Video 24](https://www.youtube.com/watch?v=RwF1esLCG4U&t=307s) · [Video 25](https://www.youtube.com/watch?v=7LlpTABc27s)

**Further reading:** ESL chapter 12 develops margin methods. [Welling, Kernel Ridge Regression · PDF](https://web2.qatar.cmu.edu/~gdicaro/10315-Fall19/additional/welling-notes-on-kernel-ridge.pdf) supplies the kernel-ridge derivation. Cornell video 21, linked under model selection, introduces kernels; continue with videos 22–25 here.

- **Python example:** [SVM Margins Example](https://scikit-learn.org/stable/auto_examples/svm/plot_svm_margin.html).

## <a id="10-decision-and-regression-trees"></a>10. Decision and regression trees

**Topics:** Recursive partitioning; entropy, information gain and Gini impurity; greedy split selection; categorical and continuous features; regression leaves; stopping and pruning; depth, instability and overfitting. Relate the objective to the tree-building algorithm.

- **CMU, Jan 14: Decision trees and overfitting.** [Slides · PDF](https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/02_Overfitting_ProbReview-1-14-2015.pdf) · [Video](https://www.youtube.com/watch?v=xXEjtF_hs-4) Read the tree/overfitting part; the probability review is optional.
- **Cornell topic 17: Decision and regression trees.** [Notes · HTML](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote17.html) · [Video 28](https://www.youtube.com/watch?v=E1_WCdUAtyE) · [Video 29](https://www.youtube.com/watch?v=a3ioGSwfVpE)

**Further reading:** ESL §9.2 supplies a fuller CART treatment, including pruning. The introductory CMU lecture in topic 1 provides the first tree example.

- **Python example:** [Decision Tree Regression](https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html).

## <a id="11-bagging-and-random-forests"></a>11. Bagging and random forests

**Topics:** Bootstrap samples; averaging unstable learners; variance and correlation; bagged trees; feature subsampling; random forests; out-of-bag evaluation; feature importance and its limitations.

- **Cornell topic 18: Bagging and random forests.** [Notes · HTML](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote18.html) · [Video 30](https://www.youtube.com/watch?v=0LB1cy2sCXc) · [Video 31](https://www.youtube.com/watch?v=4EOCQJgqAOY)

**Further reading:** ESL §8.7 covers bagging and chapter 15 treats random forests. [Breiman, Random Forests (2001) · PDF](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) is the optional original paper.

- **Python example:** [OOB Errors for Random Forests](https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html).

## <a id="12-adaboost-and-gradient-boosting"></a>12. AdaBoost and gradient boosting

**Topics:** Weak and strong learning; weighted training examples; AdaBoost updates and exponential loss; training-error guarantees and margins; additive models; gradient descent in function space; gradient-boosted regression trees; shrinkage, tree depth and stopping.

- **CMU, Mar 16: Boosting.** [Slides · PDF](https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/15_boosting_3-16-2015.pdf) · [Video](https://www.youtube.com/watch?v=28aAjtyzrRE)
- **CMU, Mar 18: AdaBoost guarantees and margins.** [Slides · PDF](https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/16_boosting-percepton-margins_03-18-2015.pdf) · [Video](https://www.youtube.com/watch?v=qI_QNnDN26k) The perceptron portion revisits topic 3.
- **Cornell topic 19: Gradient boosting and AdaBoost.** [Notes · HTML](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote19.html) · [Video 32](https://www.youtube.com/watch?v=dosOtgSdbnY) · [Video 33](https://www.youtube.com/watch?v=Vd6hzcwEa2k) · [Video 34](https://www.youtube.com/watch?v=toOAToTaGV4)

**Further reading:** ESL chapter 10 supplies the statistical/additive-model view. [Schapire, The Boosting Approach to Machine Learning: An Overview · PDF](https://www.cs.princeton.edu/courses/archive/spr08/cos424/readings/Schapire2003.pdf) develops the theory and margin perspective.

- **Python example:** [Gradient Boosting regression](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html).

## <a id="13-principal-components-and-dimensionality-reduction"></a>13. Principal components and dimensionality reduction

**Topics:** Centering; variance maximization and reconstruction error; SVD and PCA scores; retained dimension; scaling versus standardization; principal-component regression; kernel PCA. Treat PCA as a fitted transformation whose parameters come from training data.

- **CMU, Apr 8: PCA and kernel PCA.** [Slides · PDF](https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/22_pca-04-09-2015.pdf) · [Video](https://www.youtube.com/watch?v=ds8l8prOsvU)

**Further reading:** ESL §14.5.1 gives the PCA formulation and §3.5 discusses regression on derived inputs. The Foundations linear-algebra chapter contains the required SVD background.

- **Python example:** [PCA followed by logistic regression in a pipeline](https://scikit-learn.org/stable/auto_examples/compose/plot_digits_pipe.html).

## <a id="14-clustering"></a>14. Clustering

**Topics:** Clustering objectives and the role of distance; k-means and Lloyd iterations; initialization, local optima and k-means++; hierarchical/agglomerative clustering and linkage; cluster number; stability and interpretation.

- **CMU, Apr 6: Partitional and hierarchical clustering.** [Slides · PDF](https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/21_clustering_4-6-2015.pdf) · [Video](https://www.youtube.com/watch?v=p3cB23CzpzQ)

**Further reading:** ESL §14.3 gives broader clustering coverage. Keep objective value, cluster stability and agreement with external labels conceptually separate.

- **Python example:** [A demo of K-Means clustering on the handwritten digits data](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html).

## <a id="15-gaussian-mixtures-and-expectation-maximization"></a>15. Gaussian mixtures and expectation maximization

**Topics:** Latent component labels; mixture likelihood; soft assignments; complete versus observed data; E and M steps; the lower-bound/Jensen derivation; likelihood monotonicity; initialization, singular covariance and model selection; comparison with k-means.

- **CMU, Feb 25: Learning with hidden variables and EM.** [Slides · PDF](https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/13_GrMod3_2-25-2015.pdf) · [Video](https://www.youtube.com/watch?v=E16cqVHXow8) Read the latent-variable/EM development. General Bayesian-network representation and inference belong in the AI module.
- **CMU, Mar 4: Gaussian mixtures, EM and clustering.** [Slides · PDF](https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/14_GrMod4_3-4-2015.pdf) · [Video](https://www.youtube.com/watch?v=ek0JtB1Ftn0)

**Further reading:** ESL §8.5 develops EM, starting with a Gaussian-mixture example. Separate a guarantee of nondecreasing likelihood from a guarantee of a global optimum.

- **Python example:** [Gaussian Mixture Model Selection](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html).

## <a id="16-gaussian-processes"></a>16. Gaussian processes

**Topics:** Bayesian linear regression as motivation; distributions over functions; covariance kernels; conditioning for prediction; predictive mean and uncertainty; observation noise and hyperparameter fitting. Bayesian optimization is an optional extension.

- **Cornell topic 15: Gaussian processes.** [Notes · HTML](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote15.html) · [Video 26](https://www.youtube.com/watch?v=R-NUdqxKjos) · [Video 27](https://www.youtube.com/watch?v=BzHJ57QCdVo) The later k-d-tree/ball-tree portion of video 27 belongs to optional topic 18.
- **Textbook chapter:** [Rasmussen and Williams, Gaussian Processes for Machine Learning, chapter 2 · PDF](https://gaussianprocess.org/gpml/chapters/RW2.pdf). Sections 2.1–2.3 are the initial reading.

**Further reading:** The [GPML chapter index](https://gaussianprocess.org/gpml/chapters/) links chapter 4 on covariance functions and chapter 5 on hyperparameters.

- **Python example:** [Gaussian Processes regression: basic introductory example](https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html).

## <a id="17-semi-supervised-and-active-learning-optional"></a>17. Semi-supervised and active learning — optional

**Topics:** How unlabeled data can help under additional assumptions; co-training and graph-based methods; querying labels; uncertainty sampling; selection bias and label complexity. This is an extension after the supervised and unsupervised core.

- **CMU, Mar 30: Semi-supervised learning.** [Slides · PDF](https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/19_ssl_03-30-2015.pdf) · [Video](https://www.youtube.com/watch?v=gnNLjX50F7U)
- **CMU, Apr 1: Active learning.** [Slides · PDF](https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/20_al_4-1-2015.pdf) · [Video](https://www.youtube.com/watch?v=2BZhsEakEH8)

**Further reading:** [Zhu, Semi-Supervised Learning · PDF](https://pages.cs.wisc.edu/~jerryzhu/pub/SSL_EoML.pdf) and [Settles, Active Learning Literature Survey · PDF](https://burrsettles.com/pub/settles.activelearning.pdf) are optional surveys.

## <a id="18-further-classical-methods-optional"></a>18. Further classical methods — optional

**Topics:** Kernel density estimation and bandwidth; local regression; basis expansions, splines and generalized additive models; efficient nearest-neighbor search. These extend the breadth of the selected courses.

- **Cornell topic 16: k-d trees and ball trees.** [Notes · HTML](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote16.html) Use the relevant portions of video 27 (topic 16) and video 28 (topic 10).

**Further reading:** ESL chapter 5 covers basis expansions and splines; chapter 6 covers kernel smoothing and density estimation; §9.1 covers generalized additive models. Quant Prep’s nonparametric-regression, density-estimation and evaluation sections are an additional local reference.

- **Python example:** [Simple 1D Kernel Density Estimation](https://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html).
- **Python example:** [Polynomial and Spline interpolation](https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html).

## <a id="exercises-and-local-references"></a>Exercises and local references

- [CMU homework archive](https://www.cs.cmu.edu/~ninamf/courses/601sp15/homeworks.shtml) and [recitations](https://www.cs.cmu.edu/~ninamf/courses/601sp15/recitations.shtml). Homework PDFs are available; the archive’s solution links returned 404 when checked. Some programming exercises use Octave/MATLAB and would need adaptation for Python. The theory and derivation questions remain useful.
- Foundations overview links the mathematical and numerical prerequisites. Its existing notes are references during this module.
- NumPy implementations of small algorithms can accompany the notes you write. The linked scikit-learn notebooks provide concrete comparisons, plots and evaluation examples.

## <a id="connections-to-later-modules"></a>Connections to later modules

Multilayer networks and deep-learning training belong in DL; general graphical-model inference in AI; sequence and language models in NLP; deep generative models in Generative AI; bandits, MDPs and control in RL. Classical generative classifiers, Gaussian mixtures, EM, the perceptron and generalization theory remain in this module.
