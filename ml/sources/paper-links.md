[ML Mastery Notes](../../README.md) › [Machine Learning](../README.md)

# Paper links

# <a id="papers"></a>Papers

The notes link these works where their ideas arise. This index collects the original papers and principal analyses, grouped by the chapters that use them; surveys that make good follow-up reading are marked.

## <a id="prediction-nearest-neighbors-and-linear-methods"></a>Prediction, nearest neighbors, and linear methods

| Work | Idea developed in the module |
| --- | --- |
| Cover and Hart, [*Nearest Neighbor Pattern Classification*](https://doi.org/10.1109/TIT.1967.1053964) (1967) | The asymptotic 1-NN error lies between $R^*$ and $2R^*(1-R^*)$ (chapter 1). |
| Stone, [*Consistent Nonparametric Regression*](https://doi.org/10.1214/aos/1176343886) (1977) | Conditions under which local averaging rules, including $k$-NN, are universally consistent (chapter 1). |
| Stone, [*Optimal Global Rates of Convergence for Nonparametric Regression*](https://doi.org/10.1214/aos/1176345969) (1982) | The minimax rate $n^{-2/(d+2)}$ for Lipschitz regression functions and its dependence on dimension (chapter 1). |
| Beyer, Goldstein, Ramakrishnan, and Shaft, [*When Is "Nearest Neighbor" Meaningful?*](https://minds.wisconsin.edu/handle/1793/60174) (1999) | Concentration of distances in high dimension (chapter 1). |
| Kpotufe, [*k-NN Regression Adapts to Local Intrinsic Dimension*](https://proceedings.neurips.cc/paper/2011/hash/05f971b5ec196b8c65b75d2ef8267331-Abstract.html) (2011) | Why nearest neighbors can work on high-dimensional data of low intrinsic dimension (chapter 1). |
| Rosenblatt, [*The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain*](https://doi.org/10.1037/h0042519) (1958) | The perceptron (chapter 2). |
| Freund and Schapire, [*Large Margin Classification Using the Perceptron Algorithm*](https://link.springer.com/article/10.1023/A:1007662407062) (1999) | Mistake bounds without separability, voted and averaged perceptrons, and kernelization (chapters 2 and 8). |
| Hoerl and Kennard, [*Ridge Regression: Biased Estimation for Nonorthogonal Problems*](https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634) (1970) | Ridge regression (chapter 3). |
| Tibshirani, [*Regression Shrinkage and Selection via the Lasso*](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x) (1996) | The lasso (chapter 3). |
| Efron, Hastie, Johnstone, and Tibshirani, [*Least Angle Regression*](https://doi.org/10.1214/009053604000000067) (2004) | The piecewise-linear lasso path (chapter 3). |
| Zou and Hastie, [*Regularization and Variable Selection via the Elastic Net*](https://doi.org/10.1111/j.1467-9868.2005.00503.x) (2005) | The elastic net and grouped selection of correlated features (chapter 3). |
| Friedman, Hastie, and Tibshirani, [*Regularization Paths for Generalized Linear Models via Coordinate Descent*](https://www.jstatsoft.org/v33/i01/) (2010) | Coordinate descent with warm starts along a penalty path (chapter 3). |
| Fisher, [*The Use of Multiple Measurements in Taxonomic Problems*](https://onlinelibrary.wiley.com/doi/10.1111/j.1469-1809.1936.tb02137.x) (1936) | Fisher's linear discriminant (chapter 4). |
| Domingos and Pazzani, [*On the Optimality of the Simple Bayesian Classifier under Zero-One Loss*](https://link.springer.com/article/10.1023/A:1007413511361) (1997) | Why naive Bayes can classify well with poor probability estimates (chapter 4). |
| McCallum and Nigam, [*A Comparison of Event Models for Naive Bayes Text Classification*](https://aaai.org/papers/041-ws98-05-007/) (1998) | Bernoulli and multinomial event models (chapter 4). |
| Friedman, [*Regularized Discriminant Analysis*](https://www.tandfonline.com/doi/abs/10.1080/01621459.1989.10478752) (1989) | Shrinking between QDA, LDA, and a scaled identity (chapter 4). |
| Ng and Jordan, [*On Discriminative vs. Generative Classifiers: A Comparison of Logistic Regression and Naive Bayes*](https://proceedings.neurips.cc/paper/2001/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html) (2001) | Generative estimates converge faster to a worse asymptote (chapters 4–5). |
| Efron, [*The Efficiency of Logistic Regression Compared to Normal Discriminant Analysis*](https://www.tandfonline.com/doi/abs/10.1080/01621459.1975.10480319) (1975) | The statistical cost of discriminative estimation when the Gaussian model is true (chapter 5). |
| Albert and Anderson, [*On the Existence of Maximum Likelihood Estimates in Logistic Regression Models*](https://academic.oup.com/biomet/article-abstract/71/1/1/349338) (1984) | Separation and the nonexistence of the MLE (chapter 5). |
| Soudry, Hoffer, Nacson, Gunasekar, and Srebro, [*The Implicit Bias of Gradient Descent on Separable Data*](https://jmlr.org/papers/v19/18-188.html) (2018) | Gradient descent on separable data converges in direction to the maximum-margin separator (chapter 5). |
| Gneiting and Raftery, [*Strictly Proper Scoring Rules, Prediction, and Estimation*](https://www.tandfonline.com/doi/abs/10.1198/016214506000001437) (2007) | Proper scoring rules (chapter 5). |
| Niculescu-Mizil and Caruana, [*Predicting Good Probabilities with Supervised Learning*](https://dl.acm.org/doi/10.1145/1102351.1102430) (2005) | Calibration of common classifiers, Platt scaling, and isotonic regression (chapter 5). |
| Guo, Pleiss, Sun, and Weinberger, [*On Calibration of Modern Neural Networks*](https://arxiv.org/abs/1706.04599) (2017) | Temperature scaling and the miscalibration of large models (chapter 5). |

## <a id="model-assessment-and-learning-theory"></a>Model assessment and learning theory

| Work | Idea developed in the module |
| --- | --- |
| Zhang, [*Statistical Behavior and Consistency of Classification Methods Based on Convex Risk Minimization*](https://www.semanticscholar.org/paper/Statistical-behavior-and-consistency-of-methods-on-Zhang/7678da8b2eb70a5383f203d948564d8f48c0c62a) (2004) | What each convex surrogate estimates (chapter 6). |
| Bartlett, Jordan, and McAuliffe, [*Convexity, Classification, and Risk Bounds*](https://www.tandfonline.com/doi/abs/10.1198/016214505000000907) (2006) | Classification calibration of surrogate losses (chapter 6). |
| Belkin, Hsu, Ma, and Mandal, [*Reconciling Modern Machine-Learning Practice and the Classical Bias–Variance Trade-off*](https://www.pnas.org/doi/abs/10.1073/pnas.1903070116) (2019) | Double descent (chapter 6). |
| Hastie, Montanari, Rosset, and Tibshirani, [*Surprises in High-Dimensional Ridgeless Least Squares Interpolation*](https://arxiv.org/abs/1903.08560) (2022) | Exact risk of minimum-norm interpolation (chapter 6). |
| Efron, [*The Estimation of Prediction Error: Covariance Penalties and Cross-Validation*](https://www.tandfonline.com/doi/abs/10.1198/016214504000000692) (2004) | Optimism and covariance penalties (chapter 6). |
| Bates, Hastie, and Tibshirani, [*Cross-Validation: What Does It Estimate and How Well Does It Do It?*](https://arxiv.org/abs/2104.00673) (2023) | What the cross-validation estimate targets, why its naive standard error is too small, and nested cross-validation for honest intervals (chapter 6). |
| Cawley and Talbot, [*On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*](https://www.jmlr.org/papers/v11/cawley10a.html) (2010) | Selection bias and nested cross-validation (chapter 6). |
| Kaufman, Rosset, Perlich, and Stitelman, [*Leakage in Data Mining: Formulation, Detection, and Avoidance*](https://dl.acm.org/doi/10.1145/2382577.2382579) (2012) | Data leakage (chapter 6). |
| Fawcett, [*An Introduction to ROC Analysis*](https://www.sciencedirect.com/science/article/abs/pii/S016786550500303X) (2006) | ROC curves and AUC (chapter 6). |
| Dietterich, [*Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms*](https://dl.acm.org/doi/10.1162/089976698300017197) (1998) | McNemar's test and the pitfalls of resampled $t$ tests (chapter 6). |
| Nadeau and Bengio, [*Inference for the Generalization Error*](https://link.springer.com/article/10.1023/A:1024068626366) (2003) | The corrected resampled $t$ test (chapter 6). |
| Blumer, Ehrenfeucht, Haussler, and Warmuth, [*Learnability and the Vapnik–Chervonenkis Dimension*](https://dl.acm.org/doi/10.1145/76359.76371) (1989) | Finite VC dimension characterizes PAC learnability (chapter 7). |
| Sauer, [*On the Density of Families of Sets*](https://www.sciencedirect.com/science/article/pii/0097316572900192) (1972) | Sauer's lemma (chapter 7). |
| Ehrenfeucht, Haussler, Kearns, and Valiant, [*A General Lower Bound on the Number of Examples Needed for Learning*](https://www.sciencedirect.com/science/article/pii/0890540189900023) (1989) | The $\Omega(v/\varepsilon)$ realizable lower bound for VC dimension $v$ (chapter 7). |
| Hanneke, [*The Optimal Sample Complexity of PAC Learning*](https://jmlr.org/papers/v17/15-389.html) (2016) | Removing the $\log(1/\varepsilon)$ factor from the realizable upper bound (chapter 7). |
| Pitt and Valiant, [*Computational Limitations on Learning from Examples*](https://dl.acm.org/doi/10.1145/48014.63140) (1988) | Representation-dependent hardness of learning (chapter 7). |
| Littlestone, [*Learning Quickly When Irrelevant Attributes Abound: A New Linear-Threshold Algorithm*](https://link.springer.com/article/10.1023/A:1022869011914) (1988) | The mistake-bound model and the Littlestone dimension (chapter 7). |
| Littlestone and Warmuth, [*The Weighted Majority Algorithm*](https://www.sciencedirect.com/science/article/pii/S0890540184710091) (1994) | Learning from expert advice (chapter 7). |

## <a id="kernels-trees-and-ensembles"></a>Kernels, trees, and ensembles

| Work | Idea developed in the module |
| --- | --- |
| Boser, Guyon, and Vapnik, [*A Training Algorithm for Optimal Margin Classifiers*](https://dl.acm.org/doi/10.1145/130385.130401) (1992) | The kernelized maximum-margin classifier (chapter 8). |
| Cortes and Vapnik, [*Support-Vector Networks*](https://link.springer.com/article/10.1007/BF00994018) (1995) | The soft-margin SVM (chapter 8). |
| Platt, [*Sequential Minimal Optimization*](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/) (1998) | Solving the SVM dual two variables at a time (chapter 8). |
| Aronszajn, [*Theory of Reproducing Kernels*](https://www.ams.org/journals/tran/1950-068-03/S0002-9947-1950-0051437-7/) (1950) | Reproducing kernel Hilbert spaces (chapter 8). |
| Schölkopf, Herbrich, and Smola, [*A Generalized Representer Theorem*](https://link.springer.com/chapter/10.1007/3-540-44581-1_27) (2001) | The representer theorem (chapter 8). |
| Williams and Seeger, [*Using the Nyström Method to Speed Up Kernel Machines*](https://proceedings.neurips.cc/paper/2000/hash/19de10adbaa1b2ee13f77f679fa1483a-Abstract.html) (2000) | Low-rank kernel approximation (chapter 8). |
| Rahimi and Recht, [*Random Features for Large-Scale Kernel Machines*](https://proceedings.neurips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html) (2007) | Random Fourier features (chapter 8). |
| Crammer and Singer, [*On the Algorithmic Implementation of Multiclass Kernel-Based Vector Machines*](https://www.jmlr.org/papers/v2/crammer01a.html) (2001) | The multiclass SVM (chapter 8). |
| Hyafil and Rivest, [*Constructing Optimal Binary Decision Trees Is NP-Complete*](https://www.sciencedirect.com/science/article/abs/pii/0020019076900958) (1976) | Why trees are grown greedily (chapter 9). |
| Quinlan, [*Induction of Decision Trees*](https://link.springer.com/article/10.1007/BF00116251) (1986) | ID3 and information gain (chapter 9). |
| Loh, [*Fifty Years of Classification and Regression Trees*](https://onlinelibrary.wiley.com/doi/abs/10.1111/insr.12016) (2014) | Survey of tree algorithms and their selection biases (chapter 9). |
| Breiman, [*Bagging Predictors*](https://link.springer.com/article/10.1007/BF00058655) (1996) | Bootstrap aggregation of unstable learners (chapter 10). |
| Krogh and Vedelsby, [*Neural Network Ensembles, Cross Validation, and Active Learning*](https://proceedings.neurips.cc/paper/1994/hash/b8c37e33defde51cf91e1e03e51657da-Abstract.html) (1994) | The ambiguity decomposition (chapter 10). |
| Breiman, [*Random Forests*](https://link.springer.com/article/10.1023/A:1010933404324) (2001) | Random forests, strength and correlation, and out-of-bag estimates (chapter 10). |
| Geurts, Ernst, and Wehenkel, [*Extremely Randomized Trees*](https://link.springer.com/article/10.1007/s10994-006-6226-1) (2006) | Random thresholds as further decorrelation (chapter 10). |
| Lin and Jeon, [*Random Forests and Adaptive Nearest Neighbors*](https://www.tandfonline.com/doi/abs/10.1198/016214505000001230) (2006) | Forests as adaptive weighted neighbors (chapter 10). |
| Biau and Scornet, [*A Random Forest Guided Tour*](https://link.springer.com/article/10.1007/s11749-016-0481-7) (2016) | Survey of the theory of random forests (chapter 10). |
| Strobl, Boulesteix, Zeileis, and Hothorn, [*Bias in Random Forest Variable Importance Measures*](https://pmc.ncbi.nlm.nih.gov/articles/PMC1796903/) (2007) | The bias of impurity importance toward features with many split points (chapter 10). |
| Schapire, [*The Strength of Weak Learnability*](https://link.springer.com/article/10.1007/BF00116037) (1990) | Weak learning implies strong learning (chapter 11). |
| Freund and Schapire, [*A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting*](https://www.sciencedirect.com/science/article/pii/S002200009791504X) (1997) | AdaBoost and its training-error bound (chapter 11). |
| Schapire, Freund, Bartlett, and Lee, [*Boosting the Margin*](https://projecteuclid.org/journals/annals-of-statistics/volume-26/issue-5/Boosting-the-margin--a-new-explanation-for-the-effectiveness/10.1214/aos/1024691352.full) (1998) | Margin-based generalization bounds for voting classifiers (chapter 11). |
| Friedman, Hastie, and Tibshirani, [*Additive Logistic Regression: A Statistical View of Boosting*](https://projecteuclid.org/journals/annals-of-statistics/volume-28/issue-2/Additive-logistic-regression--a-statistical-view-of-boosting-With/10.1214/aos/1016218223.full) (2000) | AdaBoost as stagewise minimization of the exponential loss (chapter 11). |
| Friedman, [*Greedy Function Approximation: A Gradient Boosting Machine*](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full) (2001) | Gradient boosting (chapter 11). |
| Friedman, [*Stochastic Gradient Boosting*](https://www.sciencedirect.com/science/article/abs/pii/S0167947301000652) (2002) | Subsampling in gradient boosting (chapter 11). |
| Chen and Guestrin, [*XGBoost: A Scalable Tree Boosting System*](https://dl.acm.org/doi/abs/10.1145/2939672.2939785) (2016) | Newton steps and regularized tree objectives (chapter 11). |
| Grinsztajn, Oyallon, and Varoquaux, [*Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data?*](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html) (2022) | The practical standing of boosted trees on tabular data (chapter 11). |

## <a id="unsupervised-learning-and-probabilistic-models"></a>Unsupervised learning and probabilistic models

| Work | Idea developed in the module |
| --- | --- |
| Pearson, [*On Lines and Planes of Closest Fit to Systems of Points in Space*](https://www.tandfonline.com/doi/abs/10.1080/14786440109462720) (1901) | PCA as best low-dimensional fit (chapter 12). |
| Hotelling, [*Analysis of a Complex of Statistical Variables into Principal Components*](https://doi.org/10.1037/h0071325) (1933) | PCA as variance maximization (chapter 12). |
| Johnstone, [*On the Distribution of the Largest Eigenvalue in Principal Components Analysis*](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-2/On-the-distribution-of-the-largest-eigenvalue-in-principal-components/10.1214/aos/1009210544.full) (2001) | The spiked covariance model and the largest noise eigenvalue (chapter 12). |
| Baik, Ben Arous, and Péché, [*Phase Transition of the Largest Eigenvalue for Nonnull Complex Sample Covariance Matrices*](https://projecteuclid.org/journals/annals-of-probability/volume-33/issue-5/Phase-transition-of-the-largest-eigenvalue-for-nonnull-complex-sample/10.1214/009117905000000233.full) (2005) | The detection threshold for a spike (chapter 12). |
| Tipping and Bishop, [*Probabilistic Principal Component Analysis*](https://academic.oup.com/jrsssb/article-abstract/61/3/611/7083217) (1999) | PCA as a latent-variable model (chapter 12). |
| Minka, [*Automatic Choice of Dimensionality for PCA*](https://papers.nips.cc/paper/1853-automatic-choice-of-dimensionality-for-pca) (2000) | A Bayesian choice of the number of components (chapter 12). |
| Schölkopf, Smola, and Müller, [*Nonlinear Component Analysis as a Kernel Eigenvalue Problem*](https://direct.mit.edu/neco/article/10/5/1299/6193/Nonlinear-Component-Analysis-as-a-Kernel) (1998) | Kernel PCA (chapter 12). |
| Jolliffe and Cadima, [*Principal Component Analysis: A Review and Recent Developments*](https://royalsocietypublishing.org/rsta/article/374/2065/20150202/115142/Principal-component-analysis-a-review-and-recent) (2016) | Survey (chapter 12). |
| Kleinberg, [*An Impossibility Theorem for Clustering*](https://proceedings.neurips.cc/paper/2002/hash/43e4e6a6f341e00671e123714de019a8-Abstract.html) (2002) | No clustering function satisfies three natural axioms (chapter 13). |
| Lloyd, [*Least Squares Quantization in PCM*](https://ieeexplore.ieee.org/document/1056489/) (1982) | Lloyd's algorithm (chapter 13). |
| Arthur and Vassilvitskii, [*k-means++: The Advantages of Careful Seeding*](https://dl.acm.org/doi/10.5555/1283383.1283494) (2007) | $k$-means++ and its $O(\log k)$ guarantee (chapter 13). |
| Ward, [*Hierarchical Grouping to Optimize an Objective Function*](https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500845) (1963) | Ward linkage (chapter 13). |
| Tibshirani, Walther, and Hastie, [*Estimating the Number of Clusters in a Data Set via the Gap Statistic*](https://academic.oup.com/jrsssb/article/63/2/411/7083348) (2001) | The gap statistic (chapter 13). |
| Rousseeuw, [*Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis*](https://www.sciencedirect.com/science/article/pii/0377042787901257) (1987) | Silhouette widths (chapter 13). |
| Hubert and Arabie, [*Comparing Partitions*](https://link.springer.com/article/10.1007/BF01908075) (1985) | The adjusted Rand index (chapter 13). |
| von Luxburg, [*A Tutorial on Spectral Clustering*](https://link.springer.com/article/10.1007/s11222-007-9033-z) (2007) | Spectral clustering and graph Laplacians (chapter 13). |
| Dempster, Laird, and Rubin, [*Maximum Likelihood from Incomplete Data via the EM Algorithm*](https://academic.oup.com/jrsssb/article/39/1/1/7027539) (1977) | The EM algorithm (chapter 14). |
| Wu, [*On the Convergence Properties of the EM Algorithm*](https://projecteuclid.org/journals/annals-of-statistics/volume-11/issue-1/On-the-Convergence-Properties-of-the-EM-Algorithm/10.1214/aos/1176346060.full) (1983) | What EM's monotonicity does and does not imply about convergence (chapter 14). |
| Neal and Hinton, [*A View of the EM Algorithm That Justifies Incremental, Sparse, and Other Variants*](https://link.springer.com/chapter/10.1007/978-94-011-5014-9_12) (1998) | EM as coordinate ascent on a lower bound (chapter 14). |
| Kanagawa, Hennig, Sejdinovic, and Sriperumbudur, [*Gaussian Processes and Kernel Methods: A Review on Connections and Equivalences*](https://arxiv.org/abs/1807.02582) (2018) | Gaussian-process regression and kernel ridge regression (chapter 15). |
| Jones, Schonlau, and Welch, [*Efficient Global Optimization of Expensive Black-Box Functions*](https://link.springer.com/article/10.1023/A:1008306431147) (1998) | Expected improvement (chapter 15). |
| Srinivas, Krause, Kakade, and Seeger, [*Gaussian Process Optimization in the Bandit Setting*](https://arxiv.org/abs/0912.3995) (2010) | GP-UCB and regret bounds (chapter 15). |
| Snoek, Larochelle, and Adams, [*Practical Bayesian Optimization of Machine Learning Algorithms*](https://proceedings.neurips.cc/paper/2012/file/05311655a15b75fab86956663e1819cd-Paper.pdf) (2012) | Bayesian optimization for hyperparameter tuning (chapter 15). |

## <a id="optional-chapters"></a>Optional chapters

| Work | Idea developed in the module |
| --- | --- |
| Schölkopf, Janzing, Peters, Sgouritsa, Zhang, and Mooij, [*On Causal and Anticausal Learning*](https://arxiv.org/abs/1206.6471) (2012) | When unlabeled inputs carry information about the labels (chapter 16). |
| Nigam, McCallum, Thrun, and Mitchell, [*Text Classification from Labeled and Unlabeled Documents Using EM*](https://link.springer.com/article/10.1023/A:1007692713085) (2000) | Semi-supervised naive Bayes with EM (chapter 16). |
| Blum and Mitchell, [*Combining Labeled and Unlabeled Data with Co-Training*](https://dl.acm.org/doi/10.1145/279943.279962) (1998) | Co-training (chapter 16). |
| Zhu, Ghahramani, and Lafferty, [*Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions*](https://dl.acm.org/doi/10.5555/3041838.3041953) (2003) | Harmonic label propagation (chapter 16). |
| Oliver, Odena, Raffel, Cubuk, and Goodfellow, [*Realistic Evaluation of Deep Semi-Supervised Learning Algorithms*](https://proceedings.neurips.cc/paper/2018/hash/c1fea270c48e8079d8ddf7d06d26ab52-Abstract.html) (2018) | Evaluation pitfalls for semi-supervised methods (chapter 16). |
| Zhu, [*Semi-Supervised Learning*](https://pages.cs.wisc.edu/~jerryzhu/pub/SSL_EoML.pdf) | Survey (chapter 16). |
| Lewis and Gale, [*A Sequential Algorithm for Training Text Classifiers*](https://link.springer.com/chapter/10.1007/978-1-4471-2099-5_1) (1994) | Uncertainty sampling (chapter 16). |
| Dasgupta, [*Two Faces of Active Learning*](https://www.sciencedirect.com/science/article/pii/S0304397510007620) (2011) | Label complexity and exploiting cluster structure (chapter 16). |
| Beygelzimer, Dasgupta, and Langford, [*Importance Weighted Active Learning*](https://arxiv.org/abs/0812.4952) (2009) | Correcting the sampling bias of active queries (chapter 16). |
| Settles, [*Active Learning Literature Survey*](https://burrsettles.com/pub/settles.activelearning.pdf) (2009) | Survey (chapter 16). |
| Cleveland, [*Robust Locally Weighted Regression and Smoothing Scatterplots*](https://www.tandfonline.com/doi/abs/10.1080/01621459.1979.10481038) (1979) | LOESS (chapter 17). |
| Hastie and Tibshirani, [*Generalized Additive Models*](https://projecteuclid.org/journals/statistical-science/volume-1/issue-3/Generalized-Additive-Models/10.1214/ss/1177013604.full) (1986) | Additive models and backfitting (chapter 17). |
| Friedman, Bentley, and Finkel, [*An Algorithm for Finding Best Matches in Logarithmic Expected Time*](https://dl.acm.org/doi/10.1145/355744.355745) (1977) | $k$-d tree search (chapter 17). |
| Malkov and Yashunin, [*Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs*](https://arxiv.org/abs/1603.09320) (2016) | Graph-based approximate search (chapter 17). |
