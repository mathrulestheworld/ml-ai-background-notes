[Background Notes](../../README.md) › [Foundations](../README.md)

# Book and documentation links

# <a id="books-and-documentation"></a>Books and documentation

The chapter notes develop the module's main material. These books and official documentation provide alternative explanations, fuller proofs, and details of numerical interfaces.

## <a id="mathematical-references"></a>Mathematical references

| Source | Relevant material |
| --- | --- |
| Deisenroth, Faisal, and Ong, [*Mathematics for Machine Learning*](https://mml-book.github.io/) | Part I: linear algebra, geometry, decompositions, vector calculus, probability, and optimization. The local notes specify their numerator-layout derivative convention explicitly. |
| Mitchell, [*Machine Learning*](https://www.cs.cmu.edu/~tom/mlbook.html) | Chapter 1: tasks, experience, and performance measures. |
| Härdle, [*Smoothing Techniques: With Implementation in S*](https://link.springer.com/book/10.1007/978-1-4612-4432-5) | Kernel density estimation and smoothing; the source cited for the Old Faithful waiting times in chapter 1. |
| Blitzstein and Hwang, [*Introduction to Probability*](https://probabilitybook.net/) | Probability laws, conditioning, random variables, distribution families, and expectation. |
| Durrett, [*Probability: Theory and Examples*](https://sites.math.duke.edu/~rtd/PTE/pte.html) | Chapter 1: measure-theoretic foundations, including σ-algebras, the construction of Lebesgue measure, random variables as measurable maps, and the integral behind expectation. |
| Wasserman, [*All of Statistics*](https://www.stat.cmu.edu/~larry/all-of-statistics/) | Statistical inference, empirical distributions, estimation, confidence intervals, and testing. |
| Murphy, [*Probabilistic Machine Learning: An Introduction*](https://probml.github.io/pml-book/book1.html) | Probability, statistics, decision theory, and information-theoretic foundations of probabilistic models. |
| Cover and Thomas, [*Elements of Information Theory*](https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X) | Entropy, typical sequences, source coding, entropy rates, channels, differential entropy, and rate–distortion. |
| MacKay, [*Information Theory, Inference, and Learning Algorithms*](https://www.inference.org.uk/mackay/itila/book.html) | Connections among coding, probabilistic inference, and learning. |
| Shalev-Shwartz and Ben-David, [*Understanding Machine Learning*](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) | Chapters 2–8 for statistical learning; further chapters on stability, online learning, and Rademacher complexity. |
| Mohri, Rostamizadeh, and Talwalkar, [*Foundations of Machine Learning*](https://cs.nyu.edu/~mohri/mlbook/) | Growth functions, Rademacher complexity, margins, and generalization. |

## <a id="numerical-libraries"></a>Numerical libraries

| Documentation | Relevant material |
| --- | --- |
| [NumPy array object](https://numpy.org/doc/stable/reference/arrays.ndarray.html) | Shape, axes, arithmetic types, and the array representation. |
| [NumPy copies and views](https://numpy.org/doc/stable/user/basics.copies.html) | Shared storage, indexing, reshaping, and mutation. |
| [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) | Alignment of axes and singleton dimensions. |
| [NumPy matrix multiplication](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html) | Matrix products, vector special cases, and batch dimensions. |
| [NumPy least squares](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html) | Least-squares solutions, numerical rank, and singular values. |
| [NumPy linear algebra](https://numpy.org/doc/stable/reference/routines.linalg.html) | Factorizations, solves, eigenvalues, singular values, norms, and condition numbers used in chapter 2. |
| [SciPy log-sum-exp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html) | Stable logarithmic probability calculations. |
| [PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) | Recorded dependencies, leaf gradients, accumulation, and gradient modes. |
| [PyTorch `torch.autograd`](https://docs.pytorch.org/docs/stable/autograd.html) | `grad`, `backward`, and the tensor attributes used in chapter 3's backpropagation example. |
| [PyTorch JVP](https://docs.pytorch.org/docs/stable/generated/torch.func.jvp.html) and [VJP](https://docs.pytorch.org/docs/stable/generated/torch.func.vjp.html) | Derivative products with explicit input directions and output seeds. |
| [PyTorch Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) | Output-by-input parameter storage and the batched affine map. |
| [PyTorch cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) | Logits, target representations, weighting, and reductions. |
| [PyTorch optimizers](https://docs.pytorch.org/docs/stable/optim.html) | Gradient clearing, parameter updates, and optimizer state. |
| [PyTorch data utilities](https://docs.pytorch.org/docs/stable/data.html) | Dataset indexing, batching, and sampling order. |

The computing examples notebook accompanies chapter 6. Additional API links appear beside the corresponding definitions in that chapter. Chapters 2 and 3 link each NumPy and PyTorch function where the text first discusses it and list every function their examples use in a final collapsed appendix.
