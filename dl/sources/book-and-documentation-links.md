[Background Notes](../../README.md) › [Deep Learning](../README.md)

# Book and documentation links

> [!WARNING]
> Work in progress: this part of the notes is still being revised.

# <a id="books-and-documentation"></a>Books and documentation

The chapter notes develop the module's main material. These books and official documentation provide alternative explanations, fuller derivations, and the details of the library interfaces used in the code. The [reading plan](../reading-plan.md) lists the lecture slides and readings for each topic.

## <a id="main-references"></a>Main references

| Source | Relevant material |
| --- | --- |
| Prince, [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/) (MIT Press, 2023) | The primary textbook: chapters 3–4 (chapter 1), 6–7 (chapters 2–3), 8–9 (chapter 5), 10 (chapter 6), 11 (chapters 4 and 7), 12 (chapter 9), 13 (chapter 13), 14 (chapter 10), and 20 (chapters 5 and 15). The free PDF and Python notebooks are linked from the book page. |
| Goodfellow, Bengio, and Courville, [*Deep Learning*](https://www.deeplearningbook.org/) (MIT Press, 2016) | Chapters 6 (chapter 1), 7 (chapter 5), 8 (chapters 2–4), 9 (chapter 6), 10 (chapter 8), 11 (chapter 12), and 14–15 (chapter 10). Older, but thorough on the classical material; free HTML chapters. |
| Fleuret, [*The Little Book of Deep Learning*](https://fleuret.org/francois/lbdl.html) | A compact summary of components, architectures, training, and efficient computation that accompanies the UNIGE course. |
| Zhang, Lipton, Li, and Smola, [*Dive into Deep Learning*](https://d2l.ai/) | An interactive book with runnable PyTorch code for every model: optimization, convolutional and recurrent networks, attention and transformers, computational performance, and computer vision. |

## <a id="specialized-books-and-surveys"></a>Specialized books and surveys

| Source | Relevant material |
| --- | --- |
| Bronstein, Bruna, Cohen, and Veličković, [*Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges*](https://arxiv.org/abs/2104.13478) | Architectures derived from the symmetries of their domain: convolutional networks, transformers, and graph networks in one framework (chapters 6, 9, and 13). |
| Hamilton, [*Graph Representation Learning*](https://www.cs.mcgill.ca/~wlh/grl_book/) | Graph neural networks, spectral methods, and the Weisfeiler–Lehman connection (chapter 13). |
| Roberts, Yaida, and Hanin, [*The Principles of Deep Learning Theory*](https://arxiv.org/abs/2106.10165) | Signal propagation at initialization, the infinite-width limit, and its $`1/n`$ corrections (chapters 2 and 15). |
| Balestriero et al., [*A Cookbook of Self-Supervised Learning*](https://arxiv.org/abs/2304.12210) | The families of self-supervised methods and the training details that make them work (chapter 10). |
| Phuong and Hutter, [*Formal Algorithms for Transformers*](https://arxiv.org/abs/2207.09238) | Precise pseudocode for every transformer variant (chapter 9). |
| Nagel et al., [*A White Paper on Neural Network Quantization*](https://arxiv.org/abs/2106.08295) | Quantization schemes, post-training quantization, and quantization-aware training (chapter 11). |

## <a id="notes-and-guides"></a>Notes and guides

| Source | Relevant material |
| --- | --- |
| Horace He, [*Making Deep Learning Go Brrrr From First Principles*](https://horace.io/brrr_intro.html) | Compute, memory bandwidth, and overhead as the three costs of a GPU program (chapter 11). |
| Olah, [*Understanding LSTM Networks*](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | Illustrated LSTM and GRU gates (chapter 8). |
| Rush et al., [*The Annotated Transformer*](https://nlp.seas.harvard.edu/annotated-transformer/) and Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) | Compact, complete transformer implementations (chapter 9). |
| Weng, [*Contrastive Representation Learning*](https://lilianweng.github.io/posts/2021-05-31-contrastive/) and [*Some Math behind Neural Tangent Kernel*](https://lilianweng.github.io/posts/2022-09-08-ntk/) | Surveys of contrastive objectives (chapter 10) and derivations of the NTK results (chapter 15). |
| Dumoulin and Visin, [*A Guide to Convolution Arithmetic for Deep Learning*](https://arxiv.org/abs/1603.07285) | Output sizes and illustrations for padding, stride, dilation, and transposed convolutions (chapters 6 and 14). |
| Odena, Dumoulin, and Olah, [*Deconvolution and Checkerboard Artifacts*](https://distill.pub/2016/deconv-checkerboard/) | Why transposed convolutions produce checkerboard patterns (chapter 14). |

## <a id="library-documentation"></a>Library documentation

| Source | Relevant material |
| --- | --- |
| [PyTorch documentation](https://pytorch.org/docs/stable/index.html) | `torch.nn` layers and initialization, autograd, `torch.optim` and learning-rate schedulers, `torch.nn.functional.scaled_dot_product_attention`, and `torch.utils.checkpoint`. |
| [PyTorch automatic mixed precision](https://pytorch.org/docs/stable/amp.html) and [`torch.compile`](https://pytorch.org/docs/stable/torch.compiler.html) | `torch.autocast`, gradient scaling, and compilation with kernel fusion (chapter 11). |
| [PyTorch distributed overview](https://pytorch.org/tutorials/beginner/dist_overview.html) | Distributed data parallelism, fully sharded data parallelism, and tensor parallelism (chapter 11). |
| [PyTorch reproducibility notes](https://pytorch.org/docs/stable/notes/randomness.html) | Seeds, deterministic algorithms, and sources of nondeterminism (chapter 12). |
| [NetworkX documentation](https://networkx.org/documentation/stable/) | Graph construction, random graph models, and the karate-club graph used in chapter 13. |
| [scikit-learn datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html) | The bundled digits dataset used throughout, and `load_sample_image` (chapter 9). |
