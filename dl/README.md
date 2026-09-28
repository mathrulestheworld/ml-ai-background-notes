[Background Notes](../README.md)

# Deep Learning

Deep Learning develops the networks that learn their own features: how they are built, why they can be trained, how they generalize despite fitting their training data, and how they are made to work in practice and at scale. It builds on the differentiation, optimization, and PyTorch interfaces of [Foundations](../foundations/README.md) and on the losses, regularization, model selection, and kernels of [Machine Learning](../ml/README.md). Twelve core chapters follow the main topics of the [reading plan](reading-plan.md), and three optional chapters cover its extensions.

## <a id="chapters"></a>Chapters

| Chapter | Main content | Reading plan |
| --- | --- | --- |
| [1. Deep Feedforward Networks](01-deep-feedforward-networks.md) | Learned features; the multilayer perceptron and backpropagation in practice; activation functions; universal approximation and its limits; depth separations and linear regions; output layers and losses; nonconvexity and symmetry. | [Topic 1](reading-plan.md#1-deep-feedforward-networks) |
| [2. Initialization and Signal Propagation](02-initialization-and-signal-propagation.md) | Forward and backward variance recursions; Glorot and He initialization; saturation; correlations between inputs and the ReLU correlation map; Jacobian spectra and dynamical isometry; dead units. | [Topic 2](reading-plan.md#2-initialization-and-signal-propagation) |
| [3. Optimization for Deep Networks](03-optimization-for-deep-networks.md) | Minibatch noise, the gradient noise scale, and the critical batch size; learning-rate schedules and warmup; curvature, the stability threshold, and the edge of stability; adaptive and second-order methods; saddles, plateaus, and mode connectivity. | [Topic 3](reading-plan.md#3-optimization-for-deep-networks) |
| [4. Normalization and Residual Connections](04-normalization-and-residual-connections.md) | Batch, layer, RMS, and group normalization; scale invariance and the effective learning rate; residual connections, the scale of the residual stream, and identity initializations; pre- and post-normalization. | [Topic 4](reading-plan.md#4-normalization-and-residual-connections) |
| [5. Regularization and Generalization in Deep Networks](05-regularization-and-generalization-in-deep-networks.md) | Memorization of random labels; double descent; weight decay, dropout, augmentation, label smoothing, and early stopping; ensembles and weight averaging; implicit bias, flat minima, and benign overfitting. | [Topic 5](reading-plan.md#5-regularization-and-generalization) |
| [6. Convolutional Networks](06-convolutional-networks.md) | Locality and weight sharing; the convolution operation, padding, stride, and dilation; equivariance, pooling, and aliasing; receptive fields; convolution as a matrix product and its cost. | [Topic 6](reading-plan.md#6-convolutional-networks) |
| [7. Convolutional Architectures and Transfer Learning](07-convolutional-architectures-and-transfer-learning.md) | From LeNet to ConvNeXt; bottlenecks, residual and efficient blocks; compound scaling and training recipes; pretrained features, linear probes, and fine-tuning. | [Topic 7](reading-plan.md#7-convolutional-architectures-and-transfer-learning) |
| [8. Recurrent Networks](08-recurrent-networks.md) | Recurrence and weight sharing over time; backpropagation through time and vanishing gradients; LSTM and GRU; encoder–decoder models and teacher forcing; temporal convolutions and state-space models. | [Topic 8](reading-plan.md#8-recurrent-networks) |
| [9. Attention and Transformers](09-attention-and-transformers.md) | Attention as kernel smoothing; scaled dot-product and multi-head self-attention; masks; position encodings and RoPE; the transformer block and its three families; efficient and approximate attention; vision transformers. | [Topic 9](reading-plan.md#9-attention-and-transformers) |
| [10. Self-Supervised Representation Learning](10-self-supervised-representation-learning.md) | Pretext tasks; linear autoencoders and PCA; denoising; contrastive learning, InfoNCE, and mutual information; collapse and methods without negatives; masked modeling; evaluating representations. | [Topic 10](reading-plan.md#10-self-supervised-representation-learning) |
| [11. Training at Scale and Efficient Inference](11-training-at-scale-and-efficient-inference.md) | The roofline model and fusion; floating-point formats and mixed precision; memory accounting and checkpointing; data, sharded, tensor, and pipeline parallelism; quantization, pruning, and distillation. | [Topic 11](reading-plan.md#11-training-at-scale-and-efficient-inference) |
| [12. Practical Methodology](12-practical-methodology.md) | A workflow and sanity checks; reading learning curves and internal statistics; the learning-rate range test; hyperparameter search strategies; seed variance, silent bugs, and reproducibility. | [Topic 12](reading-plan.md#12-practical-methodology) |
| [13. Graph Neural Networks](13-graph-neural-networks.md) *(optional)* | Permutation symmetry; message passing; graph convolutional and attention networks; the Weisfeiler–Lehman bound; oversmoothing and oversquashing. | [Topic 13](reading-plan.md#13-graph-neural-networks-optional) |
| [14. Detection and Segmentation](14-detection-and-segmentation.md) *(optional)* | IoU and average precision; two-stage, one-stage, and anchor-free detectors; the focal loss; DETR and set prediction; fully convolutional networks, U-Net, and transposed convolutions; instance and promptable segmentation. | [Topic 14](reading-plan.md#14-detection-and-segmentation-optional) |
| [15. Infinite Width and the Neural Tangent Kernel](15-infinite-width-and-the-neural-tangent-kernel.md) *(optional)* | Wide networks as Gaussian processes; linearization and the neural tangent kernel; training as kernel regression; spectral bias; lazy and feature-learning regimes; μP. | [Topic 15](reading-plan.md#15-infinite-width-and-the-neural-tangent-kernel-optional) |

Chapters 1–5 treat the deep feedforward network as a trainable object: what it can represent, how its initialization, optimizer, normalization, and residual connections make training possible, and why it generalizes. Chapters 6–9 develop the architectures that build structure into the network, convolution for images, recurrence for sequences, and attention for everything. Chapters 10–12 turn to representations learned without labels, to the engineering of training and serving large models, and to the practice of making a network work. The optional chapters extend the architectures to graphs and to dense prediction and give the infinite-width theory. Proofs and longer derivations appear in collapsed appendices at the end of each chapter.

## <a id="shared-conventions"></a>Shared conventions

- A network $`f(x;\theta)`$ maps an input $`x`$ to an output with parameters $`\theta`$; a layer computes $`h^{(l+1)}=\phi\bigl(W^{(l)}h^{(l)}+b^{(l)}\bigr)`$ with activation $`\phi`$. Widths are $`n`$ or $`d`$, the depth is $`L`$, and a batch has $`B`$ examples. Data follow the conventions of [ML](../ml/README.md#shared-conventions).
- In code, tensors put the batch first: $`(B,d)`$ for vectors, $`(B,C,H,W)`$ for images, and $`(B,T,d)`$ for sequences, as PyTorch's `batch_first=True` layers expect; a sequence written as a matrix $`X\in\mathbb R^{T\times d}`$ has one token per row.
- Losses are averaged over the examples of a batch unless stated otherwise, and cross-entropy is computed from logits. Learning rates refer to the optimizer named alongside them.
- Logarithms are natural. Vectors are columns in the mathematics, and $`\|\cdot\|`$ without a subscript is the Euclidean norm (the Frobenius norm for matrices).
- Each code block runs on its own on a CPU. Seeds are fixed, and the comment lines at the end of a block record what it printed in the environment described in the [computing setup](sources/computing-setup.md). Experiments use the scikit-learn digits, synthetic data, or small bundled graphs and images, so that each finishes in seconds or minutes; their results illustrate the chapters' claims and are not benchmarks.

## <a id="examples-and-supporting-resources"></a>Examples and supporting resources

The chapter text contains the definitions, derivations, and worked examples, and every figure is generated by a script in `Sources/Figure code`. The [computing setup](sources/computing-setup.md) records the Python environment and how to regenerate the figures, and [figure sources](sources/figure-sources.md) records the data and construction behind each figure.

The [reading plan](reading-plan.md) lists the lectures and readings for each topic. [Course links](sources/course-links.md) collects the courses and practical guides it draws on, [video links](sources/video-links.md) the lecture recordings by chapter, [books and documentation](sources/book-and-documentation-links.md) the reference texts and library guides, and [papers](sources/paper-links.md) the principal research behind each chapter. The [UMich EECS 498-007 assignments](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2019/) implement fully connected, convolutional, recurrent, and attention networks from scratch in PyTorch with automatic checks, and the UNIGE course page provides practical sessions with solutions; both fit the chapters directly.

## <a id="connections-to-later-modules"></a>Connections to later modules

**Artificial intelligence** treats search, planning, logic, and probabilistic reasoning, the parts of intelligent behavior that are not learned end to end. Networks from this module appear there as learned heuristics and evaluation functions. The [AI overview](../ai/README.md) lists its chapters.

**[NLP and large language models](../nlp-llms/README.md)** build directly on the transformer of chapter 9, the self-supervised pretraining of chapter 10, and the training at scale of chapter 11: tokenization, language-model pretraining, scaling laws, fine-tuning, and in-context learning.

**[Generative AI](../generative-ai/README.md)** extends the autoencoders and denoising of chapter 10 to variational autoencoders and diffusion models, and adds adversarial networks and normalizing flows; its image models use the U-Net of chapter 14 and the transformers of chapter 9.

**Reinforcement learning** uses networks as value functions and policies. Its instabilities, from nonstationary targets to exploding value estimates, are diagnosed with the tools of chapters 2–4 and 12.

**Safety and frontier** research studies what trained networks compute and whether they can be trusted: mechanistic interpretability starts from the residual stream and attention heads of chapter 9, and the theory of deep learning continues from the generalization puzzles of chapter 5 and the infinite-width limits of chapter 15.

## Reading plan and sources

- [Reading plan](reading-plan.md)
- [Book and documentation links](sources/book-and-documentation-links.md)
- [Computing setup](sources/computing-setup.md)
- [Course links](sources/course-links.md)
- [Figure sources](sources/figure-sources.md)
- [Paper links](sources/paper-links.md)
- [Video links](sources/video-links.md)
