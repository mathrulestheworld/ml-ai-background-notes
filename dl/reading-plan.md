[Background Notes](../README.md) › [Deep Learning](README.md)

# Reading plan

## <a id="courses"></a>Courses

- **Main course — UMich EECS 498-007 / 598-005, Fall 2019: Justin Johnson, *Deep Learning for Computer Vision*.** [Course homepage](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2019/) · [Schedule with slides and readings](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2019/schedule.html). Supplies the main lecture sequence: neural networks, backpropagation in practice, training, convolutional and recurrent networks, and attention. The lectures use images as the running example, but most of the material is about deep learning in general.
- **Supplement — UNIGE 14x050: François Fleuret, *Deep Learning*.** [Course page with slides, handouts and screencasts](https://fleuret.org/dlc/). Short, PyTorch-centered sections, numbered as on the course page; adds depth, dropout, normalization, autoencoders, and attention from a second angle.

The order below follows the chapters of this module. Each block lists the material to cover and its primary lectures; further reading and PyTorch examples provide additional depth. Blocks 1–12 form the main plan; 13–15 are optional extensions. Timing is flexible.

Foundations covered reverse-mode differentiation, backpropagation as an algorithm, and the PyTorch tensor, autograd, and module interfaces; ML covered linear models, losses, regularization, and model selection. Review those chapters as needed in place of the corresponding introductory lectures (UMich lectures 2, 3 and 6; UNIGE modules 1–2). Deep generative models belong to the Generative AI module, language models to NLP and LLMs, and deep reinforcement learning to RL.

## <a id="topic-list"></a>Topic list

- [ ] [1. Deep feedforward networks](#1-deep-feedforward-networks)
- [ ] [2. Initialization and signal propagation](#2-initialization-and-signal-propagation)
- [ ] [3. Optimization for deep networks](#3-optimization-for-deep-networks)
- [ ] [4. Normalization and residual connections](#4-normalization-and-residual-connections)
- [ ] [5. Regularization and generalization](#5-regularization-and-generalization)
- [ ] [6. Convolutional networks](#6-convolutional-networks)
- [ ] [7. Convolutional architectures and transfer learning](#7-convolutional-architectures-and-transfer-learning)
- [ ] [8. Recurrent networks](#8-recurrent-networks)
- [ ] [9. Attention and transformers](#9-attention-and-transformers)
- [ ] [10. Self-supervised representation learning](#10-self-supervised-representation-learning)
- [ ] [11. Training at scale and efficient inference](#11-training-at-scale-and-efficient-inference)
- [ ] [12. Practical methodology](#12-practical-methodology)
- [ ] [13. Graph neural networks — optional](#13-graph-neural-networks-optional)
- [ ] [14. Detection and segmentation — optional](#14-detection-and-segmentation-optional)
- [ ] [15. Infinite width and the neural tangent kernel — optional](#15-infinite-width-and-the-neural-tangent-kernel-optional)

### <a id="reference-books-used-below"></a>Reference books used below

- **UDL:** Prince, [Understanding Deep Learning](https://udlbook.github.io/udlbook/) (MIT Press, 2023). The free PDF and Python notebooks are linked from the book page. The primary textbook for this module.
- **DLB:** Goodfellow, Bengio and Courville, [Deep Learning](https://www.deeplearningbook.org/) (MIT Press, 2016). Free HTML chapters; older, but thorough on feedforward networks, regularization, optimization, and sequence models.
- **LBDL:** Fleuret, [The Little Book of Deep Learning](https://fleuret.org/francois/lbdl.html). A compact summary to accompany the UNIGE course.

## <a id="1-deep-feedforward-networks"></a>1. Deep feedforward networks

**Topics:** From linear models to multilayer perceptrons; activation functions; the network as a composition of affine maps and nonlinearities; output layers and losses as likelihoods; universal approximation; the benefits of depth, including linear regions and depth-separation results; parameter counting.

- **UMich lecture 5: Neural networks.** [Slides · PDF](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture05.pdf) · [Video](https://www.youtube.com/watch?v=eMJm-Eoacdc)
- **UNIGE 3.4 Multi-layer perceptrons, 6.1 Benefits of depth.** [Course page](https://fleuret.org/dlc/)

**Further reading:** UDL chapters 3–5 develop shallow networks, deep networks, and losses as likelihoods, with interactive figures. DLB chapter 6 covers the same ground with more history. Foundations' calculus chapter contains the backpropagation derivation used here.

- **PyTorch example:** [Build the neural network](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html).

## <a id="2-initialization-and-signal-propagation"></a>2. Initialization and signal propagation

**Topics:** How activations and gradients change in scale with depth; vanishing and exploding gradients; variance-preserving initialization (Glorot and He); saturation and dead units; activation statistics as a diagnostic; orthogonal initialization.

- **UMich lecture 10: Training neural networks I.** [Slides · PDF](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture10.pdf) · [Video](https://www.youtube.com/watch?v=ux2-9QbsMPI) Activation functions, data preprocessing, and weight initialization; regularization belongs with block 5.
- **UNIGE 5.5 Parameter initialization, 6.2 Rectifiers.** [Course page](https://fleuret.org/dlc/)
- **Karpathy, *Building makemore part 3: activations and gradients, BatchNorm*.** [Video](https://www.youtube.com/watch?v=P6sfmUTpUmc) A hands-on diagnosis of saturation and initialization.

**Further reading:** UDL chapter 7 derives the He initialization and discusses exploding gradients.

## <a id="3-optimization-for-deep-networks"></a>3. Optimization for deep networks

**Topics:** Minibatch gradient noise; momentum and Nesterov momentum; adaptive methods and AdamW; learning-rate schedules, warmup, and decay; batch size and the learning rate; gradient clipping; the geometry of neural loss landscapes.

- **UMich lecture 4: Optimization.** [Slides · PDF](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture04.pdf) · [Video](https://www.youtube.com/watch?v=mvR07KHB75E)
- **UMich lecture 11: Training neural networks II.** [Slides · PDF](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture11.pdf) · [Video](https://www.youtube.com/watch?v=q9_bxtae0ok) Learning-rate schedules and large-batch training; hyperparameter search belongs with block 12.
- **UNIGE 5.2 Stochastic gradient descent, 5.3 PyTorch optimizers.** [Course page](https://fleuret.org/dlc/)

**Further reading:** UDL chapter 6 covers the optimizers; DLB chapter 8 discusses the obstacles specific to neural networks. Foundations' calculus chapter covers convergence of gradient descent, momentum, and Adam on convex problems.

## <a id="4-normalization-and-residual-connections"></a>4. Normalization and residual connections

**Topics:** Batch normalization in training and evaluation; layer, group, and RMS normalization; residual connections and why they make depth trainable; pre-normalization and post-normalization; the residual network family.

- **UMich lecture 8: CNN architectures.** [Slides · PDF](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture08.pdf) · [Video](https://www.youtube.com/watch?v=Y78DttDmkr8) Residual networks.
- **UNIGE 6.4 Batch normalization, 6.5 Residual networks.** [Course page](https://fleuret.org/dlc/)

**Further reading:** UDL chapter 11 treats residual networks and normalization together.

## <a id="5-regularization-and-generalization"></a>5. Regularization and generalization

**Topics:** Weight decay; dropout; data augmentation and mixup; early stopping; label smoothing; ensembles and weight averaging; memorization of random labels; implicit regularization by stochastic gradient descent; flat and sharp minima; double descent in deep networks.

- **UMich lecture 10: Training neural networks I.** [Slides · PDF](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture10.pdf) · [Video](https://www.youtube.com/watch?v=ux2-9QbsMPI) The regularization part: weight decay, dropout, data augmentation, stochastic depth, and mixup.
- **UMich lecture 11: Training neural networks II.** [Slides · PDF](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture11.pdf) · [Video](https://www.youtube.com/watch?v=q9_bxtae0ok) Model ensembles.
- **UNIGE 5.4 L2 and L1 penalties, 6.3 Dropout.** [Course page](https://fleuret.org/dlc/)

**Further reading:** UDL chapters 8, 9 and 20 cover measuring performance, regularization, and why deep networks generalize. DLB chapter 7 is a catalogue of regularizers. ML's model-selection chapter covers the bias–variance decomposition and double descent for linear models.

## <a id="6-convolutional-networks"></a>6. Convolutional networks

**Topics:** Convolution and cross-correlation; locality, weight sharing, and translation equivariance; padding, stride, and dilation; channels and 1×1 convolutions; pooling and invariance; receptive fields; the cost of a convolutional layer; depthwise-separable convolutions.

- **UMich lecture 7: Convolutional networks.** [Slides · PDF](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture07.pdf) · [Video](https://www.youtube.com/watch?v=gJ5UENsAQGY)
- **UNIGE 4.4 Convolutions, 4.5 Pooling.** [Course page](https://fleuret.org/dlc/)

**Further reading:** UDL chapter 10; DLB chapter 9; the [CS231n notes on convolutional networks](https://cs231n.github.io/convolutional-networks/).

## <a id="7-convolutional-architectures-and-transfer-learning"></a>7. Convolutional architectures and transfer learning

**Topics:** LeNet, AlexNet, VGG, GoogLeNet, residual networks, MobileNet, EfficientNet, and ConvNeXt; design principles and compute–accuracy trade-offs; pretraining and fine-tuning; feature extraction and linear probes.

- **UMich lecture 8: CNN architectures.** [Slides · PDF](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture08.pdf) · [Video](https://www.youtube.com/watch?v=Y78DttDmkr8)
- **UMich lecture 11: Training neural networks II.** [Slides · PDF](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture11.pdf) · [Video](https://www.youtube.com/watch?v=q9_bxtae0ok) Transfer learning.
- **UNIGE 8.1 Computer vision tasks, 8.2 Networks for image classification, 8.5 DataLoader and neuro-surgery.** [Course page](https://fleuret.org/dlc/)

**Further reading:** UDL chapter 10 (later sections) and chapter 11.

- **PyTorch example:** [Transfer learning for computer vision](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

## <a id="8-recurrent-networks"></a>8. Recurrent networks

**Topics:** Sequence modeling with shared weights across time; backpropagation through time; vanishing and exploding gradients over time steps; gated units (LSTM and GRU); gradient clipping; bidirectional and stacked networks; encoder–decoder models and teacher forcing.

- **UMich lecture 12: Recurrent networks.** [Slides · PDF](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture12.pdf) · [Video](https://www.youtube.com/watch?v=tQUetA6A4ts)
- **UNIGE 12.1 Recurrent neural networks, 12.2 LSTM and GRU.** [Course page](https://fleuret.org/dlc/)

**Further reading:** DLB chapter 10 is the most complete textbook treatment. Language modeling with recurrent networks is developed in the NLP and LLMs module.

## <a id="9-attention-and-transformers"></a>9. Attention and transformers

**Topics:** Attention as differentiable lookup; attention in encoder–decoder models; scaled dot-product self-attention; multiple heads; positional information; the transformer block with residual connections and normalization; masking; computational cost; vision transformers.

- **UMich lecture 13: Attention.** [Slides · PDF](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture13.pdf) · [Video](https://www.youtube.com/watch?v=NrXmXcaYBg8)
- **UNIGE 13.1 Attention for memory and sequence translation, 13.2 Attention mechanisms, 13.3 Transformer networks.** [Course page](https://fleuret.org/dlc/)

**Further reading:** UDL chapter 12 covers transformers for text and images. Tokenization, pretraining, and language models are developed in the NLP and LLMs module.

## <a id="10-self-supervised-representation-learning"></a>10. Self-supervised representation learning

**Topics:** Representations and how to evaluate them; autoencoders and denoising; pretext tasks; contrastive learning and the InfoNCE objective; collapse and non-contrastive methods; masked image modeling; linear probes and fine-tuning.

- **UNIGE 7.2 Deep autoencoders, 7.3 Denoising autoencoders.** [Course page](https://fleuret.org/dlc/)
- **Balestriero et al., *A Cookbook of Self-Supervised Learning*.** [Paper · arXiv](https://arxiv.org/abs/2304.12210) A survey of the method families and their training details.

**Further reading:** UDL chapter 14 frames unsupervised learning; [Lilian Weng, *Contrastive representation learning*](https://lilianweng.github.io/posts/2021-05-31-contrastive/) collects the contrastive objectives. Variational autoencoders are developed in the Generative AI module.

## <a id="11-training-at-scale-and-efficient-inference"></a>11. Training at scale and efficient inference

**Topics:** Arithmetic intensity and the memory hierarchy; mixed-precision training; memory for parameters, gradients, optimizer state, and activations; activation checkpointing; data, tensor, and pipeline parallelism; sharded optimizer state; quantization, pruning, and distillation for inference.

- **UMich lecture 9: Hardware and software.** [Slides · PDF](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture09.pdf) · [Video](https://www.youtube.com/watch?v=Sb8c7-n-QHA)
- **UNIGE 6.6 Using GPUs.** [Course page](https://fleuret.org/dlc/)
- **Hugging Face, *The Ultra-Scale Playbook*.** [Interactive book](https://huggingface.co/spaces/nanotron/ultrascale-playbook) Memory accounting and every form of parallelism, with measurements.
- **Google DeepMind, *How to Scale Your Model*.** [Online book](https://jax-ml.github.io/scaling-book/) Rooflines, sharding, and the costs of transformer training and inference on accelerators.

**Further reading:** The [PyTorch automatic mixed precision guide](https://pytorch.org/docs/stable/amp.html) documents the numerical formats and loss scaling.

## <a id="12-practical-methodology"></a>12. Practical methodology

**Topics:** Baselines and sanity checks; overfitting a single batch; monitoring losses, activations, and gradients; learning-rate range tests; hyperparameter search and its budget; interpreting learning curves; reproducibility and seeds; experiment tracking.

- **UMich lecture 11: Training neural networks II.** [Slides · PDF](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture11.pdf) · [Video](https://www.youtube.com/watch?v=q9_bxtae0ok) Hyperparameter search and reading learning curves.
- **UNIGE 5.6 Architecture choice and training protocol.** [Course page](https://fleuret.org/dlc/)
- **Karpathy, *A Recipe for Training Neural Networks*.** [Blog post](https://karpathy.github.io/2019/04/25/recipe/)
- **Godbole, Dahl, Gilmer, Shallue, and Nado, *Deep Learning Tuning Playbook*.** [Repository](https://github.com/google-research/tuning_playbook)

**Further reading:** DLB chapter 11 (practical methodology).

## <a id="13-graph-neural-networks-optional"></a>13. Graph neural networks — optional

**Topics:** Graphs, adjacency, and permutation symmetry; message passing; graph convolutional and attention networks; node, edge, and graph prediction; expressive power and the Weisfeiler–Lehman test; oversmoothing.

- **Stanford CS224W: Jure Leskovec, *Machine Learning with Graphs* (2021).** [Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOP-ImU-O1rYRg2RFxomvFp) Lectures 6–9 cover graph neural networks.

**Further reading:** UDL chapter 13.

## <a id="14-detection-and-segmentation-optional"></a>14. Detection and segmentation — optional

**Topics:** Dense prediction; bounding-box regression and intersection over union; two-stage and one-stage detectors; anchors and non-maximum suppression; fully convolutional networks, U-Net, and upsampling; instance segmentation.

- **UMich lecture 15: Object detection.** [Slides · PDF](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture15.pdf) · [Video](https://www.youtube.com/watch?v=MshSNnwF1Qg)
- **UMich lecture 16: Detection and segmentation.** [Slides · PDF](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture16.pdf) · [Video](https://www.youtube.com/watch?v=zHSjrqS0jAY)
- **UNIGE 7.1 Transposed convolutions, 8.3 Networks for object detection, 8.4 Networks for semantic segmentation.** [Course page](https://fleuret.org/dlc/)

## <a id="15-infinite-width-and-the-neural-tangent-kernel-optional"></a>15. Infinite width and the neural tangent kernel — optional

**Topics:** Wide networks at initialization as Gaussian processes; linearization around initialization; the neural tangent kernel and lazy training; what the kernel limit explains and what it misses; feature learning and parameterization.

**Further reading:** [Lilian Weng, *Some math behind neural tangent kernel*](https://lilianweng.github.io/posts/2022-09-08-ntk/) derives the main results. The Gaussian-process chapter of ML supplies the kernel background.

## <a id="exercises-and-local-references"></a>Exercises and local references

- [UMich EECS 498-007 assignments](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2019/) implement networks, convolution, batch normalization, recurrent networks, and attention from scratch in PyTorch, with automatic checks.
- The UNIGE course page provides six practical sessions with solutions.
- Foundations and ML supply the mathematical and statistical prerequisites; their chapters are references during this module.

## <a id="connections-to-later-modules"></a>Connections to later modules

Tokenization, pretraining, and language models belong in NLP and LLMs; variational autoencoders, adversarial networks, normalizing flows, and diffusion models in Generative AI; deep reinforcement learning in RL; interpretability, robustness, and alignment in Safety and Frontier. Architectures, optimization, normalization, and training practice common to all of them remain in this module.
