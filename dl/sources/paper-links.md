[ML Mastery Notes](../../README.md) › [Deep Learning](../README.md)

# Paper links

# <a id="papers"></a>Papers

The notes link several hundred papers where their ideas arise. This index collects the original papers and principal analyses behind each chapter's main ideas, grouped by chapter; the chapters cite further work inline.

## <a id="networks-initialization-and-optimization"></a>Networks, initialization, and optimization

| Work | Idea developed in the module |
| --- | --- |
| Rumelhart, Hinton, and Williams, [*Learning Representations by Back-propagating Errors*](https://doi.org/10.1038/323533a0) (1986) | Backpropagation trains multilayer networks and makes them learn internal representations (chapter 1). |
| Cybenko, [*Approximation by Superpositions of a Sigmoidal Function*](https://doi.org/10.1007/BF02551274) (1989), and Hornik, Stinchcombe, and White, [*Multilayer Feedforward Networks Are Universal Approximators*](https://www.sciencedirect.com/science/article/abs/pii/0893608089900208) (1989) | Universal approximation with one hidden layer (chapter 1). |
| Barron, [*Universal Approximation Bounds for Superpositions of a Sigmoidal Function*](https://doi.org/10.1109/18.256500) (1993) | Approximation rates that do not degrade with the input dimension, for functions with a bounded Fourier moment (chapter 1). |
| Telgarsky, [*Benefits of Depth in Neural Networks*](https://arxiv.org/abs/1602.04485) (2016) | Functions that deep networks compute with few units and shallow networks need exponentially many units for (chapter 1). |
| Montúfar, Pascanu, Cho, and Bengio, [*On the Number of Linear Regions of Deep Neural Networks*](https://arxiv.org/abs/1402.1869) (2014), and Hanin and Rolnick, [*Complexity of Linear Regions in Deep Networks*](https://arxiv.org/abs/1901.09021) (2019) | The maximal and the typical number of linear pieces of ReLU networks (chapter 1). |
| Glorot and Bengio, [*Understanding the Difficulty of Training Deep Feedforward Neural Networks*](https://proceedings.mlr.press/v9/glorot10a.html) (2010) | Variance-preserving initialization and saturation of deep networks (chapter 2). |
| He, Zhang, Ren, and Sun, [*Delving Deep into Rectifiers*](https://arxiv.org/abs/1502.01852) (2015) | Initialization for ReLU networks (chapter 2). |
| Saxe, McClelland, and Ganguli, [*Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks*](https://arxiv.org/abs/1312.6120) (2014) | Learning dynamics of deep linear networks and orthogonal initialization (chapter 2). |
| Poole et al., [*Exponential Expressivity in Deep Neural Networks through Transient Chaos*](https://arxiv.org/abs/1606.05340) (2016), and Schoenholz et al., [*Deep Information Propagation*](https://arxiv.org/abs/1611.01232) (2017) | Mean-field signal propagation: order, chaos, and the depth scales of trainability (chapters 2 and 15). |
| Pennington, Schoenholz, and Ganguli, [*Resurrecting the Sigmoid in Deep Learning through Dynamical Isometry*](https://arxiv.org/abs/1711.04735) (2017) | The whole singular-value spectrum of the input–output Jacobian at initialization (chapter 2). |
| Kingma and Ba, [*Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980) (2015), and Loshchilov and Hutter, [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101) (2019) | Adam and AdamW, the default optimizers of the module (introduced in Foundations; chapters 3, 9, and 11). |
| McCandlish, Kaplan, and Amodei, [*An Empirical Model of Large-Batch Training*](https://arxiv.org/abs/1812.06162) (2018) | The gradient noise scale and the critical batch size (chapters 3 and 11). |
| Goyal et al., [*Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*](https://arxiv.org/abs/1706.02677) (2017) | Linear learning-rate scaling with the batch size, and warmup (chapters 3 and 11). |
| Shallue et al., [*Measuring the Effects of Data Parallelism on Neural Network Training*](https://jmlr.org/papers/v20/18-789.html) (2019) | Steps to a target as a function of the batch size, across workloads (chapter 3). |
| Loshchilov and Hutter, [*SGDR: Stochastic Gradient Descent with Warm Restarts*](https://arxiv.org/abs/1608.03983) (2017) | Cosine learning-rate schedules (chapter 3). |
| Cohen et al., [*Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability*](https://arxiv.org/abs/2103.00065) (2021) | Progressive sharpening and training at the stability threshold $2/\eta$ (chapter 3). |
| Dauphin et al., [*Identifying and Attacking the Saddle Point Problem in High-Dimensional Non-Convex Optimization*](https://arxiv.org/abs/1406.2572) (2014) | Saddle points, rather than poor local minima, as the obstacle in high dimension (chapter 3). |
| Garipov et al., [*Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs*](https://arxiv.org/abs/1802.10026) (2018), and Entezari et al., [*The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks*](https://arxiv.org/abs/2110.06296) (2022) | Low-loss paths between minima and the role of permutation symmetry (chapter 3). |

## <a id="normalization-residual-connections-and-regularization"></a>Normalization, residual connections, and regularization

| Work | Idea developed in the module |
| --- | --- |
| Ioffe and Szegedy, [*Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*](https://arxiv.org/abs/1502.03167) (2015) | Batch normalization (chapter 4). |
| Santurkar, Tsipras, Ilyas, and Madry, [*How Does Batch Normalization Help Optimization?*](https://arxiv.org/abs/1805.11604) (2018) | Smoothing of the loss landscape rather than reduced covariate shift (chapter 4). |
| Ba, Kiros, and Hinton, [*Layer Normalization*](https://arxiv.org/abs/1607.06450) (2016), and Zhang and Sennrich, [*Root Mean Square Layer Normalization*](https://arxiv.org/abs/1910.07467) (2019) | Normalization over features, independent of the batch (chapters 4 and 9). |
| He, Zhang, Ren, and Sun, [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385) (2016) | Residual connections and networks with over 100 layers (chapters 4 and 7). |
| De and Smith, [*Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks*](https://arxiv.org/abs/2002.10444) (2020) | Why normalization helps residual networks, and SkipInit (chapter 4). |
| Xiong et al., [*On Layer Normalization in the Transformer Architecture*](https://arxiv.org/abs/2002.04745) (2020) | Pre-normalization and the need for warmup in post-normalization transformers (chapters 3, 4, and 9). |
| Zhang, Bengio, Hardt, Recht, and Vinyals, [*Understanding Deep Learning Requires Rethinking Generalization*](https://arxiv.org/abs/1611.03530) (2017) | Networks fit random labels, so capacity alone does not explain generalization (chapter 5). |
| Nakkiran et al., [*Deep Double Descent: Where Bigger Models and More Data Hurt*](https://arxiv.org/abs/1912.02292) (2020) | Double descent in model size, training time, and data (chapter 5). |
| Srivastava et al., [*Dropout: A Simple Way to Prevent Neural Networks from Overfitting*](https://jmlr.org/papers/v15/srivastava14a.html) (2014) | Dropout (chapters 5 and 8). |
| Szegedy et al., [*Rethinking the Inception Architecture for Computer Vision*](https://arxiv.org/abs/1512.00567) (2016) | Label smoothing (chapters 5 and 9). |
| Lakshminarayanan, Pritzel, and Blundell, [*Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles*](https://arxiv.org/abs/1612.01474) (2017), and Izmailov et al., [*Averaging Weights Leads to Wider Optima and Better Generalization*](https://arxiv.org/abs/1803.05407) (2018) | Ensembles and stochastic weight averaging (chapter 5). |
| Soudry et al., [*The Implicit Bias of Gradient Descent on Separable Data*](https://jmlr.org/papers/v19/18-188.html) (2018) | Convergence to the maximum-margin direction (chapter 5). |
| Foret, Kleiner, Mobahi, and Neyshabur, [*Sharpness-Aware Minimization for Efficiently Improving Generalization*](https://arxiv.org/abs/2010.01412) (2021) | Minimizing the worst loss in a neighborhood (chapter 5). |
| Power et al., [*Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*](https://arxiv.org/abs/2201.02177) (2022) | Generalization long after the training data are fitted (chapter 5). |

## <a id="convolutional-and-recurrent-networks"></a>Convolutional and recurrent networks

| Work | Idea developed in the module |
| --- | --- |
| Fukushima, [*Neocognitron*](https://doi.org/10.1007/BF00344251) (1980), and LeCun et al., [*Backpropagation Applied to Handwritten Zip Code Recognition*](https://doi.org/10.1162/neco.1989.1.4.541) (1989) | Local receptive fields, weight sharing, and pooling; convolutional networks trained by backpropagation (chapter 6). |
| LeCun, Bottou, Bengio, and Haffner, [*Gradient-Based Learning Applied to Document Recognition*](https://doi.org/10.1109/5.726791) (1998) | LeNet-5 (chapter 7). |
| Krizhevsky, Sutskever, and Hinton, [*ImageNet Classification with Deep Convolutional Neural Networks*](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) (2012) | AlexNet: GPUs, ReLU, and dropout win ImageNet (chapters 1 and 7). |
| Zhang, [*Making Convolutional Networks Shift-Invariant Again*](https://arxiv.org/abs/1904.11486) (2019) | Aliasing from strided layers and anti-aliased pooling (chapter 6). |
| Simonyan and Zisserman, [*Very Deep Convolutional Networks for Large-Scale Image Recognition*](https://arxiv.org/abs/1409.1556) (2015) | Depth with $3\times3$ convolutions (chapter 7). |
| Howard et al., [*MobileNets*](https://arxiv.org/abs/1704.04861) (2017), and Sandler et al., [*MobileNetV2: Inverted Residuals and Linear Bottlenecks*](https://arxiv.org/abs/1801.04381) (2018) | Depthwise separable convolutions and inverted residual blocks (chapters 6 and 7). |
| Tan and Le, [*EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*](https://arxiv.org/abs/1905.11946) (2019) | Compound scaling of depth, width, and resolution (chapter 7). |
| Liu et al., [*A ConvNet for the 2020s*](https://arxiv.org/abs/2201.03545) (2022) | ConvNeXt: a convolutional network modernized step by step to match vision transformers (chapters 7 and 9). |
| Yosinski, Clune, Bengio, and Lipson, [*How Transferable Are Features in Deep Neural Networks?*](https://arxiv.org/abs/1411.1792) (2014), and Kornblith, Shlens, and Le, [*Do Better ImageNet Models Transfer Better?*](https://arxiv.org/abs/1805.08974) (2019) | What transfers from pretrained networks, layer by layer and across architectures (chapter 7). |
| Hochreiter and Schmidhuber, [*Long Short-Term Memory*](https://doi.org/10.1162/neco.1997.9.8.1735) (1997) | The LSTM and its constant error carousel (chapter 8). |
| Pascanu, Mikolov, and Bengio, [*On the Difficulty of Training Recurrent Neural Networks*](https://arxiv.org/abs/1211.5063) (2013) | Vanishing and exploding gradients, and gradient clipping (chapter 8). |
| Cho et al., [*Learning Phrase Representations Using RNN Encoder–Decoder for Statistical Machine Translation*](https://arxiv.org/abs/1406.1078) (2014), and Sutskever, Vinyals, and Le, [*Sequence to Sequence Learning with Neural Networks*](https://arxiv.org/abs/1409.3215) (2014) | The GRU and encoder–decoder models (chapter 8). |
| Gu, Goel, and Ré, [*Efficiently Modeling Long Sequences with Structured State Spaces*](https://arxiv.org/abs/2111.00396) (2022), and Gu and Dao, [*Mamba: Linear-Time Sequence Modeling with Selective State Spaces*](https://arxiv.org/abs/2312.00752) (2023) | State-space models: linear recurrences computed as convolutions or scans (chapter 8). |

## <a id="attention-self-supervision-and-scale"></a>Attention, self-supervision, and scale

| Work | Idea developed in the module |
| --- | --- |
| Bahdanau, Cho, and Bengio, [*Neural Machine Translation by Jointly Learning to Align and Translate*](https://arxiv.org/abs/1409.0473) (2015) | Attention in encoder–decoder models (chapter 9). |
| Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017) | The transformer: scaled dot-product and multi-head attention, sinusoidal positions (chapter 9). |
| Su et al., [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864) (2021) | Rotary position embeddings (chapter 9). |
| Elhage et al., [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html) (2021) | The residual stream and heads as independent read–write operations (chapter 9; developed in Safety and Frontier). |
| Dao et al., [*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*](https://arxiv.org/abs/2205.14135) (2022) | Exact attention computed block by block in on-chip memory (chapters 9 and 11). |
| Dosovitskiy et al., [*An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929) (2021) | The vision transformer (chapters 9 and 14). |
| van den Oord, Li, and Vinyals, [*Representation Learning with Contrastive Predictive Coding*](https://arxiv.org/abs/1807.03748) (2018) | The InfoNCE loss (chapter 10). |
| Chen, Kornblith, Norouzi, and Hinton, [*A Simple Framework for Contrastive Learning of Visual Representations*](https://arxiv.org/abs/2002.05709) (2020) | SimCLR (chapter 10). |
| Grill et al., [*Bootstrap Your Own Latent*](https://arxiv.org/abs/2006.07733) (2020), and Chen and He, [*Exploring Simple Siamese Representation Learning*](https://arxiv.org/abs/2011.10566) (2021) | Self-supervised learning without negatives (chapter 10). |
| Bardes, Ponce, and LeCun, [*VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*](https://arxiv.org/abs/2105.04906) (2022) | Preventing collapse with explicit penalties on the embedding statistics (chapter 10). |
| Wang and Isola, [*Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*](https://arxiv.org/abs/2005.10242) (2020) | The two terms of the contrastive loss (chapter 10). |
| He et al., [*Masked Autoencoders Are Scalable Vision Learners*](https://arxiv.org/abs/2111.06377) (2022) | Masked image modeling (chapter 10). |
| Radford et al., [*Learning Transferable Visual Models from Natural Language Supervision*](https://arxiv.org/abs/2103.00020) (2021) | CLIP: contrastive image–text pretraining (chapters 10 and 14). |
| Micikevicius et al., [*Mixed Precision Training*](https://arxiv.org/abs/1710.03740) (2018) | Master weights and loss scaling (chapter 11). |
| Rajbhandari, Rasley, Ruwase, and He, [*ZeRO: Memory Optimizations toward Training Trillion Parameter Models*](https://arxiv.org/abs/1910.02054) (2020) | Sharding optimizer state, gradients, and parameters (chapter 11). |
| Shoeybi et al., [*Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*](https://arxiv.org/abs/1909.08053) (2019) | Tensor parallelism for transformer layers (chapter 11). |
| Huang et al., [*GPipe: Efficient Training of Giant Neural Networks Using Pipeline Parallelism*](https://arxiv.org/abs/1811.06965) (2019) | Pipeline parallelism with micro-batches (chapter 11). |
| Chen, Xu, Zhang, and Guestrin, [*Training Deep Nets with Sublinear Memory Cost*](https://arxiv.org/abs/1604.06174) (2016) | Activation checkpointing (chapter 11). |
| Kaplan et al., [*Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361) (2020) | The $6N$ operations per token and the compute of transformers (chapters 9 and 11; developed in NLP and LLMs). |
| Jacob et al., [*Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*](https://arxiv.org/abs/1712.05877) (2018) | Integer quantization of weights and activations (chapter 11). |
| Frankle and Carbin, [*The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks*](https://arxiv.org/abs/1803.03635) (2019) | Sparse subnetworks that train from their original initialization (chapter 11). |
| Hinton, Vinyals, and Dean, [*Distilling the Knowledge in a Neural Network*](https://arxiv.org/abs/1503.02531) (2015) | Knowledge distillation with softened targets (chapter 11). |

## <a id="methodology-and-optional-topics"></a>Methodology and optional topics

| Work | Idea developed in the module |
| --- | --- |
| Bergstra and Bengio, [*Random Search for Hyper-Parameter Optimization*](https://jmlr.org/papers/v13/bergstra12a.html) (2012) | Why random search beats grid search when few hyperparameters matter (chapter 12). |
| Smith, [*Cyclical Learning Rates for Training Neural Networks*](https://arxiv.org/abs/1506.01186) (2017) | The learning-rate range test (chapter 12). |
| Li et al., [*Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization*](https://arxiv.org/abs/1603.06560) (2018) | Early stopping of poor trials (chapter 12). |
| Dodge et al., [*Show Your Work: Improved Reporting of Experimental Results*](https://arxiv.org/abs/1909.03004) (2019), and Bouthillier et al., [*Accounting for Variance in Machine Learning Benchmarks*](https://arxiv.org/abs/2103.03098) (2021) | Reporting results as a function of the tuning budget, and the variance across seeds (chapter 12). |
| Yang et al., [*Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer*](https://arxiv.org/abs/2203.03466) (2022) | μP and hyperparameter transfer across widths (chapters 9, 12, and 15). |
| Gilmer et al., [*Neural Message Passing for Quantum Chemistry*](https://arxiv.org/abs/1704.01212) (2017) | The message-passing framework (chapter 13). |
| Kipf and Welling, [*Semi-Supervised Classification with Graph Convolutional Networks*](https://arxiv.org/abs/1609.02907) (2017) | The graph convolutional network (chapter 13). |
| Xu, Hu, Leskovec, and Jegelka, [*How Powerful Are Graph Neural Networks?*](https://arxiv.org/abs/1810.00826) (2019) | Message passing is at most as powerful as the Weisfeiler–Lehman test; GIN (chapter 13). |
| Ren, He, Girshick, and Sun, [*Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*](https://arxiv.org/abs/1506.01497) (2015) | Region proposal networks and anchors (chapter 14). |
| Lin, Goyal, Girshick, He, and Dollár, [*Focal Loss for Dense Object Detection*](https://arxiv.org/abs/1708.02002) (2017) | The focal loss and RetinaNet (chapter 14). |
| Carion et al., [*End-to-End Object Detection with Transformers*](https://arxiv.org/abs/2005.12872) (2020) | DETR: detection as set prediction with Hungarian matching (chapter 14). |
| Ronneberger, Fischer, and Brox, [*U-Net: Convolutional Networks for Biomedical Image Segmentation*](https://arxiv.org/abs/1505.04597) (2015) | Encoder–decoder segmentation with skip connections (chapter 14). |
| He, Gkioxari, Dollár, and Girshick, [*Mask R-CNN*](https://arxiv.org/abs/1703.06870) (2017) | Instance segmentation and RoIAlign (chapter 14). |
| Kirillov et al., [*Segment Anything*](https://arxiv.org/abs/2304.02643) (2023) | Promptable segmentation trained on 1.1 billion masks (chapter 14). |
| Lee et al., [*Deep Neural Networks as Gaussian Processes*](https://arxiv.org/abs/1711.00165) (2018) | The NNGP kernel of deep networks (chapter 15). |
| Jacot, Gabriel, and Hongler, [*Neural Tangent Kernel: Convergence and Generalization in Neural Networks*](https://arxiv.org/abs/1806.07572) (2018) | The neural tangent kernel and its constancy at infinite width (chapter 15). |
| Chizat, Oyallon, and Bach, [*On Lazy Training in Differentiable Programming*](https://arxiv.org/abs/1812.07956) (2019) | Lazy training and its dependence on the output scale (chapter 15). |
| Yang and Hu, [*Feature Learning in Infinite-Width Neural Networks*](https://arxiv.org/abs/2011.14522) (2021) | Parameterizations and the maximal-update limit (chapter 15). |
