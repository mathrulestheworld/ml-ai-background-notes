[Background Notes](../../README.md) › [Generative AI](../README.md)

# Paper links

# <a id="papers"></a>Papers

The notes cite the original papers where their ideas arise. This index collects the principal ones by topic; the chapters give further references inline.

## <a id="foundations-and-divergences"></a>Foundations and divergences

| Work | Idea developed in the module |
| --- | --- |
| Bengio, Courville, and Vincent, [*Representation Learning: A Review and New Perspectives*](https://arxiv.org/abs/1206.5538) (2013) | The manifold hypothesis: natural data concentrate near sets of much lower dimension than the space they occupy (chapter 1). |
| Theis, van den Oord, and Bethge, [*A Note on the Evaluation of Generative Models*](https://arxiv.org/abs/1511.01844) (2016) | Dequantization, which makes the likelihoods of continuous models on discrete pixels meaningful, and evaluation with respect to the intended application (chapters 1 and 13). |
| Minka, [*Divergence Measures and Message Passing*](https://www.microsoft.com/en-us/research/publication/divergence-measures-and-message-passing/) (2005) | The mode-seeking minimizers of the reverse KL divergence, against the mass-covering forward divergence (chapter 1). |
| Nowozin, Cseke, and Tomioka, [*f-GAN: Training Generative Neural Samplers Using Variational Divergence Minimization*](https://arxiv.org/abs/1606.00709) (2016) | The variational form of f-divergences, which lets any of them be estimated and minimized adversarially from samples (chapters 1 and 5). |
| Arjovsky and Bottou, [*Towards Principled Methods for Training Generative Adversarial Networks*](https://arxiv.org/abs/1701.04862) (2017) | Divergences built on density ratios give no signal when the supports are disjoint, where the Jensen–Shannon divergence is stuck at $`\log 2`$ (chapter 1). |
| Gretton et al., [*A Kernel Two-Sample Test*](https://jmlr.org/papers/v13/gretton12a.html) (2012) | The maximum mean discrepancy and its closed-form estimate from kernel evaluations (chapter 1). |
| Mohamed and Lakshminarayanan, [*Learning in Implicit Generative Models*](https://arxiv.org/abs/1610.03483) (2016) | A classifier trained to tell data from model samples estimates the log density ratio (chapter 1). |
| Xiao, Kreis, and Vahdat, [*Tackling the Generative Learning Trilemma with Denoising Diffusion GANs*](https://arxiv.org/abs/2112.07804) (2022) | The trilemma among sample quality, coverage of the modes, and fast sampling (chapter 1). |

## <a id="autoregressive-models"></a>Autoregressive models

| Work | Idea developed in the module |
| --- | --- |
| Larochelle and Murray, [*The Neural Autoregressive Distribution Estimator*](https://proceedings.mlr.press/v15/larochelle11a.html) (2011) | NADE: one hidden layer shared across all conditionals, so that all of them cost as much as one ordinary layer (chapters 1 and 2). |
| Germain et al., [*MADE: Masked Autoencoder for Distribution Estimation*](https://arxiv.org/abs/1502.03509) (2015) | Masked weights in an ordinary autoencoder give all conditionals in one pass (chapters 2 and 4). |
| van den Oord, Kalchbrenner, and Kavukcuoglu, [*Pixel Recurrent Neural Networks*](https://arxiv.org/abs/1601.06759) (2016) | PixelRNN and PixelCNN: pixels in raster order, with two-dimensional LSTMs or masked convolutions and a 256-way softmax (chapters 1 and 2). |
| van den Oord et al., [*Conditional Image Generation with PixelCNN Decoders*](https://arxiv.org/abs/1606.05328) (2016) | The gated PixelCNN, whose vertical and horizontal stacks remove the blind spot of masked convolutions (chapter 2). |
| Salimans et al., [*PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications*](https://arxiv.org/abs/1701.05517) (2017) | A discretized mixture of logistic distributions in place of the softmax over intensities (chapter 2). |
| Child et al., [*Generating Long Sequences with Sparse Transformers*](https://arxiv.org/abs/1904.10509) (2019) | Self-attention over pixels factorized into strided patterns of cost $`O(n\sqrt n)`$ (chapter 2). |
| Chen et al., [*Generative Pretraining from Pixels*](https://proceedings.mlr.press/v119/chen20s.html) (2020) | Image GPT: next-pixel prediction learns features that are good for classification (chapter 2). |
| van den Oord et al., [*WaveNet: A Generative Model for Raw Audio*](https://arxiv.org/abs/1609.03499) (2016) | Dilated causal convolutions and μ-law quantization for raw speech (chapter 2). |
| van den Oord et al., [*Parallel WaveNet: Fast High-Fidelity Speech Synthesis*](https://arxiv.org/abs/1711.10433) (2018) | A WaveNet distilled into an inverse autoregressive flow that generates all samples in parallel (chapters 2 and 4). |
| Tian et al., [*Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction*](https://arxiv.org/abs/2404.02905) (2024) | Autoregression over scales, predicting the whole image at the next finer scale at each step (chapter 2). |
| Li et al., [*Autoregressive Image Generation without Vector Quantization*](https://arxiv.org/abs/2406.11838) (2024) | Masked autoregressive models of continuous tokens, with a small diffusion model for each token's distribution (chapter 2). |

## <a id="variational-autoencoders"></a>Variational autoencoders

| Work | Idea developed in the module |
| --- | --- |
| Kingma and Welling, [*Auto-Encoding Variational Bayes*](https://arxiv.org/abs/1312.6114) (2014), and Rezende, Mohamed, and Wierstra, [*Stochastic Backpropagation and Approximate Inference in Deep Generative Models*](https://arxiv.org/abs/1401.4082) (2014) | The variational autoencoder: amortized inference by an encoder, trained on the ELBO with reparameterized gradients (chapters 1 and 3). |
| Dayan et al., [*The Helmholtz Machine*](https://doi.org/10.1162/neco.1995.7.5.889) (1995) | A generative network paired with a recognition network, the origin of amortized inference (chapters 1 and 3). |
| Williams, [*Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*](https://doi.org/10.1007/BF00992696) (1992) | REINFORCE, the score-function gradient estimator, which needs only samples and log-probabilities but has large variance (chapters 3 and 12). |
| Burda, Grosse, and Salakhutdinov, [*Importance Weighted Autoencoders*](https://arxiv.org/abs/1509.00519) (2016) | The importance-weighted bound, which tightens as samples are added and estimates held-out log-likelihoods (chapter 3). |
| Alemi et al., [*Fixing a Broken ELBO*](https://arxiv.org/abs/1711.00464) (2018) | The negative ELBO as distortion plus rate, the terms of lossy compression (chapter 3). |
| Higgins et al., [*β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*](https://openreview.net/forum?id=Sy2fzU9gl) (2017), and Locatello et al., [*Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations*](https://arxiv.org/abs/1811.12359) (2019) | A weight on the rate to encourage disentangled codes, and the proof that unsupervised disentanglement is impossible without inductive biases (chapter 3). |
| Bowman et al., [*Generating Sentences from a Continuous Space*](https://arxiv.org/abs/1511.06349) (2016) | Posterior collapse under a powerful autoregressive decoder, and KL annealing with dropped decoder inputs as a remedy (chapter 3). |
| Hoffman and Johnson, [*ELBO Surgery: Yet Another Way to Carve Up the Variational Evidence Lower Bound*](http://approximateinference.org/2016/accepted/HoffmanJohnson2016.pdf) (2016) | The rate as mutual information plus the divergence of the aggregate posterior from the prior, whose mismatch leaves holes in the prior (chapter 3). |
| Sohn, Lee, and Yan, [*Learning Structured Output Representation Using Deep Conditional Generative Models*](https://proceedings.neurips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html) (2015) | The conditional VAE, with a latent for the variation that the condition leaves open (chapter 3). |
| Vahdat and Kautz, [*NVAE: A Deep Hierarchical Variational Autoencoder*](https://arxiv.org/abs/2007.03898) (2020), and Child, [*Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images*](https://arxiv.org/abs/2011.10650) (2021) | Hierarchical VAEs made deep enough to generate large natural images and to beat PixelCNN in likelihood (chapter 3). |

## <a id="normalizing-flows"></a>Normalizing flows

| Work | Idea developed in the module |
| --- | --- |
| Rezende and Mohamed, [*Variational Inference with Normalizing Flows*](https://arxiv.org/abs/1505.05770) (2015) | Invertible maps that transform the data step by step into a standard normal, with densities from the change of variables (chapter 4). |
| Dinh, Krueger, and Bengio, [*NICE: Non-Linear Independent Components Estimation*](https://arxiv.org/abs/1410.8516) (2015), and Dinh, Sohl-Dickstein, and Bengio, [*Density Estimation Using Real NVP*](https://arxiv.org/abs/1605.08803) (2017) | Coupling layers, additive and then affine, whose triangular Jacobians reduce the log-determinant to a sum (chapters 1 and 4). |
| Kingma and Dhariwal, [*Glow: Generative Flow with Invertible 1x1 Convolutions*](https://arxiv.org/abs/1807.03039) (2018) | Learned invertible $`1\times1`$ convolutions between couplings, and attributes of faces changed along directions in the latent space (chapter 4). |
| Papamakarios, Pavlakou, and Murray, [*Masked Autoregressive Flow for Density Estimation*](https://arxiv.org/abs/1705.07057) (2017) | An autoregressive flow built on MADE, fast to evaluate and slow to sample (chapter 4). |
| Kingma et al., [*Improving Variational Inference with Inverse Autoregressive Flow*](https://arxiv.org/abs/1606.04934) (2016) | The inverse autoregressive flow, fast to sample, and free bits against posterior collapse (chapters 3 and 4). |
| Durkan et al., [*Neural Spline Flows*](https://arxiv.org/abs/1906.04032) (2019) | Monotonic rational-quadratic splines in place of affine maps in each layer (chapter 4). |
| Chen et al., [*Residual Flows for Invertible Generative Modeling*](https://arxiv.org/abs/1906.02735) (2019) | Residual blocks made invertible by a Lipschitz constraint, with a random power-series estimate of the log-determinant (chapter 4). |
| Chen et al., [*Neural Ordinary Differential Equations*](https://arxiv.org/abs/1806.07366) (2018) | Flows defined by an ordinary differential equation, and the instantaneous change of variables (chapter 4). |
| Grathwohl et al., [*FFJORD: Free-Form Continuous Dynamics for Scalable Reversible Generative Models*](https://arxiv.org/abs/1810.01367) (2019) | Hutchinson's trace estimator for the log-density of continuous flows (chapter 4). |
| Cornish et al., [*Relaxing Bijectivity Constraints with Continuously Indexed Normalising Flows*](https://arxiv.org/abs/1909.13833) (2020) | A bijection preserves topology, so a flow joins separated clusters by a filament (chapter 4). |
| Ho et al., [*Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design*](https://arxiv.org/abs/1902.00275) (2019) | Dequantization noise learned with a variational distribution (chapter 4). |
| Noé et al., [*Boltzmann Generators: Sampling Equilibrium States of Many-Body Systems with Deep Learning*](https://doi.org/10.1126/science.aaw1147) (2019) | Flows trained on a known energy to sample molecular configurations, with exact densities to reweight the samples (chapter 4). |
| Zhai et al., [*Normalizing Flows Are Capable Generative Models*](https://arxiv.org/abs/2412.06329) (2025) | TarFlow: a transformer autoregressive flow with samples comparable to those of diffusion models (chapter 4). |

## <a id="generative-adversarial-networks"></a>Generative adversarial networks

| Work | Idea developed in the module |
| --- | --- |
| Goodfellow et al., [*Generative Adversarial Networks*](https://arxiv.org/abs/1406.2661) (2014) | A generator trained against a discriminator, with no density to evaluate (chapters 1 and 5). |
| Mescheder, Geiger, and Nowozin, [*Which Training Methods for GANs Do Actually Converge?*](https://arxiv.org/abs/1801.04406) (2018) | The Dirac GAN, in which simultaneous gradient steps circle the equilibrium, and gradient penalties that make training converge (chapter 5). |
| Metz et al., [*Unrolled Generative Adversarial Networks*](https://arxiv.org/abs/1611.02163) (2017) | A generator that anticipates unrolled discriminator updates, and the eight Gaussians on a circle as a test of mode coverage (chapter 5). |
| Salimans et al., [*Improved Techniques for Training GANs*](https://arxiv.org/abs/1606.03498) (2016) | Minibatch discrimination against mode collapse, and the Inception score (chapters 5 and 13). |
| Arjovsky, Chintala, and Bottou, [*Wasserstein GAN*](https://arxiv.org/abs/1701.07875) (2017) | A critic that estimates the Wasserstein-1 distance, which decreases smoothly even when the distributions do not overlap (chapters 1 and 5). |
| Gulrajani et al., [*Improved Training of Wasserstein GANs*](https://arxiv.org/abs/1704.00028) (2017) | A gradient penalty in place of weight clipping to enforce the Lipschitz constraint (chapter 5). |
| Miyato et al., [*Spectral Normalization for Generative Adversarial Networks*](https://arxiv.org/abs/1802.05957) (2018) | Each weight matrix divided by its largest singular value to bound the discriminator's Lipschitz constant (chapter 5). |
| Radford, Metz, and Chintala, [*Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*](https://arxiv.org/abs/1511.06434) (2016) | DCGAN: convolutional architectures that trained reliably, and arithmetic on latent codes (chapter 5). |
| Karras et al., [*Progressive Growing of GANs for Improved Quality, Stability, and Variation*](https://arxiv.org/abs/1710.10196) (2018) | Layers added for higher resolutions during training, up to $`1024\times1024`$ faces (chapter 5). |
| Karras, Laine, and Aila, [*A Style-Based Generator Architecture for Generative Adversarial Networks*](https://arxiv.org/abs/1812.04948) (2019), and Karras et al., [*Analyzing and Improving the Image Quality of StyleGAN*](https://arxiv.org/abs/1912.04958) (2020) | StyleGAN: a mapping network, style modulation at every layer, and noise inputs, and the removal of its artifacts in StyleGAN2 (chapter 5). |
| Mirza and Osindero, [*Conditional Generative Adversarial Nets*](https://arxiv.org/abs/1411.1784) (2014) | The condition given to both the generator and the discriminator (chapter 5). |
| Brock, Donahue, and Simonyan, [*Large Scale GAN Training for High Fidelity Natural Image Synthesis*](https://arxiv.org/abs/1809.11096) (2019) | BigGAN: class-conditional GANs at scale, and the truncation trick (chapter 5). |
| Isola et al., [*Image-to-Image Translation with Conditional Adversarial Networks*](https://arxiv.org/abs/1611.07004) (2017), and Zhu et al., [*Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks*](https://arxiv.org/abs/1703.10593) (2017) | Translation of paired images with a patch discriminator, and of unpaired collections with cycle consistency (chapter 5). |

## <a id="energy-based-models-and-score-matching"></a>Energy-based models and score matching

| Work | Idea developed in the module |
| --- | --- |
| LeCun et al., [*A Tutorial on Energy-Based Learning*](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf) (2006) | Energy-based models: any network with a scalar output, at the price of an intractable partition function (chapter 6). |
| Ackley, Hinton, and Sejnowski, [*A Learning Algorithm for Boltzmann Machines*](https://doi.org/10.1207/s15516709cog0901_7) (1985) | The Boltzmann machine, an energy-based model of binary units with pairwise interactions (chapters 1 and 6). |
| Hinton, [*Training Products of Experts by Minimizing Contrastive Divergence*](https://doi.org/10.1162/089976602760128018) (2002) | Contrastive divergence: short chains started at training examples (chapter 6). |
| Du and Mordatch, [*Implicit Generation and Generalization in Energy-Based Models*](https://arxiv.org/abs/1903.08689) (2019) | Convolutional energies trained with Langevin chains and a replay buffer of past samples (chapter 6). |
| Gutmann and Hyvärinen, [*Noise-Contrastive Estimation: A New Estimation Principle for Unnormalized Statistical Models*](https://proceedings.mlr.press/v9/gutmann10a.html) (2010) | Logistic regression against a known noise distribution, with the log partition function as one more parameter (chapter 6). |
| Hyvärinen, [*Estimation of Non-Normalized Statistical Models by Score Matching*](https://jmlr.org/papers/v6/hyvarinen05a.html) (2005) | Score matching, an objective in the model's score and the data alone (chapter 6). |
| Vincent, [*A Connection between Score Matching and Denoising Autoencoders*](https://doi.org/10.1162/NECO_a_00142) (2011) | Denoising score matching: regressing onto the score of the added noise learns the score of the noisy data (chapters 6 and 7). |
| Efron, [*Tweedie's Formula and Selection Bias*](https://doi.org/10.1198/jasa.2011.tm11181) (2011) | Tweedie's formula, by which the optimal denoiser and the score of noisy data determine each other (chapters 6 and 7). |
| Song and Ermon, [*Generative Modeling by Estimating Gradients of the Data Distribution*](https://arxiv.org/abs/1907.05600) (2019) | Noise-conditional score networks and annealed Langevin dynamics, and the blindness of the score to the weights of separated modes (chapters 1 and 6). |
| Song and Ermon, [*Improved Techniques for Training Score-Based Generative Models*](https://arxiv.org/abs/2006.09011) (2020) | A largest noise level as large as the greatest distance between training images, for generation at up to $`256\times256`$ (chapter 6). |

## <a id="denoising-diffusion"></a>Denoising diffusion

| Work | Idea developed in the module |
| --- | --- |
| Sohl-Dickstein et al., [*Deep Unsupervised Learning Using Nonequilibrium Thermodynamics*](https://arxiv.org/abs/1503.03585) (2015) | Generation by learning to reverse a gradual noising process (chapters 1 and 7). |
| Ho, Jain, and Abbeel, [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239) (2020) | DDPM: training reparameterized as denoising, connected to score matching, with high-quality samples (chapters 1 and 7). |
| Nichol and Dhariwal, [*Improved Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2102.09672) (2021) | The cosine schedule, and reverse variances learned by the network (chapter 7). |
| Lin et al., [*Common Diffusion Noise Schedules and Sample Steps Are Flawed*](https://arxiv.org/abs/2305.08891) (2024) | Schedules that leave signal at the last step, and the rescaling of guided predictions to limit overexposure (chapters 7 and 10). |
| Kingma et al., [*Variational Diffusion Models*](https://arxiv.org/abs/2107.00630) (2021) | The continuous-time bound, which depends on the schedule only through its endpoints, and a schedule learned to reduce variance (chapter 7). |
| Kingma and Gao, [*Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation*](https://arxiv.org/abs/2303.00848) (2023) | Weightings that decrease with the log signal-to-noise ratio as bounds on the likelihood of noise-augmented data (chapter 7). |
| Hang et al., [*Efficient Diffusion Training via Min-SNR Weighting Strategy*](https://arxiv.org/abs/2303.09556) (2023) | Min-SNR weighting, which balances the noise levels as competing tasks (chapter 7). |
| Song, Meng, and Ermon, [*Denoising Diffusion Implicit Models*](https://arxiv.org/abs/2010.02502) (2021) | DDIM: non-Markovian processes that share the DDPM marginals, and the deterministic sampler used for fast sampling and inversion (chapters 7, 8, 9, and 10). |
| Dhariwal and Nichol, [*Diffusion Models Beat GANs on Image Synthesis*](https://arxiv.org/abs/2105.05233) (2021) | An ablated U-Net with adaptive group normalization, and classifier guidance (chapters 7 and 10). |

## <a id="sdes-samplers-and-distillation"></a>SDEs, samplers, and distillation

| Work | Idea developed in the module |
| --- | --- |
| Song et al., [*Score-Based Generative Modeling through Stochastic Differential Equations*](https://arxiv.org/abs/2011.13456) (2021) | Noising as an SDE, the reverse-time SDE, and the probability-flow ODE with exact likelihoods (chapters 8 and 10). |
| Anderson, [*Reverse-Time Diffusion Equation Models*](https://www.sciencedirect.com/science/article/pii/0304414982900515) (1982) | A diffusion run backward in time is again a diffusion, with the score in its drift (chapter 8). |
| Karras et al., [*Elucidating the Design Space of Diffusion-Based Generative Models*](https://arxiv.org/abs/2206.00364) (2022) | EDM: every noising process as a scaling and a noise level, $`\sigma(t)=t`$, preconditioning, and a tunable amount of Langevin noise in the sampler (chapter 8). |
| Zhang and Chen, [*Fast Sampling of Diffusion Models with Exponential Integrator*](https://arxiv.org/abs/2204.13902) (2023), and Lu et al., [*DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps*](https://arxiv.org/abs/2206.00927) (2022) | Exponential integrators that expand the denoiser in the log signal-to-noise ratio (chapter 8). |
| Lu et al., [*DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models*](https://arxiv.org/abs/2211.01095) (2022) | A multistep solver on the predicted data, stable under strong guidance (chapter 8). |
| Salimans and Ho, [*Progressive Distillation for Fast Sampling of Diffusion Models*](https://arxiv.org/abs/2202.00512) (2022) | Distillation that halves the number of steps in each round, and the velocity parameterization it required (chapters 7 and 8). |
| Song et al., [*Consistency Models*](https://proceedings.mlr.press/v202/song23a.html) (2023) | A network that maps every point of a probability-flow path to its origin, by distillation or by training alone (chapter 8). |
| Song and Dhariwal, [*Improved Techniques for Training Consistency Models*](https://arxiv.org/abs/2310.14189) (2023) | Consistency training without a teacher, with a Pseudo-Huber loss and log-normal noise levels (chapter 8). |
| Yin et al., [*One-Step Diffusion with Distribution Matching Distillation*](https://arxiv.org/abs/2311.18828) (2024) | A one-step generator trained with the difference between the teacher's score and the score of its own outputs (chapter 8). |
| Sauer et al., [*Adversarial Diffusion Distillation*](https://arxiv.org/abs/2311.17042) (2023) | Score distillation combined with a discriminator for generation in one to four steps, the method behind SDXL Turbo (chapters 5, 8, and 11). |
| Meng et al., [*On Distillation of Guided Diffusion Models*](https://arxiv.org/abs/2210.03142) (2023) | Guidance distillation: the guided prediction in one pass (chapter 11). |
| Luo et al., [*Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference*](https://arxiv.org/abs/2310.04378) (2023) | Stable Diffusion distilled into a consistency model that samples in 2 to 4 steps (chapter 11). |

## <a id="flow-matching-and-optimal-transport"></a>Flow matching and optimal transport

| Work | Idea developed in the module |
| --- | --- |
| Lipman et al., [*Flow Matching for Generative Modeling*](https://arxiv.org/abs/2210.02747) (2023) | Regressing the velocity field of a chosen path of distributions from noise to data (chapters 1 and 9). |
| Liu, Gong, and Liu, [*Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*](https://arxiv.org/abs/2209.03003) (2023) | Rectified flow, and reflow, which straightens the paths by retraining on the flow's own pairing of noise and data (chapter 9). |
| Albergo and Vanden-Eijnden, [*Building Normalizing Flows with Stochastic Interpolants*](https://arxiv.org/abs/2209.15571) (2023), and Albergo, Boffi, and Vanden-Eijnden, [*Stochastic Interpolants: A Unifying Framework for Flows and Diffusions*](https://arxiv.org/abs/2303.08797) (2023) | Stochastic interpolants, with extra noise along the path and sources that need not be Gaussian (chapter 9). |
| Liu et al., [*InstaFlow: One Step Is Enough for High-Quality Diffusion-Based Text-to-Image Generation*](https://arxiv.org/abs/2309.06380) (2024) | Reflow and distillation applied to Stable Diffusion for one-step text-to-image generation (chapter 9). |
| Benamou and Brenier, [*A Computational Fluid Mechanics Solution to the Monge–Kantorovich Mass Transfer Problem*](https://doi.org/10.1007/s002110050002) (2000) | The dynamic formulation of optimal transport: the velocity field of least kinetic energy, with straight paths (chapter 9). |
| Tong et al., [*Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport*](https://arxiv.org/abs/2302.00482) (2024) | Noise and data re-paired by optimal transport within each minibatch to straighten the paths (chapter 9). |
| Frans et al., [*One Step Diffusion via Shortcut Models*](https://arxiv.org/abs/2410.12557) (2025), and Geng et al., [*Mean Flows for One-Step Generative Modeling*](https://arxiv.org/abs/2505.13447) (2025) | Large steps of the flow map learned directly, from the step size as an input or from the average velocity over an interval (chapter 9). |
| Chen and Lipman, [*Flow Matching on General Geometries*](https://arxiv.org/abs/2302.03660) (2024) | Riemannian flow matching along geodesics on spheres, tori, and rotations (chapters 9 and 14). |
| Lipman et al., [*Flow Matching Guide and Code*](https://arxiv.org/abs/2412.06264) (2024) | The general theory of flow matching with a reference implementation (chapter 9). |

## <a id="guidance-inverse-problems-editing-and-control-signals"></a>Guidance, inverse problems, editing, and control signals

| Work | Idea developed in the module |
| --- | --- |
| Saharia et al., [*Image Super-Resolution via Iterative Refinement*](https://arxiv.org/abs/2104.07636) (2022), and [*Palette: Image-to-Image Diffusion Models*](https://arxiv.org/abs/2111.05826) (2022) | SR3 and Palette: a conditioning image concatenated with the noisy input (chapter 10). |
| Ho and Salimans, [*Classifier-Free Diffusion Guidance*](https://arxiv.org/abs/2207.12598) (2022) | One network trained with and without the condition, and extrapolation from the unconditional prediction (chapters 10, 11, and 14). |
| Nichol et al., [*GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models*](https://arxiv.org/abs/2112.10741) (2022) | Classifier-free guidance preferred by human raters to guidance by CLIP for text-to-image generation (chapter 10). |
| Chidambaram et al., [*What Does Guidance Do? A Fine-Grained Analysis in a Simple Setting*](https://arxiv.org/abs/2409.13074) (2024), and Bradley and Nakkiran, [*Classifier-Free Guidance Is a Predictor-Corrector*](https://arxiv.org/abs/2408.09000) (2024) | Guidance does not sample the tilted distribution, and in the limit of small steps it is a DDIM step followed by Langevin corrector steps (chapter 10). |
| Kynkäänniemi et al., [*Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models*](https://arxiv.org/abs/2404.07724) (2024) | Guidance applied only at intermediate noise levels (chapter 10). |
| Karras et al., [*Guiding a Diffusion Model with a Bad Version of Itself*](https://arxiv.org/abs/2406.02507) (2024) | Autoguidance, which extrapolates away from a smaller, less-trained version of the same model (chapter 10). |
| Lugmayr et al., [*RePaint: Inpainting Using Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2201.09865) (2022) | Inpainting by replacing the observed pixels, harmonized by resampling (chapter 10). |
| Chung et al., [*Diffusion Posterior Sampling for General Noisy Inverse Problems*](https://arxiv.org/abs/2209.14687) (2023) | DPS: the likelihood evaluated at the denoiser's prediction, with gradients through the denoiser (chapter 10). |
| Meng et al., [*SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations*](https://arxiv.org/abs/2108.01073) (2022) | Editing by noising a guide partway and denoising it (chapter 10). |
| Mokady et al., [*Null-Text Inversion for Editing Real Images Using Guided Diffusion Models*](https://arxiv.org/abs/2211.09794) (2023), and Hertz et al., [*Prompt-to-Prompt Image Editing with Cross Attention Control*](https://arxiv.org/abs/2208.01626) (2022) | Inversion that remains accurate under guidance, and edits that reuse the cross-attention maps of the original prompt (chapter 10). |
| Zhang, Rao, and Agrawala, [*Adding Conditional Control to Text-to-Image Diffusion Models*](https://arxiv.org/abs/2302.05543) (2023) | ControlNet: spatial conditions through a trainable copy of the encoder, joined to the locked model by zero-initialized convolutions (chapter 10). |
| Gal et al., [*An Image Is Worth One Word: Personalizing Text-to-Image Generation Using Textual Inversion*](https://arxiv.org/abs/2208.01618) (2023), and Ruiz et al., [*DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation*](https://arxiv.org/abs/2208.12242) (2023) | A subject learned from a few images, as the embedding of a new word or by fine-tuning with a prior-preservation loss (chapter 10). |
| Black et al., [*Training Diffusion Models with Reinforcement Learning*](https://arxiv.org/abs/2305.13301) (2024) | DDPO: the denoising chain as a sequence of actions, trained by policy gradients against a reward (chapter 10). |
| Xu et al., [*ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation*](https://arxiv.org/abs/2304.05977) (2023) | A reward model trained on expert comparisons, backpropagated through a single denoising step by ReFL (chapter 10). |
| Wallace et al., [*Diffusion Model Alignment Using Direct Preference Optimization*](https://arxiv.org/abs/2311.12908) (2024) | Diffusion-DPO, with variational bounds in place of the intractable likelihoods (chapter 10). |

## <a id="latent-diffusion-architectures-and-video"></a>Latent diffusion, architectures, and video

| Work | Idea developed in the module |
| --- | --- |
| Rombach et al., [*High-Resolution Image Synthesis with Latent Diffusion Models*](https://arxiv.org/abs/2112.10752) (2022) | Perceptual compression by an autoencoder and semantic compression by a diffusion model on its latents (chapters 1 and 11). |
| Ho et al., [*Cascaded Diffusion Models for High Fidelity Image Generation*](https://arxiv.org/abs/2106.15282) (2022) | Cascades of super-resolution models trained with conditioning augmentation (chapter 11). |
| Saharia et al., [*Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding*](https://arxiv.org/abs/2205.11487) (2022) | Imagen: a frozen T5-XXL text encoder, a cascade of upsamplers, and dynamic thresholding (chapters 10 and 11). |
| Ramesh et al., [*Hierarchical Text-Conditional Image Generation with CLIP Latents*](https://arxiv.org/abs/2204.06125) (2022) | DALL·E 2: a diffusion prior from caption to CLIP image embedding, and a diffusion decoder (chapter 11). |
| Hoogeboom, Heek, and Salimans, [*simple diffusion: End-to-End Diffusion for High Resolution Images*](https://arxiv.org/abs/2301.11093) (2023) | Pixel-space diffusion at high resolution by shifting the noise schedule with the resolution (chapter 11). |
| Chen, [*On the Importance of Noise Scheduling for Diffusion Models*](https://arxiv.org/abs/2301.10972) (2023) | Scaling the input to shift the log signal-to-noise ratio for high resolutions (chapter 11). |
| Esser et al., [*Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*](https://arxiv.org/abs/2403.03206) (2024) | Stable Diffusion 3: rectified flow with logit-normal times, the MM-DiT, and 16-channel latents (chapters 9 and 11). |
| Yu et al., [*Representation Alignment for Generation: Training Diffusion Transformers Is Easier than You Think*](https://arxiv.org/abs/2410.06940) (2025) | REPA: hidden states aligned with the features of a self-supervised encoder to speed up training (chapter 11). |
| Vahdat, Kreis, and Kautz, [*Score-Based Generative Modeling in Latent Space*](https://arxiv.org/abs/2106.05931) (2021) | A VAE and its diffusion prior trained jointly (chapter 11). |
| Podell et al., [*SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis*](https://arxiv.org/abs/2307.01952) (2023) | Two CLIP text encoders with a 2.6-billion-parameter U-Net (chapter 11). |
| Betker et al., [*Improving Image Generation with Better Captions*](https://cdn.openai.com/papers/dall-e-3.pdf) (2023) | DALL·E 3: a captioner that writes detailed descriptions, and a generator retrained mostly on its captions (chapter 11). |
| Peebles and Xie, [*Scalable Diffusion Models with Transformers*](https://arxiv.org/abs/2212.09748) (2023) | DiT: a vision transformer on latent patches, conditioned through adaLN-Zero, whose FID improves with compute (chapter 11). |
| Karras et al., [*Analyzing and Improving the Training Dynamics of Diffusion Models*](https://arxiv.org/abs/2312.02696) (2024) | Layers that preserve the magnitudes of activations and weights, and the averaging length chosen after training (chapter 11). |
| Chen et al., [*PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis*](https://arxiv.org/abs/2310.00426) (2024) | A text-to-image transformer trained in separate stages on recaptioned data at a fraction of the usual cost (chapter 11). |
| Ho et al., [*Video Diffusion Models*](https://arxiv.org/abs/2204.03458) (2022), and [*Imagen Video: High Definition Video Generation with Diffusion Models*](https://arxiv.org/abs/2210.02303) (2022) | A 3D U-Net factorized into spatial layers and temporal attention, trained on videos and images, and a cascade of video models (chapter 11). |
| Brooks et al., [*Video Generation Models as World Simulators*](https://openai.com/index/video-generation-models-as-world-simulators/) (2024) | Sora: a diffusion transformer on spacetime patches of a compressed video latent (chapter 11). |
| Polyak et al., [*Movie Gen: A Cast of Media Foundation Models*](https://arxiv.org/abs/2410.13720) (2024) | A 30-billion-parameter transformer trained by flow matching on latents compressed in time and space (chapter 11). |

## <a id="discrete-tokens-masked-and-discrete-diffusion-and-multimodal-models"></a>Discrete tokens, masked and discrete diffusion, and multimodal models

| Work | Idea developed in the module |
| --- | --- |
| Jang, Gu, and Poole, [*Categorical Reparameterization with Gumbel-Softmax*](https://arxiv.org/abs/1611.01144) (2017), and Maddison, Mnih, and Teh, [*The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables*](https://arxiv.org/abs/1611.00712) (2017) | The Gumbel-softmax relaxation of categorical samples (chapter 12). |
| Bengio, Léonard, and Courville, [*Estimating or Propagating Gradients through Stochastic Neurons for Conditional Computation*](https://arxiv.org/abs/1308.3432) (2013) | The straight-through estimator: the hard sample forward, the soft one or the identity backward (chapter 12). |
| van den Oord, Vinyals, and Kavukcuoglu, [*Neural Discrete Representation Learning*](https://arxiv.org/abs/1711.00937) (2017) | The VQ-VAE: a learned codebook, straight-through gradients, and a commitment loss (chapter 12). |
| Razavi, van den Oord, and Vinyals, [*Generating Diverse High-Fidelity Images with VQ-VAE-2*](https://arxiv.org/abs/1906.00446) (2019) | A hierarchy of token grids for global structure and detail (chapter 12). |
| Esser, Rombach, and Ommer, [*Taming Transformers for High-Resolution Image Synthesis*](https://arxiv.org/abs/2012.09841) (2021) | VQGAN: perceptual and adversarial losses for sharp reconstructions from few tokens, and the autoencoder recipe of latent diffusion (chapters 11 and 12). |
| Yu et al., [*Vector-Quantized Image Modeling with Improved VQGAN*](https://arxiv.org/abs/2110.04627) (2022) | Codebook lookup in a low-dimensional, normalized space against codebook collapse (chapter 12). |
| Ramesh et al., [*Zero-Shot Text-to-Image Generation*](https://arxiv.org/abs/2102.12092) (2021) | DALL·E: a discrete VAE and a transformer over text and image tokens (chapter 12). |
| Mentzer et al., [*Finite Scalar Quantization: VQ-VAE Made Simple*](https://arxiv.org/abs/2309.15505) (2024) | Rounding a few bounded numbers per position in place of a learned codebook (chapter 12). |
| Yu et al., [*Language Model Beats Diffusion – Tokenizer Is Key to Visual Generation*](https://arxiv.org/abs/2310.05737) (2024) | Lookup-free quantization, with which a language-model-style generator beat diffusion models (chapter 12). |
| Zeghidour et al., [*SoundStream: An End-to-End Neural Audio Codec*](https://arxiv.org/abs/2107.03312) (2021), and Défossez et al., [*High Fidelity Neural Audio Compression*](https://arxiv.org/abs/2210.13438) (2022) | Neural audio codecs with residual vector quantization (chapter 12). |
| Yu et al., [*Scaling Autoregressive Models for Content-Rich Text-to-Image Generation*](https://arxiv.org/abs/2206.10789) (2022) | Parti: an encoder–decoder transformer over image tokens scaled to 20 billion parameters (chapter 12). |
| Borsos et al., [*AudioLM: A Language Modeling Approach to Audio Generation*](https://arxiv.org/abs/2209.03143) (2023) | Semantic tokens predicted before coarse and fine acoustic tokens (chapter 12). |
| Wang et al., [*Neural Codec Language Models Are Zero-Shot Text to Speech Synthesizers*](https://arxiv.org/abs/2301.02111) (2023), and Copet et al., [*Simple and Controllable Music Generation*](https://arxiv.org/abs/2306.05284) (2023) | Speech and music as language modeling over codec tokens, with the codebook streams offset in time (chapter 12). |
| Chang et al., [*MaskGIT: Masked Generative Image Transformer*](https://arxiv.org/abs/2202.04200) (2022), and [*Muse: Text-to-Image Generation via Masked Generative Transformers*](https://arxiv.org/abs/2301.00704) (2023) | Masked token prediction with confidence-based unmasking in a few parallel steps (chapter 12). |
| Austin et al., [*Structured Denoising Diffusion Models in Discrete State-Spaces*](https://arxiv.org/abs/2107.03006) (2021) | D3PM: discrete forward processes defined by transition matrices, including an absorbing mask state (chapter 12). |
| Lou, Meng, and Ermon, [*Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution*](https://arxiv.org/abs/2310.16834) (2024) | SEDD: ratios between neighboring states in place of the score, fitted by score entropy (chapter 12). |
| Sahoo et al., [*Simple and Effective Masked Diffusion Language Models*](https://arxiv.org/abs/2406.07524) (2024), and Shi et al., [*Simplified and Generalized Masked Diffusion for Discrete Data*](https://arxiv.org/abs/2406.04329) (2024) | Masked diffusion and its simple likelihood bound (chapter 12). |
| Nie et al., [*Large Language Diffusion Models*](https://arxiv.org/abs/2502.09992) (2025) | LLaDA: an 8-billion-parameter masked diffusion language model trained from scratch (chapter 12). |
| Gat et al., [*Discrete Flow Matching*](https://arxiv.org/abs/2407.15595) (2024) | Flow matching over discrete states (chapter 12). |
| Alayrac et al., [*Flamingo: A Visual Language Model for Few-Shot Learning*](https://arxiv.org/abs/2204.14198) (2022) | A frozen language model with gated cross-attention to image features, and in-context learning with images (chapter 12). |
| Liu et al., [*Visual Instruction Tuning*](https://arxiv.org/abs/2304.08485) (2023) | LLaVA: image features projected into a language model's input and fine-tuned on visual instructions (chapter 12). |
| Chameleon Team, [*Chameleon: Mixed-Modal Early-Fusion Foundation Models*](https://arxiv.org/abs/2405.09818) (2024) | Early fusion: one transformer trained from the start on interleaved text and image tokens (chapter 12). |
| Zhou et al., [*Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model*](https://arxiv.org/abs/2408.11039) (2024) | One transformer with a next-token loss on text and a diffusion loss on continuous image patches (chapter 12). |

## <a id="evaluation-and-memorization"></a>Evaluation and memorization

| Work | Idea developed in the module |
| --- | --- |
| Nalisnick et al., [*Do Deep Generative Models Know What They Don't Know?*](https://arxiv.org/abs/1810.09136) (2019) | Models trained on CIFAR-10 assign higher likelihood to SVHN, so likelihood cannot detect unusual inputs (chapters 2 and 13). |
| Heusel et al., [*GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*](https://arxiv.org/abs/1706.08500) (2017) | The Fréchet inception distance, and separate learning rates for the two networks of a GAN (chapters 5 and 13). |
| Chong and Forsyth, [*Effectively Unbiased FID and Inception Score and Where to Find Them*](https://arxiv.org/abs/1911.07023) (2020) | The bias of FID at finite sample sizes, which depends on the model and can reverse rankings (chapter 13). |
| Parmar, Zhang, and Zhu, [*On Aliased Resizing and Surprising Subtleties in GAN Evaluation*](https://arxiv.org/abs/2104.11222) (2022) | The sensitivity of FID to resizing filters and compression (chapter 13). |
| Kynkäänniemi et al., [*The Role of ImageNet Classes in Fréchet Inception Distance*](https://arxiv.org/abs/2203.06026) (2023) | FID lowered by choosing samples in the Inception feature space, with no image improved (chapter 13). |
| Bińkowski et al., [*Demystifying MMD GANs*](https://arxiv.org/abs/1801.01401) (2018) | The kernel inception distance, an unbiased maximum mean discrepancy with a cubic kernel (chapter 13). |
| Jayasumana et al., [*Rethinking FID: Towards a Better Evaluation Metric for Image Generation*](https://arxiv.org/abs/2401.09603) (2024) | CMMD: maximum mean discrepancy with a Gaussian kernel on CLIP embeddings (chapter 13). |
| Stein et al., [*Exposing Flaws of Generative Model Evaluation Metrics and Their Unfair Treatment of Diffusion Models*](https://arxiv.org/abs/2306.04675) (2023) | Metrics compared with large-scale human judgments, and DINOv2 features in place of Inception ones (chapter 13). |
| Sajjadi et al., [*Assessing Generative Models via Precision and Recall*](https://arxiv.org/abs/1806.00035) (2018), and Kynkäänniemi et al., [*Improved Precision and Recall Metric for Assessing Generative Models*](https://arxiv.org/abs/1904.06991) (2019) | Fidelity and diversity measured separately, with supports estimated by nearest-neighbor balls (chapter 13). |
| Hessel et al., [*CLIPScore: A Reference-Free Evaluation Metric for Image Captioning*](https://arxiv.org/abs/2104.08718) (2021) | The agreement of an image with its prompt as the similarity of their CLIP embeddings (chapter 13). |
| Hu et al., [*TIFA: Accurate and Interpretable Text-to-Image Faithfulness Evaluation with Question Answering*](https://arxiv.org/abs/2303.11897) (2023), and Ghosh, Hajishirzi, and Schmidt, [*GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment*](https://arxiv.org/abs/2310.11513) (2023) | Prompts decomposed into questions for a visual question-answering model, or into objects, counts, positions, and colors checked by a detector (chapter 13). |
| Kirstain et al., [*Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation*](https://arxiv.org/abs/2305.01569) (2023) | PickScore: a preference model trained on user choices, as a proxy for human evaluation (chapter 13). |
| Meehan, Chaudhuri, and Dasgupta, [*A Non-Parametric Test to Detect Data-Copying in Generative Models*](https://arxiv.org/abs/2004.05675) (2020) | A test for samples systematically closer to training points than held-out data are (chapter 13). |
| Somepalli et al., [*Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models*](https://arxiv.org/abs/2212.03860) (2023) | Near-copies of training images among the generations of Stable Diffusion (chapter 13). |
| Carlini et al., [*Extracting Training Data from Diffusion Models*](https://arxiv.org/abs/2301.13188) (2023) | Deliberate extraction of memorized training images, concentrated on duplicated ones (chapter 13). |

## <a id="science-and-control"></a>Science and control

| Work | Idea developed in the module |
| --- | --- |
| Chi et al., [*Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*](https://arxiv.org/abs/2303.04137) (2023) | A conditional diffusion model over action sequences that samples one demonstrated behavior instead of their mean, run with a receding horizon (chapter 14). |
| Zhao et al., [*Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*](https://arxiv.org/abs/2304.13705) (2023) | ACT: chunks of actions predicted by a conditional VAE with a transformer (chapter 14). |
| Janner et al., [*Planning with Diffusion for Flexible Behavior Synthesis*](https://arxiv.org/abs/2205.09991) (2022) | Diffuser: planning by sampling trajectories, guided toward high return or inpainted toward a goal (chapter 14). |
| Ajay et al., [*Is Conditional Generative Modeling All You Need for Decision-Making?*](https://arxiv.org/abs/2211.15657) (2023) | States generated with classifier-free guidance on the return, and actions recovered by inverse dynamics (chapter 14). |
| Black et al., [*π0: A Vision-Language-Action Flow Model for General Robot Control*](https://arxiv.org/abs/2410.24164) (2024) | A flow-matching action expert attached to a vision–language model (chapter 14). |
| Satorras, Hoogeboom, and Welling, [*E(n) Equivariant Graph Neural Networks*](https://arxiv.org/abs/2102.09844) (2021) | Cheap updates that commute with rotations, translations, and reorderings of atoms (chapter 14). |
| Hoogeboom et al., [*Equivariant Diffusion for Molecule Generation in 3D*](https://arxiv.org/abs/2203.17003) (2022) | Positions and atom types generated jointly, with diffusion in the subspace of zero center of mass (chapter 14). |
| Corso et al., [*DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking*](https://arxiv.org/abs/2210.01776) (2023) | Docking as diffusion over translations, rotations, and torsions, a product of manifolds (chapter 14). |
| Buttenschoen, Morris, and Deane, [*PoseBusters: AI-Based Docking Methods Fail to Generate Physically Valid Poses or Generalise to Novel Sequences*](https://arxiv.org/abs/2308.05777) (2024) | Checks of chemical and physical plausibility that many deep-learning poses failed (chapter 14). |
| Jumper et al., [*Highly Accurate Protein Structure Prediction with AlphaFold*](https://doi.org/10.1038/s41586-021-03819-2) (2021) | Protein structure predicted from sequence with accuracy close to experiment (chapter 14). |
| Yim et al., [*SE(3) Diffusion Model with Application to Protein Backbone Generation*](https://arxiv.org/abs/2302.02277) (2023) | FrameDiff: diffusion on residue frames, with Gaussian noise on positions and Brownian motion on rotations (chapter 14). |
| Watson et al., [*De Novo Design of Protein Structure and Function with RFdiffusion*](https://doi.org/10.1038/s41586-023-06415-8) (2023) | A structure-prediction network fine-tuned as a denoiser, conditioned on targets, symmetries, and motifs, with designs tested in the laboratory (chapter 14). |
| Ingraham et al., [*Illuminating Protein Space with a Programmable Generative Model*](https://doi.org/10.1038/s41586-023-06728-8) (2023) | Chroma: composable conditioners in the spirit of guidance (chapter 14). |
| Abramson et al., [*Accurate Structure Prediction of Biomolecular Interactions with AlphaFold 3*](https://doi.org/10.1038/s41586-024-07487-w) (2024) | Structure prediction as generation, a diffusion model over the raw coordinates of every atom (chapter 14). |
| Lam et al., [*Learning Skillful Medium-Range Global Weather Forecasting*](https://doi.org/10.1126/science.adi2336) (2023) | GraphCast: a deterministic forecast trained with squared error, which predicts the mean of possible futures (chapter 14). |
| Price et al., [*Probabilistic Weather Forecasting with Machine Learning*](https://doi.org/10.1038/s41586-024-08252-9) (2025) | GenCast: ensembles sampled from a conditional diffusion model of the global atmosphere (chapter 14). |
| Gneiting and Raftery, [*Strictly Proper Scoring Rules, Prediction, and Estimation*](https://doi.org/10.1198/016214506000001437) (2007) | Proper scoring rules and the continuous ranked probability score (chapter 14). |
| Lorenz, [*Deterministic Nonperiodic Flow*](https://doi.org/10.1175/1520-0469%281963%29020%3C0130%3ADNF%3E2.0.CO%3B2) (1963) | The chaotic system on which the code compares single forecasts with ensembles (chapter 14). |

## <a id="theory"></a>Theory

| Work | Idea developed in the module |
| --- | --- |
| Chen et al., [*Sampling Is as Easy as Learning the Score: Theory for Diffusion Models with Minimal Data Assumptions*](https://arxiv.org/abs/2209.11215) (2023) | Polynomial convergence of the DDPM sampler given an accurate score, without log-concavity (chapter 15). |
| Benton et al., [*Nearly d-Linear Convergence Bounds for Diffusion Models via Stochastic Localization*](https://arxiv.org/abs/2308.03686) (2024) | A step count linear in the dimension up to logarithms, assuming only a finite second moment (chapter 15). |
| Chen et al., [*The Probability Flow ODE Is Provably Fast*](https://arxiv.org/abs/2305.11798) (2023) | Guarantees for the deterministic sampler with Langevin correction steps (chapter 15). |
| Oko, Akiyama, and Suzuki, [*Diffusion Models Are Minimax Optimal Distribution Estimators*](https://arxiv.org/abs/2303.01861) (2023) | Diffusion models reach the minimax rate of density estimation for smooth densities (chapter 15). |
| Biroli et al., [*Dynamical Regimes of Diffusion Models*](https://arxiv.org/abs/2402.18491) (2024) | Speciation and collapse onto training points under the exact empirical score (chapter 15). |
| Kadkhodaie et al., [*Generalization in Diffusion Models Arises from Geometry-Adaptive Harmonic Representations*](https://arxiv.org/abs/2310.02557) (2024) | The transition from memorization to generalization as the training set grows, and geometry-adaptive harmonic bases (chapter 15). |
| Kamb and Ganguli, [*An Analytic Theory of Creativity in Convolutional Diffusion Models*](https://arxiv.org/abs/2412.20292) (2025) | The equivariant local score machine, whose patch mosaics predict the outputs of trained convolutional models (chapter 15). |
| Wang and Vastola, [*The Hidden Linear Structure in Score-Based Models and Its Application*](https://arxiv.org/abs/2311.10892) (2023), and Li, Dai, and Qu, [*Understanding Generalizability of Diffusion Models Requires Rethinking the Hidden Gaussian Structure*](https://arxiv.org/abs/2410.24060) (2024) | Learned denoisers close to the linear denoiser of a Gaussian with the mean and covariance of the data, at high noise and increasingly as models generalize (chapter 15). |
| van der Schaaf and van Hateren, [*Modelling the Power Spectra of Natural Images: Statistics and Information*](https://www.sciencedirect.com/science/article/pii/0042698996000028) (1996), and Dieleman, [*Diffusion Is Spectral Autoregression*](https://sander.ai/2024/09/02/spectral-autoregression.html) (2024) | The power-law spectrum of natural images, and diffusion as generation from coarse to fine frequencies (chapter 15). |
| Rissanen, Heinonen, and Solin, [*Generative Modelling with Inverse Heat Dissipation*](https://arxiv.org/abs/2206.13397) (2023) | Generation by reversing a blurring process instead of a noising one (chapter 15). |
| Léonard, [*A Survey of the Schrödinger Problem and Some of Its Connections with Optimal Transport*](https://arxiv.org/abs/1308.0215) (2014) | The Schrödinger bridge as entropy-regularized optimal transport (chapter 15). |
| De Bortoli et al., [*Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling*](https://arxiv.org/abs/2106.01357) (2021), and Shi et al., [*Diffusion Schrödinger Bridge Matching*](https://arxiv.org/abs/2303.16852) (2023) | Bridges computed by iterative proportional fitting, and by iterative Markovian fitting (chapter 15). |
