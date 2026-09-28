[Background Notes](../README.md) › [Generative AI](README.md)

# Reading plan

## <a id="courses"></a>Courses

- **Main course — Stanford CS236, Fall 2023: Stefano Ermon, *Deep Generative Models*.** [Course homepage](https://deepgenerativemodels.github.io/) · [Syllabus with slides](https://deepgenerativemodels.github.io/syllabus.html) · [Lecture playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rPOWA-omMM6STXaWW4FvJT8) · [Course notes](https://deepgenerativemodels.github.io/notes/). The standard graduate course on the families of generative models: autoregressive models, variational autoencoders, normalizing flows, adversarial networks, energy-based models, score-based and diffusion models, evaluation, and discrete latent variables. Its notes cover the first half of the course.
- **Main course — MIT 6.S184, IAP 2026: Peter Holderrieth, *Introduction to Flow Matching and Diffusion Models*.** [Course homepage](https://diffusion.csail.mit.edu/2026/index.html) · [Lecture notes · PDF](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Lecture playlist](https://www.youtube.com/playlist?list=PL57nT7tSGAAXwjhDYcxEycx5W7YoSrZyt) · [Labs with solutions](https://github.com/eje24/iap-diffusion-labs/tree/2026). A short, self-contained course on modern diffusion and flow models through differential equations: flow matching, score matching, guidance, latent spaces and architectures, and discrete diffusion. Its notes ([arXiv version](https://arxiv.org/abs/2506.02070)) are the best single text for chapters 8–12. The [IAP 2025 version](https://diffusion.csail.mit.edu/2025/index.html) adds guest lectures on robotics and protein design.
- **Supplement — MIT 6.S978, Fall 2024: Kaiming He, *Deep Generative Models*.** [Course page with slides and readings](https://mit-6s978.github.io/). Lectures on VAEs, autoregressive models, adversarial networks, and diffusion, and reading sessions on tokenizers, flow matching, discrete diffusion, and applications. No recordings.
- **Supplement — UC Berkeley CS294-158, Spring 2024: Pieter Abbeel, Wilson Yan, Kevin Frans, and Philipp Wu, *Deep Unsupervised Learning*.** [Course page with slides](https://sites.google.com/view/berkeley-cs294-158-sp24/home) · [Lecture playlist](https://www.youtube.com/playlist?list=PLwRJQ4m4UJjPIvv4kgBkvu_uygrV3ut_U) · [Homework](https://github.com/rll/deepul). Recorded lectures on each family of models and on video generation, compression, multimodal models, and applications in science; its four homeworks implement autoregressive, latent-variable, adversarial, and diffusion models.
- **Supplement — Stanford CME 296, Spring 2026: Afshine and Shervine Amidi, *Diffusion and Large Vision Models*.** [Course homepage](https://cme296.stanford.edu/) · [Lecture playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rNdy8rt2rZ4T2xM0OjADnfu). A practical course on current image and video generators: diffusion, score matching, flow matching, latent spaces and guidance, architectures, training, and evaluation.

The order below follows the chapters of this module. Each block lists the material to cover and its primary lectures; further reading and exercises provide additional depth. Blocks 1–13 form the main plan; 14–15 are optional extensions. Timing is flexible.

Slide links are the courses' original files; four CS236 lectures are available only as PowerPoint files. Several prerequisites come from earlier modules: the evidence lower bound, Markov chain Monte Carlo, and Langevin-type samplers from AI chapter 10; mixtures, EM, and kernel density estimates from ML; autoencoders, contrastive image–text models, U-Nets, and vision transformers from DL; and autoregressive transformers, tokenization, sampling, and preference optimization from NLP and LLMs. The corresponding introductory lectures serve as review.

## <a id="topic-list"></a>Topic list

- [ ] [1. Foundations of generative modeling](#1-foundations-of-generative-modeling)
- [ ] [2. Autoregressive models](#2-autoregressive-models)
- [ ] [3. Variational autoencoders](#3-variational-autoencoders)
- [ ] [4. Normalizing flows](#4-normalizing-flows)
- [ ] [5. Generative adversarial networks](#5-generative-adversarial-networks)
- [ ] [6. Energy-based models and score matching](#6-energy-based-models-and-score-matching)
- [ ] [7. Denoising diffusion models](#7-denoising-diffusion-models)
- [ ] [8. Diffusion SDEs and fast sampling](#8-diffusion-sdes-and-fast-sampling)
- [ ] [9. Flow matching](#9-flow-matching)
- [ ] [10. Guidance and conditional generation](#10-guidance-and-conditional-generation)
- [ ] [11. Latent diffusion and large-scale generation](#11-latent-diffusion-and-large-scale-generation)
- [ ] [12. Discrete tokens and multimodal generation](#12-discrete-tokens-and-multimodal-generation)
- [ ] [13. Evaluating generative models](#13-evaluating-generative-models)
- [ ] [14. Generative models for science and control — optional](#14-generative-models-for-science-and-control-optional)
- [ ] [15. The theory of diffusion models — optional](#15-the-theory-of-diffusion-models-optional)

### <a id="reference-books-used-below"></a>Reference books used below

- **PML2:** Kevin Murphy, [Probabilistic Machine Learning: Advanced Topics](https://probml.github.io/pml-book/book2.html) (MIT Press, 2023). Free draft; part IV, chapters 20–26, covers every family of generative models with consistent notation.
- **UDL:** Simon Prince, [Understanding Deep Learning](https://udlbook.github.io/udlbook/) (MIT Press, 2023). Free; chapters 14–18 on unsupervised learning, adversarial networks, normalizing flows, variational autoencoders, and diffusion models, with notebooks.
- **6.S184 notes:** Holderrieth and Erives, [An Introduction to Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) (2026). Free; flow and diffusion models, flow matching, score matching, guidance, architectures, and discrete diffusion.
- **PDM:** Lai, Song, Kim, Mitsufuji, and Ermon, [The Principles of Diffusion Models](https://arxiv.org/abs/2510.21890) (2025). Free; a monograph that unifies the variational, score-based, and flow-based views of diffusion and covers guidance and fast solvers.
- **FMGC:** Lipman et al., [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264) (2024). Free; flow matching in continuous, discrete, and Riemannian settings with a PyTorch library.

## <a id="1-foundations-of-generative-modeling"></a>1. Foundations of generative modeling

**Topics:** What a generative model is for: density estimation, sampling, and representation; the data manifold and the curse of dimensionality; maximum likelihood as minimizing the forward KL divergence; forward and reverse KL, mode covering and mode seeking; f-divergences and integral probability metrics, including maximum mean discrepancy and the Wasserstein distance; explicit, approximate, and implicit models; the trade-offs among sample quality, diversity, likelihood, and speed.

- **CS236 lecture 1: Introduction.** [Slides · PPTX](https://deepgenerativemodels.github.io/assets/slides/cs236_lecture1_2023.pptx) · [Video](https://www.youtube.com/watch?v=XZ0PMRWXBEU)
- **CS236 lecture 2: Background.** [Slides · PDF](https://deepgenerativemodels.github.io/assets/slides/cs236_lecture2.pdf) · [Video](https://www.youtube.com/watch?v=rNEujZmD2Tg)
- **CS236 lecture 4: Maximum likelihood learning.** [Slides · PDF](https://deepgenerativemodels.github.io/assets/slides/cs236_lecture4.pdf) · [Video](https://www.youtube.com/watch?v=bt3dqcbMLa0)
- **6.S978 lecture 1: Introduction.** [Slides · PDF](https://mit-6s978.github.io/assets/pdfs/lec1_intro.pdf)
- **PML2 chapter 20: Generative models, an overview.**

**Further reading:** CS236 [notes, introduction](https://deepgenerativemodels.github.io/notes/introduction/); Goodfellow, [NIPS 2016 tutorial: generative adversarial networks](https://arxiv.org/abs/1701.00160), section 2, for a taxonomy of models by how they use likelihood; Bond-Taylor et al., [Deep generative modelling: a comparative review](https://arxiv.org/abs/2103.04922) (2022).

## <a id="2-autoregressive-models"></a>2. Autoregressive models

**Topics:** The chain rule for images and audio; fully visible belief networks and NADE; masked autoencoders for distribution estimation (MADE); PixelRNN and PixelCNN, masked convolutions and the blind spot, gated and conditional PixelCNN; discretized logistic mixtures; WaveNet and dilated causal convolutions; transformers over pixels; orderings; exact likelihood in bits per dimension; the cost of sequential sampling.

- **CS236 lecture 3: Autoregressive models.** [Slides · PDF](https://deepgenerativemodels.github.io/assets/slides/cs236_lecture3.pdf) · [Video](https://www.youtube.com/watch?v=tRArbBf-AbI) · [Notes](https://deepgenerativemodels.github.io/notes/autoregressive/)
- **CS294-158 lecture 2: Autoregressive models.** [Video](https://www.youtube.com/watch?v=2ojJUSMf-_g)
- **6.S978 lecture 3: Autoregressive models.** [Slides · PDF](https://mit-6s978.github.io/assets/pdfs/lec3_ar.pdf)
- **PML2 chapter 22: Autoregressive models.**

**Further reading:** Germain et al., [MADE](https://arxiv.org/abs/1502.03509) (2015); van den Oord, Kalchbrenner, and Kavukcuoglu, [Pixel recurrent neural networks](https://arxiv.org/abs/1601.06759) (2016); van den Oord et al., [WaveNet](https://arxiv.org/abs/1609.03499) (2016); Salimans et al., [PixelCNN++](https://arxiv.org/abs/1701.05517) (2017). CS294-158 homework 1 implements autoregressive models of images.

## <a id="3-variational-autoencoders"></a>3. Variational autoencoders

**Topics:** Latent-variable models and the intractable marginal likelihood; the evidence lower bound and amortized inference; the reparameterization trick and its variance; Gaussian and Bernoulli decoders; posterior collapse and KL annealing; the rate–distortion view and β-VAE; importance-weighted bounds; conditional VAEs; hierarchical VAEs.

- **CS236 lectures 5–6: Variational autoencoders.** [Slides 5 · PDF](https://deepgenerativemodels.github.io/assets/slides/cs236_lecture5.pdf) · [Video 5](https://www.youtube.com/watch?v=MAGBUh77bNg) · [Slides 6 · PDF](https://deepgenerativemodels.github.io/assets/slides/cs236_lecture6.pdf) · [Video 6](https://www.youtube.com/watch?v=8cO61e_8oPY) · [Notes](https://deepgenerativemodels.github.io/notes/vae/)
- **6.S978 lecture 2: Variational autoencoders.** [Slides · PDF](https://mit-6s978.github.io/assets/pdfs/lec2_vae.pdf)
- **CS294-158 lecture 4: Latent variable models.** [Video](https://www.youtube.com/watch?v=NlIqjtbjjRE)
- **UDL chapter 17: Variational autoencoders; PML2 chapter 21.**

**Further reading:** Kingma and Welling, [An introduction to variational autoencoders](https://arxiv.org/abs/1906.02691) (2019); Burda, Grosse, and Salakhutdinov, [Importance weighted autoencoders](https://arxiv.org/abs/1509.00519) (2016); Alemi et al., [Fixing a broken ELBO](https://arxiv.org/abs/1711.00464) (2018); Vahdat and Kautz, [NVAE](https://arxiv.org/abs/2007.03898) (2020). CS294-158 homework 2 trains VAEs on images.

## <a id="4-normalizing-flows"></a>4. Normalizing flows

**Topics:** The change-of-variables formula; triangular Jacobians; coupling layers in NICE, RealNVP, and Glow; autoregressive flows, MAF and IAF, and their duality; spline flows; residual and continuous-time flows, neural ODEs, and trace estimation; dequantization; what invertibility costs.

- **CS236 lectures 7–8: Normalizing flows.** [Slides 7 · PDF](https://deepgenerativemodels.github.io/assets/slides/cs236_lecture7.pdf) · [Video 7](https://www.youtube.com/watch?v=m6dKKRsZwBQ) · [Slides 8 · PDF](https://deepgenerativemodels.github.io/assets/slides/cs236_lecture8.pdf) · [Video 8](https://www.youtube.com/watch?v=qgTvgBCOyn8) · [Notes](https://deepgenerativemodels.github.io/notes/flow/)
- **CS294-158 lecture 3: Flow models.** [Video](https://www.youtube.com/watch?v=SkSDCzz41Vs)
- **UDL chapter 16: Normalizing flows; PML2 chapter 23.**

**Further reading:** Papamakarios et al., [Normalizing flows for probabilistic modeling and inference](https://arxiv.org/abs/1912.02762) (2021); Dinh, Sohl-Dickstein, and Bengio, [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803) (2017); Chen et al., [Neural ordinary differential equations](https://arxiv.org/abs/1806.07366) (2018); Lilian Weng, [Flow-based deep generative models](https://lilianweng.github.io/posts/2018-10-13-flow-models/).

## <a id="5-generative-adversarial-networks"></a>5. Generative adversarial networks

**Topics:** The minimax game and the optimal discriminator; the Jensen–Shannon divergence; the non-saturating loss; training dynamics, oscillation, and mode collapse; f-GANs; Wasserstein GANs, weight clipping, and gradient penalties; spectral normalization; DCGAN, progressive growing, StyleGAN, and BigGAN; conditional and image-to-image GANs; GAN inversion; adversarial losses inside other models.

- **CS236 lectures 9–10: Generative adversarial networks.** [Slides 9 · PDF](https://deepgenerativemodels.github.io/assets/slides/cs236_lecture9.pdf) · [Video 9](https://www.youtube.com/watch?v=3Zv-gokhLu8) · [Slides 10 · PDF](https://deepgenerativemodels.github.io/assets/slides/cs236_lecture10.pdf) · [Video 10](https://www.youtube.com/watch?v=M3Fkvu78ZXc) · [Notes](https://deepgenerativemodels.github.io/notes/gan/)
- **6.S978 lecture 4: Generative adversarial networks.** [Slides · PDF](https://mit-6s978.github.io/assets/pdfs/lec4_gan.pdf)
- **CS294-158 lecture 5: GANs and implicit models.** [Video](https://www.youtube.com/watch?v=lFAHPJS2HHc)
- **UDL chapter 15: Generative adversarial networks; PML2 chapter 26.**

**Further reading:** Goodfellow et al., [Generative adversarial nets](https://arxiv.org/abs/1406.2661) (2014); Arjovsky, Chintala, and Bottou, [Wasserstein GAN](https://arxiv.org/abs/1701.07875) (2017); Mescheder, Geiger, and Nowozin, [Which training methods for GANs do actually converge?](https://arxiv.org/abs/1801.04406) (2018); Karras, Laine, and Aila, [StyleGAN](https://arxiv.org/abs/1812.04948) (2019). CS294-158 homework 3 trains adversarial networks.

## <a id="6-energy-based-models-and-score-matching"></a>6. Energy-based models and score matching

**Topics:** Energy functions and the partition function; the maximum-likelihood gradient and its negative phase; Langevin dynamics and contrastive divergence; noise-contrastive estimation; the score function and the Fisher divergence; score matching, sliced score matching, and denoising score matching; Tweedie's formula; the failure of Langevin sampling in low-density regions; noise-conditional score networks and annealed Langevin dynamics.

- **CS236 lectures 11–12: Energy-based models.** [Slides 11 · PDF](https://deepgenerativemodels.github.io/assets/slides/cs236_lecture11.pdf) · [Video 11](https://www.youtube.com/watch?v=m61KiAMCJ5Q) · [Slides 12 · PDF](https://deepgenerativemodels.github.io/assets/slides/cs236_lecture12.pdf) · [Video 12](https://www.youtube.com/watch?v=Nci1Bepcy0g)
- **CS236 lectures 13–14: Score-based models (the syllabus lists lecture 14 as energy-based models).** [Slides 13 · PPTX](https://deepgenerativemodels.github.io/assets/slides/lecture%2013.pptx) · [Video 13](https://www.youtube.com/watch?v=8G-OsDs1RLI) · [Slides 14 · PPTX](https://deepgenerativemodels.github.io/assets/slides/lecture_14_comp.pptx) · [Video 14](https://www.youtube.com/watch?v=E69Lp_T9nVg)
- **6.S184 lecture 3, first part: Score functions and score matching.** [Slides · PDF](https://diffusion.csail.mit.edu/2026/docs/20260123_Lecture_03.pdf) · [Video](https://www.youtube.com/watch?v=ngC3QnYSVNM)
- **Yang Song, *Generative modeling by estimating gradients of the data distribution*.** [Blog post](https://yang-song.net/blog/2021/score/)
- **PML2 chapter 24: Energy-based models.**

**Further reading:** Song and Kingma, [How to train your energy-based models](https://arxiv.org/abs/2101.03288) (2021); Hyvärinen, [Estimation of non-normalized statistical models by score matching](https://jmlr.org/papers/v6/hyvarinen05a.html) (2005); Vincent, [A connection between score matching and denoising autoencoders](https://doi.org/10.1162/NECO_a_00142) (2011); Song and Ermon, [Generative modeling by estimating gradients of the data distribution](https://arxiv.org/abs/1907.05600) (2019).

## <a id="7-denoising-diffusion-models"></a>7. Denoising diffusion models

**Topics:** The forward noising process and its closed form; the reverse process as a hierarchical latent-variable model; the evidence lower bound and its decomposition; predicting the noise, the data, or the velocity; the simplified loss and its link to denoising score matching; noise schedules and signal-to-noise weighting; ancestral sampling; DDIM and deterministic sampling; U-Net denoisers with time conditioning.

- **CS236 lecture 16: Score-based diffusion models.** [Slides · PPTX](https://deepgenerativemodels.github.io/assets/slides/lecture16-2023-comp.pptx) · [Video](https://www.youtube.com/watch?v=VsllsC2JMGY)
- **6.S978 lecture 5: Energy-based models, score matching, and diffusion models.** [Slides · PDF](https://mit-6s978.github.io/assets/pdfs/lec5_diffusion.pdf)
- **CS294-158 lecture 6: Diffusion models.** [Video](https://www.youtube.com/watch?v=DsEDMjdxOv4)
- **CME 296 lecture 1: Diffusion.** [Video](https://www.youtube.com/watch?v=tr-CUpw--ck)
- **UDL chapter 18: Diffusion models; PML2 chapter 25.**

**Further reading:** Ho, Jain, and Abbeel, [Denoising diffusion probabilistic models](https://arxiv.org/abs/2006.11239) (2020); Song, Meng, and Ermon, [Denoising diffusion implicit models](https://arxiv.org/abs/2010.02502) (2021); Kingma et al., [Variational diffusion models](https://arxiv.org/abs/2107.00630) (2021); Luo, [Understanding diffusion models: a unified perspective](https://arxiv.org/abs/2208.11970) (2022); Lilian Weng, [What are diffusion models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/). CS294-158 homework 4 trains a diffusion model.

## <a id="8-diffusion-sdes-and-fast-sampling"></a>8. Diffusion SDEs and fast sampling

**Topics:** Stochastic differential equations and their Euler–Maruyama discretization; variance-preserving and variance-exploding processes; the Fokker–Planck equation; reverse-time SDEs; the probability-flow ODE and exact likelihoods; numerical solvers, Heun's method, and exponential integrators; the design space of noise levels, preconditioning, and time steps; distillation, consistency models, and few-step generation.

- **6.S184 lecture 1: Flow and diffusion models.** [Slides · PDF](https://diffusion.csail.mit.edu/2026/docs/20260120_Lecture_01.pdf) · [Video](https://www.youtube.com/watch?v=9eJQQVrUUoI)
- **6.S184 notes chapters 2 and 4: Flow and diffusion models; score functions and score matching.**
- **CME 296 lecture 2: Score matching.** [Video](https://www.youtube.com/watch?v=_WaR2fjZpEQ)
- **6.S978 guest lecture: Yang Song, consistency models.** [Slides · PDF](https://mit-6s978.github.io/assets/pdfs/CM_lecture.pdf)
- **PDM:** the chapters on the score SDE and on fast sampling.

**Further reading:** Song et al., [Score-based generative modeling through stochastic differential equations](https://arxiv.org/abs/2011.13456) (2021); Karras et al., [Elucidating the design space of diffusion-based generative models](https://arxiv.org/abs/2206.00364) (2022); Lu et al., [DPM-Solver](https://arxiv.org/abs/2206.00927) (2022); Song et al., [Consistency models](https://arxiv.org/abs/2303.01469) (2023).

## <a id="9-flow-matching"></a>9. Flow matching

**Topics:** Flows as solutions of ordinary differential equations; the continuity equation; probability paths and conditional paths; the conditional flow matching loss and why it has the same gradient as the marginal one; Gaussian paths and their relation to diffusion; rectified flows and straightening; stochastic interpolants; couplings from optimal transport; sampling with few steps.

- **6.S184 lecture 2: Flow matching.** [Slides · PDF](https://diffusion.csail.mit.edu/2026/docs/20260122_Lecture_02.pdf) · [Video](https://www.youtube.com/watch?v=PNkMKWW8Khw)
- **6.S184 notes chapter 3: Flow matching.**
- **CME 296 lecture 3: Flow matching.** [Video](https://www.youtube.com/watch?v=agN3AlfGFrk)
- **FMGC sections 1–4.**

**Further reading:** Lipman et al., [Flow matching for generative modeling](https://arxiv.org/abs/2210.02747) (2023); Liu, Gong, and Liu, [Flow straight and fast](https://arxiv.org/abs/2209.03003) (2023); Albergo and Vanden-Eijnden, [Building normalizing flows with stochastic interpolants](https://arxiv.org/abs/2209.15571) (2023); Tong et al., [Improving and generalizing flow-based generative models with minibatch optimal transport](https://arxiv.org/abs/2302.00482) (2024). The 6.S184 labs implement flow matching and score matching from scratch.

## <a id="10-guidance-and-conditional-generation"></a>10. Guidance and conditional generation

**Topics:** Conditional generative models; classifier guidance; classifier-free guidance and its geometry; the trade-off between fidelity and diversity; negative prompts; solving inverse problems with a diffusion prior: inpainting, super-resolution, and posterior sampling; editing by partial noising and inversion; adding control signals; fine-tuning for personalization; aligning generators with rewards and preferences.

- **6.S184 lecture 3, second part: Classifier-free guidance.** [Slides · PDF](https://diffusion.csail.mit.edu/2026/docs/20260123_Lecture_03.pdf) · [Video](https://www.youtube.com/watch?v=8oWZ1bHwyRI)
- **6.S184 notes chapter 5: Guidance.**
- **CME 296 lecture 4: Latent space and guidance.** [Video](https://www.youtube.com/watch?v=WUUq6TVAu8U)
- **Sander Dieleman, *Guidance: a cheat code for diffusion models* and *The geometry of diffusion guidance*.** [Post 1](https://sander.ai/2022/05/26/guidance.html) · [Post 2](https://sander.ai/2023/08/28/geometry.html)

**Further reading:** Dhariwal and Nichol, [Diffusion models beat GANs on image synthesis](https://arxiv.org/abs/2105.05233) (2021); Ho and Salimans, [Classifier-free diffusion guidance](https://arxiv.org/abs/2207.12598) (2022); Chung et al., [Diffusion posterior sampling for general noisy inverse problems](https://arxiv.org/abs/2209.14687) (2023); Zhang, Rao, and Agrawala, [Adding conditional control to text-to-image diffusion models](https://arxiv.org/abs/2302.05543) (2023).

## <a id="11-latent-diffusion-and-large-scale-generation"></a>11. Latent diffusion and large-scale generation

**Topics:** Why generate in a latent space; autoencoders for latent diffusion and their losses; text conditioning with frozen text encoders and cross-attention; U-Nets and diffusion transformers; rectified-flow transformers at scale; resolution, noise schedules, and data; video generation with spatiotemporal latents; the cost of training and sampling; distillation for deployment.

- **6.S184 lecture 4: Latent spaces and neural network architectures.** [Slides · PDF](https://diffusion.csail.mit.edu/2026/docs/20260128_Lecture_04_edited.pdf) · [Video](https://www.youtube.com/watch?v=g0MB1CCBmsI)
- **6.S184 notes chapter 6: Building large-scale image or video generators.**
- **CME 296 lectures 5–6: Architectures, model training.** [Video 5](https://www.youtube.com/watch?v=HpFdSlMeXzQ) · [Video 6](https://www.youtube.com/watch?v=IvXTl3yj-4Y)
- **CS294-158 lecture 9: Video generation.** [Video](https://www.youtube.com/watch?v=8ibaG_DPly8)

**Further reading:** Rombach et al., [High-resolution image synthesis with latent diffusion models](https://arxiv.org/abs/2112.10752) (2022); Peebles and Xie, [Scalable diffusion models with transformers](https://arxiv.org/abs/2212.09748) (2023); Esser et al., [Scaling rectified flow transformers for high-resolution image synthesis](https://arxiv.org/abs/2403.03206) (2024); Sander Dieleman, [Generative modelling in latent space](https://sander.ai/2025/04/15/latents.html).

## <a id="12-discrete-tokens-and-multimodal-generation"></a>12. Discrete tokens and multimodal generation

**Topics:** Discrete latent variables, the Gumbel-softmax relaxation, and the straight-through estimator; vector quantization, VQ-VAE, and VQGAN; residual and finite scalar quantization; neural audio codecs; autoregressive transformers over image and audio tokens; masked generative transformers; discrete diffusion, masked diffusion, and diffusion language models; vision–language models and unified models that generate several modalities.

- **CS236 lecture 17: Discrete latent variable models.** [Slides · PDF](https://deepgenerativemodels.github.io/assets/slides/cs236_lecture17.pdf) · [Video](https://www.youtube.com/watch?v=vBv7Mf1zsg8)
- **CS236 lecture 18: Diffusion models for discrete data (Aaron Lou).** [Slides · PDF](https://deepgenerativemodels.github.io/assets/slides/cs236_lecture18.pdf) · [Video](https://www.youtube.com/watch?v=mCaRNnEnYwA)
- **6.S184 lecture 5: Discrete diffusion models.** [Slides · PDF](https://diffusion.csail.mit.edu/2026/docs/20260130_Lecture_05.pdf) · [Video](https://www.youtube.com/watch?v=d0kmyEJN2hI)
- **CS294-158 lecture 12a: Multimodal models.** [Video](https://www.youtube.com/watch?v=A36T-MPihTo)

**Further reading:** van den Oord, Vinyals, and Kavukcuoglu, [Neural discrete representation learning](https://arxiv.org/abs/1711.00937) (2017); Esser, Rombach, and Ommer, [Taming transformers for high-resolution image synthesis](https://arxiv.org/abs/2012.09841) (2021); Chang et al., [MaskGIT](https://arxiv.org/abs/2202.04200) (2022); Sahoo et al., [Simple and effective masked diffusion language models](https://arxiv.org/abs/2406.07524) (2024); Sander Dieleman, [Diffusion language models](https://sander.ai/2023/01/09/diffusion-language.html).

## <a id="13-evaluating-generative-models"></a>13. Evaluating generative models

**Topics:** Likelihood and bits per dimension, and why good likelihoods and good samples can disagree; the Inception score; the Fréchet inception distance, its bias, and the choice of feature space; kernel inception distance; precision and recall for distributions; text–image alignment scores; human preference studies; memorization, data copying, and extracting training data.

- **CS236 lecture 15: Evaluation of generative models.** [Slides · PDF](https://deepgenerativemodels.github.io/assets/slides/lecture15.pdf) · [Video](https://www.youtube.com/watch?v=MJt_ahtO-to)
- **CME 296 lecture 7: Evaluation.** [Video](https://www.youtube.com/watch?v=iNaRBp4T57Q)

**Further reading:** Theis, van den Oord, and Bethge, [A note on the evaluation of generative models](https://arxiv.org/abs/1511.01844) (2016); Heusel et al., [GANs trained by a two time-scale update rule converge to a local Nash equilibrium](https://arxiv.org/abs/1706.08500) (2017), which introduced FID; Kynkäänniemi et al., [Improved precision and recall metric for assessing generative models](https://arxiv.org/abs/1904.06991) (2019); Stein et al., [Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models](https://arxiv.org/abs/2306.04675) (2023); Carlini et al., [Extracting training data from diffusion models](https://arxiv.org/abs/2301.13188) (2023).

## <a id="14-generative-models-for-science-and-control-optional"></a>14. Generative models for science and control — optional

**Topics:** Generative models as priors and samplers outside media: diffusion policies for robot control and planning; protein structure generation and design; molecules and docking; weather and climate ensembles; equivariance to rotations and translations; conditioning on physical constraints; how such models are validated.

- **6.S184 (IAP 2025) lecture 5: Generative robotics.** [Video](https://www.youtube.com/watch?v=7tsCN2hRBMg)
- **6.S184 (IAP 2025) lecture 6: Generative protein design.** [Slides · PDF](https://diffusion.csail.mit.edu/2025/docs/slides_lecture_6.pdf) · [Video](https://www.youtube.com/watch?v=hi_s4T1-gBY)
- **CS294-158 lecture 13a: AI for science (John Ingraham).** [Video](https://www.youtube.com/watch?v=4dyf_SALtxM)

**Further reading:** Chi et al., [Diffusion policy](https://arxiv.org/abs/2303.04137) (2023); Janner et al., [Planning with diffusion for flexible behavior synthesis](https://arxiv.org/abs/2205.09991) (2022); Watson et al., [De novo design of protein structure and function with RFdiffusion](https://doi.org/10.1038/s41586-023-06415-8) (2023); Price et al., [Probabilistic weather forecasting with machine learning](https://doi.org/10.1038/s41586-024-08252-9) (2025).

## <a id="15-the-theory-of-diffusion-models-optional"></a>15. The theory of diffusion models — optional

**Topics:** Convergence of diffusion sampling given an accurate score; the role of score error and discretization; why the optimal denoiser memorizes the training set and why trained networks generalize; inductive biases of convolutional denoisers; diffusion as spectral autoregression; Schrödinger bridges and connections to optimal transport.

- **PDM:** the chapters that relate the variational, score, and flow views and that analyze sampling.
- **UT Austin IFML, Fall 2025: Sanjay Shakkottai, *Diffusion models for generative AI*.** [Course page](https://www.ifml.institute/node/551) · [Lecture playlist](https://www.youtube.com/playlist?list=PL8lIiiIWuabLxhJreBRZwNW-d02dJwbMb) A mathematical course on Langevin dynamics, functional inequalities, and the analysis of diffusion samplers.
- **Sander Dieleman, *Diffusion is spectral autoregression*.** [Blog post](https://sander.ai/2024/09/02/spectral-autoregression.html)

**Further reading:** Chen et al., [Sampling is as easy as learning the score](https://arxiv.org/abs/2209.11215) (2023); Kadkhodaie et al., [Generalization in diffusion models arises from geometry-adaptive harmonic representations](https://arxiv.org/abs/2310.02557) (2024); Kamb and Ganguli, [An analytic theory of creativity in convolutional diffusion models](https://arxiv.org/abs/2412.20292) (2025); De Bortoli et al., [Diffusion Schrödinger bridge with applications to score-based generative modeling](https://arxiv.org/abs/2106.01357) (2021).

## <a id="exercises-and-local-references"></a>Exercises and local references

- The four [CS294-158 homeworks](https://github.com/rll/deepul) implement autoregressive models, VAEs, adversarial networks, and diffusion models on images, with starter code and tests.
- The three [6.S184 labs](https://github.com/eje24/iap-diffusion-labs/tree/2026) build flow and diffusion models from simulating SDEs through flow matching, score matching, and a conditional image generator with classifier-free guidance.
- Jakub Tomczak's [code for *Deep Generative Modeling*](https://github.com/jmtomczak/intro_dgm) gives short, self-contained notebooks for each family of models.
- Foundations supplies the KL divergence, entropy, and the change of variables; ML supplies mixtures, EM, and kernel density estimates; DL supplies autoencoders, U-Nets, vision transformers, and contrastive image–text learning; AI supplies variational inference and Markov chain Monte Carlo; NLP and LLMs supplies autoregressive transformers, tokenizers, sampling, and preference optimization.

## <a id="connections-to-later-modules"></a>Connections to later modules

Reinforcement learning uses generative models as policies and world models, and fine-tuning a diffusion model against a reward is a reinforcement-learning problem developed in RL. Watermarking, provenance, deepfakes, the misuse of generators, and the memorization of copyrighted or private data are taken up in Safety and Frontier. The families of generative models, their training objectives, samplers, guidance, architectures, and evaluation remain in this module.
