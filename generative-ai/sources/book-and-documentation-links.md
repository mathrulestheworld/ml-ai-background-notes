[ML Mastery Notes](../../README.md) › [Generative AI](../README.md)

# Book and documentation links

# <a id="books-and-documentation"></a>Books and documentation

The chapter notes develop the module's main material. These books, tutorials, and documentation pages provide alternative explanations, fuller derivations, and the tools used to build real generative models. The reading plan lists the lecture slides and readings for each topic.

## <a id="main-references"></a>Main references

| Source | Relevant material |
| --- | --- |
| Murphy, [*Probabilistic Machine Learning: Advanced Topics*](https://probml.github.io/pml-book/book2.html) (MIT Press, 2023) | Free draft; part IV treats every family with consistent notation: chapter 20, an overview (chapter 1), 21 on variational autoencoders (chapter 3), 22 on autoregressive models (chapter 2), 23 on normalizing flows (chapter 4), 24 on energy-based models (chapter 6), 25 on diffusion models (chapters 7–8), and 26 on adversarial networks (chapter 5). |
| Prince, [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/) (MIT Press, 2023) | Free, with notebooks: chapter 14 on unsupervised learning (chapter 1), 15 on adversarial networks (chapter 5), 16 on normalizing flows (chapter 4), 17 on variational autoencoders (chapter 3), and 18 on diffusion models (chapter 7). |
| Holderrieth and Erives, [*An Introduction to Flow Matching and Diffusion Models*](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) (2026), the notes of MIT 6.S184 | Free; flow and diffusion models as differential equations, score matching, flow matching, guidance, architectures, and discrete diffusion (chapters 6 and 8–12), with the conversion formulas between velocities, scores, and denoisers used in chapter 9. |
| Lai, Song, Kim, Mitsufuji, and Ermon, [*The Principles of Diffusion Models*](https://arxiv.org/abs/2510.21890) (2025) | Free monograph; the variational, score-based, and flow-based views of diffusion and their equivalence, guidance, fast solvers, and distillation (chapters 6–10 and 15). |
| Lipman et al., [*Flow Matching Guide and Code*](https://arxiv.org/abs/2412.06264) (2024) | Free; flow matching in continuous, discrete, and Riemannian settings, with a PyTorch library (chapters 9 and 12). |
| Tomczak, [*Deep Generative Modeling*](https://link.springer.com/book/10.1007/978-3-031-64087-2) (Springer, second edition, 2024), with [code](https://github.com/jmtomczak/intro_dgm) | A short book with a notebook for each family of models (chapters 2–7 and 12). |

## <a id="tutorials-and-surveys"></a>Tutorials and surveys

| Source | Relevant material |
| --- | --- |
| Luo, [*Understanding Diffusion Models: A Unified Perspective*](https://arxiv.org/abs/2208.11970) (2022) | A step-by-step derivation of diffusion as a hierarchical VAE, its variational bound, and its equivalence to score matching (chapters 3, 6, and 7). |
| Lilian Weng, [*What are Diffusion Models?*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) (2021, updated) | A compact survey of DDPM, score models, DDIM, guidance, and latent diffusion with consistent notation (chapters 7, 8, 10, and 11). |
| Yang Song, [*Generative Modeling by Estimating Gradients of the Data Distribution*](https://yang-song.net/blog/2021/score/) (2021) | Score matching, annealed Langevin dynamics, and the SDE view, by one of their authors (chapters 6 and 8). |
| Sander Dieleman, [blog](https://sander.ai/) | Essays on guidance and its geometry, the perspectives on diffusion, latent spaces for generation, and diffusion as spectral autoregression (chapters 10, 11, and 15). |
| Gao, Hoogeboom, Heek, De Bortoli, Murphy, and Salimans, [*Diffusion Meets Flow Matching: Two Sides of the Same Coin*](https://diffusionflow.github.io/) (2024) | An interactive explanation of why Gaussian flow matching and diffusion coincide, including their samplers and loss weightings (chapter 9). |
| Karras, Aittala, Aila, and Laine, [*Elucidating the Design Space of Diffusion-Based Generative Models*](https://arxiv.org/abs/2206.00364) (2022) | Read as a tutorial: the separation of noise schedule, scaling, preconditioning, loss weighting, and sampler that chapter 8 follows. |

## <a id="libraries-and-tools"></a>Libraries and tools

| Source | Relevant material |
| --- | --- |
| Hugging Face [Diffusers](https://huggingface.co/docs/diffusers/index) | Pretrained pipelines for text-to-image, image-to-image, and video models, interchangeable schedulers (DDPM, DDIM, DPM-Solver++, Euler, Heun), guidance, ControlNet, LoRA fine-tuning, and training scripts (chapters 7–11). |
| [k-diffusion](https://github.com/crowsonkb/k-diffusion) | Reference implementations of the samplers and preconditioning of Karras et al. (chapter 8). |
| [EDM2 code](https://github.com/NVlabs/edm2) | The training and sampling code of Karras et al. (2024), with post-hoc weight averaging and autoguidance (chapters 10 and 11). |
| [Flow Matching library](https://github.com/facebookresearch/flow_matching) and [TorchCFM](https://github.com/atong01/conditional-flow-matching) | Flow matching with Gaussian, optimal-transport, and discrete paths, and minibatch optimal-transport couplings (chapter 9). |
| [clean-fid](https://github.com/GaParmar/clean-fid) and [dgm-eval](https://github.com/layer6ai-labs/dgm-eval) | FID and KID computed with consistent resizing, and the metrics and feature spaces compared by Stein et al. (chapter 13). |
| [AudioCraft](https://github.com/facebookresearch/audiocraft) | EnCodec and MusicGen, neural audio codecs and a transformer over their tokens (chapter 12). |
| [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) and [LeRobot](https://github.com/huggingface/lerobot) | Training and evaluating diffusion policies and other imitation-learning policies for robots (chapter 14). |
| [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) and [WeatherNext](https://github.com/google-deepmind/weathernext) | Protein design with RFdiffusion, and the weather models of Google DeepMind, whose repository includes GraphCast and GenCast (chapter 14). |
