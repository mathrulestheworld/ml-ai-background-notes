[Background Notes](../../README.md) › [Generative AI](../README.md)

# Course links

# <a id="course-links"></a>Course links

| Course | Links | Role |
| --- | --- | --- |
| Stanford CS236, Fall 2023 — Stefano Ermon, *Deep Generative Models* | [Course](https://deepgenerativemodels.github.io/) · [Syllabus with slides](https://deepgenerativemodels.github.io/syllabus.html) · [Lecture playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rPOWA-omMM6STXaWW4FvJT8) · [Course notes](https://deepgenerativemodels.github.io/notes/) | Main course: the families of generative models from maximum likelihood to diffusion, in the order of chapters 1–7, with lectures on evaluation and on discrete latent variables and discrete diffusion (chapters 12–13). Its written notes cover the first half of the course. |
| MIT 6.S184, IAP 2026 — Peter Holderrieth, *Introduction to Flow Matching and Diffusion Models* | [Course](https://diffusion.csail.mit.edu/2026/index.html) · [Lecture notes · PDF](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Lecture playlist](https://www.youtube.com/playlist?list=PL57nT7tSGAAXwjhDYcxEycx5W7YoSrZyt) · [Labs](https://github.com/eje24/iap-diffusion-labs/tree/2026) · [IAP 2025 edition](https://diffusion.csail.mit.edu/2025/index.html) | Main course: diffusion and flow models through differential equations, score matching, flow matching, guidance, latent spaces and architectures, and discrete diffusion (chapters 6, 8–12). Its notes are the best single text for chapters 8–12; the 2025 edition adds guest lectures on robotics and protein design (chapter 14). Three labs build a conditional image generator step by step. |
| MIT 6.S978, Fall 2024 — Kaiming He, *Deep Generative Models* | [Course page with slides and readings](https://mit-6s978.github.io/) | Supplement: lectures on VAEs, autoregressive models, adversarial networks, and diffusion, a guest lecture on consistency models, and reading sessions on tokenizers, flow matching, and discrete diffusion (chapters 1–3, 5, 7–8, 12). No recordings. |
| UC Berkeley CS294-158, Spring 2024 — Pieter Abbeel, Wilson Yan, Kevin Frans, and Philipp Wu, *Deep Unsupervised Learning* | [Course page with slides](https://sites.google.com/view/berkeley-cs294-158-sp24/home) · [Lecture playlist](https://www.youtube.com/playlist?list=PLwRJQ4m4UJjPIvv4kgBkvu_uygrV3ut_U) · [Homework](https://github.com/rll/deepul) | Supplement: recorded lectures on each family of models and on video generation, multimodal models, and science (chapters 2–5, 7, 11–12, 14). Its four homeworks implement autoregressive, latent-variable, adversarial, and diffusion models on images. |
| Stanford CME 296, Spring 2026 — Afshine and Shervine Amidi, *Diffusion and Large Vision Models* | [Course](https://cme296.stanford.edu/) · [Lecture playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rNdy8rt2rZ4T2xM0OjADnfu) | Supplement: a practical course on current image and video generators, from diffusion, score matching, and flow matching to latent spaces, guidance, architectures, training, and evaluation (chapters 7–11, 13). |
| UT Austin IFML, Fall 2025 — Sanjay Shakkottai, *Diffusion Models for Generative AI* | [Course page](https://www.ifml.institute/node/551) · [Lecture playlist](https://www.youtube.com/playlist?list=PL8lIiiIWuabLxhJreBRZwNW-d02dJwbMb) | Optional: a mathematical course on Langevin dynamics, functional inequalities, and the analysis of diffusion samplers (chapter 15). |

Code resources complement the courses:

| Resource | Link | Role |
| --- | --- | --- |
| CS294-158 homeworks | [Repository](https://github.com/rll/deepul) | Four homeworks with starter code and tests: autoregressive models, VAEs and a VQ-VAE with a transformer prior, adversarial networks, and diffusion models on images (chapters 2, 3, 5, 7, and 12). |
| 6.S184 labs | [Repository](https://github.com/eje24/iap-diffusion-labs/tree/2026) | Simulating SDEs, flow matching, score matching, and a conditional image generator with a diffusion transformer and classifier-free guidance, with solutions (chapters 8–11). |
| Jakub Tomczak, code for *Deep Generative Modeling* | [Repository](https://github.com/jmtomczak/intro_dgm) | Short, self-contained notebooks for each family of models (chapters 2–7). |
| Flow Matching Guide and Code | [Paper](https://arxiv.org/abs/2412.06264) · [Library](https://github.com/facebookresearch/flow_matching) | A PyTorch library for continuous, discrete, and Riemannian flow matching with worked examples (chapters 9 and 12). |

The reading plan maps individual lectures, slides, and readings to the study sequence. Video links arranges the recordings by chapter, and the Generative AI overview maps the reading-plan topics to the chapters.
