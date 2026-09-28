[ML Mastery Notes](../../README.md) › [Generative AI](../README.md)

# Computing setup

# <a id="computing-setup"></a>Computing setup

The Generative AI code blocks and figure scripts use PyTorch, NumPy, SciPy, Matplotlib, and scikit-learn, and everything runs on a CPU. Every model, from MADE and the VAE to the DDPM, the flow-matching networks, the distilled students, and the masked diffusion model, is implemented from scratch in a few dozen lines rather than taken from a library such as Diffusers, so that the code shows the method. The environment below is a **CPU reproducibility snapshot**, tested on Linux x86_64 with CPython 3.12.3. Every code block in the fifteen chapters was run in it with one and with two threads, and its printed output matched the comment lines at the end of the block. The version pins are not a claim that these are the newest releases, and a GPU is neither needed nor tested.

| Component | Tested version |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.2.2 (CPU execution) |
| NumPy | 1.26.4 |
| SciPy | 1.11.4 |
| Matplotlib | 3.8.4 |
| scikit-learn | 1.4.2 |

This is the environment of the DL and NLP and LLMs modules, so either of the following works.

## <a id="reuse-an-existing-environment"></a>Reuse an existing environment

If the DL or NLP environment exists, or the Foundations environment with scikit-learn added as described for ML, nothing more is needed: activate it and run the Generative AI code from the `5. Generative AI` folder. The Foundations snapshot was built on macOS x86_64, for which PyTorch 2.2.2 is the last release; the outputs here were verified on Linux, and tiny differences in the last printed digit are possible on other platforms.

## <a id="or-create-a-separate-environment"></a>Or create a separate environment

Open a terminal in the `5. Generative AI` folder and use Python 3.12. The environment stays inside this folder.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r Sources/Environment/requirements.txt
python -m pip check
```

[requirements.txt](environment/requirements.txt) records the resolved versions. On Linux, the PyTorch wheel from PyPI also downloads NVIDIA runtime libraries of about 2 GB; to avoid them, install PyTorch first from the CPU index, `python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu`, and then the requirements. Do not upgrade NumPy independently: PyTorch 2.2 and the pinned SciPy are built against NumPy 1.x.

## <a id="running-the-code"></a>Running the code

Each code block is self-contained: copy it into a file or a notebook cell and run it. No block reads files; the digits come with scikit-learn and every other distribution is generated in the code (Data sources). Random numbers come from seeded generators, and blocks that train PyTorch models call `torch.set_num_threads(1)` where their results would otherwise depend slightly on the number of threads, so the printed numbers are reproducible; all blocks were checked to print the same output with one and with two threads. Most blocks finish in a few seconds. The longest train small networks: the MADE of chapter 2, the VAE of chapter 3, the two-moons DDPM of chapter 7, the two flow-matching models of chapter 9, the latent diffusion model of chapter 11, and the masked diffusion model of chapter 12 each take 20 to 45 seconds on one core.

The figures are regenerated from the `Sources/Figure code` folder, where the shared style module `genfig.py` lives:

```bash
cd "Sources/Figure code"
python ch07_diffusion.py                # both figures of chapter 7
python ch10_guidance.py                 # both figures of chapter 10
```

The scripts write PNG files to `Sources/Images` and read no arguments. Most take seconds to two minutes; the longest train models on the digits, about three minutes for the VAEs of chapter 3, four for the DDPM of chapter 7, and six for the guided model of chapter 10, as noted at the top of each script and in Figure sources.

## <a id="beyond-this-snapshot"></a>Beyond this snapshot

The models here have at most about a million parameters, work on $`8\times8`$ images or points in the plane, and train in minutes on a laptop; the same code runs on a GPU after moving the model and the data to the device, and scales to images by replacing the multilayer perceptrons with U-Nets or transformers (chapter 7, chapter 11). Real work with generative models uses libraries that the notes do not require: Hugging Face Diffusers for pretrained pipelines, schedulers, guidance, ControlNet, and LoRA fine-tuning (chapter 10, chapter 11); k-diffusion and the EDM2 code for the samplers and training recipes of Karras et al. (chapter 8); the Flow Matching library and TorchCFM (chapter 9); AudioCraft for neural audio codecs and music generation (chapter 12); clean-fid and dgm-eval for evaluation (chapter 13); and LeRobot, RFdiffusion, and the WeatherNext repository for control, proteins, and weather (chapter 14). The book and documentation links point to them.
