[Background Notes](../../README.md) › [Deep Learning](../README.md)

# Computing setup

# <a id="computing-setup"></a>Computing setup

The DL code blocks and figure scripts use PyTorch, NumPy, SciPy, Matplotlib, scikit-learn, and NetworkX, and everything runs on a CPU. The environment below is a **CPU reproducibility snapshot**, tested on Linux x86_64 with CPython 3.12.3. Every code block in the fifteen chapters was run in it with one and with two threads, and its printed output matched the comment lines at the end of the block. The version pins are not a claim that these are the newest releases, and a GPU is neither needed nor tested.

| Component | Tested version |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.2.2 (CPU execution) |
| NumPy | 1.26.4 |
| SciPy | 1.11.4 |
| Matplotlib | 3.8.4 |
| scikit-learn | 1.4.2 |
| NetworkX | 3.7 |

These are the versions of the Foundations environment (PyTorch, NumPy, SciPy, Matplotlib, and NetworkX, which PyTorch installs) plus the scikit-learn of the ML environment, so either of the following works.

## <a id="reuse-the-foundations-environment"></a>Reuse the Foundations environment

If the Foundations environment exists and scikit-learn has been added to it as described for ML, nothing more is needed: activate it from the `0. Foundations` folder and run the DL code from there. The Foundations snapshot was built on macOS x86_64, for which PyTorch 2.2.2 is the last release; the DL outputs were verified on Linux, and tiny differences in the last printed digit are possible on other platforms.

## <a id="or-create-a-separate-environment"></a>Or create a separate environment

Open a terminal in the `2. DL` folder and use Python 3.12. The environment stays inside this folder.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r Sources/Environment/requirements.txt
python -m pip check
```

[requirements.txt](environment/requirements.txt) records the resolved versions. On Linux, the PyTorch wheel from PyPI also downloads NVIDIA runtime libraries of about 2 GB; to avoid them, install PyTorch first from the CPU index, `python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu`, and then the requirements. Do not upgrade NumPy independently: PyTorch 2.2 and the pinned SciPy are built against NumPy 1.x.

## <a id="running-the-code"></a>Running the code

Each code block is self-contained: copy it into a file or a notebook cell and run it. Blocks whose training could depend on the order of floating-point additions call `torch.set_num_threads(1)`, so that their printed numbers do not depend on the number of cores; all blocks were checked to print the same output with one and with two threads. The blocks use only the digits and sample images bundled with scikit-learn, the karate-club graph bundled with NetworkX, and synthetic data, so no downloads are needed. The longest blocks take about 20 seconds.

The figures are regenerated from the `Sources/Figure code` folder, where the shared style module `dlfig.py` lives:

```bash
cd "Sources/Figure code"
python ch09_attention.py        # all figures of chapter 9
python ch09_attention.py 5      # only figure 5 (scripts for chapters 3 to 15 accept figure numbers)
```

The scripts write PNG files to `Sources/Images`. Most figures take seconds; those that train many networks take several minutes each, as noted at the top of each script and in Figure sources. The longest, the self-supervised comparison of chapter 10 and the spectral-bias experiment of chapter 15, take about ten minutes on one core.

## <a id="beyond-this-snapshot"></a>Beyond this snapshot

The examples are sized for a laptop CPU. The same code runs on a GPU after moving the model and the data to the device (`model.to("cuda")`, `x.to("cuda")`), and the chapters on scale (chapter 11) and methodology (chapter 12) describe what changes for real workloads: mixed precision, larger batches, data loaders with worker processes, and nondeterministic GPU kernels. Libraries used for larger work, such as torchvision, Hugging Face `transformers`, PyTorch Geometric, and experiment trackers, are not required by these notes.
