[ML Mastery Notes](../../README.md) › [Foundations](../README.md)

# Computing setup

# <a id="computing-setup"></a>Computing setup

The foundations notebooks use Python, NumPy, SciPy, Matplotlib, and PyTorch. The environment below is a **CPU reproducibility snapshot**, tested on macOS x86_64 with CPython 3.12.1. It preserves the course's numerical examples; the version pins are not a claim that these are the newest releases. GPU execution and other operating systems were not part of this validation.

| Component | Tested version |
| --- | --- |
| Python | 3.12.1 |
| NumPy | 1.26.4 |
| SciPy | 1.11.4 |
| Matplotlib | 3.8.4 |
| PyTorch | 2.2.2 |
| JupyterLab | 4.1.6 |
| IPython kernel | 6.27.1 |
| nbclient / nbformat | 0.9.0 / 5.9.2 |

## <a id="create-the-environment"></a>Create the environment

Open a terminal in the `0. Foundations` folder. Use Python 3.12 for these commands; the exact tested interpreter was 3.12.1. The environment stays inside this folder, so installing its packages does not change your global Python installation.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r Sources/Environment/requirements.txt
python -m pip check
python -m ipykernel install --prefix "$VIRTUAL_ENV" --name foundations --display-name "Python (Foundations CPU)"
python -m jupyterlab
```

[requirements.txt](environment/requirements.txt) records the resolved package versions, including dependencies. Python itself is selected when you create the environment. Do not upgrade NumPy independently inside this snapshot: the tested PyTorch and SciPy versions require a compatible NumPy installation for their native extensions.

In JupyterLab, open [NumPy and PyTorch examples](notebooks/NumPy-and-PyTorch-examples.ipynb), select **Python (Foundations CPU)**, and run all cells from the beginning. The first code cell reports the environment and checks NumPy/PyTorch shared storage. It sets one CPU thread for this small lab. The remaining cells check their results with assertions. A seed supports repeatability within this environment; it does not promise identical results on different library versions or hardware.

The accompanying [exercise sources](companion-sources.md) describe the portable exercise notes and notebooks. Use the same kernel for those labs. If you move the vault or the virtual environment to another location, recreate the environment and register its kernel again because a kernel records its Python executable's location.

## <a id="keep-the-notebook-synchronized"></a>Keep the notebook synchronized

The chapter markdown is the source for the companion notebook's explanations. After editing the chapter, regenerate the notebook from the same `0. Foundations` folder:

```bash
python Sources/Environment/build_notebook.py
```

The converter preserves literal inline and fenced code when replacing Obsidian wiki links, rebases links for the notebook's directory, shows the chapter's figures from `Sources/Images`, points appendix links at the notebook's appendix headings, and retains saved outputs only when a code cell's source is unchanged. It also adds the environment check. Run the notebook again after changing executable code; retained outputs are a convenience, not fresh verification. The chapter's Obsidian styling metadata is omitted from the notebook.

## <a id="regenerate-the-added-figures"></a>Regenerate the added figures

The code in `Sources/Figure code` regenerates the figures added to chapters 1 to 6. The chapter 1 script also prints the numbers that chapter quotes from its figures, and it needs scikit-learn; add it to this environment as described in the ML computing setup. Then, from the `0. Foundations` folder:

```bash
python "Sources/Figure code/ch01_terminology.py"
```

The drawn figures of chapters 2 to 6 are made in JavaScript and need Node.js 22 or later. From the same folder, install the pinned packages once and then render; Playwright supplies the headless Chromium:

```bash
cd "Sources/Figure code/js"
npm install
npx playwright install chromium
npm run render
```

`npm install` creates a `node_modules` folder that contains Markdown files of its own; add `node_modules` to Obsidian's excluded files (Settings → Files and links → Excluded files) to keep them out of search and the graph. From the `0. Foundations` folder, `python "Sources/Figure code/ch03_capture_distill.py"` and `python "Sources/Figure code/ch03_crop_baydin.py"` refresh the imported Distill and survey figures of chapter 3; they need Python Playwright with Chromium, Pillow, `curl`, Poppler's `pdftoppm`, and network access. Inside the Foundations environment, `python "Sources/Figure code/ch06_training_curve.py"` reruns the chapter 6 training example and prints the data embedded in its training-curve figure.

Each figure script writes its image files into `Sources/Images`, replacing the stored versions; the chapter 6 script only prints data. The chapter 1 script was tested with Python 3.12.3, NumPy 1.26.4, SciPy 1.11.4, Matplotlib 3.8.4, and scikit-learn 1.4.2. Figure sources lists what each script produces and the versions used for the drawn figures.
