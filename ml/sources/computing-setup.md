[Background Notes](../../README.md) › [Machine Learning](../README.md)

# Computing setup

# <a id="computing-setup"></a>Computing setup

The ML code blocks and figure scripts use Python, NumPy, SciPy, and scikit-learn; the environment also pins Matplotlib, and PyTorch is not needed. The figures themselves are drawn with Node.js, as described [below](#regenerating-the-figures). The environment below is a **CPU reproducibility snapshot**, tested on Linux x86_64 with CPython 3.12.3. Every code block in the seventeen chapters was run in it with one and with two BLAS threads, and its printed output matched the comment lines at the end of the block. The version pins are not a claim that these are the newest releases.

| Component | Tested version |
| --- | --- |
| Python | 3.12.3 |
| NumPy | 1.26.4 |
| SciPy | 1.11.4 |
| Matplotlib | 3.8.4 |
| scikit-learn | 1.4.2 |
| joblib / threadpoolctl | 1.6.0 / 3.7.0 |

NumPy, SciPy, and Matplotlib have the same versions as in the [Foundations environment](../../foundations/sources/computing-setup.md), so either of the following works.

## <a id="add-scikit-learn-to-the-foundations-environment"></a>Add scikit-learn to the Foundations environment

If the Foundations environment already exists, activate it from the `0. Foundations` folder and add scikit-learn at the tested versions:

```bash
source .venv/bin/activate
python -m pip install scikit-learn==1.4.2 joblib==1.6.0 threadpoolctl==3.7.0 cloudpickle==3.1.2
python -m pip check
```

These packages do not change the NumPy or SciPy installation that the Foundations PyTorch build depends on.

## <a id="or-create-a-separate-environment"></a>Or create a separate environment

Open a terminal in the `1. ML` folder and use Python 3.12. The environment stays inside this folder.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r Sources/Environment/requirements.txt
python -m pip check
```

[requirements.txt](environment/requirements.txt) records the resolved package versions, including dependencies. Do not upgrade NumPy independently: the pinned SciPy and scikit-learn wheels are built against NumPy 1.x.

## <a id="running-the-examples"></a>Running the examples

Each code block in a chapter is self-contained. Copy it into a Python session or a notebook cell and run it; the final comment lines show the output to expect. All blocks use synthetic data or datasets bundled with scikit-learn, so they run offline.

Random seeds are fixed, so the results are repeatable within this environment. Other library versions can change printed values in the last digits, and occasionally more, for example when two candidate splits of a tree tie or when a solver's default tolerance changes. A mismatch in the last printed digit is not an error in the notes.

## <a id="regenerating-the-figures"></a>Regenerating the figures

All figures are drawn with D3 and KaTeX, using the drawing library and packages of the Foundations figures. They need Node.js 22 or later. Install the Foundations figure packages once, as described in [the Foundations computing setup](../../foundations/sources/computing-setup.md#regenerate-the-added-figures).

Each chapter has a script in `Sources/Figure code`, from `ch01_nearest_neighbors.py` to `ch17_smoothing.py`. With the Python environment active, a script writes the data of the chapter's figures that show fitted models (scikit-learn estimators, EM runs, Gaussian-process fits, and the like) to `js/data/<name>.js`, and prints every number that the chapter and its captions quote from its figures:

```bash
cd "Sources/Figure code"         # from the 1. ML folder
python ch08_svm_kernels.py        # chapter 8: writes js/data/*.js and prints the quoted numbers
```

Chapter 1's script writes no data files, because all of its figures compute their data in JavaScript. Most scripts finish in under half a minute; those of chapters 5, 6, 10, 11, and 16 take up to about a minute. Rerunning a script in the tested environment reproduces its data files, apart from differences in the last digits of a few values that come from floating-point roundoff. The data files are kept in the vault, so the figures can be rebuilt without running the scripts first. To build and render them:

```bash
cd "Sources/Figure code/js"      # from the 1. ML folder
sh build.sh
node render.js                    # all figures; or name some: node render.js svm-rbf-grid svm-cv-heatmap
```

`build.sh` wraps each `src/<name>.js`, preceded by `data/<name>.js` when that file exists, into `build/<name>.html`. `render.js` renders each page at twice its CSS size into `Sources/Images`, reporting any overlapping or clipped labels and any label crossed by a drawn line. [Figure sources](figure-sources.md) records the data and construction of each figure. The module `mlfig.py` belonged to the earlier Matplotlib versions of the figures and is no longer used.
