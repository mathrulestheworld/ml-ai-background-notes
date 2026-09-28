[Background Notes](../../README.md) › [Artificial Intelligence](../README.md)

# Computing setup

# <a id="computing-setup"></a>Computing setup

The AI code blocks and figure scripts use only NumPy, SciPy, Matplotlib, and NetworkX; every algorithm, from A\* and DPLL to variable elimination, particle filtering, and the PC algorithm, is implemented from scratch in plain Python so that the code shows the algorithm rather than a library call. Everything runs on a CPU. The environment below is a **CPU reproducibility snapshot**, tested on Linux x86_64 with CPython 3.12.3. Every code block in the fifteen chapters was run in it with one and with two threads, and its printed output matched the comment lines at the end of the block. The version pins are not a claim that these are the newest releases.

| Component | Tested version |
| --- | --- |
| Python | 3.12.3 |
| NumPy | 1.26.4 |
| SciPy | 1.11.4 |
| Matplotlib | 3.8.4 |
| NetworkX | 3.7 |

These packages are all part of the Foundations environment, so either of the following works.

## <a id="reuse-the-foundations-environment"></a>Reuse the Foundations environment

If the Foundations environment exists, nothing more is needed: activate it from the `0. Foundations` folder and run the AI code from there. The Foundations snapshot was built on macOS; the AI outputs were verified on Linux, and differences in the last printed digit are possible on other platforms.

## <a id="or-create-a-separate-environment"></a>Or create a separate environment

Open a terminal in the `3. AI` folder and use Python 3.12. The environment stays inside this folder.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r Sources/Environment/requirements.txt
python -m pip check
```

[requirements.txt](environment/requirements.txt) records the resolved versions. Do not upgrade NumPy independently of SciPy: the pinned SciPy is built against NumPy 1.x.

## <a id="running-the-code"></a>Running the code

Each code block is self-contained: copy it into a file or a notebook cell and run it. Random numbers come from seeded NumPy generators or `random.Random` instances, so the printed numbers are reproducible, and the code avoids iterating over sets of strings where Python's hash randomization would change the order of results. The blocks use no datasets and no downloads: the problems are puzzles, games, small networks, and synthetic data generated in the block. Most blocks finish in under a second; the longest, the eight-queens local-search comparison of chapter 3, takes about 15 seconds.

The figures are regenerated from the `Sources/Figure code` folder, where the shared style module `aifig.py` lives:

```bash
cd "Sources/Figure code"
python ch11_temporal.py        # all figures of chapter 11
python ch11_temporal.py 2      # only figure 2 (every script accepts figure numbers)
```

The scripts write PNG files to `Sources/Images`. Most figures take seconds; those that run many searches take minutes, as noted at the top of each script and in Figure sources. The longest, the blocks-world comparison of chapter 7, takes about ten minutes on one core, because its planning heuristics are recomputed in pure Python for every state.

## <a id="beyond-this-snapshot"></a>Beyond this snapshot

Pure Python is slow for search and inference: the implementations here expand thousands of nodes per second where compiled solvers expand millions. The tools for larger problems list mature solvers, planners, inference libraries, and probabilistic programming languages to use once the algorithms are understood.
