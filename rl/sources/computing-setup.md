[Background Notes](../../README.md) › [Reinforcement Learning](../README.md)

# Computing setup

# <a id="computing-setup"></a>Computing setup

The RL code blocks, figure scripts, and labs use NumPy, SciPy, Matplotlib, PyTorch, and Gymnasium, and everything runs on a CPU. Every agent, from value iteration and Q-learning to DQN, PPO, SAC, AlphaZero, CFR, and the GRPO of the language-model lab, is implemented from scratch rather than taken from a library such as Stable-Baselines3 or CleanRL, so that the code shows the method; Gymnasium supplies only environments, and several environments are written in NumPy in the code itself. The environment below is a **CPU reproducibility snapshot**, tested on Linux x86_64 with CPython 3.12.3. Every code block in the 31 chapters was run in it and its printed output matches the comment lines at the end of the block, and every lab script printed the results quoted on its page. The version pins are not a claim that these are the newest releases, and a GPU is neither needed nor tested.

| Component | Tested version |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.2.2 (CPU execution) |
| NumPy | 1.26.4 |
| SciPy | 1.11.4 |
| Matplotlib | 3.8.4 |
| scikit-learn | 1.4.2 (one figure script) |
| Gymnasium | 1.2.2, with the Box2D and MuJoCo environments |
| MuJoCo | 3.3.7 |
| box2d-py | 2.3.5 |
| MinAtar | 1.0.15 |

This is the environment of the DL, NLP and LLMs, and Generative AI modules with four additions: Gymnasium, MuJoCo, Box2D, and MinAtar.

## <a id="extend-an-existing-environment"></a>Extend an existing environment

If the DL, NLP, or Generative AI environment exists, activate it and add the RL packages:

```bash
python -m pip install swig==4.5.0
python -m pip install "gymnasium[box2d,mujoco]==1.2.2" MinAtar==1.0.15
python -m pip check
```

`swig` must be installed first, because `box2d-py` compiles against it; only Lab 10, with `LunarLander-v3`, uses Box2D, so the lab can be skipped if the build fails. MuJoCo is used by Labs 10 and 11 (`HalfCheetah-v5`) and MinAtar by Lab 8. The chapters' code blocks need only NumPy, SciPy, PyTorch, and Gymnasium's basic environments.

## <a id="or-create-a-separate-environment"></a>Or create a separate environment

Open a terminal in the `6. RL` folder and use Python 3.12. The environment stays inside this folder.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install swig==4.5.0
python -m pip install -r Sources/Environment/requirements.txt
python -m pip check
```

[requirements.txt](environment/requirements.txt) records the resolved versions. On Linux, the PyTorch wheel from PyPI also downloads NVIDIA runtime libraries of about 2 GB; to avoid them, install PyTorch first from the CPU index, `python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu`, and then the requirements. Do not upgrade NumPy independently: PyTorch 2.2 and the pinned SciPy are built against NumPy 1.x.

## <a id="running-the-code"></a>Running the code

Each code block is self-contained: copy it into a file or a notebook cell and run it. No block reads files; environments are either Gymnasium's (`FrozenLake`, `Taxi`, `Blackjack`, `MountainCar`, and `CartPole`) or written in the block, from gridworlds, chains, and bandits to Kuhn poker, RiverSwim, and the bit-flipping task of chapter 31. Random numbers come from seeded generators, and blocks that train PyTorch models call `torch.set_num_threads(1)` so that the printed numbers do not depend on the number of threads. Most blocks finish in a few seconds. The longest take one to two and a half minutes on one core: the DQN runs on CartPole in chapter 16, the Deep Sea experiment of chapter 22, POMCP in an exercise of chapter 14, and the hindsight experience replay of chapter 31.

The labs are larger. Each lab page links its script in `Labs/code`, which prints the lab's expected results; run it from that folder:

```bash
cd Labs/code
python lab03_tabular_control.py      # about four minutes on one core
python lab17_safe_robust.py          # about 16 minutes on two cores
```

Labs 1–7, 9, and 15 take between half a minute and four minutes on one core. The deep RL labs use two worker processes and take 15 to 45 minutes: Lab 8 (DQN on CartPole and MinAtar) about 25 minutes, Lab 10 (PPO on LunarLander and HalfCheetah) about 45, Lab 11 (DDPG, TD3, and SAC on HalfCheetah) about 40, Lab 12 (PETS and MBPO) about 15, Lab 13 (AlphaZero) about 17, Lab 14 (imitation and offline RL) about 25, Lab 16 (RL for a small language model) about 25, and Lab 17 (safe and robust RL) about 16. Each script states its time at the top. Results of the deep RL labs can differ slightly across platforms and library builds, since small numerical differences are amplified over training; the qualitative conclusions on each page should hold.

The figures are regenerated from the `Sources/Figure code` folder, where the shared style module `rlfig.py` lives:

```bash
cd "Sources/Figure code"
python ch16_dqn.py                   # both figures of chapter 16
python ch31_realworld.py             # the figure of chapter 31
```

The scripts write PNG files to `Sources/Images` and read no arguments. Most take seconds; the longest train agents and take one to a few minutes, as noted at the top of each script and in Figure sources.

## <a id="beyond-this-snapshot"></a>Beyond this snapshot

The code here trains agents with at most a few hundred thousand parameters on environments that simulate in microseconds, so that every experiment runs on a laptop. Research-scale RL separates acting from learning and simulates thousands of environments in parallel, often on accelerators. The tools the notes point to, without requiring them, are: [CleanRL](https://docs.cleanrl.dev/) and Stable-Baselines3 for tested single-agent implementations (chapter 21); EnvPool, Brax, and Isaac Gym for batched simulation (chapter 19, chapter 31); the Arcade Learning Environment and the DeepMind Control Suite for standard benchmarks (chapter 16, chapter 21); `mctx` for batched tree search (chapter 24); D4RL, Minari, and OGBench for offline data (chapter 26); OpenSpiel and PettingZoo for games and multi-agent RL (chapter 27); and TRL and verl for reinforcement learning of language models (chapter 28). The book and documentation links point to them.
