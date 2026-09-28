[Background Notes](../../README.md) › [NLP and Large Language Models](../README.md)

# Computing setup

> [!WARNING]
> Work in progress: this part of the notes is still being revised.

# <a id="computing-setup"></a>Computing setup

The NLP code blocks and figure scripts use PyTorch, NumPy, SciPy, Matplotlib, and scikit-learn, and everything runs on a CPU. Tokenizers, n-gram models, retrieval indexes, parsers, and the small transformers are implemented from scratch rather than taken from libraries such as Hugging Face `transformers`, so that the code shows the method. The environment below is a **CPU reproducibility snapshot**, tested on Linux x86_64 with CPython 3.12.3. Every code block in the sixteen chapters was run in it with one and with two threads, and its printed output matched the comment lines at the end of the block. The version pins are not a claim that these are the newest releases, and a GPU is neither needed nor tested.

| Component | Tested version |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.2.2 (CPU execution) |
| NumPy | 1.26.4 |
| SciPy | 1.11.4 |
| Matplotlib | 3.8.4 |
| scikit-learn | 1.4.2 |

This is the environment of the [DL module](../../dl/sources/computing-setup.md), so either of the following works.

## <a id="reuse-the-dl-or-foundations-environment"></a>Reuse the DL or Foundations environment

If the DL environment exists, or the Foundations environment with scikit-learn added as described for ML, nothing more is needed: activate it and run the NLP code from the `4. NLP&LLMs` folder. The Foundations snapshot was built on macOS x86_64, for which PyTorch 2.2.2 is the last release; the NLP outputs were verified on Linux, and tiny differences in the last printed digit are possible on other platforms.

## <a id="or-create-a-separate-environment"></a>Or create a separate environment

Open a terminal in the `4. NLP&LLMs` folder and use Python 3.12. The environment stays inside this folder.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r Sources/Environment/requirements.txt
python -m pip check
```

[requirements.txt](environment/requirements.txt) records the resolved versions. On Linux, the PyTorch wheel from PyPI also downloads NVIDIA runtime libraries of about 2 GB; to avoid them, install PyTorch first from the CPU index, `python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu`, and then the requirements. Do not upgrade NumPy independently: PyTorch 2.2 and the pinned SciPy are built against NumPy 1.x.

## <a id="running-the-code"></a>Running the code

Each code block is self-contained: copy it into a file or a notebook cell and run it **from the `4. NLP&LLMs` folder**, since blocks that read Tiny Shakespeare or the trained character model open them by the relative paths `Sources/Data/tinyshakespeare.txt` and `Sources/Data/shakespeare-char-transformer.pt` ([Data sources](data-sources.md)). Random numbers come from seeded generators, so the printed numbers are reproducible, and all blocks were checked to print the same output with one and with two threads. Most blocks finish in a few seconds. The longest, which train the small character transformer of chapter 4, takes about 50 seconds on one core; the tokenizer training of chapter 1, the skip-gram training of chapter 3, the mixture-of-experts load-balancing comparison of chapter 4, and the fine-tuning comparisons of chapter 10 take 20 to 30 seconds each.

The figures are regenerated from the `Sources/Figure code` folder, where the shared style module `nlpfig.py` lives:

```bash
cd "Sources/Figure code"
python ch04_transformer_lm.py           # the chapter 4 figure; trains and saves the model if the checkpoint is missing
python ch12_reasoning.py 2              # only figure 2 of chapter 12
```

The scripts write PNG files to `Sources/Images`. Chapter 4's script must run before those of chapters 5, 8, and 10 if the checkpoint is missing. Most figures take seconds to a few minutes; the longest are training the character transformer, about 20 minutes on two CPU cores, and the parity experiment of chapter 12, about ten minutes, as noted at the top of each script and in [Figure sources](figure-sources.md).

## <a id="beyond-this-snapshot"></a>Beyond this snapshot

The models here have at most about a million parameters and train in minutes on a laptop; the same code runs on a GPU after moving the model and the data to the device. Real work with language models uses libraries that the notes do not require: Hugging Face `transformers`, `tokenizers`, and `datasets` for pretrained models and data; `tiktoken` and SentencePiece for production tokenizers; vLLM and similar servers for efficient inference ([chapter 8](../08-decoding-and-text-generation.md#the-cost-of-generation)); PEFT for LoRA and other adapters ([chapter 10](../10-fine-tuning-and-parameter-efficient-adaptation.md)); TRL for preference tuning and reinforcement learning ([chapter 11](../11-learning-from-human-preferences.md)); FAISS for vector search ([chapter 13](../13-retrieval-tools-and-agents.md)); the Language Model Evaluation Harness for benchmarks ([chapter 14](../14-evaluating-language-models.md)); and SacreBLEU and Stanza for translation metrics and parsing ([chapter 15](../15-machine-translation-and-multilingual-models.md) and [chapter 16](../16-syntactic-parsing.md)). The [book and documentation links](book-and-documentation-links.md) point to their documentation.
