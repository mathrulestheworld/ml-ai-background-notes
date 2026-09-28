[Background Notes](../../README.md) › [NLP and Large Language Models](../README.md)

# Data sources

# <a id="data-sources"></a>Data sources

The NLP code blocks and figure scripts use one text corpus and one trained model, both stored in `Sources/Data`. Everything else, from the toy English–Spanish corpus of chapter 15 to the simulated benchmarks of chapter 14, is generated inside the code that uses it, so nothing needs to be downloaded.

| File | Size | Used in |
| --- | --- | --- |
| `tinyshakespeare.txt` | 1,115,394 bytes | Code and figures of chapters 1–6, 8, 10, and 13 |
| `shakespeare-char-transformer.pt` | 3.4 MB | Code and figures of chapters 4, 5, 8, and 10 |

## <a id="tiny-shakespeare"></a>Tiny Shakespeare

**Tiny Shakespeare** is a plain-text file of about 40,000 lines of dialogue from Shakespeare's plays, with each speech preceded by the speaker's name in capitals. It was assembled by Andrej Karpathy for his character-level recurrent network experiments and is distributed with the [char-rnn repository](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare) as `input.txt`; the copy here is byte-for-byte that file. Shakespeare's plays are in the public domain, and the repository is released under the MIT license.

| Property | Value |
| --- | --- |
| Size | 1,115,394 bytes, all ASCII, so also 1,115,394 characters |
| Alphabet | 65 distinct characters: the 52 letters, the digit 3, space, newline, and 10 punctuation marks |
| Words | about 204,000 lowercase word tokens and 12,400 types ([chapter 1](../01-text-tokens-and-tokenization.md#words-types-and-tokens)) |
| Lines | 40,000, with excerpts of several plays concatenated play by play |

Most chapters split the file by position. The first 90% (1,003,854 characters) is the training text and the last 10% (111,540 characters) the held-out text. The split is not random, so the held-out text comes from different plays than the training text: it consists mostly of *The Taming of the Shrew*, and its last 34,657 characters are from *The Tempest*. This makes held-out scores measure generalization to unseen plays rather than to unseen lines of familiar ones, and it explains some numbers in the notes: the best Kneser–Ney character model needs 2.52 bits per character on the last 10% when trained on the first 80%, but 2.25 when trained on the first 90%, because the added tenth contains the opening of *The Taming of the Shrew* ([chapter 2](../02-n-gram-language-models-and-perplexity.md#comparing-the-methods)), and chapters 5 and 10 use *The Tempest* as the unseen play to adapt to. The file is small enough for every experiment in the module to run on a laptop CPU, and large enough for a character-level model to learn spelling, word boundaries, and the layout of a script.

## <a id="the-character-transformer"></a>The character transformer

`shakespeare-char-transformer.pt` holds a small transformer language model over the 65 characters of Tiny Shakespeare, trained by `Sources/Figure code/ch04_transformer_lm.py` on the first 90% of the file ([chapter 4](../04-transformer-language-models.md)). It is saved with `torch.save` as a dictionary with four entries:

| Key | Contents |
| --- | --- |
| `config` | `{"V": 65, "T": 128, "d": 128, "L": 4, "heads": 4}`: vocabulary size, context length, width, layers, and attention heads |
| `chars` | The 65 characters in the order of their token ids |
| `state_dict` | The weights of the `CharLM` class defined in chapter 4's figure script and in the code blocks that load it: learned token and position embeddings, four pre-norm transformer blocks with GELU feedforward layers four times the width, a final layer norm, and an output layer tied to the token embeddings; 818,048 parameters |
| `curves` | Training and held-out loss logged during training, which the chapter 4 figure plots |

The model was trained for 4,000 steps of 32 windows of 128 characters with AdamW (weight decay 0.1) and a one-cycle learning-rate schedule peaking at $`2\times10^{-3}`$. It reaches 2.19 bits per character on the held-out text, against 2.25 for the best Kneser–Ney model of chapter 2. Chapter 5 fine-tunes it on *The Tempest*, chapter 8 decodes from it, and chapter 10 adapts it with LoRA and task vectors.

The file can be inspected without the model class:

```python
import torch

ckpt = torch.load("Sources/Data/shakespeare-char-transformer.pt")
print(sorted(ckpt), ckpt["config"])
print(repr("".join(ckpt["chars"])))
print(f"{len(ckpt['curves']['steps'])} logged steps; final held-out loss {ckpt['curves']['val'][-1]:.3f} bits per character")
# ['chars', 'config', 'curves', 'state_dict'] {'V': 65, 'T': 128, 'd': 128, 'L': 4, 'heads': 4}
# "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# 21 logged steps; final held-out loss 2.189 bits per character
```

The code blocks that use the model rebuild it with `model = CharLM(**ckpt["config"])` and `model.load_state_dict(ckpt["state_dict"])`, with the `CharLM` class defined in each block; like all the module's code, they open the file by a path relative to the `4. NLP&LLMs` folder. If the file is missing, running `python ch04_transformer_lm.py` from `Sources/Figure code` trains it again in about 20 minutes on two CPU cores; with `--retrain` it overwrites an existing checkpoint. A retrained model will print slightly different numbers from those in the notes, since training on a CPU with several threads is not bit-for-bit reproducible.
