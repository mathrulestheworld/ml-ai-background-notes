[ML Mastery Notes](../../README.md) › [NLP and Large Language Models](../README.md)

# Book and documentation links

# <a id="books-and-documentation"></a>Books and documentation

The chapter notes develop the module's main material. These books and documentation pages provide alternative explanations, fuller treatments of the classical methods, and the tools used for real work with language models. The reading plan lists the lecture slides and readings for each topic.

## <a id="main-references"></a>Main references

| Source | Relevant material |
| --- | --- |
| Jurafsky and Martin, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/), third edition draft of August 19, 2026 | The standard textbook, free as PDFs by chapter with slides: chapter 2 (chapter 1), 3 and appendix C (chapter 2), 5 and appendix J (chapter 3), 7 (chapter 4), 9 (chapter 5), 11 (chapter 13), 13 (chapter 15), and 18–20 with appendix E (chapter 16). Chapter numbers change between drafts. |
| Eisenstein, [*Introduction to Natural Language Processing*](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf) (MIT Press, 2019) | Free draft PDF, stronger on the classical methods: chapter 6 (chapter 2), 14 (chapter 3), 18 (chapter 15), and 7–11 (chapter 16). |
| Xiao and Zhu, [*Foundations of Large Language Models*](https://arxiv.org/abs/2501.09223) (2025) | Free; pretraining, generative models, prompting, alignment, and inference (chapters 4–12). |
| Lambert, [*Reinforcement Learning from Human Feedback*](https://rlhfbook.com/) | Free; instruction tuning, reward models, policy-gradient and direct preference optimization, and post-training recipes (chapters 10–12). |
| Manning, Raghavan, and Schütze, [*Introduction to Information Retrieval*](https://nlp.stanford.edu/IR-book/) (Cambridge, 2008) | Free; inverted indexes, tf-idf weighting, probabilistic retrieval, and evaluation (chapter 13). |

## <a id="specialized-books-and-surveys"></a>Specialized books and surveys

| Source | Relevant material |
| --- | --- |
| Raschka, *Build a Large Language Model (From Scratch)* (Manning, 2024), with [code](https://github.com/rasbt/LLMs-from-scratch) | A GPT built step by step in PyTorch: tokenization, attention, pretraining, and fine-tuning for classification and instructions (chapters 1, 4, and 10). |
| Tunstall, von Werra, and Wolf, *Natural Language Processing with Transformers* (O'Reilly, 2022), with [notebooks](https://github.com/nlp-with-transformers/notebooks) | Practical use of the Hugging Face libraries for classification, named-entity recognition, generation, summarization, and question answering (chapters 5, 8, 10, and 13). |
| Alammar and Grootendorst, *Hands-On Large Language Models* (O'Reilly, 2024), with [code](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models) | Illustrated treatment of tokens and embeddings, prompting, retrieval-augmented generation, and fine-tuning (chapters 1, 5, 9–10, and 13). |
| Welleck et al., [*From Decoding to Meta-Generation: Inference-Time Algorithms for Large Language Models*](https://arxiv.org/abs/2406.16838) (2024) | A survey of token-level decoding, sampling and search over whole outputs, and efficient generation (chapters 8 and 12). |
| Robertson and Zaragoza, [*The Probabilistic Relevance Framework: BM25 and Beyond*](https://doi.org/10.1561/1500000019) (2009) | The derivation of BM25 and its extensions (chapter 13). |
| Koehn, [*Statistical Machine Translation*](https://doi.org/10.1017/CBO9780511815829) (Cambridge, 2010), and [*Neural Machine Translation*](https://doi.org/10.1017/9781108608480) (Cambridge, 2020) | Word alignment, phrase-based models, and decoding; then neural encoder–decoder translation, its training, and its evaluation (chapter 15). |
| Kübler, McDonald, and Nivre, *Dependency Parsing* (Morgan & Claypool, 2009) | Transition-based and graph-based dependency parsing in depth (chapter 16). |

## <a id="library-documentation"></a>Library documentation

| Source | Relevant material |
| --- | --- |
| [PyTorch documentation](https://pytorch.org/docs/stable/) | `nn.TransformerEncoderLayer` for the character transformer (chapters 4, 5, 8, and 10), `torch.nn.utils.parametrize` for LoRA (chapter 10), and optimizers and schedulers. |
| [NumPy documentation](https://numpy.org/doc/stable/) | Arrays and random generation; used in almost every chapter. |
| [SciPy documentation](https://docs.scipy.org/doc/scipy/) | `kmeans2` for inverted files and product quantization (chapter 13), `logsumexp` (chapter 16), and `minimize` for fitting scaling laws (chapter 7). |
| [scikit-learn t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) | The embedding map of chapter 3. |

## <a id="tools-for-larger-problems"></a>Tools for larger problems

The notes implement every method from scratch on small examples. For real models and data, mature implementations exist:

| Tool | Use |
| --- | --- |
| [Hugging Face Transformers](https://huggingface.co/docs/transformers), [Tokenizers](https://huggingface.co/docs/tokenizers), and [Datasets](https://huggingface.co/docs/datasets) | Pretrained models, fast tokenizers, and datasets (chapters 1, 4–5, and 10). |
| [tiktoken](https://github.com/openai/tiktoken) and [SentencePiece](https://github.com/google/sentencepiece) | Production byte-level BPE and unigram tokenizers (chapter 1). |
| [KenLM](https://kheafield.com/code/kenlm/) | Large Kneser–Ney n-gram models (chapter 2). |
| [DataTrove](https://github.com/huggingface/datatrove) | Filtering and deduplication of web-scale text, as used to build FineWeb (chapter 6). |
| [vLLM](https://docs.vllm.ai/) | Efficient serving with paged key–value caches, continuous batching, and speculative decoding (chapter 8). |
| [PEFT](https://huggingface.co/docs/peft) and [TRL](https://huggingface.co/docs/trl) | LoRA and other adapters; supervised fine-tuning, reward modeling, DPO, and GRPO (chapters 10–12). |
| [FAISS](https://github.com/facebookresearch/faiss) | Approximate nearest-neighbor search with inverted files, product quantization, and graphs (chapter 13). |
| [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) | Standardized benchmark evaluation (chapter 14). |
| [SacreBLEU](https://github.com/mjpost/sacrebleu) | Reproducible BLEU and chrF scores (chapter 15). |
| [Stanza](https://stanfordnlp.github.io/stanza/) and [spaCy](https://spacy.io/) | Tokenization, tagging, named entities, and dependency parsing in many languages (chapter 16). |
