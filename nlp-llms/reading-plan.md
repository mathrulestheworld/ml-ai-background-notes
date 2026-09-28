[Background Notes](../README.md) › [NLP and Large Language Models](README.md)

# Reading plan

## <a id="courses"></a>Courses

- **Main course — Stanford CS224N, Spring 2024: Christopher Manning, *Natural Language Processing with Deep Learning*.** [Course homepage](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/) · [Lecture playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D). Supplies word vectors, language models, pretraining, post-training, evaluation, reasoning and agents, machine translation, and parsing. Its four assignments cover word vectors, dependency parsing, neural machine translation, and pretraining and fine-tuning a small transformer.
- **Main course — Stanford CS336, Spring 2025: Percy Liang and Tatsunori Hashimoto, *Language Modeling from Scratch*.** [Course homepage](https://cs336.stanford.edu/spring2025/) · [Lecture playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_). Builds a language model end to end: tokenization, resource accounting, architectures, mixture of experts, scaling laws, inference, evaluation, data, and alignment with reinforcement learning. Several lectures are executable Python files viewed in the course's trace viewer. The [Spring 2026 recordings](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV) update the same sequence.
- **Supplement — CMU 11-711, Fall 2025: Sean Welleck, *Advanced Natural Language Processing*.** [Course page with slides and code](https://cmu-l3.github.io/anlp-fall2025/). Adds in-context learning, fine-tuning, decoding and inference-time algorithms, retrieval, evaluation and experimental design, reinforcement learning for language models, and agents.

The order below follows the chapters of this module. Each block lists the material to cover and its primary lectures; further reading and exercises provide additional depth. Blocks 1–14 form the main plan; 15–16 are optional extensions. Timing is flexible.

Slide links are the courses' original PDFs; CS224N videos are the Spring 2024 recordings except where a block names the 2023 recording. Deep Learning already covered backpropagation, recurrent networks, attention, the transformer block, position encodings, self-supervised learning, and training at scale, so CS224N lectures 3, 5 (second half), and 8 and CS336 lectures 5–8 (GPUs, kernels, and parallelism) serve as review. Policy-gradient methods, which preference tuning and reasoning training use, are developed in the RL module; this module states the objectives they optimize.

## <a id="topic-list"></a>Topic list

- [ ] [1. Text, tokens, and tokenization](#1-text-tokens-and-tokenization)
- [ ] [2. N-gram language models and perplexity](#2-n-gram-language-models-and-perplexity)
- [ ] [3. Word embeddings](#3-word-embeddings)
- [ ] [4. Transformer language models](#4-transformer-language-models)
- [ ] [5. Pretraining and transfer](#5-pretraining-and-transfer)
- [ ] [6. Pretraining data](#6-pretraining-data)
- [ ] [7. Scaling laws](#7-scaling-laws)
- [ ] [8. Decoding and text generation](#8-decoding-and-text-generation)
- [ ] [9. In-context learning and prompting](#9-in-context-learning-and-prompting)
- [ ] [10. Fine-tuning and parameter-efficient adaptation](#10-fine-tuning-and-parameter-efficient-adaptation)
- [ ] [11. Learning from human preferences](#11-learning-from-human-preferences)
- [ ] [12. Reasoning and test-time compute](#12-reasoning-and-test-time-compute)
- [ ] [13. Retrieval, tools, and agents](#13-retrieval-tools-and-agents)
- [ ] [14. Evaluating language models](#14-evaluating-language-models)
- [ ] [15. Machine translation and multilingual models — optional](#15-machine-translation-and-multilingual-models-optional)
- [ ] [16. Syntactic parsing — optional](#16-syntactic-parsing-optional)

### <a id="reference-books-used-below"></a>Reference books used below

- **SLP3:** Jurafsky and Martin, [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/), third edition draft of August 19, 2026. Free PDFs by chapter, with slides for many; chapter numbers below refer to this draft and change between drafts.
- **Eisenstein:** Jacob Eisenstein, *Introduction to Natural Language Processing* (MIT Press, 2019); a free [draft · PDF](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf). Stronger on the classical methods of blocks 2, 15, and 16.
- **FLLM:** Xiao and Zhu, [Foundations of Large Language Models](https://arxiv.org/abs/2501.09223) (2025). Free; pretraining, generative models, prompting, alignment, and inference, for blocks 4–12.
- **RLHF book:** Nathan Lambert, [Reinforcement Learning from Human Feedback](https://rlhfbook.com/). Free; reward models, preference optimization, and post-training recipes for blocks 10–12.
- **IIR:** Manning, Raghavan, and Schütze, [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/) (Cambridge, 2008). Free; the reference for block 13's classical retrieval.

## <a id="1-text-tokens-and-tokenization"></a>1. Text, tokens, and tokenization

**Topics:** Characters, Unicode, and bytes; normalization; words, types, and tokens; Zipf's and Heaps' laws; word, character, and byte vocabularies; subword tokenization with byte-pair encoding, WordPiece, and the unigram language model; pre-tokenization and special tokens; compression rate; artifacts of tokenization in arithmetic, code, and non-English text; tokenizer-free models.

- **CS336 lecture 1: Overview, tokenization.** [Executable lecture](http://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_01.json) · [Video](https://www.youtube.com/watch?v=SQ3fZ1sAqXI)
- **Andrej Karpathy: Let's build the GPT tokenizer.** [Video](https://www.youtube.com/watch?v=zduSFxRajkE) Builds a byte-level BPE tokenizer and examines the GPT-2 and GPT-4 tokenizers and their failure modes.
- **SLP3 chapter 2: Words and tokens.** [Chapter · PDF](https://web.stanford.edu/~jurafsky/slp3/2.pdf) · [Slides · PDF](https://web.stanford.edu/~jurafsky/slp3/slides/tokens_jan26.pdf)

**Further reading:** Sennrich, Haddow, and Birch, [Neural machine translation of rare words with subword units](https://arxiv.org/abs/1508.07909) (2016), introduced BPE for neural models; Kudo, [Subword regularization](https://arxiv.org/abs/1804.10959) (2018), the unigram tokenizer; Kudo and Richardson, [SentencePiece](https://arxiv.org/abs/1808.06226) (2018). CS336 assignment 1 implements a byte-level BPE tokenizer.

## <a id="2-n-gram-language-models-and-perplexity"></a>2. N-gram language models and perplexity

**Topics:** Language models and the chain rule; the Markov assumption; maximum-likelihood estimation; sparsity and smoothing: add-$`k`$, backoff, interpolation, and Kneser–Ney; perplexity and cross-entropy; the entropy of English; sampling from a language model; the noisy channel.

- **SLP3 chapter 3 and appendix C: N-gram language models, Kneser–Ney smoothing.** [Chapter 3 · PDF](https://web.stanford.edu/~jurafsky/slp3/3.pdf) · [Slides · PDF](https://web.stanford.edu/~jurafsky/slp3/slides/lm_jan25.pdf) · [Appendix C · PDF](https://web.stanford.edu/~jurafsky/slp3/C.pdf)
- **CS224N lecture 5: Recurrent neural networks, first half on language models and n-grams.** [Slides · PDF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture05-rnnlm.pdf) · [Video](https://www.youtube.com/watch?v=fyc0Jzr74y4)
- **Andrej Karpathy: The spelled-out intro to language modeling.** [Video](https://www.youtube.com/watch?v=PaCmpygFfXo) Count-based and neural bigram models of names, with sampling and the negative log-likelihood.

**Further reading:** Eisenstein chapter 6; Chen and Goodman, [An empirical study of smoothing techniques for language modeling](https://doi.org/10.1006/csla.1999.0128) (1999), the standard comparison that established modified Kneser–Ney.

## <a id="3-word-embeddings"></a>3. Word embeddings

**Topics:** The distributional hypothesis; term–document and word–context matrices; pointwise mutual information and PPMI; truncated SVD and latent semantic analysis; word2vec skip-gram and CBOW; negative sampling and its equivalence to factorizing shifted PMI; GloVe; subword embeddings; similarity and analogy evaluation; social biases in embeddings; from static to contextual representations.

- **CS224N lecture 1: Intro and word vectors.** [Slides · PDF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture01-wordvecs1.pdf) · [Notes · PDF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/readings/cs224n_winter2023_lecture1_notes_draft.pdf) · [Video](https://www.youtube.com/watch?v=DzpHeXVSC5I)
- **CS224N lecture 2: Word vectors and language models.** [Slides · PDF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture02-wordvecs2.pdf) · [Notes · PDF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/readings/cs224n-2019-notes02-wordvecs2.pdf) · [Video](https://www.youtube.com/watch?v=nBor4jfWetQ)
- **SLP3 chapter 5 and appendix J: Embeddings, PPMI.** [Chapter 5 · PDF](https://web.stanford.edu/~jurafsky/slp3/5.pdf) · [Slides · PDF](https://web.stanford.edu/~jurafsky/slp3/slides/vector25aug.pdf) · [Appendix J · PDF](https://web.stanford.edu/~jurafsky/slp3/J.pdf)

**Further reading:** Mikolov et al., [Distributed representations of words and phrases](https://arxiv.org/abs/1310.4546) (2013); Pennington, Socher, and Manning, [GloVe](https://aclanthology.org/D14-1162/) (2014); Levy, Goldberg, and Dagan, [Improving distributional similarity with lessons learned from word embeddings](https://aclanthology.org/Q15-1016/) (2015), which shows that the hyperparameters matter more than the algorithm. CS224N assignment 1 compares co-occurrence and word2vec vectors.

## <a id="4-transformer-language-models"></a>4. Transformer language models

**Topics:** Neural language models from feedforward and recurrent networks to transformers; the decoder-only language model: embeddings, blocks, unembedding, and weight tying; next-token loss and teacher forcing; counting parameters, FLOPs, and memory; the architecture and hyperparameter choices of current models; training instabilities and their fixes; mixture-of-experts layers, routing, and load balancing; attention alternatives and long context.

- **CS336 lecture 2: PyTorch, resource accounting.** [Executable lecture](http://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_02.json) · [Video](https://www.youtube.com/watch?v=msHyYioAyNE)
- **CS336 lecture 3: Architectures, hyperparameters.** [Slides · PDF](https://github.com/stanford-cs336/spring2025-lectures/blob/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [Video](https://www.youtube.com/watch?v=ptFiH_bHnJw)
- **CS336 lecture 4: Mixture of experts.** [Slides · PDF](https://github.com/stanford-cs336/spring2025-lectures/blob/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [Video](https://www.youtube.com/watch?v=LPv1KfUXLCo)
- **Andrej Karpathy: Let's build GPT, from scratch, in code, spelled out.** [Video](https://www.youtube.com/watch?v=kCc8FmEb1nY) Trains a character-level transformer on Tiny Shakespeare, the corpus used in this module's code.
- **SLP3 chapter 7: Transformers and pretraining.** [Chapter · PDF](https://web.stanford.edu/~jurafsky/slp3/7.pdf) · [Slides · PDF](https://web.stanford.edu/~jurafsky/slp3/slides/transformer_aug26.pdf)

**Further reading:** Bengio et al., [A neural probabilistic language model](https://www.jmlr.org/papers/v3/bengio03a.html) (2003); Radford et al., [Language models are unsupervised multitask learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, 2019); Karpathy, [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU), a complete training run with its engineering. FLLM chapter 2. CS336 assignment 1 implements the model, AdamW, and the training loop.

## <a id="5-pretraining-and-transfer"></a>5. Pretraining and transfer

**Topics:** Pretrain, then adapt; contextual embeddings; generative pretraining followed by fine-tuning; masked language modeling and its refinements; replaced-token detection; span corruption and the text-to-text format; encoder, decoder, and encoder–decoder models compared; fine-tuning for classification; sentence embeddings; probing what pretraining learns.

- **CS224N lecture 9: Pretraining.** [Slides · PDF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture09-pretraining-updated.pdf) · [Video, 2023 recording](https://www.youtube.com/watch?v=DGfCRXuNA2w)
- **SLP3 chapter 9: Masked language models.** [Chapter · PDF](https://web.stanford.edu/~jurafsky/slp3/9.pdf) · [Slides · PDF](https://web.stanford.edu/~jurafsky/slp3/slides/mlmjan25.pdf)
- **CMU 11-711 lecture 6: Pretraining.** [Slides · PDF](https://cmu-l3.github.io/anlp-fall2025/static_files/anlp-f2025-06-pretraining.pdf)

**Further reading:** Devlin et al., [BERT](https://arxiv.org/abs/1810.04805) (2019); Raffel et al., [Exploring the limits of transfer learning with a unified text-to-text transformer](https://arxiv.org/abs/1910.10683) (T5, 2020), a systematic comparison of objectives and architectures; Clark et al., [ELECTRA](https://arxiv.org/abs/2003.10555) (2020). CS224N assignment 4 pretrains a small transformer and fine-tunes it.

## <a id="6-pretraining-data"></a>6. Pretraining data

**Topics:** Sources: web crawls, books, code, and curated collections; extracting text from HTML; language identification; heuristic and model-based quality filters; exact and near-duplicate removal with MinHash and locality-sensitive hashing; data mixtures and how to choose them; repeating data; contamination of evaluations; synthetic data; licensing, consent, and privacy.

- **CS336 lecture 13: Data.** [Executable lecture](http://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_13.json) · [Video](https://www.youtube.com/watch?v=WePxmeXU1xg)
- **CS336 lecture 14: Data, filtering and deduplication.** [Executable lecture](http://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_14.json) · [Video](https://www.youtube.com/watch?v=9Cd0THLS1t0)
- **Penedo et al., *The FineWeb datasets* (2024).** [Paper](https://arxiv.org/abs/2406.17557) A documented web-data pipeline with ablations of every filtering and deduplication step.

**Further reading:** Li et al., [DataComp-LM](https://arxiv.org/abs/2406.11794) (2024), a benchmark for data curation that established classifier-based filtering; Lee et al., [Deduplicating training data makes language models better](https://arxiv.org/abs/2107.06499) (2022); Muennighoff et al., [Scaling data-constrained language models](https://arxiv.org/abs/2305.16264) (2023). CS336 assignment 4 builds a filtering and deduplication pipeline on Common Crawl.

## <a id="7-scaling-laws"></a>7. Scaling laws

**Topics:** Power laws in loss; scaling with parameters, data, and compute; compute-optimal allocation; IsoFLOP profiles and parametric fits; training beyond the compute-optimal point for cheaper inference; scaling with limited data; scaling of hyperparameters and μP; predicting downstream performance; emergent abilities and the choice of metric.

- **CS336 lecture 9: Scaling laws.** [Slides · PDF](https://github.com/stanford-cs336/spring2025-lectures/blob/fb79eb018fa047bf99c4c785dcbbd62fff361e54/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [Video](https://www.youtube.com/watch?v=6Q-ESEmDf4Q)
- **CS336 lecture 11: Scaling laws, details.** [Slides · PDF](https://github.com/stanford-cs336/spring2025-lectures/blob/00191bba00d6d64621dc46ccaed9122681413a24/nonexecutable/2025%20Lecture%2011%20-%20Scaling%20details.pdf) · [Video](https://www.youtube.com/watch?v=OSYuUqGBQxw)

**Further reading:** Kaplan et al., [Scaling laws for neural language models](https://arxiv.org/abs/2001.08361) (2020); Hoffmann et al., [Training compute-optimal large language models](https://arxiv.org/abs/2203.15556) (Chinchilla, 2022); Besiroglu et al., [Chinchilla scaling: a replication attempt](https://arxiv.org/abs/2404.10102) (2024); Schaeffer, Miranda, and Koyejo, [Are emergent abilities of large language models a mirage?](https://arxiv.org/abs/2304.15004) (2023). CS336 assignment 3 fits scaling laws under a fixed budget of training runs.

## <a id="8-decoding-and-text-generation"></a>8. Decoding and text generation

**Topics:** Generation as search and as sampling; greedy and beam search; the likelihood trap and degenerate text; temperature, top-$`k`$, nucleus, and min-$`p`$ sampling; repetition penalties; constrained decoding and structured output; speculative decoding and why it is exact; the cost of generation: prefill and decode, batching, and paged key–value caches.

- **CMU 11-711 lecture 9: Decoding algorithms.** [Slides · PDF](https://cmu-l3.github.io/anlp-fall2025/static_files/anlp-f2025-09-decoding.pdf)
- **CS336 lecture 10: Inference.** [Executable lecture](http://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_10.json) · [Video](https://www.youtube.com/watch?v=fcgPYo3OtV0)
- **Welleck et al., *From decoding to meta-generation*, sections 1–3.** [Paper](https://arxiv.org/abs/2406.16838) A survey of token-level decoding, from greedy search to sampling adapters.

**Further reading:** Holtzman et al., [The curious case of neural text degeneration](https://arxiv.org/abs/1904.09751) (2020); Meister, Cotterell, and Vieira, [If beam search is the answer, what was the question?](https://arxiv.org/abs/2010.02650) (2020); Leviathan, Kalman, and Matias, [Fast inference from transformers via speculative decoding](https://arxiv.org/abs/2211.17192) (2023); Kwon et al., [Efficient memory management for large language model serving with PagedAttention](https://arxiv.org/abs/2309.06180) (2023).

## <a id="9-in-context-learning-and-prompting"></a>9. In-context learning and prompting

**Topics:** Few-shot prompting and its dependence on scale; zero-shot instructions; sensitivity to format, order, and label balance; calibration; what the demonstrations contribute; in-context learning as implicit Bayesian inference; transformers that implement learning algorithms in their forward pass; induction heads; the data properties that give rise to in-context learning; prompt design and its limits.

- **CMU 11-711 lecture 7: In-context learning and prompting.** [Slides · PDF](https://cmu-l3.github.io/anlp-fall2025/static_files/anlp-f2025-07-icl-prompting.pdf)
- **CS224N lecture 10: Post-training, first part on prompting and instruction following.** [Slides · PDF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture10-prompting-rlhf.pdf) · [Video](https://www.youtube.com/watch?v=35X6zlhoCy4)

**Further reading:** Brown et al., [Language models are few-shot learners](https://arxiv.org/abs/2005.14165) (GPT-3, 2020); Min et al., [Rethinking the role of demonstrations](https://arxiv.org/abs/2202.12837) (2022); Garg et al., [What can transformers learn in-context?](https://arxiv.org/abs/2208.01066) (2022); von Oswald et al., [Transformers learn in-context by gradient descent](https://arxiv.org/abs/2212.07677) (2023). FLLM chapter 3.

## <a id="10-fine-tuning-and-parameter-efficient-adaptation"></a>10. Fine-tuning and parameter-efficient adaptation

**Topics:** Full fine-tuning; supervised instruction tuning and chat templates; instruction data, collected, templated, and synthetic; forgetting; parameter-efficient methods: adapters, prompt and prefix tuning, LoRA, and QLoRA; the low intrinsic dimension of fine-tuning; model merging and task arithmetic; distilling a stronger model.

- **CMU 11-711 lecture 8: Fine-tuning and distillation.** [Slides · PDF](https://cmu-l3.github.io/anlp-fall2025/static_files/anlp-f2025-08-finetuning.pdf)
- **CS224N lecture 12: Efficient training, section on parameter-efficient fine-tuning.** [Slides · PDF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture12-training-shikhar.pdf) · [Video](https://www.youtube.com/watch?v=UVX7SYGCKkA)

**Further reading:** Wei et al., [Finetuned language models are zero-shot learners](https://arxiv.org/abs/2109.01652) (FLAN, 2022); Zhou et al., [LIMA: less is more for alignment](https://arxiv.org/abs/2305.11206) (2023); Hu et al., [LoRA](https://arxiv.org/abs/2106.09685) (2022); Dettmers et al., [QLoRA](https://arxiv.org/abs/2305.14314) (2023). RLHF book chapters on instruction tuning.

## <a id="11-learning-from-human-preferences"></a>11. Learning from human preferences

**Topics:** Why imitation is not enough; collecting preferences; the Bradley–Terry reward model; KL-regularized reward maximization and its optimal policy; RLHF with a policy-gradient optimizer; direct preference optimization and its variants; best-of-$`n`$ sampling; reward overoptimization; AI feedback and constitutional methods; what preference tuning changes in a model.

- **CS224N lecture 10: Post-training, second part on RLHF and DPO.** [Slides · PDF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture10-prompting-rlhf.pdf) · [Video](https://www.youtube.com/watch?v=35X6zlhoCy4)
- **CS336 lecture 15: Alignment, SFT and RLHF.** [Slides · PDF](https://github.com/stanford-cs336/spring2025-lectures/blob/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [Video](https://www.youtube.com/watch?v=Dfu7vC9jo4w)
- **CS224N lecture 15: Life after DPO (Nathan Lambert).** [Slides · PDF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture15-life-after-dpo-lambert.pdf) · [Video](https://www.youtube.com/watch?v=dnF463_Ar9I)

**Further reading:** Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT, 2022); Rafailov et al., [Direct preference optimization](https://arxiv.org/abs/2305.18290) (2023); Gao, Schulman, and Hilton, [Scaling laws for reward model overoptimization](https://arxiv.org/abs/2210.10760) (2023); Bai et al., [Constitutional AI](https://arxiv.org/abs/2212.08073) (2022). RLHF book chapters on reward modeling, regularization, and direct alignment.

## <a id="12-reasoning-and-test-time-compute"></a>12. Reasoning and test-time compute

**Topics:** Chain-of-thought prompting and why intermediate tokens add computation; self-consistency and majority voting; best-of-$`n`$ with verifiers; outcome and process reward models; search over reasoning steps; reinforcement learning from verifiable rewards; test-time scaling; the limits and faithfulness of reasoning traces.

- **CS336 lecture 16: Alignment, reinforcement learning from verifiable rewards.** [Slides · PDF](https://github.com/stanford-cs336/spring2025-lectures/blob/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [Video](https://www.youtube.com/watch?v=46f2QTDB08Q)
- **CS336 lecture 17: Alignment, policy gradients for language models.** [Executable lecture](http://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_17.json) · [Video](https://www.youtube.com/watch?v=JdGFdViaOJk)
- **CS224N lecture 14: Reasoning and agents, first half.** [Slides · PDF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture14-agents-shikhar-updated.pdf) · [Video](https://www.youtube.com/watch?v=I0tj4Y7xaOQ)
- **CMU 11-711 lectures 18 and 26: Reinforcement learning for language models, advanced inference.** [Slides 18 · PDF](https://cmu-l3.github.io/anlp-fall2025/static_files/anlp-f2025-15-rl-llms.pdf) · [Slides 26 · PDF](https://cmu-l3.github.io/anlp-fall2025/static_files/anlp-f2025-22-inference.pdf)

**Further reading:** Wei et al., [Chain-of-thought prompting elicits reasoning](https://arxiv.org/abs/2201.11903) (2022); Wang et al., [Self-consistency improves chain of thought reasoning](https://arxiv.org/abs/2203.11171) (2023); Lightman et al., [Let's verify step by step](https://arxiv.org/abs/2305.20050) (2023); Snell et al., [Scaling LLM test-time compute optimally](https://arxiv.org/abs/2408.03314) (2024); DeepSeek-AI, [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (2025). Welleck et al. sections 4–7. CS336 assignment 5 trains a model on mathematics with expert iteration and GRPO.

## <a id="13-retrieval-tools-and-agents"></a>13. Retrieval, tools, and agents

**Topics:** Sparse retrieval with TF-IDF and BM25; dense retrieval with dual encoders; approximate nearest-neighbor search; measuring retrieval; retrieval-augmented generation; retrieval versus long context; tool use and function calling; agents that plan, act, and observe; benchmarks for agents; failures, including prompt injection.

- **SLP3 chapter 11: Information retrieval and retrieval-augmented generation.** [Chapter · PDF](https://web.stanford.edu/~jurafsky/slp3/11.pdf) · [Slides · PDF](https://web.stanford.edu/~jurafsky/slp3/slides/ir_nov25.pdf)
- **CMU 11-711 lecture 10: Retrieval and RAG (Akari Asai).** [Slides · PDF](https://cmu-l3.github.io/anlp-fall2025/static_files/akari_anlp_2025_rag_lecture.pdf)
- **CS224N lecture 14: Reasoning and agents, second half.** [Slides · PDF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture14-agents-shikhar-updated.pdf) · [Video](https://www.youtube.com/watch?v=I0tj4Y7xaOQ)
- **CMU 11-711 lecture 19: Agents.** [Slides · PDF](https://cmu-l3.github.io/anlp-fall2025/static_files/anlp-f2025-17-agents.pdf)

**Further reading:** IIR chapters 6 and 11; Karpukhin et al., [Dense passage retrieval](https://arxiv.org/abs/2004.04906) (2020); Lewis et al., [Retrieval-augmented generation for knowledge-intensive NLP tasks](https://arxiv.org/abs/2005.11401) (2020); Yao et al., [ReAct](https://arxiv.org/abs/2210.03629) (2023); Jimenez et al., [SWE-bench](https://arxiv.org/abs/2310.06770) (2024).

## <a id="14-evaluating-language-models"></a>14. Evaluating language models

**Topics:** Perplexity and bits per byte; kinds of benchmarks: multiple choice, free generation, and execution; the pass@$`k`$ estimator; contamination; human and model judges and their biases; pairwise comparisons and rating systems; the statistics of evaluation: standard errors, clustered questions, and paired comparisons; holistic evaluation; Goodhart's law and saturated benchmarks.

- **CS224N lecture 11: Benchmarking and evaluation (Yann Dubois).** [Slides · PDF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture11-evaluation-yann.pdf) · [Video](https://www.youtube.com/watch?v=TO0CqzqiArM)
- **CS336 lecture 12: Evaluation.** [Executable lecture](http://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_12.json) · [Video](https://www.youtube.com/watch?v=x-R5l2HsXqM)
- **CMU 11-711 lectures 13–14: Evaluation techniques, experimental design.** [Slides 13 · PDF](https://cmu-l3.github.io/anlp-fall2025/static_files/anlp-f2025-13-evaluation.pdf) · [Slides 14 · PDF](https://cmu-l3.github.io/anlp-fall2025/static_files/anlp-f2025-14-experimentation.pdf)

**Further reading:** Miller, [Adding error bars to evals](https://arxiv.org/abs/2411.00640) (2024); Chen et al., [Evaluating large language models trained on code](https://arxiv.org/abs/2107.03374) (2021), which defines pass@$`k`$; Zheng et al., [Judging LLM-as-a-judge](https://arxiv.org/abs/2306.05685) (2023); Chiang et al., [Chatbot Arena](https://arxiv.org/abs/2403.04132) (2024); Biderman et al., [Lessons from the trenches on reproducible evaluation of language models](https://arxiv.org/abs/2405.14782) (2024).

## <a id="15-machine-translation-and-multilingual-models-optional"></a>15. Machine translation and multilingual models — optional

**Topics:** Translation as a noisy channel; word alignment and IBM Model 1 trained with EM; phrase-based translation in brief; neural machine translation with encoder–decoder transformers; evaluation with BLEU, chrF, and learned metrics; back-translation; multilingual pretraining and cross-lingual transfer; the curse of multilinguality; low-resource languages.

- **CS224N lecture 6: Sequence-to-sequence models and machine translation.** [Slides · PDF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture06-fancy-rnn.pdf) · [Video](https://www.youtube.com/watch?v=Ba6Fn1-Jsfw)
- **SLP3 chapter 13: Machine translation.** [Chapter · PDF](https://web.stanford.edu/~jurafsky/slp3/13.pdf)

**Further reading:** Eisenstein chapter 18; Brown et al., [The mathematics of statistical machine translation](https://aclanthology.org/J93-2003/) (1993); Papineni et al., [BLEU](https://aclanthology.org/P02-1040/) (2002); Conneau et al., [Unsupervised cross-lingual representation learning at scale](https://arxiv.org/abs/1911.02116) (XLM-R, 2020). CS224N assignment 3 builds a neural translation system with attention.

## <a id="16-syntactic-parsing-optional"></a>16. Syntactic parsing — optional

**Topics:** Part-of-speech tagging and named-entity recognition as sequence labeling; constituency, context-free grammars, and treebanks; CKY parsing and probabilistic grammars; dependency grammar and Universal Dependencies; transition-based parsing; graph-based parsing with maximum spanning trees; neural parsers; evaluation; syntax inside language models.

- **CS224N lecture 4: Dependency parsing.** [Slides · PDF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture04-dep-parsing.pdf) · [Notes · PDF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/readings/cs224n-2019-notes04-dependencyparsing.pdf) · [Video](https://www.youtube.com/watch?v=KVKvde-_MYc)
- **CS224N lecture 16: ConvNets, tree recursive networks, and constituency parsing.** [Slides · PDF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture16-CNN-TreeRNN.pdf) · [Video](https://www.youtube.com/watch?v=S8d-7v3f5MQ)
- **SLP3 chapters 18–20 and appendix E: Sequence labeling, constituency parsing, dependency parsing, statistical parsing.** [Chapter 18 · PDF](https://web.stanford.edu/~jurafsky/slp3/18.pdf) · [Chapter 19 · PDF](https://web.stanford.edu/~jurafsky/slp3/19.pdf) · [Chapter 20 · PDF](https://web.stanford.edu/~jurafsky/slp3/20.pdf) · [Appendix E · PDF](https://web.stanford.edu/~jurafsky/slp3/E.pdf)

**Further reading:** Eisenstein chapters 7–11; Chen and Manning, [A fast and accurate dependency parser using neural networks](https://aclanthology.org/D14-1082/) (2014); Dozat and Manning, [Deep biaffine attention for neural dependency parsing](https://arxiv.org/abs/1611.01734) (2017); Hewitt and Manning, [A structural probe for finding syntax in word representations](https://aclanthology.org/N19-1419/) (2019). CS224N assignment 2 implements a neural transition-based parser.

## <a id="exercises-and-local-references"></a>Exercises and local references

- The five [CS336 assignments](https://cs336.stanford.edu/spring2025/) build, in order, a BPE tokenizer and transformer language model with its training loop, systems optimizations, a scaling-law study, a data pipeline, and supervised and reinforcement-learning post-training for mathematical reasoning; each comes with tests and a handout.
- The four CS224N assignments, linked from the [course homepage](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/), cover word vectors, a neural dependency parser, attention-based translation, and pretraining and fine-tuning a small transformer.
- The [CMU 11-711 code repository](https://github.com/cmu-l3/anlp-fall2025-code) has short notebooks for each lecture, including decoding, fine-tuning, retrieval, and reinforcement learning with language models.
- Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and [minbpe](https://github.com/karpathy/minbpe) are minimal, readable implementations of a GPT training loop and a BPE tokenizer.
- Foundations supplies probability and information theory for perplexity and cross-entropy; ML supplies logistic regression and EM; DL supplies the transformer, self-supervised learning, and training at scale; AI supplies search for decoding, hidden Markov models, and conditional random fields.

## <a id="connections-to-later-modules"></a>Connections to later modules

Diffusion language models, multimodal models, and the generative models behind images and audio belong in Generative AI. The policy-gradient methods that optimize the objectives of blocks 11–12 belong in RL. Interpretability of language models, evaluations for dangerous capabilities, the failure modes of preference training, and the theory of alignment belong in Safety and Frontier. Tokenization, language modeling, pretraining, scaling, adaptation, reasoning, retrieval, and evaluation remain in this module.
