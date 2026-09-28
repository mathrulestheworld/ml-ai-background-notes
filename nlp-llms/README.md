[Background Notes](../README.md)

# NLP and Large Language Models

Natural Language Processing and Large Language Models studies how machines model, generate, and use human language, from counting words to training, adapting, and evaluating the large language models at the center of current AI. It builds on the probability and information theory of Foundations, on the logistic regression and EM of Machine Learning, on the transformer, self-supervised learning, and training at scale of Deep Learning, and on the search, hidden Markov models, and conditional random fields of Artificial Intelligence. Fourteen core chapters follow the main topics of the reading plan, and two optional chapters cover the classical tasks of translation and parsing.

## <a id="chapters"></a>Chapters

| Chapter | Main content | Reading plan |
| --- | --- | --- |
| 1. Text, Tokens, and Tokenization | Unicode, bytes, and normalization; words, types, and Zipf's and Heaps' laws; byte-pair encoding, WordPiece, and the unigram language model; vocabulary size and compression; numbers, code, and languages; bits per byte; byte-level models. | Topic 1 |
| 2. N-gram Language Models and Perplexity | The chain rule and the Markov assumption; maximum likelihood and sampling; perplexity and cross-entropy; the entropy of English; add-$`k`$, Good–Turing, interpolation, backoff, and Kneser–Ney smoothing; large count-based models; the noisy channel. | Topic 2 |
| 3. Word Embeddings | The distributional hypothesis; co-occurrence matrices, PMI, and truncated SVD; skip-gram with negative sampling as implicit matrix factorization; GloVe and subword vectors; similarity, analogies, and social biases; contextual embeddings. | Topic 3 |
| 4. Transformer Language Models | Neural and recurrent language models; the decoder-only transformer and its training objective; a character model of Shakespeare; weight tying, the softmax bottleneck, and calibration; current architectures and training recipes; mixture-of-experts layers; long contexts. | Topic 4 |
| 5. Pretraining and Transfer | Why next-token prediction teaches so much; causal, masked, replaced-token, and span-corruption objectives; fine-tuning and continued pretraining; sentence embeddings; probing classifiers, minimal pairs, and knowledge in the parameters. | Topic 5 |
| 6. Pretraining Data | Web crawls and other sources; text extraction and language identification; heuristic and model-based filters; exact and near-duplicate removal with MinHash and LSH; mixtures, repetition, and synthetic data; contamination, licensing, and documentation. | Topic 6 |
| 7. Scaling Laws | Power laws in loss; model size and data together; compute-optimal allocation, Kaplan versus Chinchilla, and the reliability of fits; overtraining for inference, limited data, and hyperparameter transfer; predicting downstream performance and emergence. | Topic 7 |
| 8. Decoding and Text Generation | Greedy and beam search and the trouble with the most likely sequence; temperature, top-$`k`$, and nucleus sampling; repetition; constrained generation; speculative decoding; the cost of prefill and decode, batching, and memory. | Topic 8 |
| 9. In-Context Learning and Prompting | Few-shot prompting, scoring, and calibration; sensitivity to the prompt; in-context learning as inference over latent concepts and as learning algorithms in the forward pass; induction heads; instructions, decomposition, and prompt search. | Topic 9 |
| 10. Fine-Tuning and Parameter-Efficient Adaptation | Instruction tuning and its data; what fine-tuning changes, and forgetting; adapters, soft prompts, and LoRA; why low rank suffices; quantized adaptation; task vectors and model merging; distillation. | Topic 10 |
| 11. Learning from Human Preferences | Collecting preferences; Bradley–Terry reward models; the KL-regularized objective, RLHF, and best-of-$`n`$; direct preference optimization and its variants; reward hacking and overoptimization; constitutional AI and feedback from models. | Topic 11 |
| 12. Reasoning and Test-Time Compute | Chain of thought and why intermediate tokens help; faithfulness; self-consistency, verifiers, search, and refinement; scaling test-time compute; expert iteration, reinforcement learning with verifiable rewards and GRPO; reasoning models. | Topic 12 |
| 13. Retrieval, Tools, and Agents | BM25, dense retrieval, and approximate nearest-neighbor search; retrieval-augmented generation and its failure modes; function calling and code execution; the agent loop, memory, and planning; agent benchmarks; compounding errors and prompt injection. | Topic 13 |
| 14. Evaluating Language Models | Likelihood, multiple-choice scoring, and execution; pass@$`k`$; benchmarks, saturation, and contamination; reference-based metrics, human evaluation, and model judges; pairwise ratings with Bradley–Terry and Elo; standard errors, clustering, and paired comparisons. | Topic 14 |
| 15. Machine Translation and Multilingual Models *(optional)* | The noisy channel; word alignment and IBM Model 1 with EM; phrase-based translation; neural translation and back-translation; BLEU, chrF, learned metrics, and expert evaluation; multilingual models, language sampling, the curse of multilinguality, and low-resource languages. | Topic 15 |
| 16. Syntactic Parsing *(optional)* | Sequence labeling; context-free grammars, treebanks, PCFGs, and the CKY and inside–outside algorithms; lexicalized and neural constituency parsers; dependency grammar and Universal Dependencies; transition-based and graph-based parsing with Eisner's algorithm; syntax inside language models. | Topic 16 |

Chapters 1–3 turn text into units a model can use and build the first models of it: tokens, counts, and vectors. Chapters 4–7 build and train the transformer language model: its architecture, its pretraining objectives, the data it learns from, and the laws that govern how it scales. Chapters 8–10 use and adapt a trained model: decoding its output, prompting it, and fine-tuning it. Chapters 11–13 turn it into an assistant that follows preferences, reasons at length, and acts with retrieval and tools, and chapter 14 asks how any of this is measured. The optional chapters cover translation and parsing, the classical tasks from which much of the field's machinery came. Proofs and longer derivations appear in collapsed appendices at the end of each chapter.

## <a id="shared-conventions"></a>Shared conventions

- A text is a sequence of tokens $`x_1,\dots,x_T`$ from a vocabulary $`\mathcal V`$ of size $`V`$; $`x_{<t}`$ is the prefix before position $`t`$, and a language model $`p_\theta(x_t\mid x_{<t})`$ gives the next-token distribution, computed as a softmax of logits $`z_t\in\mathbb R^V`$. Words are written $`w`$ when the units are words, and $`|\,\cdot\,|`$ is a length in tokens unless stated otherwise.
- Losses are cross-entropies in nats when written with $`\ln`$ and in bits with $`\log_2`$; perplexity is $`2`$ to the power of the cross-entropy in bits, and bits per byte or per character normalize by the length of the text rather than the number of tokens.
- For models, $`N`$ is the number of parameters, $`D`$ the number of training tokens, and $`C\approx6ND`$ the training compute in floating-point operations; $`d`$ is the model width and $`L`$ the number of layers.
- A prompt is $`x`$ and a response $`y`$; $`\pi_\theta`$ is a policy being trained, $`\pi_{\mathrm{ref}}`$ a reference model, $`r`$ a reward, and $`\beta`$ the weight of a KL penalty.
- Each code block runs on its own on a CPU with PyTorch, NumPy, and SciPy from the `4. NLP&LLMs` folder, and implements its method from scratch. Blocks that need text use Tiny Shakespeare and the character transformer trained in chapter 4 (data sources); the others generate synthetic data. Seeds are fixed, and the comment lines at the end of a block record what it printed in the environment described in the computing setup. Models have at most about a million parameters, so results illustrate the chapters' claims and are not benchmarks.

## <a id="examples-and-supporting-resources"></a>Examples and supporting resources

The chapter text contains the definitions, derivations, and worked examples, and every figure is generated by a script in `Sources/Figure code`. The computing setup records the Python environment and how to regenerate the figures, data sources describes the corpus and the trained model, and figure sources records the data and construction behind each figure.

The reading plan lists the lectures and readings for each topic. Course links collects the courses it draws on, video links the lecture recordings by chapter, books and documentation the reference texts, libraries, and tools for real models, and papers the principal research behind each chapter. The five [CS336 assignments](https://cs336.stanford.edu/spring2025/) build a tokenizer, a transformer, a scaling-law study, a data pipeline, and post-training for reasoning, and fit chapters 1, 4, 6–7, and 11–12 directly.

## <a id="connections-to-later-modules"></a>Connections to later modules

**Generative AI** extends language modeling to other kinds of data and other generative families. Autoregressive transformers over image, audio, and video tokens reuse chapters 1 and 4; variational autoencoders, normalizing flows, generative adversarial networks, and diffusion models provide alternatives to next-token prediction, including diffusion language models; and multimodal models connect language to images through the contrastive training of chapter 5.

**Reinforcement learning** develops the policy-gradient methods that chapter 11 and chapter 12 apply to language models: REINFORCE, baselines and advantages, PPO, and value functions, together with the exploration problems that GRPO's silent groups illustrate. Agents that act over many steps (chapter 13) are reinforcement-learning agents with language as their action space.

**Safety and frontier** research starts from the failure modes this module documents: reward hacking and overoptimization (chapter 11), unfaithful reasoning (chapter 12), prompt injection and unreliable agents (chapter 13), and the difficulty of measuring capabilities (chapter 14). It adds interpretability of the representations that probes only sample, evaluations of dangerous capabilities, and the theory of alignment.

## Reading plan and sources

- [Reading plan](reading-plan.md)
- [Book and documentation links](sources/book-and-documentation-links.md)
- [Computing setup](sources/computing-setup.md)
- [Course links](sources/course-links.md)
- [Data sources](sources/data-sources.md)
- [Figure sources](sources/figure-sources.md)
- [Paper links](sources/paper-links.md)
- [Video links](sources/video-links.md)
