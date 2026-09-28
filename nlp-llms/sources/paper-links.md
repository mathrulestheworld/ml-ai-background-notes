[ML Mastery Notes](../../README.md) › [NLP and Large Language Models](../README.md)

# Paper links

# <a id="papers"></a>Papers

The notes cite the original papers where their ideas arise. This index collects the principal ones by topic; the chapters give further references inline.

## <a id="tokenization-and-n-gram-models"></a>Tokenization and n-gram models

| Work | Idea developed in the module |
| --- | --- |
| Gage, [*A New Algorithm for Data Compression*](https://dl.acm.org/doi/10.5555/177910.177914) (1994), and Sennrich, Haddow, and Birch, [*Neural Machine Translation of Rare Words with Subword Units*](https://arxiv.org/abs/1508.07909) (2016) | Byte-pair encoding, from a compression algorithm to subword vocabularies for translation (chapters 1 and 15). |
| Schuster and Nakajima, [*Japanese and Korean Voice Search*](https://doi.org/10.1109/ICASSP.2012.6289079) (2012) | WordPiece, which chooses merges by the likelihood of the training data (chapter 1). |
| Kudo, [*Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates*](https://arxiv.org/abs/1804.10959) (2018) | The unigram language model tokenizer and the sampling of segmentations (chapter 1). |
| Kudo and Richardson, [*SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing*](https://arxiv.org/abs/1808.06226) (2018) | BPE and unigram tokenizers trained directly on raw text, with spaces as an ordinary symbol (chapter 1). |
| Petrov et al., [*Language Model Tokenizers Introduce Unfairness between Languages*](https://arxiv.org/abs/2305.15425) (2023) | Token counts for the same text that differ by up to a factor of 15 across languages (chapter 1). |
| Xue et al., [*ByT5: Towards a Token-Free Future with Pre-Trained Byte-to-Byte Models*](https://arxiv.org/abs/2105.13626) (2022) | A transformer trained directly on UTF-8 bytes, without a tokenizer (chapter 1). |
| Shannon, [*A Mathematical Theory of Communication*](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x) (1948), and [*Prediction and Entropy of Printed English*](https://doi.org/10.1002/j.1538-7305.1951.tb01366.x) (1951) | Approximations to English of increasing order, and the entropy of English estimated from human guessing (chapter 2). |
| Good, [*The Population Frequencies of Species and the Estimation of Population Parameters*](https://doi.org/10.1093/biomet/40.3-4.237) (1953) | Good–Turing estimation from the count of counts (chapter 2). |
| Katz, [*Estimation of Probabilities from Sparse Data for the Language Model Component of a Speech Recognizer*](https://doi.org/10.1109/TASSP.1987.1165125) (1987) | Backoff with Good–Turing discounting (chapter 2). |
| Kneser and Ney, [*Improved Backing-Off for M-Gram Language Modeling*](https://doi.org/10.1109/ICASSP.1995.479394) (1995), and Chen and Goodman, [*An Empirical Study of Smoothing Techniques for Language Modeling*](https://doi.org/10.1006/csla.1999.0128) (1999) | Kneser–Ney smoothing with continuation counts, and its modified form with three discounts (chapter 2). |

## <a id="word-embeddings"></a>Word embeddings

| Work | Idea developed in the module |
| --- | --- |
| Harris, [*Distributional Structure*](https://doi.org/10.1080/00437956.1954.11659520) (1954) | The distributional hypothesis (chapter 3). |
| Church and Hanks, [*Word Association Norms, Mutual Information, and Lexicography*](https://aclanthology.org/J90-1003/) (1990) | Pointwise mutual information between words and their contexts (chapter 3). |
| Mikolov et al., [*Efficient Estimation of Word Representations in Vector Space*](https://arxiv.org/abs/1301.3781) (2013), and [*Distributed Representations of Words and Phrases and Their Compositionality*](https://arxiv.org/abs/1310.4546) (2013) | Word2vec: the skip-gram model and negative sampling (chapter 3). |
| Levy and Goldberg, [*Neural Word Embedding as Implicit Matrix Factorization*](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization) (2014) | Skip-gram with negative sampling factorizes a shifted PMI matrix (chapter 3). |
| Levy, Goldberg, and Dagan, [*Improving Distributional Similarity with Lessons Learned from Word Embeddings*](https://aclanthology.org/Q15-1016/) (2015) | With matched design choices, count-based and predictive vectors perform alike (chapter 3). |
| Pennington, Socher, and Manning, [*GloVe: Global Vectors for Word Representation*](https://aclanthology.org/D14-1162/) (2014) | An explicit weighted factorization of log co-occurrence counts (chapter 3). |
| Bojanowski et al., [*Enriching Word Vectors with Subword Information*](https://arxiv.org/abs/1607.04606) (2017) | FastText: word vectors built from character n-grams (chapter 3). |
| Bolukbasi et al., [*Man Is to Computer Programmer as Woman Is to Homemaker? Debiasing Word Embeddings*](https://arxiv.org/abs/1607.06520) (2016) | Social biases in embeddings, and projecting out a gender direction (chapter 3). |

## <a id="transformer-language-models"></a>Transformer language models

| Work | Idea developed in the module |
| --- | --- |
| Bengio et al., [*A Neural Probabilistic Language Model*](https://www.jmlr.org/papers/v3/bengio03a.html) (2003) | The first neural language model: learned word vectors and a feedforward network over a fixed window (chapters 2 and 4). |
| Radford et al., [*Improving Language Understanding by Generative Pre-Training*](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (2018) | GPT: a pretrained transformer decoder fine-tuned for downstream tasks (chapters 4 and 5). |
| Radford et al., [*Language Models Are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019) | GPT-2: byte-level BPE, and tasks performed without fine-tuning (chapters 1 and 4). |
| Brown et al., [*Language Models Are Few-Shot Learners*](https://arxiv.org/abs/2005.14165) (2020) | GPT-3: 175 billion parameters, 300 billion tokens, and few-shot prompting (chapters 4, 6, and 9). |
| OpenAI, [*GPT-4 Technical Report*](https://arxiv.org/abs/2303.08774) (2023) | Calibration of the pretrained model, and final loss predicted from much smaller runs (chapters 4 and 7). |
| Touvron et al., [*Llama 2: Open Foundation and Fine-Tuned Chat Models*](https://arxiv.org/abs/2307.09288) (2023) | An open reference architecture, and post-training with rejection sampling, PPO, and separate reward models (chapters 4 and 11). |
| Llama Team, [*The Llama 3 Herd of Models*](https://arxiv.org/abs/2407.21783) (2024) | The tokenizer, document masking, and 15.6-trillion-token data mixture of a current open model (chapters 1, 4, and 6). |
| Shazeer et al., [*Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*](https://arxiv.org/abs/1701.06538) (2017), and Fedus, Zoph, and Shazeer, [*Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*](https://arxiv.org/abs/2101.03961) (2022) | Mixture-of-experts layers, top-$`k`$ routing, and the load-balancing loss (chapter 4). |
| DeepSeek-AI, [*DeepSeek-V3 Technical Report*](https://arxiv.org/abs/2412.19437) (2024) | Many small experts, balancing without an auxiliary loss, and latent attention at scale (chapter 4). |
| Chen et al., [*Extending Context Window of Large Language Models via Positional Interpolation*](https://arxiv.org/abs/2306.15595) (2023) | Extending rotary position embeddings beyond the training length (chapter 4). |
| Liu et al., [*Lost in the Middle: How Language Models Use Long Contexts*](https://arxiv.org/abs/2307.03172) (2023) | Information in the middle of a long context is used least (chapter 4). |

## <a id="pretraining-and-transfer"></a>Pretraining and transfer

| Work | Idea developed in the module |
| --- | --- |
| Peters et al., [*Deep Contextualized Word Representations*](https://arxiv.org/abs/1802.05365) (2018) | ELMo: contextual word vectors from bidirectional LSTM language models (chapters 3 and 5). |
| Howard and Ruder, [*Universal Language Model Fine-Tuning for Text Classification*](https://arxiv.org/abs/1801.06146) (2018) | ULMFiT: fine-tuning a whole pretrained language model, first on the target domain and then on the task (chapter 5). |
| Devlin et al., [*BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding*](https://arxiv.org/abs/1810.04805) (2019) | Masked language modeling and the bidirectional encoder (chapter 5). |
| Liu et al., [*RoBERTa: A Robustly Optimized BERT Pretraining Approach*](https://arxiv.org/abs/1907.11692) (2019) | Longer training on more data, without next-sentence prediction (chapter 5). |
| Clark et al., [*ELECTRA: Pre-Training Text Encoders as Discriminators Rather than Generators*](https://arxiv.org/abs/2003.10555) (2020) | Replaced-token detection, a training signal at every position (chapter 5). |
| Raffel et al., [*Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*](https://arxiv.org/abs/1910.10683) (2020) | T5: span corruption, the text-to-text format, and the C4 corpus (chapters 5 and 6). |
| Wang et al., [*GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding*](https://arxiv.org/abs/1804.07461) (2018) | The benchmark on which pretrained encoders were compared, and its rapid saturation (chapter 5). |
| Reimers and Gurevych, [*Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks*](https://arxiv.org/abs/1908.10084) (2019) | Sentence embeddings from BERT fine-tuned as a bi-encoder (chapter 5). |
| Tenney, Das, and Pavlick, [*BERT Rediscovers the Classical NLP Pipeline*](https://arxiv.org/abs/1905.05950) (2019) | Probing which layers encode which linguistic properties (chapter 5). |
| Linzen, Dupoux, and Goldberg, [*Assessing the Ability of LSTMs to Learn Syntax-Sensitive Dependencies*](https://arxiv.org/abs/1611.01368) (2016) | Subject–verb agreement across intervening nouns as a test of syntax (chapters 5 and 16). |

## <a id="pretraining-data"></a>Pretraining data

| Work | Idea developed in the module |
| --- | --- |
| Gao et al., [*The Pile: An 800GB Dataset of Diverse Text for Language Modeling*](https://arxiv.org/abs/2101.00027) (2020) | A curated mixture of 22 sources (chapter 6). |
| Penedo et al., [*The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only*](https://arxiv.org/abs/2306.01116) (2023), and [*The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*](https://arxiv.org/abs/2406.17557) (2024) | Filtered and deduplicated web text alone, with every step of the pipeline ablated (chapter 6). |
| Rae et al., [*Scaling Language Models: Methods, Analysis & Insights from Training Gopher*](https://arxiv.org/abs/2112.11446) (2021) | Heuristic quality rules that remain a common template (chapter 6). |
| Lee et al., [*Deduplicating Training Data Makes Language Models Better*](https://arxiv.org/abs/2107.06499) (2022) | Deduplication reduces memorization and training cost (chapter 6). |
| Carlini et al., [*Quantifying Memorization across Neural Language Models*](https://arxiv.org/abs/2202.07646) (2023) | Memorization grows log-linearly with duplication and with model size (chapter 6). |
| Broder, [*On the Resemblance and Containment of Documents*](https://doi.org/10.1109/SEQUEN.1997.666900) (1997) | MinHash estimates of Jaccard similarity for near-duplicate detection (chapter 6). |
| Xie et al., [*DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining*](https://arxiv.org/abs/2305.10429) (2023) | Domain weights learned with a small proxy model (chapter 6). |
| Muennighoff et al., [*Scaling Data-Constrained Language Models*](https://arxiv.org/abs/2305.16264) (2023) | The value of repeated data, and a scaling law for several epochs (chapters 6 and 7). |
| Shumailov et al., [*AI Models Collapse When Trained on Recursively Generated Data*](https://www.nature.com/articles/s41586-024-07566-y) (2024) | Model collapse under recursive training on generated text (chapter 6). |

## <a id="scaling-laws"></a>Scaling laws

| Work | Idea developed in the module |
| --- | --- |
| Hestness et al., [*Deep Learning Scaling Is Predictable, Empirically*](https://arxiv.org/abs/1712.00409) (2017) | Test loss as a power law in the amount of data, across domains (chapter 7). |
| Kaplan et al., [*Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361) (2020) | Power laws in parameters, data, and compute for transformer language models (chapters 4 and 7). |
| Hoffmann et al., [*Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556) (2022) | The joint law in model size and data, and compute-optimal allocation (chapter 7). |
| Porian et al., [*Resolving Discrepancies in Compute-Optimal Scaling of Language Models*](https://arxiv.org/abs/2406.19146) (2024) | Why the Kaplan and Chinchilla allocations differ (chapter 7). |
| Besiroglu et al., [*Chinchilla Scaling: A Replication Attempt*](https://arxiv.org/abs/2404.10102) (2024) | A refit of the Chinchilla law from data reconstructed from the paper's figures (chapter 7). |
| Sardana et al., [*Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*](https://arxiv.org/abs/2401.00448) (2024) | Counting inference favors smaller models trained on more data (chapter 7). |
| Yang et al., [*Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer*](https://arxiv.org/abs/2203.03466) (2022) | μP: hyperparameters tuned on a small proxy and transferred to a large model (chapter 7). |
| Wei et al., [*Emergent Abilities of Large Language Models*](https://arxiv.org/abs/2206.07682) (2022), and Schaeffer, Miranda, and Koyejo, [*Are Emergent Abilities of Large Language Models a Mirage?*](https://arxiv.org/abs/2304.15004) (2023) | Abrupt gains with scale, and how much of them the choice of metric creates (chapter 7). |

## <a id="decoding-and-inference"></a>Decoding and inference

| Work | Idea developed in the module |
| --- | --- |
| Wu et al., [*Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation*](https://arxiv.org/abs/1609.08144) (2016) | A production neural translation system and the length penalty for beam search (chapters 8 and 15). |
| Koehn and Knowles, [*Six Challenges for Neural Machine Translation*](https://arxiv.org/abs/1706.03872) (2017) | The beam search curse and the weaknesses of early neural translation (chapters 8 and 15). |
| Stahlberg and Byrne, [*On NMT Search Errors and Model Errors: Cat Got Your Tongue?*](https://arxiv.org/abs/1908.10090) (2019) | Exact search shows that the most probable translation is often the empty string (chapter 8). |
| Holtzman et al., [*The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751) (2020) | Degeneration under maximization, and nucleus (top-$`p`$) sampling (chapter 8). |
| Fan, Lewis, and Dauphin, [*Hierarchical Neural Story Generation*](https://arxiv.org/abs/1805.04833) (2018) | Top-$`k`$ sampling (chapter 8). |
| Willard and Louf, [*Efficient Guided Generation for Large Language Models*](https://arxiv.org/abs/2307.09702) (2023) | Constrained decoding with the allowed tokens precomputed for each automaton state (chapter 8). |
| Leviathan, Kalman, and Matias, [*Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192) (2023), and Chen et al., [*Accelerating Large Language Model Decoding with Speculative Sampling*](https://arxiv.org/abs/2302.01318) (2023) | Speculative decoding: a draft model proposes tokens and the target model verifies them without changing its distribution (chapter 8). |
| Kwon et al., [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180) (2023) | The key–value cache stored in blocks allocated on demand (chapter 8). |

## <a id="in-context-learning-and-prompting"></a>In-context learning and prompting

| Work | Idea developed in the module |
| --- | --- |
| Zhao et al., [*Calibrate Before Use: Improving Few-Shot Performance of Language Models*](https://arxiv.org/abs/2102.09690) (2021) | Majority-label, recency, and common-token biases, and contextual calibration (chapter 9). |
| Lu et al., [*Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity*](https://arxiv.org/abs/2104.08786) (2022) | The sensitivity of few-shot accuracy to the order of demonstrations (chapter 9). |
| Min et al., [*Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?*](https://arxiv.org/abs/2202.12837) (2022), and Wei et al., [*Larger Language Models Do In-Context Learning Differently*](https://arxiv.org/abs/2303.03846) (2023) | What demonstrations convey: format and label space for smaller models, the input–label mapping for larger ones (chapter 9). |
| Xie et al., [*An Explanation of In-Context Learning as Implicit Bayesian Inference*](https://arxiv.org/abs/2111.02080) (2022) | In-context learning as inference over latent concepts (chapter 9). |
| Garg et al., [*What Can Transformers Learn In-Context? A Case Study of Simple Function Classes*](https://arxiv.org/abs/2208.01066) (2022), and von Oswald et al., [*Transformers Learn In-Context by Gradient Descent*](https://arxiv.org/abs/2212.07677) (2023) | Transformers trained on regression prompts, and linear attention layers that perform gradient steps (chapter 9). |
| Olsson et al., [*In-Context Learning and Induction Heads*](https://arxiv.org/abs/2209.11895) (2022) | Induction heads and the phase change in which they form (chapter 9). |
| Chan et al., [*Data Distributional Properties Drive Emergent In-Context Learning in Transformers*](https://arxiv.org/abs/2205.05055) (2022) | Burstiness and many rare classes in the training data produce in-context learning (chapter 9). |

## <a id="fine-tuning-and-adaptation"></a>Fine-tuning and adaptation

| Work | Idea developed in the module |
| --- | --- |
| Wei et al., [*Finetuned Language Models Are Zero-Shot Learners*](https://arxiv.org/abs/2109.01652) (2022) | FLAN: instruction tuning on NLP datasets rewritten as instructions (chapter 10). |
| Wang et al., [*Self-Instruct: Aligning Language Models with Self-Generated Instructions*](https://arxiv.org/abs/2212.10560) (2023) | Instruction data generated by a model from a few seed examples (chapter 10). |
| Zhou et al., [*LIMA: Less Is More for Alignment*](https://arxiv.org/abs/2305.11206) (2023) | Instruction tuning on 1,000 curated examples (chapter 10). |
| Houlsby et al., [*Parameter-Efficient Transfer Learning for NLP*](https://arxiv.org/abs/1902.00751) (2019) | Adapters: small bottleneck networks trained inside a frozen model (chapter 10). |
| Li and Liang, [*Prefix-Tuning: Optimizing Continuous Prompts for Generation*](https://arxiv.org/abs/2101.00190) (2021) | Soft prompts: trained key and value vectors prepended at every layer (chapter 10). |
| Hu et al., [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685) (2022) | Low-rank updates to frozen weight matrices (chapter 10). |
| Aghajanyan, Zettlemoyer, and Gupta, [*Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*](https://arxiv.org/abs/2012.13255) (2021) | Fine-tuning succeeds within a low-dimensional subspace (chapter 10). |
| Dettmers et al., [*QLoRA: Efficient Finetuning of Quantized LLMs*](https://arxiv.org/abs/2305.14314) (2023) | LoRA on a base model stored in 4-bit precision (chapter 10). |
| Ilharco et al., [*Editing Models with Task Arithmetic*](https://arxiv.org/abs/2212.04089) (2023) | Task vectors, added and negated in weight space (chapter 10). |
| Kim and Rush, [*Sequence-Level Knowledge Distillation*](https://arxiv.org/abs/1606.07947) (2016) | Fine-tuning a student on outputs generated by the teacher (chapter 10). |

## <a id="learning-from-preferences"></a>Learning from preferences

| Work | Idea developed in the module |
| --- | --- |
| Christiano et al., [*Deep Reinforcement Learning from Human Preferences*](https://arxiv.org/abs/1706.03741) (2017) | A reward model learned from human comparisons (chapter 11). |
| Stiennon et al., [*Learning to Summarize from Human Feedback*](https://arxiv.org/abs/2009.01325) (2020) | Reinforcement learning from human feedback applied to summarization (chapter 11). |
| Ouyang et al., [*Training Language Models to Follow Instructions with Human Feedback*](https://arxiv.org/abs/2203.02155) (2022) | InstructGPT: demonstrations, a reward model, and PPO for a general assistant (chapters 10, 11, and 14). |
| Bai et al., [*Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback*](https://arxiv.org/abs/2204.05862) (2022) | Pairwise comparisons of helpfulness and harmlessness in dialogue (chapter 11). |
| Schulman et al., [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347) (2017), and Ahmadian et al., [*Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs*](https://arxiv.org/abs/2402.14740) (2024) | PPO, and simpler estimators that replace the value network by the mean reward of several samples (chapter 11). |
| Rafailov et al., [*Direct Preference Optimization: Your Language Model Is Secretly a Reward Model*](https://arxiv.org/abs/2305.18290) (2023) | DPO: a loss on the policy alone, with no separate reward model (chapter 11). |
| Gao, Schulman, and Hilton, [*Scaling Laws for Reward Model Overoptimization*](https://arxiv.org/abs/2210.10760) (2023) | Goodhart's law measured against a gold reward model (chapter 11). |
| Sharma et al., [*Towards Understanding Sycophancy in Language Models*](https://arxiv.org/abs/2310.13548) (2023) | Sycophancy learned from human preferences (chapter 11). |
| Bai et al., [*Constitutional AI: Harmlessness from AI Feedback*](https://arxiv.org/abs/2212.08073) (2022) | Critiques, revisions, and preferences from a model guided by written principles (chapter 11). |

## <a id="reasoning-and-test-time-compute"></a>Reasoning and test-time compute

| Work | Idea developed in the module |
| --- | --- |
| Wei et al., [*Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*](https://arxiv.org/abs/2201.11903) (2022), and Kojima et al., [*Large Language Models Are Zero-Shot Reasoners*](https://arxiv.org/abs/2205.11916) (2022) | Chain-of-thought prompting, with worked demonstrations and with *Let's think step by step* alone (chapter 12). |
| Cobbe et al., [*Training Verifiers to Solve Math Word Problems*](https://arxiv.org/abs/2110.14168) (2021) | GSM8K, and verifiers that rank sampled solutions (chapters 12 and 14). |
| Turpin et al., [*Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting*](https://arxiv.org/abs/2305.04388) (2023) | Chains of thought that omit what drove the answer (chapter 12). |
| Wang et al., [*Self-Consistency Improves Chain of Thought Reasoning in Language Models*](https://arxiv.org/abs/2203.11171) (2023) | Majority voting over sampled chains of thought (chapter 12). |
| Lightman et al., [*Let's Verify Step by Step*](https://arxiv.org/abs/2305.20050) (2023) | Process reward models that score individual steps (chapter 12). |
| Snell et al., [*Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters*](https://arxiv.org/abs/2408.03314) (2024) | Test-time computation allocated by difficulty, traded against model size (chapter 12). |
| Zelikman et al., [*STaR: Bootstrapping Reasoning with Reasoning*](https://arxiv.org/abs/2203.14465) (2022) | Fine-tuning on the model's own successful rationales (chapter 12). |
| Shao et al., [*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*](https://arxiv.org/abs/2402.03300) (2024) | GRPO: policy optimization with rewards normalized within a group of samples (chapter 12). |
| DeepSeek-AI, [*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*](https://arxiv.org/abs/2501.12948) (2025) | Long reasoning learned from verifiable rewards, starting from a base model (chapter 12). |

## <a id="retrieval-tools-and-agents"></a>Retrieval, tools, and agents

| Work | Idea developed in the module |
| --- | --- |
| Robertson and Zaragoza, [*The Probabilistic Relevance Framework: BM25 and Beyond*](https://doi.org/10.1561/1500000019) (2009) | BM25 and sparse retrieval (chapter 13). |
| Karpukhin et al., [*Dense Passage Retrieval for Open-Domain Question Answering*](https://arxiv.org/abs/2004.04906) (2020) | Dense retrieval with separate question and passage encoders (chapter 13). |
| Jégou, Douze, and Schmid, [*Product Quantization for Nearest Neighbor Search*](https://doi.org/10.1109/TPAMI.2010.57) (2011), and Malkov and Yashunin, [*Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs*](https://arxiv.org/abs/1603.09320) (2020) | Compressed vectors and graph indexes for approximate nearest-neighbor search (chapter 13). |
| Lewis et al., [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401) (2020) | A retriever and a generator trained jointly, with the passage as a latent variable (chapter 13). |
| Borgeaud et al., [*Improving Language Models by Retrieving from Trillions of Tokens*](https://arxiv.org/abs/2112.04426) (2022) | RETRO: retrieval built into the model through cross-attention (chapter 13). |
| Schick et al., [*Toolformer: Language Models Can Teach Themselves to Use Tools*](https://arxiv.org/abs/2302.04761) (2023) | A model that learns by itself when to call tools (chapter 13). |
| Yao et al., [*ReAct: Synergizing Reasoning and Acting in Language Models*](https://arxiv.org/abs/2210.03629) (2023) | Agents that interleave reasoning with actions and observations (chapter 13). |
| Jimenez et al., [*SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*](https://arxiv.org/abs/2310.06770) (2024) | Real repository issues, checked by the repositories' tests, as a benchmark for agents (chapter 13). |
| Greshake et al., [*Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection*](https://arxiv.org/abs/2302.12173) (2023) | Indirect prompt injection (chapter 13). |

## <a id="evaluation"></a>Evaluation

| Work | Idea developed in the module |
| --- | --- |
| Hendrycks et al., [*Measuring Massive Multitask Language Understanding*](https://arxiv.org/abs/2009.03300) (2021), and [*Measuring Mathematical Problem Solving with the MATH Dataset*](https://arxiv.org/abs/2103.03874) (2021) | The MMLU and MATH benchmarks (chapter 14). |
| Chen et al., [*Evaluating Large Language Models Trained on Code*](https://arxiv.org/abs/2107.03374) (2021) | HumanEval and checking code by execution (chapter 14). |
| Liang et al., [*Holistic Evaluation of Language Models*](https://arxiv.org/abs/2211.09110) (2022) | HELM: many metrics beyond accuracy, across many scenarios (chapter 14). |
| Biderman et al., [*Lessons from the Trenches on Reproducible Evaluation of Language Models*](https://arxiv.org/abs/2405.14782) (2024) | Evaluation protocols and shared evaluation code (chapter 14). |
| Zheng et al., [*Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*](https://arxiv.org/abs/2306.05685) (2023) | Language models as judges, and their position and verbosity biases (chapter 14). |
| Chiang et al., [*Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference*](https://arxiv.org/abs/2403.04132) (2024) | A leaderboard fitted to crowdsourced pairwise comparisons (chapter 14). |
| Miller, [*Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations*](https://arxiv.org/abs/2411.00640) (2024) | Standard errors and confidence intervals for benchmark scores (chapter 14). |

## <a id="machine-translation-and-multilingual-models"></a>Machine translation and multilingual models

| Work | Idea developed in the module |
| --- | --- |
| Brown et al., [*The Mathematics of Statistical Machine Translation: Parameter Estimation*](https://aclanthology.org/J93-2003/) (1993) | The IBM alignment models trained by EM (chapter 15). |
| Och and Ney, [*A Systematic Comparison of Various Statistical Alignment Models*](https://aclanthology.org/J03-1002/) (2003) | The GIZA++ cascade of alignment models (chapter 15). |
| Koehn, Och, and Marcu, [*Statistical Phrase-Based Translation*](https://aclanthology.org/N03-1017/) (2003) | Phrase-based translation from word alignments (chapter 15). |
| Sutskever, Vinyals, and Le, [*Sequence to Sequence Learning with Neural Networks*](https://arxiv.org/abs/1409.3215) (2014) | The encoder–decoder introduced for translation (chapter 15). |
| Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017) | The transformer, introduced as a translation model (chapter 15). |
| Sennrich, Haddow, and Birch, [*Improving Neural Machine Translation Models with Monolingual Data*](https://arxiv.org/abs/1511.06709) (2016) | Back-translation (chapter 15). |
| Papineni et al., [*Bleu: A Method for Automatic Evaluation of Machine Translation*](https://aclanthology.org/P02-1040/) (2002) | BLEU: clipped n-gram precisions and a brevity penalty (chapter 15). |
| Rei et al., [*COMET: A Neural Framework for MT Evaluation*](https://arxiv.org/abs/2009.09025) (2020), and Freitag et al., [*Results of WMT22 Metrics Shared Task: Stop Using BLEU – Neural Metrics Are Better and More Robust*](https://aclanthology.org/2022.wmt-1.2/) (2022) | Learned metrics and their closer agreement with expert judgments (chapter 15). |
| Johnson et al., [*Google's Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation*](https://arxiv.org/abs/1611.04558) (2017) | One model for many language pairs, and zero-shot translation (chapter 15). |
| Conneau et al., [*Unsupervised Cross-Lingual Representation Learning at Scale*](https://arxiv.org/abs/1911.02116) (2020) | XLM-R and the curse of multilinguality (chapter 15). |
| NLLB Team, [*No Language Left Behind: Scaling Human-Centered Machine Translation*](https://arxiv.org/abs/2207.04672) (2022) | Translation for 200 languages, including evaluation sets and mined parallel data (chapter 15). |

## <a id="syntactic-parsing"></a>Syntactic parsing

| Work | Idea developed in the module |
| --- | --- |
| Marcus, Santorini, and Marcinkiewicz, [*Building a Large Annotated Corpus of English: The Penn Treebank*](https://aclanthology.org/J93-2004/) (1993) | The treebank on which statistical parsers were trained and compared (chapter 16). |
| Lari and Young, [*The Estimation of Stochastic Context-Free Grammars Using the Inside-Outside Algorithm*](https://www.sciencedirect.com/science/article/pii/088523089090022X) (1990) | Learning a probabilistic grammar from unannotated sentences with EM (chapter 16). |
| Johnson, [*PCFG Models of Linguistic Tree Representations*](https://aclanthology.org/J98-4004/) (1998) | Parent annotation and the independence assumptions of treebank grammars (chapter 16). |
| Collins, [*Head-Driven Statistical Models for Natural Language Parsing*](https://aclanthology.org/J03-4003/) (2003) | Lexicalized probabilistic grammars (chapter 16). |
| Kitaev and Klein, [*Constituency Parsing with a Self-Attentive Encoder*](https://arxiv.org/abs/1805.01052) (2018) | Neural span scores decoded with CKY-style dynamic programming (chapter 16). |
| Nivre et al., [*Universal Dependencies v1: A Multilingual Treebank Collection*](https://aclanthology.org/L16-1262/) (2016) | Dependency annotation that is consistent across languages (chapter 16). |
| Nivre, [*Incrementality in Deterministic Dependency Parsing*](https://aclanthology.org/W04-0308/) (2004) | The arc-standard transition system (chapter 16). |
| Chen and Manning, [*A Fast and Accurate Dependency Parser Using Neural Networks*](https://aclanthology.org/D14-1082/) (2014) | A neural network in place of sparse features in a transition-based parser (chapter 16). |
| McDonald et al., [*Non-Projective Dependency Parsing Using Spanning Tree Algorithms*](https://aclanthology.org/H05-1066/) (2005), and Eisner, [*Three New Probabilistic Models for Dependency Parsing: An Exploration*](https://aclanthology.org/C96-1058/) (1996) | Graph-based parsing: maximum spanning arborescences, and Eisner's algorithm for projective trees (chapter 16). |
| Dozat and Manning, [*Deep Biaffine Attention for Neural Dependency Parsing*](https://arxiv.org/abs/1611.01734) (2017) | The biaffine arc scorer (chapter 16). |
| Hewitt and Manning, [*A Structural Probe for Finding Syntax in Word Representations*](https://aclanthology.org/N19-1419/) (2019) | Tree distances recovered by a linear transformation of contextual word vectors (chapter 16). |
