[Background Notes](../../README.md) › [NLP and Large Language Models](../README.md)

# Figure sources

# <a id="figure-sources"></a>Figure sources

Every figure in the NLP and LLMs chapters is an original plot generated for these notes on 24 September 2026 with NumPy, SciPy, Matplotlib, PyTorch, and scikit-learn; no external artwork or course slides are reproduced. Each chapter's figures come from one script in `Sources/Figure code`, which is the complete record of the construction. The shared module `nlpfig.py` sets the palette, fonts, and export (white background, 200 dpi PNG). The ten figures of chapters 1–6, 8, and 10 are computed from Tiny Shakespeare (`Sources/Data/tinyshakespeare.txt`, 1,115,394 characters from Karpathy's char-rnn repository), and chapters 5, 8, and 10 reuse the character transformer trained in chapter 4 (`Sources/Data/shakespeare-char-transformer.pt`, 4 layers, 4 heads, width 128, context 128). [Data sources](data-sources.md) describes both files. The tables below summarize the data, seeds, and model settings of each figure so that it can be read and regenerated without opening the script; the [computing setup](computing-setup.md) gives the environment and commands.

Seeds refer to `torch.manual_seed` ("torch seed"), `torch.Generator().manual_seed` ("generator seed"), `numpy.random.default_rng` ("NumPy seed"), or Python's `random.Random` ("Python seed"); scikit-learn's `random_state` and the `seed` of SciPy's `kmeans2` are named where they are used, layers whose initialization is not described keep PyTorch's default, and the scripts of chapters 1, 2, and 16 involve no randomness. Only three scripts read command-line arguments. `ch01_tokenization.py` and `ch12_reasoning.py` accept figure numbers (for example `python ch12_reasoning.py 1`), numbered in the order of the figures in the script, which is also their order in the chapter, and regenerate both figures without arguments. `ch04_transformer_lm.py` loads the checkpoint if it exists and otherwise trains the model and saves it; `--retrain` forces training and overwrites the checkpoint. The other scripts ignore arguments and always write all their figures; `ch05_transfer.py`, `ch08_decoding.py`, and `ch10_lora.py` need the checkpoint, so `ch04_transformer_lm.py` must run first. According to their docstrings, `ch01_tokenization.py`, `ch03_embeddings.py`, and `ch07_scaling.py` take about a minute each, `ch02_ngrams.py` about four minutes, training in `ch04_transformer_lm.py` about 20 minutes on 2 CPU cores, `ch05_transfer.py` about eight minutes on 2 CPU cores, `ch08_decoding.py` about two, `ch09_icl.py` about four, `ch10_lora.py` about five, and `ch12_reasoning.py` about ten minutes on two CPU cores for figure 1 and under a minute for figure 2; `ch13_retrieval.py` takes about twenty seconds and the others seconds. Training results are single runs unless an average is stated; they illustrate the claims in the notes and are not benchmarks.

## <a id="1-text-tokens-and-tokenization"></a>1. Text, tokens, and tokenization

Script: `ch01_tokenization.py`.

| Local figure | Data and construction |
| --- | --- |
| [nlp-tok-zipf.png](images/nlp-tok-zipf.png) | Words are the runs of lowercase letters in the lowercased Tiny Shakespeare, with an optional apostrophe suffix (`[a-z]+(?:'[a-z]+)?`): 203,839 tokens of 12,373 types. Left: frequency against rank on log–log axes, with a least-squares line in log–log coordinates fitted to ranks 10–2,000 and the words *the*, *love*, *sword*, and *tempest* labeled. Right: the number of distinct words among the first $`N`$ tokens for 40 logarithmically spaced $`N`$ from 100 to the whole text, a power law fitted for $`N\ge1000`$, and the line $`V=N`$. |
| [nlp-tok-compression.png](images/nlp-tok-compression.png) | Tokenizers trained on the first 90% of the characters and evaluated on the last 10% (111,540 bytes, 30,838 pre-tokens), both split into pre-tokens by a GPT-2-style regular expression (English contractions, and runs of letters, digits, or punctuation with an optional leading space, and whitespace). Byte-level BPE: 8,000 merges learned from pre-token counts with incremental pair counts (ties toward the larger pair of ids), truncated to vocabularies of 256 (bytes only), 300, 400, 500, 700, 1,000, 1,500, 2,000, 3,000, 4,000, 6,000, and 8,256; encoding repeatedly applies the earliest learned merge present in the pre-token. Unigram language model: a seed vocabulary of every substring of up to 10 characters of the training pre-tokens that occurs at least twice, plus the 65 characters; rounds of two EM iterations (expected piece counts by forward–backward over each pre-token's segmentation lattice), each round followed by keeping the 70% most probable pieces (always all characters, at least 300); held-out compression is measured with Viterbi segmentation after every round with at most 9,000 pieces, down to 300. The dashed line is one token per pre-token (3.62 bytes) and the dotted line one token per byte. |

## <a id="2-n-gram-language-models-and-perplexity"></a>2. N-gram language models and perplexity

Script: `ch02_ngrams.py`.

| Local figure | Data and construction |
| --- | --- |
| [nlp-ngram-smoothing.png](images/nlp-ngram-smoothing.png) | Tiny Shakespeare split into the first 80% of the characters for training, the next 10% for tuning, and the last 10% for testing. Character models use every character; word models use the lowercase words (letter runs with an optional apostrophe suffix) and the punctuation marks `. , ! ? ; :` as tokens, with words seen only once in training mapped to `<unk>` in all three parts. Sequences are padded with $`n-1`$ start symbols. Left: the share of test $`n`$-gram occurrences, $`n=1,\dots,6`$, whose $`n`$-gram never occurs in training. Middle and right: held-out cross-entropy of add-$`k`$ smoothing ($`k\in\{1,0.1,0.01,0.001\}`$ chosen on the tuning part), Jelinek–Mercer interpolation of all orders down to the uniform distribution with one weight $`\lambda\in\{0.3,0.5,0.7,0.9\}`$ chosen on the tuning part, and interpolated Kneser–Ney with continuation counts for the lower orders and one discount per order, $`D=n_1/(n_1+2n_2)`$ from the counts of counts. Character models of orders 1–10 in bits per character; word models of orders 1–5 as perplexity on a log scale. The vocabulary size in the uniform base and in add-$`k`$ is the number of training types plus one. |

## <a id="3-word-embeddings"></a>3. Word embeddings

Script: `ch03_embeddings.py`. Both figures use one model: skip-gram with negative sampling on the lowercase words of Tiny Shakespeare (letter runs with an optional apostrophe suffix), keeping the 3,322 words that occur at least five times and dropping the others before pairing; every (word, context) pair within two positions on either side, 100 dimensions, $`k=5`$ negatives per pair from the unigram distribution raised to the power 0.75, word vectors uniform on $`[-0.5/d,0.5/d]`$ and context vectors zero, Adam with learning rate $`3\times10^{-3}`$, and 8 epochs over the shuffled pairs in batches of 4,096 (torch seed 0). For comparison, the script builds PPMI vectors from counts of the same pairs, with the context distribution raised to 0.75, factored by SVD and truncated to $`U_{100}\Sigma_{100}^{1/2}`$.

| Local figure | Data and construction |
| --- | --- |
| [nlp-emb-pmi.png](images/nlp-emb-pmi.png) | For every distinct training pair, the shifted PMI $`\log\bigl(n(w,c)/n(w)\bigr)-\log P_n(c)-\log k`$, with $`n`$ counting training pairs and $`P_n`$ the noise distribution, against the learned dot product of the word vector of $`w`$ and the context vector of $`c`$. Left: the 2,476 pairs seen at least 30 times, with the line $`w\cdot c=\mathrm{PMI}(w,c)-\log k`$. Right: the Pearson correlation of the two within bins of pairs seen 1, 2–4, 5–9, 10–29, 30–99, and 100 or more times. |
| [nlp-emb-map.png](images/nlp-emb-map.png) | 66 hand-chosen words in six categories (*Romeo and Juliet*, kings and nobles, family, body, emotions, time), of which the 65 in the vocabulary are shown (*evening* occurs fewer than five times). Their normalized skip-gram vectors are mapped to the plane by scikit-learn's t-SNE (perplexity 8, cosine metric, PCA initialization, `random_state=0`), and each label is placed at the least crowded of four positions around its point. The script also prints, for skip-gram and PPMI with SVD, the share of the words whose nearest neighbor by cosine similarity is in the same category, and the answers to eight analogies. |

## <a id="4-transformer-language-models"></a>4. Transformer language models

Script: `ch04_transformer_lm.py`.

| Local figure | Data and construction |
| --- | --- |
| [nlp-lm-training.png](images/nlp-lm-training.png) | The 65 characters of Tiny Shakespeare, the first 90% for training and the last 10% for validation. Model: token and learned position embeddings of width 128 (initialized $`\mathcal N(0,0.02^2)`$), four pre-normalization `nn.TransformerEncoderLayer` blocks with 4 heads, a GELU MLP of width 512, no dropout, and a causal mask, then LayerNorm and a bias-free output layer tied to the token embeddings; 818,048 parameters, context 128 (torch seed 0). Training: AdamW (learning rate $`2\times10^{-3}`$, betas $`(0.9,0.95)`$, weight decay 0.1) under PyTorch's `OneCycleLR` with warm-up over the first 5% of steps (which by default also cycles the first beta), gradient clipping at norm 1, and 4,000 steps of 32 random windows of 128 characters; the weights and curves are saved to the checkpoint. Baseline: interpolated Kneser–Ney 6-gram model on the same 90% (continuation counts for lower orders, discount $`n_1/(n_1+2n_2)`$ per order), given at most the five preceding characters within each window. Left: every 200 steps, the mean training loss over the preceding 200 steps and the loss on 128 fixed validation windows (generator seed 123), with Kneser–Ney on the same windows as a dashed line. Right: validation loss by position in 4,096 windows (generator seed 7), for positions 1–8 singly and averaged over 9–16, 17–32, 33–64, and 65–128, each bin plotted at the geometric mean of its ends, for the transformer and Kneser–Ney. |

## <a id="5-pretraining-and-transfer"></a>5. Pretraining and transfer

Script: `ch05_transfer.py`. It loads the chapter 4 checkpoint.

| Local figure | Data and construction |
| --- | --- |
| [nlp-pretrain-transfer.png](images/nlp-pretrain-transfer.png) | Left: *The Tempest*, the last 34,657 characters of Tiny Shakespeare, lies in the held-out 10% that pretraining never saw; its first 24,000 characters are available for training, and its last 10,657 are held out and scored on windows of 128 characters with stride 64. For training sets of the first 1,000, 2,000, 4,000, 8,000, 16,000, and 24,000 characters, the chapter 4 architecture is trained from scratch (torch seed 0) with AdamW at learning rate $`2\times10^{-3}`$ for 600 steps, and the pretrained model is fine-tuned with AdamW at $`10^{-4}`$ for 200 steps; both use weight decay 0.1 and batches of 16 random windows of 128 characters (generator seed equal to the training size), and each point is the lowest held-out loss among the start and checkpoints every 50 steps (from scratch) or 10 steps (fine-tuning). The dashed line is the pretrained model without adaptation. Right: 1,000 lines of the held-out 10% with at least six words, at most 100 characters, and no final colon (which marks speaker names), taken after a shuffle (NumPy seed 0). Each is corrupted by exchanging two adjacent words, replacing one word by a different one of the 200 most frequent lowercase words of the held-out text, shuffling all words (never the identity), or exchanging two adjacent different characters, and pairs left unchanged are dropped. Both versions are scored by their total log-probability between newline characters under the pretrained transformer and under the Kneser–Ney 6-gram model of chapter 4 (at most five characters of context), and the bars give the share of pairs in which the original is more probable. |

## <a id="6-pretraining-data"></a>6. Pretraining data

Script: `ch06_dedup.py`.

| Local figure | Data and construction |
| --- | --- |
| [nlp-data-minhash.png](images/nlp-data-minhash.png) | Tiny Shakespeare split at whitespace and cut into consecutive 80-word passages. 4,000 pairs (Python seed 0): a random passage and a copy in which a fraction uniform on $`[0,0.6]`$ of the word positions, chosen without repetition, is replaced by random words of the corpus. Documents are sets of lowercased word 5-gram shingles hashed with CRC32. MinHash uses 256 hash functions $`h(x)=(ax+b)\bmod(2^{31}-1)`$ with $`a`$ and $`b`$ drawn uniformly (NumPy seed 0), and two signatures are compared position by position. Left: for the 336 pairs with exact Jaccard similarity between 0.4 and 0.6, the root-mean-square error of the estimate from the first $`k`$ hashes, $`k=8,16,\dots,256`$, against the square root of the mean of $`J(1-J)/k`$ over those pairs. Right: the first 112 hashes split into $`b`$ bands of $`r`$ rows for $`(b,r)=(28,4)`$, $`(14,8)`$, and $`(8,14)`$, a pair becoming a candidate when all rows of some band agree; points are candidate rates in bins of Jaccard similarity of width 0.05 with at least 20 pairs, curves are $`1-(1-s^r)^b`$, and the legend gives the threshold $`(1/b)^{1/r}`$. |

## <a id="7-scaling-laws"></a>7. Scaling laws

Script: `ch07_scaling.py`.

| Local figure | Data and construction |
| --- | --- |
| [nlp-scaling-isoflop.png](images/nlp-scaling-isoflop.png) | Inputs with $`M=100{,}000`$ independent Gaussian coordinates of variance $`\lambda_i=i^{-1.6}`$ ($`\alpha=0.6`$) and targets $`y=w\cdot x+\varepsilon`$ with $`w_i\sim\mathcal N(0,1)`$ and noise variance $`10^{-4}`$. A model of size $`N`$ sees the $`N`$ highest-variance coordinates and is fit by ridge regression on $`D`$ samples with the Bayes-optimal penalty (the unrepresented variance plus the noise); its test loss is computed exactly from the spectrum, and every point averages 20 draws of $`w`$ and the data (NumPy seeds 0–19). Left: $`N=8,16,\dots,1024`$ against $`D=16,32,\dots,4096`$ (72 points), loss minus the noise floor, with the viridis colormap. The law $`L=E+AN^{-a}+BD^{-b}`$ is fitted by scanning $`a`$ and $`b`$ over 71 values each from 0.3 to 1.0, solving for $`E`$, $`A`$, $`B`$ by least squares in relative error, discarding fits with a negative coefficient, and keeping the one with the smallest squared error of $`\log L`$. Right: budgets $`C=ND=2^{12}`$, $`2^{14}`$, $`2^{16}`$, and $`2^{18}`$, with $`N`$ a power of 2 from 4 to 1024 such that $`8\le C/N\le16{,}384`$ and the same 20 draws; stars mark the best $`N`$, and a parabola in $`\log N`$ fitted to up to five points around each minimum gives the optimal size, whose growth with $`C`$ is fitted on log–log axes. |

## <a id="8-decoding-and-text-generation"></a>8. Decoding and text generation

Script: `ch08_decoding.py`. It loads the chapter 4 checkpoint.

| Local figure | Data and construction |
| --- | --- |
| [nlp-decoding.png](images/nlp-decoding.png) | The character model of chapter 4, conditioned on at most its last 128 characters. Prompts are the 40 characters at 24 random positions of the held-out 10% (generator seed 0), each continued for 300 characters under 15 settings: greedy; temperature 0.5, 0.7, 0.9, 1.0, and 1.2; top-$`k`$ with $`k=3`$, 10, and 30; top-$`p`$ with $`p=0.5`$, 0.8, and 0.95; and min-$`p`$ with thresholds 0.02, 0.1, and 0.3 relative to the most probable character (sampling generator seed 0 for each setting). The held-out reference is the true 300 characters after each prompt, scored in the same way. Left: the empirical cumulative distribution of the untruncated model's probability of each generated character, for held-out text, sampling at $`\tau=1`$, top-$`p`$ at 0.8, and greedy decoding. Right: for each setting, the share of generated words (runs of lowercase letters) that occur anywhere in Tiny Shakespeare against the model's mean negative log-likelihood of the generated characters in nats. The script also prints the share of word trigrams that repeat an earlier one in the same sample. |

## <a id="9-in-context-learning-and-prompting"></a>9. In-context learning and prompting

Script: `ch09_icl.py`.

| Local figure | Data and construction |
| --- | --- |
| [nlp-icl-regression.png](images/nlp-icl-regression.png) | Prompts of 20 examples $`(x_i,y_i)`$ with $`x_i\sim\mathcal N(0,I_5)`$, $`y_i=w^\top x_i`$ without noise, and a new $`w\sim\mathcal N(0,I_5)`$ for each prompt. The sequence $`x_1,y_1,\dots,x_{20},y_{20}`$ is 40 six-dimensional vectors (an $`x`$ padded with a zero, or a $`y`$ in the last coordinate with zeros elsewhere), embedded by a linear layer plus learned positions and processed by four pre-normalization blocks of width 64 with 4 heads, a GELU MLP of width 256, no dropout, and a causal mask, then LayerNorm and a linear readout at each $`x`$ position that predicts its $`y`$ (torch seed 0). Adam with learning rate $`10^{-3}`$ on the mean squared error, 4,000 steps of 64 prompts (generator seed 1); 2,000 test prompts (generator seed 2). Errors are mean squared errors divided by $`d=5`$. Left: the error for $`y_k`$ given the $`k`$ preceding examples, $`k=0,\dots,19`$, for the transformer, minimum-norm least squares by pseudoinverse, one gradient step from zero with the best step size, $`\hat w=X^\top y/(k+d+1)`$, and the mean target of the three nearest examples (zero at $`k=0`$). Right: test error with 3 and 10 examples every 100 steps, with the least-squares values as dotted lines. |

## <a id="10-fine-tuning-and-parameter-efficient-adaptation"></a>10. Fine-tuning and parameter-efficient adaptation

Script: `ch10_lora.py`. It loads the chapter 4 checkpoint.

| Local figure | Data and construction |
| --- | --- |
| [nlp-finetune-forgetting.png](images/nlp-finetune-forgetting.png) | The chapter 4 model adapted on the first 8,000 characters of *The Tempest* (the last 34,657 characters of the corpus). Learning is measured on the last 10,657 characters of the play and forgetting on the rest of the held-out 10%, the other plays (76,883 characters), both on non-overlapping windows of 128 characters. Full fine-tuning with AdamW without weight decay at learning rates $`10^{-4}`$, $`3\times10^{-4}`$, and $`10^{-3}`$; LoRA at learning rate $`3\times10^{-3}`$ with ranks 1, 4, and 16, $`\alpha=8`$ (scale $`\alpha/r`$), $`A\sim\mathcal N(0,1/n_{\text{in}})`$ and $`B=0`$, attached with `torch.nn.utils.parametrize` to the attention input and output projections and both MLP matrices of all four blocks, all other weights frozen (8,192 trained parameters per unit of rank, against 818,048 for full fine-tuning). Every run starts from torch seed 0 and takes 150 steps of 16 random windows of 128 characters (generator seed 0), with both losses recorded every 10 steps and plotted against each other from the pretrained model (star); the axes are limited to 2.15–2.8 and 1.98–2.9 bits. |

## <a id="11-learning-from-human-preferences"></a>11. Learning from human preferences

Script: `ch11_overoptimization.py`.

| Local figure | Data and construction |
| --- | --- |
| [nlp-rlhf-overoptimization.png](images/nlp-rlhf-overoptimization.png) | Responses are feature vectors $`\phi\sim\mathcal N(0,I_8)`$ under the reference policy, and the gold reward is $`w\cdot\phi+2\phi_0-0.5\phi_0^2`$ with $`w_0=0`$ and $`w_i\sim\mathcal N(0,0.3^2)`$ otherwise (NumPy seed 0). The linear proxy $`\theta\cdot\phi`$ is fitted by Bradley–Terry maximum likelihood (3,000 gradient-ascent steps of size 0.5 from zero) to 2,000 comparisons of pairs of reference samples whose winners are drawn with the Bradley–Terry probability of the gold rewards; its agreement with the gold ranking is measured on 20,000 random pairs of 20,000 reference samples. Best-of-$`n`$ uses the distinct integer parts of 26 logarithmically spaced values from 1 to $`10^5`$, with $`\max(200,\lfloor400{,}000/n\rfloor)`$ trials each and $`\mathrm{KL}=\log n-(n-1)/n`$. The KL-regularized optimum $`\pi_\beta\propto\pi_{\mathrm{ref}}\exp(r/\beta)`$ is computed by reweighting a pool of a million reference samples (NumPy seed 1) for 28 values of $`\beta`$ logarithmically spaced from $`10^{1.5}`$ down to $`10^{-1.2}`$, keeping those with KL below 9 nats so that the pool still resolves the policy. The curves show mean gold (solid) and proxy (dashed) rewards against $`\sqrt{\mathrm{KL}}`$; the dotted line is the gold reward under the reference. |

## <a id="12-reasoning-and-test-time-compute"></a>12. Reasoning and test-time compute

Script: `ch12_reasoning.py`.

| Local figure | Data and construction |
| --- | --- |
| [nlp-reasoning-parity.png](images/nlp-reasoning-parity.png) | Tokens 0 and 1 are bits and 2 is "=". Direct answering: the input is $`x_1,\dots,x_n,{=}`$ and the loss is on the parity predicted at "=". Chain of thought: the sequence $`x_1,\dots,x_n,{=},r_1,\dots,r_n`$ with $`r_k`$ the parity of the first $`k`$ bits, trained with teacher forcing on all of $`r_1,\dots,r_n`$; at test time the model writes its own chain greedily and $`r_n`$ is its answer. A two-layer causal transformer of width 64 with 4 heads, a GELU MLP of width 256, pre-normalization, learned positions, and a two-way output (torch seed 0), trained with Adam at learning rate $`10^{-3}`$ on 128 random strings per step (generator seed 1); accuracy of the final answer on 500 fixed test strings (generator seed 99). Direct answering for $`n=4`$, 8, 12, and 16 over 4,000 steps, evaluated every 100 steps; chain of thought for $`n=16`$ and 32 over 1,000 steps, evaluated every 50. |
| [nlp-reasoning-scaling.png](images/nlp-reasoning-scaling.png) | 2,000 simulated problems with six answers, one of them correct (NumPy seed 0). The probability of the correct answer is drawn from Beta(0.6, 1), and the remaining probability is spread over the five distractors by a Dirichlet(0.5, …, 0.5) draw. The verifier's score for a sample is a systematic value for its answer, 1.5 for the correct one and standard normal for each distractor, fixed per problem, plus independent standard normal noise. For $`N=1,2,4,\dots,256`$ and 1,024 samples per problem, averaged over five repetitions: coverage (some sample correct), majority vote (ties broken at random), best-of-$`N`$ by verifier score, and a vote weighted by the exponentiated scores. The dotted lines are the share of problems whose correct answer is the model's most probable one and the share whose systematic verifier values rank the correct answer first. |

## <a id="13-retrieval-tools-and-agents"></a>13. Retrieval, tools, and agents

Script: `ch13_retrieval.py`.

| Local figure | Data and construction |
| --- | --- |
| [nlp-retrieval-ann.png](images/nlp-retrieval-ann.png) | The synthetic collection of the chapter's code: 500 centers drawn from $`\mathcal N(0,I_{64})`$, and 20,000 vectors and 200 queries, each a random center plus $`\mathcal N(0,I_{64})`$ noise, stored as float32 (NumPy seed 0); a query's true ten nearest neighbors by Euclidean distance define recall@10. Inverted file: SciPy's `kmeans2` with 256 cells (`seed=1`, initial centers drawn from the points). Product quantization with $`M`$ blocks of $`64/M`$ dimensions, each with a 256-code `kmeans2` codebook (`seed` equal to the block index), so $`M`$ bytes per vector; distances by table lookup, optionally followed by exact reranking of the best 100. Left: probing the 1, 2, 4, …, 256 cells nearest the query, recall with exact distances, with 8-byte codes, and with 8-byte codes and reranking, against the mean percentage of the collection scanned. Right: a scan of every vector with codes of 2, 4, 8, 16, and 32 bytes, with and without reranking. |

## <a id="14-evaluating-language-models"></a>14. Evaluating language models

Script: `ch14_evaluation.py`.

| Local figure | Data and construction |
| --- | --- |
| [nlp-eval-statistics.png](images/nlp-eval-statistics.png) | Left: the half-width $`1.96\sqrt{2p(1-p)(1-\rho)/n}`$ of a 95% interval for a difference in accuracy at $`p=0.7`$, for 60 logarithmically spaced $`n`$ from 100 to 100,000 and correlations $`\rho=0`$, 0.5, and 0.8 between the two models' results on the same questions, with guide lines at 1, 2, and 5 points; the script prints the values at 164, 1,319, and 14,042 questions. Right: 12 models with strengths drawn from $`\mathcal N(0,1)`$ on the logit scale (NumPy seed 0); 4,000 random pairs, of which the 3,633 with two different models are kept, each winner drawn with the Bradley–Terry probability. Strengths are fitted by 500 steps of gradient ascent on the Bradley–Terry log-likelihood with slight shrinkage toward zero, centered after each step, and 95% intervals are percentiles of 200 bootstrap resamples of the votes. Estimates, intervals, and centered true strengths are shown on the Elo scale ($`400/\ln10`$ points per logit), sorted by estimate. |

## <a id="15-machine-translation-and-multilingual-models"></a>15. Machine translation and multilingual models

Script: `ch15_translation.py`.

| Local figure | Data and construction |
| --- | --- |
| [nlp-mt-alignment.png](images/nlp-mt-alignment.png) | The toy corpus of the chapter's code: 300 English–Spanish sentence pairs of the form noun phrase, verb, noun phrase (NumPy seed 0), built from 8 nouns with grammatical gender, 6 adjectives, the articles *the* and *a*, and 4 verbs; a noun phrase has an adjective with probability 0.5, placed after the noun in Spanish, where articles and adjectives agree in gender. IBM Model 1 with a NULL word, translation probabilities initialized uniformly over the Spanish vocabulary, and 15 EM updates. After 1, 2, 4, and 15 updates, the panels show the posterior over the English words and NULL for each Spanish word of "un gato rojo ve la mesa grande" given "a red cat sees the big table", on a white-to-blue scale; entries of at least 0.1 are printed. |

## <a id="16-syntactic-parsing"></a>16. Syntactic parsing

Script: `ch16_parsing.py`.

| Local figure | Data and construction |
| --- | --- |
| [nlp-parsing-ambiguity.png](images/nlp-parsing-ambiguity.png) | No data are used. The binary rules of the chapter's PCFG (S → NP VP 1.0, VP → V NP 0.6, VP → VP PP 0.4, NP → Det N 0.8, NP → NP PP 0.2, PP → P NP 1.0) and the lexical rules the sentence needs (Det → *the* 0.7, V → *saw* 0.6, N → *man* 0.3, *dog* 0.3, and *telescope* 0.2, P → *with* 0.6), with the same probabilities as in the chapter. Top: the verb-attachment and noun-attachment trees of "the man saw the dog with the telescope", written as bracketed strings, each with its probability as the product of its rule probabilities, drawn with the words in order and each parent centered over its children; the edges to the PP are orange. Bottom: the dependency tree of the first reading with Universal Dependencies labels (det, nsubj, obj, case, obl, and root), drawn as arcs above the words, and the arc of the second reading, nmod from *dog* to *telescope*, dashed below. |

## <a id="shared-style"></a>Shared style

`nlpfig.py` selects Matplotlib's Agg backend, and its `setup()` sets the font DejaVu Sans (10.5 pt, titles 11.5 pt, DejaVu Sans math text), dark axes of line width 0.8 without top and right spines, dark ticks, frameless legends, and white figure, axes, and saved backgrounds. Its `save()` writes each figure to `Sources/Images` as a 200 dpi PNG with a tight bounding box and 0.08 inches of padding. The palette is BLUE `#2267a5`, ORANGE `#d46a23`, GREEN `#39805a`, PURPLE `#995aa4`, RED `#b32b35`, DARK `#152330` for axes and ticks, GRAY `#586473`, and LIGHT `#c1c8cf`, with light tints of the first five colors defined for fills; the default color cycle is blue, orange, green, purple, red, and gray. Individual scripts set smaller font sizes for titles, annotations, and legends.
