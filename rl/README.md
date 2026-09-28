[Background Notes](../README.md)

# Reinforcement Learning

Reinforcement learning studies agents that learn to act from the consequences of their actions: an agent observes, acts, receives rewards, and must find behavior that maximizes the rewards it will collect, without being told what the right actions are. It builds on the probability, optimization, and linear algebra of [Foundations](../foundations/README.md), on the supervised learning and function approximation of [Machine Learning](../ml/README.md) and [Deep Learning](../dl/README.md), on the decision theory, filtering, game trees, Monte Carlo tree search, and game theory of [Artificial Intelligence](../ai/README.md), on the preference learning and reasoning of [NLP and LLMs](../nlp-llms/README.md), and on the generative models of [Generative AI](../generative-ai/README.md). The module is about twice the size of the others: 31 chapters in two parts follow the topics of the [reading plan](reading-plan.md), and 17 labs apply them to working agents.

## <a id="chapters"></a>Chapters

**Part I — Classical reinforcement learning.**

| Chapter | Main content | Reading plan |
| --- | --- | --- |
| [1. Markov Decision Processes](01-markov-decision-processes.md) | The reward hypothesis; Markov chains and reward processes; returns and discounting; policies, value functions, and the Bellman equations; occupancy measures; optimal values and why deterministic Markov policies suffice; what discounts, rewards, and time limits specify; partial observability, previewed. | [Topic 1](reading-plan.md#1-markov-decision-processes) |
| [2. Dynamic Programming](02-dynamic-programming.md) | Policy evaluation by iteration and by linear algebra; the policy improvement theorem; policy iteration as Newton's method; value iteration, error bounds, and asynchronous updates; generalized policy iteration; linear programming; finite horizons and average reward; complexity and the curse of dimensionality. | [Topic 2](reading-plan.md#2-dynamic-programming) |
| [3. Multi-Armed Bandits](03-multi-armed-bandits.md) | Regret; ε-greedy, optimism, and explore-then-commit; UCB1, KL-UCB, and the Lai–Robbins bound; minimax regret; Thompson sampling and its regret; gradient bandits; nonstationarity; best-arm identification. | [Topic 3](reading-plan.md#3-multi-armed-bandits) |
| [4. Contextual, Bayesian, and Adversarial Bandits](04-contextual-bayesian-and-adversarial-bandits.md) | The bandit as a belief MDP and the Gittins index; adversarial bandits and Exp3; contextual bandits, LinUCB, and linear Thompson sampling; off-policy evaluation and learning from logs. | [Topic 4](reading-plan.md#4-contextual-bayesian-and-adversarial-bandits) |
| [5. Monte Carlo Methods](05-monte-carlo-methods.md) | First-visit and every-visit prediction; blackjack; exploring starts and ε-soft policies; importance sampling, ordinary and weighted, and its variance; the role of the Markov property. | [Topic 5](reading-plan.md#5-monte-carlo-methods) |
| [6. Temporal-Difference Learning](06-temporal-difference-learning.md) | The TD error and bootstrapping; bias and variance of targets; why TD is often faster than Monte Carlo; stochastic approximation and step sizes; afterstates; TD-Gammon; TD errors and dopamine. | [Topic 6](reading-plan.md#6-temporal-difference-learning) |
| [7. Model-Free Control](07-model-free-control.md) | SARSA, Q-learning, and Expected SARSA, with their convergence; cliff walking and on-policy caution; maximization bias and double Q-learning; exploration, step sizes, and time limits in practice. | [Topic 7](reading-plan.md#7-model-free-control) |
| [8. Multi-Step Bootstrapping and Eligibility Traces](08-multi-step-bootstrapping-and-eligibility-traces.md) | n-step returns and how to choose n; λ-returns and the forward view; eligibility traces and the backward view; true online TD(λ); SARSA(λ) and Q(λ). | [Topic 8](reading-plan.md#8-multi-step-bootstrapping-and-eligibility-traces) |
| [9. Off-Policy Learning](09-off-policy-learning.md) | Importance sampling for returns and its products of ratios; control variates and doubly robust estimators; the curse of horizon; tree backup, Q(σ), and a general form of off-policy traces; Retrace and its descendants. | [Topic 9](reading-plan.md#9-off-policy-learning) |
| [10. Planning and Learning with Tabular Models](10-planning-and-learning-with-tabular-models.md) | Models and Dyna; wrong models and exploration bonuses; prioritized sweeping; trajectory sampling and real-time dynamic programming; rollout algorithms, sparse sampling, and Monte Carlo tree search. | [Topic 10](reading-plan.md#10-planning-and-learning-with-tabular-models) |
| [11. Value Function Approximation](11-value-function-approximation.md) | The prediction objective; gradient Monte Carlo and semi-gradient TD; linear methods and the TD fixed point; features, tile coding, and step sizes; least-squares TD and policy iteration; control with approximation and average reward. | [Topic 11](reading-plan.md#11-value-function-approximation) |
| [12. The Deadly Triad and Gradient-TD Methods](12-the-deadly-triad-and-gradient-td-methods.md) | Baird's counterexample and the deadly triad; the geometry of value functions; residual gradients and why the Bellman error is not learnable; gradient-TD methods and emphatic TD; fitted value iteration and its error bounds. | [Topic 12](reading-plan.md#12-the-deadly-triad-and-gradient-td-methods) |
| [13. Policy Gradient and Actor-Critic Methods](13-policy-gradient-and-actor-critic-methods.md) | Parameterized policies and the policy gradient theorem; REINFORCE and baselines; actor–critic methods with traces; compatible features and the natural policy gradient; Gaussian and deterministic policies. | [Topic 13](reading-plan.md#13-policy-gradient-and-actor-critic-methods) |
| [14. Partially Observable Environments](14-partially-observable-environments.md) | Agent states and memory; POMDPs and value functions over beliefs; exact and point-based value iteration; online planning with POMCP; finite-state controllers; learned memory and beliefs; Bayes-adaptive MDPs; Gaussian beliefs, the Kalman filter, and the separation of estimation and control. | [Topic 14](reading-plan.md#14-partially-observable-environments) |
| [15. Optimal Control and Trajectory Optimization](15-optimal-control-and-trajectory-optimization.md) | The linear–quadratic regulator and the Riccati recursion; LQR as a laboratory for RL; shooting, collocation, DDP, and iLQR; the cross-entropy method and MPPI; model predictive control, stability, and constraints; where control and RL meet. | [Topic 15](reading-plan.md#15-optimal-control-and-trajectory-optimization) |

**Part II — Modern reinforcement learning.**

| Chapter | Main content | Reading plan |
| --- | --- | --- |
| [16. Deep Q-Learning](16-deep-q-learning.md) | DQN: fitted Q-iteration with networks, experience replay, and target networks; measuring progress on Atari; overestimation and double DQN; dueling networks; the deadly triad in practice and the pathologies of deep value learning. | [Topic 16](reading-plan.md#16-deep-q-learning) |
| [17. Distributional Reinforcement Learning](17-distributional-reinforcement-learning.md) | The distributional Bellman equation and its contraction; categorical and quantile representations; C51, QR-DQN, and IQN; why distributions help; risk-sensitive decisions and CVaR. | [Topic 17](reading-plan.md#17-distributional-reinforcement-learning) |
| [18. Data-Efficient and Scalable Value-Based Agents](18-data-efficient-and-scalable-value-based-agents.md) | Prioritized replay, multi-step returns, noisy networks, and Rainbow; distributed replay, R2D2, and Agent57; Atari 100k and data efficiency; replay ratios, resets, plasticity, and scaling; evaluating agents. | [Topic 18](reading-plan.md#18-data-efficient-and-scalable-value-based-agents) |
| [19. Deep Actor-Critic and Distributed RL](19-deep-actor-critic-and-distributed-rl.md) | A2C and A3C; generalized advantage estimation; rollouts, truncation, and termination; policy lag and V-trace; IMPALA and SEED; simulation on accelerators; OpenAI Five. | [Topic 19](reading-plan.md#19-deep-actor-critic-and-distributed-rl) |
| [20. Trust Regions and Proximal Policy Optimization](20-trust-regions-and-proximal-policy-optimization.md) | The performance difference lemma and conservative policy iteration; TRPO and the natural gradient; PPO's clipped objective and what it does; the implementation details that matter; MPO and regularized MDPs; PPO as the default. | [Topic 20](reading-plan.md#20-trust-regions-and-proximal-policy-optimization) |
| [21. Continuous Control and Maximum-Entropy RL](21-continuous-control-and-maximum-entropy-rl.md) | Deterministic policy gradients and DDPG; overestimation and TD3; the maximum-entropy objective and soft actor–critic; control as inference and linearly solvable MDPs; ensembles, high update-to-data ratios, and normalization; benchmarks and reproducibility. | [Topic 21](reading-plan.md#21-continuous-control-and-maximum-entropy-rl) |
| [22. Exploration in Deep RL](22-exploration-in-deep-rl.md) | Why dithering fails; counts, pseudo-counts, and random network distillation; curiosity and the noisy TV; episodic novelty; posterior sampling, bootstrapped DQN, and randomized priors; Go-Explore; exploration without rewards. | [Topic 22](reading-plan.md#22-exploration-in-deep-rl) |
| [23. Model-Based RL and World Models](23-model-based-rl-and-world-models.md) | Why model errors compound; probabilistic ensembles, PETS, and MBPO; latent world models and the Dreamer agents; value-equivalent models and TD-MPC; transformer, diffusion, and interactive world models; when models help. | [Topic 23](reading-plan.md#23-model-based-rl-and-world-models) |
| [24. Planning with Learned Models](24-planning-with-learned-models.md) | AlphaGo, AlphaGo Zero, and AlphaZero; MuZero and its extensions; what visit counts approximate; Gumbel search and policy improvement with few simulations. | [Topic 24](reading-plan.md#24-planning-with-learned-models-alphazero-and-muzero) |
| [25. Imitation Learning and Inverse RL](25-imitation-learning-and-inverse-rl.md) | Behavior cloning and compounding errors; DAgger; inverse RL, feature matching, and maximum-entropy IRL; GAIL, AIRL, and IQ-Learn; expressive policies and learning from video; rewards from human preferences. | [Topic 25](reading-plan.md#25-imitation-learning-and-inverse-rl) |
| [26. Offline Reinforcement Learning](26-offline-reinforcement-learning.md) | Why off-policy algorithms fail offline; policy constraints, conservative Q-learning, and implicit Q-learning; model-based offline RL; pessimism and coverage; sequence models and expressive policies; evaluation and offline-to-online learning. | [Topic 26](reading-plan.md#26-offline-reinforcement-learning) |
| [27. Multi-Agent RL and Self-Play](27-multi-agent-rl-and-self-play.md) | Solution concepts and the dynamics of learning in games; counterfactual regret minimization and poker; self-play, populations, PSRO, and leagues; cooperative MARL and centralized training; conventions, social dilemmas, and negotiation; self-play for language models. | [Topic 27](reading-plan.md#27-multi-agent-rl-and-self-play) |
| [28. Reinforcement Learning for Language Models and Reasoning](28-reinforcement-learning-for-language-models-and-reasoning.md) | Generation as a decision problem and the KL-regularized objective; from PPO to critic-free methods such as GRPO; normalizations, importance ratios, and entropy; learned, verifiable, and process rewards; reasoning models, search, and multi-turn agents; reward hacking. | [Topic 28](reading-plan.md#28-reinforcement-learning-for-language-models-and-reasoning) |
| [29. Generalist Agents, Meta-RL, and Open-Endedness](29-generalist-agents-meta-rl-and-open-endedness.md) | Meta-RL as Bayes-adaptive control; RL², MAML, and task inference; in-context RL and discovered algorithms; goal-conditioned and zero-shot RL; generalist and pretrained agents; robot foundation models; novelty, quality diversity, and environment generation. | [Topic 29](reading-plan.md#29-generalist-agents-meta-rl-and-open-endedness) |
| [30. The Theory of Reinforcement Learning](30-the-theory-of-reinforcement-learning.md) | Sample complexity with a generative model; regret and PAC bounds, optimism, and posterior sampling; linear MDPs and the limits of realizability; complexity measures; offline data and pessimism; convergence of policy optimization. | [Topic 30](reading-plan.md#30-the-theory-of-reinforcement-learning) |
| [31. Reinforcement Learning in the Real World](31-reinforcement-learning-in-the-real-world.md) | Reward design and potential-based shaping; reward hacking; goals and hindsight relabeling; options and hierarchies; constrained MDPs, safe exploration, and robustness; sim-to-real transfer, domain randomization, and adaptation; deployed systems and the challenges of real-world RL. | [Topic 31](reading-plan.md#31-reinforcement-learning-in-the-real-world) |

Part I builds the subject from its foundations. Chapters 1–2 define the problem and solve it with a known model; chapters 3–4 isolate exploration in bandits; chapters 5–10 learn from experience without a model, by Monte Carlo, temporal differences, multi-step and off-policy methods, and planning with learned tabular models; chapters 11–13 add function approximation and policy gradients, with the instabilities that come with them; and chapters 14–15 treat hidden state and continuous control. Part II follows deep RL to the current frontier: value-based agents (chapters 16–18), actor-critic and policy optimization methods (19–21), exploration (22), models and planning (23–24), learning from demonstrations and fixed data (25–26), many agents (27), language models (28), and generalist agents (29), and it closes with the theory that explains what is possible (30) and the practice of deploying RL in the world (31). Proofs and longer derivations appear in collapsed appendices at the end of each chapter, and every chapter ends with exercises whose solutions are in collapsed callouts.

## <a id="labs"></a>Labs

The labs are larger projects, each a page in the `Labs` folder with instructions, expected results, commentary on what the results show, and extensions, and a reference solution in `Labs/code` that prints the expected results on a laptop CPU.

| Lab | What you build | Chapters | Time |
| --- | --- | --- | --- |
| [1. Dynamic Programming on FrozenLake and Taxi](labs/lab-01-dynamic-programming-on-frozenlake-and-taxi.md) | Exact models from Gymnasium environments, solved by policy and value iteration and checked by simulation. | 1–2 | ½ min |
| [2. Bandit Algorithms in Practice](labs/lab-02-bandit-algorithms-in-practice.md) | A library of bandit algorithms, stress tests, a contextual bandit from real data, and off-policy evaluation. | 3–4 | 1 min |
| [3. Tabular Control with Monte Carlo and TD Methods](labs/lab-03-tabular-control-with-monte-carlo-and-td-methods.md) | Monte Carlo, SARSA, Q-learning, and double Q-learning on four environments with known optimal policies. | 5–7 | 4 min |
| [4. Planning with Learned and Given Models](labs/lab-04-planning-with-learned-and-given-models.md) | Dyna, rollouts, Monte Carlo tree search, and real-time dynamic programming. | 10 | 1½ min |
| [5. Tile Coding and Linear Control on Mountain Car](labs/lab-05-tile-coding-and-linear-control-on-mountain-car.md) | Tile coding, SARSA(λ), true online SARSA(λ), and least-squares policy iteration. | 8, 11 | 3 min |
| [6. Policy Gradient and Actor-Critic Methods on CartPole](labs/lab-06-policy-gradient-and-actor-critic-methods-on-cartpole.md) | REINFORCE estimators, an online actor–critic, GAE, and the natural policy gradient. | 13 | 2½ min |
| [7. System Identification, LQR, and Model Predictive Control](labs/lab-07-system-identification-lqr-and-model-predictive-control.md) | A learned linear model with LQR, and model predictive control of a pendulum with given and learned models. | 15 | 2 min |
| [8. Deep Q-Networks on CartPole and MinAtar](labs/lab-08-deep-q-networks-on-cartpole-and-minatar.md) | DQN from scratch, its ablations and overestimation, and double DQN on a miniature Atari game. | 16 | 25 min |
| [9. Advantage Actor-Critic with Parallel and Stale Actors](labs/lab-09-advantage-actor-critic-with-parallel-and-stale-actors.md) | A2C, policy lag, V-trace, and clipping. | 19 | 4 min |
| [10. Proximal Policy Optimization from Scratch](labs/lab-10-proximal-policy-optimization-from-scratch.md) | PPO on LunarLander with ablations, and a Gaussian policy for HalfCheetah. | 20 | 45 min |
| [11. Off-Policy Actor-Critics for Continuous Control](labs/lab-11-off-policy-actor-critics-for-continuous-control.md) | DDPG, TD3, and SAC on a pendulum and on HalfCheetah. | 21 | 40 min |
| [12. Model-Based RL with Learned Ensembles](labs/lab-12-model-based-rl-with-learned-ensembles.md) | PETS with probabilistic ensembles, and MBPO. | 23 | 15 min |
| [13. AlphaZero and Gumbel Search from Scratch](labs/lab-13-alphazero-and-gumbel-search-from-scratch.md) | AlphaZero on tic-tac-toe and Connect Four, and Gumbel search with few simulations. | 24 | 17 min |
| [14. Imitation and Offline Reinforcement Learning](labs/lab-14-imitation-and-offline-reinforcement-learning.md) | Behavior cloning, DAgger, TD3+BC, and implicit Q-learning on datasets of different quality. | 25–26 | 25 min |
| [15. Regret Minimization and Self-Play in Poker](labs/lab-15-regret-minimization-and-self-play-in-poker.md) | Leduc poker with exact exploitability; CFR, CFR+, and Monte Carlo CFR; regularized self-play and PSRO. | 27 | 3 min |
| [16. Reinforcement Learning for a Small Language Model](labs/lab-16-reinforcement-learning-for-a-small-language-model.md) | A small transformer fine-tuned with REINFORCE, RLOO, GRPO, and Dr. GRPO against a verifier. | 28 | 25 min |
| [17. Safe and Robust Reinforcement Learning](labs/lab-17-safe-and-robust-reinforcement-learning.md) | Lagrangian PPO, a model-based shield, and sim-to-real transfer with domain randomization and RMA. | 31 | 16 min |

Times are for the reference solutions; the labs from 8 on use two cores. Chapters without a lab of their own (9, 12, 14, 17, 18, 22, 29, and 30) have code blocks and experimental exercises that play the same role.

## <a id="shared-conventions"></a>Shared conventions

- A Markov decision process has states $`s\in\mathcal S`$, actions $`a\in\mathcal A`$, dynamics $`p(s'\mid s,a)`$ (or $`p(s',r\mid s,a)`$ with the reward), expected rewards $`r(s,a)`$, and a discount $`\gamma\in[0,1)`$; episodic tasks may use $`\gamma=1`$. The return from time $`t`$ is $`G_t=\sum_{k\ge0}\gamma^kR_{t+k+1}`$, and $`\pi(a\mid s)`$ is a policy, with parameters $`\theta`$ when parameterized.
- Part I follows Sutton and Barto's notation: $`v_\pi`$ and $`q_\pi`$ for the values of a policy, $`v_*`$ and $`q_*`$ for the optimal values, and $`\mathbf w`$ for the weights of an approximate value function $`\hat v(s,\mathbf w)`$. The deep RL and theory chapters follow their literature: $`V^\pi`$, $`Q^\pi`$, $`V^*`$, $`Q^*`$, transition kernels $`P(s'\mid s,a)`$, and the advantage $`A^\pi=Q^\pi-V^\pi`$. The two notations mean the same things. Finite-horizon settings use $`H`$ steps per episode and $`K`$ episodes.
- The discounted occupancy measure of a policy from a start distribution $`\mu`$ is written $`d^\pi_\mu`$ (normalized by $`1-\gamma`$ where stated), and on-policy state distributions under approximation are written $`\mu_\pi`$.
- Regret is the total shortfall of the rewards or values collected against an optimal policy or arm; bounds written $`\tilde O(\cdot)`$ ignore logarithmic factors.
- Each code block runs on its own on a CPU with NumPy, SciPy, PyTorch, and Gymnasium, and implements its method from scratch. Seeds are fixed, and the comment lines at the end of a block record what it printed in the environment described in the [computing setup](sources/computing-setup.md). Environments are small enough that exact answers are often known, so results illustrate the chapters' claims; they are not benchmarks.

## <a id="examples-and-supporting-resources"></a>Examples and supporting resources

The chapter text contains the definitions, derivations, and worked examples, and every figure is generated by a script in `Sources/Figure code`. The [computing setup](sources/computing-setup.md) records the Python environment and how to run the code, the labs, and the figure scripts, and [figure sources](sources/figure-sources.md) records the construction behind each figure.

The [reading plan](reading-plan.md) lists the lectures and readings for each topic. [Course links](sources/course-links.md) collects the courses it draws on, [video links](sources/video-links.md) the lecture recordings by chapter, [books and documentation](sources/book-and-documentation-links.md) the reference texts, libraries, and tutorials, and [papers](sources/paper-links.md) the principal research behind each chapter. Beyond the module's own labs, the [CS285 homeworks](https://rail.eecs.berkeley.edu/deeprlcourse-fa23/) and [Spinning Up](https://spinningup.openai.com/) are the best further exercises, and [CleanRL](https://docs.cleanrl.dev/) is the best reference for complete, benchmarked implementations.

## <a id="connections-to-later-modules"></a>Connections to later modules

**Safety and frontier** research starts from problems this module documents: rewards that are misspecified and proxies that are hacked ([chapter 31](31-reinforcement-learning-in-the-real-world.md#when-the-proxy-is-optimized), [chapter 28](28-reinforcement-learning-for-language-models-and-reasoning.md#reward-hacking-and-oversight)), open-ended systems that generate their own objectives ([chapter 29](29-generalist-agents-meta-rl-and-open-endedness.md#open-endedness-with-foundation-models)), agents that learn in the world and must stay within constraints while they do ([chapter 31](31-reinforcement-learning-in-the-real-world.md#safe-exploration)), and the language models trained with human feedback and verifiable rewards ([chapter 28](28-reinforcement-learning-for-language-models-and-reasoning.md)). It adds goal misgeneralization, the risks of agents that pursue goals in the world, and the oversight and evaluation of systems trained with reinforcement learning.

## Labs

- [Lab 1. Dynamic Programming on FrozenLake and Taxi](labs/lab-01-dynamic-programming-on-frozenlake-and-taxi.md)
- [Lab 2. Bandit Algorithms in Practice](labs/lab-02-bandit-algorithms-in-practice.md)
- [Lab 3. Tabular Control with Monte Carlo and TD Methods](labs/lab-03-tabular-control-with-monte-carlo-and-td-methods.md)
- [Lab 4. Planning with Learned and Given Models](labs/lab-04-planning-with-learned-and-given-models.md)
- [Lab 5. Tile Coding and Linear Control on Mountain Car](labs/lab-05-tile-coding-and-linear-control-on-mountain-car.md)
- [Lab 6. Policy Gradient and Actor-Critic Methods on CartPole](labs/lab-06-policy-gradient-and-actor-critic-methods-on-cartpole.md)
- [Lab 7. System Identification, LQR, and Model Predictive Control](labs/lab-07-system-identification-lqr-and-model-predictive-control.md)
- [Lab 8. Deep Q-Networks on CartPole and MinAtar](labs/lab-08-deep-q-networks-on-cartpole-and-minatar.md)
- [Lab 9. Advantage Actor-Critic with Parallel and Stale Actors](labs/lab-09-advantage-actor-critic-with-parallel-and-stale-actors.md)
- [Lab 10. Proximal Policy Optimization from Scratch](labs/lab-10-proximal-policy-optimization-from-scratch.md)
- [Lab 11. Off-Policy Actor-Critics for Continuous Control](labs/lab-11-off-policy-actor-critics-for-continuous-control.md)
- [Lab 12. Model-Based RL with Learned Ensembles](labs/lab-12-model-based-rl-with-learned-ensembles.md)
- [Lab 13. AlphaZero and Gumbel Search from Scratch](labs/lab-13-alphazero-and-gumbel-search-from-scratch.md)
- [Lab 14. Imitation and Offline Reinforcement Learning](labs/lab-14-imitation-and-offline-reinforcement-learning.md)
- [Lab 15. Regret Minimization and Self-Play in Poker](labs/lab-15-regret-minimization-and-self-play-in-poker.md)
- [Lab 16. Reinforcement Learning for a Small Language Model](labs/lab-16-reinforcement-learning-for-a-small-language-model.md)
- [Lab 17. Safe and Robust Reinforcement Learning](labs/lab-17-safe-and-robust-reinforcement-learning.md)

## Reading plan and sources

- [Reading plan](reading-plan.md)
- [Book and documentation links](sources/book-and-documentation-links.md)
- [Computing setup](sources/computing-setup.md)
- [Course links](sources/course-links.md)
- [Figure sources](sources/figure-sources.md)
- [Paper links](sources/paper-links.md)
- [Video links](sources/video-links.md)
