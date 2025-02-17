---
{"dg-publish":true,"permalink":"/Course Notes/Deep Reinforcement Learning/Notes/","created":"2024-04-01T08:10:58.475-04:00"}
---

### some definitions
- markov decision process (MDP)
    - $S$: set of **states**
    - $O$: set of **observations**
    - $A$: set of **actions**
    - $T(s,a,s')$: transition function
    - $R(s,a,s')$: reward function
    - markov: depend only on current state
- fully/partial observable
    - partial observable markov decision process (POMDP)
- policy: $\pi:S\rightarrow A$
    - optimal policy: $\pi^*$
    - sometimes use $\pi(a| s)$ to describe a action distribution
- discount factor
    - utility(sum of rewards) ${U}([r_0,r_1,r_2,...])=r_0+\gamma r_1+\gamma^2r_2+\cdots$
- value of a state $s$: $V^*(s)$ expected utility starting in $s$ and act optimally  **$s$ 出发能获得的最大收益（$\gamma$ 衰减）**
- Q value: $Q^*(s,a)$ expected utility taking action $a$ from $s$ and then act optimally  **$s$ 出发，第一步做 action $a$ 能获得的最大收益**
- $V^*(s) = \max_a Q^*(s, a)$
- $Q^{*}(s,a)=\sum_{s'}T(s,a,s^{\prime})[R(s,a,s^{\prime})+\gamma V^{*}(s^{\prime})]$
### value iteration
> to calculate $V^*$ and the corresponding policy
- define $V_k(s)$ 表示 $s$ 出发走 $k$ 步能获得的最大收益，然后动态规划
- $V_{\mathrm{k}+1}(s)\leftarrow\operatorname*{max}_{a}\sum_{s^{\prime}}T(s,a,s^{\prime})[R(s,a,s^{\prime})+\gamma V_{\mathrm{k}}(s^{\prime})]$
- repeat until convergence
    - 策略会更早收敛
- 复杂度每个 iteration $O(S^2A)$ 
- 问题
    - slow
    - policy converges much earlier than value，做了很多多余的
### policy iteration
> jointly optimize policy and value
- $V^{\pi}(s)$: $s$ 的 value under policy $\pi$
- $V^{\pi}(s) = \sum_{s'} T(s,\pi(s),s') [R(s,\pi(s),s') + \gamma V^{\pi}(s')]$
- 注意 $T$ 中的随机性
- steps
    - 1. policy evaluation $O(S^2)$ each iteration
        - or 直接解方程
    - 2. policy improvement
        - $\pi'(s) = \arg \max \sum_{s'} T(s,a,s') [R(s,a,s') + \gamma V^{\pi}(s')]$
        - use previous value (not optimal) to update policy
- repeat until policy converges
- converge faster
- 问题
    - what if we don't know $T$
    - need to learn from environment
        - => **Reinforcement Learning**

- Model-free vs Model-based
    - Model-based: learn the MDP transition
    - Model-free: learn value or policy w/o learn the transition
### Monte-Carlo Policy Evaluation (model free)
> learn from experience
- 对于一串 experience $s_1, a_1, r_1, s_2, a_2, r_2, \cdots, s_k \sim \pi$ 
    - under policy $\pi$ 表示按照 $\pi$ sample 得到
    - return $G_t = r_{t+1} + \gamma r_{t+2} + \cdots$
        - 从当前节点开始往后的 reward
    - $v_\pi (s)=\mathbb{E}_\pi[G_t\mid S_t=s]$
- 根据 experience 估计 $v_\pi (s)$
    - monte-carlo tree search
    - 对于每一次搜索，对于所有路径上的节点
        - N (s) = N (s) + 1
        - S (s) = S (s) + G_t
        - V (s) = S (s) / N (s)
    - 当搜索次数 趋于无穷时，V (s) 趋于 $V_\pi (s)$
- good
    - simple
    - only from experience
- bad
    - too long time
    - all episodes must terminate
    - 没用到 MDP 的性质
### Temporal Difference Learning
- simplest TD (0)
    - $V (S_t)\leftarrow V (S_t)+\alpha (R_{t+1}+\gamma V (S_{t+1})-V (S_t))$
        - Here $R_{t+1} + \gamma V (S_{t+1})$ can be seen as gt value of this try: **TD Target**
        - pull toward it
        - $\delta_t = R_{t+1}+\gamma V (S_{t+1})-V (S_t)$ is called **TD error**
##### MC vs TD
- MC
    - correct expectation, high variance (randomly choose each step)
    - 每一次 search 都搜到底，不需要靠自己的 $V$，no bootstrap
- TD
    - can update for each step
    - use own $V$ for update
    - biased, but bias $\rightarrow$ 0
    - lower variance
![Pasted image 20240417135254.png|625](/img/user/Course%20Notes/Deep%20Reinforcement%20Learning/assets/Pasted%20image%2020240417135254.png)
- which to choose
    - bad initialization, choose MC
    - value is already good, choose TD
##### n-step return
- TD (n)
    - $G_t^{(n)}=R_{t+1}+\gamma R_{t+2}+\ldots+\gamma^{n-1}R_{t+n}+\gamma^nV (S_{t+n})$
    - $V (S_t)\leftarrow V (S_t)+\alpha\left (G_t^{(n)}-V (S_t)\right)$
    - larger $n$ means lower bias but higher variance
##### combine MC and TD (n): TD ($\lambda$)
- $G_t^\lambda=(1-\lambda)\sum_{n=1}^\infty\lambda^{n-1}G_t^{(n)}$
- weighted sum of TD (n)
- but $TD (\lambda)$ need full sequence, 这样我们做 TD 的意义就没了
##### backward TD ($\lambda$)
- 由于每一步的误差其实都跟之前的 choice 有关，把误差传回去
- $\begin{aligned}E_0 (s)&=0\\E_t (s)&=\gamma\lambda E_{t-1}(s)+\mathbf{1}(S_t=s)\end{aligned}$
    - 其实就是  $E_t (s_i) = (\gamma \lambda) ^ {t-i}$
$$\begin{aligned}\delta_t&=R_{t+1}+\gamma V (S_{t+1})-V (S_t)\\V (s)&\leftarrow V (s)+\alpha\delta_tE_t (s)\end{aligned}$$
- 对于 $V (s_t)$ 就是 $V (s_t) \leftarrow V (s_t)+\alpha \delta_t$
- 往前每一级乘上一个 $\gamma \lambda$ 的系数
![Pasted image 20240417143258.png|525](/img/user/Course%20Notes/Deep%20Reinforcement%20Learning/assets/Pasted%20image%2020240417143258.png)
- **After one whole episode, forward == backward**
- **but in the middle maybe not**

> The previous MC and TD can help us estimate value of a fixed policy $\pi$. But how to find $\pi$? Need something to **update policy**

- Recall policy update
    - we use $\arg \max_a$, and iteratively update value and policy
    - to get best action easier, we choose to update $Q$ value
- explore or exploit?
### $\epsilon$ - Greedy 
$$
\pi(a\mid s)=\begin{cases}\epsilon/m+1-\epsilon&\mathrm{~if~}a^*=\operatorname{argmax}Q(s,a)\\\epsilon/m&\mathrm{~otherwise}&\end{cases}
$$
- $1-\epsilon$ choose best action, $\epsilon$ choose randomly
##### GLIE (Greedy in the Limit with Infinite Exploration)
- 一个性质，指当实验次数 $k \rightarrow \infty$ 时
    - 每个 (state, action) pair 都被 exploit 了无数次
    - $\pi_k (\arg\max_a \mid s) \rightarrow 1$
- Ex.  $\epsilon_k=1/k$ 的 $\epsilon$ - greedy 就是 GLIE
### Q-learning
- TD update + $\epsilon$ - greedy
1. start with random $Q$ value
2. 当前状态 s，take action a sampled from $\epsilon$ -greedy, 到达 s'
3. update Q
    $Q(s,a ) = Q(s,a) + \alpha (R(s') + \gamma \max_{a'} Q(s', a') - Q(s,a))$
- it's off-policy
    - not depend on next action, 只需要 (s, a, s')
##### State-Action-Reward-State-Action (SARSA)
-  $Q_{n+1}(s_n,a_n)=Q_n(s_n,a_n)+\alpha_n[r(s_n,a_n)+\lambda Q_n(s_{n+1},a_{n+1})-Q_n(s_n,a_n)]$
- 用 $s_{n+1}, a_{n+1}$ 来更新，而不是 $\max_{a'} Q (s', a')$
- on-policy: need $a_{n+1}$
- problems
    - state, action too many
    - only cope with discrete scenes
### Deep RL
- Use neural networks & parameters to replace $V (s)$ and $Q (s, a)$
- Use loss function and gradient descent instead of discrete update
- mean squared error: $J(\mathbf{w})=\mathbb{E}_\pi\left[(q_\pi(S,A)-\hat{q}(S,A,\mathbf{w}))^2\right]$
### Deep Q-Network (DQN)
- 几乎和 Q-learning 相同
- TD update:
$$
\mathcal{L}_i(w_i)=\left(r+\gamma\max_{a^{\prime}}Q{\left(s^{\prime},a^{\prime};w_i\right)}-Q(s,a;w_i)\right)^2
$$
- Problems
    - 在一段时间内，遇到的状态可能都差不多，容易导致 NN overfit，陷入一个 local 的状况不出去
        - **experience replay**: 存储游戏历史，每次 update 从整个 storage 里 sample
    - Q value 的极小变化可能导致 policy 的完全改变（policy 可能不连续），policy 的改变又会导致 data distribution 短时间变化巨大
        - 主要影响 update 的时候的 $\max_a$，我们要保证这个稳定
        - 用旧的参数去计算  $\max$: $\mathcal{L}_i(w_i)=\mathbb{E}_{s,a,r,s^{\prime}\sim\mathcal{D}_i}\left[\left(r+\gamma\max_{a^{\prime}}Q\left(s^{\prime},a^{\prime};w_i^-\right)-Q(s,a;w_i)\right)^2\right]$
        - 称作 **target network**
    - reward 太大可能导致 gradient 太大，参数的 update 不稳定
        - **clip**
- DQN
    - TD update + exp replay + target network + clip
- Problems
    - Overestimation
        - TD update 用的是 TD (0)，可能导致 bias 太大（overestimate）
        - 你的 policy 并不是每次都能选最好的 action
        - solution
            - 用 current Q network 选择 action，用 old Q network 计算 Q value
    - 使用 replay buffer 的效率太低
        - 大部分都是没啥用的 state
        - solution
            - 按照 DQN error 给定权重，优先使用 error 大的历史来 update
    - split DQN into two channels (use **advantage**)
        - $V (s)$ + $A (s, a)$
        - $A$ 用来衡量在这个状态下不同 action 之间谁更好
            - $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$
            - 都是负的，最好的那个 action 是零，其他的是和最好的 action 之间的差距
        - 我觉得主要作用在解释性强
    - n-step return
### Policy Gradient
- parameterize the policy:  $\pi_\theta(s,a)=\mathbb{P}[a\mid s,\theta]$
- can cope with continuous action space
- how to evaluate policy (objective function) (average reward per step)
$$J_{avR}(\theta)=\sum_sd^{\pi_\theta}(s)\sum_a\pi_\theta(s,a)\mathcal{R}_s^a$$
- $d^{\pi_\theta}(s)$ is stationary distribution of Markov chain for $\pi_\theta$
- then simply calculate gradient and update $\theta$
    - $\Delta\theta=\alpha\nabla_\theta J(\theta)$
- unable to calculate gradient, because we cannot get gradient from the world or the reward
    - make the world differentiable
    - find another way to calculate gradient
- $$
\nabla_\theta\mathbb{E}_\tau[R(\tau)] = \mathbb{E}_\tau\left[R(\tau)\nabla_\theta\sum_{t=0}^{T-1}\log\pi(a_t\mid s_t,\theta)\right]
$$
1. sample a trajectory from $\pi_\theta$
2. calculate $J (\theta)$ and $\nabla J (\theta)$ according to above equation
3. update $\theta$

- 类似于 MC，方差比较大（每次都需要一整条 trajectory）
    - 注意到 $\mathbb{E}_\tau\left[B\cdot \nabla_\theta\sum_{t=0}^{T-1}\log\pi (a_t\mid s_t,\theta)\right] = 0$
    - 这是因为相当于 $B \cdot \mathbb{E}_\tau\left[\nabla_\theta\sum_{t=0}^{T-1}\log\pi(a_t\mid s_t,\theta)\right]$ 右边变形之后相当于对所有 trajectory 的概率之和求梯度，但这个概率和必然为 1，梯度必然等于 0
    - 我们加入一个 bias 项来减少梯度 $\mathbb{E}_\tau\left[(R (\tau)-B(S))\nabla_\theta\sum_{t=0}^{T-1}\log\pi (a_t\mid s_t,\theta)\right]$
        - 取 $B (S) = V (s)$ 就很好
    - 另一种方法：reduce variance
        - reward 只算 $t'=t \sim T$ 的部分，因为 $t<t'$ 的部分不会有贡献
- PG is on-policy，我们需要根据当前的 $\pi$ 来 sample trajectory
##### Off-policy PG
- importance sampling
- $E_p[f(x)] = E_q[f(x) \frac {p(x)} {q(x)}]$
-  $$\nabla _ {\theta'} J(\theta') = E_{\tau\sim p_\theta(\tau)}{\left[\sum_{t=1}^T\nabla_{\theta^{\prime}}\log\pi_{\theta^{\prime}}(\mathbf{a}_t\mid\mathbf{s}_t){\left(\prod_{t^{\prime}=1}^t\frac{\pi_{\theta^{\prime}}(\mathbf{a}_{t^{\prime}}\mid\mathbf{s}_{t^{\prime}})}{\pi_\theta(\mathbf{a}_{t^{\prime}}\mid\mathbf{s}_{t^{\prime}})}\right)}{\left(\sum_{t^{\prime}=t}^Tr(\mathbf{s}_{t^{\prime}},\mathbf{a}_{t^{\prime}})\right)}\right]}$$
### Actor-Critic
- Bias set to Value $V (s)$
- Use TD to update value (together train policy & value)
- **advantage = Q - V**
1. take action $a \sim \pi_\theta (a|s)$, get $(s, a, s', r)$ and store in $R$
2. sample a batch $\{s_i, a_i, s'_i, r_i\}$ from $R$ 
3. update $Q$ using targets $y_i = r_i + \gamma Q (s_i', a_i')$ for each data
4. sample new $a_i^\pi$ according to current $\pi$, and $\nabla_\theta J(\theta)\approx\frac1N\sum_i\nabla_\theta\log\pi_\theta(\mathbf{a}_i^\pi\mid\mathbf{s}_i)\hat{Q}^\pi(\mathbf{s}_i,\mathbf{a}_i^\pi)$
5. update $\theta$
- problems
    - unstable
        - large step -> bad policy, bad data
        - small step -> too slow
##### DDPG
- learn $\mu_\theta$ (action) to deal with continuous action space
##### TD3
- overestimation
    - use 2 Q, get min of them
- unstable
    - update interval
    - delayed policy updates
- add noise to smooth the policy

### Proximal Policy Optimization
$$\mathrm{maximize}\quad\hat{\mathbb{E}}_t{\left[\frac{\pi_\theta (a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}\left (a_t\mid s_t\right)}\hat{A}_t\right]}-\beta\hat{\mathbb{E}}_t{\left[\mathrm{KL}[\pi_{\theta_{\mathrm{old}}}\left (\cdot\mid s_t\right),\pi_\theta (\cdot\mid s_t)]\right]}$$







### Model-based RL
> when we know or can simulate the environment
> know $T (s, a, s')$
- open-loop: not using future feedbacks, 在开始前决定好 action 序列
- closed-loop: 每一步根据这一步转移的结果，决定下一步的 action（only in Stochastic scenarios）
#### open-loop planning
- $f (s, a)=s'$，deterministic world
$$
\mathbf{a}_1,\ldots,\mathbf{a}_T=\arg\max_{\mathbf{a}_1,\ldots,\mathbf{a}_T}\sum_{t=1}^Tr(\mathbf{s}_t,\mathbf{a}_t)\mathrm{~s.t.~}\mathbf{s}_{t+1}=f(\mathbf{s}_t,\mathbf{a}_t)
$$
##### random shooting
- guess $\{a_i\}$, check the reward
- find the best one
##### cross-entropy method
- 用参数 $w$ 控制选择 action 的 策略
- 例如：状态是一个 向量 $S$，我们用一个向量 $w$ 表示我们的 policy，根据 $S\cdot w$ 的结果选择做哪一个 action
- 我们控制用来 sample $w$ 的参数，例如方差均值等
- 每次 sample N 组，选取 top-k 的 w，根据他们的方差和均值 update 
- problems
    - 如果维数太大，或者 action space 太大等等
    - 只能做 open-loop
        - 无法应对扰动
        - 无法根据反馈调整策略
#### closed-loop control
##### MCTS
- 根据当前策略找到最优的叶子结点，并且通过随机模拟扩展
- Go
##### Linear Quadratic Regulator
- 解决线性问题
- $s_{t+1} = As_t + Ba_t$
- $cost(s_t, u_t) = s_t^TQ s_t + u_t ^ T R u_t$
- cost 和 reward 本质是等价的
- 用类似 value iteration 的方式，每次更新应该用的 action 都可以直接计算出来
    - 快，closed-loop
    - 但是不能解决非线性问题，并且无法应对环境扰动
#### model is unknown
- easiest algorithm
    - use random policy to explore, collect data $(s, a, s')$
    - use collected data to learn a model $f (s, a)$
    - use random algorithm to plan with $f$
- 在 world 比较复杂的时候显然不行
- improve
    - 在 plan 的时候，把新获得的数据加入 dataset，用于训练 $f$
##### MPC
- change to closed-loop
![Pasted image 20240423231757.png|750](/img/user/Course%20Notes/Deep%20Reinforcement%20Learning/assets/Pasted%20image%2020240423231757.png)
- 由于每次把 $(s, a, s')$ 放进了 dataset，我们的 environment 就知道了 $f (s, a)=s'$，那么在 plan 的时候就可以用这个信息
##### model-based RL with policy gradient
1. run base policy to collect $D$
2. learn $f (s, a)$
3. use $f (s, a)$ and $\pi_\theta$ to generate trajectories and do policy gradient
4. run $\pi_\theta$ and collect new data
- problems
    - $f (s, a)$ biased 导致 $\pi$ overfit
    - long trajectories will accumulate error
###### to solve the biased problem
- uncertainty
    - We are uncertain about the real world
- measure uncertainty?
- **model ensemble**
    - use multiple models and see if they agree with each other
    - 初始化和 sgd 本身的不确定性就能够提供 different models
    - Huazhe's work: model ensemble 本身是在 smooth value function
###### to solve accumulated error problem
- use short trajectories
- Ideal loss (**value equivalence**)
    - 我们用 MSE loss 是为了模拟真实环境
    - 但是我们是否需要全部的环境？我们只需要针对我们的任务去得到部分的环境和 dynamics 就可以
    - 怎么区分哪些和我们的 task 相关?
        - 我们的环境下的 value function 与真实环境下 value function 接近
        - ideal loss: $|V^{\pi, M} - V^{\pi, M^*}|$
### imitation learning
> learn from demos
- demos: $\{o_t, a_t\}$
- **behavior cloning**
    - learn a policy $\pi_\theta(a|o)$ to maximize $\pi (a_t|o_t)$
- problems
    - 每个 scene 下都会有 small error, accumulate 会造成大问题
    - cannot work in unknown scenes (even cannot get it back to known ones)
##### better data collection
- DAgger: dataset aggregation
    - 类似之前的想法，边做边把新的数据（以及人类关于这个数据的 gt action）加到 dataset 中
    - 问题：
        - 直接按照一个并不好的 policy 在现实中做并不安全
        - 需要人随时看着准备接手
- Dataset Resampling/Reweighting
    - 不同的数据重要性不同，给数据一个权重
    - 用 error 给定权重
- Data Augmentation
- 更好的收集数据的设备
- Pre-training inverse dynamics models
    - 用一个小数据集训练一个 $f (s_t, s_{t+1})=a_t$：用来根据 observation 推测 action
    - 用大量的视频数据推测 action 之后，加入到数据集中
##### better input (提供更多的信息)
- 如果只提供一帧的 image 不够，我们可以提供连续的 images，并且可以用 RNN 等来处理这个序列信息
- confusion
    - 人踩刹车亮刹车灯的例子
- multi-view inputs
##### better model （更好的 representation, loss）
- Time-Contrastive Networks (TCN)
    - multi-view images
    - 让同一帧的不同角度的 image 的 representation 尽量接近
    - 时间上相距较远的 image 的 representation 尽量 far away
- R3M 
    - 除了 TCN 的 loss 之外，再加上一个 image to text 的 loss，用来提取语义信息
- Diffusion policy
- solve multi-modal problem
    - Xu: 开车，相同场景可以向左拐也可以向右拐
    - 传统的 MSE loss: 直接往前撞墙
    - Solutions
        - CE loss，分类问题
        - mixture of gaussian instead of one single gaussian
        - use a latent variable to control the mode
        - autoregressive discretization
            - 把 action 离散化
##### Deep Q-learning from demos (DofD)
- 要求其他 action 对应的 value 一定显式地比 expert 的策略 action 的 value 低一个 margin

