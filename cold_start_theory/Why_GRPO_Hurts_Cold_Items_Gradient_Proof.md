# 从梯度推导解释GRPO后训练为何降低冷门物品推荐准确率

---

## 1. 问题与结论预览

**实验现象**：GRPO后训练在三个数据集上整体优于SFT，但最冷门物品的Hit@5下降15%–24%。

**本文目标**：通过梯度推导，直接证明冷门物品的生成概率在GRPO训练下净减小。

**核心结论**（将在下文严格推导）：

> 在Trie的任一节点上（含 $K$ 个合法后续Token），任一Token $v$ 经过一组GRPO更新后的概率变化为：
>
> $$\Delta\pi_v = \eta\,\pi_v\,(S_v - \bar{S})$$
>
> 其中 $S_v$ 是经过Token $v$ 的候选的净优势之和，$\bar{S} = \sum_k \pi_k S_k$ 是概率加权平均。冷门物品的Token因为 $S_v$ 低而 $\bar{S}$ 被热门Token拉高，导致 $\Delta\pi_v < 0$——**即使没有任何候选探索过该Token**。

---

## 2. 符号与设定

### 2.1 生成式推荐模型

每个物品 $i$ 编码为 $L$ 个Token的序列：$i \leftrightarrow (c_1^i, c_2^i, \ldots, c_L^i)$（本系统中 $L=4$）。所有物品的Token序列构成前缀树（Trie）$\mathcal{T}$。

给定用户历史 $x$，物品 $i$ 被生成的概率是各层条件概率之积：

$$P_\theta(i \mid x) = \prod_{l=1}^L \pi_\theta(c_l^i \mid x, c_{<l}^i) \tag{1}$$

### 2.2 Softmax参数化

在某个Trie节点（状态 $s$），有 $K$ 个合法后续Token $\{v_1,\ldots,v_K\}$，每个Token对应Trie中的一棵**子树**，包含若干物品。模型输出logits $\mathbf{z} = (z_1,\ldots,z_K)$，概率为：

$$\pi_k = \frac{\exp(z_k)}{\sum_{j=1}^K \exp(z_j)}, \quad \sum_{k=1}^K \pi_k = 1 \tag{2}$$

### 2.3 关于"冷热"定义的说明

**"冷"和"热"是item级别的概念**（物品在训练集中的出现频率），而非token级别的。在Trie中：

- **浅层**（第1-2层）：每个Token对应一棵大的子树，包含**各种流行度的物品混合**。一个冷门物品的前缀Token可能与大量热门物品相同。
- **深层**（第3-4层）：每个Token对应的子树越来越小，到最后一层，一个Token基本对应一个具体物品。

因此本分析**不假设**某个Token"是热门Token"或"是冷门Token"。我们直接分析：**一个特定冷门物品 $i_c$ 在Trie各层的Token概率如何变化**。

### 2.4 GRPO采样与优势

对每个训练样本 $(x, i^*)$，生成 $G$ 个Beam Search候选 + 1个Ground Truth (GT)，共 $G{+}1$ 个候选。第 $g$ 个候选的优势：

$$\hat{A}_g = \frac{r_g - \bar{r}}{\sigma_r + \epsilon} \tag{3}$$

**奖励函数（NDCG Reward）**：本系统使用的奖励不是简单的0/1二值奖励，而是：

$$r_g = \begin{cases} (1-w) \cdot 1 + w \cdot 0 = 1-w & \text{若候选 } g \text{ 匹配目标} \\ (1-w) \cdot 0 + w \cdot p_{\text{rank}(g)} & \text{若不匹配} \end{cases} \tag{4}$$

其中 $w$ 是NDCG权重（如0.5），$p_{\text{rank}(g)} < 0$ 是依赖于Beam Search排名的归一化惩罚项（排名越靠前，惩罚绝对值越大）。

**GRPO归一化的关键性质**：每个组内优势之和恒为零：

$$\sum_{g=1}^{G+1} \hat{A}_g = 0 \tag{5}$$

---

## 3. 核心推导

### 3.1 GRPO对logits的梯度

对单个候选（状态 $s$，选择Token $a$，优势 $\hat{A}$），logit $z_v$ 的梯度上升更新为：

$$\Delta z_v = \eta\,\hat{A}\,\frac{\partial \log \pi_a}{\partial z_v} = \eta\,\hat{A}\,(\mathbb{1}[v{=}a] - \pi_v) \tag{6}$$

### 3.2 一组候选的聚合logit更新

对一个训练样本的整组 $G{+}1$ 个候选，聚合logit更新为：

$$\Delta z_v = \eta \sum_{g=1}^{G+1} \hat{A}_g\,(\mathbb{1}[a_g{=}v] - \pi_v)$$

利用性质(5)（$\sum_g \hat{A}_g = 0$），$\pi_v$ 项消去，化简为：

$$\boxed{\Delta z_v = \eta\,S_v} \tag{7}$$

其中定义：

$$S_v \triangleq \sum_{g:\,a_g = v} \hat{A}_g$$

即**经过Token $v$ 的所有候选的净优势之和**。若没有候选经过 $v$，则 $S_v = 0$。

由(5)立刻可得：$\sum_{k=1}^K S_{v_k} = 0$。

### 3.3 概率变化的推导（通用 $K$-Token引理）

利用softmax的Jacobian $\frac{\partial \pi_v}{\partial z_j} = \pi_v(\mathbb{1}[v{=}j] - \pi_j)$，概率的一阶变化为：

$$\Delta\pi_v = \sum_{j=1}^K \frac{\partial \pi_v}{\partial z_j}\,\Delta z_j = \pi_v \sum_{j=1}^K (\mathbb{1}[v{=}j] - \pi_j)\,\eta\,S_j$$

$$= \eta\,\pi_v \left[S_v - \sum_{j=1}^K \pi_j\,S_j\right]$$

定义 $\bar{S} \triangleq \sum_{j=1}^K \pi_j\,S_j$（**概率加权的净优势均值**），得：

$$\boxed{\Delta\pi_v = \eta\,\pi_v\,(S_v - \bar{S})} \tag{8}$$

> **引理（通用概率变化公式）**：在任一含 $K$ 个合法Token的Trie节点上，Token $v$ 经过一组GRPO候选更新后的概率变化为 $\eta\pi_v(S_v - \bar{S})$。
>
> - $S_v > \bar{S}$：概率**上升**（该Token的候选表现优于加权平均）
> - $S_v < \bar{S}$：概率**下降**
> - $S_v = 0$（无候选经过）且 $\bar{S} > 0$：概率**下降**（仅通过softmax归一化的间接效应）

---

## 4. 对冷门物品Token的分析

现在利用式(8)分析一个**特定冷门物品** $i_c$ 在Trie各层的Token概率如何变化。

### 4.1 关键机制：$\bar{S}$ 被热门Token拉高

在一个训练样本中，$\bar{S} = \sum_k \pi_k S_k$。这是一个以概率为权重的加权平均。

由 $\sum_k S_k = 0$，无权平均为零。但概率加权平均 $\bar{S}$ 的符号取决于 $\pi_k$ 与 $S_k$ 的**相关性**：

$$\bar{S} = \sum_k \pi_k S_k = \text{Cov}_{\text{uniform}}(\pi, S) \cdot K$$

当高概率Token倾向于有高 $S_k$（正净优势）时，$\bar{S} > 0$。

**这恰好是GRPO训练中的典型情况**：

- 热门物品的Token子树概率高（$\pi_k$ 大）
- 这些子树中的候选更容易匹配目标（因为目标也倾向于是热门物品），因此获得更多正优势（$S_k$ 大）
- 因此 $\bar{S} > 0$

### 4.2 按训练样本类型分析

#### 情形A：目标是高频物品（频率 $\phi_h$，高）

设目标物品 $i^*$ 位于Token $v_{k^*}$ 对应的子树中，且 $\pi_{k^*}$ 较大。

Beam Search的 $G$ 个候选大多经过高概率Token（不一定是 $v_{k^*}$），GT经过 $v_{k^*}$。用NDCG奖励：

- 匹配候选（GT和可能的Beam候选）：奖励 $r = 1-w > 0$
- 不匹配候选：奖励 $r = w \cdot p_{\text{rank}} < 0$（排名依赖的负惩罚）

匹配候选的优势为正（$r > \bar{r}$），不匹配候选的优势为负。由于 $v_{k^*}$ 高概率且有匹配候选，$S_{k^*} > 0$。

**对冷门物品 $i_c$ 在此节点的Token $v_c$ 的效应**：

- 若 $v_c \neq v_{k^*}$（冷门物品不在目标子树中）：通常没有候选经过 $v_c$，$S_c = 0$
- $\bar{S} > 0$（因为 $S_{k^*} > 0$ 且 $\pi_{k^*}$ 大，正值被高权重放大）
- 因此 $\Delta\pi_c = \eta\pi_c(0 - \bar{S}) < 0$

**冷门Token概率下降——即使没有任何候选探索过它。** 这纯粹是softmax归一化的间接效应：其他Token的logit增大 → softmax分母增大 → 冷门Token概率被挤压。

#### 情形B：目标是低频物品（频率 $\phi_c$，低）

设目标恰好是冷门物品 $i_c$，GT经过 $v_c$。

Beam Search的 $G$ 个候选大多经过高概率Token，不匹配 $i_c$。只有GT匹配。

- $S_c > 0$（GT的正优势）
- 高概率Token上的候选都不匹配，$S_k < 0$
- $\bar{S} = \sum_k \pi_k S_k$：此时高概率Token有负 $S_k$，权重大；冷门Token有正 $S_c$，权重小
- 因此 $\bar{S} < 0$
- $\Delta\pi_c = \eta\pi_c(S_c - \bar{S}) > 0$（$S_c > 0 > \bar{S}$）

**冷门Token概率上升。**

### 4.3 数值示例（使用NDCG奖励）

以一个含3个合法Token的Trie节点为例：$v_1$（$\pi_1=0.6$，热门子树），$v_2$（$\pi_2=0.3$，中频子树），$v_3$（$\pi_3=0.1$，冷门物品 $i_c$ 的Token）。

$G{+}1 = 5$（4个Beam + 1个GT），NDCG权重 $w = 0.5$。

归一化NDCG惩罚（5个位置）：$p_0 = -0.339$，$p_1 = -0.214$，$p_2 = -0.170$，$p_3 = -0.146$，$p_4 = -0.131$。

#### 示例A：热门目标（经过 $v_1$，2个候选匹配）

候选分布：3个经过 $v_1$（1个匹配）+ 1个经过 $v_2$ + GT经过 $v_1$（匹配）。共2个匹配。

| 候选 | Token | 匹配 | 奖励 | 优势 |
|------|-------|------|------|------|
| beam1 ($v_1$) | $v_1$ | ✓ | $0.5$ | $+1.095$ |
| beam2 ($v_1$) | $v_1$ | ✗ | $0.5 \times (-0.214) = -0.107$ | $-0.788$ |
| beam3 ($v_1$) | $v_1$ | ✗ | $0.5 \times (-0.170) = -0.085$ | $-0.720$ |
| beam4 ($v_2$) | $v_2$ | ✗ | $0.5 \times (-0.146) = -0.073$ | $-0.682$ |
| GT ($v_1$) | $v_1$ | ✓ | $0.5$ | $+1.095$ |

（均值 $\bar{r} = 0.147$，标准差 $\sigma = 0.322$，优势之和 $= 0$ ✓）

$$S_1 = 1.095 + (-0.788) + (-0.720) + 1.095 = 0.682$$
$$S_2 = -0.682, \quad S_3 = 0$$

$$\bar{S} = 0.6 \times 0.682 + 0.3 \times (-0.682) + 0.1 \times 0 = 0.205$$

冷门Token $v_3$ 的概率变化：

$$\boxed{\Delta\pi_3 = \eta \times 0.1 \times (0 - 0.205) = -0.0205\,\eta < 0}$$

**$v_3$ 没有任何候选经过，概率仍然下降。**

#### 示例B：冷门目标（经过 $v_3$，仅GT匹配）

候选分布：3个经过 $v_1$ + 1个经过 $v_2$ + GT经过 $v_3$（匹配）。

| 候选 | Token | 匹配 | 奖励 | 优势 |
|------|-------|------|------|------|
| beam1 ($v_1$) | $v_1$ | ✗ | $0.5 \times (-0.339) = -0.170$ | $-0.666$ |
| beam2 ($v_1$) | $v_1$ | ✗ | $0.5 \times (-0.214) = -0.107$ | $-0.436$ |
| beam3 ($v_1$) | $v_1$ | ✗ | $0.5 \times (-0.170) = -0.085$ | $-0.356$ |
| beam4 ($v_2$) | $v_2$ | ✗ | $0.5 \times (-0.146) = -0.073$ | $-0.313$ |
| GT ($v_3$) | $v_3$ | ✓ | $0.5$ | $+1.771$ |

（均值 $\bar{r} = 0.013$，标准差 $\sigma = 0.275$，优势之和 $= 0$ ✓）

$$S_1 = -1.458, \quad S_2 = -0.313, \quad S_3 = 1.771$$

$$\bar{S} = 0.6 \times (-1.458) + 0.3 \times (-0.313) + 0.1 \times 1.771 = -0.792$$

$$\boxed{\Delta\pi_3 = \eta \times 0.1 \times (1.771 - (-0.792)) = +0.256\,\eta > 0}$$

**冷门目标样本提供强力保护。**

### 4.4 聚合：损害 vs. 保护的频率之争

将两种情形在训练分布上聚合（设 $\phi_1 = 0.8$（$v_1$ 子树物品为目标的频率），$\phi_3 = \phi_c$（$v_3$ 子树物品为目标的频率），简化忽略 $v_2$）：

$$\mathbb{E}[\Delta\pi_3] \approx \phi_1 \times (-0.0205\,\eta) + \phi_c \times (+0.256\,\eta) \tag{9}$$

冷门概率净减小的条件（$\mathbb{E}[\Delta\pi_3] < 0$）：

$$\phi_1 \times 0.0205 > \phi_c \times 0.256$$

$$\boxed{\frac{\phi_1}{\phi_c} > \frac{0.256}{0.0205} \approx 12.5} \tag{10}$$

**含义**：当热门子树目标频率与冷门子树目标频率之比超过约12.5时，冷门Token概率净下降。

对比之前使用二值奖励的阈值（$5.5/q$），NDCG奖励下的阈值（$\approx 12.5$）略有不同，但**数量级一致**。

---

## 5. 在Trie不同层的分析

前面的分析是对**单个Trie节点**的。冷门物品 $i_c$ 的最终生成概率是 $L$ 层条件概率之积（式1）。需要逐层分析。

### 5.1 浅层（第1-2层）：共享前缀，冷门物品搭便车

RQ-VAE的语义编码按**语义相似度**组织，而非按流行度。因此在浅层：

- 冷门物品 $i_c$ 的Token $c_l^{i_c}$ 与大量热门物品**共享**（它们在语义上相似，被分到同一个粗粒度簇）
- 该Token对应的子树包含大量物品，其中热门物品居多
- 因此 $S_{c_l^{i_c}}$ 可能为**正**（子树内有大量匹配候选贡献正优势）

$$\Rightarrow \Delta\pi_{c_l^{i_c}} \text{ 可能 } \geq 0 \quad (\text{浅层冷门物品的Token概率不降反升})$$

**浅层不是冷门物品受损的位置。**

### 5.2 深层（第3-4层）：Token粒度接近物品，核心损害发生在此

在深层，每个Token对应的子树越来越小，最终（第 $L$ 层）一个Token基本对应一个物品。此时：

- 冷门物品 $i_c$ 的Token $c_L^{i_c}$ 几乎是**唯一**的——只有目标为 $i_c$ 的样本的GT才会经过它
- $S_{c_L^{i_c}} > 0$ 仅来自目标为 $i_c$ 的样本的GT（频率 $\phi_c$，很低）
- 大部分训练样本中，$S_{c_L^{i_c}} = 0$（没有候选经过）
- 但 $\bar{S} > 0$（被热门物品的Token拉高）

因此由式(8)：**大部分样本中 $\Delta\pi_{c_L^{i_c}} < 0$**。少量冷门目标样本的保护不足以抵消。

$$\Rightarrow \text{第4节的核心分析在此层成立，冷门Token概率净减小}$$

### 5.3 物品概率的层级乘积

$$\frac{P^{\text{GRPO}}(i_c \mid x)}{P^{\text{SFT}}(i_c \mid x)} = \prod_{l=1}^L \rho_l \tag{11}$$

| 层 | Token特点 | $\rho_l$ |
|----|---------|---------|
| 浅层（$l = 1, 2$） | 与热门物品共享，子树大 | $\geq 1$（搭便车效应） |
| 深层（$l = 3, 4$） | 接近物品级，子树小 | $< 1$（核心损害） |

**净效应**：深层损害是否超过浅层增益？

浅层增益：$\rho_l \geq 1$，但增量通常不大（因为共享子树中冷门物品的贡献很小，Token概率的增加主要惠及子树内的热门物品）。

深层损害：$\rho_l < 1$，且由第4节分析，在 $\phi_h/\phi_c > 12.5$ 时损害显著。

**在幂律分布下，对最冷门物品**（$\phi_h/\phi_c \gg 12.5$），深层损害明显大于浅层增益，净效应为概率下降。

---

## 6. 为什么SFT不会产生同样的损害

### 6.1 SFT的梯度结构

SFT使用交叉熵损失。对目标Token $v^*$，每个训练样本等效于"一组中只有1个候选（目标本身），优势为1"。

SFT对Trie节点中Token $v$ 的概率更新：

$$\Delta\pi_v^{\text{SFT}} = \eta\,\pi_v\,(\mathbb{1}[v = v^*] - \pi_v) \tag{12}$$

（这是式(8)的特例：$S_v = \mathbb{1}[v=v^*]$，$\bar{S} = \pi_{v^*}$。）

### 6.2 SFT在训练分布上的聚合

对冷门物品 $i_c$ 在深层的Token $v_c$，聚合SFT梯度：

$$\mathbb{E}[\Delta\pi_c]^{\text{SFT}} = \eta\pi_c\left[\phi_c(1 - \pi_c) - (1 - \phi_c)\pi_c\right] = \eta\pi_c\left[\phi_c - \pi_c\right] \tag{13}$$

当SFT收敛（$\Delta\pi_c = 0$）时：$\pi_c = \phi_c$。

**SFT收敛到概率与训练频率成正比的平衡态。**

### 6.3 GRPO打破SFT平衡的原因

GRPO以SFT模型为起点（$\pi_c \approx \phi_c$）。但GRPO引入了SFT中不存在的梯度来源：

**(a) Softmax间接挤压（SFT中不存在等价机制）**

SFT中，每个样本只更新**一个**Token的logit（目标Token），其余Token的logit不变。概率变化完全通过 $\mathbb{1}[v=v^*] - \pi_v$ 中的 $-\pi_v$ 项间接影响。

GRPO中，一组 $G{+}1$ 个候选同时更新**多个**Token的logit（所有被选过的Token）。当多个热门Token同时获得正 $S_k$ 时，$\bar{S}$ 被大幅拉高，冷门Token通过 $-\bar{S}$ 项受到更强的挤压。

**(b) 优势函数的放大**

SFT的等效"优势"恒为1。GRPO中，由于组内归一化，匹配候选的优势可远大于1。以 $G{+}1=5$、仅1个匹配为例（示例B中 $\hat{A}_{\text{GT}} = 1.771$），强化力度是SFT的约 **1.8倍**。

**(c) 梯度-采样正反馈（SFT中不存在）**

SFT的梯度仅取决于训练数据 $(x, i^*)$。GRPO的梯度额外取决于Beam Search输出（模型的当前采样分布）。$\pi_c$ 下降 → Beam Search更不可能生成冷门物品 → 冷门Token的 $S_c$ 更小 → $\Delta\pi_c$ 更负 → 正反馈循环。

### 6.4 SFT的自平衡 vs. GRPO的自弱化

SFT具有天然的**负反馈**：$\pi_c$ 被压低时，梯度项 $\phi_c - \pi_c$ 变大（正向修复力增强），自动拉回平衡。

GRPO的保护信号 $\Delta\pi_c^{(\text{protect})} \propto \pi_c$——$\pi_c$ 越低，保护越弱（**正反馈**），加速偏离平衡。

---

## 7. 阈值条件与实验对照

### 7.1 冷门概率净减小的条件

由式(9)的推广，冷门物品 $i_c$ 在深层Token处概率净减小的充分条件为：

$$\frac{\phi_{\text{others}}}{\phi_c} > \frac{\text{单次保护量}}{\text{单次损害量}} \tag{14}$$

数值示例（第4.3节）给出该阈值约为 **12.5**。实际值取决于 $G{+}1$、NDCG权重 $w$、节点分支数 $K$、概率分布等。

### 7.2 与实验数据的对照

| 数据集 | 最冷桶 | 预计 $\phi_{\text{others}}/\phi_c$ | 是否超过阈值 | 实验结果 |
|--------|-------|----------------------------------|------------|---------|
| Beauty | [5-9] | $\sim$50–80 | ✓ 远超 | Hit@5 **-24%** ✓ |
| Sports | [9-17] | $\sim$30–60 | ✓ 超过 | Hit@5 **-20%** ✓ |
| Toys | [5-8] | $\sim$40–75 | ✓ 远超 | Hit@5 **-15%** ✓ |
| Beauty | [10-20] | $\sim$5–15 | ≈ 阈值附近 | Hit@5 +12% ✓ |
| Toys | [9-16] | $\sim$5–12 | ≈ 阈值附近 | Hit@5 +6% ✓ |

- **最冷桶**（$\phi_{\text{others}}/\phi_c \gg 12.5$）：深层损害远超浅层增益 → 概率净下降 → Hit@5 下降 ✓
- **中间桶**（$\phi_{\text{others}}/\phi_c \approx 5\text{–}15$）：接近阈值，浅层增益部分抵消深层损害 → 概率微升或不变 ✓
- **最热桶**：它们自身的Token $S_k$ 大且 $\pi_k$ 高 → $S_k > \bar{S}$ → 概率上升 ✓

---

## 8. 完整因果链

$$\text{幂律分布: 冷门物品频率 } \phi_c \text{ 极低}$$
$$\downarrow$$
$$\text{SFT平衡态: 冷门物品深层Token概率 } \pi_c \approx \phi_c \text{ (小但合理)}$$
$$\downarrow$$
$$\text{GRPO训练: 每个样本生成 }G{+}1\text{ 个候选，计算组内优势}$$
$$\downarrow$$
$$\text{大部分样本的目标是高频物品 → 热门Token获得正 }S_k$$
$$\downarrow$$
$$\bar{S} = \sum_k \pi_k S_k > 0 \text{ (被高概率、高}S_k\text{的热门Token拉高)}$$
$$\downarrow$$
$$\text{冷门物品深层Token: } S_c \approx 0 \text{ (无候选探索) } \Rightarrow \Delta\pi_c = \eta\pi_c(0 - \bar{S}) < 0$$
$$\downarrow$$
$$\text{少量冷门目标样本的保护 }(\phi_c \times \Delta\pi_c^{+}) \text{ 不足以抵消}$$
$$\text{（因 }\phi_{\text{others}}/\phi_c > 12.5\text{）}$$
$$\downarrow$$
$$\text{深层Token概率净下降，乘以}L\text{层 → 物品概率下降 15–24\%}$$
$$\downarrow$$
$$\text{正反馈: }\pi_c\downarrow \;\to\; \text{保护更弱} \;\to\; \pi_c\downarrow\downarrow$$
$$\downarrow$$
$$\text{推理时Beam Search排名下降 → Hit@K 下降}$$

---

## 附录：关键公式的验证

### A.1 概率变化验证

取 $K=3$，$\boldsymbol{\pi} = (0.6, 0.3, 0.1)$，$\mathbf{S} = (0.682, -0.682, 0)$（示例A）：

$$\Delta\pi_1 = \eta \times 0.6 \times (0.682 - 0.205) = 0.286\,\eta$$
$$\Delta\pi_2 = \eta \times 0.3 \times (-0.682 - 0.205) = -0.266\,\eta$$
$$\Delta\pi_3 = \eta \times 0.1 \times (0 - 0.205) = -0.021\,\eta$$

验证：$\Delta\pi_1 + \Delta\pi_2 + \Delta\pi_3 = (0.286 - 0.266 - 0.021)\eta \approx 0$ ✓（微小误差来自四舍五入）

### A.2 聚合阈值验证

设 $\phi_1 = 0.9$（热门目标频率），$\phi_3 = 0.1$（冷门目标频率），$\phi_1/\phi_3 = 9$。

$$\mathbb{E}[\Delta\pi_3] = 0.9 \times (-0.021\,\eta) + 0.1 \times (0.256\,\eta) = (-0.019 + 0.026)\eta = +0.007\,\eta > 0$$

**$\phi_1/\phi_3 = 9 < 12.5$**：冷门概率微升 ✓（未超阈值，保护足够）

设 $\phi_1 = 0.98$，$\phi_3 = 0.02$，$\phi_1/\phi_3 = 49$：

$$\mathbb{E}[\Delta\pi_3] = 0.98 \times (-0.021\,\eta) + 0.02 \times (0.256\,\eta) = (-0.0206 + 0.0051)\eta = -0.0155\,\eta < 0$$

**$\phi_1/\phi_3 = 49 > 12.5$**：冷门概率下降 ✓（超过阈值，保护不足）
