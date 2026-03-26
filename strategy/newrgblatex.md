# 方案 C': Agreement -> MLP -> 双超参数调制注意力 -> 单 token 输出 -> 空间注入

你提出的改进（先由 G 生成两个调制超参数，去调制 $A^{rgb}, A^{ir}$，再得到一个 $Y \in \mathbb{R}^{B \times 1 \times d}$）是成立的，而且很像高光谱融合里常见的 modulation / FiLM / dynamic weighting 思路。

下面给你一个“完整方案”，满足你所有约束：
*   text 作 query (桥梁)
*   不加显式对齐损失（只靠检测损失反向优化）
*   最终只产生 1 个 token 的 $Y$ (避免 C 倍膨胀)
*   还能恢复成与 RGB 特征图尺寸的注入项（注意：这一步不能简单 reshape(Y)，需要一个“从 token 回到空间”的展平操作；我会给你一个最干净的闭环公式）

---

## Step 0: 输入与 text 原型

在 neck 某一层：
$$ F^{rgb}, F^{ir} \in \mathbb{R}^{B \times D \times H \times W}, \quad N = H \cdot W $$
展平为 token:
$$ X^{rgb}, X^{ir} \in \mathbb{R}^{B \times N \times D} $$
准备冻结 text 类别原型（或文本编码器输出）：
$$ T \in \mathbb{R}^{C \times d_t} $$

---

## Step 1: Text-as-Query 得到两路注意力 $A^{rgb}, A^{ir}$

**构造 Query (共享):**
$$ Q = \text{Norm}(T W_q) \in \mathbb{R}^{C \times d_k} $$

**构造 Key/Value:**
$$ K^m = X^m W_k^m \in \mathbb{R}^{B \times N \times d_k}, \quad V^m = X^m W_v^m \in \mathbb{R}^{B \times N \times d_v}, \quad m \in \{rgb, ir\} $$

**Cross-attn 得到两路注意力:**
$$ A^m = \text{Softmax}_N \left( \frac{Q (K^m)^\top}{\sqrt{d_k}} \right) \in \mathbb{R}^{B \times C \times N} $$

到这里为止仍然有 $C$（因为你需要同一 text 类别下的注意力图来比较一致性），但马上把它压缩回 1。

---

## Step 2: 从 $A^{rgb}, A^{ir}$ 计算一致性 $G$ (仍然是 $B \times C$)

对每个 query 类别 $c$，取空间注意力向量：
$$ a^{rgb}_{b,c,:} = A^{rgb}_{b,c,:} \in \mathbb{R}^N, \quad a^{ir}_{b,c,:} = A^{ir}_{b,c,:} \in \mathbb{R}^N $$

**一致性 (cosine):**
$$ g_{b,c} = \frac{\langle a^{rgb}_{b,c,:}, a^{ir}_{b,c,:} \rangle}{\|a^{rgb}_{b,c,:}\|_2 \|a^{ir}_{b,c,:}\|_2 + \epsilon} \in [-1, 1] $$

得到：
$$ G \in \mathbb{R}^{B \times C} $$

---

## Step 3: 用 MLP 把 $G$ 压缩为两个超参数（你想要的“调制操作”）

这里有两种你可以选的“压缩方式”。我建议用 3.1（更稳、更符合“避免 C 放大”）。

### 3.1 (推荐) 先把 $G$ 做统计汇聚，再 MLP 输出两个标量

计算几个统计量（不依赖 $C$ 的维度大小）：
$$ u_b = [\text{mean}(G_b), \text{max}(G_b), \text{std}(G_b)] \in \mathbb{R}^3 $$

**MLP 输出两条调制参数:**
$$ [\eta^{rgb}_b, \eta^{ir}_b] = \text{MLP}(u_b) \in \mathbb{R}^2 $$

**做归一化 (避免数值发散):**
$$ [\lambda^{rgb}, \lambda^{ir}] = \text{Softmax}([\eta^{rgb}_b, \eta^{ir}_b]) \quad \Rightarrow \quad \lambda^{rgb}, \lambda^{ir} \in (0,1), \quad \lambda^{rgb} + \lambda^{ir} = 1 $$

---

## Step 4: 用两个超参数去“调制” $A^{rgb}, A^{ir}$，并把 $C$ 压成 1 个注意力图

**关键点：你要从 $B \times C \times N$ 得到 $B \times 1 \times N$。**

一个非常自然的做法是：先用 $G$ 生成一个权重 $w$，把 $C$ 个 query 的注意力加权汇聚成“单张空间注意力图”。

**先把一致性 $G$ 变成类权重 (softmax over C):**
$$ w_{b,:} = \text{Softmax}\left( \frac{G_{b,:}}{\tau_g} \right) \in \mathbb{R}^C, \quad \sum_c w_{b,c} = 1 $$

**对两路注意力分别做类维度汇聚（得到单注意力图）:**
$$ A^{rgb}_{b,1,:} = \sum_{c=1}^C w_{b,c} A^{rgb}_{b,c,:} \in \mathbb{R}^N $$
$$ A^{ir}_{b,1,:} = \sum_{c=1}^C w_{b,c} A^{ir}_{b,c,:} \in \mathbb{R}^N $$

**然后用你 MLP 生成的两超参数做调制融合（你说的“调制操作”）:**
$$ \bar{A} = \lambda^{rgb} \odot A^{rgb} + \lambda^{ir} \odot A^{ir} \in \mathbb{R}^{B \times 1 \times N} $$

**最后把它重新归一化成标准注意力分布（对 $N$ 做归一）:**
$$ A^{fuse} = \frac{\bar{A}}{\sum_{n=1}^N \bar{A}_{:,:,n} + \epsilon} \in \mathbb{R}^{B \times 1 \times N} $$

到这里，$C$ 已经被你完全去掉，只剩一张融合注意力图 $A^{fuse}$。

---

## 注入逻辑与综合公式

# Step 5: 得到你想要的单 token 输出 

#### $Y \in \mathbb{R}^{B \times 1 \times d_v}$



你要求 “$Y \in \mathbb{R}^{B \times 1 \times d}$” ，对应就是：

先把 value 融合（同样用两超参数调制）：

$$
V^{fuse} = \lambda^{rgb} \odot V^{rgb} + \lambda^{ir} \odot V^{ir} \in \mathbb{R}^{B \times N \times d_v}
$$

再做加权求和（注意力乘 value）：

$$
\boxed{Y = A^{fuse} V^{fuse} \in \mathbb{R}^{B \times 1 \times d_v}}
$$

这一步严格符合你想要的“最后只生成 1 个 $Y$”——不会出现 $C$ 倍 token。

---

# Step 6: 把单 token $Y$ 回注入成与原 RGB 特征图同尺寸（关键闭环）

这里必须讲清楚：不能直接 `reshape(Y)` 变成 $B \times D \times H \times W$，除非你强行设 $d_v = DHW$（这会极不合理也极重）。

正确做法是使用一个“token $\rightarrow$ 空间展开”的操作。最干净、且与你的 attention 结构一致的是：

$$
X^{inj} = (A^{fuse})^\top Y
$$

注意维度：|

注意维度：

*   $(A^{fuse})^\top \in \mathbb{R}^{B \times N \times 1}$

**然后 reshape 回空间:**
$$ X^{inj} \in \mathbb{R}^{B \times N \times d_v} $$
$$ F^{inj} = \text{reshape}(X^{inj}) \in \mathbb{R}^{B \times d_v \times H \times W} $$

**再用一个 $1 \times 1$ conv 投回通道数 $D$ 并残差注入（RGB 主干，IR 辅助，符合你“IR 可能有问题”的设定）：**
$$ \Delta F = \psi(F^{inj}) \in \mathbb{R}^{B \times D \times H \times W} $$
$$ F^{out} = F^{rgb} + \Delta F $$

---

## 你要的“综合相加公式”版本（把核心串成一条主线）

为了满足你希望“一条相加公式展示”的风格，我给你一个论文里可用的紧凑表达（省略中间维度的求和符号）：

$$ F^{out} = F^{rgb} + \psi \left( \text{reshape} \left( (A^{fuse})^\top (A^{fuse} (X^{rgb} V^{rgb} + \lambda^{ir} V^{ir})) \right) \right) $$

**其中：**
$$ A^{fuse} = \text{Norm}_N \left( \lambda^{rgb} \sum_c w_c A^{rgb}_c + \lambda^{ir} \sum_c w_c A^{ir}_c \right) $$
$$ w = \text{Softmax}\left(\frac{G}{\tau_g}\right), \quad [\lambda^{rgb}, \lambda^{ir}] = \text{Softmax}(\text{MLP}(\text{Pool}(G))) $$

