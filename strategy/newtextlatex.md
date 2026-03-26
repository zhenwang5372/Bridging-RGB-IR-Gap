# 1. 输入与符号

在某个尺度（或多尺度可扩展）：
*   **RGB 特征 token:**
    $$ X^{rgb} \in \mathbb{R}^{B \times N \times D}, \quad N = H \cdot W $$
*   **IR (可信) 特征 token:**
    $$ X^{ir} \in \mathbb{R}^{B \times N \times D} $$
*   **文本类别原型 (冻结或预训练):**
    $$ T = [t_1, \dots, t_C] \in \mathbb{R}^{C \times d_t}, \quad \|t_c\|_2 = 1 $$

**可学习投影:**
*   $W_{ir \to t} \in \mathbb{R}^{D \times d_t}$
*   $W_q \in \mathbb{R}^{d_t \times d_k}$
*   $W_k \in \mathbb{R}^{D \times d_k}, \quad W_v \in \mathbb{R}^{D \times d_v}$
*   MLP: $\mathbb{R}^{d_v} \to \mathbb{R}^{d_t}$

**超参:**
*   $\tau > 0$: IR $\to$ text 类权重温度
*   $d_k, d_v$: attention 维度
*   $\gamma \in (0, 1]$: 视觉写入强度（可以常数，也可以由 IR 置信度自适应）

---

# 2. 计算流程 (与你给出的公式一一对应)

## Step A: IR 作为语义锚点，得到类别权重 $w$

1.  从 IR 得到一个全局语义向量（池化 + 投影 + 归一化）：
    $$ u^{ir} = \text{Norm}(\text{Pool}(X^{ir}) W_{ir \to t}) \in \mathbb{R}^{B \times d_t} $$
    *   **Pool(-)** 推荐 mean pooling: $\text{Pool}(X) = \frac{1}{N} \sum_{n=1}^N X_{:, n, :}$
    *   也可使用 attention pooling（更强但更重），初版建议 mean。

2.  与所有类别文本原型做相似度，并 softmax 得到类别权重：
    $$ w = \text{Softmax}\left( \frac{u^{ir} T^\top}{\tau} \right) \in \mathbb{R}^{B \times C} $$
    *   **解释**: $w$ 是“样本在 IR 视角下的语义分布”，它是后续所有检索/更新的锚点。
    *   这里的$$w$$你结合后面的任务，你认为是使用softmax还是sigmoid好

---

## 下面是先对更新后的IR 进行了处理

### 建议你用 "IR-Guided CBAM"?

在你的任务（RGB + IR 融合）中，我们不能直接照搬原始的 CBAM，而是要做一点点魔改，这正是你创新的地方：

1.  **对于通道注意力 (Step 1):**
    *   **让 RGB 自己做**。因为 RGB 的颜色和纹理信息最丰富，它需要自己判断哪些通道有效（比如去掉因光照损坏的通道）。
2.  **对于空间注意力 (Step 2):**
    *   **不要用 RGB 做，要用 IR 做!**
    *   原始 CBAM 是从 Input 自己提取 Spatial Mask。
    *   **你的方案是：** 用 IR 特征图来生成这个 Spatial Mask，然后乘到 RGB 上。

### 为何加残差?
使用 IR-Guided CBAM 对 RGB 进行处理并进行残差：

$$ X'_{rgb} = X^{rgb} + \alpha \cdot [\text{ChannelAttn}(X^{rgb}) \cdot \text{SpatialMask}(X^{ir})] $$

或者更简单的变体（CBAM 标准残差）：

$$ X_{temp} = X^{rgb} \cdot \text{ChannelAttn}(X^{rgb}) $$
$$ X_{final} = X_{temp} + X_{temp} \cdot \text{SpatialMask}(X^{ir}) $$

你需要评估这两个方案选择其一即可

---

## Step B: Text 作为桥梁，检索 RGB 视觉证据 (Text $\to$ RGB Cross-Attention)

1.  **构造 query (来自文本原型):**
    $$ Q = T W_q \in \mathbb{R}^{C \times d_k} $$
2.  **构造 RGB 的 key/value:**
    $$ K^{rgb} = X^{rgb} W_k \in \mathbb{R}^{B \times N \times d_k}, \quad V^{rgb} = X^{rgb} W_v \in \mathbb{R}^{B \times N \times d_v} $$
3.  **text $\to$ RGB cross-attention 得到每个类别的空间注意力与类别条件视觉上下文:**
    $$ A^{rgb} = \text{Softmax}_N \left( \frac{Q (K^{rgb})^\top}{\sqrt{d_k}} \right) \in \mathbb{R}^{B \times C \times N} $$
    $$ Y^{rgb} = A^{rgb} V^{rgb} \in \mathbb{R}^{B \times C \times d_v} $$
4.  $$y^{rgb}\in \mathbb{R}^{B\times 1\times N}$$
    *   **解释**: $Y^{rgb}_{b,c,:}$ 是“类别 $c$ 在 RGB 特征上检索到的视觉证据摘要”。

---

## Step C: 逐类别残差更新 (Class-wise Update)

*   不再聚合 $t_{base}$。直接对原始的 $C$ 个文本原型 $t_c$ 进行一对一更新。
*   **公式：**
    $$ t^{new}_c = \text{Norm}(t_c + \gamma \cdot w\cdot\text{MLP}(y^{rgb}_c)) $$
*   这里 MLP 将视觉维度的 $Y^{rgb}_c$ 映射回文本维度，作为残差加到原始文本上。
*   **最终输出：** $T^{new} \in \mathbb{R}^{B \times C \times d_t}$。即 Batch 中每张图片都有自己专属的一套“被视觉微调过”的 $C$ 个类别文本向量，用于后续 Head 的匹配。

---

上面两张图片，使用更新后的 IR 和经过新华 IR 处理的 RGB 求出 $Y^{rgb}$，并对 $C$ 个 text 进行更新。