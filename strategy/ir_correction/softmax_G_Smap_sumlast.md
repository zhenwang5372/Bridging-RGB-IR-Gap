{
type: "file",
fileName: "softmax_G_Smap_sumlast.md",
content: """

# 方案五：G求积归一化加卷积 (Softmax-G-Smap-SumLast)

## 1. 方案核心思想
本方案名为 **"G-Product-Norm-Conv" (G求积归一化加卷积)**。
它结合了全局语义一致性 ($G$) 与细粒度空间共识 (Hadamard Product)，旨在挖掘并增强 RGB 和 IR 模态在语义和空间上都达成“共识”的区域。

- **Softmax**: 使用 Softmax Attention 生成概率分布形式的语义图。
- **G (Global)**: 利用文本计算类别一致性 $G_c$，作为可信度权重。
- **Smap (Product)**: 利用哈达姆积 ($A \odot B$) 挖掘空间共鸣。
- **SumLast**: 在计算完所有类别的加权共识后，最后进行求和聚合 ($N \to 1$)。
- **Norm & Conv**: 引入归一化和卷积层，将共识图转化为高质量的特征门控。

---

## 2. 详细步骤与数学推导

### Step 1: 文本引导的语义激活 (Semantic Activation)
**目的**: 将文本特征映射到图像空间，生成类别注意力图。

输入: 
- $X_{rgb}, X_{ir} \in \mathbb{R}^{B \times C \times H \times W}$
- $Text_{emb} \in \mathbb{R}^{N \times d}$ (N classes)

$$
Q = Text_{emb} W_Q
$$
$$
K_{rgb} = \text{Conv}(X_{rgb}), \quad K_{ir} = \text{Conv}(X_{ir})
$$

计算 Softmax Attention Maps:
$$
A_{rgb} = \text{Softmax}\left(\frac{Q K_{rgb}^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{B \times N \times HW}
$$
$$
A_{ir} = \text{Softmax}\left(\frac{Q K_{ir}^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{B \times N \times HW}
$$

---

### Step 2: 语义一致性度量 (Consistency Measure G)
**目的**: 判断哪些类别在当前两模态中是一致可信的。

使用余弦相似度计算类别权重 $G_c$：
$$
G_c = \frac{A_{rgb}[:, c, :] \cdot A_{ir}[:, c, :]}{\|A_{rgb}[:, c, :]\|_2 \|A_{ir}[:, c, :]\|_2 + \epsilon}
$$
* $G \in \mathbb{R}^{B \times N}$
* $G_c$ 值越高，代表该类别在 RGB 和 IR 中语义对齐度越好，其空间特征越值得增强。

---

### Step 3: 加权求积与归一化 (Weighted Product & Normalization)
**目的**: 生成“共识热力图” ($S_{map}$)。这是本方案的核心差异点。

1.  **加权求积聚合 (SumLast)**:
    不再计算差值，而是计算**积**。只有当 RGB 响应高、IR 响应高、且全局一致性 $G$ 高时，该区域才被激活。
    $$
    S_{raw} = \sum_{c=1}^{N} \underbrace{G_c}_{\text{Weight}} \cdot (\underbrace{A_{rgb}^c \odot A_{ir}^c}_{\text{Spatial Product}})
    $$
    * $S_{raw} \in \mathbb{R}^{B \times HW}$

2.  **归一化 (Normalization)**:
    为了防止数值范围过大导致后续卷积层不稳定，对聚合后的图进行 Min-Max 归一化。
    $$
    S_{norm} = \frac{S_{raw} - \min(S_{raw})}{\max(S_{raw}) - \min(S_{raw}) + \epsilon}
    $$
    $$
    S_{map} = \text{Reshape}(S_{norm}, [B, 1, H, W])
    $$

---

### Step 4: 卷积细化 (Convolutional Refinement)
**目的**: 将稀疏或粗糙的共识图转化为连续的门控信号。

$$
M_{final} = \text{Sigmoid}\left( \text{Conv}_{3\times3}\left( \text{ReLU}\left( \text{Conv}_{1\times1}(S_{map}) \right) \right) \right)
$$

* 第一层 $1 \times 1$ Conv 用于特征投影。
* 第二层 $3 \times 3$ Conv 用于利用上下文信息平滑空间噪声。

---

### Step 5: 特征增强 (Feature Enhancement)
**目的**: 利用生成的门控增强 IR 特征。

由于 $M_{final}$ 代表的是“语义和空间都高度一致的可信区域”，我们采取**增强 (Enhancement)** 策略，而非抑制。

$$
X_{ir}^{out} = X_{ir} \cdot (1 + \alpha \cdot M_{final})
$$

* $\alpha$: 可学习参数，建议初始化为正值 (e.g., 0.1)。
* **物理含义**: 在确信正确的区域（共识区），放大红外特征的表达能力。

---

## 3. PyTorch 实现伪代码

```python
class SoftmaxGSmapSumLast(nn.Module):
    def forward(self, x_rgb, x_ir, text_emb):
        # --- Step 1: Semantic Activation ---
        # Q: [B, N, C], K: [B, C, HW] -> Attn: [B, N, HW]
        a_rgb = self.get_attention(x_rgb, text_emb)
        a_ir = self.get_attention(x_ir, text_emb)
        
        # --- Step 2: Global Consistency G ---
        # Normalize for cosine similarity
        a_rgb_n = F.normalize(a_rgb, dim=-1)
        a_ir_n = F.normalize(a_ir, dim=-1)
        # G: [B, N, 1]
        g_score = (a_rgb_n * a_ir_n).sum(dim=-1, keepdim=True)
        
        # --- Step 3: Weighted Product & SumLast ---
        # Element-wise product: [B, N, HW]
        spatial_product = a_rgb * a_ir
        
        # Weighted sum: sum(G * Product) -> [B, HW]
        # Broadcasting g_score: [B, N, 1] * [B, N, HW] -> sum(dim=1)
        s_raw = (g_score * spatial_product).sum(dim=1)
        
        # Normalization (Min-Max per sample)
        s_min = s_raw.min(dim=-1, keepdim=True)[0]
        s_max = s_raw.max(dim=-1, keepdim=True)[0]
        s_norm = (s_raw - s_min) / (s_max - s_min + 1e-6)
        
        # Reshape to [B, 1, H, W]
        s_map = s_norm.view(B, 1, H, W)
        
        # --- Step 4: Conv Refinement ---
        # Conv1x1 -> ReLU -> Conv3x3 -> Sigmoid
        m_final = self.conv_block(s_map)
        
        # --- Step 5: Enhancement ---
        # x_ir * (1 + alpha * mask)
        out = x_ir * (1 + self.alpha * m_final)
        
        return out