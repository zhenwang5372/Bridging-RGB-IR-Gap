{
type: "file",
fileName: "softmax_G_sumfirst_product.md",
content: """

# G加权先求和后求积 (Softmax-G-SumFirst-Product)

## 1. 方案核心思想
本方案名为 **"G-Weighted SumFirst-Product"**。
与之前的 "SumLast"（先积后和）不同，本方案采用 **"SumFirst"（先和后积）** 的策略。

- **逻辑差异**：
    - **SumLast**: $\sum (G \cdot A \cdot B)$。强调**逐类对应**，只有当 RGB 和 IR 在**同一个类别**上都有响应时才激活。
    - **SumFirst**: $(\sum G \cdot A) \cdot (\sum G \cdot B)$。强调**综合显著性**，先生成 RGB 和 IR 各自的“可信综合热力图”，再计算这两个热力图的空间重合度。
- **优势**: 计算量更小（在 $N \to 1$ 压缩后进行 $HW$ 级别的乘法），且容忍度更高，能捕捉到两模态在不同类别通道上的隐含共鸣（只要是高置信度类别）。

---

## 2. 详细步骤与数学推导

### Step 1: 文本引导的语义激活 (Semantic Activation)
*(保持不变)*
输入: $X_{rgb}, X_{ir}$, $Text_{emb}$
$$
Q = Text_{emb} W_Q
$$
$$
A_{rgb} = \text{Softmax}\left(\frac{Q K_{rgb}^T}{\sqrt{d_k}}\right), \quad A_{ir} = \text{Softmax}\left(\frac{Q K_{ir}^T}{\sqrt{d_k}}\right)
$$
* $A_{rgb}, A_{ir} \in \mathbb{R}^{B \times N \times HW}$

---

### Step 2: 语义一致性度量 (Consistency Measure G)
*(保持不变)*
$$
G_c = \frac{A_{rgb}[:, c, :] \cdot A_{ir}[:, c, :]}{\|A_{rgb}[:, c, :]\|_2 \|A_{ir}[:, c, :]\|_2 + \epsilon}
$$
* $G \in \mathbb{R}^{B \times N}$：衡量每个类别在两个模态间的一致性。

---

### Step 3: G加权求和与空间求积 (G-Weighted SumFirst & Product)
*(**核心修改**: 加权前置，求和前置)*

1.  **模态内加权聚合 (Intra-Modality Weighted Aggregation)**:
    分别利用 $G$ 对 RGB 和 IR 的注意力图进行加权，并沿类别维度 $N$ 求和，生成单通道的“可信语义图”。
    
    $$
    M_{rgb} = \sum_{c=1}^{N} G_c \cdot A_{rgb}^c \quad \in \mathbb{R}^{B \times HW}
    $$
    $$
    M_{ir} = \sum_{c=1}^{N} G_c \cdot A_{ir}^c \quad \in \mathbb{R}^{B \times HW}
    $$
    
    * **物理含义**: $M_{rgb}$ 仅保留了那些**在 IR 中也得到确认（$G$ 高）**的 RGB 类别特征，滤除了单模态的幻觉噪声。

2.  **空间共识求积 (Spatial Consensus Product)**:
    计算两个综合图的哈达姆积。
    
    $$
    S_{raw} = M_{rgb} \odot M_{ir} \quad \in \mathbb{R}^{B \times HW}
    $$

3.  **归一化 (Normalization)**:
    对乘积结果进行归一化。
    
    $$
    S_{norm} = \frac{S_{raw} - \min(S_{raw})}{\max(S_{raw}) - \min(S_{raw}) + \epsilon}
    $$
    $$
    S_{map} = \text{Reshape}(S_{norm}, [B, 1, H, W])
    $$

---

### Step 4: 卷积细化 (Convolutional Refinement)
*(保持不变)*
$$
M_{final} = \text{Sigmoid}\left( \text{Conv}_{3\times3}\left( \text{ReLU}\left( \text{Conv}_{1\times1}(S_{map}) \right) \right) \right)
$$

---

### Step 5: 特征增强 (Feature Enhancement)
*(保持不变)*
$$
X_{ir}^{out} = X_{ir} \cdot (1 + \alpha \cdot M_{final})
$$

---

## 3. PyTorch 实现伪代码

```python
class SoftmaxGSumFirstProduct(nn.Module):
    def forward(self, x_rgb, x_ir, text_emb):
        # 1. Semantic Activation
        a_rgb = self.get_attention(x_rgb, text_emb) # [B, N, HW]
        a_ir = self.get_attention(x_ir, text_emb)   # [B, N, HW]
        
        # 2. Global Consistency G
        a_rgb_n = F.normalize(a_rgb, dim=-1)
        a_ir_n = F.normalize(a_ir, dim=-1)
        g_score = (a_rgb_n * a_ir_n).sum(dim=-1, keepdim=True) # [B, N, 1]
        
        # 3. SumFirst & Product (核心差异)
        # 3.1 加权 (Weighting)
        # 将 G 广播应用到每个模态
        w_rgb = a_rgb * g_score  # [B, N, HW]
        w_ir = a_ir * g_score    # [B, N, HW]
        
        # 3.2 求和/聚合 (Summation / Collapse)
        m_rgb = w_rgb.sum(dim=1) # [B, HW] (BXHW在此处产生)
        m_ir = w_ir.sum(dim=1)   # [B, HW]
        
        # 3.3 空间求积 (Spatial Product)
        s_raw = m_rgb * m_ir     # [B, HW]
        
        # 3.4 归一化 (Normalization)
        s_min = s_raw.min(dim=-1, keepdim=True)[0]
        s_max = s_raw.max(dim=-1, keepdim=True)[0]
        s_norm = (s_raw - s_min) / (s_max - s_min + 1e-6)
        
        s_map = s_norm.view(B, 1, H, W)
        
        # 4. Conv Refinement
        m_final = self.conv_block(s_map)
        
        # 5. Enhancement
        out = x_ir * (1 + self.alpha * m_final)
        
        return out