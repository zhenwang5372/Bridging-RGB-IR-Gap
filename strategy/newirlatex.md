# New IR 完整方案: Text-bridged RGB knowledge + Physics-modulated IR correction

## 0. 输入/输出与目标

*   **输入：**
    *   原始红外图像（含错误/污染）：$I^{ir} \in \mathbb{R}^{B \times 1 \times H \times W}$
    *   RGB 图像：$I^{rgb} \in \mathbb{R}^{B \times 3 \times H \times W}$
    *   文本类别原型（或 prompt 编码）：$T = [t_1, \dots, t_C] \in \mathbb{R}^{C \times d_t}$
*   **输出：**
    *   纠错后的红外：$I^{ir}_{new} \in \mathbb{R}^{B \times 1 \times H \times W}$
*   **目标（你定义的“去除错误部分”）在工程上可理解为：**
    *   抑制 IR 中的 **伪热点/反射/背景热干扰/传感器偏置** 等非目标信号；
    *   保留并增强与 **文本语义一致** 的目标相关热信号；
    *   该步骤本身不需要单独监督损失，允许由下游检测损失端到端反传（也可以加非常轻的稳定正则，但不是必须）。

---

## 1) 特征准备 (轻量、只为生成参数与语义图)

用浅层 backbone 或 neck 的某一层特征即可（不需要很深）：

$$ X^{rgb} \in \mathbb{R}^{B \times N \times D}, \quad X^{ir} \in \mathbb{R}^{B \times N \times D}, \quad N = H \cdot W $$

这里的 $X^{ir}$ 来自“原始 IR”也没问题，因为它只是做“粗语义锚定 + 统计信息”，真正的纠错是靠 RGB + text + 物理模型完成的。

---

## (A) IR 借助 Text 做语义锚定：得到类别权重 $w$

这一步的目的不是“相信 IR 的细节”，而是用 IR 提供一个 **场景语义方向**（例如此图更像 person / vehicle / drone / fire / ...），为后续从 RGB 取证提供“要取什么”的依据。

$$ u^{ir} = \text{Norm}(\text{Pool}(X^{ir}) W_{ir \to t}) \in \mathbb{R}^{B \times d_t} $$

$$ w = \text{Softmax}\left( \frac{u^{ir} T^\top}{\tau} \right) \in \mathbb{R}^{B \times C} $$

*   **Pool**: 推荐 mean pooling (稳定、便宜)
*   **$\tau$**: 温度，控制分布尖锐程度（越小越“挑类”）
*   **解释**: $w$ 是“IR 视角下的语义分布”，它只用来告诉我们：哪些文本类别更相关。

---

## (B) Text 作为桥梁，从 RGB 取“语义一致的视觉知识”：得到 $y^{rgb}$ 与语义区域图 $M$

这里沿用你前面已经熟悉的 Text-as-Query：

**构造 query/key/value:**
$$ Q = T W_q \in \mathbb{R}^{C \times d_k} $$
$$ K^{rgb} = X^{rgb} W_k \in \mathbb{R}^{B \times N \times d_k}, \quad V^{rgb} = X^{rgb} W_v \in \mathbb{R}^{B \times N \times d_v} $$

**Cross-attention:**
$$ A^{rgb} = \text{Softmax}_N \left( \frac{Q (K^{rgb})^\top}{\sqrt{d_k}} \right) \in \mathbb{R}^{B \times C \times N} $$
$$ Y^{rgb} = A^{rgb} V^{rgb} \in \mathbb{R}^{B \times C \times d_v} $$

**用 IR 的类别权重 $w$ 聚合得到单向量 "RGB 证据":**
$$ y^{rgb} = \sum_{c=1}^{C} w_c Y^{rgb}_c \in \mathbb{R}^{B \times 1 \times d_v} $$

**同时，用 $w$ 对注意力图聚合得到语义区域图（用来做空间级纠错）：**
$$ a^{rgb} = \sum_{c=1}^{C} w_c A^{rgb}_c \in \mathbb{R}^{B \times 1 \times N} $$

**Reshape 回空间并做平滑/卷积得到 mask:**
$$ M = \text{Smooth}(\text{reshape}(a^{rgb})) \in \mathbb{R}^{B \times 1 \times H \times W} $$

**解释：**
*   $y^{rgb}$: 是“与 IR 语义一致的 RGB 外观证据”（全局）。
*   $M$: 是“目标/语义相关区域在哪里”（空间级），用于做“突出正确、屏蔽错误”的空间门控。

---

## (C) MLP 生成物理模型超参数，作用于红外成像/校正公式：得到 $I^{ir}_{new}$

这是你提出的关键：MLP modulation $\to$ physical formula.
我给你一个既“物理可解释”又“工程可控”的公式模板，它本质上对应红外成像常见的：增益/偏置校正 + 反射项扣除 + 大气/环境项扣除 + 发射项增强。

### 3.1 先定义一个“物理启发的 IR 纠错核” $f_{phys}$

我们把 IR 的错误来源抽象成三类：
1.  **传感器响应误差**：gain/offset (增益/偏置)
2.  **反射/可见光相关污染**：与 RGB 亮度/照度相关（典型伪热点来源之一）
3.  **路径辐射/环境热背景**：低频背景/背景噪音

先把 RGB 变成一个可用于“反射项”的亮度图（不需要学习）：
$$ L^{rgb} = \text{Norm01}(\text{Gray}(I^{rgb})) \in \mathbb{R}^{B \times 1 \times H \times W} $$

**定义物理纠错（像素级）：**

1.  **增益/偏置校正：**
    $$ \tilde{I} = \frac{I^{ir} - o}{g} $$
2.  **扣除反射项与环境项：**
    $$ I_{emit} = \tilde{I} - \kappa \cdot L^{rgb} - b $$
3.  **最终输出（Clamp 保证范围）：**
    $$ I^{ir}_{new} = \text{Clamp}\left( \frac{I_{emit}}{\tau \cdot \epsilon + \epsilon_0}, 0, 1 \right) $$

**其中参数含义非常明确：**
*   $g > 0$: 传感器增益
*   $o$: 传感器偏置
*   $\kappa \ge 0$: 反射污染系数 (RGB 亮度泄漏到 IR 的强度)
*   $b$: 路径辐射/环境背景项
*   $\tau \in (0, 1]$: 大气透过率 (或整体衰减)
*   $\epsilon \in (0, 1]$: 等效发射率 (控制“真实热辐射”的占比)
*   $\epsilon_0$: 极小值，防止除零

你可以把它解释为：把 IR 里“看起来热但其实是反射/环境”的部分减掉，把“真实发射项”按物理意义重新归一化。

### 3.2 关键：这些物理参数由 MLP 从 "RGB 证据 + 语义区域" 预测出来

你提出 "MLP 调制生成超参用于物理模型"，我们就这么做：

**参数生成器 (Hyper-Param Generator):**
$$ \theta = \text{MLP}([y^{rgb}, u^{ir}, s(I^{ir})]) $$

*   $y^{rgb}$: RGB 语义证据（你希望从 RGB 获取的“知识”核心来源）
*   $u^{ir}$: IR 的语义方向（锚点）
*   $s(I^{ir})$: IR 的简单统计（可选但很有用）：均值、方差、分位数、动态范围等，帮助参数稳定。

**将 $\theta$ 映射物理参数并做范围约束 (很关键，保证物理意义和训练稳定):**
*   $g = \text{Softplus}(\theta_g) + 1$ (初始化接近 1)
*   $o = \theta_o$ (初始化接近 0)
*   $\kappa = \text{Softplus}(\theta_\kappa)$ (非负)
*   $b = \theta_b$
*   $\tau = \sigma(\theta_\tau)$ (范围 0-1)
*   $\epsilon = \sigma(\theta_\epsilon)$ (范围 0-1)

### 3.3 让“突出正确、屏蔽错误”在空间上发生：用 $M$ 做前景/背景双参数或空间混合

这是你目标落到实处的关键：错误往往发生在背景或非目标区域，你需要语义区域差异化的校正。

**写法 1：前景/背景两套物理参数 (经典、可解释)**
由 MLP 输出两套参数：$\theta_{fg}, \theta_{bg}$。用 $M$ 进行空间混合：
$$ \theta(x) = M(x) \cdot \theta_{fg} + (1 - M(x)) \cdot \theta_{bg} $$
然后对每个像素用对应的 $\theta(x)$ 做 $f_{phys}$。

**直觉非常强：**
*   **目标区域 (语义一致)** 用“增强真实热信号”的参数；
*   **背景区域** 用“强抑制反射/环境”的参数。

---

## 4) 把整个 new IR 方案写成一条“完整公式链” (论文可直接用)

1.  $$ u^{ir} = \text{Norm}(\text{Pool}(X^{ir})W_{ir \to t}), \quad w = \text{Softmax}\left( \frac{u^{ir} T^\top}{\tau} \right) $$
2.  $$ A^{rgb} = \text{Softmax}_N \left( \frac{(T W_q) (X^{rgb} W_k)^\top}{\sqrt{d_k}} \right), \quad Y^{rgb} = A^{rgb} (X^{rgb} W_v) $$
3.  $$ y^{rgb} = \sum_c w_c Y^{rgb}_c, \quad M = \text{Smooth}\left( \text{reshape}\left( \sum_c w_c A^{rgb}_c \right) \right) $$
4.  $$ \theta = \text{MLP}([y^{rgb}, u^{ir}, s(I^{ir})]) \Rightarrow \{g, o, \kappa, b, \tau, \epsilon\} $$
5.  $$ I^{ir}_{new} = f_{phys}(I^{ir}, L^{rgb}; \theta, M) $$

其中 $f_{phys}$ 用你选择的空间混合方式实现（前景/背景双参数 or mask 注入）。

---

## 5) 训练与稳定性 (你不可显式加 new IR 损失也能训，但要避免退化)

因为 new IR 是上游模块，下游检测损失会反传回来；这通常可行，但容易出现两个退化行为：
1.  **参数把 IR 拉到饱和** (全黑/全白) 以“讨好”某些梯度方向。
2.  **参数发散导致训练不稳定**。

**因此我建议你做三个不改变形式、但极大提升稳定性的工程措施：**
*   **参数范围约束**（上面已经做：sigmoid/softplus）。
*   **接近恒等初始化**: $g \approx 1, o \approx 0, \kappa \approx 0, b \approx 0, \tau \approx 1, \epsilon \approx 1$。让网络一开始“不破坏 IR”，再逐步学会纠错。
*   **轻量正则 (可选)**: $\| \theta - \theta_0 \|$ 或对 $M$ 做 TV 平滑，避免变化剧烈（不需要单独监督）。