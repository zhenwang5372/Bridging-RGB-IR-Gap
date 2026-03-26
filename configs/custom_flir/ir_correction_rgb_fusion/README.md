# Text-Guided RGB-IR Fusion (Scheme 2)

## 方案概述

本方案实现了**基于文本加权的逐类哈达姆门控 (Text-Weighted Class-wise Hadamard Gating)**，是对 IR 纠错方案的改进。

### 与方案一（V4）的区别

| 特性 | 方案一（V4 IR Correction） | 方案二（Text-Guided Fusion） |
|------|--------------------------|------------------------------|
| **架构** | `ir_correction` + `fusion_module` 分离 | 统一的 `text_guided_fusion` 模块 |
| **核心思想** | IR 特征纠错（负残差） | RGB-IR 协同融合（门控调制） |
| **Step 1** | 文本引导的语义激活 | ✅ 相同 |
| **Step 2** | 语义一致性度量（G） | ⭐ 类别重要性计算（w_c） |
| **Step 3** | 加权差异图（M_err） | ⭐ 加权哈达姆对齐（S_map） |
| **Step 4** | 错误特征提取（Error_map） | ⭐ 门控生成（Mask） |
| **Step 5** | 特征纠正（X_ir - α×Error_map） | ⭐ 特征融合（X_rgb×Mask + X_rgb） |
| **输出** | 纠正后的 IR 特征 | 融合后的特征（直接用于检测） |

## 核心流程

```
Input: RGB Features + IR Features + Text Features
  ↓
Step 1: 文本引导的语义激活
  Q = Text·W_Q
  K_rgb = Conv(X_rgb), K_ir = Conv(X_ir)
  A_rgb = Softmax(Q·K_rgb^T / √d_k)  [B, N, HW]
  A_ir = Softmax(Q·K_ir^T / √d_k)    [B, N, HW]
  ↓
Step 2: 类别重要性计算（⭐ 新增）
  gap_rgb = GlobalAvgPool(A_rgb)  [B, N]
  gap_ir = GlobalAvgPool(A_ir)    [B, N]
  w_c = Sigmoid(MLP([gap_rgb, gap_ir]))  [B, N, 1]
  ↓
Step 3: 加权哈达姆对齐（⭐ 新增）
  Ã_rgb = w_c · A_rgb
  Ã_ir = w_c · A_ir
  S_map = Σ(Ã_rgb ⊙ Ã_ir)  [B, 1, H, W]
  ↓
Step C: 门控生成（⭐ 新增）
  Mask = σ(β·X_ir + γ·S_map)  [B, C, H, W]
  ↓
Step D: 特征融合（⭐ 新增）
  X_fused = X_rgb · Mask + X_rgb
  ↓
Output: Fused Features [B, C, H, W]
```

## 关键参数

### β 和 γ（可学习参数）

```python
# 门控生成公式
Mask = σ(β · X_ir + γ · S_map)
```

- **β (beta)**: X_ir 的权重系数
  - **初始值**: 1.0
  - **原因**: IR backbone 已有 BN 归一化，特征在稳定范围内
  - **作用**: X_ir 提供基础的红外结构信息（主要信号）

- **γ (gamma)**: S_map 的权重系数
  - **初始值**: 0.5
  - **原因**: S_map 是一致性校验信号，作为辅助
  - **作用**: S_map 提供语义一致性引导（辅助信号）

### 类别权重 MLP

```python
self.class_weight_mlp = nn.Sequential(
    nn.Linear(2, 16),      # 输入: [gap_rgb, gap_ir]
    nn.ReLU(inplace=True),
    nn.Linear(16, 1),      # 输出: w_c ∈ [0, 1]
    nn.Sigmoid()
)
```

**物理含义**:
- 输入: 某个类别在 RGB 和 IR 上的平均激活强度
- 输出: 该类别的重要性权重
- 学习目标: 识别哪些类别在当前图像中重要/存在

## 文件结构

```
yolo_world/models/
├── necks/
│   └── ir_correction_rgb_fusion/
│       ├── __init__.py
│       └── text_guided_rgb_ir_fusion.py
│           ├── SingleLevelTextGuidedFusion
│           └── TextGuidedRGBIRFusion
│
└── backbones/
    └── ir_correction_rgb_fusion/
        ├── __init__.py
        └── dual_stream_backbone_with_text_guided_fusion.py
            └── DualStreamMultiModalYOLOBackboneWithTextGuidedFusion

configs/custom_flir/
└── ir_correction_rgb_fusion/
    ├── README.md
    └── yolow_v2_rgb_ir_flir_with_text_guided_fusion.py
```

## 使用方法

### 训练

```bash
cd /home/ssd1/users/wangzhen01/YOLO-World-master_2

# 单卡训练
python tools/train.py configs/custom_flir/ir_correction_rgb_fusion/yolow_v2_rgb_ir_flir_with_text_guided_fusion.py

# 多卡训练
bash ./tools/dist_train.sh configs/custom_flir/ir_correction_rgb_fusion/yolow_v2_rgb_ir_flir_with_text_guided_fusion.py 4
```

### 推理

```bash
python tools/test.py \
    configs/custom_flir/ir_correction_rgb_fusion/yolow_v2_rgb_ir_flir_with_text_guided_fusion.py \
    work_dirs/xxx/best_coco_bbox_mAP_50_epoch_xxx.pth
```

## 配置说明

### 关键配置项

```python
# 方案二融合参数
fusion_beta = 1.0   # X_ir 权重（已有 BN）
fusion_gamma = 0.5  # S_map 权重（一致性校验）

model = dict(
    backbone=dict(
        type='DualStreamMultiModalYOLOBackboneWithTextGuidedFusion',
        
        # RGB backbone: YOLOv8-s
        image_model=dict(
            type='YOLOv8CSPDarknet',
            deepen_factor=0.33,
            widen_factor=0.5,
        ),
        
        # IR backbone: LiteFFTIRBackbone
        ir_model=dict(
            type='LiteFFTIRBackbone',
            base_channels=32,  # P3=64, P4=128, P5=256
        ),
        
        # ⭐ 方案二融合模块
        text_guided_fusion=dict(
            type='TextGuidedRGBIRFusion',
            rgb_channels=[128, 256, 512],
            ir_channels=[64, 128, 256],
            text_dim=512,
            num_classes=4,
            beta=fusion_beta,
            gamma=fusion_gamma,
        ),
        
        # Text model: CLIP
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name='openai/clip-vit-base-patch32',
        ),
    ),
)
```

## 优势

1. **统一架构**: 避免了 `ir_correction` + `fusion_module` 的模块冲突
2. **细粒度控制**: 逐类别计算权重，保留最细粒度的语义信息
3. **可解释性强**: 
   - w_c 表示类别重要性
   - S_map 表示语义一致性
   - Mask 表示融合强度
4. **参数可学习**: β 和 γ 设为 `nn.Parameter`，网络自动学习最优值

## 实验记录

### 训练日志

训练完成后，日志会保存在：
```
work_dirs/ir_correction_rgb_fusion_scheme2/YYYYMMDD_HHMMSS/
├── YYYYMMDD_HHMMSS.log
├── vis_data/
└── best_coco_bbox_mAP_50_epoch_xxx.pth
```

### 预期输出示例

```
[TextGuidedRGBIRFusion] Building P3: RGB=128, IR=64, Text=512, Classes=4
[SingleLevelTextGuidedFusion] IR Channel Align: 64 -> 128
[TextGuidedRGBIRFusion] Building P4: RGB=256, IR=128, Text=512, Classes=4
[SingleLevelTextGuidedFusion] IR Channel Align: 128 -> 256
[TextGuidedRGBIRFusion] Building P5: RGB=512, IR=256, Text=512, Classes=4
[SingleLevelTextGuidedFusion] IR Channel Align: 256 -> 512
[TextGuidedRGBIRFusion] Initialized with 3 levels. Beta=1.0, Gamma=0.5
```

## 调试建议

### 1. 检查模块加载

```python
from yolo_world.models.necks.ir_correction_rgb_fusion import TextGuidedRGBIRFusion
from yolo_world.models.backbones.ir_correction_rgb_fusion import DualStreamMultiModalYOLOBackboneWithTextGuidedFusion

print("✅ Modules imported successfully!")
```

### 2. 可视化参数学习

训练过程中，β 和 γ 会自动学习。可以在代码中添加：

```python
# 在 TextGuidedRGBIRFusion.forward() 中
if self.training and iter_count % 100 == 0:
    logger.info(f"[Fusion Params] Beta={self.beta.item():.4f}, Gamma={self.gamma.item():.4f}")
```

### 3. 验证 S_map 范围

确保 S_map 归一化正确：

```python
# 在 SingleLevelTextGuidedFusion.forward() 中
print(f"S_map range: [{S_map.min():.4f}, {S_map.max():.4f}]")
print(f"S_map_norm range: [{S_map_norm.min():.4f}, {S_map_norm.max():.4f}]")
```

## 参考

- 方案文档: `strategy/ir_correction.md` - 方案二部分
- 方案一实现: `yolo_world/models/necks/text_guided_ir_correction/text_guided_ir_correction_v4.py`
- 原始融合: `yolo_world/models/necks/rgb_ir_fusion.py`
