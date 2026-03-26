# 排除特定图片后计算mAP

本脚本用于从日志文件中提取需要排除的图片（漏检图片和高置信度FP图片），然后重新计算mAP、mAP50、mAP75。

## 功能说明

脚本会：
1. 从日志文件中解析出需要排除的图片：
   - 漏检（FN）的图片
   - 置信度大于0.5的误检（FP）图片
2. 对于 score_thr=0 和 score_thr=0.001 分别处理
3. 排除这些图片后，重新计算 mAP、mAP50、mAP75
4. 输出详细的统计信息

## 文件说明

- `calculate_map_after_filtering.py`: 主程序脚本
- `run_calculate_map.sh`: 运行脚本（需要配置路径）
- `README_MAP_FILTER.md`: 本说明文档

## 使用方法

### 方法1：使用Shell脚本（推荐）

1. 编辑 `run_calculate_map.sh`，修改以下变量：

```bash
# PKL文件路径（需要根据实际情况修改）
PKL_SCORE0="path/to/your/results_score0.pkl"
PKL_SCORE0001="path/to/your/results_score0001.pkl"

# COCO标注文件路径（如果路径不对，也需要修改）
ANN_FILE="data/LLVIP/coco_annotations/test.json"
```

2. 运行脚本：

```bash
cd /home/ssd1/users/wangzhen01/YOLO-World-master_2
bash tools/analysis/with_text_update/run_calculate_map.sh
```

### 方法2：直接运行Python脚本

```bash
cd /home/ssd1/users/wangzhen01/YOLO-World-master_2

python tools/analysis/with_text_update/calculate_map_after_filtering.py \
    --log-file tools/analysis/with_text_update/map50_analysis_report_text_update.log \
    --pkl-score0 <你的pkl文件路径_score0> \
    --pkl-score0001 <你的pkl文件路径_score0001> \
    --ann-file data/LLVIP/coco_annotations/test.json \
    --output tools/analysis/with_text_update/filtered_map_results.txt
```

## 参数说明

- `--log-file`: 日志文件路径（默认：`map50_analysis_report_text_update.log`）
- `--pkl-score0`: score_thr=0 的预测结果pkl文件路径（必需）
- `--pkl-score0001`: score_thr=0.001 的预测结果pkl文件路径（必需）
- `--ann-file`: COCO格式的标注文件路径（必需）
- `--output`: 输出结果文件路径（默认：`filtered_map_results.txt`）

## 输出说明

脚本会输出以下信息：

### 终端输出
- 排除的图片统计（漏检、高置信度FP、合并后总数）
- 原始数据统计（图片数、标注数、预测数）
- 过滤后数据统计
- mAP、mAP50、mAP75 的计算结果

### 文件输出
在指定的输出文件中保存详细的统计信息和mAP结果。

## 示例输出

```
================================================================================
计算排除特定图片后的mAP
================================================================================

步骤 1: 解析日志文件...

排除的图片统计:
  score_thr=0:
    漏检图片数: 19
    高置信度FP图片数: 567
    合并后排除图片数: 580

  score_thr=0.001:
    漏检图片数: 63
    高置信度FP图片数: 404
    合并后排除图片数: 450

================================================================================
处理 score_thr=0
================================================================================
...
结果:
  mAP:    0.xxxx
  mAP50:  0.xxxx
  mAP75:  0.xxxx

================================================================================
处理 score_thr=0.001
================================================================================
...
结果:
  mAP:    0.xxxx
  mAP50:  0.xxxx
  mAP75:  0.xxxx
```

## 依赖环境

脚本需要以下Python包：
- pickle
- json
- numpy
- pycocotools

如果缺少依赖，可以使用以下命令安装：

```bash
pip install numpy pycocotools
```

## 注意事项

1. **PKL文件路径**: 脚本需要原始的预测结果pkl文件，这些文件通常是运行检测模型时生成的。你需要找到这些文件的实际路径。

2. **日志文件格式**: 脚本假设日志文件的格式与 `map50_analysis_report_text_update.log` 一致。

3. **COCO标注格式**: 标注文件必须是标准的COCO格式JSON文件。

4. **内存使用**: 对于大型数据集，脚本可能需要较多内存。

## 常见问题

### Q: 找不到PKL文件怎么办？
A: PKL文件通常在运行检测模型时生成。请检查你的模型训练/测试输出目录，或者重新运行检测生成这些文件。

### Q: 如何生成PKL文件？
A: 通常需要运行 MMDetection 的测试脚本，并设置输出选项。例如：
```bash
python tools/test.py <config> <checkpoint> --out results.pkl
```

### Q: 输出的mAP值为0？
A: 可能的原因：
- PKL文件格式不正确
- 没有检测结果
- 图片ID映射出现问题

请检查终端输出的详细信息来定位问题。

## 联系方式

如有问题，请联系开发者或查看项目文档。
