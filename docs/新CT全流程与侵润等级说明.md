# 新 CT（NIfTI）全流程：当前能做什么、还缺什么

## 您问的是否都能做到？

| 环节 | 当前状态 | 说明 |
|------|----------|------|
| **投入新 CT（NIfTI 格式）** | ✅ 支持 | 推理与分割脚本均支持 `.nii` / `.nii.gz`。 |
| **自动勾画/分割结节** | ✅ 支持 | `ct_to_detection_and_contour.py`：nnU-Net 预测 mask → 轮廓叠加图 + 结节统计。需已训练好 nnU-Net 2D 并安装 nnUNetv2。 |
| **生成病例报告** | ✅ 支持 | `inference.py`：CT（取 2D slice）→ VLM → 所见/结论/建议/病理倾向。需已完成 VLM 训练（Vision+Bridge）。 |
| **按报告内容做侵润等级诊断** | ⚠️ 未实现 | 当前**没有**「侵润等级」专用模块；报告里只有自由文本的病理倾向，没有自动输出 AIS/MIA/IA 等分级。可后续加「从报告文本推断侵润倾向」或单独分类器。 |

结论：**在完成训练与参数学习后**，投入新 CT 可以做到：**自动分割/勾画结节 + 生成病例报告**。**侵润等级诊断**目前需要您额外设计（例如根据报告文本解析或加分类模型）。

---

## 一键全流程（推荐）

一条命令完成：**分割+勾画 → 报告 → 侵润倾向占位**（倾向为基于报告关键词的占位，非临床侵润等级）：

```powershell
python scripts/ct_full_pipeline.py --image "D:/path/to/new_ct.nii.gz" [--output_dir "D:/mamba-res/full_pipeline"]
```

输出目录下会有：`predicted_mask.nii.gz`、`overlay_contour.png`、`nodules.csv`、`report.txt`、`invasiveness_placeholder.txt` / `.json`。  
若已有 mask、只做报告+侵润占位：`--skip_segment --mask "mask路径"`。

---

## 分步执行（可选）

### 1. 分割 + 勾画

```powershell
python scripts/ct_to_detection_and_contour.py --image "D:/path/to/new_ct.nii.gz" --output_dir "D:/mamba-res/out"
```

得到：`predicted_mask.nii.gz`、`overlay_contour.png`、`nodules.csv`。

### 2. 病例报告生成（需已训练 VLM）

```powershell
python inference.py --image "D:/path/to/new_ct.nii.gz"
```

得到：文字报告（所见、结论、建议、病理倾向）。

---

## 侵润等级诊断：当前与可选做法

- **当前**：已提供**占位实现**：`scripts/infer_invasiveness_from_report.py` 根据报告文本做**关键词匹配**，输出「侵润/风险倾向」标签（如炎性倾向、肿瘤性待排等），并在一键全流程中写入 `invasiveness_placeholder.txt` / `.json`。**这不是临床侵润等级**（如 AIS/MIA/IA），仅供流程串联与后续替换。
- **可选扩展**：
  1. **替换占位逻辑**：在 `infer_invasiveness_from_report.py` 中改为您的规则或小模型，从报告（或影像）输出真实分级。
  2. **单独分类模型**：用影像+报告或仅影像训练侵润等级分类器，在流程中替换或补充当前占位步骤。

---

## 小结

- **是的**：在您完成训练与学习参数后，**投入新 CT（NIfTI）**可以：
  - **自动分割并勾画结节**（依赖 nnU-Net 2D + nnUNetv2）；
  - **生成病例报告**（依赖已训练的 VLM）；
  - **按报告内容做侵润/风险倾向占位**（关键词推断，见 `infer_invasiveness_from_report.py`；真实侵润等级需您替换为该逻辑或专用模型）。
- **一键全流程**：`scripts/ct_full_pipeline.py --image <新CT路径>` → 分割+勾画 → 报告 → 侵润倾向占位，结果统一落在 `--output_dir`。
