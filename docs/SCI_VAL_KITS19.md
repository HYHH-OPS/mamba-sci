# 最终模型在公共数据集 KiTS19 上的验证（SCI 论文用）

## 1. 公共验证集说明

- **数据集**：KiTS19（公共数据集），对应配置中的 `caption_csv_val_public_kits19`。
- **CSV 路径**：`/autodl-tmp/caption_val_linux.csv`（在 `config/paths.yaml` 中已配置）。
- **CSV 列**：`image_path`, `question`, `answer`, `mask_path`（无 `grade` 列时仅计算报告质量指标，不计算分级准确率/F1）。

## 2. 最终模型验证命令（KiTS19，含 WandB）

使用**与训练一致**的 checkpoint 与超参，在 KiTS19 上跑验证，结果落盘并同步到 WandB，便于写论文。

```bash
conda activate mamba_a800
cd /root/autodl-tmp/mamba-sci

python inference.py \
  --val_sample \
  --val_public \
  --num_val 50 \
  --checkpoint /root/autodl-tmp/mamba-res/stage2_grade4_auto/vision_bridge_vlm_final.pt \
  --mamba_model /autodl-tmp/models/mamba-2.8b-hf \
  --max_visual_tokens 144 \
  --max_new_tokens 512 \
  --num_beams 1 \
  --out_dir /root/autodl-tmp/mamba-res/val_public_kits19
```

- `--val_public`：自动使用 `paths.yaml` 中的 KiTS19 验证集（`caption_csv_val_public_kits19`）。
- `--num_val 50`：验证样本数，可按 KiTS19 验证集实际条数改为全量（如 100）。
- 若 checkpoint 不在 `mamba-res`，请将 `--checkpoint` 改为你最终模型权重路径。

## 3. 论文所需指标与落盘位置

验证结束后会生成以下内容，供 SCI 论文使用。

### 3.1 本地文件（`out_dir/run_YYYYMMDD_HHMMSS/`）

| 文件 | 用途 |
|------|------|
| **val_metrics.json** | 汇总指标：`bleu4`、`rouge_l`；若 CSV 含 `grade` 还有 `grade_accuracy`、`grade_macro_f1`、`confusion_matrix` |
| **meta.json** | 含 `val_metrics` 与各样本信息，便于复现与补充材料 |
| **sample_*_gen.txt / sample_*_ref.txt** | 逐条生成与参考报告，用于定性分析或示例展示 |
| **wandb_run_info.json** | 该次 run 的 WandB 链接与名称，便于在方法/附录中注明 |

### 3.2 WandB 中可见

- **val/bleu4**：报告生成质量（生成 vs 参考）。
- **val/rouge_l**：报告生成质量（需安装 `rouge-score`）。
- **val/samples_table**：每条样本的 `grade_pred`、`grade_gt`、生成/参考预览。
- 若验证集带 `grade` 列：**val/grade_accuracy**、**val/grade_macro_f1**、**val/confusion_matrix**（图或表）。

### 3.3 论文中可写的表述与表格

- **方法**：说明最终模型在**公共数据集 KiTS19**上验证，报告质量采用 BLEU-4、ROUGE-L；若使用带分级的验证集，可补充分级准确率与宏 F1。
- **结果**：从 `val_metrics.json` 或 WandB 摘录 **Val (KiTS19)** 的 BLEU-4、ROUGE-L（及可选的分级指标），制成表格；混淆矩阵可直接使用 WandB 或本地导出的图。
- **复现**：在附录或补充材料中注明 WandB run 链接（见 `wandb_run_info.json`）及 `caption_csv_val_public_kits19` 指向的 CSV 路径。

## 4. 可选：仅指定 CSV 而不改 config

若希望临时指定其他 KiTS19 格式 CSV，可不用 `--val_public`，直接传 `--csv_val`：

```bash
python inference.py \
  --val_sample \
  --csv_val /path/to/your_kits19_val.csv \
  --num_val 50 \
  --checkpoint /root/autodl-tmp/mamba-res/stage2_grade4_auto/vision_bridge_vlm_final.pt \
  --mamba_model /autodl-tmp/models/mamba-2.8b-hf \
  --max_visual_tokens 144 \
  --max_new_tokens 512 \
  --out_dir /root/autodl-tmp/mamba-res/val_public_kits19
```

## 5. 依赖（指标计算）

- **BLEU-4**：`pip install nltk`，必要时 `python -c "import nltk; nltk.download('punkt')"`。
- **ROUGE-L**：`pip install rouge-score`。
- **分级指标**（仅当 CSV 含 `grade` 列）：`scikit-learn`（通常已安装）。

上述命令与配置已按「公共数据集 = KiTS19、最终模型验证、SCI 论文所需指标」设置好，可直接用于最终模型验证与论文撰写。
