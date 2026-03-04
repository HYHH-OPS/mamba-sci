# mamba-sci 运行指令速查

所有结果默认落在 **`/root/autodl-tmp/mamba-res/`**，在 Cursor 中打开 `autodl-tmp → mamba-res` 即可查看。

环境与项目：
```bash
# 环境
/root/miniconda3/envs/mamba_a800/bin/python -V
cd /root/autodl-tmp/mamba-sci
```

**WandB**：训练时会自动从 `mamba-res/.wandb_key` 或环境变量 `WANDB_API_KEY` 读取密钥，记录 Loss、学习率、显存等；运行信息会写入当前输出目录下的 `wandb_run_info.json`，便于论文与汇报。详见 `mamba-res/WANDB_README.md`。禁用时加 `--no_wandb`。

---

## 一、训练

### 1. Stage 2 四分类训练（VLM，推荐）

产物目录：`/root/autodl-tmp/mamba-res/stage2_grade4_auto/`  
（checkpoint、日志、config 均在此）

```bash
cd /root/autodl-tmp/mamba-sci

/root/miniconda3/envs/mamba_a800/bin/python train_vlm.py \
  --epochs 20 \
  --batch_size 1 \
  --lr 2e-5 \
  --max_visual_tokens 144 \
  --mamba_model /autodl-tmp/models/mamba-2.8b-hf \
  --output_dir /root/autodl-tmp/mamba-res/stage2_grade4_auto
```

不写 `--output_dir` 时默认输出到：`/root/autodl-tmp/mamba-res/train_outputs/`。

### 2. Stage 1 训练（仅 Vision+Bridge）

```bash
cd /root/autodl-tmp/mamba-sci

/root/miniconda3/envs/mamba_a800/bin/python train.py \
  --epochs 3 \
  --batch_size 4 \
  --lr 1e-5 \
  --output_dir /root/autodl-tmp/mamba-res/stage1_outputs
```

---

## 二、验证

### 1. 验证私有数据（config 中验证集 CSV）

使用 `config/paths.yaml` 里的 `caption_csv_val`（当前为私有验证集）。

**快速验证（例如 3 条）：**
```bash
cd /root/autodl-tmp/mamba-sci

/root/miniconda3/envs/mamba_a800/bin/python inference.py \
  --val_sample \
  --num_val 3 \
  --checkpoint /root/autodl-tmp/mamba-res/stage2_grade4_auto/vision_bridge_vlm_final.pt \
  --mamba_model /autodl-tmp/models/mamba-2.8b-hf \
  --max_visual_tokens 144 \
  --max_new_tokens 512 \
  --out_dir /root/autodl-tmp/mamba-res/val_private
```

**全量验证（例如 23 条，建议后台）：**
```bash
cd /root/autodl-tmp/mamba-sci

screen -dmS val_private bash -lc '
cd /root/autodl-tmp/mamba-sci &&
/root/miniconda3/envs/mamba_a800/bin/python inference.py \
  --val_sample \
  --num_val 23 \
  --checkpoint /root/autodl-tmp/mamba-res/stage2_grade4_auto/vision_bridge_vlm_final.pt \
  --mamba_model /autodl-tmp/models/mamba-2.8b-hf \
  --max_visual_tokens 144 \
  --max_new_tokens 512 \
  --num_beams 1 \
  --out_dir /root/autodl-tmp/mamba-res/val_private \
  > /root/autodl-tmp/mamba-res/val_private/val_console.log 2>&1
'
# 查看进度
tail -f /root/autodl-tmp/mamba-res/val_private/val_console.log
```

若 checkpoint 不在 mamba-res 下，请把 `--checkpoint` 换成实际路径（如 `/autodl-tmp/outputs/stage2_grade4_auto/vision_bridge_vlm_final.pt`）。

### 2. 验证公共数据（指定验证集 CSV）

使用 `--csv_val` 指定公共验证集 CSV（如 `caption_val_linux.csv`），不传则使用 config 中的私有验证集。

```bash
cd /root/autodl-tmp/mamba-sci

/root/miniconda3/envs/mamba_a800/bin/python inference.py \
  --val_sample \
  --num_val 10 \
  --csv_val /autodl-tmp/caption_val_linux.csv \
  --checkpoint /root/autodl-tmp/mamba-res/stage2_grade4_auto/vision_bridge_vlm_final.pt \
  --mamba_model /autodl-tmp/models/mamba-2.8b-hf \
  --max_visual_tokens 144 \
  --out_dir /root/autodl-tmp/mamba-res/val_public
```

若 checkpoint 不在 mamba-res 下，请将 `--checkpoint` 改为实际路径。

---

## 三、功能验证（勾画、报告、诊断）

### 1. 单病例：报告 + 四分类诊断 + 结节勾画

一条命令同时得到：生成报告、侵润等级（AIS/MIA/IA/IAC）、结节勾画图与统计。

```bash
cd /root/autodl-tmp/mamba-sci

/root/miniconda3/envs/mamba_a800/bin/python inference.py \
  --image /root/autodl-tmp/ct/0000719802_20260124/ct.nii.gz \
  --mask /root/autodl-tmp/datasets/private_masks_aligned/0000719802_20260124.nii.gz \
  --draw_nodule_contour \
  --checkpoint /root/autodl-tmp/mamba-res/stage2_grade4_auto/vision_bridge_vlm_final.pt \
  --mamba_model /autodl-tmp/models/mamba-2.8b-hf \
  --max_visual_tokens 144 \
  --out_dir /root/autodl-tmp/mamba-res/single_case
```

结果在：`/root/autodl-tmp/mamba-res/single_case/run_YYYYMMDD_HHMMSS/`  
- **报告**：`generated.txt`  
- **诊断（四分类）**：`meta.json` 中的 `grade`（label、probs）  
- **勾画**：`nodule_contour/overlay_contour.png`、`nodule_contour/nodules.csv`

### 2. 仅报告（不勾画）

不加 `--draw_nodule_contour`、可不传 `--mask`（若 config 允许）：

```bash
/root/miniconda3/envs/mamba_a800/bin/python inference.py \
  --image /path/to/ct.nii.gz \
  --checkpoint /root/autodl-tmp/mamba-res/stage2_grade4_auto/vision_bridge_vlm_final.pt \
  --mamba_model /autodl-tmp/models/mamba-2.8b-hf \
  --max_visual_tokens 144 \
  --out_dir /root/autodl-tmp/mamba-res/single_report
```

### 3. 仅结节勾画（不跑 VLM）

使用脚本只做勾画与统计，不生成报告：

```bash
cd /root/autodl-tmp/mamba-sci

/root/miniconda3/envs/mamba_a800/bin/python scripts/nodule_overlay_and_stats.py \
  --image /path/to/ct.nii.gz \
  --mask /path/to/mask.nii.gz \
  --output_dir /root/autodl-tmp/mamba-res/nodules_only
```

输出：`nodules_only/overlay_contour.png`、`nodules_only/nodules.csv`。

### 4. 新 CT 全流程（分割 + 勾画 + 报告 + 侵润占位）

```bash
cd /root/autodl-tmp/mamba-sci

/root/miniconda3/envs/mamba_a800/bin/python scripts/ct_full_pipeline.py \
  --image /path/to/new_ct.nii.gz \
  --output_dir /root/autodl-tmp/mamba-res/full_pipeline \
  --checkpoint /root/autodl-tmp/mamba-res/stage2_grade4_auto/vision_bridge_vlm_final.pt
```

若已有 mask，可跳过分割只做报告+勾画：

```bash
/root/miniconda3/envs/mamba_a800/bin/python scripts/ct_full_pipeline.py \
  --image /path/to/ct.nii.gz \
  --mask /path/to/mask.nii.gz \
  --skip_segment \
  --output_dir /root/autodl-tmp/mamba-res/full_pipeline
```

---

## 四、结果位置速查

| 用途           | 默认/指定输出目录                     | 关键文件 |
|----------------|--------------------------------------|----------|
| Stage 2 训练   | `mamba-res/stage2_grade4_auto/`       | `vision_bridge_vlm_final.pt`、`stage2_train_log.csv` |
| Stage 1 训练   | `mamba-res/stage1_outputs/` 或自选   | `vision_bridge_final.pt` |
| 验证私有/公共  | `mamba-res/val_private` 或 `val_public` | `run_*/sample_*_gen.txt`、`run_*/meta.json` |
| 单病例全功能   | `mamba-res/single_case/run_*/`       | `generated.txt`、`meta.json`、`nodule_contour/` |
| 仅勾画         | `mamba-res/nodules_only/`            | `overlay_contour.png`、`nodules.csv` |

终端粘贴请用 **Ctrl+Shift+V**（或右键粘贴）。
