# Windows + Conda 下直接训练（不依赖 WSL）

在 **Windows** 下用 **Conda 环境**（如 `mamba5090`）即可完成 Stage 2 训练，无需 WSL。路径使用 Windows 格式（如 `D:\mamba`、`D:/mamba/...`）。

## 一、环境准备

1. **激活 Conda 环境：**
   ```powershell
   conda activate mamba5090
   ```

2. **进入项目目录：**
   ```powershell
   cd D:\mamba
   ```

3. **（可选）使用本地 Mamba 模型**（避免连 Hugging Face）：  
   若已将 Mamba 下载到 `D:\mamba\models\mamba-2.8b-hf`，可设置环境变量，脚本会自动传给训练：
   ```powershell
   $env:MAMBA_MODEL = "D:/mamba/models/mamba-2.8b-hf"
   ```
   也可在运行训练时直接加参数（见下）。

## 二、训练前数据清洗（强烈建议）

若生成出现「电话、客服、左肺上叶重复」等幻觉，多半是训练集含垃圾文本或图文错位。训练前请先清洗 CSV：

```powershell
python scripts/clean_caption_csv.py
# 若 CSV 有 image_path 列，建议同时剔除文件不存在的行，防止图文错位：
python scripts/clean_caption_csv.py --drop_missing_paths
```

清洗后会在同目录生成 `*_clean.csv`。将 `config/paths.yaml` 中 `caption_csv_train` / `caption_csv_val` 改为指向 `*_clean.csv` 后再训练。

## 三、一键续训（推荐）

```powershell
.\run_stage2_continue.ps1
```

该脚本会：
- 切到 `D:\mamba`（若存在）
- 设置 `KMP_DUPLICATE_LIB_OK=TRUE` 避免 OpenMP 冲突
- 调用 `train_vlm.py`；若已设置 `MAMBA_MODEL`，会自动加上 `--mamba_model`

## 四、手动运行训练（等价命令）

若不想用脚本，可完全在 Conda 下执行：

```powershell
cd D:\mamba
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
conda activate mamba5090

# 使用本地 Mamba（二选一）
$env:MAMBA_MODEL = "D:/mamba/models/mamba-2.8b-hf"

python train_vlm.py --epochs 20 --batch_size 1 --lr 2e-5 --max_visual_tokens 96 --max_text_len 512
```

若 **loss 震荡或幻觉严重**，可降低学习率：`--lr 1e-5`。

若未设置 `MAMBA_MODEL`，且要用本地模型，可直接写：

```powershell
python train_vlm.py --epochs 20 --batch_size 1 --lr 2e-5 --max_visual_tokens 96 --mamba_model "D:/mamba/models/mamba-2.8b-hf"
```

## 五、路径说明

- **config/paths.yaml** 里使用 Windows 路径即可，例如 `d:/mamba/...`、`d:/nnunet_raw`。  
- **CSV 里的 image_path** 可以是 Windows 路径（如 `C:\...` 或 `D:\...`）。  
- 在 Windows 下运行不会做 WSL 的 `/mnt/d/` 转换，代码会原样使用这些路径。

## 六、Stage 1 与视觉退化

若你跑过 **Stage 1**（`train.py` 的 proxy loss）且 loss 很快掉到 0.001 以下，视觉特征可能退化为近零，Mamba 会「看不见图」而盲猜报告。此时 Stage 2 请用 **`--from_scratch`** 重新初始化 Vision+Bridge，不要用 Stage 1 的 checkpoint：

```powershell
python train_vlm.py --from_scratch --epochs 20 --batch_size 1 --lr 2e-5 --max_visual_tokens 96
```

## 七、显存与 RTX 5090 / 关闭 CMI 防 OOM

- **显存爆炸（31GB 顶满）**：多为 `config/paths.yaml` 中 **use_cmi: true** 导致。CMI 模块极耗显存，建议改为 **use_cmi: false**，并设 **roi_side: null**、**cmi_compress_to: null**，显存可降至约 12–16GB。随后可用 **`.\run_stage2_stable.ps1`**（from_scratch + 梯度累积 8，无 8-bit）。
- 若未安装 mamba-ssm / causal-conv1d 的 CUDA 扩展，程序会自动用 **reference/slow path**，可正常训练，仅速度较慢。  
- 显存紧张可加：`--max_visual_tokens 64` 或 `--max_text_len 256`，或 `--gradient_accumulation_steps 2`。  
- 更多 PyTorch / 5090 说明见 `docs/STAGE2_AND_PYTORCH.md`。

## 八、测试推理与重复/幻觉

若推理时出现**大量重复**（如「左肺上叶…」循环）或电话号码等乱码：
- 提高重复惩罚：`--repetition_penalty 1.5`
- 可尝试采样打破循环：`--do_sample --temperature 0.7 --top_p 0.9`
- 确保训练前已跑 `clean_caption_csv.py` 并改用 `*_clean.csv`；观察训练首行打印的 `[debug] 样本0: prompt token 数=..., answer token 数=...` 确认 Loss 只对回答部分计算。

若**生成内容单薄**（只有「两肺对称」等、缺病灶定位/尺寸）：
- 推理时可试：`--do_sample --temperature 0.7` 增加多样性。
- 训练端：增加 epoch（默认已改为 30，可设 `$env:STAGE2_EPOCHS=40`）；并用下面「数据丰富性」减少无异常样本。

## 九、进一步优化（数据平衡 / 显存 / 视觉退化）

- **数据平衡**：若 caption 里大量「无异常」短报告，模型会学捷径。生成 CSV 或清洗时只保留病理丰富的样本：
  - `python scripts/clean_caption_csv.py --min_answer_chars 50 --require_keywords "结节|mm|肺叶|段|病灶"`（输出仍为 `*_clean.csv`）
  - 或生成时：`python scripts/excel_total_to_caption_csv.py ... --min_answer_chars 50 --require_keywords "结节|mm|肺叶|段|病灶"`
- **显存吃紧（如 31/31.5 GB 封顶）**：训练前设置 `$env:LOW_VRAM = "1"` 再运行 `.\run_stage2_continue.ps1`（自动改为 `--max_visual_tokens 64 --max_text_len 256`）；或加 `$env:LLM_8BIT = "1"` 启用 8-bit 加载。推理时同样加 `--max_visual_tokens 64` 与训练一致。
- **视觉特征退化**：若 Stage 1 proxy loss 曾极低且生成与图像对不上，Stage 2 请用 `--from_scratch` 重新初始化 Bridge（见第六节）。
- **训练轮数**：默认 30 epoch；需更多可设 `$env:STAGE2_EPOCHS = "40"` 再运行脚本。
- **一键「重新对齐 + 省显存」**（生成截断/看图不准时建议）：直接运行  
  `.\run_stage2_from_scratch_lowvram.ps1`  
  等效于：`--from_scratch --llm_8bit --max_visual_tokens 64 --max_text_len 256 --batch_size 1`。推理时须用相同参数：`--max_visual_tokens 64 --llm_8bit`。

## 十、测试推理

训练完成后可用同一 Conda 环境测试（若用了 LOW_VRAM，此处改为 `--max_visual_tokens 64`）：

```powershell
python inference.py --val_sample --num_val 1 --checkpoint outputs/vision_bridge_vlm_final.pt --max_visual_tokens 96 --llm_device auto
```

若使用本地 Mamba：

```powershell
python inference.py --val_sample --num_val 1 --checkpoint outputs/vision_bridge_vlm_final.pt --max_visual_tokens 96 --mamba_model "D:/mamba/models/mamba-2.8b-hf" --llm_device auto
```

以上全部在 **Windows + Conda** 下完成，无需 WSL。
