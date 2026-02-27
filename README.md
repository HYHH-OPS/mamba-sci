# 医学 VLM：nnU-Net（轻下采样）+ Vim Bridge + Mamba-2

本仓库实现「视觉侧 nnU-Net 保持高分辨率（28×28）+ Vim 桥接 + 预训练 Mamba-2 不从头训练」的医学视觉-语言架构，面向 `d:\nnunet_*` 与 `d:\unn-net` 数据与权重。

## 创新点

- **Llama / Attention VLM**：受 O(L²) 限制，视觉特征被压成 **14×14**。
- **本方案（Mamba）**：O(L) 线性复杂度，保留 **28×28** 甚至更高分辨率特征。

## 架构概览

见 [ARCHITECTURE.md](ARCHITECTURE.md)。

- **Vision**：nnU-Net 2D 编码器，轻下采样 → 输出 28×28 或 32×32 特征图。
- **Bridge**：Vim (Vision Mamba) Block，2D → 1D 序列 → 双向 Mamba。
- **LLM**：Mamba-2.8B（或 OpenElm）从 HuggingFace 加载，不从头训练。

## 路径与配置

在 `config/paths.yaml` 中配置：

- `nnunet_raw`, `nnunet_preprocessed`, `nnunet_results`, `nnunet_data`
- `nnunet_encoder_checkpoint`：2D 最佳权重（如 Dataset503 的 fold_0）
- `caption_csv_train` / `caption_csv_val`：RadFM 风格 CSV（可指向 `d:\unn-net`）
- `mamba_hf_model`：如 `state-spaces/mamba-2.8b-hf`

## 依赖

```powershell
pip install -r requirements.txt
```

可选：安装 `mamba-ssm` 与 `causal-conv1d` 以启用 Vim 中的真实 Mamba 块（否则为 Linear 占位）。

## 使用示例

```python
import torch
from model import MedicalVLM
from data import MedicalVLMDataset, load_paths_config

# 视觉 + 桥接
config = load_paths_config()
config["bridge_d_model"] = 2560  # 与 Mamba-2.8B hidden_size 对齐
vlm = MedicalVLM(
    encoder_checkpoint=config.get("nnunet_encoder_checkpoint"),
    encoder_target_spatial=28,
    bridge_d_model=2560,
)
x = torch.randn(2, 1, 512, 512)
visual_tokens = vlm(x)  # [2, 784, 2560]

# 数据
ds = MedicalVLMDataset(
    config["caption_csv_train"],
    prompt_json_file=config["caption_prompt_json"],
)
sample = ds[0]
```

LLM 加载与视觉 token 与文本的融合见 `llm/mamba_loader.py` 与 `inference.py`。

---

## 训练得到的指标与验证、图像→文本（旧版，仅 Stage 1）

- **训练时**：每 50 step 打印 **train_loss**；每 epoch 结束打印 **train_loss_avg** 与 **val_loss**（验证集 loss）。最佳验证 loss 会保存为 `outputs/vision_bridge_best_val.pt`。
- **验证**：验证集 61 条会参与每轮 val_loss 计算；如需看生成质量，用下面的推理脚本。
- **图像→文本生成**：训练完 Vision+Bridge 后，运行推理脚本接上 Mamba 做报告生成：
  ```powershell
  # 单张图像
  python inference.py --image D:/nnunet_raw/Dataset503_.../imagesTr/xxx.nii.gz
  # 从验证集抽几条跑生成（问题+生成+参考）
  python inference.py --val_sample --num_val 5
  ```
  生成结果会打印在终端。注意：当前仅 Vision+Bridge 被训练，Mamba 未微调，生成质量有限；完整效果需后续接上「视觉+文本」联合训练（如 caption loss）。

## 训练流程总览

本仓库采用 **两阶段** 思路：

- **Stage 1（可选，结构验证）**：只训练 Vision+Bridge，使用代理损失，主要用于验证 nnU-Net 编码器 + Vim Bridge 前向是否正常、不会 OOM。
- **Stage 2（主训练）**：在冻结 Mamba 的前提下，**端到端解冻 Vision+Bridge**，用「图像 + 问题 → 报告」的 caption loss 做真正的医学 VLM 训练。

### Stage 1：Vision+Bridge 代理训练（可选）

- 脚本：`train.py`
- 作用：让 nnU-Net 编码器 + Vim Bridge 在你的数据上跑通，检查显存占用、前向/反向是否正常。
- 命令示例（Windows 本机）：

```powershell
conda activate mamba5090
cd D:\mamba
python train.py --epochs 3 --batch_size 4 --lr 1e-5
```

- 训练日志：每 10 step 打印 `train_loss`，每 epoch 结束打印 `train_loss_avg` 和（如有验证集）`val_loss`。
- Checkpoint：
  - `outputs/vision_bridge_stepXXXX.pt`
  - `outputs/vision_bridge_best_val.pt`
  - `outputs/vision_bridge_final.pt`
- 注意：如果 Stage 1 的 proxy loss 非常接近 0，说明视觉特征几乎被压扁，不建议拿这个权重继续做 Stage 2，可直接在 Stage 2 从随机初始化开始训练。

### Stage 2：医学 VLM 图文对齐训练（主训练）

- 脚本：`train_vlm.py`
- 数据格式：由 `data/medical_vlm_dataset.py` 定义，训练 CSV 至少包含：
  - `image_path`: 指向 CT 的 `.nii` / `.nii.gz`
  - `answer`: 医学报告文本
  - 可选 `question`: 提示问题；若为空，则自动从 `caption_prompt_json` 或内置中文模板中采样
  - 可选 `mask_path`: 病灶 mask，用于自动选择最佳切片
  - 可选 `grade`: 四级分级标签（AAH/AIS/MIA/IAC），用于侵润/分级辅助任务

- 关键参数说明：
  - `--epochs`: 训练轮数，医学报告较复杂，可设为 20–40
  - `--batch_size`: batch 大小，受显存限制
  - `--lr`: 学习率，端到端解冻 Vision+Bridge 时建议 `1e-5`
  - `--max_visual_tokens`: 视觉 token 上限，建议和池化大小匹配（如 12×12=144）
  - `--max_text_len`: 问题 + 报告总 token 长度上限，默认 512，如报告较长可设为 640/768
  - `--gradient_accumulation_steps`: 梯度累积步数，小显存机器可以用累积换 batch
  - `--save_every_steps`: 每 N step 额外保存一次权重，0 表示仅在每个 epoch 结束时保存
  - `--csv`: 直接指定训练 CSV；若不给则从 `config/paths.yaml` 的 `caption_csv_train` 读取
  - `--mamba_model`: Mamba 预训练权重（HF id 或本地路径），如 `state-spaces/mamba-2.8b-hf`
  - `--llm_8bit`: 以 8-bit 量化方式加载 LLM，显著节省显存
  - `--align_vocab`: 训练端对齐 tokenizer/embedding 词表，推荐开启

- 命令示例（Windows / WSL，本地有 `state-spaces/mamba-2.8b-hf`）：  

```powershell
cd D:\mamba
python train_vlm.py `
  --epochs 30 `
  --batch_size 4 `
  --lr 1e-5 `
  --max_visual_tokens 144 `
  --max_text_len 512 `
  --gradient_accumulation_steps 1
```

或者显式指定 CSV 与本地 Mamba 路径（推荐在服务器上这样做）：

```bash
python train_vlm.py \
  --epochs 30 \
  --batch_size 4 \
  --lr 1e-5 \
  --max_visual_tokens 144 \
  --max_text_len 512 \
  --csv /data/ct_reports/train_radfm.csv \
  --mamba_model /data/models/mamba-2.8b-hf \
  --align_vocab \
  --llm_8bit
```

- 训练时输出：
  - 首先会统计文本 token 长度分布，提示是否有严重截断风险。
  - 每个 step 打印 `total_loss`、`caption_loss`、`cls_loss`。
  - 每个 epoch 结束保存 `outputs/vision_bridge_vlm_final.pt`，并记录最近若干 epoch 的平均 loss。
  - 训练日志保存在 `outputs/stage2_train_log.csv`，可自行画 loss 曲线。

## 图像→文本推理与验证

当 Stage 2 训练完成后，可使用 `inference.py` 加载训练好的 Vision+Bridge + Mamba 做图像→报告生成。

- 单张图像推理（Windows 示例）：

```powershell
python inference.py --image D:/nnunet_raw/Dataset503_.../imagesTr/xxx.nii.gz
```

- 从验证集随机抽样若干条，查看生成与参考报告（需在 `config/paths.yaml` 中配置 `caption_csv_val`）：

```powershell
python inference.py --val_sample --num_val 5
```

- 常用选项：
  - `--checkpoint`: 指定 Vision+Bridge 的 checkpoint，默认在 `outputs/` 下自动寻找 `vision_bridge_vlm_final.pt` / `vision_bridge_best_val.pt`。
  - `--mamba_model`: 与训练时一致，确保加载相同 Mamba 权重。
  - `--constrained_decode`: 强制输出“所见/结论/建议/病理倾向”四段标题。
  - `--draw_nodule_contour` + `--mask`: 同时输出结节勾画 PNG 与结节统计 CSV。

生成结果会打印在终端，并落盘到 `D:/mamba-res/run_时间戳/`。

## 在服务器 / WSL 上训练与验证

- **路径配置**：
  - Windows 本机默认使用 `d:\nnunet_*` 与 `d:\unn-net`，可在 `config/paths.yaml` 中改为你自己的路径。
  - 代码在 Linux / WSL 下会自动把形如 `D:\...` 的路径转换为 `/mnt/d/...`，前提是你在 WSL 中挂载了对应盘符。
  - 如果是在纯 Linux 服务器（没有 Windows 盘），建议手动编辑 `config/paths.yaml`，把所有路径改成服务器上的绝对路径。

- **Mamba 模型离线使用**：
  - 推荐在一台能访问 HuggingFace 的机器上先下载 `state-spaces/mamba-2.8b-hf` 到本地目录（例如 `/data/models/mamba-2.8b-hf`），
  - 然后在服务器上通过 `--mamba_model /data/models/mamba-2.8b-hf` 指定该本地路径，并按需配置 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`。

- **训练/验证流程建议**：
  1. 在本机或服务器上完成数据 CSV + NIfTI 的准备，并测试 `MedicalVLMDataset` 是否能正常读取一两个样本。
  2. 可选：运行 `python train.py` 做一次小规模 Stage 1 训练，确认 Vision+Bridge 前向/显存正常。
  3. 运行 `python train_vlm.py` 做主训练，观察 `caption_loss` 是否随 epoch 稳定下降。
  4. 训练完成后，用 `python inference.py --val_sample` 或 `--image` 做图像→报告生成，人工检查报告质量。

