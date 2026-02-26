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

## 训练得到的指标与验证、图像→文本

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
