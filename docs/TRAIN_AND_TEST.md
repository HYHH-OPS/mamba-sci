# 训练指令与检测（图像→文本）

## 一、训练指令

### 1. 环境

```powershell
cd D:\mamba
conda activate mamba5090
```

### 2. 基础训练（若干轮 + 验证）

```powershell
python train.py --epochs 10 --batch_size 4 --lr 1e-4
```

### 3. 夜间 300 轮 + 超参筛选（多组 lr/batch_size，选最佳）

```powershell
./run_full_train.ps1
```

只跑 300 轮、不搜参：

```powershell
python train_vlm.py --epochs 30 --batch_size 4 --lr 1e-5
```

### 4. 训练产出

| 文件 | 说明 |
|------|------|
| `outputs/vision_bridge_best_val.pt` | 验证集 val_loss 最佳权重 |
| `outputs/vision_bridge_final.pt` | 最后一轮权重 |
| `outputs/vision_bridge_vlm_final.pt` | 夜间脚本筛选出的最佳参数对应权重 |
| `outputs/overnight_best_summary.txt` | 夜间多组参数对比与最佳 lr/batch_size |

---

## 二、检测代码：丢入图像能否生成文本

训练完成后，用下面任一方式检测「图像 → 文本」是否可用。

### 方式 A：单张 NIfTI 图像

```powershell
python inference.py --image "D:\nnunet_raw\Dataset503_TBLesion_327\imagesTr\某例_0000.nii.gz"
```

指定用某次训练得到的权重（不指定则自动用 `outputs/vision_bridge_best_val.pt` 或 `vision_bridge_final.pt`）：

```powershell
python inference.py --checkpoint outputs/vision_bridge_vlm_final.pt --image "D:\nnunet_raw\Dataset503_TBLesion_327\imagesTr\某例_0000.nii.gz"
```

### 方式 B：用验证集几条做检测（推荐）

不指定图像，从验证集抽几条跑一遍，看「问题 + 生成 + 参考」：

```powershell
python inference.py --val_sample --num_val 3
```

### 方式 C：一键检测脚本（最小命令）

```powershell
python inference.py --val_sample --num_val 1
```

会默认用 `outputs/vision_bridge_best_val.pt`（或 `overnight_best` / `final`），从验证集取 1 张图，打印「输入图像」和「生成文本」，用于确认流程是否跑通。

指定图像和权重：

```powershell
python inference.py --checkpoint outputs/vision_bridge_vlm_final.pt --image "D:\nnunet_raw\Dataset503_TBLesion_327\imagesTr\xxx_0000.nii.gz"
```

---

## 三、预期输出示例

检测通过时，终端会看到类似：

```
加载 Vision+Bridge...
加载 Mamba LLM...
输入图像: D:\nnunet_raw\...\xxx_0000.nii.gz
问题: 请分析这幅胸部CT影像，并给出详细的诊断报告。
生成文本: 影像所见: ... 诊断意见: ...
```

若只训练了 Vision+Bridge、未微调 Mamba，生成内容可能较泛；能稳定输出上述格式即表示「参数 + 图像 → 文本」流程可用。

---

## 四、Hugging Face 连接失败 (WinError 10060)

若运行检测/推理时出现「连接超时」「WinError 10060」等，说明无法访问 Hugging Face 下载 Mamba。解决办法见 **README.md**，常用两种：

1. **使用国内镜像**（运行前执行）：
   ```powershell
   $env:HF_ENDPOINT = "https://hf-mirror.com"
   python inference.py --val_sample --num_val 1
   ```
   或直接运行：先设置 `HF_ENDPOINT` 再运行 `inference.py`

2. **使用本地已下载模型**：
   ```powershell
   python inference.py --val_sample --num_val 1 --mamba_model D:/mamba/models/mamba-2.8b-hf
   ```
