"""
医学 VLM 训练：使用 config/paths.yaml 中的 2D/3D 数据与病例文档（CSV）进行训练。

数据与文档路径见 docs/DATA_PATHS.md。
用法（RTX 5090）:
  1. conda activate mamba5090  或先执行 .\setup_mamba_5090.ps1
  2. cd D:\mamba
  3. python train.py [--epochs 3] [--batch_size 4] [--lr 1e-4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 保证项目根在 path 中
REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
from torch.utils.data import DataLoader

from data.medical_vlm_dataset import MedicalVLMDataset, load_paths_config
from model.forward_medical_vlm import build_medical_vlm_from_config


def get_config():
    """加载 config/paths.yaml，若不存在则用默认路径。"""
    config = load_paths_config(REPO / "config" / "paths.yaml")
    # 默认训练相关
    config.setdefault("encoder_output_stage", 4)
    config.setdefault("encoder_target_spatial", 28)
    config.setdefault("bridge_d_model", 2560)
    config.setdefault("bridge_bidirectional", True)
    return config


def main():
    parser = argparse.ArgumentParser(description="医学 VLM 训练 (nnU-Net + Vim + Mamba)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5, help="Stage 1 建议 1e-5；过大(如 1e-4)+小数据易使 proxy loss 趋近 0，视觉特征退化，不宜作 Stage 2 初始化")
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/mamba-res/train_outputs", help="训练产物目录，默认 /root/autodl-tmp/mamba-res/train_outputs")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--no_wandb", action="store_true", help="禁用 WandB 记录")
    args = parser.parse_args()

    config = get_config()
    csv_train = config.get("caption_csv_train")
    csv_val = config.get("caption_csv_val")
    prompt_json = config.get("caption_prompt_json")

    # 检查数据与文档路径
    for name, p in [("train CSV", csv_train), ("val CSV", csv_val)]:
        if not Path(p).exists():
            print(f"[WARN] {name} 不存在: {p}，请检查 config/paths.yaml 或 docs/DATA_PATHS.md")
    if not Path(csv_train).exists():
        print("训练 CSV 缺失，退出。")
        return 1

    # 数据集（使用你处理好的 2D/3D 对应文档：CSV 的 image_path 指向 raw NIfTI，取 2D slice）
    train_ds = MedicalVLMDataset(
        csv_train,
        prompt_json_file=prompt_json,
        patch_size=args.patch_size,
    )
    val_ds = None
    if Path(csv_val).exists():
        val_ds = MedicalVLMDataset(csv_val, prompt_json_file=prompt_json, patch_size=args.patch_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0) if val_ds else None

    # 模型：Vision + Vim Bridge（与 Mamba 融合的完整 VLM 可在后续接上）
    config["bridge_d_model"] = 2560
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_medical_vlm_from_config(config)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 简单代理损失：视觉 token 的 L2 正则（验证前向与梯度）；后续可接 LLM 做 caption loss
    def proxy_loss(visual_tokens: torch.Tensor) -> torch.Tensor:
        return visual_tokens.pow(2).mean()

    out_dir = Path(args.output_dir or "/root/autodl-tmp/mamba-res/train_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_config = {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "output_dir": str(out_dir)}
    use_wandb = False
    wandb_run_info_path = out_dir / "wandb_run_info.json"
    if not getattr(args, "no_wandb", False):
        import os
        wandb_key = os.environ.get("WANDB_API_KEY", "").strip()
        if len(wandb_key) < 40:
            wandb_key = ""
        if not wandb_key:
            for key_file in [out_dir / ".wandb_key", out_dir.parent / ".wandb_key", Path("/root/autodl-tmp/mamba-res/.wandb_key")]:
                if key_file.exists():
                    try:
                        wandb_key = key_file.read_text(encoding="utf-8").strip().split("\n")[0].strip()
                        if len(wandb_key) >= 40:
                            break
                        wandb_key = ""
                    except Exception:
                        pass
        if wandb_key and len(wandb_key) >= 40:
            try:
                import wandb
                os.environ["WANDB_API_KEY"] = wandb_key
                wandb.init(project="mamba-sci-vlm", name=out_dir.name or "stage1", config=train_config)
                use_wandb = True
                print("WandB 已启用。", flush=True)
            except Exception as e:
                print(f"WandB 初始化失败: {e}", flush=True)
    print(f"训练样本: {len(train_ds)}，验证: {len(val_ds) if val_ds else 0}，device: {device}", flush=True)
    print(f"输出目录: {out_dir}", flush=True)
    print("指标: 每 10 step 打印 train_loss；每 epoch 结束打印 val_loss（若有验证集）", flush=True)

    global_step = 0
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        print(f"开始 epoch {epoch+1}/{args.epochs} ...", flush=True)
        model.train()
        epoch_loss_sum = 0.0
        epoch_count = 0
        for i, batch in enumerate(train_loader):
            image = batch["image"].to(device)
            if image.dim() == 3:
                image = image.unsqueeze(1)
            visual_tokens = model(image)
            loss = proxy_loss(visual_tokens)
            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step += 1
            epoch_loss_sum += loss.item()
            epoch_count += 1
            if global_step == 1 or global_step % 10 == 0:
                print(f"epoch {epoch+1} step {global_step} train_loss {loss.item():.6f}", flush=True)
                if use_wandb:
                    try:
                        import wandb
                        log_d = {"train/loss": loss.item(), "train/epoch": epoch + 1, "step": global_step, "train/lr": opt.param_groups[0]["lr"]}
                        if device.type == "cuda":
                            log_d["system/gpu_alloc_gb"] = torch.cuda.memory_allocated(device) / (1024 ** 3)
                        wandb.log(log_d, step=global_step)
                    except Exception:
                        pass
            if args.save_every and global_step % args.save_every == 0:
                ckpt = out_dir / f"vision_bridge_step{global_step}.pt"
                torch.save({"step": global_step, "model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict()}, ckpt)
                print(f"  saved {ckpt}", flush=True)

        train_avg = epoch_loss_sum / max(epoch_count, 1)
        if train_avg < 0.001:
            print("  [注意] Stage 1 proxy loss 已极低，视觉特征可能退化为近零，不建议用本阶段权重初始化 Stage 2；Stage 2 请用 --from_scratch", flush=True)
        # 验证：每 epoch 在验证集上算一次 loss
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    image = batch["image"].to(device)
                    if image.dim() == 3:
                        image = image.unsqueeze(1)
                    visual_tokens = model(image)
                    val_loss_sum += proxy_loss(visual_tokens).item()
                    val_count += 1
            val_loss = val_loss_sum / max(val_count, 1)
            model.train()
            print(f"epoch {epoch+1} 结束 | train_loss_avg {train_avg:.6f} | val_loss {val_loss:.6f}", flush=True)
            if use_wandb:
                try:
                    import wandb
                    wandb.log({"train/epoch_avg_loss": train_avg, "val/loss": val_loss, "train/epoch": epoch + 1, "step": global_step}, step=global_step)
                except Exception:
                    pass
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_best = out_dir / "vision_bridge_best_val.pt"
                torch.save({"step": global_step, "model_state_dict": model.state_dict(), "val_loss": val_loss}, ckpt_best)
                print(f"  [best] 保存 best val: {ckpt_best}", flush=True)
        else:
            print(f"epoch {epoch+1} 结束 | train_loss_avg {train_avg:.6f}", flush=True)
            if use_wandb:
                try:
                    import wandb
                    wandb.log({"train/epoch_avg_loss": train_avg, "train/epoch": epoch + 1, "step": global_step}, step=global_step)
                except Exception:
                    pass

    if use_wandb:
        try:
            import wandb
            run_info = {"run_id": wandb.run.id, "run_url": wandb.run.get_url(), "project": wandb.run.project, "name": wandb.run.name, "config": train_config}
            wandb_run_info_path.write_text(__import__("json").dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"WandB 运行信息已保存: {wandb_run_info_path}", flush=True)
            wandb.finish()
        except Exception as e:
            print(f"WandB 收尾失败: {e}", flush=True)
    ckpt_final = out_dir / "vision_bridge_final.pt"
    torch.save({"step": global_step, "model_state_dict": model.state_dict()}, ckpt_final)
    print(f"训练完成，保存: {ckpt_final}", flush=True)
    train_avg_final = epoch_loss_sum / max(epoch_count, 1) if epoch_count else 0
    if train_avg_final < 0.001:
        print("提示: 本次 Stage 1 最终 loss 极低，该权重可能已「练废」（视觉输出趋近零）。Stage 2 请用 train_vlm.py --from_scratch，勿用本 checkpoint。", flush=True)
    print("验证与生成: 运行 python inference.py --image <nifti路径> 或 python inference.py --val_sample 进行图像→文本生成", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
