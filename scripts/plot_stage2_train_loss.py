#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 stage2 训练日志 CSV 绘制训练 loss 曲线，保存到 mamba-res，便于 SCI 使用。
用法:
  python scripts/plot_stage2_train_loss.py [--csv PATH] [--out PATH]
  --csv: 训练日志 CSV 路径，默认自动查找 mamba-res 或 outputs 下 *stage2*train*log*.csv
  --out: 输出图路径，默认 mamba-res/stage2_train_loss.png
"""
import argparse
import csv
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Plot stage2 train loss from CSV to mamba-res")
    p.add_argument("--csv", type=str, default="", help="Path to stage2_train_log.csv")
    p.add_argument("--out", type=str, default="", help="Output figure path (default: mamba-res/stage2_train_loss.png)")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    # mamba-res / outputs 在 autodl-tmp 下
    base_dir = repo_root.parent  # autodl-tmp
    mamba_res = base_dir / "mamba-res"
    outputs = base_dir / "outputs"

    csv_path = None
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"CSV not found: {csv_path}", file=sys.stderr)
            sys.exit(1)
    else:
        for base in (mamba_res, outputs):
            base = Path(base)
            if not base.exists():
                continue
            for f in base.rglob("*stage2*train*log*.csv"):
                csv_path = f
                break
            if csv_path is not None:
                break
        if csv_path is None:
            print("No stage2 train log CSV found under mamba-res or outputs.", file=sys.stderr)
            sys.exit(1)

    out_path = args.out
    if not out_path:
        # 默认: mamba-res/stage2_train_loss.png
        out_path = mamba_res / "stage2_train_loss.png"
        mamba_res.mkdir(parents=True, exist_ok=True)
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs, losses = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)  # skip header if any
        for row in r:
            if len(row) < 2:
                continue
            try:
                step = int(row[0])
                loss = float(row[1])
            except (ValueError, IndexError):
                continue
            epochs.append(step)
            losses.append(loss)

    if not epochs:
        print("No epoch/loss rows in CSV.", file=sys.stderr)
        sys.exit(1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skip figure.", file=sys.stderr)
        sys.exit(0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, losses, color="tab:blue", linewidth=1.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Caption loss")
    ax.set_title("Stage2 training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Training loss curve saved to: {out_path}")
    print("(SCI: use this path for 'Training loss curve' in Methods or Figure caption.)")

if __name__ == "__main__":
    main()
