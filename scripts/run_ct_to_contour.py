"""
新 CT → 结节勾画 流程入口（仅勾画，不跑分割）。

- 若提供 --image 与 --mask：直接调用 nodule_overlay_and_stats，生成轮廓叠加图与结节统计。
- 若仅提供 --image：提示使用一键流程脚本 ct_to_detection_and_contour.py（先 nnU-Net 预测再勾画）。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(
        description="CT + mask → 结节轮廓叠加图与统计（仅勾画）。要「自动检测+勾画」请用 ct_to_detection_and_contour.py"
    )
    ap.add_argument("--image", required=True, help="CT NIfTI 路径")
    ap.add_argument("--mask", default=None, help="结节分割 mask NIfTI 路径；不传则提示使用一键流程")
    ap.add_argument("--output_dir", default="D:/mamba-res/nodules", help="输出目录")
    ap.add_argument("--line_width", type=float, default=1.8)
    ap.add_argument("--fill_alpha", type=float, default=0.22)
    args = ap.parse_args()

    if not args.mask:
        print("未提供 --mask，无法勾画。", file=sys.stderr)
        print("一键流程（新 CT → 自动检测 → 勾画）：", file=sys.stderr)
        print("  python scripts/ct_to_detection_and_contour.py --image <CT路径> [--output_dir ...]", file=sys.stderr)
        print("若已有 mask，可加 --mask 再运行本脚本，或：", file=sys.stderr)
        print("  python scripts/ct_to_detection_and_contour.py --image <CT> --skip_nnunet --mask <mask路径>", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        str(REPO / "scripts" / "nodule_overlay_and_stats.py"),
        "--image", args.image,
        "--mask", args.mask,
        "--output_dir", args.output_dir,
        "--line_width", str(args.line_width),
        "--fill_alpha", str(args.fill_alpha),
    ]
    return subprocess.run(cmd, cwd=str(REPO)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
