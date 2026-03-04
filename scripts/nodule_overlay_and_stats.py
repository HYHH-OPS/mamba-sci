"""
Generate nodule contour overlay and component stats from image + mask.

Usage (必须使用真实存在的文件路径，不能使用占位符 xxx):
  python scripts/nodule_overlay_and_stats.py ^
    --image "D:/nnunet_raw/Dataset503_TBLesion_327/imagesTr/0000719802_20260124_0000.nii.gz" ^
    --mask "D:/nnunet_raw/Dataset503_TBLesion_327/labelsTr/0000719802_20260124.nii.gz" ^
    --output_dir "/root/autodl-tmp/mamba-res/nodules"

路径可从 caption CSV 的 image_path / mask_path 列获取，或从 nnunet_raw/.../imagesTr 与 labelsTr 中选取。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from vision.nodule_contour import generate_nodule_contour_outputs


def _resolve_path(p: str) -> Path:
    # Keep symlink/junction path text instead of dereferencing to real path.
    # This avoids issues in some Windows setups where SimpleITK fails on
    # resolved non-ASCII paths.
    return Path(os.path.abspath(str(Path(p).expanduser())))


def main() -> int:
    ap = argparse.ArgumentParser(description="Nodule contour overlay + nodule stats")
    ap.add_argument("--image", required=True, help="CT NIfTI 路径（需真实存在，勿用占位符 xxx）")
    ap.add_argument("--mask", required=True, help="分割 mask NIfTI 路径（需真实存在）")
    ap.add_argument("--output_dir", default="/root/autodl-tmp/mamba-res/nodules", help="Output directory")
    ap.add_argument("--line_width", type=float, default=1.8, help="Contour line width")
    ap.add_argument("--fill_alpha", type=float, default=0.22, help="Overlay fill alpha")
    args = ap.parse_args()

    image_path = _resolve_path(args.image)
    mask_path = _resolve_path(args.mask)

    if not image_path.is_file():
        print("错误: 图像文件不存在:", image_path, file=sys.stderr)
        if "xxx" in str(image_path).lower():
            print("  您可能使用了文档中的占位符 xxx，请替换为实际病例 ID。", file=sys.stderr)
            print("  示例: --image \"D:/nnunet_raw/Dataset503_TBLesion_327/imagesTr/0000719802_20260124_0000.nii.gz\"", file=sys.stderr)
        print("  可从 caption CSV 的 image_path 列复制路径，或从 imagesTr 目录中选取存在的 .nii.gz 文件。", file=sys.stderr)
        return 1
    if not mask_path.is_file():
        print("错误: Mask 文件不存在:", mask_path, file=sys.stderr)
        if "xxx" in str(mask_path).lower():
            print("  请使用与 image 对应的 labelsTr 中的 .nii.gz 路径。", file=sys.stderr)
            print("  示例: --mask \"D:/nnunet_raw/Dataset503_TBLesion_327/labelsTr/0000719802_20260124.nii.gz\"", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading image and mask ...")
    result = generate_nodule_contour_outputs(
        image_path=str(image_path),
        mask_path=str(mask_path),
        output_dir=out_dir,
        line_width=args.line_width,
        fill_alpha=args.fill_alpha,
    )
    print(f"Saved: {result['overlay_png']}")
    print(f"Saved: {result['stats_csv']}")
    print(f"Nodule count: {result['nodule_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
