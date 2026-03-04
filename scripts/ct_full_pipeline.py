"""
新 CT（NIfTI）一键全流程：分割+勾画 → 病例报告 → 侵润倾向占位。

步骤：
  1. 调用 ct_to_detection_and_contour：nnU-Net 预测 mask → 轮廓叠加图 + 结节统计
  2. 调用 inference：VLM 生成病例报告（所见/结论/建议/病理倾向）
  3. 根据报告文本做侵润/风险倾向占位推断（关键词），并写入 invasiveness_placeholder.txt

前提：nnUNetv2 已安装且 Dataset503 2D 已训练；VLM 已训练（vision_bridge_vlm_final.pt 等）。

用法：
  python scripts/ct_full_pipeline.py --image "D:/path/to/new_ct.nii.gz"
  python scripts/ct_full_pipeline.py --image "/path/to/ct.nii.gz" --output_dir "/root/autodl-tmp/mamba-res/full_out" --skip_segment --mask "已有mask.nii.gz"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _abs_no_resolve(p: str | Path) -> Path:
    return Path(__import__("os").path.abspath(str(Path(p).expanduser())))


def main() -> int:
    ap = argparse.ArgumentParser(description="新 CT → 分割+勾画 → 报告 → 侵润倾向占位")
    ap.add_argument("--image", required=True, help="新 CT 的 NIfTI 路径")
    ap.add_argument("--output_dir", default="/root/autodl-tmp/mamba-res/full_pipeline", help="输出目录（所有结果落在此目录）")
    ap.add_argument("--skip_segment", action="store_true", help="跳过分割+勾画，仅做报告+侵润占位（需同时提供 --mask 或该目录下已有 predicted_mask.nii.gz）")
    ap.add_argument("--mask", default=None, help="已有 mask；与 --skip_segment 一起使用时跳过 nnU-Net 预测")
    ap.add_argument("--checkpoint", default=None, help="VLM checkpoint，默认 outputs/vision_bridge_vlm_final.pt")
    args = ap.parse_args()

    image_path = _abs_no_resolve(args.image)
    if not image_path.is_file():
        print("错误: 图像文件不存在:", image_path, file=sys.stderr)
        return 1

    out_dir = _abs_no_resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_path: Path | None = None

    # Step 1: 分割 + 勾画
    if not args.skip_segment:
        print("Step 1: 分割 + 勾画 ...")
        cmd_seg = [
            sys.executable,
            str(REPO / "scripts" / "ct_to_detection_and_contour.py"),
            "--image", str(image_path),
            "--output_dir", str(out_dir),
        ]
        if args.mask:
            cmd_seg += ["--skip_nnunet", "--mask", str(_abs_no_resolve(args.mask))]
        ret = subprocess.run(cmd_seg, cwd=str(REPO))
        if ret.returncode != 0:
            print("分割/勾画失败，终止流程。", file=sys.stderr)
            return ret.returncode
        mask_path = out_dir / "predicted_mask.nii.gz"
        if not mask_path.is_file():
            print("未找到 predicted_mask.nii.gz，请检查 Step 1 输出。", file=sys.stderr)
            return 1
    else:
        mask_path = _abs_no_resolve(args.mask) if args.mask else (out_dir / "predicted_mask.nii.gz")
        if not mask_path.is_file():
            print("--skip_segment 时需提供 --mask 或确保 output_dir 下已有 predicted_mask.nii.gz。", file=sys.stderr)
            return 1
        print("Step 1: 跳过分割，使用已有 mask:", mask_path)

    # Step 2: 病例报告
    print("Step 2: 生成病例报告 ...")
    ckpt = args.checkpoint
    if not ckpt:
        for name in ["vision_bridge_vlm_final.pt", "vision_bridge_best_val.pt", "vision_bridge_final.pt"]:
            p = REPO / "outputs" / name
            if p.exists():
                ckpt = str(p)
                break
    if not ckpt or not Path(ckpt).exists():
        print("未找到 VLM checkpoint，跳过报告生成；仅完成分割+勾画。", file=sys.stderr)
        report_text = ""
    else:
        cmd_inf = [
            sys.executable, str(REPO / "inference.py"),
            "--image", str(image_path),
            "--checkpoint", ckpt,
            "--out_dir", str(out_dir),
        ]
        ret = subprocess.run(cmd_inf, cwd=str(REPO), capture_output=True, text=True, timeout=300)
        # 报告会写在 out_dir/run_xxx/generated.txt，找最新 run 目录
        run_dirs = sorted(out_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        report_text = ""
        for run_d in run_dirs:
            gen_file = run_d / "generated.txt"
            if gen_file.is_file():
                report_text = gen_file.read_text(encoding="utf-8")
                break
        if not report_text and ret.stdout:
            report_text = ret.stdout.strip()[-5000:]  # fallback
        report_out = out_dir / "report.txt"
        report_out.write_text(report_text or "(未生成报告)", encoding="utf-8")
        print("报告已写入:", report_out)

    # Step 3: 侵润倾向占位（基于报告文本）
    print("Step 3: 侵润/风险倾向占位（基于报告关键词）...")
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    from infer_invasiveness_from_report import infer_invasiveness_from_report, format_invasiveness_output

    inv_result = infer_invasiveness_from_report(report_text)
    inv_path = out_dir / "invasiveness_placeholder.txt"
    inv_path.write_text(format_invasiveness_output(inv_result), encoding="utf-8")
    (out_dir / "invasiveness_placeholder.json").write_text(
        json.dumps(inv_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("侵润倾向占位:", inv_result.get("label", ""))
    print("已写入:", inv_path)

    print("\n全流程结束。输出目录:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
