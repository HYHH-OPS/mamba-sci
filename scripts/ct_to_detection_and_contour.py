"""
新 CT → 自动检测（nnU-Net 分割）→ 自动勾画结节轮廓。

流程：传入新 CT → 调用 nnUNetv2 预测结节 mask → 用本仓库脚本生成轮廓叠加图与结节统计。

依赖：
  - 已安装 nnUNetv2（pip install nnunetv2），并已设置环境变量：
    NNUNET_RESULTS_FOLDER、nnUNet_preprocessed、nnUNet_raw（或通过 config/paths.yaml 推断）
  - Dataset503_TBLesion_327 的 2D 模型已训练好（nnunet_results/.../fold_0/）

用法：
  python scripts/ct_to_detection_and_contour.py --image "D:/path/to/new_ct.nii.gz"
  python scripts/ct_to_detection_and_contour.py --image "D:/path/to/ct.nii.gz" --output_dir "D:/mamba-res/nodules/my_case"
  python scripts/ct_to_detection_and_contour.py --image "..." --skip_nnunet --mask "已有mask.nii.gz"   # 仅勾画，不预测
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# 默认 task 与 config（与 paths.yaml 中 Dataset503_TBLesion_327、2d 一致）
DEFAULT_NNUNET_TASK_ID = 503
DEFAULT_NNUNET_CONFIG = "2d"
DEFAULT_FOLD = 0


def _load_config() -> dict:
    from data.medical_vlm_dataset import load_paths_config
    return load_paths_config(REPO / "config" / "paths.yaml")


def _run_nnunetv2_predict(
    input_dir: Path,
    output_dir: Path,
    task_id: int = DEFAULT_NNUNET_TASK_ID,
    config: str = DEFAULT_NNUNET_CONFIG,
    fold: int = DEFAULT_FOLD,
    nnunet_results: str | None = None,
    nnunet_preprocessed: str | None = None,
    nnunet_raw: str | None = None,
) -> Path | None:
    """
    调用 nnUNetv2 预测。返回预测结果中第一个 *_seg.nii.gz 的路径，失败返回 None。
    先尝试 CLI nnUNetv2_predict，再尝试 python -m nnUNetv2.predict。
    """
    env = os.environ.copy()
    if nnunet_results:
        env["nnUNet_results"] = str(nnunet_results)
        env["NNUNET_RESULTS_FOLDER"] = str(nnunet_results)
    if nnunet_preprocessed:
        env["nnUNet_preprocessed"] = str(nnunet_preprocessed)
        env["NNUNET_PREPROCESSED_FOLDER"] = str(nnunet_preprocessed)
    if nnunet_raw:
        env["nnUNet_raw"] = str(nnunet_raw)
        env["nnUNET_RAW_FOLDER"] = str(nnunet_raw)

    # nnUNetv2 常见调用方式（不同版本可能不同）
    attempts = [
        ["nnUNetv2_predict", "-i", str(input_dir), "-o", str(output_dir), "-d", str(task_id), "-c", config, "-f", str(fold)],
        [sys.executable, "-m", "nnUNetv2.predict", "-i", str(input_dir), "-o", str(output_dir), "-d", str(task_id), "-c", config, "-f", str(fold)],
    ]
    ran_ok = False
    for cmd in attempts:
        try:
            subprocess.run(cmd, env=env, check=True, cwd=str(REPO), timeout=600)
            ran_ok = True
            break
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError as e:
            print(f"nnUNetv2 预测失败 (exit {e.returncode}): {e}", file=sys.stderr)
            return None
        except subprocess.TimeoutExpired:
            print("nnUNetv2 预测超时（10 分钟）。", file=sys.stderr)
            return None

    if not ran_ok:
        print("未找到 nnUNetv2 命令行（请确认已安装 nnunetv2 并可用 nnUNetv2_predict 或 python -m nnUNetv2.predict）。", file=sys.stderr)
        return None

    # 查找输出：*_seg.nii.gz 或 *.nii.gz
    segs = list(output_dir.glob("*_seg.nii.gz")) or list(output_dir.glob("*.nii.gz"))
    if not segs:
        print("未在 nnUNetv2 输出目录找到预测 mask 文件。", file=sys.stderr)
        return None
    return segs[0]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="新 CT → nnU-Net 分割 → 结节轮廓勾画与统计（一键流程）"
    )
    ap.add_argument("--image", required=True, help="新 CT 的 NIfTI 路径（.nii 或 .nii.gz）")
    ap.add_argument("--output_dir", default="D:/mamba-res/nodules", help="输出目录（叠加图、结节统计、预测 mask 均在此）")
    ap.add_argument("--task_id", type=int, default=DEFAULT_NNUNET_TASK_ID, help=f"nnU-Net 任务 ID，默认 {DEFAULT_NNUNET_TASK_ID}（Dataset503）")
    ap.add_argument("--config", default=DEFAULT_NNUNET_CONFIG, help=f"nnU-Net 配置，默认 {DEFAULT_NNUNET_CONFIG}")
    ap.add_argument("--fold", type=int, default=DEFAULT_FOLD, help="使用的 fold，默认 0")
    ap.add_argument("--skip_nnunet", action="store_true", help="跳过 nnU-Net 预测，仅做勾画（需同时提供 --mask）")
    ap.add_argument("--mask", default=None, help="已有 mask 路径；与 --skip_nnunet 一起使用时仅执行勾画")
    ap.add_argument("--line_width", type=float, default=1.8)
    ap.add_argument("--fill_alpha", type=float, default=0.22)
    ap.add_argument("--keep_pred", action="store_true", help="保留 nnUNetv2 的临时预测目录（便于调试）")
    args = ap.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        print("错误: 图像文件不存在:", image_path, file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_path: Path | None = None

    if args.skip_nnunet:
        if not args.mask:
            print("错误: --skip_nnunet 时必须提供 --mask（已有 mask 路径）", file=sys.stderr)
            return 1
        mask_path = Path(args.mask).expanduser().resolve()
        if not mask_path.is_file():
            print("错误: Mask 文件不存在:", mask_path, file=sys.stderr)
            return 1
        print("跳过 nnU-Net 预测，使用已有 mask:", mask_path)
    else:
        # Step 1: nnUNetv2 预测
        try:
            import nnUNetv2  # noqa: F401
        except ImportError:
            print("未检测到 nnUNetv2。请安装: pip install nnunetv2", file=sys.stderr)
            print("安装后需设置环境变量 nnUNet_results / nnUNet_preprocessed / nnUNet_raw，或使用本脚本传入的 config。", file=sys.stderr)
            print("若您已有 mask，可运行: python scripts/ct_to_detection_and_contour.py --image ... --skip_nnunet --mask <mask路径>", file=sys.stderr)
            return 1

        config = _load_config()
        nnunet_results = config.get("nnunet_results")
        nnunet_preprocessed = config.get("nnunet_preprocessed")
        nnunet_raw = config.get("nnunet_raw")
        if not nnunet_results or not Path(nnunet_results).is_dir():
            print("警告: nnunet_results 未配置或目录不存在，nnUNetv2 可能找不到模型。", file=sys.stderr)
            print("  config/paths.yaml 中 nnunet_results 应指向含 Dataset503_... 的目录。", file=sys.stderr)

        with tempfile.TemporaryDirectory(prefix="nnunet_in_") as tmp_in:
            with tempfile.TemporaryDirectory(prefix="nnunet_out_") as tmp_out:
                # nnUNetv2 要求输入名为 *_0000.nii.gz（单模态）
                case_name = "case"
                input_nifti = Path(tmp_in) / f"{case_name}_0000.nii.gz"
                shutil.copy2(image_path, input_nifti)
                print("Step 1: 运行 nnU-Net 2D 分割预测 ...")
                mask_path = _run_nnunetv2_predict(
                    Path(tmp_in),
                    Path(tmp_out),
                    task_id=args.task_id,
                    config=args.config,
                    fold=args.fold,
                    nnunet_results=nnunet_results,
                    nnunet_preprocessed=nnunet_preprocessed,
                    nnunet_raw=nnunet_raw,
                )
                if mask_path is None:
                    return 1
                # 将预测 mask 复制到用户输出目录，便于后续使用
                pred_save = out_dir / "predicted_mask.nii.gz"
                shutil.copy2(mask_path, pred_save)
                print("预测 mask 已保存:", pred_save)
                mask_path = pred_save
                if args.keep_pred:
                    keep_dir = out_dir / "nnunet_pred"
                    keep_dir.mkdir(parents=True, exist_ok=True)
                    for f in Path(tmp_out).iterdir():
                        shutil.copy2(f, keep_dir / f.name)
                    print("已保留 nnUNetv2 原始输出至:", keep_dir)

    if mask_path is None:
        return 1

    # Step 2: 结节轮廓勾画与统计
    print("Step 2: 生成结节轮廓叠加图与统计 ...")
    from vision.nodule_contour import generate_nodule_contour_outputs

    result = generate_nodule_contour_outputs(
        image_path=str(image_path),
        mask_path=str(mask_path),
        output_dir=out_dir,
        line_width=args.line_width,
        fill_alpha=args.fill_alpha,
    )
    print("结节数量:", result["nodule_count"])
    print("轮廓图:", result["overlay_png"])
    print("统计 CSV:", result["stats_csv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
