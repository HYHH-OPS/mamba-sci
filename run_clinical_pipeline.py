from __future__ import annotations

"""
End-to-end clinical pipeline:

CT (.nii / .nii.gz) → nnU-Net 分割 → VLM 报告 + 分级 + 结节勾画与体积统计。

使用方式（示例）：

  python run_clinical_pipeline.py --image D:/nnunet_raw/DatasetXXX/imagesTr/xxx_0000.nii.gz ^
      --nnunet_cmd "nnUNetv2_predict -i {in_dir} -o {out_dir} -d 503 -c 2d -f 0"

注意：
- 本脚本不会训练任何模型，只是串联「已有 nnU-Net 分割」和「已有 VLM 推理」两步。
- 你需要事先准备好 nnU-Net 的权重，并能在命令行中用 nnUNetv2_predict 正常跑出 mask。
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

# 保证仓库根目录在 sys.path 中
REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.medical_vlm_dataset import load_paths_config
from inference import (  # type: ignore
    _build_template_force_words_ids,
    generate_from_image,
    infer_grade_from_queries,
    load_image_tensor,
    load_vision_bridge,
)
from llm.mamba_loader import load_mamba_lm  # type: ignore
from vision.nodule_contour import generate_nodule_contour_outputs  # type: ignore


def run_nnunet_segmentation(
    image_path: Path,
    temp_root: Path,
    nnunet_cmd: Optional[str],
) -> Path:
    """
    调用外部 nnUNet 进行分割。

    参数
    ----
    image_path: 原始 CT NIfTI 路径 (.nii 或 .nii.gz)
    temp_root: 临时工作目录；本函数会在其中创建 input/ 与 output/ 子目录
    nnunet_cmd: 完整的 nnUNetv2_predict 命令模板；支持占位符：
        {in_dir}: 输入目录（包含待预测的 .nii.gz）
        {out_dir}: 输出目录（nnUNet 写入预测 mask 的目录）

    返回
    ----
    mask_path: 预测得到的 mask NIfTI 路径
    """
    temp_in = temp_root / "input"
    temp_out = temp_root / "output"
    temp_in.mkdir(parents=True, exist_ok=True)
    temp_out.mkdir(parents=True, exist_ok=True)

    # 将待预测的 CT 拷贝到临时输入目录（避免改动原始数据结构）
    ct_copy = temp_in / image_path.name
    if not ct_copy.exists():
        shutil.copy2(image_path, ct_copy)

    if nnunet_cmd is None or not nnunet_cmd.strip():
        # 预留命令模板位置：请根据你自己的 nnU-Net 模型修改 DATASET_ID / 配置等参数
        nnunet_cmd = (
            "nnUNetv2_predict "
            "-i {in_dir} "
            "-o {out_dir} "
            "-d DATASET_ID "
            "-c 2d "
            "-f 0"
        )

    cmd_str = nnunet_cmd.format(in_dir=str(temp_in), out_dir=str(temp_out))
    print("\n========== Stage 1: nnU-Net 分割 ==========")
    print("即将执行的 nnU-Net 推理命令（请根据你的环境修改 --nnunet_cmd 或脚本内模板）：")
    print(f"  {cmd_str}\n")

    result = subprocess.run(cmd_str, shell=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"nnUNet 推理命令执行失败（returncode={result.returncode}）。\n"
            "请检查 nnUNet 安装、权重路径与命令参数是否正确。"
        )

    # 在输出目录中寻找第一个 .nii / .nii.gz 作为 mask
    candidates = sorted(
        list(temp_out.glob("*.nii.gz")) + list(temp_out.glob("*.nii"))
    )
    if not candidates:
        raise FileNotFoundError(
            f"在 nnUNet 输出目录中未找到任何 .nii / .nii.gz 文件: {temp_out}"
        )

    mask_path = candidates[0]
    print(f"nnUNet 分割完成，使用的 mask: {mask_path}\n")
    return mask_path


def pretty_print_nodule_stats(stats_csv_path: Path, max_rows: int = 5) -> None:
    """从 nodules.csv 中读取前几条结节统计并打印。"""
    if not stats_csv_path.exists():
        print("未找到结节统计 CSV:", stats_csv_path)
        return

    with open(stats_csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("未检测到阳性结节（mask 中没有连通域 > 0）。")
        return

    print(f"共检测到 {len(rows)} 个结节，前 {min(max_rows, len(rows))} 个如下（按体积排序）：")
    header = f"{'ID':>4} {'Vol(mm^3)':>12} {'Eq.D(mm)':>10} {'Center(z,y,x)mm':>30}"
    print(header)
    print("-" * len(header))
    for row in rows[:max_rows]:
        nid = int(row.get("nodule_id", 0))
        vol = float(row.get("volume_mm3", 0.0))
        d_eq = float(row.get("equivalent_diameter_mm", 0.0))
        cz = row.get("center_z_mm", "?")
        cy = row.get("center_y_mm", "?")
        cx = row.get("center_x_mm", "?")
        print(
            f"{nid:>4} {vol:>12.2f} {d_eq:>10.2f} "
            f"{f'({cz}, {cy}, {cx})':>30}"
        )


def run_vlm_inference_with_mask(
    image_path: Path,
    mask_path: Path,
    out_root: Path,
    checkpoint: Optional[str],
    mamba_model: Optional[str],
    llm_device: str = "auto",
) -> None:
    """
    第二阶段：调用 VLM 完成「报告生成 + 分级 + 结节勾画」。
    """
    print("========== Stage 2: VLM 推理 ==========\n")

    config = load_paths_config(REPO / "config" / "paths.yaml")
    config.setdefault("encoder_output_stage", 4)
    config.setdefault("encoder_target_spatial", 28)
    config["bridge_d_model"] = 2560
    config.setdefault("nnunet_encoder_checkpoint", None)
    config.setdefault("use_cmi", False)
    config.setdefault("roi_side", None)
    config.setdefault("cmi_compress_to", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / ("clinical_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    # 选择 Vision+Bridge checkpoint
    ckpt = checkpoint
    if ckpt is None:
        default_out = REPO / "outputs"
        for name in [
            "vision_bridge_vlm_final.pt",
            "vision_bridge_best_val.pt",
            "vision_bridge_final.pt",
        ]:
            p = default_out / name
            if p.exists():
                ckpt = str(p)
                break
    if ckpt is None or not Path(ckpt).exists():
        raise FileNotFoundError(
            "未找到 Vision+Bridge checkpoint。\n"
            "请先完成 train_vlm.py 的 Stage 2 训练，或通过 --checkpoint 显式指定路径。"
        )

    print(f"加载 Vision+Bridge checkpoint: {ckpt}")
    vision_bridge = load_vision_bridge(ckpt, config, device)

    # 加载 Mamba LLM
    mamba_name = (
        mamba_model
        or config.get("mamba_hf_model")
        or "state-spaces/mamba-2.8b-hf"
    )
    print(f"加载 Mamba LLM: {mamba_name} (device={llm_device})")
    llm_device_map = "cpu" if llm_device == "cpu" else llm_device  # "auto"/"cuda"
    llm_model, tokenizer = load_mamba_lm(
        mamba_name,
        device_map=llm_device_map,
    )
    llm_model.eval()

    # 再次确保 lm_head 权重绑定（与 inference.py 一致）
    if (
        hasattr(llm_model, "backbone")
        and hasattr(llm_model.backbone, "embeddings")
        and hasattr(llm_model, "lm_head")
    ):
        llm_model.lm_head.weight = llm_model.backbone.embeddings.weight

    # 图像加载与预处理（与 inference.py 保持一致）
    image_tensor = load_image_tensor(str(image_path), mask_path=str(mask_path)).to(
        device
    )

    # 生成报告文本
    print("开始生成报告文本（可能需要数十秒）...\n")
    force_words_ids = _build_template_force_words_ids(tokenizer)
    text = generate_from_image(
        image_tensor,
        vision_bridge,
        llm_model,
        tokenizer,
        prompt=None,
        max_new_tokens=512,
        device=device,
        max_visual_tokens=196,
        do_sample=False,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.2,
        length_penalty=1.1,
        no_repeat_ngram_size=0,
        suppress_eos_steps=128,
        num_beams=4,
        force_words_ids=force_words_ids,
        min_chars=180,
        max_retries=2,
        force_template=True,
        raw_out=None,
        debug_vision=False,
    )

    # 分级信息（如果训练/权重中接入了 grade_head）
    grade_info = infer_grade_from_queries(vision_bridge)

    # 结节勾画与体积统计
    contour_dir = run_dir / "nodule_contour"
    contour_info = generate_nodule_contour_outputs(
        str(image_path),
        str(mask_path),
        contour_dir,
        line_width=1.8,
        fill_alpha=0.22,
    )

    # 落盘结果
    (run_dir / "report.txt").write_text(text, encoding="utf-8")
    meta = {
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "checkpoint": ckpt,
        "mamba_model": mamba_name,
        "grade": grade_info,
        "contour": contour_info,
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 终端打印汇总信息
    print("\n========== 结节体积与位置统计 ==========")
    stats_csv = Path(contour_info.get("stats_csv", ""))
    pretty_print_nodule_stats(stats_csv)

    print("\n========== 四级分级结果（AAH / AIS / MIA / IAC） ==========")
    if grade_info is None:
        print(
            "未检测到分级 head 或 VimBridge 未缓存 latest_queries_out。\n"
            "请确保在训练时启用了分级损失，并在保存 checkpoint 时包含 grade_head 权重。"
        )
    else:
        label = grade_info.get("label", "?")
        probs = grade_info.get("probs", [])
        print(f"预测分级: {label}")
        if probs:
            print(
                "概率向量 [AAH, AIS, MIA, IAC]: "
                + ", ".join(f"{p:.3f}" for p in probs)
            )

    print("\n========== 生成报告全文 ==========\n")
    print(text.strip() or "[报告为空]")

    print("\n结果文件已保存到：", run_dir)
    print("  - 报告文本: report.txt")
    print("  - 元信息:   meta.json")
    print("  - 结节统计: nodule_contour/nodules.csv")
    print("  - 勾画 PNG: nodule_contour/overlay_contour.png\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CT -> nnUNet 分割 + VLM 报告/分级/结节勾画 的串联临床流水线"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="原始 CT NIfTI 路径 (.nii 或 .nii.gz)",
    )
    parser.add_argument(
        "--nnunet_cmd",
        type=str,
        default=None,
        help=(
            "完整的 nnUNetv2_predict 命令模板，支持占位符 {in_dir} 与 {out_dir}。\n"
            "例如：\"nnUNetv2_predict -i {in_dir} -o {out_dir} -d 503 -c 2d -f 0\""
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Vision+Bridge VLM checkpoint 路径；若不提供则在 outputs/ 中自动搜索。",
    )
    parser.add_argument(
        "--mamba_model",
        type=str,
        default=None,
        help=(
            "Mamba 模型 id 或本地路径（如 state-spaces/mamba-2.8b-hf 或 /data/models/mamba-2.8b-hf）。"
        ),
    )
    parser.add_argument(
        "--llm_device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Mamba 运行设备：auto 使用 GPU（推荐），cpu 强制 CPU，cuda 强制 GPU。",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="流水线结果输出根目录，默认为仓库下的 outputs_clinical/。",
    )
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="是否保留 nnUNet 临时输入/输出目录（默认执行结束后自动删除）。",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"图像文件不存在: {image_path}")
        return 1

    out_root = Path(args.out_dir) if args.out_dir else REPO / "outputs_clinical"

    # 临时目录：用于 nnUNet 输入/输出
    temp_root = out_root / "temp_seg"
    temp_root.mkdir(parents=True, exist_ok=True)

    try:
        # Stage 1: nnUNet 分割
        mask_path = run_nnunet_segmentation(image_path, temp_root, args.nnunet_cmd)

        # Stage 2: VLM 推理 + 勾画/分级
        run_vlm_inference_with_mask(
            image_path=image_path,
            mask_path=mask_path,
            out_root=out_root,
            checkpoint=args.checkpoint,
            mamba_model=args.mamba_model,
            llm_device=args.llm_device,
        )
    finally:
        if not args.keep_temp and temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

