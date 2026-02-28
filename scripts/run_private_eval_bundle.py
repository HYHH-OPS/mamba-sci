"""
Post-training private validation bundle:
1) Run private validation with report generation + nodule contour + grade prediction.
2) Analyze hallucination/format quality and classification consistency.
3) Generate plots and a markdown experiment brief for teacher reporting.

Example:
  python scripts/run_private_eval_bundle.py \
    --checkpoint /autodl-tmp/outputs/stage2_private_v5_gl/vision_bridge_vlm_final.pt \
    --mamba_model /autodl-tmp/models/mamba-2.8b-hf \
    --out_dir /autodl-tmp/mamba-res/val_private_bundle \
    --num_val 40 \
    --max_visual_tokens 164 \
    --constrained_decode \
    --train_log /autodl-tmp/outputs/stage2_private_v5_gl/stage2_train_log.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

GRADE_LABELS = ["AAH", "AIS", "MIA", "IAC"]


def _run(cmd: list[str], cwd: Path) -> None:
    print(">>", " ".join(cmd), flush=True)
    ret = subprocess.run(cmd, cwd=str(cwd))
    if ret.returncode != 0:
        raise RuntimeError(f"Command failed with code {ret.returncode}: {' '.join(cmd)}")


def _find_latest_run_dir(out_dir: Path) -> Path:
    runs = sorted([p for p in out_dir.glob("run_*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No run_* directory found under: {out_dir}")
    return runs[-1]


def _safe_read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _check_sections(text: str) -> dict[str, bool]:
    return {
        "has_所见": "所见" in text,
        "has_结论": "结论" in text,
        "has_建议": "建议" in text,
        "has_病理倾向": "病理倾向" in text,
    }


def _hallucination_flags(text: str) -> list[str]:
    flags: list[str] = []
    t = text or ""
    if len(t.strip()) < 120:
        flags.append("too_short")
    if re.search(r"<[^>\n]{1,120}>", t):
        flags.append("placeholder_text")
    if "begin_of_sentence" in t:
        flags.append("template_leak")
    if re.search(r"(.)\1{7,}", t):
        flags.append("char_repeat")
    if re.search(r"(.{2,8})\1{3,}", t):
        flags.append("ngram_repeat")
    if re.search(r"(colonoscopy|embolysation|thoracotomy)", t, flags=re.IGNORECASE):
        flags.append("off_topic_english")
    return flags


def _load_meta(run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {"samples": []}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _iter_sample_indices(run_dir: Path, meta: dict[str, Any]) -> list[int]:
    if isinstance(meta.get("samples"), list) and meta["samples"]:
        idxs = []
        for s in meta["samples"]:
            try:
                idxs.append(int(s.get("idx")))
            except Exception:
                pass
        if idxs:
            return sorted(set(idxs))
    idxs = []
    for p in run_dir.glob("sample_*_gen.txt"):
        m = re.match(r"sample_(\d+)_gen\.txt$", p.name)
        if m:
            idxs.append(int(m.group(1)))
    return sorted(set(idxs))


def _compute_confusion(rows: list[dict[str, Any]]) -> tuple[list[list[int]], float | None, int]:
    mat = [[0 for _ in range(4)] for _ in range(4)]
    valid = 0
    hit = 0
    for r in rows:
        gt = r.get("grade_gt")
        pred = r.get("grade_pred")
        if isinstance(gt, int) and isinstance(pred, int) and 0 <= gt < 4 and 0 <= pred < 4:
            mat[gt][pred] += 1
            valid += 1
            if gt == pred:
                hit += 1
    acc = (hit / valid) if valid > 0 else None
    return mat, acc, valid


def analyze_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta = _load_meta(run_dir)
    sample_map: dict[int, dict[str, Any]] = {}
    for s in meta.get("samples", []) if isinstance(meta.get("samples"), list) else []:
        try:
            sample_map[int(s.get("idx"))] = s
        except Exception:
            continue

    rows: list[dict[str, Any]] = []
    for idx in _iter_sample_indices(run_dir, meta):
        gen = _safe_read(run_dir / f"sample_{idx}_gen.txt")
        ref = _safe_read(run_dir / f"sample_{idx}_ref.txt")
        m = sample_map.get(idx, {})
        sec = _check_sections(gen)
        flags = _hallucination_flags(gen)

        grade_pred = None
        grade_label = None
        if isinstance(m.get("grade"), dict):
            try:
                grade_pred = int(m["grade"].get("index"))
                grade_label = str(m["grade"].get("label"))
            except Exception:
                grade_pred = None

        grade_gt = None
        try:
            grade_gt = int(m.get("grade_gt", -1))
            if grade_gt < 0:
                grade_gt = None
        except Exception:
            grade_gt = None

        contour_ok = int(isinstance(m.get("nodule_contour"), dict))
        nodule_count = None
        if contour_ok:
            try:
                nodule_count = int(m["nodule_contour"].get("nodule_count", 0))
            except Exception:
                nodule_count = None

        rows.append(
            {
                "idx": idx,
                "gen_len": len(gen.strip()),
                "ref_len": len(ref.strip()),
                "has_所见": int(sec["has_所见"]),
                "has_结论": int(sec["has_结论"]),
                "has_建议": int(sec["has_建议"]),
                "has_病理倾向": int(sec["has_病理倾向"]),
                "section_all_ok": int(all(sec.values())),
                "hallucination_flags": "|".join(flags),
                "hallucination_any": int(len(flags) > 0),
                "grade_gt": grade_gt,
                "grade_gt_label": GRADE_LABELS[grade_gt] if isinstance(grade_gt, int) and 0 <= grade_gt < 4 else "",
                "grade_pred": grade_pred,
                "grade_pred_label": grade_label or (GRADE_LABELS[grade_pred] if isinstance(grade_pred, int) and 0 <= grade_pred < 4 else ""),
                "contour_ok": contour_ok,
                "nodule_count": nodule_count if nodule_count is not None else "",
                "image_path": str(m.get("image_path", "")),
                "mask_path": str(m.get("mask_path", "")),
            }
        )

    if not rows:
        raise RuntimeError(f"No sample_*_gen.txt found in: {run_dir}")

    gen_lens = [r["gen_len"] for r in rows]
    ref_lens = [r["ref_len"] for r in rows if r["ref_len"] > 0]
    conf, acc, cls_n = _compute_confusion(rows)

    summary = {
        "run_dir": str(run_dir),
        "num_samples": len(rows),
        "gen_len_mean": statistics.mean(gen_lens),
        "gen_len_median": statistics.median(gen_lens),
        "ref_len_mean": statistics.mean(ref_lens) if ref_lens else None,
        "ref_len_median": statistics.median(ref_lens) if ref_lens else None,
        "section_all_ok_rate": sum(r["section_all_ok"] for r in rows) / len(rows),
        "hallucination_rate": sum(r["hallucination_any"] for r in rows) / len(rows),
        "contour_success_rate": sum(r["contour_ok"] for r in rows) / len(rows),
        "classification_valid_n": cls_n,
        "classification_accuracy": acc,
        "confusion_matrix_gt_pred": conf,
    }
    return summary, rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _plot_assets(summary: dict[str, Any], rows: list[dict[str, Any]], out_dir: Path, train_log: Path | None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable, skip plots: {e}", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) generated vs reference length
    idxs = [r["idx"] for r in rows]
    gen = [r["gen_len"] for r in rows]
    ref = [r["ref_len"] for r in rows]
    plt.figure(figsize=(9, 4))
    plt.plot(idxs, gen, label="generated_len")
    plt.plot(idxs, ref, label="reference_len", alpha=0.7)
    plt.xlabel("sample_idx")
    plt.ylabel("text length")
    plt.title("Generated vs Reference Length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "length_curve.png", dpi=180)
    plt.close()

    # 2) section coverage
    labels = ["Findings", "Conclusion", "Advice", "PathTrend", "All4"]
    rates = [
        sum(r["has_所见"] for r in rows) / len(rows),
        sum(r["has_结论"] for r in rows) / len(rows),
        sum(r["has_建议"] for r in rows) / len(rows),
        sum(r["has_病理倾向"] for r in rows) / len(rows),
        sum(r["section_all_ok"] for r in rows) / len(rows),
    ]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, rates)
    plt.ylim(0, 1.0)
    plt.title("Report Structure Coverage")
    plt.tight_layout()
    plt.savefig(out_dir / "structure_coverage.png", dpi=180)
    plt.close()

    # 3) classification confusion
    if summary.get("classification_valid_n", 0) > 0:
        conf = summary.get("confusion_matrix_gt_pred", [[0] * 4 for _ in range(4)])
        plt.figure(figsize=(5, 4))
        plt.imshow(conf, cmap="Blues")
        plt.xticks(range(4), GRADE_LABELS)
        plt.yticks(range(4), GRADE_LABELS)
        plt.xlabel("Pred")
        plt.ylabel("GT")
        plt.title("Grade Confusion Matrix")
        for i in range(4):
            for j in range(4):
                plt.text(j, i, str(conf[i][j]), ha="center", va="center", color="black")
        plt.tight_layout()
        plt.savefig(out_dir / "grade_confusion.png", dpi=180)
        plt.close()

    # 4) training loss curve
    if train_log and train_log.exists():
        steps: list[int] = []
        cap: list[float] = []
        cls: list[float] = []
        with train_log.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    steps.append(int(float(row.get("step", 0))))
                    cap.append(float(row.get("caption_loss", 0.0)))
                    cls.append(float(row.get("cls_loss", 0.0)))
                except Exception:
                    continue
        if steps:
            plt.figure(figsize=(9, 4))
            plt.plot(steps, cap, label="caption_loss")
            plt.plot(steps, cls, label="cls_loss")
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.title("Training Loss Curve")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "train_loss_curve.png", dpi=180)
            plt.close()


def _write_markdown(summary: dict[str, Any], out_md: Path) -> None:
    acc = summary.get("classification_accuracy")
    acc_s = f"{acc:.4f}" if isinstance(acc, float) else "N/A"
    txt = f"""# 私有数据验证实验汇报（自动生成）

## 1. 实验概况
- 样本数: {summary.get("num_samples")}
- 结果目录: `{summary.get("run_dir")}`

## 2. 三项功能验证
- 诊断文本生成（报告四段）通过率: `{summary.get("section_all_ok_rate", 0):.2%}`
- 结节勾画成功率: `{summary.get("contour_success_rate", 0):.2%}`
- 侵润等级分类可评估样本数: `{summary.get("classification_valid_n", 0)}`
- 侵润等级分类准确率: `{acc_s}`

## 3. 幻觉与文本质量
- 幻觉触发率（规则检测）: `{summary.get("hallucination_rate", 0):.2%}`
- 生成长度均值/中位数: `{summary.get("gen_len_mean", 0):.1f}` / `{summary.get("gen_len_median", 0)}`

## 4. 图表
- `length_curve.png`
- `structure_coverage.png`
- `grade_confusion.png`（若有真值可评估）
- `train_loss_curve.png`（若提供训练日志）

## 5. 下一步建议
1. 若幻觉率 > 20%，优先启用约束解码与更严格模板检查。
2. 若分类样本不平衡，补齐 AAH/AIS/MIA/IAC 真值并做分层验证。
3. 对失败样本做 case review（检查 mask 对齐、切片定位、文本标签一致性）。
"""
    out_md.write_text(txt, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run private validation bundle and build report assets.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to vision_bridge checkpoint")
    parser.add_argument("--mamba_model", type=str, default="state-spaces/mamba-2.8b-hf")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory passed to inference.py --out_dir")
    parser.add_argument("--num_val", type=int, default=30, help="Number of private validation samples")
    parser.add_argument("--max_visual_tokens", type=int, default=164)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--length_penalty", type=float, default=1.1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--suppress_eos_steps", type=int, default=128)
    parser.add_argument("--constrained_decode", action="store_true")
    parser.add_argument("--draw_nodule_contour", action="store_true")
    parser.add_argument("--skip_infer", action="store_true", help="Only analyze an existing run_dir")
    parser.add_argument("--run_dir", type=str, default=None, help="Existing run_xxx directory to analyze")
    parser.add_argument("--train_log", type=str, default=None, help="Optional stage2_train_log.csv for plotting")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir: Path
    if args.skip_infer:
        if not args.run_dir:
            raise ValueError("--skip_infer requires --run_dir")
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        cmd = [
            sys.executable,
            str(repo / "inference.py"),
            "--val_sample",
            "--num_val",
            str(args.num_val),
            "--out_dir",
            str(out_dir),
            "--mamba_model",
            args.mamba_model,
            "--max_visual_tokens",
            str(args.max_visual_tokens),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--num_beams",
            str(args.num_beams),
            "--length_penalty",
            str(args.length_penalty),
            "--no_repeat_ngram_size",
            str(args.no_repeat_ngram_size),
            "--suppress_eos_steps",
            str(args.suppress_eos_steps),
        ]
        if args.checkpoint:
            cmd += ["--checkpoint", args.checkpoint]
        if args.constrained_decode:
            cmd += ["--constrained_decode"]
        if args.draw_nodule_contour:
            cmd += ["--draw_nodule_contour"]
        _run(cmd, cwd=repo)
        run_dir = _find_latest_run_dir(out_dir)

    summary, rows = analyze_run(run_dir)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(rows, analysis_dir / "samples.csv")
    _plot_assets(summary, rows, analysis_dir, Path(args.train_log) if args.train_log else None)
    _write_markdown(summary, analysis_dir / "EXPERIMENT_REPORT.md")

    print("\n[done] private validation bundle complete", flush=True)
    print("run_dir:", run_dir, flush=True)
    print("analysis:", analysis_dir, flush=True)
    print("section_all_ok_rate:", f"{summary['section_all_ok_rate']:.2%}", flush=True)
    print("hallucination_rate:", f"{summary['hallucination_rate']:.2%}", flush=True)
    if summary.get("classification_accuracy") is not None:
        print("classification_accuracy:", f"{summary['classification_accuracy']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
