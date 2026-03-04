"""
VLM 鍥惧儚鈫掓姤鍛婅缁冿細浠呰缁?Vision+Bridge锛孧amba 鍐荤粨銆?
杈撳叆锛氶棶棰?鎶ュ憡锛汱oss 鍙銆屾姤鍛娿€嶉儴鍒嗚绠楋紝涓庢帹鐞嗘椂銆岄棶棰?鎹㈣銆嶅悗鐢熸垚鎶ュ憡涓€鑷淬€?

鐢ㄦ硶:
  python train_vlm.py --epochs 30 --batch_size 8 --lr 1e-5 --max_visual_tokens 164
  python train_vlm.py --epochs 30 --batch_size 4 --lr 1e-5 --max_visual_tokens 164 --gradient_accumulation_steps 1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import os
import torch
# A100(sm_80) will use native CUDA fast path automatically.
# Keep this compatibility fallback only for SM120 cards (e.g., RTX 5090).
_force_cuda = os.environ.get("MAMBA_FORCE_CUDA", "0") == "1"
if not _force_cuda:
    try:
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap >= (12, 0):
                import mamba_ssm.ops.selective_scan_interface as _ssi
                _ref_fn = _ssi.selective_scan_ref
                _ssi.selective_scan_fn = _ref_fn
                import mamba_ssm.modules.mamba_simple as _mamba_simple
                _mamba_simple.selective_scan_fn = _ref_fn
                _mamba_simple.causal_conv1d_fn = None
                _mamba_simple.causal_conv1d_update = None
    except Exception:
        pass

import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    # torch>=2.0 preferred API
    from torch.amp import autocast as _autocast
    from torch.amp import GradScaler as _GradScaler
    _AMP_MODE = "torch.amp"
except Exception:
    # torch<2.0 fallback
    from torch.cuda.amp import autocast as _autocast  # type: ignore
    from torch.cuda.amp import GradScaler as _GradScaler  # type: ignore
    _AMP_MODE = "torch.cuda.amp"

from data.medical_vlm_dataset import MedicalVLMDataset, load_paths_config
from model.forward_medical_vlm import build_medical_vlm_from_config
from inference import _pool_visual_tokens

# 榛樿鏂囨湰闀垮害锛涘尰瀛︽姤鍛婅緝闀匡紝杩囧皬浼氭埅鏂€?
DEFAULT_MAX_TEXT_LEN = 512


def _autocast_ctx(device: torch.device, amp_dtype: torch.dtype = torch.float16):
    if device.type != "cuda":
        return _autocast(enabled=False)
    if _AMP_MODE == "torch.amp":
        return _autocast("cuda", dtype=amp_dtype)
    # torch.cuda.amp.autocast has no device_type arg in old torch versions.
    return _autocast(dtype=amp_dtype)


def _build_grad_scaler(device: torch.device, amp_dtype: torch.dtype = torch.float16):
    if device.type != "cuda":
        return None
    # BF16 typically does not need gradient scaling.
    if amp_dtype == torch.bfloat16:
        return None
    if _AMP_MODE == "torch.amp":
        try:
            return _GradScaler("cuda")
        except TypeError:
            return _GradScaler()
    return _GradScaler()


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {v}")


def _parse_int_tuple(v: str, n: int = 3) -> tuple[int, ...]:
    parts = [p.strip() for p in str(v).split(",") if p.strip()]
    if len(parts) != n:
        raise argparse.ArgumentTypeError(f"expected {n} comma-separated ints, got: {v}")
    try:
        vals = tuple(int(p) for p in parts)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"invalid int tuple: {v}") from e
    if any(x <= 0 for x in vals):
        raise argparse.ArgumentTypeError(f"all tuple values must be >0, got: {vals}")
    return vals


def _summarize_text_token_lengths(dataset, tokenizer, max_text_len: int, max_samples: int = 0):
    """
    Print token length stats for answer/full text and estimate truncation rate under max_text_len.
    """
    n = len(dataset)
    if n == 0:
        return
    if max_samples and max_samples > 0:
        n = min(n, max_samples)
    ans_lens = []
    full_lens = []
    trunc = 0
    for i in range(n):
        q = str(dataset.questions[i]) if i < len(dataset.questions) else ""
        a = str(dataset.answers[i]) if i < len(dataset.answers) else ""
        ans_ids = tokenizer(a, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(f"{q}\n{a}", add_special_tokens=False)["input_ids"]
        al = len(ans_ids)
        fl = len(full_ids)
        ans_lens.append(al)
        full_lens.append(fl)
        if fl > max_text_len:
            trunc += 1

    t_ans = torch.tensor(ans_lens, dtype=torch.float32)
    t_full = torch.tensor(full_lens, dtype=torch.float32)
    trunc_rate = trunc / n
    print(
        f"[length stats] samples={n}, answer p50/p90/p95/max="
        f"{int(torch.quantile(t_ans, 0.5))}/{int(torch.quantile(t_ans, 0.9))}/"
        f"{int(torch.quantile(t_ans, 0.95))}/{int(t_ans.max().item())}",
        flush=True,
    )
    print(
        f"[length stats] full   p50/p90/p95/max="
        f"{int(torch.quantile(t_full, 0.5))}/{int(torch.quantile(t_full, 0.9))}/"
        f"{int(torch.quantile(t_full, 0.95))}/{int(t_full.max().item())}",
        flush=True,
    )
    print(
        f"[length stats] max_text_len={max_text_len}, estimated truncation_rate={trunc_rate:.2%}",
        flush=True,
    )
    if trunc_rate > 0.20:
        print(
            "[warning] truncation_rate > 20%, reports may become too short. "
            "Increase --max_text_len (e.g. 640/768) or shorten question prompt.",
            flush=True,
        )


def _enable_encoder_gradient_checkpointing(encoder: torch.nn.Module) -> bool:
    """
    Best-effort enable encoder gradient checkpointing, depending on module implementation.
    """
    if encoder is None:
        return False
    enabled = False
    for fn_name in ("gradient_checkpointing_enable", "enable_gradient_checkpointing", "set_gradient_checkpointing"):
        fn = getattr(encoder, fn_name, None)
        if callable(fn):
            try:
                if fn_name == "set_gradient_checkpointing":
                    fn(True)
                else:
                    fn()
                enabled = True
            except Exception:
                pass
    for attr_name in ("gradient_checkpointing", "use_gradient_checkpointing", "use_checkpoint"):
        if hasattr(encoder, attr_name):
            try:
                setattr(encoder, attr_name, True)
                enabled = True
            except Exception:
                pass
    return enabled


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        v = float(value)
    except Exception:
        return default
    if not math.isfinite(v):
        return default
    return v


def _ema(values: list[float], alpha: float = 0.12) -> list[float]:
    if not values:
        return []
    out: list[float] = []
    prev = values[0]
    for v in values:
        prev = alpha * v + (1.0 - alpha) * prev
        out.append(prev)
    return out


def _write_training_paper_assets(log_csv: Path, epoch_csv: Path, out_dir: Path) -> None:
    """
    Build paper-friendly charts/tables from training logs.
    Safe to call repeatedly during training (e.g., every epoch).
    """
    if not log_csv.exists():
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    steps: list[int] = []
    total_loss: list[float] = []
    caption_loss: list[float] = []
    cls_loss: list[float] = []
    cls_acc: list[float] = []
    cls_mae: list[float] = []
    sps: list[float] = []
    step_time: list[float] = []

    with log_csv.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = _safe_float(row.get("step"), None)
            t = _safe_float(row.get("total_loss"), None)
            c = _safe_float(row.get("caption_loss"), None)
            cl = _safe_float(row.get("cls_loss"), None)
            if s is None:
                continue
            if t is None and c is not None and cl is not None:
                t = c + cl
            if t is None or c is None or cl is None:
                continue
            steps.append(int(s))
            total_loss.append(float(t))
            caption_loss.append(float(c))
            cls_loss.append(float(cl))
            acc = _safe_float(row.get("cls_acc"), None)
            mae = _safe_float(row.get("cls_mae"), None)
            if acc is not None:
                cls_acc.append(acc)
            if mae is not None:
                cls_mae.append(mae)
            v_sps = _safe_float(row.get("samples_per_sec"), None)
            if v_sps is not None:
                sps.append(v_sps)
            v_st = _safe_float(row.get("step_time_sec"), None)
            if v_st is not None:
                step_time.append(v_st)

    if not steps:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Main training losses with EMA for cleaner paper figures.
    plt.figure(figsize=(10, 4.8))
    plt.plot(steps, total_loss, color="#5B8FF9", alpha=0.28, linewidth=1.0, label="total_loss(raw)")
    plt.plot(steps, caption_loss, color="#61DDAA", alpha=0.22, linewidth=1.0, label="caption_loss(raw)")
    plt.plot(steps, cls_loss, color="#F6BD16", alpha=0.22, linewidth=1.0, label="cls_loss(raw)")
    plt.plot(steps, _ema(total_loss), color="#1E40AF", linewidth=2.0, label="total_loss(EMA)")
    plt.plot(steps, _ema(caption_loss), color="#047857", linewidth=1.8, label="caption_loss(EMA)")
    plt.plot(steps, _ema(cls_loss), color="#B45309", linewidth=1.8, label="cls_loss(EMA)")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training Loss Curves (Raw + EMA)")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "paper_train_loss_curves.png", dpi=220)
    plt.close()

    # 2) Classification head quality curve.
    if cls_acc and cls_mae:
        x_cls = list(range(1, min(len(cls_acc), len(cls_mae)) + 1))
        y_acc = cls_acc[: len(x_cls)]
        y_mae = cls_mae[: len(x_cls)]
        plt.figure(figsize=(9, 4.2))
        plt.plot(x_cls, _ema(y_acc), color="#2563EB", linewidth=2.0, label="cls_acc(EMA)")
        plt.plot(x_cls, _ema(y_mae), color="#DC2626", linewidth=2.0, label="cls_mae(EMA)")
        plt.ylim(0.0, 1.05)
        plt.xlabel("logged step index")
        plt.ylabel("metric")
        plt.title("Classification Branch Quality")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "paper_cls_quality.png", dpi=220)
        plt.close()

    # 3) Throughput / step time.
    if sps or step_time:
        plt.figure(figsize=(9, 4.2))
        if sps:
            plt.plot(range(1, len(sps) + 1), _ema(sps), color="#059669", linewidth=2.0, label="samples_per_sec(EMA)")
        if step_time:
            plt.plot(range(1, len(step_time) + 1), _ema(step_time), color="#7C3AED", linewidth=2.0, label="step_time_sec(EMA)")
        plt.xlabel("logged step index")
        plt.ylabel("efficiency")
        plt.title("Training Efficiency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "paper_efficiency.png", dpi=220)
        plt.close()

    # 4) Epoch-level curve if available.
    if epoch_csv.exists():
        ep: list[int] = []
        ep_total: list[float] = []
        ep_cap: list[float] = []
        ep_cls: list[float] = []
        with epoch_csv.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                e = _safe_float(row.get("epoch"), None)
                t = _safe_float(row.get("avg_total_loss"), None)
                c = _safe_float(row.get("avg_caption_loss"), None)
                cl = _safe_float(row.get("avg_cls_loss"), None)
                if e is None or t is None or c is None or cl is None:
                    continue
                ep.append(int(e))
                ep_total.append(t)
                ep_cap.append(c)
                ep_cls.append(cl)
        if ep:
            plt.figure(figsize=(8.2, 4.2))
            plt.plot(ep, ep_total, marker="o", linewidth=2.0, label="avg_total_loss")
            plt.plot(ep, ep_cap, marker="o", linewidth=1.8, label="avg_caption_loss")
            plt.plot(ep, ep_cls, marker="o", linewidth=1.8, label="avg_cls_loss")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title("Epoch-Level Training Metrics")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "paper_epoch_metrics.png", dpi=220)
            plt.close()

    summary = {
        "final_step": int(steps[-1]),
        "best_total_loss": float(min(total_loss)),
        "final_total_loss": float(total_loss[-1]),
        "best_caption_loss": float(min(caption_loss)),
        "final_caption_loss": float(caption_loss[-1]),
        "best_cls_loss": float(min(cls_loss)),
        "final_cls_loss": float(cls_loss[-1]),
        "best_cls_acc": float(max(cls_acc)) if cls_acc else None,
        "best_cls_mae": float(min(cls_mae)) if cls_mae else None,
        "median_samples_per_sec": float(sorted(sps)[len(sps) // 2]) if sps else None,
    }
    (out_dir / "paper_train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md = (
        "# Training Summary (Auto)\n\n"
        f"- final_step: `{summary['final_step']}`\n"
        f"- best_total_loss / final_total_loss: `{summary['best_total_loss']:.4f}` / `{summary['final_total_loss']:.4f}`\n"
        f"- best_caption_loss / final_caption_loss: `{summary['best_caption_loss']:.4f}` / `{summary['final_caption_loss']:.4f}`\n"
        f"- best_cls_loss / final_cls_loss: `{summary['best_cls_loss']:.4f}` / `{summary['final_cls_loss']:.4f}`\n"
        f"- best_cls_acc: `{summary['best_cls_acc'] if summary['best_cls_acc'] is not None else 'N/A'}`\n"
        f"- best_cls_mae: `{summary['best_cls_mae'] if summary['best_cls_mae'] is not None else 'N/A'}`\n"
        f"- median_samples_per_sec: `{summary['median_samples_per_sec'] if summary['median_samples_per_sec'] is not None else 'N/A'}`\n\n"
        "## Figures\n"
        "- `paper_train_loss_curves.png`\n"
        "- `paper_cls_quality.png` (if cls labels available)\n"
        "- `paper_efficiency.png`\n"
        "- `paper_epoch_metrics.png`\n"
    )
    (out_dir / "paper_train_summary.md").write_text(md, encoding="utf-8")


def _guess_lora_target_modules(model: torch.nn.Module, hints: list[str]) -> list[str]:
    """
    Auto-detect LoRA target Linear module suffix names for Mamba-like backbones.
    """
    matched: set[str] = set()
    model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    blocked_for_mamba = {"out_proj", "conv1d"} if model_type == "mamba" else set()
    hint_lower = [h.strip().lower() for h in hints if h.strip()]
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        leaf = name.split(".")[-1]
        leaf_l = leaf.lower()
        if leaf in blocked_for_mamba:
            continue
        if any(h in leaf_l for h in hint_lower):
            matched.add(leaf)
    # Reasonable fallback for common Mamba / Transformer naming.
    if not matched:
        fallback = {"in_proj", "out_proj", "x_proj", "dt_proj", "gate_proj", "up_proj", "down_proj"}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                leaf = name.split(".")[-1]
                if leaf in fallback and leaf not in blocked_for_mamba:
                    matched.add(leaf)
    return sorted(matched)


def compute_batch_loss(
    batch: dict,
    vision_bridge: torch.nn.Module,
    llm_model: torch.nn.Module,
    embed: torch.nn.Module,
    tokenizer,
    device: torch.device,
    llm_device: torch.device,
    max_visual_tokens: int,
    max_text_len: int,
    d_model: int,
    use_gradient_checkpointing: bool = False,
    grade_head: torch.nn.Module | None = None,
    class_weights: torch.Tensor | None = None,
    lambda_cls: float = 1.0,
    cls_focal_gamma: float = 0.0,
    cls_focal_alpha: float = -1.0,
) -> tuple[torch.Tensor, float, float, float, float, int]:
    """
    鍗?batch 鍓嶅悜 + caption loss銆?
    Loss 浠呭銆屽洖绛斻€嶉儴鍒嗚绠楋紙瑙嗚+闂+鎹㈣ 鍧囦负 -100锛夈€?
    """
    images = batch["image"].to(device)
    if images.dim() == 3:
        images = images.unsqueeze(1)
    roi_center = batch.get("roi_center", None)
    roi_center_3d = batch.get("roi_center_3d", None)
    if roi_center is not None:
        roi_center = torch.as_tensor(roi_center, device=device, dtype=torch.float32)
    if roi_center_3d is not None:
        roi_center_3d = torch.as_tensor(roi_center_3d, device=device, dtype=torch.float32)
    questions = batch["question"]
    answers = batch["answer"]
    B = images.shape[0]

    # Visual encoder + bridge
    if use_gradient_checkpointing and vision_bridge.training:
        if roi_center is None and roi_center_3d is None:
            vis_tokens = torch.utils.checkpoint.checkpoint(
                lambda x: vision_bridge(x),
                images,
                use_reentrant=False,
            )
        else:
            roi2d = roi_center if roi_center is not None else torch.full((B, 2), -1.0, device=device)
            roi3d = roi_center_3d if roi_center_3d is not None else torch.full((B, 3), -1.0, device=device)
            vis_tokens = torch.utils.checkpoint.checkpoint(
                lambda x, r2d, r3d: vision_bridge(x, roi_center=r2d, roi_center_3d=r3d),
                images,
                roi2d,
                roi3d,
                use_reentrant=False,
            )
    else:
        vis_tokens = vision_bridge(images, roi_center=roi_center, roi_center_3d=roi_center_3d)
    # VimBridge 鍐呴儴浼氬皢 queries 杈撳嚭缂撳瓨鍒?latest_queries_out锛屼緵鍒嗙骇浠诲姟浣跨敤
    bridge = getattr(vision_bridge, "bridge", None)
    queries_out = getattr(bridge, "latest_queries_out", None) if bridge is not None else None

    vis = _pool_visual_tokens(vis_tokens, max_visual_tokens)
    L_vis = vis.shape[1]

    # 鏂囨湰锛氫笌鎺ㄧ悊涓€鑷存牸寮忎负 "闂\n鍥炵瓟"
    full_texts = [f"{q}\n{a}" for q, a in zip(questions, answers)]
    enc = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_text_len,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # 浠呭銆屽洖绛斻€嶇畻 loss锛沺rompt 闀垮害 = "闂\n" 鐨?token 鏁帮紙涓庢帹鐞嗕竴鑷达級
    q_texts = [f"{q}\n" for q in questions]
    q_enc = tokenizer(
        q_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_text_len,
    )
    q_lens = q_enc["attention_mask"].sum(dim=1).tolist()

    # Embedding锛圠LM 宓屽叆缁撮』涓?bridge_d_model 涓€鑷达紝鍚﹀垯闇€鎶曞奖鎴?pad/trim锛?
    text_emb = embed(input_ids.to(llm_device))
    E = text_emb.shape[-1]
    if E != d_model:
        if E < d_model:
            pad = torch.zeros(B, text_emb.shape[1], d_model - E, device=text_emb.device, dtype=text_emb.dtype)
            text_emb = torch.cat([text_emb, pad], dim=-1)
        else:
            text_emb = text_emb[:, :, :d_model]
    vis = vis.to(llm_device)

    # 鍙€?CMI
    cmi = getattr(vision_bridge, "cmi_connector", None)
    if cmi is not None:
        prompt_len = int(max(q_lens)) if q_lens else 0
        prompt_embeds = text_emb[:, :prompt_len, :]
        vis = cmi(vis, prompt_embeds)
        L_vis = vis.shape[1]

    inputs_embeds = torch.cat([vis, text_emb], dim=1)
    seq_len = inputs_embeds.shape[1]
    attn_mask = torch.ones((B, seq_len), device=llm_device, dtype=torch.long)
    attn_mask[:, L_vis:] = attention_mask.to(llm_device)

    # Labels锛氫粎鍥炵瓟閮ㄥ垎鏈夋晥锛屽叾浣?-100
    labels = torch.full(
        (B, seq_len), -100, device=llm_device, dtype=torch.long
    )
    for i in range(B):
        valid_len = int(attention_mask[i].sum().item())
        q_len = int(q_lens[i])
        if q_len > valid_len:
            q_len = valid_len
        start = L_vis + q_len
        end = L_vis + valid_len
        if end > start:
            labels[i, start:end] = input_ids[i, q_len:valid_len].to(llm_device)

    # 鍓嶅悜 + loss锛坙ogits[t] 棰勬祴 position t+1锛?
    # Keep LLM in train mode only when LoRA/adapters are trainable.
    if any(p.requires_grad for p in llm_model.parameters()):
        llm_model.train()
    else:
        llm_model.eval()
    out = llm_model(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
    logits = out.logits[:, : seq_len - 1]
    shift_labels = labels[:, 1:]
    caption_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )
    # 渚垫鼎/鍒嗙骇鎹熷け锛堝彲閫夛級
    cls_loss = torch.tensor(0.0, device=llm_device)
    cls_acc = float("nan")
    cls_mae = float("nan")
    cls_valid_n = 0
    grades_raw = batch.get("grade", None)
    if grade_head is not None and grades_raw is not None:
        grades = torch.as_tensor(grades_raw, device=llm_device)
        valid_mask = grades >= 0
        if valid_mask.any():
            # Ordinal regression uses a stable visual representation as cls input.
            cls_repr = None
            if queries_out is not None:
                try:
                    cls_repr = queries_out.to(llm_device).mean(dim=1)  # [B, D]
                except Exception:
                    cls_repr = None
            if cls_repr is None or not torch.isfinite(cls_repr).all():
                # Fallback to pooled visual tokens when cached queries are invalid.
                cls_repr = vis_tokens.to(llm_device).mean(dim=1)
            if not torch.isfinite(cls_repr).all():
                # Final safety net against NaN/Inf.
                cls_repr = torch.nan_to_num(cls_repr, nan=0.0, posinf=1e4, neginf=-1e4)

            cls_input = cls_repr[valid_mask]
            cls_labels = grades[valid_mask].to(llm_device).long()
            cls_logits = grade_head(cls_input)
            ordinal_bins = int(cls_logits.shape[-1])
            # y in {0,1,2,3} -> [y>0, y>1, y>2], e.g. 2 -> [1,1,0].
            thresholds = torch.arange(ordinal_bins, device=cls_logits.device).unsqueeze(0)
            ordinal_labels = (cls_labels.unsqueeze(1) > thresholds).to(cls_logits.dtype)
            # Ordinal BCE baseline + optional focal reweighting for long-tail robustness.
            bce_raw = F.binary_cross_entropy_with_logits(
                cls_logits,
                ordinal_labels,
                reduction="none",
            )
            if cls_focal_gamma > 0:
                probs = torch.sigmoid(cls_logits)
                p_t = probs * ordinal_labels + (1.0 - probs) * (1.0 - ordinal_labels)
                focal_factor = (1.0 - p_t).pow(cls_focal_gamma)
                if 0.0 <= cls_focal_alpha <= 1.0:
                    alpha_factor = ordinal_labels * cls_focal_alpha + (1.0 - ordinal_labels) * (1.0 - cls_focal_alpha)
                    focal_factor = focal_factor * alpha_factor
                cls_loss = (focal_factor * bce_raw).mean()
            else:
                cls_loss = bce_raw.mean()
            cls_valid_n = int(cls_labels.numel())
            if cls_valid_n > 0:
                with torch.no_grad():
                    # Ordinal decode for metrics: pred in {0,1,2,3}.
                    cls_pred = (torch.sigmoid(cls_logits) > 0.5).long().sum(dim=1)
                    cls_acc = float((cls_pred == cls_labels).float().mean().item())
                    cls_mae = float((cls_pred.float() - cls_labels.float()).abs().mean().item())
        else:
            # 鐢?dummy loss 淇濊瘉璁＄畻鍥句笉鏂
            cls_loss = grade_head.weight.sum() * 0.0

    total_loss = caption_loss + lambda_cls * cls_loss
    return (
        total_loss,
        float(caption_loss.detach().cpu()),
        float(cls_loss.detach().cpu()),
        cls_acc,
        cls_mae,
        cls_valid_n,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-End VLM 璁粌锛氳В鍐?Vision+Bridge锛孧amba 鍐荤粨")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5, help="瀛︿範鐜囷紱绔埌绔В鍐绘椂寤鸿 1e-5")
    parser.add_argument(
        "--max_visual_tokens",
        type=int,
        default=164,
        help="Max visual tokens after dual-stream fusion (Global 8x8=64 + Local 10x10=100 => 164)",
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="full",
        choices=["full", "global_only", "local_only"],
        help="Visual stream ablation: full=global+local, global_only=64 tokens, local_only=100 tokens.",
    )
    parser.add_argument("--vision_bridge_type", type=str, default="vim", choices=["vim", "transformer"], help="Vision bridge type.")
    parser.add_argument("--spatial_dims", type=int, default=2, choices=[2, 3], help="2=2D slices, 3=3D patches.")
    parser.add_argument(
        "--patch_size_3d",
        type=lambda s: _parse_int_tuple(s, 3),
        default=(32, 128, 128),
        help="3D patch size as D,H,W (e.g. 32,128,128).",
    )
    parser.add_argument("--max_text_len", type=int, default=DEFAULT_MAX_TEXT_LEN, help="闂+鎶ュ憡鎬?token 涓婇檺")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="姊害绱Н姝ユ暟")
    parser.add_argument("--save_every_steps", type=int, default=0, help="Save an extra checkpoint every N steps; 0 means only save at epoch end")
    parser.add_argument("--log_every_steps", type=int, default=1, help="姣?N 姝ユ墦鍗颁竴娆?loss锛岄粯璁?1锛堟瘡姝ユ墦鍗帮級")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader worker count")
    parser.add_argument("--vision_checkpoint", type=str, default=None, help="鍙€夛細浠庡凡鏈?Vision+Bridge 鏉冮噸缁х画璁粌")
    parser.add_argument("--mamba_model", type=str, default="state-spaces/mamba-2.8b-hf")
    parser.add_argument("--llm_8bit", action="store_true", help="Load LLM in 8-bit mode (requires bitsandbytes)")
    parser.add_argument(
        "--bf16",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable bf16 autocast on CUDA (supports '--bf16' or '--bf16 True').",
    )
    parser.add_argument("--align_vocab", action="store_true", help="璁粌绔榻?tokenizer 涓?embedding 璇嶈〃澶у皬锛堟帹鑽愬紑鍚級")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--lambda_cls", type=float, default=1.0, help="classification loss weight")
    parser.add_argument("--cls_focal_gamma", type=float, default=0.0, help="Focal gamma for ordinal cls BCE (0 means plain BCE)")
    parser.add_argument(
        "--cls_focal_alpha",
        type=float,
        default=-1.0,
        help="Optional focal alpha in [0,1]; negative disables alpha weighting.",
    )
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning on LLM instead of full freeze.")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="in_proj,x_proj,dt_proj",
        help="Comma-separated module name hints for LoRA target discovery.",
    )
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        choices=["none", "all", "lora_only"],
        help="PEFT LoRA bias mode.",
    )
    parser.add_argument("--gradient_checkpointing", dest="gradient_checkpointing", action="store_true", help="鍚敤姊害妫€鏌ョ偣锛岀渷鏄惧瓨銆佺暐闄嶉€燂紙榛樿寮€鍚級")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false", help="鍏抽棴姊害妫€鏌ョ偣")
    parser.add_argument("--csv", type=str, default=None, help="鐩存帴鎸囧畾 caption CSV锛屾湁鍊兼椂浼樺厛浜?paths.yaml")
    parser.add_argument("--length_audit_samples", type=int, default=128, help="璁粌鍓嶆娊鏍风粺璁?token 闀垮害锛? 琛ㄧず鍏抽棴")
    parser.add_argument(
        "--plot_every_epochs",
        type=int,
        default=1,
        help="Generate paper-friendly training charts every N epochs (0 disables periodic plotting).",
    )
    parser.add_argument(
        "--disable_plot_assets",
        action="store_true",
        help="Disable auto-generation of training charts/tables.",
    )
    parser.set_defaults(gradient_checkpointing=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir or str(REPO / "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_paths_config(REPO / "config" / "paths.yaml")
    config.setdefault("encoder_output_stage", 4)
    config.setdefault("encoder_target_spatial", 28)
    config["bridge_d_model"] = 2560
    config["ablation_mode"] = str(args.ablation_mode)
    config["vision_bridge_type"] = str(args.vision_bridge_type).lower()
    config["spatial_dims"] = int(args.spatial_dims)
    config["patch_size_3d"] = tuple(args.patch_size_3d)

    csv_train = args.csv or config.get("caption_csv_train")
    if not Path(csv_train).exists():
        print("璁粌 CSV 涓嶅瓨鍦?", csv_train)
        return 1

    train_ds = MedicalVLMDataset(
        csv_train,
        prompt_json_file=config.get("caption_prompt_json"),
        spatial_dims=int(args.spatial_dims),
        patch_size_3d=tuple(args.patch_size_3d),
    )
    # Classification classes: AAH/AIS/MIA/IAC
    num_grades = 4
    # 鏍规嵁鏁版嵁鍒嗗竷浼拌绫诲埆鏉冮噸锛岀紦瑙ｇ被鍒笉骞宠　锛涜嫢鏃?grade 鍒楀垯涓?None
    class_weights = None
    valid_grade_count = 0
    present_grade_classes = 0
    if hasattr(train_ds, "grades"):
        counts = torch.zeros(num_grades, dtype=torch.long)
        for g in getattr(train_ds, "grades", []):
            try:
                gi = int(g)
            except Exception:
                continue
            if 0 <= gi < num_grades:
                counts[gi] += 1
        valid_grade_count = int(counts.sum().item())
        present_grade_classes = int((counts > 0).sum().item())
        if counts.sum() > 0:
            counts_f = counts.float()
            # Avoid zero-count divisions by filling zeros with non-zero mean.
            if (counts_f == 0).any() and (counts_f > 0).any():
                nz_mean = counts_f[counts_f > 0].mean()
                counts_f[counts_f == 0] = nz_mean
            inv = counts_f.sum() / (counts_f + 1e-6)
            class_weights = (inv / inv.mean()).clone()
            print(f"detected valid grade labels: {valid_grade_count}, distribution={counts.tolist()}", flush=True)
        else:
            print(
                "warning: train CSV has no valid grade(0-3); cls_loss will stay 0. Add grade column for classification training.",
                flush=True,
            )
    else:
        print(
            "warning: dataset has no grades field; cls_loss will stay 0. Add grade column to CSV for classification training.",
            flush=True,
        )

    if valid_grade_count == 0:
        class_weights = None
    if valid_grade_count > 0 and present_grade_classes < 2:
        print(
            "warning: only one grade class is present, classification head will be disabled.",
            flush=True,
        )
    enable_cls_head = (
        float(args.lambda_cls) > 0.0
        and valid_grade_count > 0
        and present_grade_classes >= 2
    )
    lambda_cls = float(args.lambda_cls) if enable_cls_head else 0.0
    print(f"lambda_cls={lambda_cls:.4f} (requested={float(args.lambda_cls):.4f})", flush=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        # 5090/Ada+ usually benefits from TF32 matmul throughput.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    nw = args.num_workers if use_cuda else 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=use_cuda,
    )

    # Vision+Bridge
    vision_bridge = build_medical_vlm_from_config(config)
    ckpt = args.vision_checkpoint
    loaded_ckpt = None
    if ckpt:
        if not Path(ckpt).exists():
            print(f"鎸囧畾鐨?--vision_checkpoint 涓嶅瓨鍦? {ckpt}", flush=True)
            return 1
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        sd = state.get("model_state_dict", state)
        vision_bridge.load_state_dict(sd, strict=False)
        if isinstance(state, dict):
            loaded_ckpt = state
        step_info = f" (浠?step {state.get('step', '?')} 缁)" if isinstance(state, dict) and state.get("step") else ""
        print(f"Loaded Vision+Bridge: {ckpt}{step_info}", flush=True)
    else:
        print("--vision_checkpoint not provided: Vision Encoder + Bridge will be initialized and trained end-to-end.", flush=True)
    vision_bridge = vision_bridge.to(device)
    print(
        f"ablation_mode={args.ablation_mode}, visual_seq_len={getattr(vision_bridge, 'visual_seq_len', 'unknown')}",
        flush=True,
    )
    # 鍒嗙骇澶达細杈撳叆缁村害涓?bridge_d_model 涓€鑷达紝杈撳嚭 4 绫伙紙AAH/AIS/MIA/IAC锛?
    ordinal_bins = num_grades - 1
    grade_head = torch.nn.Linear(config.get("bridge_d_model", 2560), ordinal_bins, device=device)
    # Resume cls head from checkpoint when available.
    if isinstance(loaded_ckpt, dict):
        grade_sd = loaded_ckpt.get("grade_head_state_dict") or loaded_ckpt.get("cls_head_state_dict")
        if isinstance(grade_sd, dict) and grade_sd:
            cur_sd = grade_head.state_dict()
            safe_sd = {}
            for k, v in grade_sd.items():
                if k in cur_sd and isinstance(v, torch.Tensor) and tuple(v.shape) == tuple(cur_sd[k].shape):
                    safe_sd[k] = v
            if safe_sd:
                miss, unexp = grade_head.load_state_dict(safe_sd, strict=False)
                print(
                    f"Loaded grade_head from checkpoint: keys={list(safe_sd.keys())}, "
                    f"missing={list(miss)}, unexpected={list(unexp)}",
                    flush=True,
                )
            else:
                print("warning: checkpoint has cls head but shape mismatches current ordinal head; skip loading.", flush=True)
    if enable_cls_head:
        print("grade_head enabled.", flush=True)
    else:
        for p in grade_head.parameters():
            p.requires_grad = False
        print("grade_head disabled (kept for checkpoint compatibility).", flush=True)
    # End-to-end: unfreeze encoder and bridge together.
    if hasattr(vision_bridge, "encoder"):
        for p in vision_bridge.encoder.parameters():
            p.requires_grad = True
        print("宸茶В鍐?nnunet_encoder锛岀鍒扮璁粌 Vision+Bridge", flush=True)
        if getattr(args, "gradient_checkpointing", False):
            if _enable_encoder_gradient_checkpointing(vision_bridge.encoder):
                print("宸插惎鐢?encoder gradient checkpointing", flush=True)
            else:
                print("encoder has no native gradient-checkpointing API; keep outer checkpoint wrapper.", flush=True)
    trainable = [p for p in vision_bridge.parameters() if p.requires_grad]
    if enable_cls_head:
        trainable.extend(list(grade_head.parameters()))

    # Mamba 鍐荤粨
    print("鍔犺浇 Mamba锛堝喕缁擄級...", flush=True)
    from llm.mamba_loader import load_mamba_lm
    llm_model, tokenizer = load_mamba_lm(
        args.mamba_model,
        device_map="auto" if device.type == "cuda" else None,
        load_in_8bit=getattr(args, "llm_8bit", False),
        align_vocab=getattr(args, "align_vocab", False),
    )
    if hasattr(llm_model, "backbone") and hasattr(llm_model.backbone, "embeddings") and hasattr(llm_model, "lm_head"):
        llm_model.lm_head.weight = llm_model.backbone.embeddings.weight
    if hasattr(llm_model, "gradient_checkpointing_enable"):
        llm_model.gradient_checkpointing_enable()
    if hasattr(llm_model, "config"):
        llm_model.config.use_cache = False
    for p in llm_model.parameters():
        p.requires_grad = False
    lora_trainable_params = 0
    if args.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except Exception as e:
            print(f"[error] --use_lora requires peft. Install with `pip install peft`. detail={e}", flush=True)
            return 1
        hints = [x.strip() for x in str(args.lora_target_modules).split(",") if x.strip()]
        target_modules = _guess_lora_target_modules(llm_model, hints)
        if not target_modules:
            print(f"[error] no LoRA target modules matched hints={hints}", flush=True)
            return 1
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            target_modules=target_modules,
            bias=str(args.lora_bias),
        )
        try:
            llm_model = get_peft_model(llm_model, lora_cfg)
        except ValueError as e:
            # Auto-correct for Mamba restrictions in PEFT.
            msg = str(e)
            if "incompatible" in msg.lower() or "mamba-based models" in msg.lower():
                safe_targets = [m for m in target_modules if m not in {"out_proj", "conv1d"}]
                if not safe_targets:
                    safe_targets = ["in_proj", "x_proj", "dt_proj"]
                print(
                    f"[warn] LoRA target modules incompatible for Mamba, retry with {safe_targets}",
                    flush=True,
                )
                lora_cfg = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=int(args.lora_r),
                    lora_alpha=int(args.lora_alpha),
                    lora_dropout=float(args.lora_dropout),
                    target_modules=safe_targets,
                    bias=str(args.lora_bias),
                )
                llm_model = get_peft_model(llm_model, lora_cfg)
                target_modules = safe_targets
            else:
                raise
        if hasattr(llm_model, "print_trainable_parameters"):
            llm_model.print_trainable_parameters()
        lora_trainable_params = int(sum(p.numel() for p in llm_model.parameters() if p.requires_grad))
        print(f"LoRA enabled on LLM. target_modules={target_modules}", flush=True)
    else:
        print("LLM fully frozen (no LoRA).", flush=True)
    llm_device = next(llm_model.parameters()).device
    embed = llm_model.get_input_embeddings()
    d_model = llm_model.config.hidden_size
    # 纭繚鍒嗙骇澶翠笌 LLM/瑙嗚鍦ㄥ悓涓€璁惧涓婏紝閬垮厤 device mismatch
    if grade_head is not None:
        grade_head = grade_head.to(llm_device)
    if args.use_lora:
        lora_trainable = [p for p in llm_model.parameters() if p.requires_grad]
        if lora_trainable:
            trainable.extend(lora_trainable)
            print(f"added LoRA params to optimizer: {lora_trainable_params}", flush=True)
        else:
            print("[warn] LoRA requested but no trainable params found.", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and bool(args.bf16)) else torch.float16
    scaler = _build_grad_scaler(device, amp_dtype=amp_dtype)
    max_visual_tokens = getattr(args, "max_visual_tokens", 164)
    max_text_len = getattr(args, "max_text_len", DEFAULT_MAX_TEXT_LEN)
    _summarize_text_token_lengths(
        train_ds,
        tokenizer,
        max_text_len=max_text_len,
        max_samples=getattr(args, "length_audit_samples", 0),
    )

    print(f"max_visual_tokens={max_visual_tokens}锛堟帹鐞嗘椂椤讳竴鑷达級", flush=True)
    print(f"max_text_len={max_text_len}, gradient_accumulation_steps={args.gradient_accumulation_steps}", flush=True)
    if device.type == "cuda":
        try:
            alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            print(
                f"褰撳墠 GPU 鏄惧瓨: 宸插垎閰?{alloc_gb:.2f} GB, 宸查鐣?{reserved_gb:.2f} GB",
                flush=True,
            )
        except Exception:
            pass

    stage2_config = {
        "max_visual_tokens": max_visual_tokens,
        "ablation_mode": str(args.ablation_mode),
        "visual_seq_len": int(getattr(vision_bridge, "visual_seq_len", max_visual_tokens)),
        "mamba_model": args.mamba_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_text_len": max_text_len,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "classifier_mode": "ordinal_bce",
        "ordinal_bins": ordinal_bins,
        "cls_focal_gamma": float(args.cls_focal_gamma),
        "cls_focal_alpha": float(args.cls_focal_alpha),
        "use_lora": bool(args.use_lora),
        "bf16": bool(args.bf16),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lora_target_modules": args.lora_target_modules,
        "plot_every_epochs": int(args.plot_every_epochs),
    }
    (out_dir / "stage2_config.json").write_text(json.dumps(stage2_config, indent=2), encoding="utf-8")

    # 鏃ュ織
    log_csv = out_dir / "stage2_train_log.csv"
    epoch_csv = out_dir / "stage2_epoch_metrics.csv"
    paper_dir = out_dir / "paper_assets"
    heartbeat_path = out_dir / "stage2_heartbeat.txt"
    log_writer = None
    epoch_writer = None
    epoch_fh = None
    try:
        log_fh = open(log_csv, "w", encoding="utf-8", newline="")
        log_writer = csv.DictWriter(
            log_fh,
            fieldnames=[
                "step",
                "epoch",
                "total_loss",
                "caption_loss",
                "cls_loss",
                "cls_acc",
                "cls_mae",
                "cls_valid_n",
                "lr",
                "step_time_sec",
                "samples_per_sec",
                "gpu_mem_alloc_gb",
                "gpu_mem_reserved_gb",
            ],
        )
        log_writer.writeheader()
        log_fh.flush()
    except OSError as e:
        log_fh = None
        print(f"璀﹀憡: 鏃犳硶鍐欐棩蹇?{log_csv}: {e}", flush=True)

    try:
        epoch_fh = open(epoch_csv, "w", encoding="utf-8", newline="")
        epoch_writer = csv.DictWriter(
            epoch_fh,
            fieldnames=[
                "epoch",
                "steps",
                "avg_total_loss",
                "avg_caption_loss",
                "avg_cls_loss",
                "avg_cls_acc",
                "avg_cls_mae",
                "avg_step_time_sec",
                "avg_samples_per_sec",
                "lr_end",
            ],
        )
        epoch_writer.writeheader()
        epoch_fh.flush()
    except OSError as e:
        epoch_fh = None
        epoch_writer = None
        print(f"[warn] cannot open epoch metrics csv {epoch_csv}: {e}", flush=True)

    use_amp = device.type == "cuda"
    use_scaler = scaler is not None
    use_grad_ckpt = getattr(args, "gradient_checkpointing", False)
    if use_grad_ckpt:
        print("gradient_checkpointing enabled", flush=True)
    global_step = 0
    epoch_avgs = []
    first_batch_done = False
    accum_steps = max(1, args.gradient_accumulation_steps)
    grade_head_for_loss = grade_head if enable_cls_head else None

    # 棣?batch 鍓嶏細纭 Loss 鍙銆屽洖绛斻€嶉儴鍒嗚绠楋紙渚夸簬鎺掓煡骞昏/鑳岃闂锛?
    try:
        one = next(iter(train_loader))
        q0 = (one["question"] if isinstance(one["question"], str) else one["question"][0])
        a0 = (one["answer"] if isinstance(one["answer"], str) else one["answer"][0])
        sample_full = f"{q0}\n{a0}"
        q_enc = tokenizer([q0 + "\n"], return_tensors="pt", truncation=True, max_length=max_text_len)
        a_enc = tokenizer([a0], return_tensors="pt", truncation=True, max_length=max_text_len)
        full_enc = tokenizer([sample_full], return_tensors="pt", truncation=True, max_length=max_text_len)
        prompt_len = q_enc["input_ids"].shape[1]
        answer_len = a_enc["input_ids"].shape[1]
        full_len = full_enc["input_ids"].shape[1]
        print(f"[debug] 鏍锋湰0: prompt(闂+鎹㈣) token 鏁?{prompt_len}, answer token 鏁?{answer_len}, full token 鏁?{full_len}; Loss 浠呭 answer 閮ㄥ垎璁＄畻", flush=True)
        # 鎵撳嵃璁粌鏂囨湰鏍煎紡锛岀‘璁ゆ槸 "question\\nanswer"銆?
        print(f"[debug] 鏍锋湰0璁粌鏂囨湰棰勮: {sample_full[:280].replace(chr(10), ' | ')}", flush=True)
        # 瑙ｇ爜 full/answer 鍓嶈嫢骞?token锛岀洿瑙傜湅妯″瀷鍦ㄥ浠€涔堟牸寮忋€?
        full_ids = full_enc["input_ids"][0]
        n_full = min(100, full_ids.size(0) if hasattr(full_ids, "size") else len(full_ids))
        full_decode = tokenizer.decode(
            full_ids[:n_full].tolist() if hasattr(full_ids, "tolist") else list(full_ids[:n_full]),
            skip_special_tokens=True,
        )
        print(f"[debug] 鏍锋湰0 full 鍓嶇害100 token 瑙ｇ爜: {full_decode[:240]}...", flush=True)
        answer_ids = a_enc["input_ids"][0]
        n_show = min(60, answer_ids.size(0) if hasattr(answer_ids, "size") else len(answer_ids))
        head_decode = tokenizer.decode(
            answer_ids[:n_show].tolist() if hasattr(answer_ids, "tolist") else list(answer_ids[:n_show]),
            skip_special_tokens=True,
        )
        print(f"[debug] 鏍锋湰0 answer 閮ㄥ垎鍓嶇害60 token 瑙ｇ爜: {head_decode[:200]}...", flush=True)
    except Exception as e:
        print(f"[debug] 棣?batch 妫€鏌ヨ烦杩? {e}", flush=True)

    for epoch in range(args.epochs):
        vision_bridge.train()
        epoch_losses = []
        epoch_caption_losses = []
        epoch_cls_losses = []
        epoch_cls_acc_vals = []
        epoch_cls_mae_vals = []
        epoch_step_times = []
        epoch_sps_vals = []
        optimizer.zero_grad(set_to_none=True)

        for batch in train_loader:
            # Heartbeat to show progress even when the first step is slow.
            try:
                heartbeat_path.write_text(
                    f"epoch={epoch+1} step={global_step+1} start\n",
                    encoding="utf-8",
                )
            except OSError:
                pass
            if global_step % args.log_every_steps == 0:
                print(f"epoch {epoch+1} step {global_step+1} start", flush=True)
            step_t0 = time.perf_counter()
            if use_amp:
                with _autocast_ctx(device, amp_dtype=amp_dtype):
                    (
                        loss,
                        cap_loss_val,
                        cls_loss_val,
                        cls_acc_val,
                        cls_mae_val,
                        cls_valid_n,
                    ) = compute_batch_loss(
                        batch,
                        vision_bridge,
                        llm_model,
                        embed,
                        tokenizer,
                        device,
                        llm_device,
                        max_visual_tokens,
                        max_text_len,
                        d_model,
                        use_gradient_checkpointing=use_grad_ckpt,
                        grade_head=grade_head_for_loss,
                        class_weights=class_weights,
                        lambda_cls=lambda_cls,
                        cls_focal_gamma=float(args.cls_focal_gamma),
                        cls_focal_alpha=float(args.cls_focal_alpha),
                    )
                loss = loss / accum_steps
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                (
                    loss,
                    cap_loss_val,
                    cls_loss_val,
                    cls_acc_val,
                    cls_mae_val,
                    cls_valid_n,
                ) = compute_batch_loss(
                    batch,
                    vision_bridge,
                    llm_model,
                    embed,
                    tokenizer,
                    device,
                    llm_device,
                    max_visual_tokens,
                    max_text_len,
                    d_model,
                    use_gradient_checkpointing=use_grad_ckpt,
                    grade_head=grade_head_for_loss,
                    class_weights=class_weights,
                    lambda_cls=lambda_cls,
                    cls_focal_gamma=float(args.cls_focal_gamma),
                    cls_focal_alpha=float(args.cls_focal_alpha),
                )
                loss = loss / accum_steps
                loss.backward()

            step_time_sec = max(time.perf_counter() - step_t0, 1e-8)
            batch_size_cur = 0
            images_obj = batch.get("image")
            if torch.is_tensor(images_obj) and images_obj.ndim >= 1:
                batch_size_cur = int(images_obj.shape[0])
            elif isinstance(batch.get("question"), list):
                batch_size_cur = len(batch["question"])
            elif isinstance(batch.get("answer"), list):
                batch_size_cur = len(batch["answer"])
            if batch_size_cur <= 0:
                batch_size_cur = int(args.batch_size)
            samples_per_sec = float(batch_size_cur) / step_time_sec

            if device.type == "cuda":
                try:
                    gpu_mem_alloc_gb = float(torch.cuda.memory_allocated() / (1024 ** 3))
                    gpu_mem_reserved_gb = float(torch.cuda.memory_reserved() / (1024 ** 3))
                except Exception:
                    gpu_mem_alloc_gb = float("nan")
                    gpu_mem_reserved_gb = float("nan")
            else:
                gpu_mem_alloc_gb = 0.0
                gpu_mem_reserved_gb = 0.0

            loss_val = loss.item() * accum_steps
            if not first_batch_done:
                first_batch_done = True
                if loss_val < 0.1:
                    print(f"璀﹀憡: 棣栨 loss={loss_val:.4f} 寮傚父浣庯紝璇锋鏌?label 鏄惁鍙鍥炵瓟閮ㄥ垎", flush=True)
                elif loss_val > 15.0:
                    print(f"note: first-step loss={loss_val:.4f} is relatively high", flush=True)

            epoch_losses.append(loss_val)
            epoch_caption_losses.append(cap_loss_val)
            epoch_cls_losses.append(cls_loss_val)
            if math.isfinite(cls_acc_val):
                epoch_cls_acc_vals.append(float(cls_acc_val))
            if math.isfinite(cls_mae_val):
                epoch_cls_mae_vals.append(float(cls_mae_val))
            epoch_step_times.append(step_time_sec)
            epoch_sps_vals.append(samples_per_sec)
            global_step += 1

            if global_step % accum_steps == 0:
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % args.log_every_steps == 0:
                lr_now = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else float("nan")
                cls_extra = ""
                if cls_valid_n > 0 and math.isfinite(cls_acc_val):
                    cls_extra = f", cls_acc={cls_acc_val:.4f}, cls_mae={cls_mae_val:.4f}, cls_n={cls_valid_n}"
                print(
                    f"epoch {epoch+1} step {global_step} total_loss {loss_val:.4f} "
                    f"(caption={cap_loss_val:.4f}, cls={cls_loss_val:.4f}{cls_extra}, "
                    f"sps={samples_per_sec:.2f}, step_time={step_time_sec:.3f}s)",
                    flush=True,
                )
                if log_fh is not None and log_writer is not None:
                    try:
                        log_writer.writerow(
                            {
                                "step": global_step,
                                "epoch": epoch + 1,
                                "total_loss": f"{loss_val:.6f}",
                                "caption_loss": f"{cap_loss_val:.6f}",
                                "cls_loss": f"{cls_loss_val:.6f}",
                                "cls_acc": f"{cls_acc_val:.6f}" if math.isfinite(cls_acc_val) else "",
                                "cls_mae": f"{cls_mae_val:.6f}" if math.isfinite(cls_mae_val) else "",
                                "cls_valid_n": cls_valid_n,
                                "lr": f"{lr_now:.8e}",
                                "step_time_sec": f"{step_time_sec:.6f}",
                                "samples_per_sec": f"{samples_per_sec:.4f}",
                                "gpu_mem_alloc_gb": f"{gpu_mem_alloc_gb:.4f}" if math.isfinite(gpu_mem_alloc_gb) else "",
                                "gpu_mem_reserved_gb": f"{gpu_mem_reserved_gb:.4f}" if math.isfinite(gpu_mem_reserved_gb) else "",
                            }
                        )
                        log_fh.flush()
                    except OSError:
                        pass

            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0 and global_step > 0:
                ckpt_path = out_dir / f"vision_bridge_vlm_step{global_step}.pt"
                torch.save(
                    {
                        "step": global_step,
                        "model_state_dict": vision_bridge.state_dict(),
                        "grade_head_state_dict": grade_head.state_dict(),
                        "cls_head_state_dict": grade_head.state_dict(),
                        "num_grades": num_grades,
                        "classifier_mode": "ordinal_bce",
                        "ordinal_bins": ordinal_bins,
                    },
                    ckpt_path,
                )
                print(f"  [step {global_step}] saved {ckpt_path} (with grade_head)", flush=True)

        # epoch 鏈熬鑻ユ湁鏈?step 鐨勭疮绉搴︼紝琛ヤ竴娆?
        if accum_steps > 1 and global_step % accum_steps != 0:
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        avg_total_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        avg_cap_loss = sum(epoch_caption_losses) / len(epoch_caption_losses) if epoch_caption_losses else 0.0
        avg_cls_loss = sum(epoch_cls_losses) / len(epoch_cls_losses) if epoch_cls_losses else 0.0
        avg_cls_acc = (
            sum(epoch_cls_acc_vals) / len(epoch_cls_acc_vals) if epoch_cls_acc_vals else float("nan")
        )
        avg_cls_mae = (
            sum(epoch_cls_mae_vals) / len(epoch_cls_mae_vals) if epoch_cls_mae_vals else float("nan")
        )
        avg_step_time = sum(epoch_step_times) / len(epoch_step_times) if epoch_step_times else 0.0
        avg_sps = sum(epoch_sps_vals) / len(epoch_sps_vals) if epoch_sps_vals else 0.0
        epoch_avgs.append(avg_total_loss)
        lr_end = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else float("nan")

        if epoch_writer is not None and epoch_fh is not None:
            try:
                epoch_writer.writerow(
                    {
                        "epoch": epoch + 1,
                        "steps": len(epoch_losses),
                        "avg_total_loss": f"{avg_total_loss:.6f}",
                        "avg_caption_loss": f"{avg_cap_loss:.6f}",
                        "avg_cls_loss": f"{avg_cls_loss:.6f}",
                        "avg_cls_acc": f"{avg_cls_acc:.6f}" if math.isfinite(avg_cls_acc) else "",
                        "avg_cls_mae": f"{avg_cls_mae:.6f}" if math.isfinite(avg_cls_mae) else "",
                        "avg_step_time_sec": f"{avg_step_time:.6f}",
                        "avg_samples_per_sec": f"{avg_sps:.4f}",
                        "lr_end": f"{lr_end:.8e}",
                    }
                )
                epoch_fh.flush()
            except OSError:
                pass

        ckpt_path = out_dir / "vision_bridge_vlm_final.pt"
        torch.save(
            {
                "step": global_step,
                "model_state_dict": vision_bridge.state_dict(),
                "grade_head_state_dict": grade_head.state_dict(),
                "cls_head_state_dict": grade_head.state_dict(),
                "num_grades": num_grades,
                        "classifier_mode": "ordinal_bce",
                        "ordinal_bins": ordinal_bins,
            },
            ckpt_path,
        )
        print(
            f"epoch {epoch+1} done, avg_total={avg_total_loss:.4f}, avg_caption={avg_cap_loss:.4f}, "
            f"avg_cls={avg_cls_loss:.4f}, avg_cls_acc={avg_cls_acc if math.isfinite(avg_cls_acc) else 'N/A'}, "
            f"saved {ckpt_path} (with grade_head)",
            flush=True,
        )
        if not args.disable_plot_assets and int(args.plot_every_epochs) > 0 and ((epoch + 1) % int(args.plot_every_epochs) == 0):
            _write_training_paper_assets(log_csv=log_csv, epoch_csv=epoch_csv, out_dir=paper_dir)

    if log_fh is not None:
        try:
            log_fh.close()
        except OSError:
            pass
    print("VLM 璁粌瀹屾垚", flush=True)
    if epoch_avgs:
        recent = epoch_avgs[-5:] if len(epoch_avgs) >= 5 else epoch_avgs
        print(f"鏈€杩?epoch 骞冲潎 loss: {[f'{x:.4f}' for x in recent]}", flush=True)
        print(f"Loss 鏇茬嚎: {log_csv}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())



