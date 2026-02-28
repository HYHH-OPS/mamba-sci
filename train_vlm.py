"""
VLM 鍥惧儚鈫掓姤鍛婅缁冿細浠呰缁?Vision+Bridge锛孧amba 鍐荤粨銆?
杈撳叆锛氶棶棰?鎶ュ憡锛汱oss 鍙銆屾姤鍛娿€嶉儴鍒嗚绠楋紝涓庢帹鐞嗘椂銆岄棶棰?鎹㈣銆嶅悗鐢熸垚鎶ュ憡涓€鑷淬€?

鐢ㄦ硶:
  python train_vlm.py --epochs 30 --batch_size 8 --lr 1e-5 --max_visual_tokens 144
  python train_vlm.py --epochs 30 --batch_size 4 --lr 1e-5 --max_visual_tokens 144 --gradient_accumulation_steps 1
"""

from __future__ import annotations

import argparse
import json
import sys
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


def _autocast_ctx(device: torch.device):
    if device.type != "cuda":
        return _autocast(enabled=False)
    if _AMP_MODE == "torch.amp":
        return _autocast("cuda", dtype=torch.float16)
    # torch.cuda.amp.autocast has no device_type arg in old torch versions.
    return _autocast(dtype=torch.float16)


def _build_grad_scaler(device: torch.device):
    if device.type != "cuda":
        return None
    if _AMP_MODE == "torch.amp":
        try:
            return _GradScaler("cuda")
        except TypeError:
            return _GradScaler()
    return _GradScaler()


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
) -> tuple[torch.Tensor, float, float]:
    """
    鍗?batch 鍓嶅悜 + caption loss銆?
    Loss 浠呭銆屽洖绛斻€嶉儴鍒嗚绠楋紙瑙嗚+闂+鎹㈣ 鍧囦负 -100锛夈€?
    """
    images = batch["image"].to(device)
    if images.dim() == 3:
        images = images.unsqueeze(1)
    questions = batch["question"]
    answers = batch["answer"]
    B = images.shape[0]

    # 瑙嗚缂栫爜 + 姹犲寲锛堝彲閫夋搴︽鏌ョ偣鐪佹樉瀛橈級
    if use_gradient_checkpointing and vision_bridge.training:
        vis_tokens = torch.utils.checkpoint.checkpoint(vision_bridge, images, use_reentrant=False)
    else:
        vis_tokens = vision_bridge(images)
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
            # Baseline version: no class weight / pos_weight.
            cls_loss = F.binary_cross_entropy_with_logits(
                cls_logits,
                ordinal_labels,
                reduction="mean",
            )
        else:
            # 鐢?dummy loss 淇濊瘉璁＄畻鍥句笉鏂
            cls_loss = grade_head.weight.sum() * 0.0

    total_loss = caption_loss + lambda_cls * cls_loss
    return total_loss, float(caption_loss.detach().cpu()), float(cls_loss.detach().cpu())


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-End VLM 璁粌锛氳В鍐?Vision+Bridge锛孧amba 鍐荤粨")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5, help="瀛︿範鐜囷紱绔埌绔В鍐绘椂寤鸿 1e-5")
    parser.add_argument("--max_visual_tokens", type=int, default=144, help="Max visual tokens after pooling (e.g. 12x12=144)")
    parser.add_argument("--max_text_len", type=int, default=DEFAULT_MAX_TEXT_LEN, help="闂+鎶ュ憡鎬?token 涓婇檺")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="姊害绱Н姝ユ暟")
    parser.add_argument("--save_every_steps", type=int, default=0, help="Save an extra checkpoint every N steps; 0 means only save at epoch end")
    parser.add_argument("--log_every_steps", type=int, default=1, help="姣?N 姝ユ墦鍗颁竴娆?loss锛岄粯璁?1锛堟瘡姝ユ墦鍗帮級")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader worker count")
    parser.add_argument("--vision_checkpoint", type=str, default=None, help="鍙€夛細浠庡凡鏈?Vision+Bridge 鏉冮噸缁х画璁粌")
    parser.add_argument("--mamba_model", type=str, default="state-spaces/mamba-2.8b-hf")
    parser.add_argument("--llm_8bit", action="store_true", help="Load LLM in 8-bit mode (requires bitsandbytes)")
    parser.add_argument("--align_vocab", action="store_true", help="璁粌绔榻?tokenizer 涓?embedding 璇嶈〃澶у皬锛堟帹鑽愬紑鍚級")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--lambda_cls", type=float, default=1.0, help="classification loss weight")
    parser.add_argument("--gradient_checkpointing", dest="gradient_checkpointing", action="store_true", help="鍚敤姊害妫€鏌ョ偣锛岀渷鏄惧瓨銆佺暐闄嶉€燂紙榛樿寮€鍚級")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false", help="鍏抽棴姊害妫€鏌ョ偣")
    parser.add_argument("--csv", type=str, default=None, help="鐩存帴鎸囧畾 caption CSV锛屾湁鍊兼椂浼樺厛浜?paths.yaml")
    parser.add_argument("--length_audit_samples", type=int, default=128, help="璁粌鍓嶆娊鏍风粺璁?token 闀垮害锛? 琛ㄧず鍏抽棴")
    parser.set_defaults(gradient_checkpointing=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir or str(REPO / "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_paths_config(REPO / "config" / "paths.yaml")
    config.setdefault("encoder_output_stage", 4)
    config.setdefault("encoder_target_spatial", 28)
    config["bridge_d_model"] = 2560

    csv_train = args.csv or config.get("caption_csv_train")
    if not Path(csv_train).exists():
        print("璁粌 CSV 涓嶅瓨鍦?", csv_train)
        return 1

    train_ds = MedicalVLMDataset(csv_train, prompt_json_file=config.get("caption_prompt_json"))
    # 渚垫鼎/鍒嗙骇绫诲埆鏁帮紙AAH/AIS/MIA/IAC锛?    num_grades = 4
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
    if ckpt:
        if not Path(ckpt).exists():
            print(f"鎸囧畾鐨?--vision_checkpoint 涓嶅瓨鍦? {ckpt}", flush=True)
            return 1
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        sd = state.get("model_state_dict", state)
        vision_bridge.load_state_dict(sd, strict=False)
        step_info = f" (浠?step {state.get('step', '?')} 缁)" if isinstance(state, dict) and state.get("step") else ""
        print(f"Loaded Vision+Bridge: {ckpt}{step_info}", flush=True)
    else:
        print("--vision_checkpoint not provided: Vision Encoder + Bridge will be initialized and trained end-to-end.", flush=True)
    vision_bridge = vision_bridge.to(device)
    # 鍒嗙骇澶达細杈撳叆缁村害涓?bridge_d_model 涓€鑷达紝杈撳嚭 4 绫伙紙AAH/AIS/MIA/IAC锛?
    ordinal_bins = num_grades - 1
    grade_head = torch.nn.Linear(config.get("bridge_d_model", 2560), ordinal_bins, device=device)
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
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scaler = _build_grad_scaler(device)

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
    llm_model.eval()
    if hasattr(llm_model, "gradient_checkpointing_enable"):
        llm_model.gradient_checkpointing_enable()
    if hasattr(llm_model, "config"):
        llm_model.config.use_cache = False
    for p in llm_model.parameters():
        p.requires_grad = False
    llm_device = next(llm_model.parameters()).device
    embed = llm_model.get_input_embeddings()
    d_model = llm_model.config.hidden_size
    # 纭繚鍒嗙骇澶翠笌 LLM/瑙嗚鍦ㄥ悓涓€璁惧涓婏紝閬垮厤 device mismatch
    if grade_head is not None:
        grade_head = grade_head.to(llm_device)
    max_visual_tokens = getattr(args, "max_visual_tokens", 144)
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
        "mamba_model": args.mamba_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_text_len": max_text_len,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "classifier_mode": "ordinal_bce",
        "ordinal_bins": ordinal_bins,
    }
    (out_dir / "stage2_config.json").write_text(json.dumps(stage2_config, indent=2), encoding="utf-8")

    # 鏃ュ織
    log_csv = out_dir / "stage2_train_log.csv"
    heartbeat_path = out_dir / "stage2_heartbeat.txt"
    try:
        log_fh = open(log_csv, "w", encoding="utf-8")
        log_fh.write("step,epoch,caption_loss,cls_loss\n")
        log_fh.flush()
    except OSError as e:
        log_fh = None
        print(f"璀﹀憡: 鏃犳硶鍐欐棩蹇?{log_csv}: {e}", flush=True)

    use_amp = device.type == "cuda" and scaler is not None
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
            if use_amp:
                with _autocast_ctx(device):
                    loss, cap_loss_val, cls_loss_val = compute_batch_loss(
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
                    )
                loss = loss / accum_steps
                scaler.scale(loss).backward()
            else:
                loss, cap_loss_val, cls_loss_val = compute_batch_loss(
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
                )
                loss = loss / accum_steps
                loss.backward()

            loss_val = loss.item() * accum_steps
            if not first_batch_done:
                first_batch_done = True
                if loss_val < 0.1:
                    print(f"璀﹀憡: 棣栨 loss={loss_val:.4f} 寮傚父浣庯紝璇锋鏌?label 鏄惁鍙鍥炵瓟閮ㄥ垎", flush=True)
                elif loss_val > 15.0:
                    print(f"note: first-step loss={loss_val:.4f} is relatively high", flush=True)

            epoch_losses.append(loss_val)
            global_step += 1

            if global_step % accum_steps == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % args.log_every_steps == 0:
                print(
                    f"epoch {epoch+1} step {global_step} total_loss {loss_val:.4f} "
                    f"(caption={cap_loss_val:.4f}, cls={cls_loss_val:.4f})",
                    flush=True,
                )
                if log_fh is not None:
                    try:
                        log_fh.write(f"{global_step},{epoch+1},{cap_loss_val:.6f},{cls_loss_val:.6f}\n")
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
                        "num_grades": num_grades,
                        "classifier_mode": "ordinal_bce",
                        "ordinal_bins": ordinal_bins,
                    },
                    ckpt_path,
                )
                print(f"  [step {global_step}] saved {ckpt_path} (with grade_head)", flush=True)

        # epoch 鏈熬鑻ユ湁鏈?step 鐨勭疮绉搴︼紝琛ヤ竴娆?
        if accum_steps > 1 and global_step % accum_steps != 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        epoch_avgs.append(avg_loss)
        ckpt_path = out_dir / "vision_bridge_vlm_final.pt"
        torch.save(
            {
                "step": global_step,
                "model_state_dict": vision_bridge.state_dict(),
                "grade_head_state_dict": grade_head.state_dict(),
                "num_grades": num_grades,
                        "classifier_mode": "ordinal_bce",
                        "ordinal_bins": ordinal_bins,
            },
            ckpt_path,
        )
        print(f"epoch {epoch+1} done, avg caption_loss {avg_loss:.4f}, saved {ckpt_path} (with grade_head)", flush=True)

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



