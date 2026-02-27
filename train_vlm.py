"""
VLM 图像→报告训练：仅训练 Vision+Bridge，Mamba 冻结。
输入：问题+报告；Loss 只对「报告」部分计算，与推理时「问题+换行」后生成报告一致。

用法:
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
from torch.amp import autocast, GradScaler

from data.medical_vlm_dataset import MedicalVLMDataset, load_paths_config
from model.forward_medical_vlm import build_medical_vlm_from_config
from inference import _pool_visual_tokens

# 默认文本长度；医学报告较长，过小会截断。
DEFAULT_MAX_TEXT_LEN = 512


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
    单 batch 前向 + caption loss。
    Loss 仅对「回答」部分计算（视觉+问题+换行 均为 -100）。
    """
    images = batch["image"].to(device)
    if images.dim() == 3:
        images = images.unsqueeze(1)
    questions = batch["question"]
    answers = batch["answer"]
    B = images.shape[0]

    # 视觉编码 + 池化（可选梯度检查点省显存）
    if use_gradient_checkpointing and vision_bridge.training:
        vis_tokens = torch.utils.checkpoint.checkpoint(vision_bridge, images, use_reentrant=False)
    else:
        vis_tokens = vision_bridge(images)
    # VimBridge 内部会将 queries 输出缓存到 latest_queries_out，供分级任务使用
    bridge = getattr(vision_bridge, "bridge", None)
    queries_out = getattr(bridge, "latest_queries_out", None) if bridge is not None else None

    vis = _pool_visual_tokens(vis_tokens, max_visual_tokens)
    L_vis = vis.shape[1]

    # 文本：与推理一致格式为 "问题\n回答"
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

    # 仅对「回答」算 loss；prompt 长度 = "问题\n" 的 token 数（与推理一致）
    q_texts = [f"{q}\n" for q in questions]
    q_enc = tokenizer(
        q_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_text_len,
    )
    q_lens = q_enc["attention_mask"].sum(dim=1).tolist()

    # Embedding（LLM 嵌入维须与 bridge_d_model 一致，否则需投影或 pad/trim）
    text_emb = embed(input_ids.to(llm_device))
    E = text_emb.shape[-1]
    if E != d_model:
        if E < d_model:
            pad = torch.zeros(B, text_emb.shape[1], d_model - E, device=text_emb.device, dtype=text_emb.dtype)
            text_emb = torch.cat([text_emb, pad], dim=-1)
        else:
            text_emb = text_emb[:, :, :d_model]
    vis = vis.to(llm_device)

    # 可选 CMI
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

    # Labels：仅回答部分有效，其余 -100
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

    # 前向 + loss（logits[t] 预测 position t+1）
    llm_model.eval()
    out = llm_model(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
    logits = out.logits[:, : seq_len - 1]
    shift_labels = labels[:, 1:]
    caption_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )
    # 侵润/分级损失（可选）
    cls_loss = torch.tensor(0.0, device=llm_device)
    grades_raw = batch.get("grade", None)
    if grade_head is not None and grades_raw is not None:
        grades = torch.as_tensor(grades_raw, device=llm_device)
        valid_mask = grades >= 0
        if valid_mask.any():
            if queries_out is not None:
                cls_repr = queries_out.to(llm_device).mean(dim=1)  # [B, D]
            else:
                # 回退：用未池化的视觉 token 均值作为表示
                cls_repr = vis_tokens.to(llm_device).mean(dim=1)
            cls_input = cls_repr[valid_mask]
            cls_labels = grades[valid_mask].to(llm_device).long()
            cls_logits = grade_head(cls_input)
            if class_weights is not None:
                w = class_weights.to(cls_logits.device)
                cls_loss = F.cross_entropy(cls_logits, cls_labels, weight=w)
            else:
                cls_loss = F.cross_entropy(cls_logits, cls_labels)
        else:
            # 用 dummy loss 保证计算图不断裂
            cls_loss = grade_head.weight.sum() * 0.0

    total_loss = caption_loss + lambda_cls * cls_loss
    return total_loss, float(caption_loss.detach().cpu()), float(cls_loss.detach().cpu())


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-End VLM 训练：解冻 Vision+Bridge，Mamba 冻结")
    parser.add_argument("--epochs", type=int, default=30, help="医学报告逻辑复杂，可增至 40 减轻欠拟合")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率；端到端解冻时建议 1e-5")
    parser.add_argument("--max_visual_tokens", type=int, default=144, help="视觉 token 上限，建议与池化配置匹配（如 12x12=144）")
    parser.add_argument("--max_text_len", type=int, default=DEFAULT_MAX_TEXT_LEN, help="问题+报告总 token 上限")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--save_every_steps", type=int, default=0, help="每 N 步额外存一次权重，0 表示仅每 epoch 结束存")
    parser.add_argument("--log_every_steps", type=int, default=1, help="每 N 步打印一次 loss，默认 1（每步打印）")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader 线程数")
    parser.add_argument("--vision_checkpoint", type=str, default=None, help="可选：从已有 Vision+Bridge 权重继续训练")
    parser.add_argument("--mamba_model", type=str, default="state-spaces/mamba-2.8b-hf")
    parser.add_argument("--llm_8bit", action="store_true", help="LLM 8-bit 加载（需 bitsandbytes）")
    parser.add_argument("--align_vocab", action="store_true", help="训练端对齐 tokenizer 与 embedding 词表大小（推荐开启）")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", dest="gradient_checkpointing", action="store_true", help="启用梯度检查点，省显存、略降速（默认开启）")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false", help="关闭梯度检查点")
    parser.add_argument("--csv", type=str, default=None, help="直接指定 caption CSV，有值时优先于 paths.yaml")
    parser.add_argument("--length_audit_samples", type=int, default=128, help="训练前抽样统计 token 长度；0 表示关闭")
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
        print("训练 CSV 不存在:", csv_train)
        return 1

    train_ds = MedicalVLMDataset(csv_train, prompt_json_file=config.get("caption_prompt_json"))
    # 侵润/分级类别数（AAH/AIS/MIA/IAC）
    num_grades = 4
    # 根据数据分布估计类别权重，缓解类别不平衡；若无 grade 列则为 None
    class_weights = None
    if hasattr(train_ds, "grades"):
        counts = torch.zeros(num_grades, dtype=torch.long)
        for g in getattr(train_ds, "grades", []):
            try:
                gi = int(g)
            except Exception:
                continue
            if 0 <= gi < num_grades:
                counts[gi] += 1
        if counts.sum() > 0:
            counts_f = counts.float()
            # 避免 0，使用非零均值替代
            if (counts_f == 0).any() and (counts_f > 0).any():
                nz_mean = counts_f[counts_f > 0].mean()
                counts_f[counts_f == 0] = nz_mean
            inv = counts_f.sum() / (counts_f + 1e-6)
            class_weights = (inv / inv.mean()).clone()

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
            print(f"指定的 --vision_checkpoint 不存在: {ckpt}", flush=True)
            return 1
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        sd = state.get("model_state_dict", state)
        vision_bridge.load_state_dict(sd, strict=False)
        step_info = f" (从 step {state.get('step', '?')} 续训)" if isinstance(state, dict) and state.get("step") else ""
        print(f"Loaded Vision+Bridge: {ckpt}{step_info}", flush=True)
    else:
        print("未指定 --vision_checkpoint：Vision Encoder + Bridge 将随机初始化并端到端训练。", flush=True)
    vision_bridge = vision_bridge.to(device)
    # 分级头：输入维度与 bridge_d_model 一致，输出 4 类（AAH/AIS/MIA/IAC）
    grade_head = torch.nn.Linear(config.get("bridge_d_model", 2560), num_grades, device=device)
    # End-to-end: unfreeze encoder and bridge together.
    if hasattr(vision_bridge, "encoder"):
        for p in vision_bridge.encoder.parameters():
            p.requires_grad = True
        print("已解冻 nnunet_encoder，端到端训练 Vision+Bridge", flush=True)
        if getattr(args, "gradient_checkpointing", False):
            if _enable_encoder_gradient_checkpointing(vision_bridge.encoder):
                print("已启用 encoder gradient checkpointing", flush=True)
            else:
                print("encoder 未提供原生 gradient checkpointing 接口，继续使用外层 checkpoint 包裹前向。", flush=True)
    trainable = [p for p in vision_bridge.parameters() if p.requires_grad]
    trainable.extend(list(grade_head.parameters()))
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scaler = GradScaler("cuda") if device.type == "cuda" else None

    # Mamba 冻结
    print("加载 Mamba（冻结）...", flush=True)
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
    # 确保分级头与 LLM/视觉在同一设备上，避免 device mismatch
    grade_head = grade_head.to(llm_device)
    max_visual_tokens = getattr(args, "max_visual_tokens", 144)
    max_text_len = getattr(args, "max_text_len", DEFAULT_MAX_TEXT_LEN)
    _summarize_text_token_lengths(
        train_ds,
        tokenizer,
        max_text_len=max_text_len,
        max_samples=getattr(args, "length_audit_samples", 0),
    )

    print(f"max_visual_tokens={max_visual_tokens}（推理时须一致）", flush=True)
    print(f"max_text_len={max_text_len}, gradient_accumulation_steps={args.gradient_accumulation_steps}", flush=True)
    if device.type == "cuda":
        try:
            alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            print(
                f"当前 GPU 显存: 已分配 {alloc_gb:.2f} GB, 已预留 {reserved_gb:.2f} GB",
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
    }
    (out_dir / "stage2_config.json").write_text(json.dumps(stage2_config, indent=2), encoding="utf-8")

    # 日志
    log_csv = out_dir / "stage2_train_log.csv"
    heartbeat_path = out_dir / "stage2_heartbeat.txt"
    try:
        log_fh = open(log_csv, "w", encoding="utf-8")
        log_fh.write("step,epoch,caption_loss,cls_loss\n")
        log_fh.flush()
    except OSError as e:
        log_fh = None
        print(f"警告: 无法写日志 {log_csv}: {e}", flush=True)

    use_amp = device.type == "cuda" and scaler is not None
    use_grad_ckpt = getattr(args, "gradient_checkpointing", False)
    if use_grad_ckpt:
        print("已启用 gradient_checkpointing（省显存）", flush=True)
    global_step = 0
    epoch_avgs = []
    first_batch_done = False
    accum_steps = max(1, args.gradient_accumulation_steps)

    # 首 batch 前：确认 Loss 只对「回答」部分计算（便于排查幻觉/背诵问题）
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
        print(f"[debug] 样本0: prompt(问题+换行) token 数={prompt_len}, answer token 数={answer_len}, full token 数={full_len}; Loss 仅对 answer 部分计算", flush=True)
        # 打印训练文本格式，确认是 "question\\nanswer"。
        print(f"[debug] 样本0训练文本预览: {sample_full[:280].replace(chr(10), ' | ')}", flush=True)
        # 解码 full/answer 前若干 token，直观看模型在学什么格式。
        full_ids = full_enc["input_ids"][0]
        n_full = min(100, full_ids.size(0) if hasattr(full_ids, "size") else len(full_ids))
        full_decode = tokenizer.decode(
            full_ids[:n_full].tolist() if hasattr(full_ids, "tolist") else list(full_ids[:n_full]),
            skip_special_tokens=True,
        )
        print(f"[debug] 样本0 full 前约100 token 解码: {full_decode[:240]}...", flush=True)
        answer_ids = a_enc["input_ids"][0]
        n_show = min(60, answer_ids.size(0) if hasattr(answer_ids, "size") else len(answer_ids))
        head_decode = tokenizer.decode(
            answer_ids[:n_show].tolist() if hasattr(answer_ids, "tolist") else list(answer_ids[:n_show]),
            skip_special_tokens=True,
        )
        print(f"[debug] 样本0 answer 部分前约60 token 解码: {head_decode[:200]}...", flush=True)
    except Exception as e:
        print(f"[debug] 首 batch 检查跳过: {e}", flush=True)

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
                with autocast("cuda", dtype=torch.float16):
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
                        grade_head=grade_head,
                        class_weights=class_weights,
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
                    grade_head=grade_head,
                    class_weights=class_weights,
                )
                loss = loss / accum_steps
                loss.backward()

            loss_val = loss.item() * accum_steps
            if not first_batch_done:
                first_batch_done = True
                if loss_val < 0.1:
                    print(f"警告: 首步 loss={loss_val:.4f} 异常低，请检查 label 是否只对回答部分", flush=True)
                elif loss_val > 15.0:
                    print(f"提示: 首步 loss={loss_val:.4f} 较高属正常", flush=True)

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
                    },
                    ckpt_path,
                )
                print(f"  [step {global_step}] 已保存 {ckpt_path}（含 grade_head）", flush=True)

        # epoch 末尾若有未 step 的累积梯度，补一次
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
            },
            ckpt_path,
        )
        print(f"epoch {epoch+1} 结束, 平均 caption_loss {avg_loss:.4f}, 保存 {ckpt_path}（含 grade_head）", flush=True)

    if log_fh is not None:
        try:
            log_fh.close()
        except OSError:
            pass
    print("VLM 训练完成", flush=True)
    if epoch_avgs:
        recent = epoch_avgs[-5:] if len(epoch_avgs) >= 5 else epoch_avgs
        print(f"最近 epoch 平均 loss: {[f'{x:.4f}' for x in recent]}", flush=True)
        print(f"Loss 曲线: {log_csv}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

