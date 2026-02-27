"""
图像 → 文本生成：加载训练好的 Vision+Bridge 与 Mamba LLM，从 CT 图像生成报告。

用法:
  python inference.py --image D:/nnunet_raw/Dataset503_.../imagesTr/xxx.nii.gz
  python inference.py --val_sample   # 从验证集抽几条跑生成并打印
  python inference.py --checkpoint outputs/vision_bridge_best_val.pt --image ...

若生成结果为英文/乱码：请确保已跑完 Stage 2 图文对齐训练（run_full_train.ps1 或 train_vlm.py），
且 caption_loss 明显下降；仅 Stage 1 时模型未学过「见图写中文报告」。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
import numpy as np

from data.medical_vlm_dataset import load_paths_config
from vision.nodule_contour import (
    generate_nodule_contour_outputs,
    load_slice_with_optional_mask,
)

try:
    from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
except Exception:
    StoppingCriteria, StoppingCriteriaList = None, None

try:
    from transformers.generation import LogitsProcessor, LogitsProcessorList
except Exception:
    LogitsProcessor, LogitsProcessorList = None, None


class _VocabSizeMaskProcessor(LogitsProcessor if LogitsProcessor else object):
    """当 config.vocab_size > len(tokenizer) 时，屏蔽超出 tokenizer 的 logits，避免生成 50277/50278/50279 等导致解码成乱码（如「外」）"""

    def __init__(self, vocab_len: int, model_vocab_size: int):
        self.vocab_len = vocab_len
        self.model_vocab_size = model_vocab_size

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if LogitsProcessor is None or self.model_vocab_size <= self.vocab_len:
            return scores
        if scores.shape[-1] > self.vocab_len:
            scores[..., self.vocab_len:] = -float("inf")
        return scores


class _SuppressEOSAtBegin(LogitsProcessor if LogitsProcessor else object):
    """前 N 步禁止 EOS，迫使模型先输出正文，避免「原始输出为空」"""

    def __init__(self, input_len: int, eos_token_id: int, suppress_steps: int = 64):
        self.input_len = input_len
        self.eos_token_id = eos_token_id
        self.suppress_steps = suppress_steps

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if LogitsProcessor is None:
            return scores
        step = input_ids.shape[1] - self.input_len
        if step < self.suppress_steps and self.eos_token_id is not None:
            scores[:, self.eos_token_id] = -float("inf")
        return scores


class _NumberListStoppingCriteria(StoppingCriteria if StoppingCriteria else object):
    """生成时若解码结果结尾已是「数字, 数字, 数字」则提前停止，避免整段 45,46,47..."""

    def __init__(self, input_len: int, tokenizer, min_len: int = 20):
        self.input_len = input_len
        self.tokenizer = tokenizer
        self.min_len = min_len

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> bool:
        if StoppingCriteria is None:
            return False
        new_len = input_ids.shape[1] - self.input_len
        if new_len < self.min_len:
            return False
        if new_len % 8 != 0 and new_len < self.min_len + 50:
            return False
        new_part = input_ids[0][self.input_len:]
        text = self.tokenizer.decode(new_part, skip_special_tokens=True)
        if len(text.strip()) < self.min_len:
            return False
        return bool(_NUMBER_LIST_TAIL.search(text.strip()))

# 无占位符的短 prompt，避免模型重复「建议：<随访或检查建议>」等模板内容而非生成真实报告
PROMPT_SHORT_NO_PLACEHOLDERS = (
    "请根据胸部CT生成报告，按四段输出：所见、结论、建议、病理倾向。\n"
)

# 生成后截断：遇到广告/电话等黑名单内容即丢弃后续，减少幻觉
_BAD_PATTERNS = [
    r"请拨打电话", r"联系客服", r"客服电话", r"微信联系", r"联系电话",
    r"\+\s*\d{2}\s*[-]?\s*\d+", r"\d{11}", r"QQ\s*\d+",
    r"悲剧情况", r"敬请关注", r"如需更多",
]
# 纯数字列表幻觉（如 45, 46, 47, ...）：整段丢弃或截断
_NUMBER_LIST_PATTERN = re.compile(r"\d+\s*,\s*\d")
# 生成时若解码结尾已是「数字, 数字, 数字」则提前停止，避免整段都是数字序列
_NUMBER_LIST_TAIL = re.compile(r"\d+\s*,\s*\d+\s*,\s*\d+\s*$")
# 模板占位符行（仅含 X：<...> 或 <...>），模型重复 prompt 时产生，需过滤
_PLACEHOLDER_LINE = re.compile(
    r"^(?:\s*(?:所见|结论|建议|病理倾向|诊断)[:：]\s*)?"
    r"<[^>]*(?:所见|结论|建议|病理|随访|检查|炎性|肿瘤|待定|征象|大小|定位|条结)[^>]*>\s*$"
)
# 整行为单一 <...> 的也视为占位符
_ONLY_ANGLE_BRACKET = re.compile(r"^\s*<[^>]+>\s*$")
# <...> 内若含真实报告特征（肺、结节、mm、IM 等）则保留该行
_REAL_CONTENT_IN_BRACKETS = re.compile(r"<[^>]*(?:肺|结节|mm|IM\d|胸廓|纵隔)[^>]*>")

# 四级分级标签（顺序需与训练时一致）
GRADE_LABELS = ["AAH", "AIS", "MIA", "IAC"]

def _drop_placeholder_lines(text: str) -> str:
    """去掉仅含模板占位符的行（如「建议：<随访或检查建议>」），保留真实报告内容。"""
    if not text or not text.strip():
        return text
    lines = text.split("\n")
    kept = []
    for line in lines:
        s = line.strip()
        if not s:
            kept.append(line)
            continue
        if _ONLY_ANGLE_BRACKET.match(s):
            continue
        if _REAL_CONTENT_IN_BRACKETS.search(s):
            kept.append(line)
            continue
        if _PLACEHOLDER_LINE.match(s):
            continue
        kept.append(line)
    return "\n".join(kept).strip()

def clean_generated(text: str) -> str:
    if not text or not text.strip():
        return text
    text = _drop_placeholder_lines(text)
    for pat in _BAD_PATTERNS:
        m = re.search(pat, text)
        if m:
            return text[: m.start()].strip()
    # 若开头就是“数字, 数字”则整段视为幻觉
    if _NUMBER_LIST_PATTERN.match(text.strip()):
        return ""
    # 若中间出现数字列表，只保留其前的有效内容
    m = _NUMBER_LIST_PATTERN.search(text)
    if m:
        before = text[: m.start()].strip()
        if len(before) > 2 and re.search(r"[一-龥\u4e00-\u9fff]", before):
            return before
        return ""
    return text.strip()


TEMPLATE_HEADERS = ("所见：", "结论：", "建议：", "病理倾向：")


def _template_complete(text: str) -> bool:
    return all(h in text for h in TEMPLATE_HEADERS)


def _build_template_force_words_ids(tokenizer) -> list[list[int]]:
    ids: list[list[int]] = []
    for phrase in TEMPLATE_HEADERS:
        toks = tokenizer(phrase, add_special_tokens=False)["input_ids"]
        if toks:
            ids.append(toks)
    return ids


def _normalize_template_output(text: str) -> str:
    """
    Ensure output has four sections. If model misses some headers, reuse the main body as fallback.
    """
    txt = (text or "").strip()
    if not txt:
        return (
            "所见：胸部CT可见异常，具体定位与征象需结合原始影像复核。\n"
            "结论：肺部病灶性质待定。\n"
            "建议：建议短期复查胸部CT并结合临床。\n"
            "病理倾向：炎性或肿瘤性待定。"
        )
    if _template_complete(txt):
        return txt

    section_map = {h: "" for h in TEMPLATE_HEADERS}
    current = None
    for raw_line in txt.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        hit = None
        for h in TEMPLATE_HEADERS:
            if line.startswith(h):
                hit = h
                section_map[h] = line[len(h):].strip()
                current = h
                break
        if hit is None and current is not None:
            section_map[current] = (section_map[current] + " " + line).strip()

    body = re.sub(r"\s+", " ", txt)
    for h in TEMPLATE_HEADERS:
        if not section_map[h]:
            section_map[h] = body

    return (
        f"所见：{section_map['所见：']}\n"
        f"结论：{section_map['结论：']}\n"
        f"建议：{section_map['建议：']}\n"
        f"病理倾向：{section_map['病理倾向：']}"
    )


# 与 dataset 一致的 2D 加载
def _load_nifti_slice(
    path: str,
    slice_axis: int = 0,
    slice_idx: int | None = None,
    mask_path: str | None = None,
) -> np.ndarray:
    path = str(path).strip()
    if path in ("...", ".", "..") or not (path.endswith(".nii.gz") or path.endswith(".nii")):
        raise ValueError(
            f'无效的图像路径 "{path}"。请传入真实的 NIfTI 文件路径（.nii 或 .nii.gz），'
            '或省略 --image 从验证集取图。'
        )
    return load_slice_with_optional_mask(
        path,
        mask_path=mask_path,
        slice_axis=slice_axis,
        slice_idx=slice_idx,
    )

def _resize_to_patch(arr: np.ndarray, patch_size: int = 512) -> torch.Tensor:
    if arr.shape[0] != patch_size or arr.shape[1] != patch_size:
        t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
        t = torch.nn.functional.interpolate(t, size=(patch_size, patch_size), mode="bilinear", align_corners=False)
        arr = t.squeeze().numpy()
    mn, mx = arr.min(), arr.max()
    if mx - mn > 1e-8:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr)
    return torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

def load_image_tensor(
    image_path: str,
    patch_size: int = 512,
    mask_path: str | None = None,
) -> torch.Tensor:
    arr = _load_nifti_slice(image_path, mask_path=mask_path)
    return _resize_to_patch(arr, patch_size)


def infer_grade_from_queries(vision_bridge: torch.nn.Module) -> dict | None:
    """
    使用 VimBridge 中缓存的 latest_queries_out 与 grade_head 进行四级分级推断。
    要求:
      - vision_bridge.bridge.latest_queries_out 已由最近一次前向更新
      - vision_bridge.grade_head 存在且已加载训练好的权重
    返回:
      dict(label=..., index=..., probs=[...]) 或 None
    """
    bridge = getattr(vision_bridge, "bridge", None)
    grade_head = getattr(vision_bridge, "grade_head", None)
    queries = getattr(bridge, "latest_queries_out", None) if bridge is not None else None
    if grade_head is None or queries is None:
        return None
    if queries.ndim != 3 or queries.shape[0] == 0:
        return None
    q = queries.mean(dim=1)  # [B, D]
    device = next(grade_head.parameters()).device
    q = q.to(device)
    with torch.inference_mode():
        logits = grade_head(q)  # [B, 4]
        probs = torch.softmax(logits, dim=-1)[0].tolist()
        idx = int(logits.argmax(dim=-1)[0])
    label = GRADE_LABELS[idx] if 0 <= idx < len(GRADE_LABELS) else str(idx)
    return {"label": label, "index": idx, "probs": probs}


def load_vision_bridge(checkpoint_path: str | Path, config: dict, device: torch.device):
    from model.forward_medical_vlm import build_medical_vlm_from_config
    model = build_medical_vlm_from_config(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    # 若模型含 CMI 等可选模块而 checkpoint 为旧版未保存，则非严格加载以兼容
    # Always allow extra keys (e.g., cmi_connector) to keep inference compatible
    load_ok = model.load_state_dict(state, strict=False)
    if hasattr(load_ok, "missing_keys") and load_ok.missing_keys:
        print(
            "注意: checkpoint 中缺少部分参数（如 cmi_connector），已忽略；缺失:",
            load_ok.missing_keys[:5],
            "...",
        )

    # 可选：加载四级分级头（AAH/AIS/MIA/IAC）
    if isinstance(ckpt, dict) and "grade_head_state_dict" in ckpt:
        try:
            num_grades = int(ckpt.get("num_grades", 4))
        except Exception:
            num_grades = 4
        d_model = int(config.get("bridge_d_model", 2560))
        grade_head = torch.nn.Linear(d_model, num_grades)
        grade_head.load_state_dict(ckpt["grade_head_state_dict"])
        model.grade_head = grade_head.to(device)
        print("已从 checkpoint 恢复 grade_head，用于 AAH/AIS/MIA/IAC 分级。", flush=True)
    else:
        # 兼容旧版 checkpoint：无分级头时直接跳过
        model.grade_head = None  # type: ignore[attr-defined]

    return model.to(device).eval()


def _pool_visual_tokens(visual_tokens: torch.Tensor, max_tokens: int) -> torch.Tensor:
    """将 [1, L, D] 的视觉 token 池化到最多 max_tokens，减轻显存（避免 OOM）。"""
    B, L, D = visual_tokens.shape
    if L <= max_tokens:
        return visual_tokens
    # 假设 L = 28*28=784，池化到 14*14=196 或 7*7=49
    side = int(L ** 0.5)
    if side * side != L:
        return visual_tokens[:, :max_tokens]
    target_side = int(max_tokens ** 0.5)
    x = visual_tokens.view(B, side, side, D).permute(0, 3, 1, 2)  # [B, D, H, W]
    x = torch.nn.functional.adaptive_avg_pool2d(x, (target_side, target_side))
    x = x.permute(0, 2, 3, 1).reshape(B, -1, D)
    return x


def generate_from_image(
    image_tensor: torch.Tensor,
    vision_bridge: torch.nn.Module,
    llm_model,
    tokenizer,
    prompt: str = None,
    max_new_tokens: int = 512,
    device: torch.device = None,
    max_visual_tokens: int = 196,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    length_penalty: float = 1.1,
    no_repeat_ngram_size: int = 0,
    suppress_eos_steps: int = 128,
    num_beams: int = 1,
    force_words_ids: list[list[int]] | None = None,
    min_chars: int = 180,
    max_retries: int = 2,
    force_template: bool = True,
    raw_out: list | None = None,
    debug_vision: bool = False,
) -> str:
    """
    单张图像 + 提示 → 生成报告文本。
    默认使用无占位符短 prompt，避免模型重复「建议：<随访或检查建议>」等模板内容。
    image_tensor: [1, 1, 512, 512]
    max_visual_tokens: 视觉 token 上限；do_sample: 采样/贪心；debug_vision: 打印视觉统计。
    """
    if prompt is None:
        prompt = PROMPT_SHORT_NO_PLACEHOLDERS
    if not prompt.endswith("\n"):
        prompt = prompt + "\n"
    if device is None:
        device = next(vision_bridge.parameters()).device
    image_tensor = image_tensor.to(device)
    vision_bridge.eval()
    with torch.inference_mode():
        visual_tokens = vision_bridge(image_tensor)  # [1, L_vis, D]
    visual_tokens = _pool_visual_tokens(visual_tokens, max_visual_tokens)
    if debug_vision:
        v = visual_tokens.detach().float()
        has_nan = torch.isnan(v).any().item()
        mean, std = v.mean().item(), v.std().item()
        print(f"[debug_vision] visual_tokens shape={tuple(visual_tokens.shape)} min={v.min().item():.4f} max={v.max().item():.4f} mean={mean:.4f} std={std:.4f} has_nan={has_nan}", flush=True)
        if has_nan or (abs(mean) < 1e-6 and std < 1e-6):
            print("[debug_vision] 可能异常：接近全 0 或含 NaN，请检查图像预处理或 Vision Encoder 是否加载权重。", flush=True)
        elif std > 100 or abs(mean) > 100:
            print("[debug_vision] 可能异常：数值过大，疑为 Bridge 梯度爆炸或溢出。", flush=True)
    L_vis = visual_tokens.shape[1]
    d_model = visual_tokens.shape[2]

    # 提示 tokenize → embeddings（截断以减短序列，降低 GPU OOM）
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=128,
    )
    llm_device = next(llm_model.parameters()).device
    # 释放 GPU 显存：vision 已算完，可清缓存再送 LLM
    if device.type == "cuda":
        torch.cuda.empty_cache()
    visual_tokens = visual_tokens.to(llm_device)
    prompt_ids = enc["input_ids"].to(llm_device)
    embed_layer = llm_model.get_input_embeddings()
    prompt_embeds = embed_layer(prompt_ids)  # [1, L_prompt, D]
    if prompt_embeds.shape[-1] != d_model:
        prompt_embeds = torch.nn.Linear(
            prompt_embeds.shape[-1],
            d_model,
            device=llm_device,
            dtype=prompt_embeds.dtype,
        )(prompt_embeds)
    # 可选：CMI 机制（文本生成 SSM 参数，视觉流过 SSM 再融合），减轻握手失败/OOM
    cmi = getattr(vision_bridge, "cmi_connector", None)
    if cmi is not None:
        # 关键：保证 CMI 与输入都在 LLM 的 device 上（cpu 推理就全 cpu，避免 cpu/cuda 混用）
        if next(cmi.parameters()).device != llm_device:
            cmi = cmi.to(llm_device)
        with torch.inference_mode():
            visual_tokens = cmi(visual_tokens.to(llm_device), prompt_embeds.to(llm_device))
        L_vis = visual_tokens.shape[1]

    # 最终拼接前再次确保同 device
    visual_tokens = visual_tokens.to(llm_device)
    prompt_embeds = prompt_embeds.to(llm_device)
    inputs_embeds = torch.cat([visual_tokens, prompt_embeds], dim=1)  # [1, L_vis+L_prompt, D]

    # 使用 transformers.generate：只解码「新生成」部分，避免把视觉/提示占位也当文本
    input_len = L_vis + prompt_embeds.shape[1]
    attn_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long)
    # 医学报告需较长输出：max_new_tokens 建议 512，length_penalty>1 鼓励长句，no_repeat_ngram_size=0 避免误伤合理重复（如「双肺」「结节」）
    gen_kw = dict(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min(128, max_new_tokens),
        use_cache=True,
        num_beams=max(1, int(num_beams)),
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size if no_repeat_ngram_size > 0 else None,
    )
    # 剔除 None，避免 generate 报错
    gen_kw = {k: v for k, v in gen_kw.items() if v is not None}
    if force_words_ids:
        # Constrained decoding works best with beam search + deterministic decode.
        gen_kw["force_words_ids"] = force_words_ids
        gen_kw["num_beams"] = max(4, int(gen_kw.get("num_beams", 1)))
        gen_kw["do_sample"] = False
    elif do_sample:
        gen_kw["do_sample"] = True
        gen_kw["temperature"] = temperature
        gen_kw["top_p"] = top_p
    else:
        gen_kw["do_sample"] = False
    if StoppingCriteriaList is not None:
        gen_kw["stopping_criteria"] = StoppingCriteriaList([_NumberListStoppingCriteria(input_len, tokenizer)])
    processors = []
    vocab_len = len(tokenizer)
    model_vocab = getattr(llm_model.config, "vocab_size", None)
    if model_vocab is not None and model_vocab > vocab_len:
        processors.append(_VocabSizeMaskProcessor(vocab_len, model_vocab))
    if LogitsProcessorList is not None and tokenizer.eos_token_id is not None:
        processors.append(_SuppressEOSAtBegin(input_len, tokenizer.eos_token_id, suppress_steps=suppress_eos_steps))
    if processors:
        gen_kw["logits_processor"] = LogitsProcessorList(processors)
    gen_ids = llm_model.generate(**gen_kw)
    # 对 inputs_embeds 场景，部分模型返回「完整序列」，部分只返回「新生成序列」。
    # 若一律按 input_len 截断，可能把本就只含新 token 的输出切成空串。
    if gen_ids.shape[1] > input_len:
        new_ids = gen_ids[0][input_len:]
    else:
        new_ids = gen_ids[0]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True)
    if raw_out is not None:
        raw_out.append(raw)
    cleaned = clean_generated(raw)
    # 诊断：若过滤后为空但原始有内容，保留原始输出便于排查（如数字列表幻觉）
    if cleaned == "" and (raw and raw.strip()):
        return raw
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="医学 VLM 图像→文本生成")
    parser.add_argument("--image", type=str, default=None, help="NIfTI 图像路径")
    parser.add_argument("--mask", type=str, default=None, help="NIfTI mask 路径（用于病灶层面选择与结节勾画）")
    parser.add_argument("--draw_nodule_contour", action="store_true", help="输出结节勾画图和结节统计 CSV（需配合 --mask）")
    parser.add_argument("--nodule_output_subdir", type=str, default="nodule_contour", help="勾画结果输出到 run_dir 下的子目录名")
    parser.add_argument("--nodule_line_width", type=float, default=1.8, help="勾画轮廓线宽")
    parser.add_argument("--nodule_fill_alpha", type=float, default=0.22, help="勾画区域填充透明度")
    parser.add_argument("--val_sample", action="store_true", help="从验证集抽几条跑生成")
    parser.add_argument("--checkpoint", type=str, default=None, help="vision_bridge 权重，默认 outputs/vision_bridge_best_val.pt 或 vision_bridge_final.pt")
    parser.add_argument("--mamba_model", type=str, default="state-spaces/mamba-2.8b-hf", help="Mamba 预训练模型")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="生成 token 上限，医学报告建议 512 以减少截断")
    parser.add_argument("--max_visual_tokens", type=int, default=196, help="视觉 token 上限，省显存防 OOM，默认 196(14×14)")
    parser.add_argument("--num_beams", type=int, default=1, help="beam size；>1 会更稳但更慢，约束解码建议 4")
    parser.add_argument("--constrained_decode", action="store_true", help="约束解码：强制输出 所见/结论/建议/病理倾向 四段标题")
    parser.add_argument("--length_penalty", type=float, default=1.1, help="长度惩罚 >1 鼓励更长输出，默认 1.1")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="禁止重复 n-gram，0=关闭(推荐)，4 会误伤医学合理重复")
    parser.add_argument("--suppress_eos_steps", type=int, default=128, help="前 N 步禁止 EOS，避免过早结束，默认 128")
    parser.add_argument("--llm_device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Mamba 设备：auto 使用 GPU（推荐），cpu 防 OOM，cuda 强制 GPU")
    parser.add_argument("--num_val", type=int, default=3)
    parser.add_argument("--use_csv_prompt", action="store_true", help="验证时使用 CSV 中的问题作 prompt（含占位符）；默认用短 prompt 避免模型重复模板")
    parser.add_argument("--do_sample", action="store_true", help="启用采样生成（可能更发散，默认关闭）")
    parser.add_argument("--no_do_sample", action="store_true", help="禁用采样并使用贪心解码（默认）")
    parser.add_argument("--temperature", type=float, default=0.6, help="采样温度，仅 do_sample 时有效，默认 0.6")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="重复惩罚，减轻重复词，默认 1.2")
    parser.add_argument("--out_dir", type=str, default="D:/mamba-res", help="生成报告落盘目录，默认 D:/mamba-res")
    args = parser.parse_args()
    args.do_sample = bool(args.do_sample)
    if args.no_do_sample:
        args.do_sample = False

    config = load_paths_config(REPO / "config" / "paths.yaml")
    config.setdefault("encoder_output_stage", 4)
    config.setdefault("encoder_target_spatial", 28)
    config["bridge_d_model"] = 2560
    config.setdefault("nnunet_encoder_checkpoint", None)
    config.setdefault("use_cmi", False)
    config.setdefault("roi_side", None)
    config.setdefault("cmi_compress_to", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = REPO / "outputs"
    ckpt = args.checkpoint
    if ckpt is None:
        for name in ["vision_bridge_vlm_final.pt", "vision_bridge_best_val.pt", "vision_bridge_final.pt"]:
            p = out_dir / name
            if p.exists():
                ckpt = str(p)
                break
    if not ckpt or not Path(ckpt).exists():
        print("未找到 vision_bridge checkpoint，请先训练或指定 --checkpoint")
        return 1

    print("加载 Vision+Bridge...")
    vision_bridge = load_vision_bridge(ckpt, config, device)
    print("加载 Mamba LLM（可能较久）...")
    from llm.mamba_loader import load_mamba_lm
    llm_device_map = "cpu" if args.llm_device == "cpu" else args.llm_device  # "auto" 或 "cuda" 使用 GPU
    llm_model, tokenizer = load_mamba_lm(args.mamba_model, device_map=llm_device_map)
    # 与 train_vlm.py 一致：推理前再次确保 lm_head 绑定，避免乱码
    if hasattr(llm_model, "backbone") and hasattr(llm_model.backbone, "embeddings") and hasattr(llm_model, "lm_head"):
        llm_model.lm_head.weight = llm_model.backbone.embeddings.weight
        print("执行权重绑定 (Tying lm_head → backbone.embeddings)...", flush=True)
    llm_model.eval()
    force_words_ids = _build_template_force_words_ids(tokenizer) if getattr(args, "constrained_decode", False) else None

    res_dir = Path(args.out_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    run_dir = res_dir / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.val_sample:
        from data.medical_vlm_dataset import MedicalVLMDataset
        csv_val = config.get("caption_csv_val")
        if not csv_val or not Path(csv_val).exists():
            print("验证集 CSV 不存在，无法 --val_sample")
            return 1
        val_ds = MedicalVLMDataset(csv_val, prompt_json_file=config.get("caption_prompt_json"))
        num = min(args.num_val, len(val_ds))
        meta_list = []
        for idx in range(num):
            sample = val_ds[idx]
            image_t = sample["image"].unsqueeze(0)
            answer_gt = sample["answer"]
            if getattr(args, "use_csv_prompt", False):
                prompt = sample.get("question") or PROMPT_SHORT_NO_PLACEHOLDERS
            else:
                prompt = PROMPT_SHORT_NO_PLACEHOLDERS
            print(f"\n--- 验证样本 {idx+1}/{num} ---")
            print(f"Prompt: {prompt.strip()}")
            gen = generate_from_image(
                image_t, vision_bridge, llm_model, tokenizer, prompt=prompt,
                max_new_tokens=args.max_new_tokens, device=device, max_visual_tokens=args.max_visual_tokens,
                do_sample=args.do_sample, temperature=args.temperature, repetition_penalty=args.repetition_penalty,
                length_penalty=getattr(args, "length_penalty", 1.1),
                no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
                suppress_eos_steps=getattr(args, "suppress_eos_steps", 128),
                num_beams=getattr(args, "num_beams", 1),
                force_words_ids=force_words_ids,
            )
            print(f"生成: {gen[:500]}...")
            print(f"参考: {answer_gt[:200]}...")
            (run_dir / f"sample_{idx+1}_gen.txt").write_text(gen, encoding="utf-8")
            (run_dir / f"sample_{idx+1}_ref.txt").write_text(answer_gt, encoding="utf-8")
            sample_meta = {
                "idx": idx + 1,
                "image_path": sample.get("image_path"),
                "mask_path": sample.get("mask_path"),
                "prompt": prompt.strip(),
            }
            grade_info = infer_grade_from_queries(vision_bridge)
            if grade_info is not None:
                sample_meta["grade"] = grade_info
                print(f"[grade] sample {idx+1}: {grade_info['label']} probs={grade_info['probs']}")
            if args.draw_nodule_contour:
                sample_img = sample.get("image_path")
                sample_mask = sample.get("mask_path")
                if sample_img and sample_mask and Path(sample_img).exists() and Path(sample_mask).exists():
                    try:
                        contour_out = run_dir / f"sample_{idx+1}_{args.nodule_output_subdir}"
                        contour_info = generate_nodule_contour_outputs(
                            sample_img,
                            sample_mask,
                            contour_out,
                            line_width=args.nodule_line_width,
                            fill_alpha=args.nodule_fill_alpha,
                        )
                        sample_meta["nodule_contour"] = contour_info
                        print(f"[nodule] sample {idx+1} saved: {contour_info['overlay_png']}")
                    except Exception as e:
                        sample_meta["nodule_contour_error"] = str(e)
                        print(f"[nodule] sample {idx+1} contour failed: {e}")
                else:
                    sample_meta["nodule_contour_skipped"] = "missing image_path or mask_path"
            meta_list.append(sample_meta)
        (run_dir / "meta.json").write_text(json.dumps({"checkpoint": ckpt, "num_val": num, "samples": meta_list}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已落盘: {run_dir}")
        return 0

    if not args.image or not Path(args.image).exists():
        print("请指定 --image <NIfTI路径> 或使用 --val_sample")
        return 1
    if args.mask and not Path(args.mask).exists():
        print(f"mask file not found: {args.mask}")
        return 1
    image_t = load_image_tensor(args.image, mask_path=args.mask).to(device)
    text = generate_from_image(
        image_t, vision_bridge, llm_model, tokenizer,
        max_new_tokens=args.max_new_tokens, device=device, max_visual_tokens=args.max_visual_tokens,
        do_sample=args.do_sample, temperature=args.temperature, repetition_penalty=args.repetition_penalty,
        length_penalty=getattr(args, "length_penalty", 1.1),
        no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
        suppress_eos_steps=getattr(args, "suppress_eos_steps", 128),
        num_beams=getattr(args, "num_beams", 1),
        force_words_ids=force_words_ids,
    )
    print("生成报告:")
    print(text)
    (run_dir / "generated.txt").write_text(text, encoding="utf-8")
    run_meta = {"checkpoint": ckpt, "image_path": args.image, "mask_path": args.mask}
    grade_info = infer_grade_from_queries(vision_bridge)
    if grade_info is not None:
        run_meta["grade"] = grade_info
        print(f"[grade] {grade_info['label']} probs={grade_info['probs']}")
    if args.draw_nodule_contour:
        if not args.mask:
            run_meta["nodule_contour_skipped"] = "--draw_nodule_contour requires --mask"
            print("[nodule] skipped: please provide --mask with --draw_nodule_contour")
        else:
            try:
                contour_out = run_dir / args.nodule_output_subdir
                contour_info = generate_nodule_contour_outputs(
                    args.image,
                    args.mask,
                    contour_out,
                    line_width=args.nodule_line_width,
                    fill_alpha=args.nodule_fill_alpha,
                )
                run_meta["nodule_contour"] = contour_info
                print(f"[nodule] contour overlay: {contour_info['overlay_png']}")
                print(f"[nodule] stats csv: {contour_info['stats_csv']}")
                print(f"[nodule] nodule count: {contour_info['nodule_count']}")
            except Exception as e:
                run_meta["nodule_contour_error"] = str(e)
                print(f"[nodule] contour failed: {e}")
    (run_dir / "meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已落盘: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
