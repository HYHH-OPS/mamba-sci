"""
鍥惧儚 鈫?鏂囨湰鐢熸垚锛氬姞杞借缁冨ソ鐨?Vision+Bridge 涓?Mamba LLM锛屼粠 CT 鍥惧儚鐢熸垚鎶ュ憡銆?
鐢ㄦ硶:
  python inference.py --image D:/nnunet_raw/Dataset503_.../imagesTr/xxx.nii.gz
  python inference.py --val_sample   # 浠庨獙璇侀泦鎶藉嚑鏉¤窇鐢熸垚骞舵墦鍗?  python inference.py --checkpoint outputs/vision_bridge_best_val.pt --image ...

鑻ョ敓鎴愮粨鏋滀负鑻辨枃/涔辩爜锛氳纭繚宸茶窇瀹?Stage 2 鍥炬枃瀵归綈璁粌锛坮un_full_train.ps1 鎴?train_vlm.py锛夛紝
涓?caption_loss 鏄庢樉涓嬮檷锛涗粎 Stage 1 鏃舵ā鍨嬫湭瀛﹁繃銆岃鍥惧啓涓枃鎶ュ憡銆嶃€?"""

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
    """褰?config.vocab_size > len(tokenizer) 鏃讹紝灞忚斀瓒呭嚭 tokenizer 鐨?logits锛岄伩鍏嶇敓鎴?50277/50278/50279 绛夊鑷磋В鐮佹垚涔辩爜锛堝銆屽銆嶏級"""

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
    """Suppress EOS for first N decode steps to avoid empty output."""

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
    """鐢熸垚鏃惰嫢瑙ｇ爜缁撴灉缁撳熬宸叉槸銆屾暟瀛? 鏁板瓧, 鏁板瓧銆嶅垯鎻愬墠鍋滄锛岄伩鍏嶆暣娈?45,46,47..."""

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

# 鏃犲崰浣嶇鐨勭煭 prompt锛岄伩鍏嶆ā鍨嬮噸澶嶃€屽缓璁細<闅忚鎴栨鏌ュ缓璁?銆嶇瓑妯℃澘鍐呭鑰岄潪鐢熸垚鐪熷疄鎶ュ憡
PROMPT_SHORT_NO_PLACEHOLDERS = (
    "Please generate a chest CT report with 4 sections: Findings, Conclusion, Recommendation, Pathologic Tendency.\n"
)

# 鐢熸垚鍚庢埅鏂細閬囧埌骞垮憡/鐢佃瘽绛夐粦鍚嶅崟鍐呭鍗充涪寮冨悗缁紝鍑忓皯骞昏
_BAD_PATTERNS = [
    r"请拨打电话",
    r"联系客服",
    r"客服电话",
    r"微信联系",
    r"联系电话",
    r"\+\s*\d{2}\s*[-]?\s*\d+",
    r"\d{11}",
    r"QQ\s*\d+",
    r"敬请关注",
    r"更多信息",
]
_NUMBER_LIST_PATTERN = re.compile(r"\d+\s*,\s*\d")
_NUMBER_LIST_TAIL = re.compile(r"\d+\s*,\s*\d+\s*,\s*\d+\s*$")
_PLACEHOLDER_LINE = re.compile(
    r"^(?:\s*(?:所见|结论|建议|病理倾向|诊断)[:：]\s*)?<[^>]+>\s*$"
)
_ONLY_ANGLE_BRACKET = re.compile(r"^\s*<[^>]+>\s*$")
_REAL_CONTENT_IN_BRACKETS = re.compile(
    r"<[^>]*(?:肺|结节|mm|IM\d|胸膜|纵隔)[^>]*>",
    re.IGNORECASE,
)

# 鍥涚骇鍒嗙骇鏍囩锛堥『搴忛渶涓庤缁冩椂涓€鑷达級
GRADE_LABELS = ["AAH", "AIS", "MIA", "IAC"]


def _as_tuple3(v, default=(32, 128, 128)):
    if v is None:
        return tuple(default)
    if isinstance(v, str):
        parts = [p.strip() for p in v.split(",") if p.strip()]
        if len(parts) == 3:
            try:
                return (int(parts[0]), int(parts[1]), int(parts[2]))
            except Exception:
                return tuple(default)
        return tuple(default)
    try:
        vals = tuple(int(x) for x in v)
        if len(vals) == 3:
            return vals
    except Exception:
        pass
    return tuple(default)

def _drop_placeholder_lines(text: str) -> str:
    """Drop placeholder-only lines while keeping real report content."""
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
    # 鑻ュ紑澶村氨鏄€滄暟瀛? 鏁板瓧鈥濆垯鏁存瑙嗕负骞昏
    if _NUMBER_LIST_PATTERN.match(text.strip()):
        return ""
    # 鑻ヤ腑闂村嚭鐜版暟瀛楀垪琛紝鍙繚鐣欏叾鍓嶇殑鏈夋晥鍐呭
    m = _NUMBER_LIST_PATTERN.search(text)
    if m:
        before = text[: m.start()].strip()
        if len(before) > 2 and re.search(r"[涓€-榫u4e00-\u9fff]", before):
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
            "鎵€瑙侊細鑳搁儴CT鍙寮傚父锛屽叿浣撳畾浣嶄笌寰佽薄闇€缁撳悎鍘熷褰卞儚澶嶆牳銆俓n"
            "缁撹锛氳偤閮ㄧ梾鐏舵€ц川寰呭畾銆俓n"
            "寤鸿锛氬缓璁煭鏈熷鏌ヨ兏閮–T骞剁粨鍚堜复搴娿€俓n"
            "病理倾向：炎性或肿瘤性待定。\n"
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
        f"{TEMPLATE_HEADERS[0]}{section_map[TEMPLATE_HEADERS[0]]}\n"
        f"{TEMPLATE_HEADERS[1]}{section_map[TEMPLATE_HEADERS[1]]}\n"
        f"{TEMPLATE_HEADERS[2]}{section_map[TEMPLATE_HEADERS[2]]}\n"
        f"{TEMPLATE_HEADERS[3]}{section_map[TEMPLATE_HEADERS[3]]}"
    )


# 涓?dataset 涓€鑷寸殑 2D 鍔犺浇
def _load_nifti_slice(
    path: str,
    slice_axis: int = 0,
    slice_idx: int | None = None,
    mask_path: str | None = None,
) -> np.ndarray:
    path = str(path).strip()
    if path in ("...", ".", "..") or not (path.endswith(".nii.gz") or path.endswith(".nii")):
        raise ValueError(f"Invalid image path: {path}. Please pass a real .nii/.nii.gz file path.")
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
    使用 VimBridge 缓存的 latest_queries_out 与 grade_head 进行分级推断。
    兼容两种头：
      - 旧版 4 类 softmax 头
      - 新版 3 阈值序数头（>=AIS, >=MIA, >=IAC）
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
    if not torch.isfinite(q).all():
        q = torch.nan_to_num(q, nan=0.0, posinf=1e4, neginf=-1e4)
    with torch.inference_mode():
        logits = grade_head(q)
        n_out = int(logits.shape[-1])
        if n_out == len(GRADE_LABELS):
            probs = torch.softmax(logits, dim=-1)[0].tolist()
            idx = int(logits.argmax(dim=-1)[0])
            mode = "softmax_ce"
        elif n_out == len(GRADE_LABELS) - 1:
            # Ordinal decode: probs=sigmoid(logits), pred=(probs>0.5).sum(dim=1)
            thr = torch.sigmoid(logits)[0]
            idx = int((thr > 0.5).sum().item())
            idx = max(0, min(idx, len(GRADE_LABELS) - 1))
            ge0, ge1, ge2 = thr.tolist()
            probs = [1.0 - ge0, ge0 - ge1, ge1 - ge2, ge2]
            probs = [float(max(0.0, min(1.0, p))) for p in probs]
            s = sum(probs)
            if s > 0:
                probs = [p / s for p in probs]
            mode = "ordinal_bce"
        else:
            probs = torch.softmax(logits, dim=-1)[0].tolist()
            idx = int(logits.argmax(dim=-1)[0])
            mode = f"unknown_{n_out}"
    label = GRADE_LABELS[idx] if 0 <= idx < len(GRADE_LABELS) else str(idx)
    return {"label": label, "index": idx, "probs": probs, "mode": mode}


def load_vision_bridge(checkpoint_path: str | Path, config: dict, device: torch.device):
    from model.forward_medical_vlm import build_medical_vlm_from_config
    model = build_medical_vlm_from_config(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    # 若模型含 CMI 等可选模块而 checkpoint 为旧版未保存，则非严格加载以兼容。
    # Always allow extra keys (e.g., cmi_connector) to keep inference compatible
    load_ok = model.load_state_dict(state, strict=False)
    if hasattr(load_ok, "missing_keys") and load_ok.missing_keys:
        print(
            "注意: checkpoint 中缺少部分参数（如 cmi_connector），已忽略；缺失:",
            load_ok.missing_keys[:5],
            "...",
        )

    # 可选：加载分级头，兼容旧版4类和新版3阈值输出。
    if isinstance(ckpt, dict) and "grade_head_state_dict" in ckpt:
        grade_sd = ckpt["grade_head_state_dict"]
        d_model = int(config.get("bridge_d_model", 2560))
        out_dim = None
        if isinstance(grade_sd, dict):
            w = grade_sd.get("weight", None)
            if torch.is_tensor(w) and w.ndim == 2:
                out_dim = int(w.shape[0])
        if out_dim is None:
            try:
                out_dim = int(ckpt.get("num_grades", 4))
            except Exception:
                out_dim = 4
        grade_head = torch.nn.Linear(d_model, out_dim)
        grade_head.load_state_dict(grade_sd)
        model.grade_head = grade_head.to(device)
        classifier_mode = ckpt.get("classifier_mode", None)
        if not isinstance(classifier_mode, str) or not classifier_mode:
            classifier_mode = "ordinal_bce" if out_dim == len(GRADE_LABELS) - 1 else "softmax_ce"
        model.classifier_mode = classifier_mode  # type: ignore[attr-defined]
        model.ordinal_bins = int(ckpt.get("ordinal_bins", max(0, out_dim - 1)))  # type: ignore[attr-defined]
        print(f"已从 checkpoint 恢复 grade_head: out_dim={out_dim}, mode={classifier_mode}", flush=True)
    else:
        # 兼容旧版 checkpoint：无分级头时直接跳过
        model.grade_head = None  # type: ignore[attr-defined]

    return model.to(device).eval()


def _pool_visual_tokens(visual_tokens: torch.Tensor, max_tokens: int) -> torch.Tensor:
    """Pool visual tokens [B, L, D] down to max_tokens to reduce memory."""
    B, L, D = visual_tokens.shape
    if L <= max_tokens:
        return visual_tokens
    # 鍋囪 L = 28*28=784锛屾睜鍖栧埌 14*14=196 鎴?7*7=49
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
    roi_center: torch.Tensor | None = None,
    roi_center_3d: torch.Tensor | None = None,
) -> str:
    """Generate report text from one image tensor and prompt."""
    if prompt is None:
        prompt = PROMPT_SHORT_NO_PLACEHOLDERS
    if not prompt.endswith("\n"):
        prompt = prompt + "\n"
    if device is None:
        device = next(vision_bridge.parameters()).device
    image_tensor = image_tensor.to(device)
    if roi_center is not None:
        roi_center = torch.as_tensor(roi_center, device=device, dtype=torch.float32)
    if roi_center_3d is not None:
        roi_center_3d = torch.as_tensor(roi_center_3d, device=device, dtype=torch.float32)
    vision_bridge.eval()
    with torch.inference_mode():
        visual_tokens = vision_bridge(image_tensor, roi_center=roi_center, roi_center_3d=roi_center_3d)  # [1, L_vis, D]
    visual_tokens = _pool_visual_tokens(visual_tokens, max_visual_tokens)
    if debug_vision:
        v = visual_tokens.detach().float()
        has_nan = torch.isnan(v).any().item()
        mean, std = v.mean().item(), v.std().item()
        print(f"[debug_vision] visual_tokens shape={tuple(visual_tokens.shape)} min={v.min().item():.4f} max={v.max().item():.4f} mean={mean:.4f} std={std:.4f} has_nan={has_nan}", flush=True)
        if has_nan or (abs(mean) < 1e-6 and std < 1e-6):
            print("[debug_vision] possible issue: near-zero tokens or NaN detected.", flush=True)
        elif std > 100 or abs(mean) > 100:
            print("[debug_vision] possible issue: unusually large activation values.", flush=True)
    L_vis = visual_tokens.shape[1]
    d_model = visual_tokens.shape[2]

    # Tokenize prompt and build text embeddings.
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=128,
    )
    llm_device = next(llm_model.parameters()).device
    # 閲婃斁 GPU 鏄惧瓨锛歷ision 宸茬畻瀹岋紝鍙竻缂撳瓨鍐嶉€?LLM
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
    # 鍙€夛細CMI 鏈哄埗锛堟枃鏈敓鎴?SSM 鍙傛暟锛岃瑙夋祦杩?SSM 鍐嶈瀺鍚堬級锛屽噺杞绘彙鎵嬪け璐?OOM
    cmi = getattr(vision_bridge, "cmi_connector", None)
    if cmi is not None:
        # Keep CMI on the same device as LLM inputs.
        if next(cmi.parameters()).device != llm_device:
            cmi = cmi.to(llm_device)
        with torch.inference_mode():
            visual_tokens = cmi(visual_tokens.to(llm_device), prompt_embeds.to(llm_device))
        L_vis = visual_tokens.shape[1]

    # 鏈€缁堟嫾鎺ュ墠鍐嶆纭繚鍚?device
    visual_tokens = visual_tokens.to(llm_device)
    prompt_embeds = prompt_embeds.to(llm_device)
    inputs_embeds = torch.cat([visual_tokens, prompt_embeds], dim=1)  # [1, L_vis+L_prompt, D]

    # 浣跨敤 transformers.generate锛氬彧瑙ｇ爜銆屾柊鐢熸垚銆嶉儴鍒嗭紝閬垮厤鎶婅瑙?鎻愮ず鍗犱綅涔熷綋鏂囨湰
    input_len = L_vis + prompt_embeds.shape[1]
    attn_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long)
    # 鍖诲鎶ュ憡闇€杈冮暱杈撳嚭锛歮ax_new_tokens 寤鸿 512锛宭ength_penalty>1 榧撳姳闀垮彞锛宯o_repeat_ngram_size=0 閬垮厤璇激鍚堢悊閲嶅锛堝銆屽弻鑲恒€嶃€岀粨鑺傘€嶏級
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
    # 鍓旈櫎 None锛岄伩鍏?generate 鎶ラ敊
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
    try:
        gen_ids = llm_model.generate(**gen_kw)
    except ValueError as e:
        err = str(e)
        if force_words_ids and "Constrained Beam Search" in err:
            # Transformers>=5 may require external custom_generate for constrained beam search.
            # Fallback to standard decoding to keep inference functional.
            fallback_kw = dict(gen_kw)
            fallback_kw.pop("force_words_ids", None)
            fallback_kw.pop("custom_generate", None)
            fallback_kw.pop("trust_remote_code", None)
            fallback_kw["do_sample"] = False
            fallback_kw["num_beams"] = max(1, int(fallback_kw.get("num_beams", 1)))
            print(
                "[warn] constrained beam search is not supported in this transformers version; fallback to standard decoding.",
                flush=True,
            )
            gen_ids = llm_model.generate(**fallback_kw)
        else:
            raise
    # 瀵?inputs_embeds 鍦烘櫙锛岄儴鍒嗘ā鍨嬭繑鍥炪€屽畬鏁村簭鍒椼€嶏紝閮ㄥ垎鍙繑鍥炪€屾柊鐢熸垚搴忓垪銆嶃€?    # 鑻ヤ竴寰嬫寜 input_len 鎴柇锛屽彲鑳芥妸鏈氨鍙惈鏂?token 鐨勮緭鍑哄垏鎴愮┖涓层€?    if gen_ids.shape[1] > input_len:
        new_ids = gen_ids[0][input_len:]
    else:
        new_ids = gen_ids[0]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True)
    if raw_out is not None:
        raw_out.append(raw)
    cleaned = clean_generated(raw)
    # If post-clean text is empty but raw text exists, keep raw for debugging.
    if cleaned == "" and (raw and raw.strip()):
        return raw
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Medical VLM image-to-report inference")
    parser.add_argument("--image", type=str, default=None, help="NIfTI 鍥惧儚璺緞")
    parser.add_argument("--mask", type=str, default=None, help="NIfTI mask 璺緞锛堢敤浜庣梾鐏跺眰闈㈤€夋嫨涓庣粨鑺傚嬀鐢伙級")
    parser.add_argument("--draw_nodule_contour", action="store_true", help="Export nodule contour overlay and stats CSV (requires --mask)")
    parser.add_argument("--nodule_output_subdir", type=str, default="nodule_contour", help="鍕剧敾缁撴灉杈撳嚭鍒?run_dir 涓嬬殑瀛愮洰褰曞悕")
    parser.add_argument("--nodule_line_width", type=float, default=1.8, help="鍕剧敾杞粨绾垮")
    parser.add_argument("--nodule_fill_alpha", type=float, default=0.22, help="Contour fill alpha")
    parser.add_argument("--val_sample", action="store_true", help="浠庨獙璇侀泦鎶藉嚑鏉¤窇鐢熸垚")
    parser.add_argument("--checkpoint", type=str, default=None, help="vision_bridge 鏉冮噸锛岄粯璁?outputs/vision_bridge_best_val.pt 鎴?vision_bridge_final.pt")
    parser.add_argument("--mamba_model", type=str, default="state-spaces/mamba-2.8b-hf", help="Mamba pretrained model path")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max generation tokens")
    parser.add_argument("--max_visual_tokens", type=int, default=196, help="瑙嗚 token 涓婇檺锛岀渷鏄惧瓨闃?OOM锛岄粯璁?196(14脳14)")
    parser.add_argument("--num_beams", type=int, default=1, help="beam size锛?1 浼氭洿绋充絾鏇存參锛岀害鏉熻В鐮佸缓璁?4")
    parser.add_argument("--constrained_decode", action="store_true", help="绾︽潫瑙ｇ爜锛氬己鍒惰緭鍑?鎵€瑙?缁撹/寤鸿/鐥呯悊鍊惧悜 鍥涙鏍囬")
    parser.add_argument("--length_penalty", type=float, default=1.1, help="闀垮害鎯╃綒 >1 榧撳姳鏇撮暱杈撳嚭锛岄粯璁?1.1")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="No-repeat ngram size; 0 disables")
    parser.add_argument("--suppress_eos_steps", type=int, default=128, help="鍓?N 姝ョ姝?EOS锛岄伩鍏嶈繃鏃╃粨鏉燂紝榛樿 128")
    parser.add_argument("--llm_device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Mamba 璁惧锛歛uto 浣跨敤 GPU锛堟帹鑽愶級锛宑pu 闃?OOM锛宑uda 寮哄埗 GPU")
    parser.add_argument("--num_val", type=int, default=3)
    parser.add_argument("--use_csv_prompt", action="store_true", help="楠岃瘉鏃朵娇鐢?CSV 涓殑闂浣?prompt锛堝惈鍗犱綅绗︼級锛涢粯璁ょ敤鐭?prompt 閬垮厤妯″瀷閲嶅妯℃澘")
    parser.add_argument("--do_sample", action="store_true", help="鍚敤閲囨牱鐢熸垚锛堝彲鑳芥洿鍙戞暎锛岄粯璁ゅ叧闂級")
    parser.add_argument("--no_do_sample", action="store_true", help="Disable sampling and use greedy decode")
    parser.add_argument("--temperature", type=float, default=0.6, help="閲囨牱娓╁害锛屼粎 do_sample 鏃舵湁鏁堬紝榛樿 0.6")
    parser.add_argument("--top_p", type=float, default=0.85, help="Top-p nucleus sampling threshold when do_sample is enabled")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="閲嶅鎯╃綒锛屽噺杞婚噸澶嶈瘝锛岄粯璁?1.2")
    parser.add_argument("--csv", type=str, default=None, help="Override validation CSV for --val_sample")
    parser.add_argument("--ablation_mode", type=str, default=None, choices=["full", "global_only", "local_only"], help="Override ablation mode at inference time")
    parser.add_argument("--roi_jitter_3d", type=int, default=0, help="Apply random voxel jitter to roi_center_3d during --val_sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for roi jitter")
    parser.add_argument("--out_dir", type=str, default="D:/mamba-res", help="鐢熸垚鎶ュ憡钀界洏鐩綍锛岄粯璁?D:/mamba-res")
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
    if args.ablation_mode:
        config["ablation_mode"] = str(args.ablation_mode)
    if args.csv:
        config["caption_csv_val"] = str(args.csv)

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
        print("鏈壘鍒?vision_bridge checkpoint锛岃鍏堣缁冩垨鎸囧畾 --checkpoint")
        return 1

    print("鍔犺浇 Vision+Bridge...")
    vision_bridge = load_vision_bridge(ckpt, config, device)
    print("鍔犺浇 Mamba LLM锛堝彲鑳借緝涔咃級...")
    from llm.mamba_loader import load_mamba_lm
    llm_device_map = "cpu" if args.llm_device == "cpu" else args.llm_device  # "auto" 鎴?"cuda" 浣跨敤 GPU
    llm_model, tokenizer = load_mamba_lm(args.mamba_model, device_map=llm_device_map)
    # Align with train_vlm.py: tie lm_head to embeddings before generation.
    if hasattr(llm_model, "backbone") and hasattr(llm_model.backbone, "embeddings") and hasattr(llm_model, "lm_head"):
        llm_model.lm_head.weight = llm_model.backbone.embeddings.weight
        print("Applied weight tying (lm_head -> backbone.embeddings)", flush=True)
    llm_model.eval()
    force_words_ids = _build_template_force_words_ids(tokenizer) if getattr(args, "constrained_decode", False) else None

    res_dir = Path(args.out_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    run_dir = res_dir / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.val_sample:
        from data.medical_vlm_dataset import MedicalVLMDataset
        csv_val = args.csv or config.get("caption_csv_val")
        if not csv_val or not Path(csv_val).exists():
            print("楠岃瘉闆?CSV 涓嶅瓨鍦紝鏃犳硶 --val_sample")
            return 1
        if getattr(args, "roi_jitter_3d", 0) > 0:
            torch.manual_seed(int(getattr(args, "seed", 42)))
            np.random.seed(int(getattr(args, "seed", 42)))
        val_ds = MedicalVLMDataset(
            csv_val,
            prompt_json_file=config.get("caption_prompt_json"),
            spatial_dims=int(config.get("spatial_dims", 2)),
            patch_size_3d=_as_tuple3(config.get("patch_size_3d", (32, 128, 128))),
        )
        num = min(args.num_val, len(val_ds))
        meta_list = []
        for idx in range(num):
            sample = val_ds[idx]
            image_t = sample["image"].unsqueeze(0)
            roi_center = sample.get("roi_center")
            roi_center_3d = sample.get("roi_center_3d")
            if isinstance(roi_center_3d, torch.Tensor) and getattr(args, "roi_jitter_3d", 0) > 0:
                jit = int(args.roi_jitter_3d)
                if roi_center_3d.numel() >= 3 and float(roi_center_3d.min().item()) >= 0:
                    delta = torch.randint(low=-jit, high=jit + 1, size=(3,), dtype=torch.int64).to(roi_center_3d.dtype)
                    roi_center_3d = roi_center_3d + delta
            answer_gt = sample["answer"]
            if getattr(args, "use_csv_prompt", False):
                prompt = sample.get("question") or PROMPT_SHORT_NO_PLACEHOLDERS
            else:
                prompt = PROMPT_SHORT_NO_PLACEHOLDERS
            print(f"\n--- 楠岃瘉鏍锋湰 {idx+1}/{num} ---")
            print(f"Prompt: {prompt.strip()}")
            gen = generate_from_image(
                image_t, vision_bridge, llm_model, tokenizer, prompt=prompt,
                max_new_tokens=args.max_new_tokens, device=device, max_visual_tokens=args.max_visual_tokens,
                do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                length_penalty=getattr(args, "length_penalty", 1.1),
                no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
                suppress_eos_steps=getattr(args, "suppress_eos_steps", 128),
                num_beams=getattr(args, "num_beams", 1),
                force_words_ids=force_words_ids,
                roi_center=roi_center,
                roi_center_3d=roi_center_3d,
            )
            print(f"鐢熸垚: {gen[:500]}...")
            print(f"鍙傝€? {answer_gt[:200]}...")
            (run_dir / f"sample_{idx+1}_gen.txt").write_text(gen, encoding="utf-8")
            (run_dir / f"sample_{idx+1}_ref.txt").write_text(answer_gt, encoding="utf-8")
            sample_meta = {
                "idx": idx + 1,
                "image_path": sample.get("image_path"),
                "mask_path": sample.get("mask_path"),
                "prompt": prompt.strip(),
            }
            # Keep GT grade for post-run evaluation if available.
            gt_grade = sample.get("grade", -1)
            try:
                gt_grade = int(gt_grade)
            except Exception:
                gt_grade = -1
            sample_meta["grade_gt"] = gt_grade
            if 0 <= gt_grade < len(GRADE_LABELS):
                sample_meta["grade_gt_label"] = GRADE_LABELS[gt_grade]
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
        print(f"\n宸茶惤鐩? {run_dir}")
        return 0

    if not args.image or not Path(args.image).exists():
        print("璇锋寚瀹?--image <NIfTI璺緞> 鎴栦娇鐢?--val_sample")
        return 1
    if args.mask and not Path(args.mask).exists():
        print(f"mask file not found: {args.mask}")
        return 1
    image_t = load_image_tensor(args.image, mask_path=args.mask).to(device)
    text = generate_from_image(
        image_t, vision_bridge, llm_model, tokenizer,
        max_new_tokens=args.max_new_tokens, device=device, max_visual_tokens=args.max_visual_tokens,
        do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p, repetition_penalty=args.repetition_penalty,
        length_penalty=getattr(args, "length_penalty", 1.1),
        no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
        suppress_eos_steps=getattr(args, "suppress_eos_steps", 128),
        num_beams=getattr(args, "num_beams", 1),
        force_words_ids=force_words_ids,
    )
    print("鐢熸垚鎶ュ憡:")
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
    print(f"\n宸茶惤鐩? {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

