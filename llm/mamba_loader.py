"""
从 HuggingFace 加载预训练 Mamba（Mamba-2.8B / OpenElm 等），不从头训练。

与 ARCHITECTURE.md 一致：使用 state-spaces/mamba-2.8b-hf 或其它 Mamba 预训练权重作为 LLM 初始化。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import os

# HuggingFace Transformers 对 Mamba 的支持（>=4.39 或 main）
try:
    from transformers import AutoTokenizer, MambaForCausalLM
    _HAS_HF_MAMBA = True
except ImportError:
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        _HAS_HF_MAMBA = True
        MambaForCausalLM = AutoModelForCausalLM
    except ImportError:
        _HAS_HF_MAMBA = False
        MambaForCausalLM = None
        AutoTokenizer = None


DEFAULT_MAMBA_HF = "state-spaces/mamba-2.8b-hf"


def _patch_transformers_mamba_slow_path_sm120():
    """
    RTX 5090 (sm_120)：transformers 的 MambaMixer 会走 cuda_kernels_forward 并调用 causal_conv1d，
    其 CUDA 内核未为 sm_120 编译时会报 no kernel image。强制将 causal_conv1d_fn/update 设为 None，
    使 MambaMixer.forward 走 slow_forward（纯 PyTorch）。
    """
    try:
        if not torch.cuda.is_available():
            return
        cap = torch.cuda.get_device_capability()
        if cap < (12, 0):
            return
        import transformers.models.mamba.modeling_mamba as _mamba_mod
        _mamba_mod.causal_conv1d_fn = None
        _mamba_mod.causal_conv1d_update = None
    except Exception:
        pass


def _patch_mamba_ssm_for_transformers():
    """
    transformers 的 MambaMixer 会访问 mamba_ssm.selective_state_update，但 pip 安装的
    mamba_ssm 只在子模块 mamba_ssm.ops.triton.selective_state_update 中提供该函数。
    在加载模型前为 mamba_ssm 补上该属性，避免 AttributeError。
    """
    try:
        import mamba_ssm
        if not hasattr(mamba_ssm, "selective_state_update"):
            try:
                from mamba_ssm.ops.triton.selective_state_update import selective_state_update
                mamba_ssm.selective_state_update = selective_state_update
            except ImportError:
                pass
    except ImportError:
        pass


def _is_local_dir(path: str) -> bool:
    """判断是否为已存在的本地目录且含 config.json（可离线加载）。"""
    p = Path(path).resolve()
    return p.is_dir() and (p / "config.json").is_file()


def load_mamba_lm(
    model_name_or_path: str = DEFAULT_MAMBA_HF,
    device_map: Optional[str] = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    cache_dir: Optional[str | Path] = None,
    load_in_8bit: bool = False,
    local_files_only: Optional[bool] = None,
    align_vocab: bool = False,
):
    """
    加载预训练 Mamba 因果语言模型与 tokenizer。

    - model_name_or_path: HuggingFace 模型 id 或本地路径（如 /mnt/d/mamba/models/mamba-2.8b-hf）
    - device_map: 如 "auto" 用于多卡/显存分配
    - torch_dtype: 如 torch.bfloat16 节省显存
    - cache_dir: HF 缓存目录
    - load_in_8bit: 是否 8-bit 量化加载（需 bitsandbytes），显著省显存、减轻 OOM

    Returns:
        model, tokenizer
    """
    if not _HAS_HF_MAMBA:
        raise ImportError(
            "Install transformers (with Mamba support, e.g. >=4.39 or from main) and optional mamba-ssm, causal-conv1d."
        )
    _patch_mamba_ssm_for_transformers()
    if torch_dtype is None and not load_in_8bit:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # 判断是否为本地目录路径（绝对路径或含路径分隔符）
    looks_local = (
        model_name_or_path.startswith("/")
        or os.path.sep in model_name_or_path
        or (len(model_name_or_path) >= 2 and model_name_or_path[1] == ":")
    )
    if looks_local:
        if _is_local_dir(model_name_or_path):
            model_name_or_path = str(Path(model_name_or_path).resolve())
            local_files_only = True
        else:
            raise FileNotFoundError(
                f"本地 Mamba 路径不存在或缺少 config.json: '{model_name_or_path}'\n"
                "请先在能访问 Hugging Face 的环境里下载模型并保存到该目录，参见 docs/WSL_TRAIN.md 第九节。"
            )
    if local_files_only is None:
        # 兼容 HuggingFace/Transformers 的离线开关
        local_files_only = os.getenv("HF_HUB_OFFLINE", "0") == "1" or os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=str(cache_dir) if cache_dir else None,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
    except Exception as e:
        if "ModelWrapper" in str(e) or "untagged enum" in str(e) or "tokenizer" in str(e).lower():
            print(f"Fast tokenizer 加载失败，改用 use_fast=False: {e}", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                cache_dir=str(cache_dir) if cache_dir else None,
                trust_remote_code=True,
                local_files_only=local_files_only,
                use_fast=False,
            )
        else:
            raise
    load_kw = dict(
        device_map=device_map,
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
            load_kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kw["torch_dtype"] = torch_dtype or torch.float16
        except ImportError:
            load_in_8bit = False
            load_kw["torch_dtype"] = torch_dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    if not load_in_8bit:
        load_kw["torch_dtype"] = torch_dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model = MambaForCausalLM.from_pretrained(model_name_or_path, **load_kw)
    # RTX 5090 (sm_120)：默认强制 LLM 走 slow_forward；若已设 MAMBA_FORCE_CUDA=1 且用 PTX 重编过，则不补丁以走 Fast Path
    if os.environ.get("MAMBA_FORCE_CUDA", "0") != "1":
        _patch_transformers_mamba_slow_path_sm120()
    if align_vocab:
        try:
            tok_len = len(tokenizer)
            emb_rows = int(model.get_input_embeddings().weight.shape[0])
            if emb_rows != tok_len:
                model.resize_token_embeddings(tok_len)
                print(f"已对齐词表: resize_token_embeddings {emb_rows} -> {tok_len}", flush=True)
        except Exception as e:
            print(f"警告: 词表对齐失败，将沿用原始 vocab。原因: {e}", flush=True)

    # 修复 lm_head MISSING：Mamba 设计为 lm_head 与 embedding 共用权重
    if hasattr(model, "backbone") and hasattr(model.backbone, "embeddings") and hasattr(model, "lm_head"):
        model.lm_head.weight = model.backbone.embeddings.weight
        print("lm_head 已绑定到 backbone.embeddings，生成将使用正确词表。", flush=True)
    else:
        embed = model.get_input_embeddings()
        head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else getattr(model, "lm_head", None)
        if embed is not None and head is not None and hasattr(head, "weight") and hasattr(embed, "weight") and head.weight.shape == embed.weight.shape:
            head.weight = embed.weight
            print("lm_head 已绑定到 input_embeddings，生成将使用正确词表。", flush=True)
    return model, tokenizer


def get_mamba_config(model_name_or_path: str = DEFAULT_MAMBA_HF) -> dict:
    """获取 Mamba 模型 hidden_size 等配置，用于桥接器 d_model 对齐。"""
    try:
        from transformers import AutoConfig
        local_only = _is_local_dir(model_name_or_path)
        cfg = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        return {
            "hidden_size": getattr(cfg, "hidden_size", getattr(cfg, "d_model", 2560)),
            "vocab_size": getattr(cfg, "vocab_size", 50280),
        }
    except Exception:
        return {"hidden_size": 2560, "vocab_size": 50280}
