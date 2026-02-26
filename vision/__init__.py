"""
vision package exports with lazy imports to avoid heavy side effects.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "NNUnetEncoderLight",
    "build_nnunet_encoder_light",
    "compute_nodule_stats",
    "find_best_slice",
    "generate_nodule_contour_outputs",
    "load_image_and_mask",
    "load_slice_with_optional_mask",
    "save_contour_overlay_png",
]


def __getattr__(name: str) -> Any:
    if name in {"NNUnetEncoderLight", "build_nnunet_encoder_light"}:
        from .nnunet_encoder import NNUnetEncoderLight, build_nnunet_encoder_light

        return {
            "NNUnetEncoderLight": NNUnetEncoderLight,
            "build_nnunet_encoder_light": build_nnunet_encoder_light,
        }[name]

    if name in {
        "compute_nodule_stats",
        "find_best_slice",
        "generate_nodule_contour_outputs",
        "load_image_and_mask",
        "load_slice_with_optional_mask",
        "save_contour_overlay_png",
    }:
        from .nodule_contour import (
            compute_nodule_stats,
            find_best_slice,
            generate_nodule_contour_outputs,
            load_image_and_mask,
            load_slice_with_optional_mask,
            save_contour_overlay_png,
        )

        return {
            "compute_nodule_stats": compute_nodule_stats,
            "find_best_slice": find_best_slice,
            "generate_nodule_contour_outputs": generate_nodule_contour_outputs,
            "load_image_and_mask": load_image_and_mask,
            "load_slice_with_optional_mask": load_slice_with_optional_mask,
            "save_contour_overlay_png": save_contour_overlay_png,
        }[name]

    raise AttributeError(f"module 'vision' has no attribute '{name}'")
