#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def wrap_text(t: str, width: int = 46) -> str:
    t = (t or '').replace('\r', '')
    lines = []
    for ln in t.split('\n'):
        ln = ln.strip()
        if not ln:
            lines.append('')
            continue
        lines.extend(textwrap.wrap(ln, width=width, break_long_words=False, replace_whitespace=False))
    return '\n'.join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample_json', type=str, required=True)
    ap.add_argument('--out_pdf', type=str, default='/autodl-tmp/mamba-sci/figure2_qualitative.pdf')
    args = ap.parse_args()

    d = json.loads(Path(args.sample_json).read_text(encoding='utf-8'))
    idx = d.get('idx', '')
    hits = d.get('hits', [])
    img = d.get('image_path', '')
    mask = d.get('mask_path', '')
    gt_grade = d.get('grade_gt', '')
    pd_grade = d.get('grade_pred', '')
    gt = d.get('ref', '')
    ours = d.get('gen', '')

    fig = plt.figure(figsize=(12, 8), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 9], wspace=0.2, hspace=0.1)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.axis('off')
    header = (
        f'Figure 2. Qualitative case (sample_{idx}) | keyword hits: {", ".join(hits)}\n'
        f'image: {img}\nmask: {mask}\n'
        f'grade_gt={gt_grade}, grade_pred={pd_grade}'
    )
    ax0.text(0.0, 1.0, header, va='top', ha='left', fontsize=10, family='DejaVu Sans Mono')

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.axis('off')
    ax1.set_title('Ground Truth Report (GT)', fontsize=12, fontweight='bold')
    ax1.text(0.0, 1.0, wrap_text(gt, 42), va='top', ha='left', fontsize=10)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.axis('off')
    ax2.set_title('MambaLung Report (Ours)', fontsize=12, fontweight='bold')
    ax2.text(0.0, 1.0, wrap_text(ours, 42), va='top', ha='left', fontsize=10)

    out = Path(args.out_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    print(str(out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
