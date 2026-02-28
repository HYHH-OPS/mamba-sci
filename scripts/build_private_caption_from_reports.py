"""
Build caption train/val CSV from private report tables and aligned CT-mask pairs.

Input sources:
1) Total report table (has report text), e.g. 2021-2024肺结节数据整理总表.xlsx, sheet "总表"
2) Mapping table from total row index -> patient id, e.g. 信息病表.xlsx
3) Aligned pairs CSV, e.g. pairs_aligned.csv (case_id,image_path,mask_path)

Output CSV columns:
- image_path
- mask_path
- question
- answer
- case_id
- seq_id
- pid10
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.medical_vlm_dataset import CAPTION_DEFAULT_QUESTION_NO_NL


def _abs_no_resolve(p: str | Path) -> Path:
    return Path(os.path.abspath(str(Path(p).expanduser())))


def _find_col(cols: list[str], key: str) -> str | None:
    for c in cols:
        if key in str(c):
            return c
    return None


def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    lines = [x.strip() for x in s.split("\n") if x.strip()]
    return "\n".join(lines)


def _thorax_only_text(s: str) -> str:
    if not s:
        return ""
    keep = re.compile(r"胸|肺|气管|支气管|肺门|胸膜|纵隔|结节|病灶|磨玻璃|实性|空洞|胸腔", re.IGNORECASE)
    drop = re.compile(r"肝|肾|脾|胰|腹|盆腔|子宫|卵巢|前列腺|膀胱|头颅|颅脑|鼻窦|咽喉|甲状腺|乳腺", re.IGNORECASE)
    lines: list[str] = []
    for ln in s.split("\n"):
        t = ln.strip()
        if not t:
            continue
        if drop.search(t):
            continue
        if keep.search(t):
            lines.append(t)
    return "\n".join(lines) if lines else s


def _make_answer(row: pd.Series, finding_col: str | None, concl_col: str | None, path_col: str | None, thorax_only: bool) -> str:
    parts: list[str] = []
    if finding_col:
        x = _normalize_text(str(row.get(finding_col, "")))
        if x:
            parts.append(x)
    if concl_col:
        x = _normalize_text(str(row.get(concl_col, "")))
        if x:
            parts.append(x)
    if path_col:
        x = _normalize_text(str(row.get(path_col, "")))
        if x and x.lower() != "nan":
            parts.append("病理诊断：" + x)
    ans = _normalize_text("\n".join(parts))
    if thorax_only:
        ans = _normalize_text(_thorax_only_text(ans))
    return ans


def main() -> int:
    ap = argparse.ArgumentParser(description="Build private caption train/val CSV with mask_path")
    ap.add_argument("--total_excel", required=True, help="Report total excel path")
    ap.add_argument("--total_sheet", type=int, default=1, help="Total sheet index, usually 1")
    ap.add_argument("--map_excel", required=True, help="Mapping excel path (e.g. 信息病表.xlsx)")
    ap.add_argument("--map_sheet", type=int, default=0)
    ap.add_argument("--map_seq_col_idx", type=int, default=0, help="Map table seq column index")
    ap.add_argument("--map_pid_col_idx", type=int, default=2, help="Map table patient id column index")
    ap.add_argument("--pairs_csv", required=True, help="Aligned pairs csv with case_id,image_path,mask_path")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thorax_only", action="store_true", help="Drop non-thorax lines from answer text")
    ap.add_argument("--min_answer_chars", type=int, default=0)
    ap.add_argument("--require_keywords", type=str, default=None)
    args = ap.parse_args()

    total_excel = _abs_no_resolve(args.total_excel)
    map_excel = _abs_no_resolve(args.map_excel)
    pairs_csv = _abs_no_resolve(args.pairs_csv)
    out_dir = _abs_no_resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = pd.read_excel(total_excel, sheet_name=args.total_sheet)
    mapdf = pd.read_excel(map_excel, sheet_name=args.map_sheet)
    pairs = pd.read_csv(pairs_csv, encoding="utf-8-sig")

    seq_col = total.columns[0]
    finding_col = _find_col([str(c) for c in total.columns], "报告所见")
    concl_col = _find_col([str(c) for c in total.columns], "报告结论")
    path_col = _find_col([str(c) for c in total.columns], "病理诊断")

    map_seq_col = mapdf.columns[args.map_seq_col_idx]
    map_pid_col = mapdf.columns[args.map_pid_col_idx]

    m = mapdf[[map_seq_col, map_pid_col]].copy()
    m = m.dropna(subset=[map_seq_col, map_pid_col])
    m[map_seq_col] = m[map_seq_col].astype(int)
    m[map_pid_col] = m[map_pid_col].astype(int)
    m["pid10"] = m[map_pid_col].astype(str).str.zfill(10)

    pairs = pairs.copy()
    if "case_id" not in pairs.columns:
        raise ValueError("pairs_csv must include case_id column")
    if "image_path" not in pairs.columns or "mask_path" not in pairs.columns:
        raise ValueError("pairs_csv must include image_path and mask_path columns")
    pairs["pid10"] = pairs["case_id"].astype(str).str.split("_").str[0]

    merged = total.merge(m[["pid10", map_seq_col]], left_on=seq_col, right_on=map_seq_col, how="left")
    merged = merged.merge(pairs[["pid10", "case_id", "image_path", "mask_path"]], on="pid10", how="left")

    out = pd.DataFrame()
    out["seq_id"] = merged[seq_col]
    out["pid10"] = merged["pid10"]
    out["case_id"] = merged["case_id"]
    out["image_path"] = merged["image_path"]
    out["mask_path"] = merged["mask_path"]
    out["question"] = CAPTION_DEFAULT_QUESTION_NO_NL
    out["answer"] = merged.apply(
        lambda r: _make_answer(r, finding_col, concl_col, path_col, args.thorax_only),
        axis=1,
    )

    out = out[out["image_path"].notna() & out["mask_path"].notna()].copy()
    out = out[out["image_path"].astype(str).str.strip() != ""]
    out = out[out["mask_path"].astype(str).str.strip() != ""]
    out = out[out["answer"].astype(str).str.strip() != ""]

    if args.min_answer_chars > 0:
        out = out[out["answer"].astype(str).str.len() >= args.min_answer_chars]
    if args.require_keywords and args.require_keywords.strip():
        kw = args.require_keywords.strip()
        try:
            out = out[out["answer"].astype(str).str.contains(kw, regex=True)]
        except re.error:
            out = out[out["answer"].astype(str).str.contains(re.escape(kw))]

    out = out.reset_index(drop=True)

    full_csv = out_dir / "caption_full_with_mask.csv"
    out.to_csv(full_csv, index=False, encoding="utf-8-sig")

    case_ids = out["case_id"].dropna().astype(str).unique().tolist()
    rs = pd.Series(case_ids).sample(frac=1.0, random_state=args.seed).tolist()
    n_val_case = max(1, int(len(rs) * args.val_ratio))
    val_cases = set(rs[:n_val_case])

    val = out[out["case_id"].isin(val_cases)].reset_index(drop=True)
    train = out[~out["case_id"].isin(val_cases)].reset_index(drop=True)

    train_csv = out_dir / "caption_train_with_mask.csv"
    val_csv = out_dir / "caption_val_with_mask.csv"
    train.to_csv(train_csv, index=False, encoding="utf-8-sig")
    val.to_csv(val_csv, index=False, encoding="utf-8-sig")

    print("total_rows_in_report:", len(total))
    print("rows_after_join_with_mask:", len(out))
    print("unique_cases_after_join:", out["case_id"].nunique())
    print("train_rows:", len(train), "train_cases:", train["case_id"].nunique())
    print("val_rows:", len(val), "val_cases:", val["case_id"].nunique())
    print("wrote:", full_csv)
    print("wrote:", train_csv)
    print("wrote:", val_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
