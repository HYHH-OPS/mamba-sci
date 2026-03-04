"""
Clean private caption CSV text fields for stable medical VLM training.

Typical issues:
- leading "???" / mojibake prefixes
- replacement characters (U+FFFD)
- template leaks like "begin_of_sentence"

Example:
  python scripts/clean_private_caption_csv.py \
    --in_csv /autodl-tmp/caption_train_private_v5_ord_all.csv \
    --out_csv /autodl-tmp/caption_train_private_v5_ord_all_clean.csv \
    --report_json /autodl-tmp/caption_train_private_v5_ord_all_clean_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


TEXT_FIELDS = (
    "question",
    "answer",
    "pathology_text",
    "report",
    "caption",
    "prompt",
)

BAD_ROW_RE = re.compile(r"(begin_of_sentence|\?{3,}|？{3,}|�)", re.IGNORECASE)


def _read_csv_with_fallback(path: Path) -> tuple[list[dict[str, str]], list[str], str]:
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames or []
            # remove BOM in header if present
            fieldnames = [fn.lstrip("\ufeff") if isinstance(fn, str) else fn for fn in fieldnames]
            fixed_rows: list[dict[str, str]] = []
            for r in rows:
                rr = {}
                for k, v in r.items():
                    kk = k.lstrip("\ufeff") if isinstance(k, str) else k
                    rr[kk] = v
                fixed_rows.append(rr)
            return fixed_rows, fieldnames, enc
        except Exception as e:
            last_err = e
    raise RuntimeError(f"failed to read csv: {path}; last_err={last_err}")


def _clean_text(s: str) -> str:
    if s is None:
        return ""
    t = str(s).replace("\ufeff", "").strip()

    # Drop repeated broken prefix markers.
    t = re.sub(r"^(?:[?？�]{2,}\s*)+", "", t)

    # If prefix is mostly broken symbols before first valid char, trim prefix.
    m = re.search(r"[\u4e00-\u9fffA-Za-z0-9]", t)
    if m and m.start() > 0:
        prefix = t[: m.start()]
        if sum(ch in "?？�" for ch in prefix) >= 3:
            t = t[m.start() :]

    # Remove explicit template leak token.
    t = re.sub(r"begin_of_sentence", "", t, flags=re.IGNORECASE)

    # Normalize visible artifacts.
    t = t.replace("�", "")
    t = re.sub(r"[?？]{3,}", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    return t.strip()


def _row_bad_score(row: dict[str, str]) -> int:
    score = 0
    for k in TEXT_FIELDS:
        if k in row:
            txt = row.get(k, "") or ""
            if BAD_ROW_RE.search(txt):
                score += 1
    return score


def clean_csv(in_csv: Path, out_csv: Path, report_json: Path | None = None) -> dict:
    rows, fieldnames, used_enc = _read_csv_with_fallback(in_csv)
    if not rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(fieldnames)
        report = {
            "input_csv": str(in_csv),
            "output_csv": str(out_csv),
            "rows": 0,
            "encoding_used": used_enc,
            "changed_rows": 0,
            "bad_rows_before": 0,
            "bad_rows_after": 0,
        }
        if report_json:
            report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    bad_before = 0
    bad_after = 0
    changed_rows = 0
    changed_cells = 0

    cleaned: list[dict[str, str]] = []
    for row in rows:
        if _row_bad_score(row) > 0:
            bad_before += 1
        new_row = dict(row)
        row_changed = False
        for k in TEXT_FIELDS:
            if k not in new_row:
                continue
            old = new_row.get(k, "") or ""
            new = _clean_text(old)
            if new != old:
                new_row[k] = new
                row_changed = True
                changed_cells += 1
        if _row_bad_score(new_row) > 0:
            bad_after += 1
        if row_changed:
            changed_rows += 1
        cleaned.append(new_row)

    # keep stable header order; append any extra keys if needed
    keys = list(fieldnames)
    for r in cleaned:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(cleaned)

    report = {
        "input_csv": str(in_csv),
        "output_csv": str(out_csv),
        "rows": len(rows),
        "encoding_used": used_enc,
        "changed_rows": changed_rows,
        "changed_cells": changed_cells,
        "bad_rows_before": bad_before,
        "bad_rows_after": bad_after,
    }
    if report_json:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean private caption CSV for VLM training.")
    parser.add_argument("--in_csv", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--report_json", type=str, default=None)
    args = parser.parse_args()

    in_csv = Path(args.in_csv).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    report_json = Path(args.report_json).expanduser().resolve() if args.report_json else None

    report = clean_csv(in_csv, out_csv, report_json)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

