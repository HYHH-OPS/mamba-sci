"""
根据报告文本做侵润/风险倾向的简单推断（占位实现，非临床诊断）。

仅供流程串联与后续扩展；真实侵润等级诊断需结合病理或专用分类模型。
"""
from __future__ import annotations

import re
from typing import Any


# 关键词到倾向的简单映射（可随需求扩展）
_KEYWORDS_RISK = [
    (["浸润", "浸润性", "侵袭", "转移", "恶性"], "高危倾向（报告提及浸润/侵袭等，需病理确认）"),
    (["肿瘤性病变", "肿瘤性", "恶性待排", "占位性病变"], "肿瘤性待排（建议进一步检查）"),
    (["磨玻璃", "部分实性", "实性结节"], "结节性质待定（建议随访或病理）"),
    (["炎性", "炎性可能", "炎性结节"], "炎性倾向"),
    (["随诊", "随访", "短期复查"], "随访建议"),
]
_KEYWORDS_LOW = ["炎性", "炎性可能", "钙化", "随诊"]
_KEYWORDS_HIGH = ["浸润", "侵袭", "恶性", "转移", "肿瘤性病变"]


def infer_invasiveness_from_report(report: str) -> dict[str, Any]:
    """
    从报告文本中做简单关键词匹配，返回倾向标签与说明。
    返回格式：{"label": str, "detail": str, "raw_report_snippet": str}
    """
    if not report or not report.strip():
        return {"label": "无法推断", "detail": "报告为空", "raw_report_snippet": ""}

    report_lower = report.strip()
    snippet = report_lower[:500] if len(report_lower) > 500 else report_lower

    for keywords, detail in _KEYWORDS_RISK:
        for kw in keywords:
            if kw in report_lower:
                return {
                    "label": detail.split("（")[0].strip(),
                    "detail": detail,
                    "raw_report_snippet": snippet,
                }

    if any(k in report_lower for k in _KEYWORDS_HIGH):
        return {"label": "肿瘤性待排", "detail": "报告含肿瘤相关描述，建议结合病理", "raw_report_snippet": snippet}
    if any(k in report_lower for k in _KEYWORDS_LOW):
        return {"label": "炎性/低危倾向", "detail": "报告含炎性/随访描述", "raw_report_snippet": snippet}

    return {
        "label": "待定（需结合临床与病理）",
        "detail": "未匹配到预设关键词，仅作占位；侵润等级需临床诊断或专用模型。",
        "raw_report_snippet": snippet,
    }


def format_invasiveness_output(result: dict[str, Any]) -> str:
    """格式化为可写文件的文本。"""
    lines = [
        "# 侵润/风险倾向（占位，非临床诊断）",
        f"label: {result['label']}",
        f"detail: {result['detail']}",
        "",
        "report_snippet:",
        result.get("raw_report_snippet", "")[:800],
    ]
    return "\n".join(lines)
