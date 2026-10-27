"""
摘要评测：覆盖率 / 可定位率 / 压缩比。

覆盖率需要一份人工参考摘要（eval/summary_ref.json），由 LLM 判断「生成摘要里有没有覆盖这个要点」。
可定位率来自摘要本身：每条要点的 quote 是否在转写里精确匹配到了。
压缩比 = 摘要字数 / 转写字数。

用法：
    python video_eval.py --summary summaries/semantic.json --ref eval/summary_ref.json \
                         --transcript transcripts/video.fixed.json
"""

import argparse
import json
import re
from pathlib import Path

from common import BASE_URL, MODEL, load_key

JUDGE = """你是一个严格但公正的评分员。

给你一个【参考要点】和一份【生成摘要】。判断生成摘要里有没有内容覆盖了这个参考要点。

判定标准：
- 意思相同或更具体，算覆盖。
- 只是提到了同一个词，但没讲清楚这件事，不算覆盖。
- 不要因为措辞不同就判为未覆盖。

只输出 JSON：
{"covered": true, "matched": "对应的摘要原句，没有就留空"}"""


def judge(client, point: str, lines: list[str]) -> tuple[bool, str]:
    body = "【参考要点】\n" + point + "\n\n【生成摘要】\n" + "\n".join(f"- {l}" for l in lines)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": JUDGE},
            {"role": "user", "content": body},
        ],
        temperature=0,
    )
    raw = resp.choices[0].message.content
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return False, ""
    try:
        d = json.loads(m.group())
        return bool(d.get("covered")), str(d.get("matched", "")).strip()
    except json.JSONDecodeError:
        return False, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="summaries/semantic.json")
    ap.add_argument("--ref", default="eval/summary_ref.json")
    ap.add_argument("--transcript", default="")
    ap.add_argument("--out", default="eval/summary_eval.json")
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    ref = json.loads(Path(args.ref).read_text(encoding="utf-8"))["points"]

    chapters = summary["chapters"]
    lines = [p["text"] for c in chapters for p in c["points"]]
    points = [p for c in chapters for p in c["points"]]

    key = load_key()
    if not key:
        raise SystemExit("没读到 DEEPSEEK_API_KEY，请检查 .env")
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=BASE_URL)

    print(f"参考要点 {len(ref)} 条，生成要点 {len(points)} 条\n")
    details = []
    for i, point in enumerate(ref, 1):
        covered, matched = judge(client, point, lines)
        details.append({"point": point, "covered": covered, "matched": matched})
        mark = "✓" if covered else "✗"
        print(f"  {mark} {point}")
        if matched:
            print(f"      对应：{matched[:50]}")

    coverage = sum(1 for d in details if d["covered"]) / len(details) if details else 0
    locatable = sum(1 for p in points if not p.get("approx", True))
    loc_rate = locatable / len(points) if points else 0

    result = {
        "coverage": round(coverage, 3),
        "covered": sum(1 for d in details if d["covered"]),
        "ref_total": len(details),
        "locatable": locatable,
        "point_total": len(points),
        "locate_rate": round(loc_rate, 3),
        "details": details,
    }

    if args.transcript:
        segs = json.loads(Path(args.transcript).read_text(encoding="utf-8"))["segments"]
        src_chars = sum(len(s["text"]) for s in segs)
        sum_chars = sum(len(p["text"]) for p in points) + len(summary.get("overview", ""))
        result["source_chars"] = src_chars
        result["summary_chars"] = sum_chars
        result["compression"] = round(sum_chars / src_chars, 3)

    print("\n" + "=" * 50)
    print(f"覆盖率      {result['covered']}/{result['ref_total']}  ({coverage:.0%})")
    print(f"可定位率    {locatable}/{len(points)}  ({loc_rate:.0%})")
    if "compression" in result:
        print(f"压缩比      {result['compression']:.1%}  ({result['summary_chars']} / {result['source_chars']} 字)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已写入 {out}")


if __name__ == "__main__":
    main()
