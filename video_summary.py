"""
视频摘要：按章节生成带时间戳的分段摘要。

精度说明：模型为每个要点摘录一句「原文」，程序拿这句原文去转写结果里精确匹配，
命中的话时间戳就是那句话真正说出来的时刻；匹配不上才退回到整块的开头，并标记 approx。

用法：
    python video_summary.py chunks/semantic.json --transcript transcripts/video.fixed.json
    python video_summary.py chunks/semantic.json --md

输出：summaries/<同名>.json（+ .md）
"""

import argparse
import json
import re
from pathlib import Path

from common import BASE_URL, MODEL, load_key

SYSTEM = """你是视频内容编辑。下面是视频某一段的转写文本，按时间顺序分成若干片段。

请输出 JSON：
{
  "title": "这一段的标题，不超过 20 字",
  "points": [
    {"text": "要点，一句话", "quote": "从原文逐字摘录的一句支撑它的话，10~30 字"}
  ]
}

规则：
1. 只依据给出的文本，不要补充文本以外的知识。
2. 3~5 条要点，每条只讲一件事，不要重复。
3. quote 必须逐字来自原文，不要改写、不要拼接。
4. 不要写时间戳，系统会自动附加。
5. 只输出 JSON，不要任何解释。"""


def fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def norm(s: str) -> str:
    return re.sub(r"[\s，。、,.!?！？：:；;「」『』\"'（）()]", "", s)


def locate(quote: str, segments: list[dict]) -> float | None:
    """把模型摘录的原文对回转写里的时间戳。先精确匹配，再退化到 8 字窗口。"""
    q = norm(quote)
    if len(q) < 6:
        return None

    for s in segments:
        if q in norm(s["text"]):
            return s["start"]

    for size in (10, 8):
        for i in range(0, max(0, len(q) - size) + 1):
            frag = q[i : i + size]
            for s in segments:
                if frag in norm(s["text"]):
                    return s["start"]
    return None


def summarize_chapter(client, chunks: list[dict], segments: list[dict]) -> dict:
    body = "\n\n".join(
        f"[{i}] {fmt(c['start'])}-{fmt(c['end'])}\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": body},
        ],
        temperature=0.3,
    )
    raw = resp.choices[0].message.content
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        data = {}

    points = []
    for p in data.get("points", [])[:5]:
        if isinstance(p, str):
            text, quote = p, ""
        else:
            text, quote = str(p.get("text", "")).strip(), str(p.get("quote", "")).strip()
        if not text:
            continue

        exact = locate(quote, segments) if (quote and segments) else None
        if exact is not None:
            start, approx = exact, False
        else:
            # 退回到「这句话提到的东西在哪个块里」
            hit = next((c for c in chunks if quote and norm(quote)[:8] in norm(c["text"])), chunks[0])
            start, approx = hit["start"], True

        points.append(
            {"text": text, "quote": quote, "start": round(start, 2),
             "time": fmt(start), "approx": approx}
        )

    return {"title": str(data.get("title", "")).strip(), "points": points}


def overview(client, chapters: list[dict]) -> str:
    lines = [f"{c['time']}  {c['title']}" for c in chapters]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "根据下面的章节列表，用 2~3 句话概括整段视频讲了什么。只输出概括文字。",
            },
            {"role": "user", "content": "\n".join(lines)},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def to_markdown(data: dict) -> str:
    out = ["# 视频摘要", "", f"**总览**：{data['overview']}", ""]
    for c in data["chapters"]:
        out.append(f"## {c['time']}　{c['title']}")
        for p in c["points"]:
            mark = "≈" if p["approx"] else ""
            out.append(f"- {p['text']}　`{mark}{p['time']}`")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks")
    ap.add_argument("--transcript", default="", help="用于把 quote 对回精确时间戳")
    ap.add_argument("--md", action="store_true", help="同时导出 Markdown")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    src = Path(args.chunks)
    data = json.loads(src.read_text(encoding="utf-8"))
    chunks = data["chunks"]

    segments: list[dict] = []
    if args.transcript:
        segments = json.loads(Path(args.transcript).read_text(encoding="utf-8"))["segments"]
        print(f"已加载 {len(segments)} 条转写片段，用于精确定位")

    by_chapter: dict[int, list[dict]] = {}
    for c in chunks:
        by_chapter.setdefault(c.get("chapter", 0), []).append(c)

    key = load_key()
    if not key:
        raise SystemExit("没读到 DEEPSEEK_API_KEY，请检查 .env")
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=BASE_URL)

    chapters = []
    for idx in sorted(by_chapter):
        part = by_chapter[idx]
        print(f"章节 {idx}: {fmt(part[0]['start'])} - {fmt(part[-1]['end'])}，{len(part)} 块")
        s = summarize_chapter(client, part, segments)
        chapters.append(
            {
                "index": idx,
                "start": part[0]["start"],
                "end": part[-1]["end"],
                "time": f"{fmt(part[0]['start'])} - {fmt(part[-1]['end'])}",
                "title": s["title"] or f"第 {idx + 1} 段",
                "points": s["points"],
            }
        )

    print("生成总览…")
    result = {
        "video": data.get("video", ""),
        "duration": data.get("duration", 0),
        "overview": overview(client, chapters),
        "chapters": chapters,
    }

    out = Path(args.out) if args.out else Path("summaries") / src.name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out}")

    if args.md:
        md = out.with_suffix(".md")
        md.write_text(to_markdown(result), encoding="utf-8")
        print(f"已写入 {md}")

    exact = sum(1 for c in chapters for p in c["points"] if not p["approx"])
    total = sum(len(c["points"]) for c in chapters)
    print(f"\n精确命中 {exact}/{total} 条要点")
    print(f"\n总览：{result['overview']}\n")
    for c in chapters:
        print(f"[{c['time']}] {c['title']}")
        for p in c["points"]:
            mark = "≈" if p["approx"] else ""
            print(f"   - {p['text']}  ({mark}{p['time']})")


if __name__ == "__main__":
    main()
