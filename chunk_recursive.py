"""
递归分块：章节 → 段落 → 句子 → 字符，逐级拆，保证每块语义完整。

说明：视频转写里没有真正的「章节标题」，只有连续的口语。所以「章节」是
用停顿 + 话题标记词检测出来的伪章节；如果你以后能拿到视频的章节信息
（比如 PPT 翻页时间），替换 detect_chapters 里的边界即可。

用法：
    python chunk_recursive.py transcripts/xxx.fixed.json
    python chunk_recursive.py transcripts/xxx.fixed.json --size 350 --no-overlap

输出：chunks/<视频名>.json
"""

import argparse
import json
import re
from pathlib import Path

SENT_END = "。！？!?；;…"
# 出现这些词，大概率是新话题的开始（必须配合足够长度才切）
MARKERS = ("接下来", "下面我们", "我们来看", "首先", "最后", "好的", "第二个", "第二点", "然后我们")

PARA_GAP = 0.6       # 句子间停顿超过这个值 -> 新段落
CHAPTER_GAP = 1.5    # 段落间停顿超过这个值 -> 新章节
PARA_CHARS = 150
CHAPTER_CHARS = 900
MIN_CHAPTER_SEC = 90


def split_sentences(seg: dict) -> list[dict]:
    """把一个 ASR 段按标点切成句子。

    时间戳按字数比例分配——ASR 只给到段级，做不到句级精确。
    """
    text, start, end = seg["text"], seg["start"], seg["end"]
    parts = [p.strip() for p in re.findall(rf"[^{SENT_END}]+[{SENT_END}]?", text)]
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return [{"text": text, "start": start, "end": end}]

    total = sum(len(p) for p in parts) or 1
    out, t = [], start
    for p in parts:
        d = (end - start) * len(p) / total
        out.append({"text": p, "start": round(t, 2), "end": round(t + d, 2)})
        t += d
    return out


def chars(para: list[dict]) -> int:
    return sum(len(s["text"]) for s in para)


def text_of(para: list[dict]) -> str:
    """拼接句子；上一句没有标点结尾时补一个空格，避免两句话粘在一起。"""
    out = ""
    for s in para:
        t = s["text"]
        if out and out[-1] not in SENT_END:
            out += " "
        out += t
    return out


def span(para: list[dict]) -> tuple[float, float]:
    return para[0]["start"], para[-1]["end"]


def build_sentences(segs: list[dict]) -> list[dict]:
    out = []
    for s in segs:
        out.extend(split_sentences(s))
    return out


def build_paragraphs(sentences: list[dict]) -> list[list[dict]]:
    paras, cur = [], []
    for s in sentences:
        if cur and (s["start"] - cur[-1]["end"] > PARA_GAP or chars(cur) + len(s["text"]) > PARA_CHARS):
            paras.append(cur)
            cur = []
        cur.append(s)
    if cur:
        paras.append(cur)
    return paras


def detect_chapters(paragraphs: list[list[dict]]) -> list[list[list[dict]]]:
    chapters, cur = [], []
    for p in paragraphs:
        if cur:
            gap = p[0]["start"] - cur[-1][-1]["end"]
            long_enough = span(cur[-1])[1] - span(cur[0])[0] > MIN_CHAPTER_SEC
            marker = text_of(p).startswith(MARKERS)
            if gap > CHAPTER_GAP or (marker and long_enough) or chars(sum(cur, [])) > CHAPTER_CHARS:
                chapters.append(cur)
                cur = []
        cur.append(p)
    if cur:
        chapters.append(cur)
    return chapters


def semantic_chapters(paragraphs: list[list[dict]], min_sec: int = 90, percentile: int = 25):
    """用相邻段落的语义相似度找话题边界。

    相似度明显下降的地方，通常就是换话题了——比按字数硬切靠谱。
    """
    from common import embed_texts

    vecs = embed_texts([text_of(p) for p in paragraphs])
    sims = [sum(a * b for a, b in zip(vecs[i], vecs[i + 1])) for i in range(len(vecs) - 1)]
    if not sims:
        return [paragraphs]

    thr = sorted(sims)[max(0, int(len(sims) * percentile / 100) - 1)]
    print(f"  相似度阈值 {thr:.3f}（第 {percentile} 百分位），最低 {min(sims):.3f}，最高 {max(sims):.3f}")

    chapters, cur = [], []
    for i, p in enumerate(paragraphs):
        if cur:
            dur = p[0]["start"] - cur[0][0]["start"]
            if sims[i - 1] <= thr and dur >= min_sec:
                chapters.append(cur)
                cur = []
        cur.append(p)
    if cur:
        chapters.append(cur)
    return chapters


def merge_tiny(chunks: list[dict], min_chars: int = 100) -> list[dict]:
    """把尾巴上过短的块并到前一块，避免出现「17 个字」这种碎片。"""
    out: list[dict] = []
    for c in chunks:
        if out and len(c["text"]) < min_chars:
            prev = out[-1]
            prev["text"] = prev["text"] + " " + c["text"]
            prev["end"] = c["end"]
            prev["sentences"] += c["sentences"]
        else:
            out.append(c)
    return out


def pack_chunks(chapters, size: int, overlap: bool) -> list[dict]:
    chunks = []
    for cid, paras in enumerate(chapters):
        cur: list[list[dict]] = []
        for p in paras:
            pieces = [p]
            # 单个段落超长：只能按句子再切
            if chars(p) > size:
                pieces, buf = [], []
                for s in p:
                    if buf and chars(buf) + len(s["text"]) > size:
                        pieces.append(buf)
                        buf = []
                    buf.append(s)
                if buf:
                    pieces.append(buf)

            for piece in pieces:
                if cur and chars(sum(cur, [])) + chars(piece) > size:
                    chunks.append(make_chunk(cur, cid))
                    cur = [[cur[-1][-1]]] if overlap else []
                cur.append(piece)
        if cur:
            chunks.append(make_chunk(cur, cid))

    for i, c in enumerate(chunks):
        c["id"] = i
    return chunks


def make_chunk(paras: list[list[dict]], chapter: int) -> dict:
    flat = sum(paras, [])
    return {
        "chapter": chapter,
        "text": text_of(flat),
        "start": flat[0]["start"],
        "end": flat[-1]["end"],
        "sentences": len(flat),
    }


def fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("--size", type=int, default=350, help="每块目标字数")
    ap.add_argument("--no-overlap", action="store_true")
    ap.add_argument("--semantic", action="store_true", help="用 embedding 检测话题边界（需要 embedding key）")
    ap.add_argument("--out", default="")
    ap.add_argument("--id", default="", help="视频 ID（默认用文件名），多视频时建议显式指定")
    ap.add_argument("--title", default="", help="显示用的标题")
    args = ap.parse_args()

    path = Path(args.transcript)
    data = json.loads(path.read_text(encoding="utf-8"))

    sentences = build_sentences(data["segments"])
    paragraphs = build_paragraphs(sentences)
    if args.semantic:
        chapters = semantic_chapters(paragraphs)
        method = "语义边界"
    else:
        chapters = detect_chapters(paragraphs)
        method = "停顿 + 长度"
    chunks = pack_chunks(chapters, args.size, not args.no_overlap)
    chunks = merge_tiny(chunks)
    for i, c in enumerate(chunks):
        c["id"] = i

    print(f"句子 {len(sentences)} -> 段落 {len(paragraphs)} -> 章节 {len(chapters)}（{method}）-> 块 {len(chunks)}")

    print("\n章节划分：")
    for i, c in enumerate(chapters):
        flat = sum(c, [])
        print(f"  #{i}  {fmt(flat[0]['start'])} - {fmt(flat[-1]['end'])}  {len(flat)}句  {text_of(flat)[:36]}")

    print("\n块统计：")
    lens = [len(c["text"]) for c in chunks]
    durs = [c["end"] - c["start"] for c in chunks]
    print(f"  字数 平均 {sum(lens)/len(lens):.0f}，最短 {min(lens)}，最长 {max(lens)}")
    print(f"  时长 平均 {sum(durs)/len(durs):.0f}s，最短 {min(durs):.0f}s，最长 {max(durs):.0f}s")
    print(f"  跨章节的块：{sum(1 for i in range(1, len(chunks)) if chunks[i]['chapter'] != chunks[i-1]['chapter'])} 个章节边界")

    vid = args.id or path.stem
    out = Path(args.out) if args.out else Path("chunks") / f"{vid}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "id": vid,
                "title": args.title or vid,
                "video": data.get("video", ""),
                "duration": data.get("duration", 0),
                "chunks": chunks,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n已写入 {out}")

    print("\n前 3 块：")
    for c in chunks[:3]:
        print(f"\n  [{c['id']}] {fmt(c['start'])} - {fmt(c['end'])}  {len(c['text'])}字")
        print(f"      {c['text'][:120]}...")


if __name__ == "__main__":
    main()
