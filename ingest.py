"""上传后的处理流水线：转写 → 纠错 → 分块 → 摘要 → 入库。

每一步都会更新 tasks 表，前端轮询 /task/{id} 就能看到进度。
"""

import json
from collections import Counter
from pathlib import Path

import store
from asr_fix import fix_by_dict, fix_by_llm
from chunk_recursive import (
    build_paragraphs,
    build_sentences,
    merge_tiny,
    pack_chunks,
    semantic_chapters,
)
from common import BASE_URL, embed_texts, load_key
from video_summary import overview, summarize_chapter
from video_transcribe import fmt, transcribe

TRANSCRIPTS = Path("transcripts")
CHUNKS = Path("chunks")
SUMMARIES = Path("summaries")


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def process(
    conn, task_id: str, vid: str, video_path: str, title: str, size: int = 350
) -> int:
    def update(stage: str, progress: float) -> None:
        store.update_task(conn, task_id, stage=stage, progress=progress)

    # 1. 转写（占总时间的大头，进度按音频位置走）
    update("transcribing", 0.01)
    doc = transcribe(video_path, on_progress=lambda p: update("transcribing", 0.01 + 0.62 * p))
    doc.update(id=vid, video=str(video_path))
    _write(TRANSCRIPTS / f"{vid}.json", doc)

    # 2. 纠错
    update("fixing", 0.66)
    hits: Counter = Counter()
    for s in doc["segments"]:
        s["text"] = fix_by_dict(s["text"], hits)
    doc["segments"], _ = fix_by_llm(doc["segments"])
    _write(TRANSCRIPTS / f"{vid}.fixed.json", doc)

    # 3. 分块
    update("chunking", 0.76)
    sentences = build_sentences(doc["segments"])
    paragraphs = build_paragraphs(sentences)
    chapters = semantic_chapters(paragraphs)
    chunks = merge_tiny(pack_chunks(chapters, size, True))
    for i, c in enumerate(chunks):
        c["id"] = i
    _write(
        CHUNKS / f"{vid}.json",
        {
            "id": vid,
            "title": title,
            "video": str(video_path),
            "duration": doc["duration"],
            "chunks": chunks,
        },
    )

    # 4. 摘要
    update("summarizing", 0.86)
    from openai import OpenAI

    client = OpenAI(api_key=load_key(), base_url=BASE_URL)

    by_chapter: dict[int, list[dict]] = {}
    for c in chunks:
        by_chapter.setdefault(c.get("chapter", 0), []).append(c)

    out_chapters = []
    for n, idx in enumerate(sorted(by_chapter)):
        part = by_chapter[idx]
        s = summarize_chapter(client, part, doc["segments"])
        out_chapters.append(
            {
                "index": idx,
                "start": part[0]["start"],
                "end": part[-1]["end"],
                "time": f"{fmt(part[0]['start'])} - {fmt(part[-1]['end'])}",
                "title": s["title"] or f"第 {idx + 1} 段",
                "points": s["points"],
            }
        )
        update("summarizing", 0.86 + 0.1 * (n + 1) / max(1, len(by_chapter)))

    _write(
        SUMMARIES / f"{vid}.json",
        {
            "id": vid,
            "title": title,
            "video": str(video_path),
            "duration": doc["duration"],
            "overview": overview(client, out_chapters),
            "chapters": out_chapters,
        },
    )

    # 5. 入库
    update("indexing", 0.97)
    store.delete_video(conn, vid)
    store.upsert_video(conn, vid, title, str(video_path), doc["duration"])
    vectors = embed_texts([c["text"] for c in chunks])
    store.insert_chunks(conn, vid, chunks, vectors)
    conn.commit()

    update("done", 1.0)
    return len(chunks)
