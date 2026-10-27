"""多视频库：把 chunks/ 和 summaries/ 里的所有视频拼成一个可检索的集合。

一个视频一个文件，文件名就是视频 ID：
    chunks/langchain-llm.json
    summaries/langchain-llm.json
"""

import json
from pathlib import Path

CHUNK_DIR = Path("chunks")
SUMMARY_DIR = Path("summaries")


def load_videos(chunk_dir: Path = CHUNK_DIR) -> list[dict]:
    videos = []
    for f in sorted(chunk_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        vid = data.get("id") or f.stem
        title = data.get("title") or vid
        chunks = data.get("chunks", [])
        for c in chunks:
            c["video_id"] = vid
            c["video_title"] = title
        videos.append(
            {
                "id": vid,
                "title": title,
                "video": data.get("video", ""),
                "duration": data.get("duration", 0),
                "chunks": chunks,
            }
        )
    return videos


def all_chunks(videos: list[dict]) -> list[dict]:
    return [c for v in videos for c in v["chunks"]]


def summary_for(vid: str, summary_dir: Path = SUMMARY_DIR) -> dict:
    path = summary_dir / f"{vid}.json"
    if not path.exists():
        return {"overview": "", "chapters": []}
    return json.loads(path.read_text(encoding="utf-8"))


def find_video(videos: list[dict], vid: str) -> dict | None:
    return next((v for v in videos if v["id"] == vid), None)
