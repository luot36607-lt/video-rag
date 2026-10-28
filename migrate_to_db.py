"""把 chunks/*.json 导入 SQLite（可重复执行，同一视频会先清空再导入）。

用法：
    python migrate_to_db.py
    python migrate_to_db.py --db videos.db --chunks chunks
"""

import argparse
import json
from pathlib import Path

import store
from common import embed_texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="videos.db")
    ap.add_argument("--chunks", default="chunks")
    args = ap.parse_args()

    files = sorted(Path(args.chunks).glob("*.json"))
    if not files:
        raise SystemExit(f"{args.chunks}/ 里没有 JSON")

    conn = store.connect(args.db)
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        vid = d.get("id") or f.stem
        title = d.get("title") or vid
        chunks = d["chunks"]

        store.delete_video(conn, vid)
        store.upsert_video(conn, vid, title, d.get("video", ""), d.get("duration", 0))
        vectors = embed_texts([c["text"] for c in chunks])
        added = store.insert_chunks(conn, vid, chunks, vectors)
        conn.commit()
        print(f"{vid:<16} {added:>3} 块  {title}")

    s = store.stats(conn)
    print(f"\n数据库：{args.db}   视频 {s['videos']} 个，块 {s['chunks']} 条")


if __name__ == "__main__":
    main()
