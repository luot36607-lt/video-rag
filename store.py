"""存储层：SQLite + sqlite-vec。

一个库同时管元数据和向量：
    videos      视频信息
    chunks      文本块（含时间戳）
    chunk_vecs  vec0 虚表，rowid = chunks.id，按余弦距离检索

为什么是 SQLite 而不是 Postgres：
    这台机器没开虚拟化（Docker/WSL2 起不来），Postgres 又要单独装。
    SQLite 零运维，但表结构、约束、UPSERT、WHERE 过滤这些概念完全一样，
    以后换 Postgres + pgvector 主要是换驱动和把 MATCH 换成 `<=>`。
"""

import hashlib
import sqlite3
import threading
from pathlib import Path

import sqlite_vec

DIM = 1024  # bge-m3 的向量维度，换模型要重建表

# FastAPI 的同步接口跑在线程池里，SQLite 连接默认不能跨线程用
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
  id         TEXT PRIMARY KEY,
  title      TEXT NOT NULL,
  path       TEXT,
  duration   REAL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  video_id  TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  chapter   INTEGER,
  start_sec REAL NOT NULL,
  end_sec   REAL NOT NULL,
  text      TEXT NOT NULL,
  hash      TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_chunks_video ON chunks(video_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vecs USING vec0(
  video_id text,
  embedding float[1024] distance_metric=cosine
);

CREATE TABLE IF NOT EXISTS tasks (
  id         TEXT PRIMARY KEY,
  video_id   TEXT,
  title      TEXT,
  stage      TEXT,
  progress   REAL DEFAULT 0,
  error      TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT
);
"""


def connect(path: str = "videos.db") -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def upsert_video(conn: sqlite3.Connection, vid: str, title: str, path: str, duration: float) -> None:
    conn.execute(
        """INSERT INTO videos(id, title, path, duration) VALUES (?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET title=excluded.title, path=excluded.path,
                                         duration=excluded.duration""",
        (vid, title, path, duration),
    )


def delete_video(conn: sqlite3.Connection, vid: str) -> None:
    """重导一个视频前先清干净，避免留下旧块。"""
    ids = [r["id"] for r in conn.execute("SELECT id FROM chunks WHERE video_id=?", (vid,))]
    for cid in ids:
        conn.execute("DELETE FROM chunk_vecs WHERE rowid=?", (cid,))
    conn.execute("DELETE FROM chunks WHERE video_id=?", (vid,))
    conn.execute("DELETE FROM videos WHERE id=?", (vid,))


def insert_chunks(conn: sqlite3.Connection, vid: str, chunks: list[dict], vectors: list[list[float]]) -> int:
    """插入块和向量，hash 相同的跳过。返回实际新增的条数。"""
    added = 0
    for c, vec in zip(chunks, vectors):
        cur = conn.execute(
            """INSERT OR IGNORE INTO chunks(video_id, chapter, start_sec, end_sec, text, hash)
               VALUES (?,?,?,?,?,?)""",
            (vid, c.get("chapter", 0), c["start"], c["end"], c["text"], _hash(c["text"])),
        )
        if cur.rowcount == 0:
            continue
        conn.execute(
            "INSERT INTO chunk_vecs(rowid, video_id, embedding) VALUES (?,?,?)",
            (cur.lastrowid, vid, sqlite_vec.serialize_float32(vec)),
        )
        added += 1
    return added


def load_chunks(conn: sqlite3.Connection) -> list[dict]:
    """把块的元数据读进内存（不含向量），用于展示和引用。"""
    rows = conn.execute(
        """SELECT c.id AS db_id, c.video_id, c.chapter, c.start_sec, c.end_sec, c.text,
                  v.title AS video_title, v.duration
           FROM chunks c JOIN videos v ON v.id = c.video_id
           ORDER BY c.video_id, c.start_sec"""
    )
    return [
        {
            "db_id": r["db_id"],
            "video_id": r["video_id"],
            "video_title": r["video_title"],
            "chapter": r["chapter"],
            "start": r["start_sec"],
            "end": r["end_sec"],
            "text": r["text"],
        }
        for r in rows
    ]


def search_chunks(
    conn: sqlite3.Connection, query_vec: list[float], k: int = 3, video_id: str | None = None
) -> list[sqlite3.Row]:
    sql = """
        SELECT c.id, c.video_id, c.start_sec, c.end_sec, c.text, v.title, cv.distance
        FROM chunk_vecs cv
        JOIN chunks c ON c.id = cv.rowid
        JOIN videos v ON v.id = c.video_id
        WHERE cv.embedding MATCH ? AND k = ?
    """
    params: list = [sqlite_vec.serialize_float32(query_vec), k]
    if video_id:
        sql += " AND cv.video_id = ?"
        params.append(video_id)
    sql += " ORDER BY cv.distance"
    with _lock:
        return list(conn.execute(sql, params))


def stats(conn: sqlite3.Connection) -> dict:
    with _lock:
        videos = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return {"videos": videos, "chunks": chunks}


def list_videos(conn: sqlite3.Connection) -> list[dict]:
    with _lock:
        rows = conn.execute(
            """SELECT v.id, v.title, v.duration, v.path,
                      (SELECT COUNT(*) FROM chunks c WHERE c.video_id = v.id) AS chunks
               FROM videos v ORDER BY v.id"""
        )
        return [dict(r) for r in rows]


def create_task(conn: sqlite3.Connection, task_id: str, video_id: str, title: str) -> None:
    with _lock:
        conn.execute(
            "INSERT INTO tasks(id, video_id, title, stage, progress) VALUES (?,?,?,?,0)",
            (task_id, video_id, title, "queued"),
        )
        conn.commit()


def update_task(
    conn: sqlite3.Connection,
    task_id: str,
    stage: str | None = None,
    progress: float | None = None,
    error: str | None = None,
) -> None:
    sets, params = [], []
    if stage is not None:
        sets.append("stage=?")
        params.append(stage)
    if progress is not None:
        sets.append("progress=?")
        params.append(float(progress))
    if error is not None:
        sets.append("error=?")
        params.append(error)
    if not sets:
        return
    sets.append("updated_at=CURRENT_TIMESTAMP")
    params.append(task_id)
    with _lock:
        conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id=?", params)
        conn.commit()


def get_task(conn: sqlite3.Connection, task_id: str) -> dict | None:
    with _lock:
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    return dict(row) if row else None


class DbIndex:
    """和 VectorIndex 一样的接口，但检索走数据库。"""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.chunks = load_chunks(conn)
        self.by_id = {c["db_id"]: i for i, c in enumerate(self.chunks)}

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, int]]:
        from common import embed_texts

        vec = embed_texts([query])[0]
        rows = search_chunks(self.conn, vec, top_k)
        return [(1 - r["distance"], self.by_id[r["id"]]) for r in rows if r["id"] in self.by_id]
