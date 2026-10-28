"""
多视频问答服务。

启动：
    .venv\\Scripts\\python.exe -m uvicorn video_api:app --reload --port 8002

打开 http://127.0.0.1:8002/

视频路径从 chunks/<id>.json 里读取，也可以用环境变量覆盖单个视频：
    set VIDEO_PATH=D:\\path\\to\\video.mp4
"""

import json
import os
import queue
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import store as db
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import ingest
from common import BASE_URL, RerankIndex, VectorIndex, load_key
from library import all_chunks, find_video, load_videos, summary_for
from video_ask import ask as ask_impl

WEB_PAGE = Path("web/video.html")
UPLOAD_DIR = Path("data/uploads")
state: dict = {}
JOB_Q: "queue.Queue[dict]" = queue.Queue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_path = Path(os.environ.get("DB_PATH", "videos.db"))
    use_db = db_path.exists() and os.environ.get("DB", "1") != "0"

    if use_db:
        conn = db.connect(str(db_path))
        base = db.DbIndex(conn)
        chunks = base.chunks
        videos = db.list_videos(conn)
        state["conn"] = conn
        print(f"存储：SQLite（{db_path}）")
    else:
        videos = load_videos()
        chunks = all_chunks(videos)
        base = VectorIndex(chunks)
        print("存储：JSON（没找到 videos.db，或设了 DB=0）")

    if not chunks:
        raise RuntimeError("没有可用内容，先跑转写和分块")

    state["videos"] = videos
    state["chunks"] = chunks
    state["index"] = RerankIndex(chunks, base=base) if os.environ.get("RERANK", "0") == "1" else base

    key = load_key()
    if not key:
        raise RuntimeError("没读到 DEEPSEEK_API_KEY，请检查 .env")
    from openai import OpenAI

    state["client"] = OpenAI(api_key=key, base_url=BASE_URL)

    print(f"已加载 {len(videos)} 个视频、{len(chunks)} 块，服务就绪")
    threading.Thread(target=_worker, daemon=True).start()
    yield
    state.clear()


app = FastAPI(title="视频问答", version="2.0", lifespan=lifespan)


def _reload_index() -> None:
    """导入新视频后刷新内存里的索引和视频列表。"""
    conn = state["conn"]
    idx = db.DbIndex(conn)
    state["index"] = RerankIndex(idx.chunks, base=idx) if os.environ.get("RERANK", "0") == "1" else idx
    state["chunks"] = idx.chunks
    state["videos"] = db.list_videos(conn)


def _worker() -> None:
    """单线程处理上传任务：转写是 CPU 密集的，并发跑只会互相拖慢。"""
    while True:
        job = JOB_Q.get()
        try:
            n = ingest.process(
                state["conn"], job["task_id"], job["vid"], job["path"], job["title"]
            )
            _reload_index()
            print(f"任务完成：{job['vid']}，{n} 块")
        except Exception as e:  # noqa: BLE001
            db.update_task(state["conn"], job["task_id"], stage="failed", error=str(e)[:300])
            print(f"任务失败：{e}")
        finally:
            JOB_Q.task_done()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(3, ge=1, le=10)
    min_score: float = Field(0.0, ge=0.0, le=1.0)


@app.get("/")
def home() -> FileResponse:
    if not WEB_PAGE.exists():
        raise HTTPException(404, f"找不到 {WEB_PAGE}")
    return FileResponse(WEB_PAGE)


@app.get("/videos")
def videos() -> list[dict]:
    return [
        {
            "id": v["id"],
            "title": v["title"],
            "duration": v["duration"],
            "chunks": v["chunks"] if isinstance(v["chunks"], int) else len(v["chunks"]),
        }
        for v in state.get("videos", [])
    ]


@app.get("/video/{vid}")
def video(vid: str) -> FileResponse:
    v = find_video(state.get("videos", []), vid)
    if not v:
        raise HTTPException(404, f"没有这个视频：{vid}")
    path = Path(os.environ.get("VIDEO_PATH") or v.get("video") or v.get("path") or "")
    if not path.exists():
        raise HTTPException(404, f"找不到视频文件：{path}")
    return FileResponse(path, media_type="video/mp4")


@app.get("/summary/{vid}")
def summary(vid: str) -> dict:
    return summary_for(vid)


@app.get("/info")
def info() -> dict:
    return {
        "videos": len(state.get("videos", [])),
        "chunks": len(state.get("chunks", [])),
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return ask_impl(
        state["client"], req.question, state["chunks"], state["index"], req.k, req.min_score
    )


@app.post("/ingest")
async def ingest_video(file: UploadFile = File(...), title: str = Form("")) -> dict:
    if "conn" not in state:
        raise HTTPException(503, "上传需要数据库模式：先跑一次 migrate_to_db.py")

    vid = uuid.uuid4().hex[:8]
    suffix = Path(file.filename or "").suffix or ".mp4"
    dest = UPLOAD_DIR / f"{vid}{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    size = 0
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
            size += len(chunk)
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "空文件")

    task_id = uuid.uuid4().hex[:8]
    name = (title or file.filename or vid).strip()
    db.create_task(state["conn"], task_id, vid, name)
    JOB_Q.put({"task_id": task_id, "vid": vid, "path": str(dest), "title": name})
    print(f"收到上传：{name}（{size / 1e6:.1f} MB），任务 {task_id}")
    return {"task_id": task_id, "video_id": vid, "bytes": size}


@app.get("/task/{task_id}")
def task(task_id: str) -> dict:
    t = db.get_task(state["conn"], task_id)
    if not t:
        raise HTTPException(404, "没有这个任务")
    return t
