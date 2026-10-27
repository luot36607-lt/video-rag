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
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from common import BASE_URL, VectorIndex, load_key
from library import all_chunks, find_video, load_videos, summary_for
from video_ask import ask as ask_impl

WEB_PAGE = Path("web/video.html")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    videos = load_videos()
    if not videos:
        raise RuntimeError("chunks/ 里没有视频，先跑转写和分块")

    chunks = all_chunks(videos)
    state["videos"] = videos
    state["chunks"] = chunks
    state["index"] = VectorIndex(chunks)

    key = load_key()
    if not key:
        raise RuntimeError("没读到 DEEPSEEK_API_KEY，请检查 .env")
    from openai import OpenAI

    state["client"] = OpenAI(api_key=key, base_url=BASE_URL)

    print(f"已加载 {len(videos)} 个视频、{len(chunks)} 块，服务就绪")
    yield
    state.clear()


app = FastAPI(title="视频问答", version="2.0", lifespan=lifespan)


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
            "chunks": len(v["chunks"]),
        }
        for v in state.get("videos", [])
    ]


@app.get("/video/{vid}")
def video(vid: str) -> FileResponse:
    v = find_video(state.get("videos", []), vid)
    if not v:
        raise HTTPException(404, f"没有这个视频：{vid}")
    path = Path(os.environ.get("VIDEO_PATH") or v.get("video") or "")
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
