"""
视频问答服务：网页播放器 + 问答 + 点击引用跳转。

启动：
    .venv\\Scripts\\python.exe -m uvicorn video_api:app --reload --port 8002

打开 http://127.0.0.1:8002/

视频路径默认指向本地文件，可以用环境变量覆盖：
    set VIDEO_PATH=D:\\path\\to\\video.mp4
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from common import BASE_URL, MODEL, VectorIndex, load_key
from video_ask import ask as ask_impl

CHUNKS_FILE = Path("chunks/semantic.json")
WEB_PAGE = Path("web/video.html")
DEFAULT_VIDEO = Path("data/video.mp4")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    data = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    chunks = data["chunks"]
    state["chunks"] = chunks
    state["index"] = VectorIndex(chunks)
    state["meta"] = {"video": data.get("video", ""), "duration": data.get("duration", 0)}

    key = load_key()
    if not key:
        raise RuntimeError("没读到 DEEPSEEK_API_KEY，请检查 .env")
    from openai import OpenAI

    state["client"] = OpenAI(api_key=key, base_url=BASE_URL)

    print(f"已加载 {len(chunks)} 块，服务就绪")
    yield
    state.clear()


app = FastAPI(title="视频问答", version="1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(3, ge=1, le=10)
    min_score: float = Field(0.0, ge=0.0, le=1.0)


@app.get("/")
def home() -> FileResponse:
    if not WEB_PAGE.exists():
        raise HTTPException(404, f"找不到 {WEB_PAGE}")
    return FileResponse(WEB_PAGE)


@app.get("/video")
def video() -> FileResponse:
    path = Path(os.environ.get("VIDEO_PATH") or DEFAULT_VIDEO)
    if not path.exists():
        raise HTTPException(404, f"找不到视频：{path}")
    return FileResponse(path, media_type="video/mp4")


@app.get("/info")
def info() -> dict:
    meta = state.get("meta", {})
    return {
        "chunks": len(state.get("chunks", [])),
        "duration": meta.get("duration", 0),
        "video": Path(meta.get("video", "")).name,
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    result = ask_impl(
        state["client"], req.question, state["chunks"], state["index"], req.k, req.min_score
    )
    return result
