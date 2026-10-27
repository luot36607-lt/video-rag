"""共享工具：模型配置、embedding、向量索引。

生成用 DeepSeek（OpenAI 兼容协议），检索用硅基流动的 bge-m3（免费）。
两个 key 都放在 .env 里。
"""

import hashlib
import json
import os
from pathlib import Path

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

EMBED_BASE_URL = "https://api.siliconflow.cn/v1"
EMBED_MODEL = "BAAI/bge-m3"
CACHE_FILE = Path("embed_cache.json")
BATCH = 32


def _env(name: str, default: str = "") -> str:
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ.get(name) or default


def load_key(name: str = "DEEPSEEK_API_KEY") -> str | None:
    return _env(name) or None


def _norm(v: list[float]) -> list[float]:
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def _key(text: str) -> str:
    return hashlib.sha1(f"{EMBED_MODEL}::{text}".encode()).hexdigest()


def _load_cache() -> dict[str, list[float]]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def embed_texts(texts: list[str]) -> list[list[float]]:
    """带本地缓存的批量 embedding：同样的文本不会重复花钱。"""
    key = load_key("SILICONFLOW_API_KEY") or load_key("EMBED_API_KEY")
    if not key:
        raise SystemExit(
            "没读到 SILICONFLOW_API_KEY。\n"
            "去 https://cloud.siliconflow.cn 注册，拿一个 key，写进 .env：\n"
            "SILICONFLOW_API_KEY=sk-xxx"
        )

    from openai import OpenAI

    cache = _load_cache()
    out: list[list[float] | None] = [None] * len(texts)
    todo: list[tuple[int, str]] = []

    for i, t in enumerate(texts):
        k = _key(t)
        if k in cache:
            out[i] = cache[k]
        else:
            todo.append((i, t))

    if todo:
        client = OpenAI(api_key=key, base_url=_env("EMBED_BASE_URL", EMBED_BASE_URL))
        print(f"  需要新算 {len(todo)} 个向量（其余走缓存）…")
        for start in range(0, len(todo), BATCH):
            batch = todo[start : start + BATCH]
            try:
                resp = client.embeddings.create(
                    model=_env("EMBED_MODEL", EMBED_MODEL), input=[t for _, t in batch]
                )
            except Exception as e:  # noqa: BLE001
                raise SystemExit(f"调用 embedding 失败：{e}") from e
            for (i, _), d in zip(batch, resp.data):
                v = _norm(d.embedding)
                cache[_key(texts[i])] = v
                out[i] = v
        CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")

    return [v for v in out if v is not None]


class VectorIndex:
    """接口和 BM25 一致：search(query, k) -> [(分数, 下标)]。"""

    def __init__(self, chunks: list[dict]):
        self.vectors = embed_texts([c["text"] for c in chunks])

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, int]]:
        q = embed_texts([query])[0]
        sims = [(sum(a * b for a, b in zip(q, v)), i) for i, v in enumerate(self.vectors)]
        sims.sort(reverse=True)
        return sims[:top_k]
