"""
视频问答：检索分块 -> 生成回答 -> 把引用映射回时间戳。

设计要点：**模型不生成时间戳**。它只输出 [1][2] 这样的引用编号，
时间戳由程序根据检索结果填回去——这样时间戳永远不会被编造。

用法：
    python video_ask.py chunks/semantic.json --q "LangChain 怎么调用通义千问？"
    python video_ask.py chunks/semantic.json              # 交互模式
    python video_ask.py chunks/semantic.json --q "..." --json
"""

import argparse
import json
import re
import time
from pathlib import Path

from common import BASE_URL, MODEL, RerankIndex, VectorIndex, load_key

SYSTEM = """你是视频内容问答助手。规则：
1. 只能依据【资料】回答，不要使用资料以外的知识。
2. 如果资料里没有答案，直接回答「视频里没有提到」，不要猜测。
3. 每条结论后面标注引用编号，如 [1]、[2]。
4. 不要自己写时间戳，时间戳由系统附加。
5. 用中文简洁回答。
"""


def fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def build_prompt(question: str, hits: list[tuple[float, dict]]) -> str:
    parts = []
    for i, (_, c) in enumerate(hits, 1):
        parts.append(f"[{i}] {fmt(c['start'])}-{fmt(c['end'])}\n{c['text']}")
    return "【资料】\n" + "\n\n".join(parts) + f"\n\n【问题】\n{question}"


def ask(
    client, question: str, chunks: list[dict], index: VectorIndex, k: int, min_score: float = 0.0
) -> dict:
    t0 = time.time()
    hits = [(score, chunks[i]) for score, i in index.search(question, k)]
    t_retrieve = (time.time() - t0) * 1000

    # 相似度太低 = 视频里大概率没有，直接拒答，省一次调用，也避免它硬编
    if not hits or hits[0][0] < min_score:
        return {
            "question": question,
            "answer": "视频里没有提到。",
            "sources": [],
            "top_score": round(float(hits[0][0]), 3) if hits else 0.0,
            "retrieve_ms": round(t_retrieve),
        }

    t1 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(question, hits)},
        ],
        temperature=0.2,
    )
    t_gen = (time.time() - t1) * 1000

    answer = resp.choices[0].message.content.strip()
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    sources = [
        {
            "index": n,
            "video_id": c.get("video_id", ""),
            "video_title": c.get("video_title", ""),
            "start": c["start"],
            "end": c["end"],
            "time": f"{fmt(c['start'])} - {fmt(c['end'])}",
            "score": round(float(score), 3),
            "snippet": c["text"][:80],
            "cited": n in cited,
        }
        for n, (score, c) in enumerate(hits, 1)
    ]
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieve_ms": round(t_retrieve),
        "generate_ms": round(t_gen),
        "tokens": resp.usage.total_tokens if resp.usage else None,
    }


def show(result: dict) -> None:
    print(f"\n问题：{result['question']}")
    print(f"\n回答：{result['answer']}")
    if result["sources"]:
        print("\n来源：")
        for s in result["sources"]:
            mark = "✓" if s["cited"] else " "
            print(f"  {mark} [{s['index']}] {s['time']}  相似度 {s['score']:.3f}")
            print(f"      {s['snippet']}…")
    if result.get("generate_ms"):
        print(
            f"\n  检索 {result['retrieve_ms']}ms · 生成 {result['generate_ms']}ms"
            f" · {result.get('tokens')} tokens"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks")
    ap.add_argument("--q", default="", help="问一次就退出")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--json", action="store_true", help="输出 JSON（给前端用）")
    ap.add_argument("--rerank", action="store_true", help="召回后用 reranker 精排")
    ap.add_argument(
        "--min-score", type=float, default=0.0,
        help="top1 相似度低于此值直接拒答，例如 0.55",
    )
    args = ap.parse_args()

    path = Path(args.chunks)
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = data["chunks"]

    print(f"视频：{Path(data.get('video', '')).name or path.stem}   共 {len(chunks)} 块")
    index = RerankIndex(chunks) if args.rerank else VectorIndex(chunks)

    key = load_key()
    if not key:
        raise SystemExit("没读到 DEEPSEEK_API_KEY，请检查 .env")
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=BASE_URL)

    if args.q:
        result = ask(client, args.q, chunks, index, args.k, args.min_score)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            show(result)
        return

    print("输入问题后回车，输入 q 退出。\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"q", "quit", "exit"}:
            break
        if q:
            show(ask(client, q, chunks, index, args.k, args.min_score))


if __name__ == "__main__":
    main()
