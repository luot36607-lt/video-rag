"""检索评测：同一批问题，比较纯向量和向量+rerank。

判定：检索到的块与标注时间窗有重叠，且视频对得上，就算命中。

用法：
    python eval_retrieval.py
    python eval_retrieval.py --ref eval/video_ref.json --k 3
"""

import argparse
import json
from pathlib import Path

from common import RerankIndex, VectorIndex
from library import all_chunks, load_videos


def hit_rank(hits, chunks, ref, k):
    for rank, (_, i) in enumerate(hits[:k], 1):
        c = chunks[i]
        if c["video_id"] != ref["video"]:
            continue
        if c["start"] <= ref["end"] and c["end"] >= ref["start"]:
            return rank
    return 0


def evaluate(index, chunks, refs, k):
    ranks = [hit_rank(index.search(r["q"], k), chunks, r, k) for r in refs]
    hit = sum(1 for r in ranks if r)
    mrr = sum(1 / r for r in ranks if r) / len(ranks)
    return hit, mrr, ranks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="eval/video_ref.json")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--depth", type=int, default=10)
    args = ap.parse_args()

    refs = json.loads(Path(args.ref).read_text(encoding="utf-8"))
    chunks = all_chunks(load_videos())
    print(f"{len(chunks)} 块，{len(refs)} 个问题，k={args.k}\n")

    vi = VectorIndex(chunks)
    ri = RerankIndex(chunks, depth=args.depth)

    results = {}
    for name, idx in (("向量", vi), ("向量+rerank", ri)):
        hit, mrr, ranks = evaluate(idx, chunks, refs, args.k)
        results[name] = (hit, mrr, ranks)
        print(f"{name:<10} hit@{args.k} {hit}/{len(refs)}   MRR {mrr:.3f}")

    print("\n逐题对比（数字是命中名次，0 = 未命中）")
    print(f"{'问题':<28} {'向量':<8} {'rerank':<8}")
    for i, r in enumerate(refs):
        a, b = results["向量"][2][i], results["向量+rerank"][2][i]
        mark = " ↑" if b and (not a or b < a) else (" ↓" if a and (not b or b > a) else "")
        print(f"{r['q'][:26]:<28} {a or '-':<8} {b or '-':<8}{mark}")


if __name__ == "__main__":
    main()
