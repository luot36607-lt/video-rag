"""
ASR 语义纠错：把语音识别错的专有名词改回来。

分两层：
  1. 词典替换（默认，免费、确定性）—— 处理已经确认的错误
  2. LLM 上下文纠错（--llm，可选）—— 处理词典没覆盖的

用法：
    python asr_fix.py transcripts/xxx.json                 # 只用词典
    python asr_fix.py transcripts/xxx.json --llm           # 词典 + LLM
    python asr_fix.py transcripts/xxx.json --out xx.json

输出：transcripts/<视频名>.fixed.json（结构和输入一致）
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

# 按「长词优先」匹配，避免 LunchinCommunity 被 Lunchin 先替换掉
TERMS = {
    "fromLunchinCommunity": "from langchain_community",
    "LunchinCommunity": "langchain_community",
    "from浪欠community": "from langchain_community",
    "浪欠community": "langchain_community",
    "Lunchin": "LangChain",
    "浪欠": "LangChain",
    "郎倩": "LangChain",
    "欧拉马": "Ollama",
    "千万三max": "qwen-max",
    "q1-max": "qwen-max",
    "ipaq": "API Key",
    "in屋可": "invoke",
    "辨量": "变量",
    "bass LLM": "BaseLLM",
    "通易": "通义",
    "签问": "千问",
    # 上下文相关：同一个词在不同位置指的不是同一个模型
    "我们不用qwen-max": "我们不用qwen3-max",
    "因为qwen-max是聊天模型": "因为qwen3-max是聊天模型",
    "内队": "类对象",
    "帽号": "冒号",
    "防卫": "访问",
    "都给它保护好了": "都给它配置好了",
    "给它关闭的": "给它关掉",
    "这个是个内": "这个是个类",
    "class内": "class",
    "没什么区别直播": "没什么区别，直接",
    "导的内": "导的包",
    "PBT": "PPT",
    "APA代码": "API 代码",
    "大语言LMS": "大语言模型 LLM",
    "controller键": "Ctrl 键",
    "controller": "Ctrl",
}
_ORDERED = sorted(TERMS, key=len, reverse=True)


def fix_by_dict(text: str, hits: Counter) -> str:
    for bad in _ORDERED:
        if bad in text:
            hits[bad] += text.count(bad)
            text = text.replace(bad, TERMS[bad])
    return text


LLM_SYSTEM = """你是语音识别结果的校对员。输入是一段视频的 ASR 输出，可能有同音字或专有名词错误。

规则：
1. 只修正明显的识别错误：专有名词、技术术语、同音字、漏字、多字。
2. 不要改写句子、不要润色、不要改变技术含义；可以删掉明显的多余字。
3. 拿不准的地方保持原样。
4. 严格按输入条目的序号返回，条数必须一致。

输出 JSON 数组：[{"i": 0, "text": "修正后的文本"}, ...]"""


def fix_by_llm(rows: list[dict], batch: int = 40) -> tuple[list[dict], int]:
    from common import BASE_URL, MODEL, load_key

    key = load_key()
    if not key:
        raise SystemExit("没读到 DEEPSEEK_API_KEY，无法使用 --llm")

    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=BASE_URL)
    fixed, changed = 0, 0

    for start in range(0, len(rows), batch):
        part = rows[start : start + batch]
        payload = [{"i": i, "text": r["text"]} for i, r in enumerate(part)]
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": LLM_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0,
        )
        raw = resp.choices[0].message.content
        m = re.search(r"\[.*\]", raw, re.S)
        if not m:
            print(f"  批次 {start} 返回无法解析，跳过")
            continue
        try:
            out = json.loads(m.group())
        except json.JSONDecodeError:
            print(f"  批次 {start} JSON 解析失败，跳过")
            continue

        # 条数或序号对不上就整批丢弃，宁可少改也不改错
        if len(out) != len(part) or {o.get("i") for o in out} != set(range(len(part))):
            print(f"  批次 {start} 条数/序号不一致，跳过")
            continue

        for o in out:
            i = o["i"]
            new = (o.get("text") or "").strip()
            old = part[i]["text"]
            # 长度变化超过 60% 说明它在改写，不是纠错
            if new and 0.5 <= len(new) / max(1, len(old)) <= 1.6:
                if new != old:
                    changed += 1
                    part[i]["text"] = new
            fixed += 1

        print(f"  已处理 {min(start + batch, len(rows))}/{len(rows)}")

    return rows, changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("--out", default="")
    ap.add_argument("--llm", action="store_true", help="启用 LLM 上下文纠错")
    args = ap.parse_args()

    path = Path(args.transcript)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["segments"]

    hits: Counter = Counter()
    for r in rows:
        r["text"] = fix_by_dict(r["text"], hits)

    print(f"词典替换：{sum(hits.values())} 处")
    for bad, n in hits.most_common():
        print(f"  {bad:<24} -> {TERMS[bad]:<26} {n} 次")

    if args.llm:
        print("\nLLM 纠错中…")
        rows, changed = fix_by_llm(rows)
        print(f"LLM 修正 {changed} 段")

    out = Path(args.out) if args.out else path.with_suffix(".fixed.json")
    data["segments"] = rows
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")

    text = "".join(r["text"] for r in rows)
    print("\n前 6 段预览：")
    for r in rows[:6]:
        print(f"  [{r['start']:6.1f}] {r['text']}")


if __name__ == "__main__":
    main()
