"""
视频转写：抽音频 -> 语音识别 -> 带时间戳的文本

用法：
    python video_transcribe.py "D:/path/video.mp4"
    python video_transcribe.py "D:/path/video.mp4" --limit 120   # 只转前 120 秒
    python video_transcribe.py "D:/path/video.mp4" --model medium

输出：transcripts/<视频名>.json
    [{"start": 12.3, "end": 18.9, "text": "..."}, ...]
"""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

# 国内访问 HuggingFace 经常不通，换成镜像（必须在 import 之前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_duration(path: Path) -> float:
    """ffmpeg 的 -i 会打印时长后退出（返回码非 0），这里只解析那一行。"""
    out = subprocess.run(
        [_ffmpeg(), "-i", str(path)], capture_output=True, text=True, errors="ignore"
    ).stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", out)
    if not m:
        return 0.0
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


def load_audio(path: Path, limit: float = 0) -> np.ndarray:
    """用 ffmpeg 直接解码成 16k 单声道 float32，绕开 PyAV 的版本兼容问题。"""
    cmd = [
        _ffmpeg(), "-nostdin", "-v", "error", "-i", str(path),
        "-vn", "-ac", "1", "-ar", "16000",
    ]
    if limit:
        cmd += ["-t", str(limit)]
    cmd += ["-f", "s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise SystemExit("ffmpeg 解码失败：\n" + proc.stderr.decode("utf-8", "ignore")[-800:])
    return np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0


def fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="small", help="tiny / base / small / medium")
    ap.add_argument("--limit", type=float, default=0, help="只转前 N 秒，0 表示全部")
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--prompt",
        default="LangChain, RAG, 向量, 余弦相似度, 大语言模型, API, 提示词, 嵌入, 检索增强生成",
        help="给模型的领域提示词，能显著改善专有名词识别；传空字符串关闭",
    )
    args = ap.parse_args()

    path = Path(args.video)
    if not path.exists():
        raise SystemExit(f"找不到文件：{path}")

    t0 = time.time()
    print(f"读取音频：{path.name}")
    duration = probe_duration(path)
    audio = load_audio(path, args.limit)
    print(
        f"总时长 {fmt(duration)}，本次转写 {fmt(len(audio) / 16000)}"
        f"（解码耗时 {time.time() - t0:.1f}s）"
    )

    from faster_whisper import WhisperModel

    print(f"加载模型 {args.model}（首次会下载，之后走缓存）…")
    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8",
        cpu_threads=max(4, (os.cpu_count() or 4)),
    )

    t1 = time.time()
    segments, info = model.transcribe(
        audio,
        language="zh",
        vad_filter=True,                      # 跳过静音，减少幻觉
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
        initial_prompt=args.prompt or None,   # 领域词表，明显改善专有名词
    )

    rows = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            rows.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text})
        if len(rows) % 20 == 0 and rows:
            print(f"  {fmt(seg.end)}  {text[:40]}")

    print(f"完成：{len(rows)} 段，转写耗时 {time.time() - t1:.1f}s")

    out = Path(args.out) if args.out else Path("transcripts") / f"{path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"video": str(path), "duration": round(duration, 2), "language": info.language,
             "segments": rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"已写入 {out}")

    chars = sum(len(r["text"]) for r in rows)
    print(f"\n平均每段 {chars / max(1, len(rows)):.1f} 字，共 {chars} 字")
    print("\n前 8 段预览：")
    for r in rows[:8]:
        print(f"  [{fmt(r['start'])} - {fmt(r['end'])}] {r['text']}")


if __name__ == "__main__":
    main()
