"""
视频转写：抽音频 -> 语音识别 -> 带时间戳的文本

用法：
    python video_transcribe.py "D:/path/video.mp4"
    python video_transcribe.py "D:/path/video.mp4" --limit 120
    python video_transcribe.py "D:/path/video.mp4" --model medium

输出：transcripts/<id>.json
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

DEFAULT_PROMPT = "LangChain, RAG, 向量, 余弦相似度, 大语言模型, API, 提示词, 嵌入, 检索增强生成"


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


def transcribe(
    video: str | Path,
    model_name: str = "small",
    prompt: str = DEFAULT_PROMPT,
    limit: float = 0,
    on_progress=None,
) -> dict:
    """转写一个视频，返回 {duration, language, segments}。

    on_progress(p) 会在转写过程中被调用，p 是 0~1 的进度，用于上传后的进度条。
    """
    path = Path(video)
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    duration = probe_duration(path)
    audio = load_audio(path, limit)
    total = max(1e-6, len(audio) / 16000)

    from faster_whisper import WhisperModel

    model = WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        cpu_threads=max(4, (os.cpu_count() or 4)),
    )
    segments, info = model.transcribe(
        audio,
        language="zh",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
        initial_prompt=prompt or None,
    )

    rows = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            rows.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text})
        if on_progress and len(rows) % 5 == 0:
            on_progress(min(0.99, seg.end / total))

    if on_progress:
        on_progress(1.0)

    return {
        "duration": round(duration, 2),
        "language": info.language,
        "segments": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="small", help="tiny / base / small / medium")
    ap.add_argument("--limit", type=float, default=0, help="只转前 N 秒，0 表示全部")
    ap.add_argument("--out", default="")
    ap.add_argument("--id", default="", help="视频 ID（默认用文件名）")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="领域提示词，传空字符串关闭")
    args = ap.parse_args()

    path = Path(args.video)
    t0 = time.time()
    print(f"读取音频：{path.name}")

    def show(p: float) -> None:
        if int(p * 100) % 10 == 0:
            print(f"  {p * 100:.0f}%")

    doc = transcribe(path, args.model, args.prompt, args.limit, on_progress=show)
    rows = doc["segments"]
    print(f"完成：{len(rows)} 段，耗时 {time.time() - t0:.1f}s")

    vid = args.id or path.stem
    out = Path(args.out) if args.out else Path("transcripts") / f"{vid}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "id": vid,
                "video": str(path),
                "duration": doc["duration"],
                "language": doc["language"],
                "segments": rows,
            },
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
