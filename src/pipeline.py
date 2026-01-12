from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
import yt_dlp
from langchain_core.prompts import PromptTemplate

LOGGER = logging.getLogger("gemini_diarization")

PROMPT_TEMPLATE = PromptTemplate.from_template(
    """Role: Bạn là một chuyên gia về Audio Processing và Speech-to-Text.
Task: Phân tích file audio, thực hiện Speaker Diarization và Transcription đồng thời.

Yêu cầu kỹ thuật:
- Nhận diện chính xác thời điểm bắt đầu (start_time) và kết thúc (end_time) của mỗi phân đoạn.
- Gán Speaker ID nhất quán (SPEAKER_01, SPEAKER_02, ...).
- Nếu có khoảng trống thì bỏ qua, bắt đầu tính start_time từ lúc có giọng nói.
- Thời gian tính theo giây (float).
- Chỉ trả về kết quả, không giải thích.

Format Output (Nghiêm ngặt - RTTM Hybrid):
<file_id> <name or position of who representation> <start_time> <end_time> <transcript> <gender>

file_id: {file_id}
segment_offset: {segment_offset}
Yêu cầu:
- start_time/end_time tính theo thời gian trong đoạn audio hiện tại (0-based).
- Nếu không có giọng nói, trả về chuỗi rỗng.
- Nếu không xác định giới tính, ghi "unknown".
"""
)


@dataclass(frozen=True)
class Segment:
    start_offset: float
    path: Path


def sanitize_file_id(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_-]", "_", value)
    return value or "audio"


def resolve_api_key(cli_key: str | None) -> str:
    if cli_key:
        return cli_key
    for env_key in ("GEMINI_API_KEY", "VERTEX_API_KEY", "GOOGLE_API_KEY"):
        value = os.getenv(env_key)
        if value:
            return value
    raise RuntimeError("Missing API key. Set GEMINI_API_KEY or pass --api-key.")


def download_youtube_audio(url: str, output_dir: str | Path) -> tuple[Path, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    if info.get("_type") == "playlist":
        raise RuntimeError("Only single video URL is supported.")

    video_id = sanitize_file_id(str(info.get("id", "audio")))
    audio_path = output_dir / f"{video_id}.mp3"
    if not audio_path.exists():
        raise RuntimeError(f"Audio not found after download: {audio_path}")
    return audio_path, video_id


def get_duration_seconds(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def split_audio(path: Path, segment_seconds: int, work_dir: Path) -> list[Segment]:
    if segment_seconds <= 0:
        return [Segment(0.0, path)]

    duration = get_duration_seconds(path)
    if duration <= segment_seconds:
        return [Segment(0.0, path)]

    work_dir.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".mp3"
    base_name = path.stem
    pattern = work_dir / f"{base_name}_part_%03d{ext}"

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "segment",
            "-segment_time",
            str(segment_seconds),
            "-c",
            "copy",
            "-reset_timestamps",
            "1",
            str(pattern),
        ],
        check=True,
    )

    segments = sorted(work_dir.glob(f"{base_name}_part_*{ext}"))
    return [Segment(idx * float(segment_seconds), seg) for idx, seg in enumerate(segments)]


def detect_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "application/octet-stream"


def build_prompt(file_id: str, segment_offset: float) -> str:
    return PROMPT_TEMPLATE.format(file_id=file_id, segment_offset=f"{segment_offset:.2f}")


def call_gemini_stream(
    api_key: str,
    model: str,
    prompt: str,
    audio_path: Path,
    timeout_seconds: int = 120,
) -> str:
    url = (
        "https://aiplatform.googleapis.com/v1/publishers/google/models/"
        f"{model}:streamGenerateContent?key={api_key}"
    )
    audio_bytes = audio_path.read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": detect_mime_type(audio_path),
                            "data": audio_b64,
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 8192,
        },
    }

    text_chunks: list[str] = []
    with requests.post(url, json=payload, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data:"):
                line = line[len("data:") :].strip()
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            for candidate in event.get("candidates", []):
                content = candidate.get("content", {})
                for part in content.get("parts", []):
                    text = part.get("text")
                    if text:
                        text_chunks.append(text)

    return "".join(text_chunks).strip()


def normalize_output(text: str, file_id: str, offset: float) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        speaker = parts[1]
        start_raw = parts[2]
        end_raw = parts[3]
        gender = parts[-1]
        transcript = " ".join(parts[4:-1]).strip()
        try:
            start_val = float(start_raw) + offset
            end_val = float(end_raw) + offset
        except ValueError:
            continue
        line_out = (
            f"{file_id} {speaker} {start_val:.2f} {end_val:.2f} {transcript} {gender}"
        )
        lines.append(line_out.strip())
    return lines


def run_pipeline(
    *,
    api_key: str,
    model: str,
    audio_path: Path,
    file_id: str,
    segment_seconds: int,
) -> list[str]:
    file_id = sanitize_file_id(file_id)
    output_lines: list[str] = []

    with tempfile.TemporaryDirectory(prefix="segments_") as tmp_dir:
        segments = split_audio(audio_path, segment_seconds, Path(tmp_dir))
        for segment in segments:
            LOGGER.info("Processing segment at %.2fs: %s", segment.start_offset, segment.path)
            prompt = build_prompt(file_id, segment.start_offset)
            response_text = call_gemini_stream(
                api_key=api_key,
                model=model,
                prompt=prompt,
                audio_path=segment.path,
            )
            output_lines.extend(
                normalize_output(response_text, file_id=file_id, offset=segment.start_offset)
            )

    return output_lines


def ensure_logging() -> None:
    if LOGGER.handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def resolve_audio_input(
    youtube_url: str | None,
    audio_file: str | None,
    output_dir: str | Path,
    file_id: str | None,
) -> tuple[Path, str]:
    if youtube_url:
        audio_path, detected_id = download_youtube_audio(youtube_url, output_dir)
        return audio_path, file_id or detected_id
    if not audio_file:
        raise RuntimeError("Missing audio input.")
    audio_path = Path(audio_file).expanduser().resolve()
    if not audio_path.exists():
        raise RuntimeError(f"Audio file not found: {audio_path}")
    return audio_path, file_id or sanitize_file_id(audio_path.stem)


def format_output(lines: Iterable[str]) -> str:
    return "\n".join(line for line in lines if line.strip())
