from __future__ import annotations

import base64
import json
import logging
import math
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
- Nếu có khoảng trống thì bỏ qua, bắt đầu tính start_time từ lúc có giọng nói.
- Thời gian tính theo giây (float).
- Chỉ trả về kết quả, không giải thích.

Format Output (Nghiêm ngặt - RTTM Hybrid):
<file_id> <name or position of who representation> <start_time> <end_time> <transcript> <gender>
Ví dụ: audio123 speaker_1_Nguyễn Văn A_Phó chủ tịch 0.00 5.23 "Xin chào mọi người" nam
Ví dụ: audio123 speaker_2__đại biểu 5.50 10.75 "Hôm nay chúng ta họp về dự án mới" nữ
...
file_id: {file_id}
segment_offset: {segment_offset}
Yêu cầu:
- start_time/end_time tính theo thời gian trong đoạn audio hiện tại (0-based).
- Nếu không có giọng nói, trả về chuỗi rỗng.
- Nếu không xác định giới tính, ghi "unknown".
- identifying distinct speakers
- Skip các đoạn quảng cáo, hát, ... và chỉ tập trung vào phần hội thoại chính.
- Không skip các đoạn quan trọng, trong lời thoại
- Không tách diarization theo câu mà theo người nói, ví dụ, nếu người nói ngắt quãng hoặc hết câu, vẫn tiếp tục tracking người đó cho đến khi người khác nói.
"""
)

TIME_RE = re.compile(
    r"(?:^|\s)(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?|\d+(?:\.\d+)?)"
)

GENDER_TOKENS = {
    "male": "male",
    "female": "female",
    "unknown": "unknown",
    "nam": "nam",
    "nu": "nu",
    "nữ": "nữ",
}


@dataclass(frozen=True)
class Segment:
    start_offset: float
    path: Path


@dataclass(frozen=True)
class UsageSummary:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost_usd: float | None = None
    output_cost_usd: float | None = None
    total_cost_usd: float | None = None
    pricing_label: str | None = None


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


def parse_time_value(value: str) -> float | None:
    value = value.strip().replace(",", ".")
    if ":" in value:
        parts = value.split(":")
        if len(parts) == 2:
            minutes, seconds = parts
            return float(minutes) * 60 + float(seconds)
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        return None
    try:
        return float(value)
    except ValueError:
        return None


def seconds_to_hms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    total_sec, ms = divmod(total_ms, 1000)
    hours, rem = divmod(total_sec, 3600)
    minutes, sec = divmod(rem, 60)
    if ms:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}.{ms:03d}"
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def format_timestamp(seconds: float, time_format: str) -> str:
    if time_format == "seconds":
        return f"{seconds:.2f}"
    if time_format == "hms":
        return seconds_to_hms(seconds)
    raise ValueError(f"Unsupported time format: {time_format}")


def normalize_gender(value: str) -> str | None:
    key = value.strip().lower()
    return GENDER_TOKENS.get(key)


def normalize_speaker(value: str) -> str:
    value = value.strip().strip(":")
    value = re.sub(r"\s+", "_", value)
    return value or "SPEAKER_01"


def clean_transcript(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
        value = value[1:-1].strip()
    return value


def parse_line(line: str, file_id: str) -> tuple[str, float, float, str, str] | None:
    parts = line.split()
    if len(parts) >= 6:
        start_val = parse_time_value(parts[2])
        end_val = parse_time_value(parts[3])
        if start_val is not None and end_val is not None:
            speaker = normalize_speaker(parts[1])
            gender = normalize_gender(parts[-1])
            if gender:
                transcript = " ".join(parts[4:-1]).strip()
            else:
                gender = "unknown"
                transcript = " ".join(parts[4:]).strip()
            transcript = clean_transcript(transcript)
            return speaker, start_val, end_val, transcript, gender

    line = line.replace("—", "-").replace("–", "-")
    line = re.sub(
        r"(\d+(?:\.\d+)?|\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)[-](\d+(?:\.\d+)?|\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)",
        r"\1 \2",
        line,
    )
    matches = list(TIME_RE.finditer(line))
    if len(matches) < 2:
        return None

    start_val = parse_time_value(matches[0].group(1))
    end_val = parse_time_value(matches[1].group(1))
    if start_val is None or end_val is None:
        return None

    prefix = line[: matches[0].start()].strip()
    suffix = line[matches[1].end() :].strip()
    speaker = ""
    if prefix:
        prefix_parts = prefix.split()
        if prefix_parts and prefix_parts[0] == file_id:
            speaker = " ".join(prefix_parts[1:]).strip()
        else:
            speaker = prefix

    speaker = normalize_speaker(speaker)
    gender = "unknown"
    transcript = suffix
    suffix_parts = suffix.split()
    if suffix_parts:
        possible_gender = normalize_gender(suffix_parts[-1])
        if possible_gender:
            gender = possible_gender
            transcript = " ".join(suffix_parts[:-1]).strip()
    transcript = clean_transcript(transcript)
    return speaker, start_val, end_val, transcript, gender


def estimate_tokens(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    return int(math.ceil(len(text) / 4))


def resolve_pricing_rates(model: str, total_tokens: int) -> tuple[float, float, str] | None:
    model_name = model.lower()
    if "gemini-3-flash-preview" in model_name:
        return 0.50, 3.0, "gemini-3-flash-preview"
    if "gemini-3-pro-preview" in model_name:
        if total_tokens <= 200_000:
            return 2.0, 12.0, "gemini-3-pro-preview-tier1"
        return 4.0, 18.0, "gemini-3-pro-preview-tier2"
    return None


def estimate_usage(input_tokens: int, output_tokens: int, model: str) -> UsageSummary:
    total_tokens = input_tokens + output_tokens
    rates = resolve_pricing_rates(model, total_tokens)
    if not rates:
        return UsageSummary(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
    input_rate, output_rate, label = rates
    input_cost = (input_tokens / 1_000_000) * input_rate
    output_cost = (output_tokens / 1_000_000) * output_rate
    return UsageSummary(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=input_cost + output_cost,
        pricing_label=label,
    )


def format_usage_summary(usage: UsageSummary) -> str:
    if usage.pricing_label is None:
        return (
            "usage input_tokens={input_tokens} output_tokens={output_tokens} "
            "total_tokens={total_tokens} pricing=unknown"
        ).format(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )
    return (
        "usage input_tokens={input_tokens} output_tokens={output_tokens} "
        "total_tokens={total_tokens} estimated_cost_usd={cost:.6f} pricing={pricing}"
    ).format(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cost=usage.total_cost_usd or 0.0,
        pricing=usage.pricing_label,
    )


def log_usage_summary(usage: UsageSummary) -> None:
    LOGGER.info(format_usage_summary(usage))


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
    raw_lines: list[str] = []
    buffer: list[str] = []
    blocked_reasons: list[str] = []
    errors: list[str] = []

    def clean_stream_line(line: str) -> str:
        line = line.strip()
        if line.startswith("data:"):
            line = line[len("data:") :].strip()
        if line.startswith(")]}'"):
            line = line[len(")]}'") :].strip()
        return line

    def append_event(event: object) -> None:
        if isinstance(event, list):
            for item in event:
                append_event(item)
            return
        if not isinstance(event, dict):
            return
        if "error" in event:
            errors.append(str(event["error"]))
        prompt_feedback = event.get("promptFeedback")
        if isinstance(prompt_feedback, dict):
            reason = prompt_feedback.get("blockReason")
            if reason:
                blocked_reasons.append(str(reason))
        for candidate in event.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                text = part.get("text")
                if text:
                    text_chunks.append(text)

    with requests.post(url, json=payload, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            raw_lines.append(line)
            cleaned = clean_stream_line(line)
            if not cleaned or cleaned == "[DONE]":
                continue
            buffer.append(cleaned)
            try:
                event = json.loads("".join(buffer))
            except json.JSONDecodeError:
                continue
            buffer.clear()
            append_event(event)

    if buffer:
        try:
            append_event(json.loads("\n".join(buffer)))
        except json.JSONDecodeError:
            pass

    if not text_chunks:
        if blocked_reasons:
            LOGGER.warning("Gemini response blocked: %s", ", ".join(sorted(set(blocked_reasons))))
        if errors:
            LOGGER.warning("Gemini response errors: %s", "; ".join(errors))
        if raw_lines:
            LOGGER.warning("Gemini response had no text parts")

    return "".join(text_chunks).strip()


def normalize_output(text: str, file_id: str, offset: float, time_format: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        parsed = parse_line(line, file_id)
        if not parsed:
            continue
        speaker, start_val, end_val, transcript, gender = parsed
        if end_val < start_val:
            continue
        start_val += offset
        end_val += offset
        line_out = (
            f"{file_id} {speaker} "
            f"{format_timestamp(start_val, time_format)} "
            f"{format_timestamp(end_val, time_format)} "
            f"{transcript} {gender}"
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
    time_format: str,
) -> list[str]:
    file_id = sanitize_file_id(file_id)
    output_lines: list[str] = []
    input_tokens = 0
    output_tokens = 0

    with tempfile.TemporaryDirectory(prefix="segments_") as tmp_dir:
        segments = split_audio(audio_path, segment_seconds, Path(tmp_dir))
        for segment in segments:
            LOGGER.info("Processing segment at %.2fs: %s", segment.start_offset, segment.path)
            prompt = build_prompt(file_id, segment.start_offset)
            input_tokens += estimate_tokens(prompt)
            response_text = call_gemini_stream(
                api_key=api_key,
                model=model,
                prompt=prompt,
                audio_path=segment.path,
            )
            output_tokens += estimate_tokens(response_text)
            normalized = normalize_output(
                response_text,
                file_id=file_id,
                offset=segment.start_offset,
                time_format=time_format,
            )
            if not normalized and response_text.strip():
                LOGGER.warning("No parsable lines at %.2fs", segment.start_offset)
            output_lines.extend(normalized)

    usage = estimate_usage(input_tokens, output_tokens, model)
    return output_lines, usage


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
