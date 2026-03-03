from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def ensure_ffmpeg_tools() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required but not found in PATH")
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe is required but not found in PATH")


def get_audio_duration_seconds(audio_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def build_segments_with_tail_merge(
    total_seconds: float,
    chunk_seconds: int = 3600,
    min_tail_seconds: int = 900,
) -> list[tuple[float, float]]:
    if total_seconds <= 0:
        return []

    if total_seconds <= chunk_seconds:
        return [(0.0, total_seconds)]

    full_chunks = int(total_seconds // chunk_seconds)
    tail = total_seconds - full_chunks * chunk_seconds
    segments: list[tuple[float, float]] = []

    if tail == 0:
        for idx in range(full_chunks):
            start = idx * chunk_seconds
            end = (idx + 1) * chunk_seconds
            segments.append((float(start), float(end)))
        return segments

    if tail < min_tail_seconds and full_chunks > 0:
        for idx in range(max(full_chunks - 1, 0)):
            start = idx * chunk_seconds
            end = (idx + 1) * chunk_seconds
            segments.append((float(start), float(end)))

        start_last = (full_chunks - 1) * chunk_seconds
        segments.append((float(start_last), total_seconds))
        return segments

    for idx in range(full_chunks):
        start = idx * chunk_seconds
        end = (idx + 1) * chunk_seconds
        segments.append((float(start), float(end)))
    segments.append((float(full_chunks * chunk_seconds), total_seconds))
    return segments


def split_audio_file(
    input_path: Path,
    output_dir: Path,
    chunk_seconds: int = 3600,
    min_tail_seconds: int = 900,
) -> list[Path]:
    duration = get_audio_duration_seconds(input_path)
    segments = build_segments_with_tail_merge(
        total_seconds=duration,
        chunk_seconds=chunk_seconds,
        min_tail_seconds=min_tail_seconds,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = input_path.stem
    ext = input_path.suffix
    output_files: list[Path] = []

    for idx, (start, end) in enumerate(segments, start=1):
        part_path = output_dir / f"{base_name}_{idx}{ext}"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{(end - start):.3f}",
            "-c",
            "copy",
            str(part_path),
        ]
        subprocess.run(cmd, check=True)
        output_files.append(part_path)

    return output_files


def list_audio_files(input_dir: Path, audio_pattern: str) -> list[Path]:
    patterns = [p.strip() for p in audio_pattern.split(",") if p.strip()]
    files: set[Path] = set()
    for pattern in patterns:
        files.update(input_dir.glob(pattern))
    return sorted(p for p in files if p.is_file())


def split_audio_folder(
    input_dir: Path,
    output_dir: Path,
    chunk_seconds: int = 3600,
    min_tail_seconds: int = 900,
    audio_pattern: str = "*.mp3,*.wav,*.m4a,*.flac,*.ogg,*.aac",
) -> dict[Path, list[Path]]:
    ensure_ffmpeg_tools()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    audio_files = list_audio_files(input_dir, audio_pattern)
    if not audio_files:
        raise RuntimeError(f"No audio files found in {input_dir} with pattern: {audio_pattern}")

    results: dict[Path, list[Path]] = {}
    for audio_file in audio_files:
        parts = split_audio_file(
            input_path=audio_file,
            output_dir=output_dir,
            chunk_seconds=chunk_seconds,
            min_tail_seconds=min_tail_seconds,
        )
        results[audio_file] = parts
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split all audio files in a folder into fixed-length chunks."
    )
    parser.add_argument(
        "--input-dir",
        default="data",
        help="Directory containing source audio files",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to save split audio files",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=3600,
        help="Target chunk size in seconds (default: 3600 = 1 hour)",
    )
    parser.add_argument(
        "--min-tail-seconds",
        type=int,
        default=900,
        help="If tail chunk is shorter than this, merge into nearest chunk (default: 900 = 15 min)",
    )
    parser.add_argument(
        "--audio-pattern",
        default="*.mp3,*.wav,*.m4a,*.flac,*.ogg,*.aac",
        help="Comma-separated glob patterns to select input files",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    results = split_audio_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        chunk_seconds=args.chunk_seconds,
        min_tail_seconds=args.min_tail_seconds,
        audio_pattern=args.audio_pattern,
    )

    print(f"Processed {len(results)} files")
    for src_path, part_paths in results.items():
        print(f"- {src_path.name}: {len(part_paths)} parts")


if __name__ == "__main__":
    main()
