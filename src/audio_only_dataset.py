from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path

from edit_audio import match_segment_to_speaker, parse_speaker_info_from_label

from .pyannote_diarization import PyannoteDiarizer, ensure_ffmpeg_exists


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def discover_audio_files(audio_dir: Path, audio_pattern: str) -> list[Path]:
    """Discover audio files from a directory and glob pattern(s)."""
    patterns = [p.strip() for p in audio_pattern.split(",") if p.strip()]
    if not patterns:
        patterns = ["*.mp3"]

    found: dict[str, Path] = {}
    for pattern in patterns:
        for file_path in audio_dir.glob(pattern):
            if file_path.is_file():
                found[str(file_path.resolve())] = file_path.resolve()

    audio_files = sorted(found.values())
    if audio_files:
        return audio_files
    joined_patterns = ", ".join(patterns)
    raise RuntimeError(
        f"No audio files found in {audio_dir} with pattern(s): {joined_patterns}"
    )


def _extract_audio_group(stem: str) -> tuple[str, int]:
    match = re.match(r"^(.+)_(\d+)$", stem)
    if match:
        return match.group(1), int(match.group(2))
    return stem, 0


def _extract_file_id(stem: str) -> str:
    match = re.match(r"^(.+)_(\d+)$", stem)
    if match:
        return match.group(1)
    return stem


def compute_audio_offsets(audio_files: list[Path]) -> dict[str, float]:
    """Compute cumulative offsets for each audio part inside the same group."""
    grouped: dict[str, list[tuple[int, Path]]] = {}
    for audio_path in audio_files:
        group_id, part_num = _extract_audio_group(audio_path.stem)
        grouped.setdefault(group_id, []).append((part_num, audio_path))

    offsets: dict[str, float] = {}
    for group_id, parts in grouped.items():
        parts.sort(key=lambda item: (item[0], item[1].name))
        cumulative = 0.0
        for _, audio_path in parts:
            key = str(audio_path.resolve())
            offsets[key] = cumulative
            cumulative += get_audio_duration(audio_path)

    return offsets


def cut_segment_to_wav(
    *,
    audio_path: Path,
    start_sec: float,
    end_sec: float,
    output_path: Path,
) -> None:
    """Cut a segment into mono 16k WAV."""
    duration = end_sec - start_sec
    if duration <= 0:
        raise ValueError(f"Invalid segment duration ({duration}) for {audio_path}")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(audio_path),
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{duration:.3f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return
    stderr = result.stderr.strip() or "unknown ffmpeg error"
    raise RuntimeError(f"Failed to cut segment {output_path.name}: {stderr}")


def _format_overlap(value: float) -> str:
    if value <= 0:
        return ""
    return f"{value:.4f}"


def _load_speaker_info(label_csv_path: Path | None) -> dict[str, list[dict]]:
    if not label_csv_path:
        return {}
    if not label_csv_path.exists():
        print(f"Warning: Label CSV not found: {label_csv_path} - speaker mapping disabled")
        return {}

    print(f"Loading speaker info from: {label_csv_path}")
    mapping = parse_speaker_info_from_label(label_csv_path)
    print(f"  Found info for {len(mapping)} audio IDs")
    return mapping


def create_audio_only_dataset(
    *,
    audio_dir: Path,
    output_dir: Path,
    dataset_name: str,
    label_csv_path: Path | None,
    audio_pattern: str,
    hf_token: str | None,
    device: str,
    seg_min_duration_off: float | None,
    clustering_threshold: float | None,
    clustering_method: str | None,
    min_cluster_size: int | None,
    merge_gap: float,
    min_segment_duration: float,
    min_overlap: float,
):
    """Run audio-only diarization pipeline and export dataset artifacts."""
    ensure_ffmpeg_exists()

    if not audio_dir.exists() or not audio_dir.is_dir():
        raise RuntimeError(f"Audio directory not found: {audio_dir}")

    audio_files = discover_audio_files(audio_dir=audio_dir, audio_pattern=audio_pattern)
    print(f"Discovered {len(audio_files)} audio files")

    audio_offset_map = compute_audio_offsets(audio_files)
    print("Computed audio offsets")

    speaker_info_map = _load_speaker_info(label_csv_path)

    diarizer = PyannoteDiarizer(
        hf_token=hf_token,
        device=device,
        seg_min_duration_off=seg_min_duration_off,
        clustering_threshold=clustering_threshold,
        clustering_method=clustering_method,
        min_cluster_size=min_cluster_size,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict] = []
    total_segments = 0
    total_matched = 0
    total_fallback = 0

    for audio_path in audio_files:
        segments = diarizer.diarize_file(
            audio_path,
            merge_gap=merge_gap,
            min_segment_duration=min_segment_duration,
        )

        file_matched = 0
        file_fallback = 0
        source_file = audio_path.name
        source_stem = audio_path.stem
        file_id = _extract_file_id(source_stem)
        speaker_list = speaker_info_map.get(file_id, [])
        offset = audio_offset_map.get(str(audio_path.resolve()), 0.0)

        for idx, segment in enumerate(segments):
            start_sec = segment.start_sec
            end_sec = segment.end_sec
            duration = end_sec - start_sec
            if duration <= 0:
                continue

            segment_id = f"{source_stem}_{idx:04d}"
            output_wav = wavs_dir / f"{segment_id}.wav"
            cut_segment_to_wav(
                audio_path=audio_path,
                start_sec=start_sec,
                end_sec=end_sec,
                output_path=output_wav,
            )

            abs_start_sec = start_sec + offset
            abs_end_sec = end_sec + offset

            speaker_id = 0
            speaker_name = ""
            speaker_gender = ""
            speaker_region = ""
            speaker_position = ""
            overlap_ratio = 0.0

            if speaker_list:
                matched_speaker, ratio = match_segment_to_speaker(
                    abs_start_sec,
                    abs_end_sec,
                    speaker_list,
                    threshold=min_overlap,
                )
                if matched_speaker:
                    speaker_id = int(matched_speaker.get("speaker_id", 0) or 0)
                    speaker_name = str(matched_speaker.get("speaker_name", "") or "")
                    speaker_gender = str(matched_speaker.get("speaker_gender", "") or "")
                    speaker_region = str(matched_speaker.get("speaker_region", "") or "")
                    speaker_position = str(matched_speaker.get("speaker_position", "") or "")
                    overlap_ratio = float(ratio)

            if speaker_name:
                speaker_label = speaker_name
                file_matched += 1
            else:
                speaker_label = segment.diarization_speaker
                file_fallback += 1

            metadata_rows.append(
                {
                    "segment_id": segment_id,
                    "audio": str(output_wav.resolve()),
                    "duration": duration,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "abs_start_sec": abs_start_sec,
                    "abs_end_sec": abs_end_sec,
                    "source_file": source_file,
                    "diarization_speaker": segment.diarization_speaker,
                    "speaker_label": speaker_label,
                    "speaker_id": speaker_id,
                    "speaker_name": speaker_name,
                    "speaker_gender": speaker_gender,
                    "speaker_region": speaker_region,
                    "speaker_position": speaker_position,
                    "overlap_ratio": overlap_ratio,
                }
            )

        file_segments = file_matched + file_fallback
        print(
            f"{source_file}: segments={file_segments} matched={file_matched} fallback={file_fallback}"
        )

        total_segments += file_segments
        total_matched += file_matched
        total_fallback += file_fallback

    if not metadata_rows:
        raise RuntimeError("No segments were generated from input audio files.")

    metadata_columns = [
        "segment_id",
        "audio",
        "duration",
        "start_sec",
        "end_sec",
        "abs_start_sec",
        "abs_end_sec",
        "source_file",
        "diarization_speaker",
        "speaker_label",
        "speaker_id",
        "speaker_name",
        "speaker_gender",
        "speaker_region",
        "speaker_position",
        "overlap_ratio",
    ]

    metadata_csv_path = output_dir / "metadata.csv"
    with open(metadata_csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(metadata_columns)
        for row in metadata_rows:
            writer.writerow(
                [
                    row["segment_id"],
                    row["audio"],
                    f"{row['duration']:.6f}",
                    f"{row['start_sec']:.6f}",
                    f"{row['end_sec']:.6f}",
                    f"{row['abs_start_sec']:.6f}",
                    f"{row['abs_end_sec']:.6f}",
                    row["source_file"],
                    row["diarization_speaker"],
                    row["speaker_label"],
                    row["speaker_id"],
                    row["speaker_name"],
                    row["speaker_gender"],
                    row["speaker_region"],
                    row["speaker_position"],
                    _format_overlap(row["overlap_ratio"]),
                ]
            )

    try:
        from datasets import Audio, Dataset, Features, Value
    except ImportError as exc:
        raise RuntimeError(
            "Missing datasets dependency. Install with: pip install datasets"
        ) from exc

    features = Features(
        {
            "segment_id": Value("string"),
            "audio": Value("string"),
            "duration": Value("float32"),
            "start_sec": Value("float32"),
            "end_sec": Value("float32"),
            "abs_start_sec": Value("float32"),
            "abs_end_sec": Value("float32"),
            "source_file": Value("string"),
            "diarization_speaker": Value("string"),
            "speaker_label": Value("string"),
            "speaker_id": Value("int32"),
            "speaker_name": Value("string"),
            "speaker_gender": Value("string"),
            "speaker_region": Value("string"),
            "speaker_position": Value("string"),
            "overlap_ratio": Value("float32"),
        }
    )

    data_dict = {column: [row[column] for row in metadata_rows] for column in metadata_columns}
    dataset = Dataset.from_dict(data_dict, features=features)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    dataset_save_path = output_dir / "hf_dataset"
    dataset.save_to_disk(dataset_save_path)

    print(f"Dataset name: {dataset_name}")
    print(f"Saved metadata: {metadata_csv_path}")
    print(f"Saved HF dataset: {dataset_save_path}")
    print(
        f"Summary: total_segments={total_segments} matched={total_matched} fallback={total_fallback}"
    )

    return dataset
