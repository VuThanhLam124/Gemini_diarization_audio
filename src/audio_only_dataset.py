from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path

from edit_audio import parse_speaker_info_from_label

from .pyannote_diarization import DiarizationSegment, PyannoteDiarizer, ensure_ffmpeg_exists


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


def resolve_audio_files_from_list(
    audio_files: list[str],
    audio_dir: Path | None,
) -> list[Path]:
    """Resolve explicit audio list with optional base directory."""
    if not audio_files:
        raise RuntimeError("Empty audio files list.")

    resolved: list[Path] = []
    missing: list[str] = []
    seen: set[str] = set()

    for item in audio_files:
        raw = item.strip()
        if not raw:
            continue

        file_path = Path(raw).expanduser()
        candidates: list[Path] = []
        if file_path.is_absolute():
            candidates.append(file_path)
        else:
            if audio_dir:
                candidates.append((audio_dir / file_path).expanduser())
            candidates.append(file_path)

        matched: Path | None = None
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                matched = candidate.resolve()
                break

        if not matched:
            missing.append(raw)
            continue

        key = str(matched)
        if key not in seen:
            seen.add(key)
            resolved.append(matched)

    if missing:
        raise RuntimeError(f"Audio file(s) not found: {', '.join(missing)}")

    if not resolved:
        raise RuntimeError("No valid audio files found from --audio-files.")

    return resolved


def resolve_audio_inputs(
    *,
    audio_dir: Path | None,
    audio_pattern: str,
    audio_files: list[str] | None,
) -> list[Path]:
    """Resolve audio inputs from explicit list or directory scan."""
    if audio_files:
        return resolve_audio_files_from_list(audio_files, audio_dir)

    if not audio_dir:
        raise RuntimeError("audio_dir is required when --audio-files is not provided.")

    if not audio_dir.exists() or not audio_dir.is_dir():
        raise RuntimeError(f"Audio directory not found: {audio_dir}")

    return discover_audio_files(audio_dir=audio_dir, audio_pattern=audio_pattern)


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


def _should_skip_audio(audio_path: Path) -> bool:
    stem = audio_path.stem.lower()
    return "phuyen" in stem


def compute_audio_offsets(audio_files: list[Path]) -> dict[str, float]:
    """Compute cumulative offsets for each audio part inside the same group."""
    grouped: dict[str, list[tuple[int, Path]]] = {}
    for audio_path in audio_files:
        group_id, part_num = _extract_audio_group(audio_path.stem)
        grouped.setdefault(group_id, []).append((part_num, audio_path))

    offsets: dict[str, float] = {}
    for _, parts in grouped.items():
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


def _build_barcoded_speaker(diarization_speaker: str, source_stem: str) -> str:
    source_code = re.sub(r"[^A-Za-z0-9_-]", "_", source_stem)
    base = diarization_speaker.split("+", 1)[0]
    return f"{base}+{source_code}"


def _format_hhmmss(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _calc_overlap_seconds(
    seg_start: float,
    seg_end: float,
    spk_start: float,
    spk_end: float,
) -> float:
    overlap_start = max(seg_start, spk_start)
    overlap_end = min(seg_end, spk_end)
    return max(0.0, overlap_end - overlap_start)


def _find_best_speaker_match(
    segment_start_sec: float,
    segment_end_sec: float,
    speaker_list: list[dict],
    *,
    min_overlap: float,
) -> tuple[dict | None, float, float]:
    """Find best speaker by overlap ratio, return (speaker, ratio, overlap_seconds)."""
    segment_duration = segment_end_sec - segment_start_sec
    if segment_duration <= 0:
        return None, 0.0, 0.0

    best_speaker = None
    best_ratio = 0.0
    best_overlap = 0.0

    for spk_info in speaker_list:
        spk_start = float(spk_info.get("start_sec", 0.0) or 0.0)
        spk_end = float(spk_info.get("end_sec", 0.0) or 0.0)
        if spk_start >= spk_end:
            continue

        overlap_seconds = _calc_overlap_seconds(
            segment_start_sec,
            segment_end_sec,
            spk_start,
            spk_end,
        )
        if overlap_seconds <= 0:
            continue

        ratio = overlap_seconds / segment_duration
        better_ratio = ratio > best_ratio
        better_tie = ratio == best_ratio and overlap_seconds > best_overlap
        if better_ratio or better_tie:
            best_speaker = spk_info
            best_ratio = ratio
            best_overlap = overlap_seconds

    if best_speaker and best_ratio >= min_overlap:
        return best_speaker, best_ratio, best_overlap
    return None, 0.0, 0.0


def _build_overlap_maps(
    file_rows: list[dict],
    speaker_list: list[dict],
    *,
    min_overlap: float,
    use_global_timeline: bool,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict]], dict[str, dict], int, float]:
    overlap_scores_by_diarization: dict[str, dict[str, float]] = {}
    profile_by_diarization_and_name: dict[str, dict[str, dict]] = {}
    row_profile_by_segment: dict[str, dict] = {}
    matched_rows = 0
    total_overlap = 0.0

    for row in file_rows:
        seg_start = float(row["abs_start_sec"] if use_global_timeline else row["start_sec"])
        seg_end = float(row["abs_end_sec"] if use_global_timeline else row["end_sec"])

        matched_speaker, _, overlap_seconds = _find_best_speaker_match(
            seg_start,
            seg_end,
            speaker_list,
            min_overlap=min_overlap,
        )
        if not matched_speaker:
            continue

        speaker_name = str(matched_speaker.get("speaker_name", "") or "").strip()
        if not speaker_name:
            continue

        speaker_gender = str(matched_speaker.get("speaker_gender", "") or "").strip()
        speaker_region = str(matched_speaker.get("speaker_region", "") or "").strip()
        diarization_speaker = str(row["diarization_speaker"])
        segment_id = str(row["segment_id"])

        score_map = overlap_scores_by_diarization.setdefault(diarization_speaker, {})
        score_map[speaker_name] = score_map.get(speaker_name, 0.0) + overlap_seconds

        profile_map = profile_by_diarization_and_name.setdefault(diarization_speaker, {})
        current_profile = profile_map.get(speaker_name)
        if not current_profile or overlap_seconds > float(current_profile.get("overlap_seconds", 0.0)):
            profile_map[speaker_name] = {
                "speaker_gender": speaker_gender,
                "speaker_region": speaker_region,
                "overlap_seconds": overlap_seconds,
            }

        row_profile_by_segment[segment_id] = {
            "speaker_name": speaker_name,
            "speaker_gender": speaker_gender,
            "speaker_region": speaker_region,
            "overlap_seconds": overlap_seconds,
        }
        matched_rows += 1
        total_overlap += overlap_seconds

    return (
        overlap_scores_by_diarization,
        profile_by_diarization_and_name,
        row_profile_by_segment,
        matched_rows,
        total_overlap,
    )


def apply_segment_length_constraints(
    segments: list[DiarizationSegment],
    *,
    min_segment_duration: float,
    max_segment_duration: float,
) -> tuple[list[DiarizationSegment], int]:
    """Split long segments and drop short chunks."""
    constrained: list[DiarizationSegment] = []
    dropped_short = 0

    for segment in segments:
        cursor = segment.start_sec
        while cursor < segment.end_sec:
            chunk_end = min(cursor + max_segment_duration, segment.end_sec)
            duration = chunk_end - cursor
            if duration >= min_segment_duration:
                constrained.append(
                    DiarizationSegment(
                        start_sec=cursor,
                        end_sec=chunk_end,
                        diarization_speaker=segment.diarization_speaker,
                    )
                )
            else:
                dropped_short += 1
            cursor = chunk_end

    return constrained, dropped_short


def write_speaker_name_mapping_csv(output_path: Path, speaker_name_map: dict[str, str]) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["diarization_speaker", "speaker_name"])
        for diarization_speaker in sorted(speaker_name_map):
            speaker_name = str(speaker_name_map[diarization_speaker] or "").strip()
            if not speaker_name:
                continue
            writer.writerow([diarization_speaker, speaker_name])


def create_audio_only_dataset(
    *,
    audio_dir: Path | None,
    audio_files: list[str] | None,
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
    max_segment_duration: float,
    min_overlap: float,
    match_timeline: str,
):
    """Run audio-only diarization pipeline and export dataset artifacts."""
    ensure_ffmpeg_exists()
    match_timeline = str(match_timeline or "auto").strip().lower()
    if match_timeline not in {"auto", "global", "local"}:
        raise RuntimeError(
            f"Invalid match_timeline '{match_timeline}'. Use one of: auto, global, local."
        )

    audio_paths = resolve_audio_inputs(
        audio_dir=audio_dir,
        audio_pattern=audio_pattern,
        audio_files=audio_files,
    )

    skipped_audio_paths = [path for path in audio_paths if _should_skip_audio(path)]
    if skipped_audio_paths:
        print(
            f"Skipping {len(skipped_audio_paths)} audio file(s) by rule: stem contains 'phuyen'"
        )
        for skipped_path in skipped_audio_paths:
            print(f"  - {skipped_path.name}")
        skipped_keys = {str(path.resolve()) for path in skipped_audio_paths}
        audio_paths = [path for path in audio_paths if str(path.resolve()) not in skipped_keys]

    if not audio_paths:
        raise RuntimeError("No audio files left after skip rules were applied.")

    print(f"Discovered {len(audio_paths)} audio files")

    audio_offset_map = compute_audio_offsets(audio_paths)
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
    speaker_name_map: dict[str, str] = {}
    all_diarization_labels: set[str] = set()
    global_name_to_diarization: dict[str, set[str]] = {}
    global_speaker_ids: dict[str, int] = {}
    next_global_speaker_id = 1

    total_segments = 0
    total_matched = 0
    total_fallback = 0
    total_dropped_short = 0
    total_conflicted_diar = 0
    total_multi_diar_name = 0

    for audio_path in audio_paths:
        segments = diarizer.diarize_file(
            audio_path,
            merge_gap=merge_gap,
            min_segment_duration=min_segment_duration,
        )
        segments, dropped_short = apply_segment_length_constraints(
            segments,
            min_segment_duration=min_segment_duration,
            max_segment_duration=max_segment_duration,
        )

        source_file = audio_path.name
        source_stem = audio_path.stem
        file_id = _extract_file_id(source_stem)
        speaker_list = speaker_info_map.get(file_id, [])
        offset = audio_offset_map.get(str(audio_path.resolve()), 0.0)
        file_rows: list[dict] = []
        overlap_scores_by_diarization: dict[str, dict[str, float]] = {}
        profile_by_diarization_and_name: dict[str, dict[str, dict]] = {}
        timeline_mode_used = "no_label"

        for idx, segment in enumerate(segments):
            start_sec = segment.start_sec
            end_sec = segment.end_sec
            duration = end_sec - start_sec
            if duration < min_segment_duration:
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
            diarization_speaker = _build_barcoded_speaker(segment.diarization_speaker, source_stem)
            all_diarization_labels.add(diarization_speaker)

            file_rows.append(
                {
                    "segment_id": segment_id,
                    "audio": str(output_wav.resolve()),
                    "duration": duration,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "abs_start_sec": abs_start_sec,
                    "abs_end_sec": abs_end_sec,
                    "start_sec_glob": _format_hhmmss(abs_start_sec),
                    "end_sec_glob": _format_hhmmss(abs_end_sec),
                    "source_file": source_file,
                    "diarization_speaker": diarization_speaker,
                    "speaker_name": "",
                    "speaker_gender": "",
                    "speaker_region": "",
                }
            )

        if speaker_list and file_rows:
            global_match = None
            local_match = None

            if match_timeline in {"auto", "global"}:
                global_match = _build_overlap_maps(
                    file_rows,
                    speaker_list,
                    min_overlap=min_overlap,
                    use_global_timeline=True,
                )
            if match_timeline in {"auto", "local"}:
                local_match = _build_overlap_maps(
                    file_rows,
                    speaker_list,
                    min_overlap=min_overlap,
                    use_global_timeline=False,
                )

            selected = None
            if match_timeline == "global":
                selected = global_match or local_match
                timeline_mode_used = "global" if global_match else "local"
            elif match_timeline == "local":
                selected = local_match or global_match
                timeline_mode_used = "local" if local_match else "global"
            else:
                global_hits = max(0, global_match[3] if global_match else 0)
                local_hits = max(0, local_match[3] if local_match else 0)

                # Auto mode chi fallback sang local khi global gan nhu khong map duoc.
                global_low_threshold = max(2, int(0.02 * len(file_rows)))
                local_margin = max(5, int(0.1 * len(file_rows)))
                use_local = (
                    bool(local_match)
                    and local_hits >= global_hits + local_margin
                    and global_hits <= global_low_threshold
                )
                if use_local and local_match:
                    selected = local_match
                    timeline_mode_used = "local"
                else:
                    selected = global_match or local_match
                    timeline_mode_used = "global" if global_match else "local"

            if selected:
                (
                    overlap_scores_by_diarization,
                    profile_by_diarization_and_name,
                    row_profile_by_segment,
                    _,
                    _,
                ) = selected
                for row in file_rows:
                    row_profile = row_profile_by_segment.get(str(row["segment_id"]), {})
                    row["speaker_name"] = str(row_profile.get("speaker_name", "") or "")
                    row["speaker_gender"] = str(row_profile.get("speaker_gender", "") or "")
                    row["speaker_region"] = str(row_profile.get("speaker_region", "") or "")
        elif speaker_list:
            timeline_mode_used = "no_segment"

        canonical_name_by_diarization: dict[str, str] = {}
        canonical_profile_by_diarization: dict[str, dict] = {}
        file_conflicted_diar = 0

        for diarization_speaker, score_map in overlap_scores_by_diarization.items():
            if len(score_map) > 1:
                file_conflicted_diar += 1
            speaker_name = max(score_map.items(), key=lambda item: (item[1], item[0]))[0]
            canonical_name_by_diarization[diarization_speaker] = speaker_name
            canonical_profile_by_diarization[diarization_speaker] = (
                profile_by_diarization_and_name.get(diarization_speaker, {}).get(speaker_name, {})
            )

        file_name_to_diarization: dict[str, set[str]] = {}
        for diarization_speaker, speaker_name in canonical_name_by_diarization.items():
            normalized_name = speaker_name.strip().lower()
            if normalized_name:
                file_name_to_diarization.setdefault(normalized_name, set()).add(diarization_speaker)
        file_multi_diar_name = sum(
            1 for diar_labels in file_name_to_diarization.values() if len(diar_labels) > 1
        )

        file_matched = 0
        file_fallback = 0
        for row in file_rows:
            diarization_speaker = row["diarization_speaker"]
            row_speaker_name = str(row.get("speaker_name", "") or "").strip()
            canonical_name = canonical_name_by_diarization.get(diarization_speaker, "")

            if row_speaker_name:
                normalized_name = row_speaker_name.lower()
                speaker_key = f"name:{normalized_name}"
                global_name_to_diarization.setdefault(normalized_name, set()).add(diarization_speaker)

                # Mapping diarization_speaker -> speaker_name giu theo canonical neu co.
                if canonical_name:
                    speaker_name_map[diarization_speaker] = canonical_name
                else:
                    speaker_name_map.setdefault(diarization_speaker, row_speaker_name)

                speaker_gender = str(row["speaker_gender"] or "")
                speaker_region = str(row["speaker_region"] or "")
                file_matched += 1
            elif canonical_name:
                normalized_name = canonical_name.strip().lower()
                speaker_key = f"name:{normalized_name}"
                speaker_name_map[diarization_speaker] = canonical_name
                global_name_to_diarization.setdefault(normalized_name, set()).add(diarization_speaker)

                profile = canonical_profile_by_diarization.get(diarization_speaker, {})
                speaker_gender = str(row["speaker_gender"] or profile.get("speaker_gender", "") or "")
                speaker_region = str(row["speaker_region"] or profile.get("speaker_region", "") or "")
                file_matched += 1
            else:
                speaker_key = f"diarization:{diarization_speaker}"
                speaker_gender = str(row["speaker_gender"] or "")
                speaker_region = str(row["speaker_region"] or "")
                file_fallback += 1

            if speaker_key not in global_speaker_ids:
                global_speaker_ids[speaker_key] = next_global_speaker_id
                next_global_speaker_id += 1

            metadata_rows.append(
                {
                    "segment_id": row["segment_id"],
                    "audio": row["audio"],
                    "duration": row["duration"],
                    "start_sec": row["start_sec"],
                    "end_sec": row["end_sec"],
                    "start_sec_glob": row["start_sec_glob"],
                    "end_sec_glob": row["end_sec_glob"],
                    "source_file": row["source_file"],
                    "diarization_speaker": diarization_speaker,
                    "speaker_id": global_speaker_ids[speaker_key],
                    "speaker_gender": speaker_gender,
                    "speaker_region": speaker_region,
                }
            )

        file_segments = len(file_rows)
        total_dropped_short += dropped_short
        total_conflicted_diar += file_conflicted_diar
        total_multi_diar_name += file_multi_diar_name
        print(
            f"{source_file}: segments={file_segments} matched={file_matched} "
            f"fallback={file_fallback} dropped_short={dropped_short} "
            f"conflicted_diar={file_conflicted_diar} multi_diar_name={file_multi_diar_name} "
            f"timeline={timeline_mode_used} label_windows={len(speaker_list)}"
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
        "start_sec_glob",
        "end_sec_glob",
        "source_file",
        "diarization_speaker",
        "speaker_id",
        "speaker_gender",
        "speaker_region",
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
                    row["start_sec_glob"],
                    row["end_sec_glob"],
                    row["source_file"],
                    row["diarization_speaker"],
                    row["speaker_id"],
                    row["speaker_gender"],
                    row["speaker_region"],
                ]
            )

    speaker_name_map_path = output_dir / "speaker_name_mapping.csv"
    write_speaker_name_mapping_csv(speaker_name_map_path, speaker_name_map)

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
            "start_sec_glob": Value("string"),
            "end_sec_glob": Value("string"),
            "source_file": Value("string"),
            "diarization_speaker": Value("string"),
            "speaker_id": Value("int32"),
            "speaker_gender": Value("string"),
            "speaker_region": Value("string"),
        }
    )

    data_dict = {column: [row[column] for row in metadata_rows] for column in metadata_columns}
    dataset = Dataset.from_dict(data_dict, features=features)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))

    dataset_save_path = output_dir / "hf_dataset"
    dataset.save_to_disk(dataset_save_path)

    print(f"Dataset name: {dataset_name}")
    print(f"Saved metadata: {metadata_csv_path}")
    print(f"Saved speaker mapping: {speaker_name_map_path}")
    print(f"Saved HF dataset: {dataset_save_path}")
    multi_diar_names_global = sum(
        1 for diar_labels in global_name_to_diarization.values() if len(diar_labels) > 1
    )
    unmapped_diarization_speakers = max(0, len(all_diarization_labels) - len(speaker_name_map))
    print(
        f"Summary: total_segments={total_segments} matched={total_matched} "
        f"fallback={total_fallback} dropped_short={total_dropped_short} "
        f"global_speakers={len(global_speaker_ids)} conflicted_diar={total_conflicted_diar} "
        f"multi_diar_name={total_multi_diar_name} multi_diar_name_global={multi_diar_names_global} "
        f"unmapped_diarization_speakers={unmapped_diarization_speakers}"
    )

    return dataset
