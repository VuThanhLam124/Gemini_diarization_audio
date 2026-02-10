import argparse
import json
import re
import sys
from pathlib import Path

from edit_audio import create_hf_dataset, parse_speaker_info_from_label, parse_time

_DURATION_CACHE: dict[str, float] = {}


def get_audio_duration(audio_path: Path) -> float:
    """Lay do dai audio bang ffprobe (seconds) voi cache."""
    key = str(audio_path)
    if key in _DURATION_CACHE:
        return _DURATION_CACHE[key]

    import subprocess

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
    duration = float(result.stdout.strip())
    _DURATION_CACHE[key] = duration
    return duration


def get_segment_end_time(segments: list[list[str]]) -> float | None:
    """Lay end time cua segment cuoi cung (seconds)."""
    if not segments:
        return None
    last_seg = segments[-1]
    if len(last_seg) >= 2:
        return parse_time(last_seg[1])
    if len(last_seg) == 1 and isinstance(last_seg[0], str) and "-" in last_seg[0]:
        _, end_str = last_seg[0].split("-", 1)
        return parse_time(end_str)
    return None


def normalize_segments_for_audio(
    segments: list[list[str]],
    segment_original: list[list[str]],
    audio_path: Path,
    json_path: Path,
    tolerance_sec: float = 0.5,
) -> tuple[list[list[str]], list[list[str]]]:
    """Chuan hoa: segments = timeline sau edit, segment_original = timeline goc."""
    if not segments or not segment_original:
        return segments, segment_original

    edit_dir = json_path.parent / "outputs_segment_with_mark"
    edited_path = edit_dir / audio_path.name
    if not edited_path.exists():
        return segments, segment_original

    try:
        orig_duration = get_audio_duration(audio_path)
        edited_duration = get_audio_duration(edited_path)
    except Exception:
        return segments, segment_original

    seg_end = get_segment_end_time(segments)
    seg_orig_end = get_segment_end_time(segment_original)
    if seg_end is None or seg_orig_end is None:
        return segments, segment_original

    # Doi chi so sao cho segments gan voi edited_duration, segment_original gan voi orig_duration
    keep_score = abs(seg_end - edited_duration) + abs(seg_orig_end - orig_duration)
    swap_score = abs(seg_end - orig_duration) + abs(seg_orig_end - edited_duration)

    if swap_score + tolerance_sec < keep_score:
        print(f"Normalize segments: swap for {audio_path.name}")
        return segment_original, segments

    return segments, segment_original


def extract_audio_id_and_part(audio_path: str) -> tuple[str, int]:
    """Trich xuat audio ID goc va so thu tu part tu ten file.

    VD: 'hatinh1_1.mp3' -> ('hatinh1', 1)
        'hatinh1_2.mp3' -> ('hatinh1', 2)
    """
    stem = Path(audio_path).stem
    match = re.match(r"^(.+)_(\d+)$", stem)
    if match:
        return match.group(1), int(match.group(2))
    return stem, 0


def calculate_audio_offsets(config_data: dict, json_path: Path) -> dict[str, float]:
    """Tinh offset cua tung audio nho trong audio goc."""
    audio_groups: dict[str, list[tuple[int, str, dict]]] = {}

    for audio_str, info in config_data.items():
        audio_path = Path(audio_str)
        if not audio_path.exists():
            audio_path_rel = json_path.parent / audio_str
            if audio_path_rel.exists():
                audio_path = audio_path_rel

        audio_id, part_num = extract_audio_id_and_part(str(audio_path))
        audio_groups.setdefault(audio_id, []).append((part_num, str(audio_path), info))

    offset_map: dict[str, float] = {}

    for audio_id, parts in audio_groups.items():
        parts.sort(key=lambda x: x[0])
        cumulative_offset = 0.0

        for part_num, audio_path, info in parts:
            _ = part_num
            offset_map[audio_path] = cumulative_offset

            audio_filename = Path(audio_path).name
            outputs_audio_path = json_path.parent / "outputs" / audio_filename

            try:
                duration = get_audio_duration(outputs_audio_path)
                cumulative_offset += duration
            except Exception:
                segments = info.get("segment_original", []) or info.get("segments", [])
                if segments:
                    last_seg = segments[-1]
                    if len(last_seg) >= 2:
                        cumulative_offset += parse_time(last_seg[1])

    return offset_map


def _resolve_label_csv_path(label_csv_arg: str, base_dir: Path | None = None) -> Path | None:
    label_csv = Path(label_csv_arg)
    if label_csv.exists():
        return label_csv

    if base_dir:
        candidate = base_dir / label_csv_arg
        if candidate.exists():
            return candidate

    return None


def run_legacy_mode(args: argparse.Namespace) -> int:
    json_path = Path(args.input_json)
    if not json_path.exists():
        print(f"Error: Khong tim thay file JSON {json_path}")
        return 1

    speaker_info_map = None
    label_csv_path = _resolve_label_csv_path(args.label_csv, base_dir=json_path.parent)

    if label_csv_path:
        print(f"Loading speaker info from: {label_csv_path}")
        speaker_info_map = parse_speaker_info_from_label(label_csv_path)
        print(f"  Found info for {len(speaker_info_map)} audio IDs")
    else:
        print(f"Warning: Label CSV not found: {args.label_csv} - speaker info se trong")

    with open(json_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    print("Calculating audio offsets...")
    audio_offset_map = calculate_audio_offsets(config_data, json_path)
    for audio_path, offset in audio_offset_map.items():
        print(f"  {Path(audio_path).name}: offset = {offset:.2f}s")

    segment_map = {}
    transcript_map = {}
    segment_original_map = {}

    print("Reading config and loading transcripts...")

    for audio_str, info in config_data.items():
        audio_path = Path(audio_str)
        if not audio_path.exists():
            audio_path_rel = json_path.parent / audio_str
            if audio_path_rel.exists():
                audio_path = audio_path_rel
            else:
                print(f"Warning: Audio file not found: {audio_str}")
                continue

        segments = info.get("segments", [])
        if not segments:
            print(f"Warning: No segments for {audio_str}")
            continue

        seg_orig = info.get("segment_original", [])
        if seg_orig and len(seg_orig) != len(segments):
            print(f"Warning: segments va segment_original lech so luong o {audio_str}")

        segments, seg_orig = normalize_segments_for_audio(
            segments=segments,
            segment_original=seg_orig,
            audio_path=audio_path,
            json_path=json_path,
        )

        trans_path_str = info.get("transcript_path", "")
        full_transcript = ""
        if trans_path_str:
            trans_path = Path(trans_path_str)
            if not trans_path.exists():
                trans_path_rel = json_path.parent / trans_path_str
                if trans_path_rel.exists():
                    trans_path = trans_path_rel
                else:
                    print(f"Warning: Transcript file not found: {trans_path_str}")

            if trans_path.exists():
                with open(trans_path, "r", encoding="utf-8") as tf:
                    full_transcript = tf.read()

        if not full_transcript:
            print(f"Warning: Empty transcript for {audio_str}")

        audio_path_key = str(audio_path)
        segment_map[audio_path_key] = segments
        transcript_map[audio_path_key] = full_transcript
        segment_original_map[audio_path_key] = seg_orig

    if not segment_map:
        print("Error: No valid data found to process.")
        return 1

    print(f"\nProcessing {len(segment_map)} audio files (legacy mode)...")
    create_hf_dataset(
        segment_map=segment_map,
        transcript_map=transcript_map,
        output_dir=Path(args.output_dir),
        dataset_name=args.dataset_name,
        speaker_info_map=speaker_info_map,
        segment_original_map=segment_original_map,
        audio_offset_map=audio_offset_map,
    )

    return 0


def run_audio_only_mode(args: argparse.Namespace) -> int:
    if not args.audio_dir:
        print("Error: --audio-dir is required when --input-json is not provided")
        return 1

    try:
        from src.audio_only_dataset import create_audio_only_dataset
    except Exception as exc:
        print(f"Error: unable to load audio-only pipeline: {exc}")
        return 1

    audio_dir = Path(args.audio_dir)
    label_csv_path = _resolve_label_csv_path(args.label_csv)

    try:
        create_audio_only_dataset(
            audio_dir=audio_dir,
            output_dir=Path(args.output_dir),
            dataset_name=args.dataset_name,
            label_csv_path=label_csv_path,
            audio_pattern=args.audio_pattern,
            hf_token=args.hf_token,
            device=args.device,
            seg_min_duration_off=args.seg_min_duration_off,
            clustering_threshold=args.clustering_threshold,
            clustering_method=args.clustering_method,
            min_cluster_size=args.min_cluster_size,
            merge_gap=args.merge_gap,
            min_segment_duration=args.min_segment_duration,
            min_overlap=args.min_overlap,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified pipeline: legacy config+transcript mode or new audio-only diarization mode"
        )
    )

    # Legacy mode arguments
    parser.add_argument(
        "--input-json",
        "-i",
        help="Path den file JSON chua config legacy (audio, segments, transcript_path)",
    )

    # Shared output arguments
    parser.add_argument("--output-dir", "-o", default="outputs_dataset", help="Thu muc output")
    parser.add_argument("--dataset-name", "-n", default="my_dataset", help="Ten dataset")
    parser.add_argument(
        "--label-csv",
        "-l",
        default="data_label_by_hand.csv",
        help="Path den file CSV chua speaker info",
    )

    # Audio-only mode arguments
    parser.add_argument(
        "--audio-dir",
        help="Thu muc audio input cho mode audio-only (required when --input-json is not set)",
    )
    parser.add_argument(
        "--audio-pattern",
        default="*.mp3",
        help="Glob pattern(s) audio file, e.g. '*.mp3' or '*.mp3,*.wav'",
    )
    parser.add_argument("--hf-token", help="Hugging Face token for pyannote model")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for pyannote inference",
    )
    parser.add_argument(
        "--seg-min-duration-off",
        type=float,
        default=None,
        help="Pyannote segmentation.min_duration_off",
    )
    parser.add_argument(
        "--clustering-threshold",
        type=float,
        default=None,
        help="Pyannote clustering.threshold",
    )
    parser.add_argument(
        "--clustering-method",
        choices=["centroid", "average", "ward", "complete", "single"],
        default=None,
        help="Pyannote clustering.method",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=None,
        help="Pyannote clustering.min_cluster_size",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=2.0,
        help="Gap toi da (giay) de gop segment cung speaker",
    )
    parser.add_argument(
        "--min-segment-duration",
        type=float,
        default=0.5,
        help="Do dai toi thieu cua segment sau merge (giay)",
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.70,
        help="Nguong overlap toi thieu de map speaker theo CSV",
    )

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.input_json:
        return run_legacy_mode(args)
    return run_audio_only_mode(args)


if __name__ == "__main__":
    sys.exit(main())
