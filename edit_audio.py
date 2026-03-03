"""
Script tach audio thanh cac file nho hon theo start/end time config.
Ten file output: {ten_file_goc}_{stt}.{ext}

Usage:
    python edit_audio.py --input audio.mp3 --output-dir outputs/ --segments "0:00-1:30,1:30-3:00,5:00-10:00"
    
    Hoac config trong code:
    SEGMENTS = [
        (0, 90),      # 0:00 - 1:30
        (90, 180),    # 1:30 - 3:00  
        (300, 600),   # 5:00 - 10:00
    ]
"""

import argparse
import subprocess
from pathlib import Path


def parse_time(time_str: str) -> float:
    """Parse time string (HH:MM:SS hoac MM:SS hoac seconds) thanh seconds."""
    time_str = time_str.strip()
    if ":" in time_str:
        parts = time_str.split(":")
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        elif len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
    return float(time_str)


def parse_segments(segments_str: str) -> list[tuple[float, float]]:
    """Parse segments string thanh list of (start, end) tuples."""
    segments = []
    for seg in segments_str.split(","):
        seg = seg.strip()
        if "-" in seg:
            start_str, end_str = seg.split("-", 1)
            start = parse_time(start_str)
            end = parse_time(end_str)
            segments.append((start, end))
    return segments


def split_audio(
    input_path: Path,
    output_dir: Path,
    segments: list[tuple[float, float]],
    output_name: str | None = None,
) -> list[Path]:
    """Tach audio thanh cac file nho theo segments.
    
    Args:
        input_path: Duong dan file audio goc
        output_dir: Thu muc luu file output
        segments: List cac (start_seconds, end_seconds)
        output_name: Ten file output (khong can extension), mac dinh dung ten file goc
        
    Returns:
        List duong dan cac file da tach
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = output_name or input_path.stem
    ext = input_path.suffix
    
    output_files = []
    
    for idx, (start, end) in enumerate(segments, start=1):
        duration = end - start
        output_name = f"{base_name}_{idx}{ext}"
        output_path = output_dir / output_name
        
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(input_path),
            "-ss", str(start),
            "-t", str(duration),
            "-c", "copy",
            str(output_path),
        ]
        
        print(f"[{idx}] Tach {start:.2f}s - {end:.2f}s -> {output_name}")
        subprocess.run(cmd, check=True)
        output_files.append(output_path)
    
    return output_files


def get_audio_duration(audio_path: Path) -> float:
    """Lay do dai audio bang ffprobe (seconds)."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def format_time(seconds: float, precise: bool = False) -> str:
    """Chuyen seconds thanh MM:SS hoac HH:MM:SS.
    
    Args:
        seconds: So giay
        precise: Neu True, tra ve seconds chinh xac (vd: 00:33.07)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if precise:
        secs = seconds % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
        return f"{minutes:02d}:{secs:05.2f}"
    else:
        secs = round(seconds % 60)
        # Xu ly truong hop lam tron len 60
        if secs == 60:
            secs = 0
            minutes += 1
        if minutes == 60:
            minutes = 0
            hours += 1
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"


def update_timestamps_after_merge(
    segments: list[list[str]],
    marker_path: Path,
    input_path: Path = None,
) -> list[tuple[str, str]]:
    """Tinh lai timestamp sau khi chen marker vao giua cac segment.
    
    Args:
        segments: List cac segment goc dang [['MM:SS-MM:SS'], ...]
        marker_path: Duong dan file marker audio
        input_path: Duong dan file audio goc (neu co se do truc tiep do dai segment)
        
    Returns:
        List cac tuple (new_start, new_end) timestamp moi
    """
    import tempfile
    
    # Lay do dai marker audio goc
    marker_duration = get_audio_duration(marker_path)
    
    # Parse segments goc thanh (start, end) seconds
    parsed_segments = []
    for seg_list in segments:
        seg_str = seg_list[0]
        start_str, end_str = seg_str.split("-", 1)
        start = parse_time(start_str)
        end = parse_time(end_str)
        parsed_segments.append((start, end))
    
    # Neu co input_path, do truc tiep do dai thuc te cua tung segment
    segment_durations = []
    if input_path and input_path.exists():
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for idx, (start, end) in enumerate(parsed_segments):
                duration = end - start
                seg_file = temp_path / f"seg_{idx:04d}.mp3"
                
                # Cat segment
                cmd = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-y",
                    "-i", str(input_path),
                    "-ss", str(start),
                    "-t", str(duration),
                    "-c:a", "libmp3lame",
                    "-ar", "44100",
                    "-ac", "2",
                    str(seg_file),
                ]
                subprocess.run(cmd, check=True)
                
                # Do do dai thuc te
                actual_duration = get_audio_duration(seg_file)
                segment_durations.append(actual_duration)
    else:
        # Fallback: dung do dai tinh toan
        for start, end in parsed_segments:
            segment_durations.append(end - start)
    
    # Tinh lai timestamp moi
    new_timestamps = []
    current_position = 0.0
    
    for idx, seg_duration in enumerate(segment_durations):
        new_start = current_position
        new_end = current_position + seg_duration
        
        new_timestamps.append((format_time(new_start), format_time(new_end)))
        
        # Cap nhat vi tri hien tai
        current_position = new_end
        
        # Them do dai marker (tru segment cuoi)
        if idx < len(segment_durations) - 1:
            current_position += marker_duration
    
    return new_timestamps


def merge_segments_with_marker(
    input_path: Path,
    segments: list[list[str]],
    marker_path: Path,
    output_path: Path,
) -> tuple[Path, list[dict]]:
    """Ghep cac segment audio voi file marker chen vao giua moi segment.
    
    Args:
        input_path: Duong dan file audio goc
        segments: List cac segment dang [['MM:SS-MM:SS'], ...] hoac [['HH:MM:SS-HH:MM:SS'], ...]
        marker_path: Duong dan file audio marker
        output_path: Duong dan file output
        
    Returns:
        Tuple (duong dan file output, list dict mapping info)
        Moi dict co keys: 'original_segment', 'merged_start', 'merged_end'
    """
    import tempfile
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Parse segments thanh list (start, end)
    # Ho tro 2 format: [["00:38-00:41"], ...] hoac [["00:38", "00:41"], ...]
    parsed_segments = []
    for seg_list in segments:
        if len(seg_list) == 1 and isinstance(seg_list[0], str) and "-" in seg_list[0]:
            # Format 1: ["00:38-00:41"]
            seg_str = seg_list[0]
            start_str, end_str = seg_str.split("-", 1)
            original_str = seg_str
        elif len(seg_list) == 2 and isinstance(seg_list[0], str) and isinstance(seg_list[1], str):
            # Format 2: ["00:38", "00:41"]
            start_str, end_str = seg_list[0], seg_list[1]
            original_str = f"{start_str}-{end_str}"
        else:
            print(f"Warning: Invalid segment format: {seg_list}")
            continue
            
        start = parse_time(start_str)
        end = parse_time(end_str)
        parsed_segments.append({
            'original_str': original_str,
            'start': start,
            'end': end
        })
    
    # Tao temp dir de luu cac file temp
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        segment_files = []
        segment_durations = []
        
        # Cat tung segment tu file goc
        for idx, seg_info in enumerate(parsed_segments):
            start = seg_info['start']
            end = seg_info['end']
            duration = end - start
            seg_file = temp_path / f"seg_{idx:04d}.mp3"
            
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-i", str(input_path),
                "-ss", str(start),
                "-t", str(duration),
                "-c:a", "libmp3lame",
                "-ar", "44100",
                "-ac", "2",
                str(seg_file),
            ]
            subprocess.run(cmd, check=True)
            segment_files.append(seg_file)
            
            # Do do dai thuc te cua segment
            actual_duration = get_audio_duration(seg_file)
            segment_durations.append(actual_duration)
        
        # Chuyen doi marker audio sang cung format
        marker_converted = temp_path / "marker_converted.mp3"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(marker_path),
            "-c:a", "libmp3lame",
            "-ar", "44100",
            "-ac", "2",
            str(marker_converted),
        ]
        subprocess.run(cmd, check=True)
        
        # Do do dai marker thuc te
        marker_duration = get_audio_duration(marker_converted)
        
        # Tao file list cho ffmpeg concat
        concat_list = temp_path / "concat_list.txt"
        with open(concat_list, "w") as f:
            for idx, seg_file in enumerate(segment_files):
                f.write(f"file '{seg_file}'\n")
                # Chen marker sau moi segment tru segment cuoi
                if idx < len(segment_files) - 1:
                    f.write(f"file '{marker_converted}'\n")
        
        # Ghep tat ca bang ffmpeg concat
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c:a", "libmp3lame",
            "-q:a", "2",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
    
    # Tinh timestamps thuc te tu do dai da do va map voi original
    mapping_info = []
    current_position = 0.0
    
    for idx, seg_duration in enumerate(segment_durations):
        new_start = current_position
        new_end = current_position + seg_duration
        
        mapping_info.append({
            'original_segment': parsed_segments[idx]['original_str'],
            'merged_start': format_time(new_start, precise=True),
            'merged_end': format_time(new_end, precise=True)
        })
        
        current_position = new_end
        if idx < len(segment_durations) - 1:
            current_position += marker_duration
    
    print(f"Da ghep {len(parsed_segments)} segments voi marker -> {output_path}")
    return output_path, mapping_info


def map_transcript_to_segments(
    full_transcript: str,
    segments: list[list[str]],
    marker_transcript: str = "Đây là đoạn âm thanh dùng để phân tách",
    source_name: str = ""
) -> list[dict]:
    """Map full transcript ve tung segment dua tren marker text.
    
    Args:
        full_transcript: Van ban transcript day du tu Gemini
        segments: List cac segment goc [['MM:SS-MM:SS'], ...]
        marker_transcript: Cau van ban cua marker audio (dung de split)
        
    Returns:
        List dict [{'segment': '00:38-00:41', 'transcript': '...'}, ...]
    """
    # Chuan hoa text de split de dang hon
    marker_clean = marker_transcript.strip().lower()
    full_clean = full_transcript.strip()
    
    import re
    
    # Escape marker text cho regex va them flag ignore case
    # Dung pattern de split, them cac ky tu nhu dau cau co the dinh kem
    pattern = re.compile(re.escape(marker_clean) + r"[\.,\s]*", re.IGNORECASE)
    
    # Split text
    # re.split co the tra ve cac chuoi rong neu marker o dau/cuoi
    parts = pattern.split(full_clean)
    
    # Loc bo cac phan tu rong neu can thiet, nhung o day ta can giu thu tu
    # Nen ta chi strip whitespace
    parts = [p.strip() for p in parts if p.strip()]  # Loc bo phan tu chi toan khoang trang
    
    if len(parts) != len(segments):
        src = f" [{source_name}]" if source_name else ""
        print(
            f"Warning{src}: transcript parts ({len(parts)}) != segments ({len(segments)}). "
            "Transcript co the bi lech index."
        )

    mapped_results = []
    
    for i, seg_list in enumerate(segments):
        # Ho tro ca 2 format: [["00:38-00:41"]] va [["00:38", "00:41"]]
        if len(seg_list) == 1 and isinstance(seg_list[0], str) and "-" in seg_list[0]:
            seg_str = seg_list[0]  # Format 1: "00:38-00:41"
        elif len(seg_list) == 2 and isinstance(seg_list[0], str) and isinstance(seg_list[1], str):
            seg_str = f"{seg_list[0]}-{seg_list[1]}"  # Format 2: convert to "00:38-00:41"
        else:
            print(f"Warning: Invalid segment format: {seg_list}")
            continue
            
        transcript_part = ""
        
        # Lay part tuong ung neu co
        if i < len(parts):
            transcript_part = parts[i]
            # Xoa cac dau cau thua o dau/cuoi neu co
            transcript_part = transcript_part.strip('.,;:- ')
            
        mapped_results.append({
            "segment": seg_str,
            "transcript": transcript_part
        })
        
    return mapped_results


def export_finetune_dataset(
    input_audio_path: Path,
    transcript_data: list[dict],
    output_dir: Path,
    dataset_name: str = "dataset",
    metadata_format: str = "jsonl"
) -> Path:
    """Xuat data finetune (cat audio + metadata).
    
    Args:
        input_audio_path: File audio goc
        transcript_data: List dict tu ham map_transcript_to_segments 
                         [{'segment': 'MM:SS-MM:SS', 'transcript': 'text'}, ...]
        output_dir: Thu muc output
        dataset_name: Prefix cho ten file audio
        metadata_format: 'jsonl' hoac 'csv'
        
    Returns:
        Duong dan file metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(exist_ok=True)
    
    metadata_rows = []
    
    import json
    import csv
    
    print(f"Bat dau xu ly {len(transcript_data)} segments...")
    
    for idx, item in enumerate(transcript_data):
        seg_str = item['segment']
        text = item['transcript']
        
        # Parse time
        if "-" in seg_str:
            start_str, end_str = seg_str.split("-", 1)
            start = parse_time(start_str)
            end = parse_time(end_str)
        else:
            print(f"Skip segment invalid format: {seg_str}")
            continue
            
        duration = end - start
        if duration <= 0:
            print(f"Skip segment duration <= 0: {seg_str}")
            continue
            
        # Audio ID
        audio_id = f"{dataset_name}_{idx:04d}"
        audio_filename = f"{audio_id}.wav"
        output_audio_path = wavs_dir / audio_filename
        
        # Cat audio (convert sang wav mono 16kHz thuong dung cho STT)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(input_audio_path),
            "-ss", str(start),
            "-t", str(duration),
            "-ac", "1",       # Mono
            "-ar", "16000",   # 16kHz
            str(output_audio_path)
        ]
        subprocess.run(cmd, check=True)
        
        # Get actual duration fallback
        # actual_dur = get_audio_duration(output_audio_path)
        
        metadata_rows.append({
            "id": audio_id,
            "audio_filepath": str(output_audio_path.relative_to(output_dir)), # Relative path clean hon
            "text": text,
            "duration": duration,
            "original_segment": seg_str
        })
        
    # Ghi metadata
    if metadata_format == "jsonl":
        meta_path = output_dir / "metadata.jsonl"
        with open(meta_path, "w", encoding="utf-8") as f:
            for row in metadata_rows:
                # Format chung cho nhieu tool: id, audio, text
                json_line = {
                    "id": row["id"],
                    "audio": row["audio_filepath"],
                    "text": row["text"],
                    "duration": row["duration"]
                }
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
    else:
        meta_path = output_dir / "metadata.csv"
        with open(meta_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerow(["id", "audio", "text"])
            for row in metadata_rows:
                writer.writerow([row["id"], row["audio_filepath"], row["text"]])
                
    print(f"Da xuat dataset tai: {output_dir}")
    print(f"Metadata file: {meta_path}")
    return meta_path


def parse_speaker_info_from_label(label_csv_path: Path | str) -> dict[str, list[dict]]:
    """Parse thong tin speaker tu file data_label_by_hand.csv.
    
    Args:
        label_csv_path: Duong dan file CSV chua label
        
    Returns:
        Dict mapping ID (vd: 'hatinh1') -> list of speaker info dicts
        Moi dict co keys: speaker_id, speaker_name, speaker_gender, speaker_region, 
                         speaker_position, start_sec, end_sec
    """
    import csv
    import re

    def _norm_header(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip()).casefold()

    def _split_multiline(text: str) -> list[str]:
        return [item.strip() for item in re.split(r"\r?\n+", str(text or "")) if item.strip()]

    def _parse_timestamp_range(text: str) -> tuple[float, float, bool]:
        raw = str(text or "").strip()
        if not raw:
            return 0.0, 0.0, False

        normalized = (
            raw.replace("–", "-")
            .replace("—", "-")
            .replace("−", "-")
            .replace(":-", ":")
        )

        # Extract first 2 time tokens to avoid breakage on noisy separators.
        time_tokens = re.findall(r"\d{1,3}:\d{1,2}(?::\d{1,2}(?:\.\d+)?)?", normalized)
        if len(time_tokens) >= 2:
            try:
                return parse_time(time_tokens[0]), parse_time(time_tokens[1]), True
            except Exception:
                pass

        if "-" in normalized:
            parts = normalized.split("-", 1)
            if len(parts) == 2:
                try:
                    return parse_time(parts[0].strip()), parse_time(parts[1].strip()), True
                except Exception:
                    pass

        return 0.0, 0.0, False

    label_csv_path = Path(label_csv_path)
    result: dict[str, list[dict]] = {}

    speaker_pattern = re.compile(
        r"^\s*(\d+)\s*[_\-\s]+(male|female)\s*[_\-\s]+(central|south|north)\s*\(?\s*(.*?)\s*\)?\s*$",
        flags=re.IGNORECASE,
    )

    bad_speaker_lines = 0
    bad_timestamp_lines = 0
    mismatched_row_counts = 0

    with open(label_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames or []
        header_lookup = {_norm_header(name): name for name in fieldnames if name}

        def _get_col(row: dict, aliases: list[str]) -> str:
            for alias in aliases:
                if alias in row:
                    return str(row.get(alias) or "")
                mapped = header_lookup.get(_norm_header(alias))
                if mapped and mapped in row:
                    return str(row.get(mapped) or "")
            return ""

        for row in reader:
            file_id = _get_col(row, ["ID", "id"]).strip()
            if not file_id:
                continue

            speaker_col = _get_col(
                row,
                ["Trình tự người nói", "Trinh tu nguoi noi", "speaker", "speakers"],
            )
            timestamp_col = _get_col(
                row,
                ["Timestamp", "TimeStamp", "timestamp", "Thời gian"],
            )

            speakers = _split_multiline(speaker_col)
            timestamps = _split_multiline(timestamp_col)

            if speakers and timestamps and len(speakers) != len(timestamps):
                mismatched_row_counts += 1

            speaker_list: list[dict] = []
            for i, speaker_line in enumerate(speakers):
                match = speaker_pattern.match(speaker_line)
                if not match:
                    bad_speaker_lines += 1
                    continue

                speaker_num = match.group(1)
                gender = match.group(2).lower()
                region = match.group(3).lower()
                name_pos = match.group(4).strip()

                # Tach name va position neu co " - "
                if " - " in name_pos:
                    name, position = name_pos.split(" - ", 1)
                    name = name.strip()
                    position = position.strip()
                else:
                    name = name_pos
                    position = ""

                ts = timestamps[i] if i < len(timestamps) else ""
                start_sec = 0.0
                end_sec = 0.0
                if ts:
                    start_sec, end_sec, ok = _parse_timestamp_range(ts)
                    if not ok:
                        bad_timestamp_lines += 1

                speaker_list.append(
                    {
                        "speaker_id": int(speaker_num),
                        "speaker_name": name,
                        "speaker_gender": gender,
                        "speaker_region": region,
                        "speaker_position": position,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                    }
                )

            result[file_id] = speaker_list

    if bad_speaker_lines or bad_timestamp_lines or mismatched_row_counts:
        print(
            "Warning: Label CSV parsed with anomalies "
            f"(mismatch_rows={mismatched_row_counts}, "
            f"bad_speaker_lines={bad_speaker_lines}, "
            f"bad_timestamp_lines={bad_timestamp_lines})"
        )

    return result


def calculate_overlap_ratio(seg_start: float, seg_end: float, 
                            spk_start: float, spk_end: float) -> float:
    """Tinh ty le overlap cua segment voi khoang thoi gian speaker.
    
    Args:
        seg_start, seg_end: Timestamp segment (seconds)
        spk_start, spk_end: Timestamp speaker (seconds)
        
    Returns:
        Ty le overlap (0.0 - 1.0)
    """
    seg_duration = seg_end - seg_start
    if seg_duration <= 0:
        return 0.0
    
    overlap_start = max(seg_start, spk_start)
    overlap_end = min(seg_end, spk_end)
    overlap_duration = max(0, overlap_end - overlap_start)
    
    return overlap_duration / seg_duration


def match_segment_to_speaker(
    segment_start_sec: float,
    segment_end_sec: float,
    speaker_list: list[dict],
    threshold: float = 0.70
) -> tuple[dict | None, float]:
    """Tim speaker phu hop cho segment dua tren timestamp voi threshold overlap.
    
    Args:
        segment_start_sec: Thoi gian bat dau segment trong audio GOC (seconds)
        segment_end_sec: Thoi gian ket thuc segment trong audio GOC (seconds)
        speaker_list: List dict speaker info voi start_sec, end_sec
        threshold: Nguong overlap toi thieu (default 70%)
        
    Returns:
        Tuple (speaker_info, overlap_ratio) hoac (None, 0.0) neu khong tim thay
    """
    best_match = None
    best_ratio = 0.0
    
    for spk_info in speaker_list:
        spk_start = spk_info.get('start_sec', 0.0)
        spk_end = spk_info.get('end_sec', 0.0)
        
        if spk_start >= spk_end:
            continue
        
        ratio = calculate_overlap_ratio(segment_start_sec, segment_end_sec, 
                                        spk_start, spk_end)
        
        if ratio >= threshold and ratio > best_ratio:
            best_ratio = ratio
            best_match = spk_info
            
    return best_match, best_ratio


def create_hf_dataset(
    segment_map: dict[str, list[list[str]]],
    transcript_map: dict[str, str],
    output_dir: Path,
    dataset_name: str = "my_audio_dataset",
    push_to_hub: bool = False,
    hub_repo_id: str = None,
    speaker_info_map: dict[str, list[dict]] = None,
    segment_original_map: dict[str, list[list[str]]] = None,
    audio_offset_map: dict[str, float] = None
):
    """Tao HuggingFace Dataset tu nhieu file audio va transcript.
    
    Args:
        segment_map: Dict { 'path/to/audio1.mp3': [['00:00-00:10'], ...], ... } - timestamp SAU edit/marker
        transcript_map: Dict { 'path/to/audio1.mp3': 'Full transcript ...', ... }
        output_dir: Thu muc output
        dataset_name: Ten dataset
        push_to_hub: Co push len HF Hub hay khong
        hub_repo_id: Repo ID (vd: 'username/dataset_name')
        speaker_info_map: Dict speaker info tu parse_speaker_info_from_label
        segment_original_map: Dict { audio_path: [['start', 'end'], ...] } - timestamp GOC trong audio nho
        audio_offset_map: Dict { audio_path: offset_seconds } - offset cua audio nho trong audio goc
        
    Returns:
        Dataset object (datasets.Dataset)
    """
    try:
        from datasets import Dataset, Audio, Features, Value
    except ImportError:
        print("Vui long cai dat 'datasets': pip install datasets")
        return None
        
    output_dir = Path(output_dir)
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    
    all_metadata = []
    
    print(f"Bat dau xu ly {len(segment_map)} files...")
    
    for audio_path_str, segments in segment_map.items():
        audio_path = Path(audio_path_str)
        if not audio_path.exists():
            print(f"Warning: File not found {audio_path}")
            continue
            
        full_transcript = transcript_map.get(audio_path_str, "")
        if not full_transcript:
            print(f"Warning: No transcript for {audio_path.name}")
            # Van tiep tuc xu ly nhung transcript se rong (hoac tuy logic)
        
        # 1. Map transcript theo timestamp SAU edit/marker
        mapped_data = map_transcript_to_segments(
            full_transcript,
            segments,
            source_name=audio_path.name
        )
        
        # 2. Process tung segment
        file_prefix = audio_path.stem
        
        for idx, item in enumerate(mapped_data):
            seg_str = item['segment']
            text = item['transcript']
            
            # Parse time (segment sau edit, dung cho fallback)
            if "-" not in seg_str:
                continue
            start_str, end_str = seg_str.split("-", 1)
            seg_after_start = parse_time(start_str)
            seg_after_end = parse_time(end_str)

            # Timestamp goc (truoc edit) de cat audio + match speaker
            seg_orig_start = seg_after_start
            seg_orig_end = seg_after_end
            seg_orig_str = seg_str
            seg_orig_list = segment_original_map.get(audio_path_str, []) if segment_original_map else []
            if idx < len(seg_orig_list):
                orig_ts = seg_orig_list[idx]
                if len(orig_ts) >= 2:
                    seg_orig_start = parse_time(orig_ts[0])
                    seg_orig_end = parse_time(orig_ts[1])
                    seg_orig_str = f"{orig_ts[0]}-{orig_ts[1]}"

            duration = seg_orig_end - seg_orig_start
            
            if duration <= 0: continue
            
            # Unique ID: filename_segmentIdx
            audio_id = f"{file_prefix}_{idx:04d}"
            audio_filename = f"{audio_id}.wav"
            output_audio_path = wavs_dir / audio_filename
            
            # Cut audio
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(audio_path),
                "-ss", str(seg_orig_start), "-t", str(duration),
                "-ac", "1", "-ar", "16000",
                str(output_audio_path)
            ]
            subprocess.run(cmd, check=True)
            
            # Tim speaker info (dung segment_original + offset de tinh timestamp trong audio goc)
            speaker_id = ""
            speaker_name = ""
            speaker_gender = ""
            speaker_region = ""
            speaker_position = ""
            overlap_ratio = 0.0
            
            if speaker_info_map:
                # Tim file_id tu ten file (vd: hatinh1_1 -> hatinh1)
                file_id = file_prefix.rsplit('_', 1)[0] if '_' in file_prefix else file_prefix
                speaker_list = speaker_info_map.get(file_id, [])
                
                if speaker_list:
                    offset = audio_offset_map.get(audio_path_str, 0.0) if audio_offset_map else 0.0

                    # Tinh timestamp trong audio GOC = segment_original + offset
                    abs_start = seg_orig_start + offset
                    abs_end = seg_orig_end + offset
                    
                    # Match speaker voi threshold 70%
                    speaker_info, ratio = match_segment_to_speaker(abs_start, abs_end, speaker_list)
                    if speaker_info:
                        speaker_id = speaker_info.get('speaker_id', '')
                        speaker_name = speaker_info.get('speaker_name', '')
                        speaker_gender = speaker_info.get('speaker_gender', '')
                        speaker_region = speaker_info.get('speaker_region', '')
                        speaker_position = speaker_info.get('speaker_position', '')
                        overlap_ratio = ratio
            
            # Metadata record
            # Luu absolute path de load dataset de dang hon
            all_metadata.append({
                "audio_id": audio_id,
                "audio": str(output_audio_path.absolute()), 
                "transcript": text,
                "duration": duration,
                "original_file": audio_path.name,
                "original_segment": seg_orig_str,
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "speaker_gender": speaker_gender,
                "speaker_region": speaker_region,
                "speaker_position": speaker_position,
                "overlap_ratio": overlap_ratio
            })
            
    # 3. Create HF Dataset
    # Define features ban dau (audio la string path)
    features_initial = Features({
        "audio_id": Value("string"),
        "audio": Value("string"), # Tam thoi de string de tranh load audio ngay
        "transcript": Value("string"),
        "duration": Value("float32"),
        "original_file": Value("string"),
        "original_segment": Value("string"),
        "speaker_id": Value("int32"),
        "speaker_name": Value("string"),
        "speaker_gender": Value("string"),
        "speaker_region": Value("string"),
        "speaker_position": Value("string"),
        "overlap_ratio": Value("float32"),
    })
    
    # Create dict valid for Dataset.from_dict
    data_dict = {
        "audio_id": [x["audio_id"] for x in all_metadata],
        "audio": [x["audio"] for x in all_metadata],
        "transcript": [x["transcript"] for x in all_metadata],
        "duration": [x["duration"] for x in all_metadata],
        "original_file": [x["original_file"] for x in all_metadata],
        "original_segment": [x["original_segment"] for x in all_metadata],
        "speaker_id": [x["speaker_id"] if x["speaker_id"] else 0 for x in all_metadata],
        "speaker_name": [x["speaker_name"] for x in all_metadata],
        "speaker_gender": [x["speaker_gender"] for x in all_metadata],
        "speaker_region": [x["speaker_region"] for x in all_metadata],
        "speaker_position": [x["speaker_position"] for x in all_metadata],
        "overlap_ratio": [x["overlap_ratio"] for x in all_metadata],
    }
    
    dataset = Dataset.from_dict(data_dict, features=features_initial)
    
    # Cast column audio sang Audio feature
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Save to disk (HF format)
    dataset_save_path = output_dir / "hf_dataset"
    dataset.save_to_disk(dataset_save_path)
    print(f"Da tao HF Dataset tai: {dataset_save_path}")
    
    # Export metadata.csv de de xem transcript
    import csv
    metadata_csv_path = output_dir / "metadata.csv"
    with open(metadata_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_id", "audio_path", "transcript", "duration", 
                         "speaker_id", "speaker_name", "speaker_gender", 
                         "speaker_region", "speaker_position", "overlap_ratio"])
        for item in all_metadata:
            writer.writerow([
                item["audio_id"],
                item["audio"], # Absolute path
                item["transcript"],
                item["duration"],
                item["speaker_id"] if item["speaker_id"] else "",
                item["speaker_name"],
                item["speaker_gender"],
                item["speaker_region"],
                item["speaker_position"],
                f"{item['overlap_ratio']:.2f}" if item["overlap_ratio"] > 0 else ""
            ])
    print(f"Da xuat file metadata CSV tai: {metadata_csv_path}")
    
    if push_to_hub and hub_repo_id:
        print(f"Pushing to Hub: {hub_repo_id}...")
        dataset.push_to_hub(hub_repo_id, private=True)
        
    return dataset


def parse_segments_from_transcript(
    transcript_path: Path | str,
    exclude_marker: bool = True,
    marker_text: str = "đây là đoạn âm thanh dùng để phân tách"
) -> list[list[str]]:
    """Trich xuat segments tu transcript file co timestamp.
    
    Parse transcript voi format:
        MM:SS - MM:SS: Speaker
        Noi dung...
        
    hoac:
        HH:MM:SS - HH:MM:SS: Speaker
        Noi dung...
    
    Args:
        transcript_path: Duong dan file transcript
        exclude_marker: Neu True, bo qua cac segment chua marker text
        marker_text: Text cua marker audio (case-insensitive)
        
    Returns:
        List segments dang [["MM:SS.mmm", "MM:SS.mmm"], ...]
    """
    import re
    
    transcript_path = Path(transcript_path)
    
    # Pattern: 00:00 - 00:03 hoac 01:23:45 - 01:45:30
    # Co the co milliseconds: 00:37.949 - 00:40.514
    time_pattern = re.compile(
        r'^(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d{1,3})?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d{1,3})?)'
    )
    
    segments = []
    current_segment = None
    current_content_lines = []
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        match = time_pattern.match(line_stripped)
        if match:
            # Luu segment truoc do (neu co)
            if current_segment is not None:
                content = ' '.join(current_content_lines).strip()
                # Kiem tra exclude marker
                if exclude_marker and marker_text.lower() in content.lower():
                    pass  # Bo qua segment marker
                else:
                    segments.append(current_segment)
            
            # Bat dau segment moi
            start_time = match.group(1)
            end_time = match.group(2)
            current_segment = [start_time, end_time]
            
            # Noi dung sau timestamp tren cung dong
            rest_of_line = line_stripped[match.end():].strip()
            # Bo dau ":" hoac ": " o dau neu co
            rest_of_line = rest_of_line.lstrip(':').strip()
            current_content_lines = [rest_of_line] if rest_of_line else []
        else:
            # Dong tiep theo cua segment hien tai
            if current_segment is not None:
                current_content_lines.append(line_stripped)
    
    # Xu ly segment cuoi cung
    if current_segment is not None:
        content = ' '.join(current_content_lines).strip()
        if exclude_marker and marker_text.lower() in content.lower():
            pass
        else:
            segments.append(current_segment)
    
    return segments


def parse_segments_from_gemini_output(
    gemini_output: str,
    exclude_marker: bool = True,
    marker_text: str = "đây là đoạn âm thanh dùng để phân tách"
) -> list[list[str]]:
    """Trich xuat segments tu chuoi output cua Gemini API.
    
    Tuong tu parse_segments_from_transcript nhung nhan input la string.
    
    Args:
        gemini_output: Chuoi transcript tu Gemini
        exclude_marker: Neu True, bo qua cac segment chua marker text
        marker_text: Text cua marker audio
        
    Returns:
        List segments dang [["MM:SS.mmm", "MM:SS.mmm"], ...]
    """
    import re
    import io
    
    time_pattern = re.compile(
        r'^(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d{1,3})?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d{1,3})?)'
    )
    
    segments = []
    current_segment = None
    current_content_lines = []
    
    for line in io.StringIO(gemini_output):
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        match = time_pattern.match(line_stripped)
        if match:
            if current_segment is not None:
                content = ' '.join(current_content_lines).strip()
                if exclude_marker and marker_text.lower() in content.lower():
                    pass
                else:
                    segments.append(current_segment)
            
            start_time = match.group(1)
            end_time = match.group(2)
            current_segment = [start_time, end_time]
            
            rest_of_line = line_stripped[match.end():].strip().lstrip(':').strip()
            current_content_lines = [rest_of_line] if rest_of_line else []
        else:
            if current_segment is not None:
                current_content_lines.append(line_stripped)
    
    if current_segment is not None:
        content = ' '.join(current_content_lines).strip()
        if exclude_marker and marker_text.lower() in content.lower():
            pass
        else:
            segments.append(current_segment)
    
    return segments


def main():
    parser = argparse.ArgumentParser(description="Tach audio thanh cac file nho")
    parser.add_argument("--input", "-i", required=True, help="File audio goc")
    parser.add_argument("--output-dir", "-o", default="outputs", help="Thu muc luu file output")
    parser.add_argument(
        "--segments", "-s",
        help="Cac doan can tach, format: 'start1-end1,start2-end2,...' (time: HH:MM:SS, MM:SS, hoac seconds)"
    )
    parser.add_argument(
        "--name", "-n",
        help="Ten file output (khong can extension), mac dinh dung ten file goc"
    )
    args = parser.parse_args()
    
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    
    if not input_path.exists():
        raise FileNotFoundError(f"Khong tim thay file: {input_path}")
    
    # Parse segments tu argument hoac dung config mac dinh
    if args.segments:
        segments = parse_segments(args.segments)
    else:
        # Config mac dinh - sua theo nhu cau
        segments = [
            (0, 60),       # 0:00 - 1:00
            (60, 120),     # 1:00 - 2:00
        ]
    
    if not segments:
        print("Khong co segment nao duoc config!")
        return
    
    print(f"Input: {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"Segments: {len(segments)}")
    print("-" * 40)
    
    output_files = split_audio(input_path, output_dir, segments, args.name)
    
    print("-" * 40)
    print(f"Da tach thanh {len(output_files)} file:")
    for f in output_files:
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
