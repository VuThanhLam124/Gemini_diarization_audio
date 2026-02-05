"""
Script them speaker information vao dataset.
Mapping segment -> absolute timestamp -> doi chieu voi CSV de lay speaker.

Usage:
    python add_speaker_info.py --csv data_label_by_hand.csv --config config_sample.json --output dataset_with_speaker
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Optional
import subprocess


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


def parse_time_to_seconds(time_str: str) -> float:
    """Parse time string (H:MM:SS, MM:SS, hoac M:SS) thanh seconds.
    
    Vi du:
        "0:16" -> 16
        "14:41" -> 881
        "1:50:39" -> 6639
        "2:09:00" -> 7740
    """
    time_str = time_str.strip()
    parts = time_str.split(":")
    
    if len(parts) == 2:
        # MM:SS hoac M:SS
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        # H:MM:SS hoac HH:MM:SS
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        return float(time_str)


def parse_segment_time(seg_str: str) -> float:
    """Parse segment time (MM:SS.ms) thanh seconds.
    
    Vi du:
        "00:02.61" -> 2.61
        "59:53.01" -> 3593.01
        "01:02:29.36" -> 3749.36
    """
    seg_str = seg_str.strip()
    
    # Xu ly format HH:MM:SS.ms
    if seg_str.count(":") == 2:
        parts = seg_str.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    
    # Format MM:SS.ms
    parts = seg_str.split(":")
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    
    return float(seg_str)


def compute_audio_offsets(outputs_dir: Path) -> dict:
    """Tinh offset cua moi file audio nho trong video goc.
    
    Returns:
        Dict {'hatinh1': {'hatinh1_1.mp3': 0, 'hatinh1_2.mp3': 3593.016, ...}, ...}
    """
    offsets = {}
    
    # Group files by video_id (hatinh1, hatinh2, ...)
    files_by_video = {}
    for f in sorted(outputs_dir.glob("*.mp3")):
        # Extract video_id: hatinh1_1.mp3 -> hatinh1
        match = re.match(r'^([a-zA-Z]+\d+)_\d+\.mp3$', f.name)
        if match:
            video_id = match.group(1)
            if video_id not in files_by_video:
                files_by_video[video_id] = []
            files_by_video[video_id].append(f)
    
    # Tinh offset cho moi file
    for video_id, files in files_by_video.items():
        # Sort theo so thu tu (_1, _2, _3, ...)
        files_sorted = sorted(files, key=lambda x: int(re.search(r'_(\d+)\.mp3$', x.name).group(1)))
        
        offsets[video_id] = {}
        current_offset = 0.0
        
        for f in files_sorted:
            offsets[video_id][f.name] = current_offset
            duration = get_audio_duration(f)
            current_offset += duration
    
    return offsets


def parse_csv_speaker_data(csv_path: Path) -> dict:
    """Parse CSV thanh speaker lookup table.
    
    Returns:
        Dict {
            'hatinh1': [
                {'speaker': '1_male_central (Nguyen Hong Linh)', 'start': 43, 'end': 243},
                ...
            ],
            ...
        }
    """
    speaker_data = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse CSV - xu ly multiline cells
    lines = content.split('\r\n')
    
    current_row = None
    rows = []
    
    for line in lines:
        if not line.strip():
            continue
        
        # Dem so dau " trong dong
        quote_count = line.count('"')
        
        if current_row is None:
            if quote_count % 2 == 0:
                # Dong hoan chinh
                rows.append(line)
            else:
                # Bat dau dong multiline
                current_row = line
        else:
            current_row += '\n' + line
            quote_count = current_row.count('"')
            if quote_count % 2 == 0:
                # Ket thuc dong multiline
                rows.append(current_row)
                current_row = None
    
    # Parse header
    if not rows:
        return speaker_data
    
    # Skip header row
    for row in rows[1:]:
        # Parse CSV row (xu ly quoted fields)
        fields = []
        current_field = ''
        in_quotes = False
        
        for char in row:
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                fields.append(current_field)
                current_field = ''
            else:
                current_field += char
        fields.append(current_field)
        
        if len(fields) < 9:
            continue
        
        video_id = fields[8].strip()  # Cot ID
        if not video_id:
            continue
        
        speakers_raw = fields[5].strip()  # Cot Trinh tu nguoi noi
        timestamps_raw = fields[6].strip()  # Cot Timestamp
        
        # Parse speakers va timestamps
        speakers = [s.strip() for s in speakers_raw.split('\n') if s.strip()]
        timestamps = [t.strip() for t in timestamps_raw.split('\n') if t.strip()]
        
        if video_id not in speaker_data:
            speaker_data[video_id] = []
        
        # Match speaker voi timestamp
        for i, speaker in enumerate(speakers):
            if i >= len(timestamps):
                break
            
            ts = timestamps[i]
            # Parse timestamp range "0:43 - 4:03"
            if ' - ' in ts:
                start_str, end_str = ts.split(' - ', 1)
                start_sec = parse_time_to_seconds(start_str)
                end_sec = parse_time_to_seconds(end_str)
                
                speaker_data[video_id].append({
                    'speaker': speaker,
                    'start': start_sec,
                    'end': end_sec
                })
    
    return speaker_data


def find_speaker(speaker_data: dict, video_id: str, absolute_time_sec: float) -> Optional[str]:
    """Tim speaker dua tren absolute timestamp.
    
    Args:
        speaker_data: Dict speaker data tu parse_csv_speaker_data
        video_id: ID video (hatinh1, hatinh2, ...)
        absolute_time_sec: Thoi gian tuyet doi trong video goc (seconds)
    
    Returns:
        Speaker string hoac None neu khong tim thay
    """
    if video_id not in speaker_data:
        return None
    
    for entry in speaker_data[video_id]:
        # Cho phep sai so 2 giay
        if entry['start'] - 2 <= absolute_time_sec <= entry['end'] + 2:
            return entry['speaker']
    
    return None


def extract_speaker_info(speaker_str: str) -> dict:
    """Trich xuat thong tin tu speaker string.
    
    Vi du: "1_male_central (Nguyen Hong Linh)" 
    -> {'id': 1, 'gender': 'male', 'region': 'central', 'name': 'Nguyen Hong Linh'}
    """
    result = {
        'speaker_id': None,
        'gender': None,
        'region': None,
        'speaker_name': None,
        'speaker_full': speaker_str
    }
    
    if not speaker_str:
        return result
    
    # Pattern: 1_male_central (Ten)
    match = re.match(r'^(\d+)_(male|female)_(south|central|north)\s*\(([^)]+)\)', speaker_str)
    if match:
        result['speaker_id'] = int(match.group(1))
        result['gender'] = match.group(2)
        result['region'] = match.group(3)
        result['speaker_name'] = match.group(4).strip()
    else:
        # Try simpler pattern: 1_male_central(Ten)
        match = re.match(r'^(\d+)_(male|female)_(south|central|north)\(([^)]+)\)', speaker_str)
        if match:
            result['speaker_id'] = int(match.group(1))
            result['gender'] = match.group(2)
            result['region'] = match.group(3)
            result['speaker_name'] = match.group(4).strip()
    
    return result


def add_speaker_to_dataset(
    config_path: Path,
    csv_path: Path,
    outputs_dir: Path,
    output_path: Path = None
):
    """Them speaker info vao config JSON.
    
    Args:
        config_path: Duong dan config_sample.json
        csv_path: Duong dan data_label_by_hand.csv
        outputs_dir: Thu muc chua cac file audio
        output_path: Duong dan file output (mac dinh ghi de config_path)
    """
    print("1. Loading config...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("2. Computing audio offsets...")
    offsets = compute_audio_offsets(outputs_dir)
    print(f"   Found {len(offsets)} videos: {list(offsets.keys())}")
    
    print("3. Parsing CSV speaker data...")
    speaker_data = parse_csv_speaker_data(csv_path)
    print(f"   Found speaker data for: {list(speaker_data.keys())}")
    
    # Debug: in ra speaker data
    for vid, entries in speaker_data.items():
        print(f"   {vid}: {len(entries)} entries")
    
    print("4. Adding speaker info to segments...")
    
    total_segments = 0
    matched_segments = 0
    
    for audio_path_str, info in config.items():
        segments = info.get('segments', [])
        if not segments:
            continue
        
        # Extract video_id va file_name
        file_name = Path(audio_path_str).name
        match = re.match(r'^([a-zA-Z]+\d+)_\d+\.mp3$', file_name)
        if not match:
            print(f"   Skipping {file_name} - invalid format")
            continue
        
        video_id = match.group(1)
        
        if video_id not in offsets or file_name not in offsets[video_id]:
            print(f"   Skipping {file_name} - no offset data")
            continue
        
        file_offset = offsets[video_id][file_name]
        
        # Them speaker_info vao moi segment
        if 'segment_speakers' not in info:
            info['segment_speakers'] = []
        
        for seg in segments:
            total_segments += 1
            
            # Parse segment start time
            if len(seg) == 2:
                start_str = seg[0]
            elif len(seg) == 1 and '-' in seg[0]:
                start_str = seg[0].split('-')[0]
            else:
                continue
            
            segment_start = parse_segment_time(start_str)
            absolute_time = file_offset + segment_start
            
            # Tim speaker
            speaker = find_speaker(speaker_data, video_id, absolute_time)
            
            if speaker:
                matched_segments += 1
                speaker_info = extract_speaker_info(speaker)
            else:
                speaker_info = extract_speaker_info(None)
            
            info['segment_speakers'].append(speaker_info)
    
    print(f"   Matched {matched_segments}/{total_segments} segments")
    
    # Luu config moi
    output_file = output_path or config_path
    print(f"5. Saving to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    
    print("Done!")
    return config


def main():
    parser = argparse.ArgumentParser(description="Them speaker info vao dataset")
    parser.add_argument("--csv", "-c", required=True, help="Path den file CSV chua speaker data")
    parser.add_argument("--config", "-i", required=True, help="Path den config JSON")
    parser.add_argument("--outputs-dir", "-d", default="outputs", help="Thu muc chua audio files")
    parser.add_argument("--output", "-o", help="Path file output (mac dinh ghi de input)")
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    csv_path = Path(args.csv)
    outputs_dir = Path(args.outputs_dir)
    output_path = Path(args.output) if args.output else None
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    if not outputs_dir.exists():
        print(f"Error: Outputs directory not found: {outputs_dir}")
        return
    
    add_speaker_to_dataset(config_path, csv_path, outputs_dir, output_path)


if __name__ == "__main__":
    main()
