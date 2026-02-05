import argparse
import json
import re
from pathlib import Path
from edit_audio import create_hf_dataset, parse_speaker_info_from_label, parse_time


def extract_audio_id_and_part(audio_path: str) -> tuple[str, int]:
    """Trich xuat audio ID goc va so thu tu part tu ten file.
    
    VD: 'hatinh1_1.mp3' -> ('hatinh1', 1)
        'hatinh1_2.mp3' -> ('hatinh1', 2)
    """
    stem = Path(audio_path).stem  # hatinh1_1
    match = re.match(r'^(.+)_(\d+)$', stem)
    if match:
        return match.group(1), int(match.group(2))
    return stem, 0


def calculate_audio_offsets(config_data: dict, json_path: Path) -> dict[str, float]:
    """Tinh offset cua tung audio nho trong audio goc.
    
    Offset duoc tinh bang cach lay timestamp cuoi cung cua audio nho truoc do.
    VD: hatinh1_1 ket thuc tai 59:51.554 -> hatinh1_2 co offset = 59:51.554
    
    Args:
        config_data: Dict config tu JSON
        json_path: Path den file JSON
        
    Returns:
        Dict { audio_path: offset_seconds }
    """
    # Nhom audio theo ID goc
    audio_groups = {}  # { 'hatinh1': [(part_num, audio_path, info), ...] }
    
    for audio_str, info in config_data.items():
        audio_path = Path(audio_str)
        if not audio_path.exists():
            audio_path_rel = json_path.parent / audio_str
            if audio_path_rel.exists():
                audio_path = audio_path_rel
        
        audio_id, part_num = extract_audio_id_and_part(str(audio_path))
        
        if audio_id not in audio_groups:
            audio_groups[audio_id] = []
        audio_groups[audio_id].append((part_num, str(audio_path), info))
    
    # Tinh offset cho tung audio nho
    offset_map = {}
    
    for audio_id, parts in audio_groups.items():
        # Sap xep theo part number
        parts.sort(key=lambda x: x[0])
        
        cumulative_offset = 0.0
        
        for i, (part_num, audio_path, info) in enumerate(parts):
            # Offset cua part hien tai
            offset_map[audio_path] = cumulative_offset
            
            # Tinh offset cho part tiep theo bang timestamp cuoi cung cua segment_original hien tai
            seg_orig = info.get('segment_original', [])
            if seg_orig and len(seg_orig) > 0:
                last_seg = seg_orig[-1]
                if len(last_seg) >= 2:
                    # Lay end time cua segment cuoi cung lam offset cho part tiep theo
                    last_end = parse_time(last_seg[1])
                    cumulative_offset += last_end
    
    return offset_map


def main():
    parser = argparse.ArgumentParser(description="Tao HF Dataset tu config JSON")
    parser.add_argument("--input-json", "-i", required=True, help="Path den file JSON chua cau hinh (audio, segments, transcript_path)")
    parser.add_argument("--output-dir", "-o", default="outputs_dataset", help="Thu muc output")
    parser.add_argument("--dataset-name", "-n", default="my_dataset", help="Ten dataset")
    parser.add_argument("--label-csv", "-l", default="data_label_by_hand.csv", help="Path den file CSV chua speaker info (data_label_by_hand.csv)")
    
    args = parser.parse_args()
    
    json_path = Path(args.input_json)
    if not json_path.exists():
        print(f"Error: Khong tim thay file JSON {json_path}")
        return
    
    # Load speaker info tu file label CSV
    speaker_info_map = None
    label_csv_path = Path(args.label_csv)
    if not label_csv_path.exists():
        label_csv_rel = json_path.parent / args.label_csv
        if label_csv_rel.exists():
            label_csv_path = label_csv_rel
    
    if label_csv_path.exists():
        print(f"Loading speaker info from: {label_csv_path}")
        speaker_info_map = parse_speaker_info_from_label(label_csv_path)
        print(f"  Found info for {len(speaker_info_map)} audio IDs")
    else:
        print(f"Warning: Label CSV not found: {args.label_csv} - speaker info se trong")
        
    with open(json_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    
    # Tinh audio offset cho tung audio nho
    print("Calculating audio offsets...")
    audio_offset_map = calculate_audio_offsets(config_data, json_path)
    for audio_path, offset in audio_offset_map.items():
        print(f"  {Path(audio_path).name}: offset = {offset:.2f}s")
        
    # Parse config
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
        
        # Get segments
        segments = info.get("segments", [])
        if not segments:
            print(f"Warning: No segments for {audio_str}")
            continue
        
        # Get segment_original (cung index voi segments)
        seg_orig = info.get("segment_original", [])
            
        # Get transcript
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
        
        # Add to maps
        audio_path_key = str(audio_path)
        segment_map[audio_path_key] = segments
        transcript_map[audio_path_key] = full_transcript
        segment_original_map[audio_path_key] = seg_orig
        
    if not segment_map:
        print("Error: No valid data found to process.")
        return

    # Create dataset
    # Dung segment_original_map de cat audio (timestamp goc, tranh marker)
    # Dung segment_map de map transcript (timestamp sau marker)  
    print(f"\nProcessing {len(segment_original_map)} audio files...")
    create_hf_dataset(
        segment_map=segment_original_map,  # Cat audio bang timestamp goc
        transcript_map=transcript_map,
        output_dir=Path(args.output_dir),
        dataset_name=args.dataset_name,
        speaker_info_map=speaker_info_map,
        segment_original_map=segment_original_map,  # Match speaker
        audio_offset_map=audio_offset_map
    )

if __name__ == "__main__":
    main()

