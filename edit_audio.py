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
