"""
Batch diarization cho cac file audio trong thu muc.
Output: file .rttm cung ten trong thu muc output.
"""

import os
from pathlib import Path

from src.pipeline import (
    build_prompt,
    call_gemini_stream,
    ensure_logging,
    normalize_output,
    resolve_api_key,
    sanitize_file_id,
)

# Config
INPUT_DIR = "outputs"
OUTPUT_DIR = "outputs_diarization"
API_KEY = "AQ.Ab8RN6JcHuQ0La4J2zAtonlVnXTguIa9RrPZTi9_u70EYMNnMA"
MODEL = "gemini-3-pro-preview"
TIME_FORMAT = "hms"

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}


def process_file(audio_path: Path, output_dir: Path) -> None:
    """Xu ly mot file audio."""
    file_id = sanitize_file_id(audio_path.stem)
    prompt = build_prompt(file_id, 0.0)
    
    print(f"Processing: {audio_path.name}")
    response = call_gemini_stream(
        api_key=API_KEY,
        model=MODEL,
        prompt=prompt,
        audio_path=audio_path,
    )
    
    lines = normalize_output(response, file_id, 0.0, TIME_FORMAT)
    
    output_path = output_dir / f"{audio_path.stem}.rttm"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {output_path.name} ({len(lines)} lines)")


def main():
    ensure_logging()
    
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    audio_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    ]
    
    if not audio_files:
        print(f"Khong tim thay file audio trong {input_dir}")
        return
    
    print(f"Tim thay {len(audio_files)} file audio")
    print(f"Model: {MODEL}")
    print("-" * 40)
    
    for audio_path in sorted(audio_files):
        try:
            process_file(audio_path, output_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print("-" * 40)
    print("Done!")


if __name__ == "__main__":
    main()
