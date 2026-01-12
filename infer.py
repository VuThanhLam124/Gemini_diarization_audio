import argparse
from pathlib import Path

from src.pipeline import (
    ensure_logging,
    format_output,
    resolve_api_key,
    resolve_audio_input,
    run_pipeline,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gemini Vertex AI diarization + transcription"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--youtube-url", help="YouTube video URL")
    input_group.add_argument("--audio-file", help="Local audio file path")

    parser.add_argument("--api-key", help="Gemini Vertex AI API key")
    parser.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="Vertex AI Gemini model name",
    )
    parser.add_argument(
        "--segment-seconds",
        type=int,
        default=600,
        help="Segment length in seconds for long audio",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory for downloaded audio",
    )
    parser.add_argument("--file-id", help="Override file_id in output")
    parser.add_argument("--output", help="Write output to file")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    ensure_logging()
    api_key = resolve_api_key(args.api_key)
    audio_path, file_id = resolve_audio_input(
        youtube_url=args.youtube_url,
        audio_file=args.audio_file,
        output_dir=args.output_dir,
        file_id=args.file_id,
    )

    lines = run_pipeline(
        api_key=api_key,
        model=args.model,
        audio_path=Path(audio_path),
        file_id=file_id,
        segment_seconds=args.segment_seconds,
    )
    output_text = format_output(lines)

    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
