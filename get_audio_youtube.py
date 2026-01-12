import argparse

from src.pipeline import download_youtube_audio


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download audio from a YouTube video")
    parser.add_argument("--youtube-url", required=True, help="YouTube video URL")
    parser.add_argument(
        "--output-dir", default="data", help="Directory to store downloaded audio"
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    audio_path, file_id = download_youtube_audio(args.youtube_url, args.output_dir)
    print(f"{file_id}\t{audio_path}")


if __name__ == "__main__":
    main()
