import argparse

from src.pipeline import download_youtube_audio


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download audio from a YouTube video")
    parser.add_argument("--youtube-url", required=True, help="YouTube video URL")
    parser.add_argument(
        "--output-dir", default="data", help="Directory to store downloaded audio"
    )
    parser.add_argument(
        "--youtube-cookies-from-browser",
        help="Browser name for yt-dlp cookies (e.g. chrome, firefox)",
    )
    parser.add_argument(
        "--youtube-cookie-file",
        help="Path to cookies.txt exported from browser",
    )
    parser.add_argument(
        "--youtube-js-runtime",
        help="yt-dlp JS runtime, e.g. deno:/home/<user>/.deno/bin/deno",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    audio_path, file_id = download_youtube_audio(
        args.youtube_url,
        args.output_dir,
        cookies_from_browser=args.youtube_cookies_from_browser,
        cookie_file=args.youtube_cookie_file,
        js_runtime=args.youtube_js_runtime,
    )
    print(f"{file_id}\t{audio_path}")


if __name__ == "__main__":
    main()
