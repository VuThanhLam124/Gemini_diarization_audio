from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "pyannote/speaker-diarization-3.1"


@dataclass(frozen=True)
class DiarizationSegment:
    start_sec: float
    end_sec: float
    diarization_speaker: str


def resolve_hf_token(cli_token: str | None, key_path: str | Path = "hugging_face_key.txt") -> str:
    """Resolve HF token from CLI, env vars, or key file."""
    candidates = [
        cli_token,
        os.getenv("HF_TOKEN"),
        os.getenv("HUGGINGFACE_TOKEN"),
        os.getenv("HUGGINGFACE_ACCESS_TOKEN"),
    ]
    for token in candidates:
        if token and token.strip():
            return token.strip()

    key_file = Path(key_path)
    if key_file.exists():
        content = key_file.read_text(encoding="utf-8").strip()
        if content:
            return content

    raise RuntimeError(
        "Missing Hugging Face token. Pass --hf-token, set HF_TOKEN/HUGGINGFACE_TOKEN/"
        "HUGGINGFACE_ACCESS_TOKEN, or create hugging_face_key.txt"
    )


def ensure_ffmpeg_exists() -> None:
    if shutil.which("ffmpeg"):
        return
    raise RuntimeError(
        "ffmpeg is required but not found in PATH. Install ffmpeg before running audio-only mode."
    )


def _run_ffmpeg(cmd: list[str], error_prefix: str) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return
    stderr = result.stderr.strip() or "unknown ffmpeg error"
    raise RuntimeError(f"{error_prefix}: {stderr}")


def convert_to_wav_16k(audio_path: Path) -> tuple[Path, Path]:
    """Convert input audio to 16k mono WAV and return (converted_path, temp_dir)."""
    ensure_ffmpeg_exists()

    tmpdir = Path(tempfile.mkdtemp(prefix="pyannote_audio_"))
    safe_stem = re.sub(r"[^A-Za-z0-9_-]", "_", audio_path.stem)
    output_path = tmpdir / f"{safe_stem}_16k.wav"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        "-f",
        "wav",
        str(output_path),
    ]
    _run_ffmpeg(cmd, f"Failed to convert audio {audio_path}")
    if not output_path.exists():
        raise RuntimeError(f"ffmpeg did not produce converted WAV: {output_path}")

    return output_path, tmpdir


def merge_adjacent_segments(
    segments: list[DiarizationSegment],
    max_gap: float = 2.0,
    min_duration: float = 0.5,
) -> list[DiarizationSegment]:
    """Merge consecutive segments from the same speaker with small gap."""
    if not segments:
        return []

    sorted_segments = sorted(segments, key=lambda x: x.start_sec)
    merged: list[DiarizationSegment] = []
    current = sorted_segments[0]

    for segment in sorted_segments[1:]:
        gap = segment.start_sec - current.end_sec
        if segment.diarization_speaker == current.diarization_speaker and gap <= max_gap:
            current = DiarizationSegment(
                start_sec=current.start_sec,
                end_sec=max(current.end_sec, segment.end_sec),
                diarization_speaker=current.diarization_speaker,
            )
        else:
            if current.end_sec - current.start_sec >= min_duration:
                merged.append(current)
            current = segment

    if current.end_sec - current.start_sec >= min_duration:
        merged.append(current)

    return merged


class PyannoteDiarizer:
    """Thin wrapper around pyannote speaker diarization pipeline."""

    def __init__(
        self,
        *,
        hf_token: str | None,
        device: str = "auto",
        seg_min_duration_off: float | None = None,
        clustering_threshold: float | None = None,
        clustering_method: str | None = None,
        min_cluster_size: int | None = None,
    ) -> None:
        self._torch = None
        self._torchaudio = None

        try:
            import torch
            import torchaudio
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Missing diarization dependencies. Install them with: "
                "pip install -r requirements_diarization.txt"
            ) from exc

        self._torch = torch
        self._torchaudio = torchaudio
        resolved_device = self._resolve_device(device)
        token = resolve_hf_token(hf_token)

        pipeline = self._load_pipeline_compat(Pipeline, MODEL_ID, token)
        if pipeline is None:
            raise RuntimeError(
                f"Failed to load {MODEL_ID}. Ensure HF token is valid and model terms are accepted."
            )

        hyperparams: dict[str, dict[str, float | int | str]] = {}
        if seg_min_duration_off is not None:
            hyperparams.setdefault("segmentation", {})["min_duration_off"] = seg_min_duration_off

        if clustering_threshold is not None:
            hyperparams.setdefault("clustering", {})["threshold"] = clustering_threshold
        if clustering_method is not None:
            hyperparams.setdefault("clustering", {})["method"] = clustering_method
        if min_cluster_size is not None:
            hyperparams.setdefault("clustering", {})["min_cluster_size"] = min_cluster_size

        if hyperparams:
            pipeline.instantiate(hyperparams)

        pipeline.to(resolved_device)
        self.pipeline = pipeline

    @staticmethod
    def _load_pipeline_compat(Pipeline, model_id: str, token: str):
        """
        Load pyannote pipeline with backward-compatible auth kwargs.
        Different pyannote versions accept either `token` or `use_auth_token`.
        """
        params = inspect.signature(Pipeline.from_pretrained).parameters

        if "token" in params:
            return Pipeline.from_pretrained(model_id, token=token)
        if "use_auth_token" in params:
            return Pipeline.from_pretrained(model_id, use_auth_token=token)

        # Fallback for very old/new variants: rely on env vars and no explicit auth kwarg.
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_ACCESS_TOKEN", token)
        return Pipeline.from_pretrained(model_id)

    def _resolve_device(self, device: str):
        if device == "cpu":
            return self._torch.device("cpu")
        if device == "cuda":
            if not self._torch.cuda.is_available():
                raise RuntimeError("--device cuda was requested but no CUDA device is available.")
            return self._torch.device("cuda")
        if device == "auto":
            return self._torch.device("cuda") if self._torch.cuda.is_available() else self._torch.device("cpu")
        raise ValueError("Invalid --device value. Use: auto, cpu, cuda")

    @staticmethod
    def _get_annotation(diarization: Any):
        if hasattr(diarization, "itertracks"):
            return diarization
        if hasattr(diarization, "speaker_diarization"):
            return diarization.speaker_diarization
        raise TypeError("Unsupported pyannote output: no annotation found.")

    def diarize_file(
        self,
        audio_path: Path,
        *,
        merge_gap: float = 2.0,
        min_segment_duration: float = 0.5,
    ) -> list[DiarizationSegment]:
        if not audio_path.exists() or not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        prepared_path, tmpdir = convert_to_wav_16k(audio_path)
        try:
            waveform, sample_rate = self._torchaudio.load(str(prepared_path))
            audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}
            diarization = self.pipeline(audio_in_memory)
            annotation = self._get_annotation(diarization)

            raw_segments: list[DiarizationSegment] = []
            for segment, _, speaker in annotation.itertracks(yield_label=True):
                start_sec = float(segment.start)
                end_sec = float(segment.end)
                if end_sec <= start_sec:
                    continue
                raw_segments.append(
                    DiarizationSegment(
                        start_sec=start_sec,
                        end_sec=end_sec,
                        diarization_speaker=str(speaker),
                    )
                )

            return merge_adjacent_segments(
                raw_segments,
                max_gap=merge_gap,
                min_duration=min_segment_duration,
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
