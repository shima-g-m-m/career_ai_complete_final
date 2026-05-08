"""
speech_to_text.py
=================
Transcribes audio files to text using one of two Whisper backends:

  1. LOCAL  — openai-whisper (runs on-device, no API key needed beyond setup)
  2. CLOUD  — OpenAI Whisper API (whisper-1, requires OPENAI_API_KEY)

The backend is selected via Config.WHISPER_BACKEND ("local" | "cloud" | "auto").
In "auto" mode (the default) the system tries local first, and if the model
weights are unavailable it silently falls back to the cloud API.

Supported input formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, ogg, flac
"""

import os
import json
import subprocess
import tempfile
from typing import Optional

from config import Config


# ---------------------------------------------------------------------------
# Validation helper (uses ffprobe)
# ---------------------------------------------------------------------------

def _validate_audio(path: str) -> dict:
    """
    Uses ffprobe to confirm the file contains a valid audio stream.

    Returns a dict with 'duration', 'codec', 'sample_rate' keys.
    Raises RuntimeError if the file has no audio stream or is unreadable.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe could not read file '{path}': {result.stderr.strip()}"
        )

    info = json.loads(result.stdout)
    audio_streams = [
        s for s in info.get("streams", [])
        if s.get("codec_type") == "audio"
    ]
    if not audio_streams:
        raise RuntimeError(
            f"File '{path}' contains no audio stream. "
            "Make sure it is a valid audio/video file."
        )

    fmt = info.get("format", {})
    a = audio_streams[0]
    return {
        "duration": float(fmt.get("duration", 0)),
        "codec": a.get("codec_name"),
        "sample_rate": int(a.get("sample_rate", 0)),
        "channels": int(a.get("channels", 1)),
    }


# ---------------------------------------------------------------------------
# Pre-processing: normalise audio to 16 kHz mono WAV for Whisper
# ---------------------------------------------------------------------------

def _normalise_to_wav(src: str, dst: str) -> None:
    """
    Converts any audio/video file to a 16 kHz, mono, 16-bit WAV.
    This is the format Whisper handles most reliably regardless of backend.
    """
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", src,
            "-ar", "16000",   # 16 kHz — Whisper's native rate
            "-ac", "1",       # mono
            "-sample_fmt", "s16",
            dst,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg normalisation failed for '{src}': {result.stderr.strip()}"
        )


# ---------------------------------------------------------------------------
# Backend 1 — Local Whisper
# ---------------------------------------------------------------------------

_local_model_cache: dict = {}


def _transcribe_local(wav_path: str, model_name: str = "base") -> str:
    """
    Transcribes using the locally installed openai-whisper package.
    The model weights are cached after the first load.

    Raises ImportError if openai-whisper is not installed.
    Raises RuntimeError if the model weights cannot be downloaded (offline env).
    """
    try:
        import whisper  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "openai-whisper is not installed. "
            "Run: pip install openai-whisper"
        ) from exc

    if model_name not in _local_model_cache:
        _local_model_cache[model_name] = whisper.load_model(model_name)

    model = _local_model_cache[model_name]
    result = model.transcribe(wav_path, fp16=False)
    # Empty string is valid — silence or non-speech audio produces no output.
    # We return it as-is; callers decide whether an empty answer is acceptable.
    return result.get("text", "").strip()


# ---------------------------------------------------------------------------
# Backend 2 — OpenAI Cloud Whisper API
# ---------------------------------------------------------------------------

def _transcribe_cloud(wav_path: str) -> str:
    """
    Transcribes using the OpenAI Whisper API (whisper-1).
    Requires a valid OPENAI_API_KEY in the environment / Config.
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "openai package is not installed. "
            "Run: pip install openai"
        ) from exc

    api_key = Config.OPENAI_API_KEY
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Cannot use cloud Whisper without an API key."
        )

    client = OpenAI(api_key=api_key)

    with open(wav_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
        )

    # response is a plain string when response_format="text"
    # Empty string is valid for silence/non-speech — callers handle it.
    return str(response).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe_audio(
    file_path: str,
    backend: Optional[str] = None,
) -> str:
    """
    Transcribes an audio file to text.

    Args:
        file_path: Path to the audio file (mp3, wav, mp4, m4a, webm, etc.)
        backend:   Override Config.WHISPER_BACKEND.
                   "local"  — use on-device Whisper model
                   "cloud"  — use OpenAI Whisper API
                   "auto"   — try local first; only fall back to cloud on a
                              hard failure (import error, model load error),
                              NOT on an empty result (silence is valid).

    Returns:
        Transcribed text as a string (may be empty for silent audio).

    Raises:
        FileNotFoundError: File does not exist.
        RuntimeError:      File has no audio stream, or all backends fail.
    """
    selected_backend = (backend or getattr(Config, "WHISPER_BACKEND", "auto")).lower()

    # ── 1. Validate the input file ─────────────────────────────────────────
    _validate_audio(file_path)  # raises on invalid file

    # ── 2. Normalise to 16 kHz mono WAV ────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        _normalise_to_wav(file_path, wav_path)

        # ── 3. Transcribe ──────────────────────────────────────────────────
        if selected_backend == "local":
            return _transcribe_local(wav_path, getattr(Config, "WHISPER_MODEL", "base"))

        if selected_backend == "cloud":
            return _transcribe_cloud(wav_path)

        # "auto" — try local; fall back to cloud ONLY on import/load errors,
        # not on empty transcriptions (silence is a valid result, not a failure).
        try:
            return _transcribe_local(wav_path, getattr(Config, "WHISPER_MODEL", "base"))
        except ImportError as local_err:
            # openai-whisper not installed — try cloud
            print(
                f"[speech_to_text] Local Whisper unavailable ({local_err}). "
                "Falling back to OpenAI cloud API..."
            )
        except Exception as local_err:
            # Model download failed, GPU error, etc. — try cloud
            print(
                f"[speech_to_text] Local Whisper failed ({local_err}). "
                "Falling back to OpenAI cloud API..."
            )

        # Cloud fallback
        try:
            return _transcribe_cloud(wav_path)
        except ImportError:
            raise RuntimeError(
                "Neither openai-whisper nor the openai package is installed.\n"
                "Fix with:  pip install openai-whisper openai"
            )

    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


# ---------------------------------------------------------------------------
# Standalone test helper
# ---------------------------------------------------------------------------

def test_pipeline(audio_path: str) -> None:
    """Quick smoke-test: validates and transcribes a given audio file."""
    print(f"\n[test] Validating: {audio_path}")
    info = _validate_audio(audio_path)
    print(f"  duration   : {info['duration']:.1f}s")
    print(f"  codec      : {info['codec']}")
    print(f"  sample_rate: {info['sample_rate']} Hz")
    print(f"  channels   : {info['channels']}")

    print("[test] Transcribing...")
    text = transcribe_audio(audio_path)
    print(f"  result: {text!r}")
