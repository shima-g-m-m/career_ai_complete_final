"""
video_processor.py
==================
Extracts the audio track from a video file using ffmpeg.

The output is always a normalised 16 kHz, mono, 16-bit WAV —
the ideal format for Whisper transcription (both local and cloud).

Supported video input formats: mp4, mov, avi, mkv, webm, flv, wmv, etc.
Any container that ffmpeg can read is accepted.
"""

import json
import os
import subprocess
import tempfile


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _probe_video(path: str) -> dict:
    """
    Runs ffprobe on the file and returns stream info.

    Returns:
        dict with keys 'has_video', 'has_audio', 'duration', 'video_codec',
        'audio_codec', 'width', 'height'

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If ffprobe cannot read the file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")

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
            f"ffprobe could not read '{path}': {result.stderr.strip()}"
        )

    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    fmt = info.get("format", {})

    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    v = video_streams[0] if video_streams else {}
    a = audio_streams[0] if audio_streams else {}

    return {
        "has_video": bool(video_streams),
        "has_audio": bool(audio_streams),
        "duration": float(fmt.get("duration", 0)),
        "video_codec": v.get("codec_name"),
        "audio_codec": a.get("codec_name"),
        "width": v.get("width"),
        "height": v.get("height"),
        "audio_sample_rate": int(a.get("sample_rate", 0)) if a else 0,
    }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_audio_from_video(
    video_path: str,
    audio_output_path: str,
    sample_rate: int = 16000,
    channels: int = 1,
) -> str:
    """
    Extracts and normalises the audio track from a video file.

    The output is saved as a WAV file (16 kHz, mono, 16-bit PCM) regardless
    of the output path extension. Whisper works best with this format.

    Args:
        video_path:        Path to the source video file.
        audio_output_path: Destination path for the extracted audio.
                           The extension is forced to .wav internally.
        sample_rate:       Target sample rate in Hz (default: 16000).
        channels:          Number of audio channels (default: 1 = mono).

    Returns:
        The actual path of the written audio file (always ends in .wav).

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError:      If the file has no audio track, or ffmpeg fails.
    """
    # ── Validate ───────────────────────────────────────────────────────────
    info = _probe_video(video_path)

    if not info["has_audio"]:
        raise RuntimeError(
            f"'{video_path}' contains no audio track. "
            "Cannot extract audio from a silent video."
        )

    # Force .wav extension — WAV is best for Whisper
    base, _ = os.path.splitext(audio_output_path)
    wav_output_path = base + ".wav"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(wav_output_path)), exist_ok=True)

    # ── Extract ────────────────────────────────────────────────────────────
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", video_path,
            "-vn",                        # no video
            "-ar", str(sample_rate),      # resample
            "-ac", str(channels),         # mono
            "-sample_fmt", "s16",         # 16-bit PCM
            wav_output_path,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to extract audio from '{video_path}':\n"
            f"{result.stderr.strip()}"
        )

    if not os.path.exists(wav_output_path):
        raise RuntimeError(
            f"ffmpeg finished without error but output file '{wav_output_path}' "
            "was not created."
        )

    return wav_output_path


# ---------------------------------------------------------------------------
# Convenience: extract to a temp file and return the path
# ---------------------------------------------------------------------------

def extract_audio_to_temp(video_path: str) -> str:
    """
    Extracts audio from a video into a temporary WAV file.

    The caller is responsible for deleting the file when done.

    Returns:
        Path to the temporary WAV file.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    return extract_audio_from_video(video_path, tmp.name)


# ---------------------------------------------------------------------------
# Probe / info (public helper)
# ---------------------------------------------------------------------------

def get_video_info(video_path: str) -> dict:
    """Returns metadata about a video file. Useful for debugging."""
    return _probe_video(video_path)


# ---------------------------------------------------------------------------
# Standalone test helper
# ---------------------------------------------------------------------------

def test_pipeline(video_path: str) -> None:
    """Quick smoke-test: probes and extracts audio from a given video."""
    print(f"\n[test] Probing: {video_path}")
    info = get_video_info(video_path)
    for k, v in info.items():
        print(f"  {k:<20}: {v}")

    print("[test] Extracting audio to temp file...")
    out = extract_audio_to_temp(video_path)
    size = os.path.getsize(out)
    print(f"  output path : {out}")
    print(f"  output size : {size:,} bytes")
    os.remove(out)
    print("[test] Temp file cleaned up. Pipeline OK.")
