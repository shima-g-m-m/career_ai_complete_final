"""
audio/audio_analysis.py
=======================
Analyses audio recordings for communication quality signals.
Uses librosa for signal processing — no external API needed.

Metrics extracted:
  - speaking_pace      : words per minute (from transcript + duration)
  - filler_word_count  : count of um/uh/like/you know/basically/literally
  - filler_rate        : fillers per minute
  - pause_count        : number of pauses > 0.5s
  - pause_rate         : pauses per minute
  - avg_pause_duration : mean pause length in seconds
  - energy_variance    : voice energy variance (low = monotone, high = expressive)
  - pitch_mean         : mean fundamental frequency in Hz (0 if unvoiced)
  - pitch_variance     : pitch variance (expressiveness)
  - speech_ratio       : fraction of recording that is active speech
  - duration           : total recording duration in seconds
"""

import os
import re
import numpy as np
from typing import Dict, Optional


def analyse_audio(audio_path: str, transcript: str = "") -> Dict:
    """
    Analyses an audio file and returns communication quality metrics.

    Args:
        audio_path : path to a WAV file (16kHz mono preferred)
        transcript : whisper transcript for pace and filler analysis

    Returns:
        dict of metrics — always returns something even on error
    """
    base = {
        "speaking_pace":       0.0,
        "filler_word_count":   0,
        "filler_rate":         0.0,
        "pause_count":         0,
        "pause_rate":          0.0,
        "avg_pause_duration":  0.0,
        "energy_variance":     0.0,
        "pitch_mean":          0.0,
        "pitch_variance":      0.0,
        "speech_ratio":        0.0,
        "duration":            0.0,
        "error":               None,
    }

    if not audio_path or not os.path.exists(audio_path):
        base["error"] = "Audio file not found"
        return base

    try:
        import librosa

        # ── Load audio ────────────────────────────────────────────────────
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(y) / sr
        base["duration"] = round(duration, 2)

        if duration < 0.5:
            base["error"] = "Audio too short to analyse"
            return base

        # ── Filler words (from transcript) ────────────────────────────────
        fillers = ["um", "uh", "like", "you know", "basically",
                   "literally", "actually", "so", "right", "okay so"]
        filler_count = 0
        if transcript:
            t_lower = transcript.lower()
            for f in fillers:
                # whole word match
                filler_count += len(re.findall(r'\b' + re.escape(f) + r'\b', t_lower))
        base["filler_word_count"] = filler_count
        base["filler_rate"] = round(filler_count / (duration / 60), 2)

        # ── Speaking pace (WPM) ───────────────────────────────────────────
        if transcript:
            word_count = len(transcript.split())
            base["speaking_pace"] = round(word_count / (duration / 60), 1)

        # ── Pause detection using RMS energy ─────────────────────────────
        frame_length = int(0.025 * sr)   # 25ms frames
        hop_length   = int(0.010 * sr)   # 10ms hop
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        # Threshold: below 5% of max RMS = silence
        threshold = float(np.max(rms)) * 0.05
        is_silent = rms < threshold

        # Convert frames to time segments
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        min_pause_duration = 0.5  # seconds

        pauses = []
        in_pause = False
        pause_start = 0.0

        for i, silent in enumerate(is_silent):
            if silent and not in_pause:
                in_pause = True
                pause_start = times[i]
            elif not silent and in_pause:
                in_pause = False
                pause_dur = times[i] - pause_start
                if pause_dur >= min_pause_duration:
                    pauses.append(pause_dur)

        base["pause_count"]        = len(pauses)
        base["pause_rate"]         = round(len(pauses) / (duration / 60), 2)
        base["avg_pause_duration"] = round(float(np.mean(pauses)), 2) if pauses else 0.0

        # ── Speech ratio ──────────────────────────────────────────────────
        speech_frames = int(np.sum(~is_silent))
        base["speech_ratio"] = round(speech_frames / len(is_silent), 3)

        # ── Energy variance (expressiveness) ─────────────────────────────
        base["energy_variance"] = round(float(np.var(rms)), 6)

        # ── Pitch analysis ────────────────────────────────────────────────
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            frame_length=frame_length * 2,
        )
        voiced_f0 = f0[voiced_flag] if f0 is not None else np.array([])
        if len(voiced_f0) > 0:
            base["pitch_mean"]     = round(float(np.mean(voiced_f0)), 2)
            base["pitch_variance"] = round(float(np.var(voiced_f0)), 2)

        return base

    except Exception as e:
        base["error"] = str(e)
        return base


def interpret_audio_metrics(metrics: Dict) -> Dict:
    """
    Converts raw audio metrics into human-readable signals and
    0-10 scores for communication dimensions.

    Returns:
        {
          "pace_score"       : 0-10,
          "fluency_score"    : 0-10,  (fewer fillers + fewer pauses = better)
          "confidence_score" : 0-10,  (energy + pitch variance)
          "observations"     : [str, ...]  human-readable observations
          "summary"          : str
        }
    """
    obs = []
    pace_score       = 7.0
    fluency_score    = 7.0
    confidence_score = 7.0

    if metrics.get("error"):
        return {
            "pace_score": 5.0, "fluency_score": 5.0, "confidence_score": 5.0,
            "observations": ["Audio analysis unavailable."], "summary": "",
        }

    duration = metrics["duration"]
    if duration < 1:
        return {
            "pace_score": 5.0, "fluency_score": 5.0, "confidence_score": 5.0,
            "observations": ["Answer too short to analyse."], "summary": "",
        }

    # ── Speaking pace ─────────────────────────────────────────────────────
    pace = metrics["speaking_pace"]
    if pace == 0:
        obs.append("No speech detected for pace analysis.")
    elif pace < 90:
        pace_score = 5.0
        obs.append(f"Speaking pace is slow ({pace:.0f} WPM). Aim for 120-160 WPM.")
    elif pace < 110:
        pace_score = 7.0
        obs.append(f"Speaking pace is slightly slow ({pace:.0f} WPM).")
    elif pace <= 160:
        pace_score = 10.0
        obs.append(f"Speaking pace is ideal ({pace:.0f} WPM).")
    elif pace <= 190:
        pace_score = 7.0
        obs.append(f"Speaking pace is slightly fast ({pace:.0f} WPM). Slow down slightly.")
    else:
        pace_score = 5.0
        obs.append(f"Speaking pace is too fast ({pace:.0f} WPM). Slow down for clarity.")

    # ── Filler words ──────────────────────────────────────────────────────
    filler_rate = metrics["filler_rate"]
    if filler_rate < 3:
        obs.append(f"Very few filler words ({metrics['filler_word_count']} total). Excellent.")
        fluency_score = min(fluency_score + 1, 10)
    elif filler_rate < 8:
        obs.append(f"Moderate filler word usage ({metrics['filler_word_count']} total, {filler_rate:.1f}/min).")
    else:
        fluency_score = max(fluency_score - 2, 2)
        obs.append(f"High filler word usage ({metrics['filler_word_count']} total, {filler_rate:.1f}/min). Work on reducing um/uh/like.")

    # ── Pauses ────────────────────────────────────────────────────────────
    pause_rate = metrics["pause_rate"]
    avg_pause  = metrics["avg_pause_duration"]
    if pause_rate < 3:
        obs.append("Good flow with minimal hesitation pauses.")
    elif pause_rate < 7:
        obs.append(f"Some pauses detected ({metrics['pause_count']} pauses, avg {avg_pause:.1f}s). Acceptable.")
        fluency_score = max(fluency_score - 0.5, 2)
    else:
        fluency_score = max(fluency_score - 2, 2)
        obs.append(f"Frequent hesitation pauses ({metrics['pause_count']} pauses, avg {avg_pause:.1f}s). Practice smoother delivery.")

    # ── Speech ratio ──────────────────────────────────────────────────────
    ratio = metrics["speech_ratio"]
    if ratio < 0.3:
        obs.append("Very little active speech detected — answer may be too brief.")
        fluency_score = max(fluency_score - 1, 2)
    elif ratio > 0.85:
        obs.append("Continuous speech throughout — good energy.")

    # ── Energy variance (expressiveness) ─────────────────────────────────
    ev = metrics["energy_variance"]
    if ev < 0.0001:
        confidence_score = max(confidence_score - 1.5, 2)
        obs.append("Voice energy is very flat — try to vary your tone for engagement.")
    elif ev > 0.001:
        obs.append("Good voice expressiveness and variation in energy.")
        confidence_score = min(confidence_score + 0.5, 10)

    # ── Pitch variance ────────────────────────────────────────────────────
    pv = metrics["pitch_variance"]
    if pv > 0 and pv < 100:
        confidence_score = max(confidence_score - 1, 2)
        obs.append("Monotone delivery detected. Vary your pitch to sound more engaging.")
    elif pv > 500:
        obs.append("Good pitch variation — voice sounds natural and engaging.")

    summary = (
        f"Pace: {pace:.0f} WPM | Fillers: {metrics['filler_word_count']} | "
        f"Pauses: {metrics['pause_count']} | Speech ratio: {metrics['speech_ratio']:.0%}"
    ) if pace > 0 else "Audio communication analysis complete."

    return {
        "pace_score":       round(min(max(pace_score, 0), 10), 1),
        "fluency_score":    round(min(max(fluency_score, 0), 10), 1),
        "confidence_score": round(min(max(confidence_score, 0), 10), 1),
        "observations":     obs,
        "summary":          summary,
        "raw_metrics":      metrics,
    }
