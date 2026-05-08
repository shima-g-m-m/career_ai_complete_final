"""
video/video_analysis.py
=======================
Analyses video frames for non-verbal communication.
Uses OpenCV face detection + brightness/motion for basic scoring.
Falls back to LLaVA if available.
Runs fast — optimised for real-time interview use.
"""

import os
import cv2
import base64
import requests
import numpy as np
from typing import Dict, List, Optional


def extract_frames(video_path: str, num_frames: int = 4) -> List[np.ndarray]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        raise ValueError("Could not read video frames")
    start = max(int(total * 0.15), 0)
    end   = min(int(total * 0.85), total - 1)
    positions = np.linspace(start, end, num_frames, dtype=int)
    frames = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def _basic_cv_analysis(frames: List[np.ndarray]) -> Dict:
    """
    Improved basic CV analysis using face detection + multiple signals.
    Returns realistic scores rather than always 6.0.
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )

    face_count     = 0
    eye_count      = 0
    brightness_vals= []
    face_sizes     = []
    face_positions = []  # track if face is centred

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        brightness_vals.append(float(np.mean(gray)))

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        if len(faces) > 0:
            face_count += 1
            fx, fy, fw, fh = faces[0]
            face_sizes.append(fw * fh)
            # Check if face is roughly centred (within middle 60% of frame)
            cx = fx + fw // 2
            cy = fy + fh // 2
            centred = (w * 0.2 < cx < w * 0.8) and (h * 0.1 < cy < h * 0.85)
            face_positions.append(centred)

            # Check eyes within face region
            face_roi = gray[fy:fy+fh, fx:fx+fw]
            eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=3)
            if len(eyes) >= 1:
                eye_count += 1

    n = len(frames)
    face_ratio     = face_count / n if n else 0
    eye_ratio      = eye_count  / n if n else 0
    centred_ratio  = sum(face_positions) / len(face_positions) if face_positions else 0
    avg_brightness = np.mean(brightness_vals) if brightness_vals else 0
    avg_face_size  = np.mean(face_sizes) if face_sizes else 0

    observations = []

    # ── Eye contact score ────────────────────────────────────────────────
    # Based on: face visible + eyes detected + face centred
    if face_ratio < 0.3:
        eye_contact_score = 3.0
        observations.append("Face was rarely visible. Position your face clearly in front of the camera.")
    elif face_ratio < 0.6:
        eye_contact_score = 5.5
        observations.append("Face was partially visible. Keep your face centred in the camera frame.")
    else:
        if eye_ratio > 0.6 and centred_ratio > 0.6:
            eye_contact_score = 8.5
            observations.append("Good eye contact maintained — face centred and eyes clearly visible.")
        elif eye_ratio > 0.3:
            eye_contact_score = 7.0
            observations.append("Decent eye contact. Try to look directly at the camera more consistently.")
        else:
            eye_contact_score = 5.5
            observations.append("Face visible but eyes not always detected. Look directly at the camera lens.")

    # ── Expression score ─────────────────────────────────────────────────
    # Proxy: face size relative to frame (closer = more engaged)
    # + face visibility consistency
    frame_area = frames[0].shape[0] * frames[0].shape[1] if frames else 1
    if avg_face_size > 0:
        face_coverage = avg_face_size / frame_area
        if face_coverage > 0.08:
            expression_score = 8.0
            observations.append("Good presence — face filling the frame appropriately.")
        elif face_coverage > 0.03:
            expression_score = 6.5
            observations.append("Consider sitting a bit closer to the camera for better presence.")
        else:
            expression_score = 5.0
            observations.append("Sitting too far from camera. Move closer for better impact.")
    else:
        expression_score = 4.0
        observations.append("Could not assess expression — face not detected consistently.")

    # ── Posture score ────────────────────────────────────────────────────
    # Proxy: face vertical position (higher in frame = upright posture)
    if face_positions:
        if centred_ratio > 0.7:
            posture_score = 8.0
            observations.append("Good posture — head position is stable and well-framed.")
        elif centred_ratio > 0.4:
            posture_score = 6.5
            observations.append("Posture is acceptable but head position shifts. Sit upright and stable.")
        else:
            posture_score = 5.0
            observations.append("Head position varies significantly. Maintain a stable, upright posture.")
    else:
        posture_score = 4.5
        observations.append("Could not assess posture — face not consistently visible.")

    # ── Lighting adjustment ──────────────────────────────────────────────
    if avg_brightness < 50:
        for score in [eye_contact_score, expression_score, posture_score]:
            score = max(score - 1.5, 1.0)
        observations.append("Lighting is too dark. Move to a brighter, well-lit environment.")
    elif avg_brightness > 210:
        observations.append("Strong backlighting detected. Avoid sitting with a bright window behind you.")
    elif 80 < avg_brightness < 180:
        observations.append("Lighting conditions are good.")

    overall = round((eye_contact_score + expression_score + posture_score) / 3, 1)

    return {
        "score":              overall,
        "eye_contact_score":  round(eye_contact_score, 1),
        "expression_score":   round(expression_score, 1),
        "posture_score":      round(posture_score, 1),
        "observations":       observations,
        "method":             "basic_cv",
        "face_ratio":         round(face_ratio, 2),
        "avg_brightness":     round(avg_brightness, 1),
    }


def _llava_analyse(frame_b64: str, ollama_url: str) -> Optional[str]:
    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={"model": "llava",
                  "prompt": "Analyse this interview frame. In 2-3 sentences describe: 1) Eye contact with camera 2) Facial expression (confident/nervous/neutral) 3) Posture. Be specific and constructive.",
                  "images": [frame_b64], "stream": False},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        return None
    except Exception:
        return None


def _llava_score(observations: List[str]) -> Dict:
    combined = " ".join(observations).lower()
    eye  = 9.0 if any(w in combined for w in ["direct","camera","good eye","maintaining"]) else 4.0 if any(w in combined for w in ["avoiding","away","distracted","down"]) else 6.5
    expr = 9.0 if any(w in combined for w in ["confident","calm","professional","engaged","relaxed"]) else 4.0 if any(w in combined for w in ["nervous","anxious","uncomfortable","tense","stressed"]) else 6.5
    pos  = 9.0 if any(w in combined for w in ["upright","straight","professional posture","good posture","well-seated"]) else 4.0 if any(w in combined for w in ["slouch","leaning","hunched","poor posture"]) else 6.5
    return {"eye_contact_score": eye, "expression_score": expr, "posture_score": pos,
            "score": round((eye + expr + pos) / 3, 1)}


def analyse_video(video_path: str, ollama_url: str = "http://localhost:11434", num_frames: int = 4) -> Dict:
    base = {"score": 5.0, "eye_contact_score": 5.0, "expression_score": 5.0,
            "posture_score": 5.0, "observations": [], "method": "error", "error": None}
    try:
        frames = extract_frames(video_path, num_frames)
    except Exception as e:
        base["error"] = str(e)
        base["observations"] = [f"Could not read video: {e}"]
        return base

    if not frames:
        base["observations"] = ["No frames extracted from video."]
        return base

    # Try LLaVA first (fast timeout)
    llava_responses = []
    for frame in frames[:2]:  # only check 2 frames for speed
        b64 = base64.b64encode(cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])[1]).decode()
        resp = _llava_analyse(b64, ollama_url)
        if resp:
            llava_responses.append(resp)

    if len(llava_responses) >= 1:
        scores = _llava_score(llava_responses)
        base.update(scores)
        base["observations"] = llava_responses
        base["method"] = "llava"
        base["error"]  = None
        return base

    # Basic CV fallback
    result = _basic_cv_analysis(frames)
    base.update(result)
    base["error"] = None
    return base


def interpret_video_metrics(result: Dict) -> Dict:
    return {
        "overall_score":     result.get("score", 5.0),
        "eye_contact_score": result.get("eye_contact_score", 5.0),
        "expression_score":  result.get("expression_score", 5.0),
        "posture_score":     result.get("posture_score", 5.0),
        "feedback":          result.get("observations", []),
        "method":            result.get("method", "basic_cv"),
    }
