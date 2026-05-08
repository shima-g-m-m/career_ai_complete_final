import os, uuid, shutil
from typing import Dict, Any, Optional
from config import Config
from graph.state import InterviewState

_sessions: Dict[str, InterviewState] = {}


def _blank_state(user_id, cv_path, interview_mode, job_description) -> dict:
    return dict(
        user_id=user_id, level="mid", level_reasoning="", level_signals=[],
        job_description=job_description, cv_path=cv_path,
        interview_mode=interview_mode, retriever=None,
        current_question="", current_answer="", current_mode=interview_mode,
        current_score=None, current_feedback="", current_improvements=[],
        current_transcription=None, current_soft_skill_notes="",
        current_audio_analysis=None, current_video_analysis=None,
        consecutive_followups=0, consecutive_low_scores=0, consecutive_high_scores=0,
        turns=[], questions_asked=0, topics_covered=[], weak_areas=[],
        soft_scores_history=[], audio_scores_history=[], video_scores_history=[],
        next_action="new_question", end_reason="",
        interview_complete=False, error=None, final_evaluation=None,
        media_file_path=None, original_video_path=None,
        current_correctness_note="", current_clarity_note="",
        current_depth_note="", current_confidence_note="",
    )


def start_session(cv_path: str, interview_mode: str, job_description: str = "") -> Dict[str, Any]:
    user_id = str(uuid.uuid4())
    saved   = _save_file(cv_path)
    state   = _blank_state(user_id, saved, interview_mode, job_description)

    from graph.nodes import load_cv, generate_question
    s = {**state, **load_cv(state)}
    if s.get("error"):
        return {"status": "error", "message": s["error"]}
    s = {**s, **generate_question(s)}
    if s.get("error"):
        return {"status": "error", "message": s["error"]}

    _sessions[user_id] = s
    return {
        "status":          "success",
        "user_id":         user_id,
        "question":        s.get("current_question", ""),
        "question_number": 1,
        "level":           s.get("level", "mid"),
        "level_reasoning": s.get("level_reasoning", ""),
        "level_signals":   s.get("level_signals", []),
    }


def submit_answer(user_id: str, answer: str = "", media_file_path: Optional[str] = None) -> Dict[str, Any]:
    if user_id not in _sessions:
        return {"status": "error", "message": "Session not found."}
    state = dict(_sessions[user_id])
    if state.get("interview_complete"):
        return {"status": "error", "message": "Interview already complete."}

    mode = state.get("interview_mode", "text")
    state["current_answer"]  = answer
    state["current_mode"]    = mode
    state["media_file_path"] = media_file_path

    updated = _run_turn(state)
    _sessions[user_id] = updated

    turns = updated.get("turns", [])
    last  = turns[-1] if turns else {}

    # ── Response: NO score/feedback/improvements per turn ──────────────────
    # Only return what's needed to show "answer received" + next question
    response = {
        "status":             "success",
        "turn":               last.get("turn", 0),
        "transcription":      last.get("transcription"),   # show what was heard
        "mode":               mode,
        "is_followup":        last.get("is_followup", False),
        "interview_complete": updated.get("interview_complete", False),
        "questions_asked":    updated.get("questions_asked", 0),
    }

    if updated.get("interview_complete"):
        response["final_evaluation"] = updated.get("final_evaluation")
        response["end_reason"]       = updated.get("end_reason", "")
        response["all_turns"]        = updated.get("turns", [])
        response["topics_covered"]   = updated.get("topics_covered", [])
    else:
        response["next_question"]   = updated.get("current_question", "")
        response["next_action"]     = updated.get("next_action", "new_question")
        response["question_number"] = updated.get("questions_asked", 0) + 1

    return response


def _run_turn(state: dict) -> dict:
    from graph.nodes import (
        process_answer, evaluate, record_turn, decide,
        generate_question, generate_followup,
        generate_final_eval, end_interview,
    )
    s = dict(state)
    s = {**s, **process_answer(s)}
    if s.get("error") and not s.get("current_answer","").strip():
        s = {**s, **end_interview(s)}
        return s
    s = {**s, **evaluate(s)}
    s = {**s, **record_turn(s)}
    s = {**s, **decide(s)}
    action = s.get("next_action","new_question")
    if action == "end":
        s = {**s, **generate_final_eval(s)}
        s = {**s, **end_interview(s)}
    elif action == "followup":
        s = {**s, **generate_followup(s)}
    else:
        s = {**s, **generate_question(s)}
    return s


def _save_file(src: str) -> str:
    os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(Config.UPLOAD_DIR, os.path.basename(src))
    if os.path.abspath(src) != os.path.abspath(dest):
        shutil.copy2(src, dest)
    return dest
