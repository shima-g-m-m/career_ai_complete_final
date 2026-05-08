from typing import TypedDict, List, Optional, Any, Dict


class TurnRecord(TypedDict):
    turn: int
    question: str
    answer: str
    mode: str
    score: float
    feedback: str
    improvements: List[str]
    transcription: Optional[str]
    is_followup: bool
    soft_skill_notes: str
    # per-question breakdown for drill-down view
    correctness_note: str
    clarity_note: str
    depth_note: str
    confidence_note: str
    # audio
    audio_pace_score: Optional[float]
    audio_fluency_score: Optional[float]
    audio_confidence_score: Optional[float]
    audio_observations: List[str]
    audio_summary: str
    # video
    video_score: Optional[float]
    video_eye_contact: Optional[float]
    video_expression: Optional[float]
    video_posture: Optional[float]
    video_observations: List[str]
    video_method: str


class FinalEvaluation(TypedDict):
    technical_average: float
    soft_skills: Dict
    overall_score: float
    strengths: List[str]
    weak_areas: List[str]
    soft_skill_summary: str
    delivery_summary: str
    hiring_recommendation: str
    summary_paragraph: str
    improvement_plan: List[Dict]


class InterviewState(TypedDict):
    user_id: str
    level: str                    # auto-detected
    level_reasoning: str          # why this level was chosen
    level_signals: List[str]      # key CV signals
    job_description: str          # keywords the user provided
    cv_path: str
    interview_mode: str           # text | audio | video — fixed for session

    retriever: Optional[Any]

    current_question: str
    current_answer: str
    current_mode: str
    current_score: Optional[float]
    current_feedback: str
    current_improvements: List[str]
    current_transcription: Optional[str]
    current_soft_skill_notes: str
    current_audio_analysis: Optional[Dict]
    current_video_analysis: Optional[Dict]
    consecutive_followups: int

    # adaptive stopping
    consecutive_low_scores: int   # count of consecutive scores below EARLY_STOP_SCORE
    consecutive_high_scores: int  # count of consecutive high scores

    turns: List[TurnRecord]
    questions_asked: int
    topics_covered: List[str]
    weak_areas: List[str]
    soft_scores_history: List[Dict]
    audio_scores_history: List[Dict]
    video_scores_history: List[Dict]

    next_action: str
    end_reason: str
    interview_complete: bool
    error: Optional[str]

    final_evaluation: Optional[FinalEvaluation]
    media_file_path: Optional[str]
    original_video_path: Optional[str]
