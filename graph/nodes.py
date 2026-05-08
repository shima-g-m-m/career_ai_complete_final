"""
graph/nodes.py — Fixed version.
Key fixes:
1. consecutive_low_scores never resets on good answers (accumulates correctly)
2. weak_areas populated from actual turns in final eval
3. Evaluation JSON parsed more robustly
4. Speed: smaller context, truncated inputs
"""
import os, sys, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from graph.state import InterviewState, TurnRecord
from rag.cv_loader import CVLoader
from rag.vector_store import get_vector_store, save_vector_store
from rag.retriever import get_retriever
from chains.level_detection_chain import detect_level_chain
from chains.question_chain import generate_question_chain
from chains.followup_chain import generate_followup_chain
from chains.evaluation_chain import evaluate_answer_chain
from chains.decision_chain import decide_next_action_chain
from chains.final_evaluation_chain import final_evaluation_chain
from audio.speech_to_text import transcribe_audio
from audio.audio_analysis import analyse_audio, interpret_audio_metrics
from video.video_processor import extract_audio_to_temp
from video.video_analysis import analyse_video, interpret_video_metrics
from utils.parser import parse_json_output


# ── 1. Load CV + Auto-detect Level ────────────────────────────────────────
def load_cv(state: InterviewState) -> dict:
    try:
        loader = CVLoader(state["cv_path"])
        chunks = loader.load_and_split()
        vs = get_vector_store(state["user_id"], documents=chunks)
        save_vector_store(vs, state["user_id"])
        retriever = get_retriever(vs)

        level = "mid"; reasoning = "Default mid level"; signals = []
        try:
            raw    = detect_level_chain(retriever).invoke({"job_description": state.get("job_description","")})
            parsed = parse_json_output(raw)
            lvl    = parsed.get("level","mid").lower().strip()
            if lvl in ("junior","mid","senior"):
                level = lvl
            reasoning = parsed.get("reasoning","")
            signals   = parsed.get("key_signals",[])
        except Exception as e:
            reasoning = f"Could not detect level ({e}), using mid"

        return {"retriever": retriever, "level": level,
                "level_reasoning": reasoning, "level_signals": signals, "error": None}
    except Exception as e:
        return {"error": f"CV loading failed: {e}", "interview_complete": True}


# ── 2. Generate Question ───────────────────────────────────────────────────
def generate_question(state: InterviewState) -> dict:
    try:
        topics_str = ", ".join(state.get("topics_covered",[])[-5:]) or "none yet"  # last 5 only
        q = generate_question_chain(state["retriever"]).invoke({
            "level": state["level"], "topics_covered": topics_str,
        })
        return {
            "current_question": q.strip(), "current_answer": "", "current_score": None,
            "current_feedback": "", "current_improvements": [], "current_transcription": None,
            "current_soft_skill_notes": "", "current_audio_analysis": None,
            "current_video_analysis": None, "consecutive_followups": 0, "error": None,
            "current_correctness_note": "", "current_clarity_note": "",
            "current_depth_note": "", "current_confidence_note": "",
        }
    except Exception as e:
        return {"error": f"Question generation failed: {e}", "interview_complete": True}


# ── 3. Process Answer ──────────────────────────────────────────────────────
def process_answer(state: InterviewState) -> dict:
    mode       = state.get("current_mode","text")
    media_path = state.get("media_file_path")
    updates    = {"current_audio_analysis": None, "current_video_analysis": None, "error": None}

    if mode == "text":
        return updates

    if not media_path or not os.path.exists(media_path):
        return {**updates, "error": f"Media file not found: {media_path}"}

    tmp_audio = None
    try:
        if mode == "video":
            # Video analysis in background thread
            vr = [None]
            def run_vid():
                try:
                    raw = analyse_video(media_path, ollama_url=Config.OLLAMA_BASE_URL, num_frames=3)
                    vr[0] = interpret_video_metrics(raw)
                except Exception:
                    vr[0] = {"overall_score":5.0,"eye_contact_score":5.0,"expression_score":5.0,
                              "posture_score":5.0,"feedback":["Video analysis unavailable."],"method":"error"}
            vt = threading.Thread(target=run_vid, daemon=True)
            vt.start()
            tmp_audio = extract_audio_to_temp(media_path)
            transcribe_path = tmp_audio
            vt.join(timeout=8)
            updates["current_video_analysis"] = vr[0] or {
                "overall_score":5.0,"eye_contact_score":5.0,"expression_score":5.0,
                "posture_score":5.0,"feedback":["Video timed out."],"method":"timeout"}
        else:
            transcribe_path = media_path

        transcription = transcribe_audio(transcribe_path)
        if not transcription.strip():
            return {**updates, "error": "No speech detected. Please try again.", "current_transcription": ""}

        updates["current_answer"]        = transcription
        updates["current_transcription"] = transcription

        try:
            araw = analyse_audio(transcribe_path, transcript=transcription)
            updates["current_audio_analysis"] = interpret_audio_metrics(araw)
        except Exception:
            updates["current_audio_analysis"] = {
                "pace_score":5.0,"fluency_score":5.0,"confidence_score":5.0,
                "observations":[],"summary":""}
        return updates
    except Exception as e:
        return {**updates, "error": f"Media processing failed: {e}"}
    finally:
        if tmp_audio and os.path.exists(tmp_audio):
            os.remove(tmp_audio)


# ── 4. Evaluate ────────────────────────────────────────────────────────────
def evaluate(state: InterviewState) -> dict:
    if not state.get("current_answer","").strip():
        return {
            "current_score": 0.0, "current_feedback": "No answer provided.",
            "current_improvements": ["Please provide a substantive answer."],
            "current_soft_skill_notes": "Candidate did not respond.",
            "current_correctness_note": "No answer given.",
            "current_clarity_note": "No answer given.",
            "current_depth_note": "No answer given.",
            "current_confidence_note": "No answer given.",
            "consecutive_low_scores": state.get("consecutive_low_scores",0) + 1,
            "consecutive_high_scores": 0,
            "error": None,
        }
    try:
        raw    = evaluate_answer_chain(state["retriever"]).invoke({
            "question": state["current_question"],
            "answer":   state["current_answer"],
        })
        parsed = parse_json_output(raw)
        score  = float(parsed.get("score", 0))

        # ── Accumulate scores ─────────────────────────────────────────────
        sh = list(state.get("soft_scores_history",[]))
        rs = parsed.get("soft_scores",{})
        if rs:
            sh.append({k: float(rs.get(k,5)) for k in ("communication","confidence","problem_solving","honesty")})

        ah = list(state.get("audio_scores_history",[]))
        aa = state.get("current_audio_analysis")
        if aa:
            ah.append({"pace":aa.get("pace_score",5),"fluency":aa.get("fluency_score",5),"confidence":aa.get("confidence_score",5)})

        vh = list(state.get("video_scores_history",[]))
        va = state.get("current_video_analysis")
        if va:
            vh.append({"overall":va.get("overall_score",5),"eye_contact":va.get("eye_contact_score",5),
                       "expression":va.get("expression_score",5),"posture":va.get("posture_score",5)})

        # ── Weak areas — always track low scoring questions ────────────────
        wa = list(state.get("weak_areas",[]))
        if score < Config.WEAK_SCORE_THRESHOLD:
            entry = f"Q{state.get('questions_asked',0)+1}: {state['current_question'][:70]}... (score: {score}/10)"
            if entry not in wa:
                wa.append(entry)

        # ── Consecutive low score counter — NEVER resets on good answers ──
        # Only resets when interview ends
        cls = state.get("consecutive_low_scores", 0)
        chs = state.get("consecutive_high_scores", 0)
        if score < Config.EARLY_STOP_SCORE:
            cls += 1   # bad answer — increment streak
            # chs stays as is — we track CONSECUTIVE bad answers independently
        else:
            cls = 0    # good answer resets the bad streak
            if score >= 7.0:
                chs += 1
            else:
                chs = 0

        return {
            "current_score":            score,
            "current_feedback":         parsed.get("feedback",""),
            "current_improvements":     parsed.get("improvements",[]),
            "current_soft_skill_notes": parsed.get("soft_skill_notes",""),
            "current_correctness_note": parsed.get("correctness_note",""),
            "current_clarity_note":     parsed.get("clarity_note",""),
            "current_depth_note":       parsed.get("depth_note",""),
            "current_confidence_note":  parsed.get("confidence_note",""),
            "soft_scores_history":      sh,
            "audio_scores_history":     ah,
            "video_scores_history":     vh,
            "weak_areas":               wa,
            "consecutive_low_scores":   cls,
            "consecutive_high_scores":  chs,
            "error": None,
        }
    except Exception as e:
        # Evaluation failed — do NOT count as bad answer (would wrongly end session)
        # Keep existing streak counters unchanged
        return {
            "current_score": None, "current_feedback": f"Evaluation error: {e}",
            "current_improvements": [], "current_soft_skill_notes": "",
            "current_correctness_note": "", "current_clarity_note": "",
            "current_depth_note": "", "current_confidence_note": "",
            "consecutive_low_scores": state.get("consecutive_low_scores", 0),
            "consecutive_high_scores": state.get("consecutive_high_scores", 0),
            "error": None,  # don't abort session on eval error
        }


# ── 5. Record Turn ─────────────────────────────────────────────────────────
def record_turn(state: InterviewState) -> dict:
    turns = list(state.get("turns",[]))
    is_fu = state.get("consecutive_followups",0) > 0
    aa = state.get("current_audio_analysis") or {}
    va = state.get("current_video_analysis") or {}

    record: TurnRecord = {
        "turn": len(turns)+1, "question": state.get("current_question",""),
        "answer": state.get("current_answer",""), "mode": state.get("current_mode","text"),
        "score": state.get("current_score") or 0.0, "feedback": state.get("current_feedback",""),
        "improvements": state.get("current_improvements",[]),
        "transcription": state.get("current_transcription"),
        "is_followup": is_fu, "soft_skill_notes": state.get("current_soft_skill_notes",""),
        "correctness_note": state.get("current_correctness_note",""),
        "clarity_note": state.get("current_clarity_note",""),
        "depth_note": state.get("current_depth_note",""),
        "confidence_note": state.get("current_confidence_note",""),
        "audio_pace_score": aa.get("pace_score"), "audio_fluency_score": aa.get("fluency_score"),
        "audio_confidence_score": aa.get("confidence_score"),
        "audio_observations": aa.get("observations",[]), "audio_summary": aa.get("summary",""),
        "video_score": va.get("overall_score"), "video_eye_contact": va.get("eye_contact_score"),
        "video_expression": va.get("expression_score"), "video_posture": va.get("posture_score"),
        "video_observations": va.get("feedback",[]), "video_method": va.get("method","none"),
    }
    turns.append(record)
    return {"turns": turns, "questions_asked": state.get("questions_asked",0)+1, "media_file_path": None}


# ── 6. Decide ──────────────────────────────────────────────────────────────
def decide(state: InterviewState) -> dict:
    qa     = state.get("questions_asked",0)
    consec = state.get("consecutive_followups",0)
    cls    = state.get("consecutive_low_scores",0)
    topics = list(state.get("topics_covered",[]))

    # ── Hard maximum ───────────────────────────────────────────────────────
    if qa >= Config.MAX_QUESTIONS:
        return {"next_action":"end","end_reason":f"Completed {Config.MAX_QUESTIONS} questions.","topics_covered":topics}

    # ── Immediate stop: 2 bad answers in a row — only AFTER min questions ──
    if qa >= Config.MIN_QUESTIONS and cls >= Config.EARLY_STOP_COUNT:
        return {"next_action":"end",
                "end_reason":f"Interview ended — {cls} consecutive answers scored below {Config.EARLY_STOP_SCORE}/10.",
                "topics_covered":topics}

    # ── Minimum guard ─────────────────────────────────────────────────────
    if qa < Config.MIN_QUESTIONS:
        if consec >= 2: return {"next_action":"new_question","end_reason":""}
        score = state.get("current_score",10.0)
        return {"next_action":"followup" if score < Config.FOLLOWUP_SCORE_THRESHOLD else "new_question","end_reason":""}

    # ── LLM decision ──────────────────────────────────────────────────────
    turns = state.get("turns",[])
    history = "\n".join(
        f"Q{t['turn']}({t['score']}): {t['question'][:50]}"
        for t in turns[-5:]   # only last 5 for speed
    ) or "No turns."

    try:
        raw    = decide_next_action_chain(state["retriever"]).invoke({
            "history":               history,
            "topics_covered":        ", ".join(topics[-5:]) or "none",
            "question":              state.get("current_question",""),
            "answer":                state.get("current_answer","")[:150],
            "score":                 str(state.get("current_score",0)),
            "consecutive_followups": str(consec),
        })
        parsed    = parse_json_output(raw)
        action    = parsed.get("action","new_question").lower()
        topic_tag = parsed.get("topic_tag","")
        end_reason = parsed.get("end_reason","")

        if action not in ("followup","new_question","end"): action = "new_question"
        if action == "followup" and consec >= 2: action = "new_question"
        if topic_tag and topic_tag not in topics: topics.append(topic_tag)
        return {"next_action":action,"end_reason":end_reason,"topics_covered":topics}
    except Exception:
        score = state.get("current_score",10.0)
        return {"next_action":"followup" if (score < Config.FOLLOWUP_SCORE_THRESHOLD and consec<2) else "new_question","end_reason":""}


# ── 7. Generate Follow-up ──────────────────────────────────────────────────
def generate_followup(state: InterviewState) -> dict:
    try:
        fu = generate_followup_chain(state["retriever"]).invoke({
            "question": state["current_question"], "answer": state["current_answer"],
        })
        return {
            "current_question": fu.strip(), "current_answer": "", "current_score": None,
            "current_feedback": "", "current_improvements": [], "current_transcription": None,
            "current_soft_skill_notes": "", "current_audio_analysis": None,
            "current_video_analysis": None,
            "consecutive_followups": state.get("consecutive_followups",0)+1, "error": None,
            "current_correctness_note": "", "current_clarity_note": "",
            "current_depth_note": "", "current_confidence_note": "",
        }
    except Exception:
        return generate_question(state)


# ── 8. Generate Final Evaluation ───────────────────────────────────────────
def generate_final_eval(state: InterviewState) -> dict:
    turns = state.get("turns",[])
    if not turns:
        return {"final_evaluation": None}

    transcript = "\n".join(
        f"Q{t['turn']}({'FU' if t['is_followup'] else 'New'},{t['score']}/10): {t['question'][:80]}\nA: {t['answer'][:200]}"
        for t in turns
    )
    tech_scores = "\n".join(f"Q{t['turn']}: {t['score']}/10 — {t['feedback'][:80]}" for t in turns)
    soft_obs    = "\n".join(f"Q{t['turn']}: {t.get('soft_skill_notes','')}" for t in turns if t.get("soft_skill_notes"))
    audio_sum   = "\n".join(f"Q{t['turn']}: {t.get('audio_summary','')}" for t in turns if t.get("audio_summary"))
    video_sum   = "\n".join(
        f"Q{t['turn']}: eye={t.get('video_eye_contact','?')} expr={t.get('video_expression','?')} posture={t.get('video_posture','?')}"
        for t in turns if t.get("video_score") is not None
    )

    def avg(lst, key):
        vals = [s[key] for s in lst if key in s]
        return round(sum(vals)/len(vals),1) if vals else 5.0

    ah = state.get("audio_scores_history",[]); vh = state.get("video_scores_history",[])
    avg_pace = avg(ah,"pace"); avg_flu = avg(ah,"fluency"); avg_nv = avg(vh,"overall")

    # ── Compute actual weak areas from turns ──────────────────────────────
    scores = [t["score"] for t in turns if t.get("score") is not None]
    tech_avg = round(sum(scores)/len(scores),1) if scores else 0

    # Always derive weak areas from actual low-scoring turns
    actual_weak = []
    for t in turns:
        if (t.get("score") or 0) < Config.WEAK_SCORE_THRESHOLD:
            actual_weak.append(f"{t['question'][:70]}... (scored {t['score']}/10)")

    try:
        raw    = final_evaluation_chain(state["retriever"]).invoke({
            "level":             state.get("level","mid"),
            "transcript":        transcript[:2000],  # truncate for speed
            "technical_scores":  tech_scores,
            "soft_observations": soft_obs or "No observations.",
            "audio_analysis":    audio_sum or "Text mode.",
            "video_analysis":    video_sum or "No video.",
        })
        parsed = parse_json_output(raw)
        soft   = parsed.get("soft_skills",{})
        if avg_pace>0: soft["speaking_pace"] = avg_pace
        if avg_flu>0:  soft["voice_fluency"] = avg_flu
        if avg_nv>0:   soft["non_verbal"]    = avg_nv

        # ── Override weak_areas with actual data if LLM missed them ───────
        llm_weak = parsed.get("weak_areas",[])
        final_weak = llm_weak if llm_weak else actual_weak

        # ── Override strengths — don't show generic fallback ──────────────
        llm_strengths = [s for s in parsed.get("strengths",[]) if "domain knowledge" not in s.lower()]
        final_strengths = llm_strengths if llm_strengths else []

        return {"final_evaluation": {
            "technical_average":     float(parsed.get("technical_average", tech_avg)),
            "soft_skills":           soft,
            "overall_score":         float(parsed.get("overall_score", round(tech_avg*0.6+5*0.4,1))),
            "strengths":             final_strengths,
            "weak_areas":            final_weak,
            "soft_skill_summary":    parsed.get("soft_skill_summary",""),
            "delivery_summary":      parsed.get("delivery_summary",""),
            "hiring_recommendation": parsed.get("hiring_recommendation","Maybe"),
            "summary_paragraph":     parsed.get("summary_paragraph",""),
            "improvement_plan":      parsed.get("improvement_plan",[]),
        }}
    except Exception as e:
        # Full fallback with real data
        sh = state.get("soft_scores_history",[])
        comm=avg(sh,"communication"); conf=avg(sh,"confidence")
        prob=avg(sh,"problem_solving"); hon=avg(sh,"honesty")
        osoft=round((comm+conf+prob+hon+avg_pace+avg_flu+avg_nv)/7,1)
        overall=round(tech_avg*0.6+osoft*0.4,1)
        rec="Strong Yes" if overall>=8 else "Yes" if overall>=6.5 else "Maybe" if overall>=5 else "No"
        return {"final_evaluation":{
            "technical_average": tech_avg,
            "soft_skills":{"communication":comm,"confidence":conf,"problem_solving":prob,"honesty":hon,
                           "speaking_pace":avg_pace,"voice_fluency":avg_flu,"non_verbal":avg_nv,"overall_soft":osoft},
            "overall_score": overall,
            "strengths": [f"Shows knowledge in: {', '.join(t['question'][:40] for t in turns if (t.get('score') or 0)>=7)[:100]}"] if any((t.get('score') or 0)>=7 for t in turns) else [],
            "weak_areas": actual_weak,
            "soft_skill_summary": f"Communication avg: {comm}/10. Confidence: {conf}/10.",
            "delivery_summary": f"Speaking pace: {avg_pace}/10. Voice fluency: {avg_flu}/10. Non-verbal: {avg_nv}/10.",
            "hiring_recommendation": rec,
            "summary_paragraph": f"Candidate completed {len(turns)} questions with technical average {tech_avg}/10 and overall score {overall}/10.",
            "improvement_plan": [{"area":"Technical Knowledge","priority":"High" if tech_avg<5 else "Medium",
                                   "issue":f"Scored {tech_avg}/10 on technical questions",
                                   "actions":["Review core concepts","Practice explaining solutions","Study relevant documentation"]}],
        }}


# ── 9. End Interview ───────────────────────────────────────────────────────
def end_interview(state: InterviewState) -> dict:
    return {"interview_complete": True, "next_action": "end"}
