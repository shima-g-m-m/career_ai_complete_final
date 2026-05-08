"""
graph/interview_graph.py
========================
Builds the LangGraph StateGraph.

The key design: the graph runs FULLY between human inputs.
process_answer is an INTERRUPT node — the graph pauses there
waiting for human input, then resumes from that exact point.

Flow per turn:
  [human provides answer via state injection]
  → process_answer → evaluate → record_turn → decide
    → followup/new_question → [PAUSE: wait for next answer]
    → end → generate_final_eval → end_interview → END
"""

from langgraph.graph import StateGraph, END
from graph.state import InterviewState
from graph.nodes import (
    load_cv, generate_question, process_answer, evaluate,
    record_turn, decide, generate_followup,
    generate_final_eval, end_interview,
)


def _route_after_load(state):
    return "end_interview" if state.get("error") else "generate_question"

def _route_after_process(state):
    return "end_interview" if state.get("error") else "evaluate"

def _route_after_decide(state):
    a = state.get("next_action", "new_question")
    if a == "end":      return "generate_final_eval"
    if a == "followup": return "generate_followup"
    return "generate_question"


def build_graph():
    g = StateGraph(InterviewState)

    g.add_node("load_cv",             load_cv)
    g.add_node("generate_question",   generate_question)
    g.add_node("process_answer",      process_answer)
    g.add_node("evaluate",            evaluate)
    g.add_node("record_turn",         record_turn)
    g.add_node("decide",              decide)
    g.add_node("generate_followup",   generate_followup)
    g.add_node("generate_final_eval", generate_final_eval)
    g.add_node("end_interview",       end_interview)

    g.set_entry_point("load_cv")

    g.add_conditional_edges("load_cv", _route_after_load, {
        "generate_question": "generate_question",
        "end_interview":     "end_interview",
    })

    # After question is generated the graph STOPS — waits for human answer
    g.add_edge("generate_question", END)

    # After followup is generated the graph also STOPS
    g.add_edge("generate_followup", END)

    # Human answer injected → graph resumes at process_answer
    g.add_conditional_edges("process_answer", _route_after_process, {
        "evaluate":      "evaluate",
        "end_interview": "end_interview",
    })
    g.add_edge("evaluate",    "record_turn")
    g.add_edge("record_turn", "decide")
    g.add_conditional_edges("decide", _route_after_decide, {
        "generate_followup":   "generate_followup",
        "generate_question":   "generate_question",
        "generate_final_eval": "generate_final_eval",
    })
    g.add_edge("generate_final_eval", "end_interview")
    g.add_edge("end_interview",       END)

    return g.compile()


interview_graph = build_graph()
