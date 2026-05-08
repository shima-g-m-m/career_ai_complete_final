import json
import re


def parse_json_output(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("Empty output from LLM.")

    cleaned = text.strip()

    # Strip markdown fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    # Extract first {...} block
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    # Remove trailing commas
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fix unescaped newlines inside strings
        cleaned = re.sub(r'(?<!\\)\n', ' ', cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to fix truncated JSON by closing open braces/brackets
            cleaned = _close_truncated_json(cleaned)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                # Last resort: extract score manually
                score = re.search(r'"score"\s*:\s*([\d.]+)', cleaned)
                feedback = re.search(r'"feedback"\s*:\s*"([^"]{5,})"', cleaned)
                if score:
                    return {
                        "score": float(score.group(1)),
                        "feedback": feedback.group(1) if feedback else "See improvements.",
                        "improvements": [],
                        "correctness_note": "", "clarity_note": "",
                        "depth_note": "", "confidence_note": "",
                        "soft_skill_notes": "",
                        "soft_scores": {"communication":5,"confidence":5,"problem_solving":5,"honesty":5}
                    }
                raise ValueError(f"Cannot parse JSON. Raw: {text[:200]}")


def _close_truncated_json(s: str) -> str:
    """Attempt to close a truncated JSON string."""
    # Count open braces and brackets
    depth_brace  = s.count('{') - s.count('}')
    depth_bracket= s.count('[') - s.count(']')
    in_string    = False
    escape_next  = False

    for ch in s:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string

    # Close open string if needed
    if in_string:
        s += '"'

    # Close open arrays and objects
    s += ']' * max(0, depth_bracket)
    s += '}' * max(0, depth_brace)
    return s
