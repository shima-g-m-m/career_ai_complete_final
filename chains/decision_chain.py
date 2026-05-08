import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import Config

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "decision_prompt.txt")

def _load_prompt():
    with open(os.path.normpath(_PROMPT_PATH)) as f: return f.read()

def decide_next_action_chain(retriever):
    prompt = ChatPromptTemplate.from_template(_load_prompt())
    llm = ChatOllama(
        model=Config.LLM_MODEL_NAME,
        temperature=0.1,
        base_url=Config.OLLAMA_BASE_URL,
        num_predict=100,
        num_ctx=1024,
    )
    def format_docs(docs): return "\n".join(d.page_content[:200] for d in docs)
    def build_inputs(inputs):
        docs = retriever.invoke("skills experience")
        return {
            "context": format_docs(docs),
            "history": inputs.get("history",""),
            "topics_covered": inputs.get("topics_covered","none"),
            "question": inputs.get("question",""),
            "answer": inputs.get("answer","")[:200],
            "score": inputs.get("score","0"),
            "consecutive_followups": inputs.get("consecutive_followups","0"),
        }
    return build_inputs | prompt | llm | StrOutputParser()
