import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import Config

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "final_evaluation_prompt.txt")

def _load_prompt():
    with open(os.path.normpath(_PROMPT_PATH)) as f: return f.read()

def final_evaluation_chain(retriever):
    prompt = ChatPromptTemplate.from_template(_load_prompt())
    llm = ChatOllama(
        model=Config.LLM_MODEL_NAME,
        temperature=0.1,
        base_url=Config.OLLAMA_BASE_URL,
        num_predict=1200,  # final eval needs most tokens
        num_ctx=3072,
    )
    def format_docs(docs): return "\n".join(d.page_content[:300] for d in docs)
    def build_inputs(inputs):
        docs = retriever.invoke("candidate skills experience background")
        return {
            "context":          format_docs(docs),
            "level":            inputs["level"],
            "transcript":       inputs["transcript"],
            "technical_scores": inputs["technical_scores"],
            "soft_observations":inputs["soft_observations"],
            "audio_analysis":   inputs["audio_analysis"],
            "video_analysis":   inputs["video_analysis"],
        }
    return build_inputs | prompt | llm | StrOutputParser()
