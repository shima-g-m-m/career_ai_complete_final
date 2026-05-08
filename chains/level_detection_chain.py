import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import Config

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "level_detection_prompt.txt")

def _load_prompt():
    with open(os.path.normpath(_PROMPT_PATH)) as f: return f.read()

def detect_level_chain(retriever):
    prompt = ChatPromptTemplate.from_template(_load_prompt())
    llm = ChatOllama(
        model=Config.LLM_MODEL_NAME,
        temperature=0.1,
        base_url=Config.OLLAMA_BASE_URL,
        num_predict=150,
        num_ctx=2048,
    )
    def format_docs(docs): return "\n".join(d.page_content[:400] for d in docs)
    def build_inputs(inputs):
        docs = retriever.invoke("experience education skills work history years")
        return {"context": format_docs(docs), "job_description": inputs.get("job_description","Not provided")}
    return build_inputs | prompt | llm | StrOutputParser()
