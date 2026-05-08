import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import Config

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "question_prompt.txt")

def _load_prompt():
    with open(os.path.normpath(_PROMPT_PATH)) as f: return f.read()

def generate_question_chain(retriever):
    prompt = ChatPromptTemplate.from_template(_load_prompt())
    llm = ChatOllama(
        model=Config.LLM_MODEL_NAME,
        temperature=Config.TEMPERATURE,
        base_url=Config.OLLAMA_BASE_URL,
        num_predict=120,
        num_ctx=2048,
    )
    def format_docs(docs): return "\n".join(d.page_content[:400] for d in docs)
    def build_inputs(inputs):
        docs = retriever.invoke("candidate background skills experience projects")
        return {"context": format_docs(docs), "level": inputs.get("level","mid"), "topics_covered": inputs.get("topics_covered","none yet")}
    return build_inputs | prompt | llm | StrOutputParser()
