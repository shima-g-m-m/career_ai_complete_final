import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import Config

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "followup_prompt.txt")

def _load_prompt():
    with open(os.path.normpath(_PROMPT_PATH)) as f: return f.read()

def generate_followup_chain(retriever):
    prompt = ChatPromptTemplate.from_template(_load_prompt())
    llm = ChatOllama(
        model=Config.LLM_MODEL_NAME,
        temperature=Config.TEMPERATURE,
        base_url=Config.OLLAMA_BASE_URL,
        num_predict=100,
        num_ctx=2048,
    )
    def format_docs(docs): return "\n".join(d.page_content[:300] for d in docs)
    def build_inputs(inputs):
        docs = retriever.invoke(f"{inputs.get('question','')} {inputs.get('answer','')}".strip()[:300])
        return {"context": format_docs(docs), "question": inputs["question"], "answer": inputs["answer"][:300]}
    return build_inputs | prompt | llm | StrOutputParser()
