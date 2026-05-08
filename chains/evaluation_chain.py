import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import Config

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "evaluation_prompt.txt")

def _load_prompt():
    with open(os.path.normpath(_PROMPT_PATH)) as f: return f.read()

def evaluate_answer_chain(retriever):
    prompt = ChatPromptTemplate.from_template(_load_prompt())
    llm = ChatOllama(
        model=Config.LLM_MODEL_NAME,
        temperature=0.1,
        base_url=Config.OLLAMA_BASE_URL,
        num_predict=800,   # raised — evaluation JSON is large
        num_ctx=2048,
    )
    def format_docs(docs): return "\n".join(d.page_content[:200] for d in docs)
    def build_inputs(inputs):
        docs = retriever.invoke(inputs.get("question","")[:150])
        return {
            "context":  format_docs(docs),
            "question": inputs["question"][:200],
            "answer":   inputs["answer"][:400],
        }
    return build_inputs | prompt | llm | StrOutputParser()
