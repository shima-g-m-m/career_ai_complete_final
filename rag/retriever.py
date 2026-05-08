from langchain_community.vectorstores import FAISS
from config import Config

def get_retriever(vector_store: FAISS, k: int = None):
    """Returns a retriever. Uses Config.RETRIEVER_K (default 3) for speed."""
    return vector_store.as_retriever(search_kwargs={"k": k or Config.RETRIEVER_K})
