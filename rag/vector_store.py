import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from config import Config


def _get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=Config.EMBEDDING_MODEL,
        base_url=Config.OLLAMA_BASE_URL,
    )


def get_vector_store(user_id: str, documents=None) -> FAISS:
    """
    Returns a FAISS vector store for the given user.
    - documents provided → create new store from them.
    - no documents       → load existing store from disk.
    """
    embeddings = _get_embeddings()
    index_path = os.path.join(Config.FAISS_INDEX_DIR, f"index_{user_id}")

    if documents:
        return FAISS.from_documents(documents, embeddings)

    if os.path.exists(index_path):
        return FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )

    raise FileNotFoundError(
        f"No vector store found for user '{user_id}'. Please upload a CV first."
    )


def save_vector_store(vector_store: FAISS, user_id: str) -> None:
    """Persists the FAISS vector store to disk."""
    index_path = os.path.join(Config.FAISS_INDEX_DIR, f"index_{user_id}")
    vector_store.save_local(index_path)
