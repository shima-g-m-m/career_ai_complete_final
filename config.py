import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class Config:
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LLM_MODEL_NAME: str  = os.getenv("LLM_MODEL_NAME", "llama3.2")
    TEMPERATURE: float   = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    # tiny is 4x faster than base, still accurate enough for interviews
    WHISPER_MODEL: str   = os.getenv("WHISPER_MODEL", "tiny")
    WHISPER_BACKEND: str = os.getenv("WHISPER_BACKEND", "local")

    FAISS_INDEX_DIR: str = "faiss_indexes"
    UPLOAD_DIR: str      = "uploads"
    os.makedirs(UPLOAD_DIR,      exist_ok=True)
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

    CHUNK_SIZE: int    = int(os.getenv("CHUNK_SIZE",    "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    RETRIEVER_K: int   = int(os.getenv("RETRIEVER_K",  "2"))   # 2 chunks = much faster

    FOLLOWUP_SCORE_THRESHOLD: float = float(os.getenv("FOLLOWUP_SCORE_THRESHOLD", "6.0"))
    WEAK_SCORE_THRESHOLD:     float = float(os.getenv("WEAK_SCORE_THRESHOLD",     "5.0"))
    EARLY_STOP_SCORE:  float = float(os.getenv("EARLY_STOP_SCORE",  "4.0"))
    # Stop after 2 consecutive bad answers (as requested)
    EARLY_STOP_COUNT:  int   = int(os.getenv("EARLY_STOP_COUNT",   "2"))
    MIN_QUESTIONS: int = int(os.getenv("MIN_QUESTIONS", "4"))
    MAX_QUESTIONS: int = int(os.getenv("MAX_QUESTIONS", "10"))
