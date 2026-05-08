import os
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config


class CVLoader:
    """Loads a CV PDF and splits it into LangChain Document chunks."""

    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CV file not found: {file_path}")
        if not file_path.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported.")
        self.file_path = file_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )

    def load_and_split(self):
        """Loads the PDF and returns a list of Document chunks."""
        loader = PDFMinerLoader(self.file_path)
        documents = loader.load()
        if not documents:
            raise ValueError("PDF appears to be empty or could not be read.")
        chunks = self.text_splitter.split_documents(documents)
        if not chunks:
            raise ValueError("No text chunks could be extracted from the CV.")
        return chunks
