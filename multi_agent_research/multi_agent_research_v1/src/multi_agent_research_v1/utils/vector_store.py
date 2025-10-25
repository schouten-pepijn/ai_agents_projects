from dotenv import load_dotenv
import os
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents.base import Document
from typing import List

load_dotenv(".env")


def prepare_vector_store(documents: List[Document]) -> FAISS:
    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBED_MODEL"), base_url=os.getenv("BASE_URL")
    )
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store
