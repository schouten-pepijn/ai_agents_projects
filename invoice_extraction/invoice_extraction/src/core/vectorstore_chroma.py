import json
import os
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

from configs import env_settings
from paths import VECTORDB_DIR

def get_vectorstore(persist_dir: str = VECTORDB_DIR) -> Chroma:
    embeddings = OllamaEmbeddings(
        model=env_settings.ollama_model,
        embed_model=env_settings.embed_model
    )
    return Chroma(collection_name="invoice_fields", persist_directory=persist_dir, embedding_function=embeddings)


def query_field_shots(field: str, k: int = 4, persist_dir: str = VECTORDB_DIR):
    vectorstore = get_vectorstore(persist_dir=persist_dir)
    
    docs = vectorstore.similarity_search(
        query=field,
        k=k,
        filter={"field": field}
    )
    
    out = []
    for d in docs:
        out.append({"text": d.page_content, "meta": d.metadata})
        
    return out