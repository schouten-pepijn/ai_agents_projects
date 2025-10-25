import logging
from core.vectorstore.vectorstore_chroma import VectorStoreChroma


logger = logging.getLogger()


def create_vectorstore(
    emb_model,
    chroma_persist_dir: str,
    chroma_collection_name: str = "invoice_chunks"
):
    """Create ChromaDB vectorstore instance"""
    logger.info(f"Creating ChromaDB vector store at {chroma_persist_dir}")
    
    return VectorStoreChroma(
        emb_model=emb_model,
        persist_dir=chroma_persist_dir,
        collection_name=chroma_collection_name
    )