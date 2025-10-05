"""streamlit run src/main.py"""
import logging
import streamlit as st
from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker

from core.dockling_extractor import DoclingRAGExtractor
from core.vectorstore_in_mem import VectorStoreInMem
from configs.llm_settings import get_ollama_chat_client, get_ollama_embed_client

llm_client = get_ollama_chat_client()
emb_client = get_ollama_embed_client()


logging.root.setLevel(logging.DEBUG)

logger = logging.getLogger()

st.set_page_config(page_title="Invoice Extraction", layout="wide")

st.sidebar.header("Configuration")
k_shots = st.sidebar.number_input("Number of RAG few-shots", min_value=0, max_value=8, value=2, step=1)

uploaded_file = st.file_uploader("Upload invoice (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=False)
run = st.button("Run Extraction")

if uploaded_file and run:
    with st.spinner("Parsing"):
        parser = DoclingRAGExtractor(
            llm_client=None,
            emb_client=None,
            converter=DocumentConverter(),
            chunker=HierarchicalChunker(),
            max_chunk_tokens=512,
        )
        
        chunks = parser.process_document(uploaded_file.getvalue())
        logger.debug(f"Parsed {len(chunks)} chunks")
    
        for i, chunk in enumerate(chunks):
            logger.debug(f"Chunk {i}: {chunk}")
        
        logger.debug("Creating in-memory vector store and adding chunks")  
        vector_store = VectorStoreInMem(emb_model=emb_client)
       
        logger.debug("Adding chunks to vector store")
        vector_store.add_chunks(chunks)
        
        logger.debug("Searching for relevant chunks for field 'Total Amount'")
        results = vector_store.search(query="Total Amount", top_k=5)

        for i, result in enumerate(results):
            logger.debug(f"Result {i}: {result}")
