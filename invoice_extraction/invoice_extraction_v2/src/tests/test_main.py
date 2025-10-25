import logging
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

from core.dockling_extractor import DoclingRAGExtractor
from core.vectorstore_in_mem import VectorStoreInMem
from configs.llm_settings import get_ollama_chat_client, get_ollama_embed_client
from core.schema import field_schema
from core.tools import ToolRegistry
from paths.paths import EXAMPLE_PDF

logging.root.setLevel(logging.DEBUG)
logger = logging.getLogger()

def test_main():
    logger.debug("Creating LLM and Embedding clients")
    llm_client = get_ollama_chat_client()
    emb_client = get_ollama_embed_client()
    
    logger.debug("Creating in-memory vector store, converter, and chunker")  
    vector_store = VectorStoreInMem(emb_model=emb_client)
    converter = DocumentConverter()
    chunker = HybridChunker(
        tokenizer="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=1024,
        merge_peers=True
    )
    tools = ToolRegistry()
    
    # chunker = HybridChunker(
    #     tokenizer="sentence-transformers/all-MiniLM-L6-v2",
    #     max_tokens=128,
    #     # merge_peers=True,
    #     # merge_list_items=True,
    #     # window_size=2
    # )
    
    logger.debug("Creating DoclingRAGExtractor")
    parser = DoclingRAGExtractor(
            llm_client=llm_client,
            emb_client=emb_client,
            converter=converter,
            vector_store=vector_store,
            tools=tools,
            chunker=chunker,
            max_chunk_tokens=512,
        )

    logger.debug("Reading example PDF")
    with open(EXAMPLE_PDF, "rb") as f:
        file_bytes = f.read()

    logger.debug("Processing document to extract chunks")
    chunks = parser.process_document(file_bytes)
    logger.debug(f"Parsed {len(chunks)} chunks")
    
    logger.debug("Adding chunks to vector store")
    vector_store.add_chunks(chunks)
    
    logger.debug("Extracting fields from invoice")
    extracted_data = parser.extract_fields(field_schema=field_schema)
    
    logger.info("\n=== EXTRACTION RESULTS ===")
    for field_name, value in extracted_data.items():
        logger.info(f"{field_name}: {value}")
    
    return extracted_data

        
test_main()