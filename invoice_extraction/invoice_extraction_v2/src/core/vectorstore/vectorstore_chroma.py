import logging
import os
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

logger = logging.getLogger()


class VectorStoreChroma:
    def __init__(self, emb_model, persist_dir: str, collection_name: str = "invoice_chunks"):
        self.emb_model = emb_model
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        os.makedirs(self.persist_dir, exist_ok=True)

        self.vectorstore = Chroma(
            embedding_function=self.emb_model,
            persist_directory=self.persist_dir,
            collection_name=self.collection_name
        )
        
        logger.info(f"Initialized Chroma vector store at '{self.persist_dir}' with collection '{self.collection_name}'")
        
    def add_chunks(self, chunks):
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Creating embeddings for {len(chunks)} chunks")
        
        documents = []
        for chunk in chunks:
            # extract text and metadata
            text = chunk.get("text", "")
            metadata = {k: v for k, v in chunk.items() if k != "text"}
           
            # Ensure metadata values are JSON-serializable
            clean_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                
                elif value is None:
                    clean_metadata[key] = ""
                
                else:
                    clean_metadata[key] = str(value)
            
            documents.append(Document(page_content=text, metadata=clean_metadata))
            
        self.vectorstore.add_documents(documents)

        logger.info(f"Added {len(chunks)} chunks to vectorstore, total {len(documents)} chunks")
        
    def search(self, query, top_k=5, min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        
        logger.info(f"Searching ChromaDB for top {top_k} chunks relevant to query: '{query[:50]}...'")

        results_with_scores = self.vectorstore.similarity_search_with_score(
            query=query,
            k=top_k
        )
        
        if not results_with_scores:
            logger.warning("No chunks found in vectorstore")
            return []
        
        formatted_results = []
        for doc, score in results_with_scores:
            similarity = 1 / (1 + score)
            
            if similarity < min_similarity:
                continue
            
            chunk = {
                "text": doc.page_content,
                "similarity_score": round(similarity, 3),
                **doc.metadata,
            }
            
            formatted_results.append(chunk)
            
        if not formatted_results and min_similarity > 0.0:
            logger.warning(f"No chunks found above minimum similarity {min_similarity}")
            
            formatted_results = []
            for doc, distance in results_with_scores:
                similarity = 1 / (1 + distance)
                chunk = {
                    "text": doc.page_content,
                    "similarity_score": round(similarity, 3),
                    **doc.metadata
                }
                
                formatted_results.append(chunk)
                
        logger.info(f"Retrieved {len(formatted_results)} relevant chunks with scores: {[r['similarity_score'] for r in formatted_results]}")
        
        return formatted_results
    
    def clear(self):
        logger.info("Clearing ChromaDB vector store")
        
        self.vectorstore.delete_collection()
        
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.emb_model,
            persist_directory=self.persist_dir
        )
        
        logger.info("ChromaDB vector store cleared")
        
    def get_collection_count(self) -> int:
        try:
            collection = self.vectorstore._collection
            return collection.count()
        
        except Exception as e:
            logger.error(f"Error getting collection count: {e}")
            return 0
            