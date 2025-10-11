import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger()


class VectorStoreInMem:
    def __init__(self, emb_model):
        self.emb_model = emb_model
        self.chunks = []
        self.embs = []
        
    def add_chunks(self, chunks):
        
        logger.info(f"Creating embeddings for {len(chunks)} chunks")
        
        txts = [chunk["text"] for chunk in chunks]
        
        embs = self.emb_model.embed_documents(txts)
        
        self.chunks.extend(chunks)
        self.embs.extend(embs)
        
        logger.info(f"Added {len(chunks)} chunks to vectorstore, total {len(self.chunks)} chunks")
        
    def search(self, query, top_k=5, min_similarity=0.0) -> List[Dict[str, Any]]:
        """Search for relevant chunks with optional minimum similarity threshold"""
        if not self.chunks:
            logger.warning("Vector store is empty, cannot search")
            return []
        
        logger.info(f"Searching for top {top_k} chunks relevant to query: '{query[:50]}...'")
        
        query_emb = self.emb_model.embed_query(query)
        
        logger.info("Calculating cosine similarities")
        
        similarities = []
        for i, doc_emb in enumerate(self.embs):
            score = self._cosine_similarity(query_emb, doc_emb)
            if score >= min_similarity:
                similarities.append((score, i))
        
        if not similarities:
            logger.warning(f"No chunks found above minimum similarity {min_similarity}")
            # Return top chunks anyway but with low scores
            similarities = [(self._cosine_similarity(query_emb, doc_emb), i) 
                          for i, doc_emb in enumerate(self.embs)]
            
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        results = []
        for score, idx in similarities[:top_k]:
            chunk = self.chunks[idx].copy()
            chunk["similarity_score"] = round(score, 3)
            results.append(chunk)
            
        logger.info(f"Retrieved {len(results)} relevant chunks with scores: {[r['similarity_score'] for r in results]}")
        
        return results
    
    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        vec1, vec2 = np.array(vec1), np.array(vec2)
        
        dot_prod = np.dot(vec1, vec2)
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_prod / (norm1 * norm2))
    
    def clear(self):
        self.chunks = []
        self.embs = []
        logger.info("Cleared vector store")
        
        
       