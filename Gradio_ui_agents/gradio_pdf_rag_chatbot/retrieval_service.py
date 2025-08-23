"""Document retrieval and question answering service."""
import uuid
import logging
from typing import Dict, List, Tuple, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from models import llm, embedder
from document_processor import PDFProcessor
from text_utils import preprocess_for_retrieval, extract_keywords_from_query, create_snippet
from config import Config

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(self):
        self.sessions: Dict[str, FAISS] = {}
        self.pdf_processor = PDFProcessor()
        
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You answer strictly based on the provided context. Be concise, factual. "
             "If the answer is not in context, say so explicitly."),
            ("human",
             "Question:\n{question}\n\nContext:\n{context}\n\n"
             "Answer in up to 6 sentences. No speculation.")
        ])
    
    def ingest_documents(self, file_objects) -> Tuple[str, str]:
        if not file_objects:
            return "", "Please upload one or more PDF files."
        
        try:
            all_chunks, processed_files = self.pdf_processor.process_files(file_objects)
            
            if not all_chunks:
                return "", "No extractable text found in uploaded files."
            
            vector_store = FAISS.from_documents(all_chunks, embedder)
            session_id = str(uuid.uuid4())
            self.sessions[session_id] = vector_store
            
            unique_files = ", ".join(sorted(set(processed_files)))
            status_msg = (
                f"Successfully indexed {len(all_chunks)} chunks from "
                f"{len(set(processed_files))} file(s): {unique_files}"
            )
            
            logger.info(f"Created session {session_id}: {status_msg}")
            return session_id, status_msg
            
        except Exception as e:
            error_msg = f"Error processing documents: {str(e)}"
            logger.error(error_msg)
            return "", error_msg
    
    def format_context(self, chunks: List[Document]) -> str:
        context_blocks = []
        
        for i, doc in enumerate(chunks, 1):
            page = doc.metadata.get("page", "?")
            file_name = doc.metadata.get("file", "unknown")
            context_blocks.append(
                f"[{i}] ({file_name}, p.{page}) {doc.page_content}"
            )
        
        return "\n\n".join(context_blocks)
    
    def build_highlight_blocks(self, query: str, retrieved_docs: List[Document]) -> str:
        terms = extract_keywords_from_query(query)
        
        if not terms:
            return "<i>No keywords found for highlighting.</i>"
        
        blocks = []
        for doc in retrieved_docs[:Config.MAX_HIGHLIGHT_BLOCKS]:
            page = doc.metadata.get("page", "?")
            file_name = doc.metadata.get("file", "unknown")
            full_page = doc.metadata.get("full_page", doc.page_content)
            
            snippet_html = create_snippet(
                full_page, 
                doc.page_content, 
                terms, 
                window_sentences=Config.SIMILARITY_WINDOW_SENTENCES
            )
            
            block = f"""
            <div style="border:1px solid #ddd; border-radius:12px; padding:12px; margin-bottom:10px;">
                <div style="font-weight:600; color:#2563eb;">
                    {file_name} - Page {page}
                </div>
                <div style="margin-top:6px; line-height:1.5;">
                    {snippet_html}
                </div>
            </div>
            """
            blocks.append(block)
        
        return "\n".join(blocks) if blocks else "<i>No relevant quotes found.</i>"
    
    def answer_question(self, session_id: str, question: str) -> Tuple[str, str]:
        if not session_id or session_id not in self.sessions:
            return ("No active session found. Please upload and index PDF files first.", 
                   "<i>No session available</i>")
        
        if not question or not question.strip():
            return "Please provide a question.", "<i>No question provided</i>"
        
        question = question.strip()
        if len(question) > Config.MAX_QUESTION_LENGTH:
            return ("Question too long. Please limit to 500 characters.", 
                   "<i>Question too long</i>")
        
        try:
            vector_store = self.sessions[session_id]
            
            clean_query = preprocess_for_retrieval(question)
            search_query = clean_query if clean_query else question
            
            retrieved_docs = vector_store.similarity_search(
                search_query, 
                k=Config.MAX_RETRIEVAL_DOCS
            )
            
            if not retrieved_docs:
                return ("No relevant information found in the documents.", 
                       "<i>No relevant passages found</i>")
            
            context = self.format_context(retrieved_docs)
            answer_response = (self.answer_prompt | llm).invoke({
                "question": question,
                "context": context
            })
            
            answer = answer_response.content.strip()
            
            highlights_html = self.build_highlight_blocks(question, retrieved_docs)
            
            logger.info(f"Answered question for session {session_id}")
            return answer, highlights_html
            
        except Exception as e:
            error_msg = f"Error answering question: {str(e)}"
            logger.error(error_msg)
            return error_msg, "<i>Error occurred during processing</i>"
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        if session_id in self.sessions:
            vector_store = self.sessions[session_id]
            
            return {
                "session_id": session_id,
                "document_count": len(vector_store.docstore._dict) if hasattr(vector_store, 'docstore') else 0
            }
            
        return None
    
    def clear_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session {session_id}")
            
            return True
        
        return False
