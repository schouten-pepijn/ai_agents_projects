from typing import List
from langchain_ollama import ChatOllama, OllamaEmbeddings
from config import SETTINGS
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.chat_message_histories import ChatMessageHistory

class LTMemory:
    def __init__(self):
        self.emb = OllamaEmbeddings(
            model=SETTINGS.embed_model,
            base_url=SETTINGS.model_url
        )
        
        self.vs = FAISS.from_texts([""], self.emb)
        
        self.vs.docstore._dict.pop(list(self.vs.docstore._dict.keys())[0], None)
    
    def add_texts(self, texts: List[str], metadata: dict | None = None):
        self.vs.add_documents([Document(page_content=t, metadata=metadata or {}) for t in texts])
        
    def retriever(self, k: int = 4):
        return self.vs.as_retriever(search_kwargs={"k": k})
    

# short term chat history per session id
_STORE: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _STORE:
        _STORE[session_id] = ChatMessageHistory()
    return _STORE[session_id]

