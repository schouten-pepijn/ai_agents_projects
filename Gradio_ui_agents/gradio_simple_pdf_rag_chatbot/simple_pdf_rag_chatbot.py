import os
import re
import uuid
from typing import Dict, List
from dotenv import load_dotenv

import gradio as gr
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

import nltk
from nltk.corpus import stopwords


try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english")) | set(stopwords.words("dutch"))
WORD_RE = re.compile(r"\w+", flags=re.UNICODE)

def preprocess_for_retrieval(q: str) -> str:
    tokens = [t.lower() for t in WORD_RE.findall(q)]
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

load_dotenv(".env")

MODEL = (
    os.getenv("MODEL_LARGE")
    or os.getenv("MODEL_MEDIUM")
    or os.getenv("MODEL_SMALL")
)
BASE_URL = os.environ["BASE_URL"]
EMBED_MODEL = os.environ["EMBED_MODEL"]

llm = ChatOllama(
    model=MODEL,
    base_url=BASE_URL,
    temperature=0.1,
)
embedder = OllamaEmbeddings(
    model=EMBED_MODEL,
    base_url=BASE_URL,
)


def load_pdf_docs(path: str, file_label: str) -> List[Document]:
    reader = PdfReader(path)
    docs: List[Document] = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        
        if not text:
            continue
        
        docs.append(Document(
            page_content=text,
            metadata={"page": i+1, "full_page": text, "file": file_label}
        ))
    
    return docs

def chunk_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        length_function=len,
        add_start_index=True
    )
    
    chunks = splitter.split_documents(docs)
    for c in chunks:
        if "full_page" not in c.metadata:
            c.metadata["full_page"] = ""
        
        if "file" not in c.metadata:
            c.metadata["file"] = "unknown"

    return chunks

SESSION_VS: Dict[str, FAISS] = {}
        
        
def ingest_pdfs(file_objs):
    if not file_objs:
        return "", "Upload one or more PDFs."
    
    sid = str(uuid.uuid4())
    all_chunks: List[Document] = []
    files_list: List[str] = []
    
    for f in file_objs:
        fname = os.path.basename(getattr(f, "name", getattr(f, "orig_name", "upload.pdf")))
        files_list.append(fname)
        pages = load_pdf_docs(f.name, file_label=fname)
        
        if not pages:
            continue
        
        chunks = chunk_docs(pages)
        all_chunks.extend(chunks)
        
    if not all_chunks:
        return "", "No extractable text found."
    
    vs = FAISS.from_documents(all_chunks, embedder)
    SESSION_VS[sid] = vs
    
    uniq_files = ", ".join(sorted(set(files_list)))
    
    return sid, f"Indexed {len(all_chunks)} chunks from {len(set(files_list))} file(s): {uniq_files}"


_SENT_SPLIT = re.compile(r"(?<=[\.\?\!])\s+")
def _keywords_from_query(q: str) -> List[str]:
    toks = [t for t in WORD_RE.findall(q) if len(t) > 2]
    seen, out = set(), []
    
    for t in toks:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            out.append(t)
            
    return out[:8]

def _highlight_html(text: str, terms: List[str]) -> str:
    if not terms:
        return text
  
    escaped = [re.escape(t) for t in terms]
    pattern = re.compile(r"(" + "|".join(escaped) + r")", flags=re.IGNORECASE)
    
    return pattern.sub(r"<mark style='background-color: #FFD580;'>\1</mark>", text)


def _make_snippet(full_page: str, chunk_text: str, terms: List[str], window_sents: int = 1) -> str:
    sentences = [s.strip() for s in _SENT_SPLIT.split(chunk_text) if s.strip()]
    
    if not sentences:
        sentences = [chunk_text.strip()]
        
    matches = []
    tl = [t.lower() for t in terms]
    for i, s in enumerate(sentences):
        s_low = s.lower()
        
        if any(t in s_low for t in tl):
            start = max(0, i - window_sents)
            end = min(len(sentences), i + 1 + window_sents)
            matches.append(" ".join(sentences[start:end]).strip())
            
    snippet = matches[0] if matches else " ".join(sentences[: min(3, len(sentences))])
    
    return _highlight_html(snippet, terms)

def build_highlight_blocks(query: str, retrieved: List[Document], max_blocks: int = 4) -> str:
    terms = _keywords_from_query(query)
    
    blocks = []
    for d in retrieved[:max_blocks]:
        page = d.metadata.get("page", "?")
        full_page = d.metadata.get("full_page") or d.page_content
        snippet_html = _make_snippet(full_page, d.page_content, terms, window_sents=1)
        
        block = f"""
        <div style="border:1px solid #ddd; border-radius:12px; padding:12px; margin-bottom:10px;">
          <div style="font-weight:600;">Page {page}</div>
          <div style="margin-top:6px; line-height:1.5;">{snippet_html}</div>
        </div>
        """
        blocks.append(block)
        
    return "\n".join(blocks) if blocks else "<i>No relevant quotes found.</i>"

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer strictly based on the provided context. Be concise, factual. "
     "If the answer is not in context, say so explicitly."),
    ("human",
     "Question:\n{question}\n\nContext:\n{context}\n\n"
     "Answer in up to 6 sentences. No speculation.")
])

def format_context(chunks: List[Document]) -> str:
    blocks = []
    
    for i, d in enumerate(chunks, start=1):
        page = d.metadata.get("page", "?")
        file = d.metadata.get("file", "unknown")
        blocks.append(f"[{i}] ({file}, p.{page}) {d.page_content}")

    return "\n\n".join(blocks)

def ask_question(session_id: str, question: str):
    if not session_id or session_id not in SESSION_VS:
        return "No active session. Upload and index a PDF first.", "<i>—</i>"
    
    if not question or not question.strip():
        return "Provide a question.", "<i>—</i>"

    vs = SESSION_VS[session_id]

    clean_q = preprocess_for_retrieval(question.strip())
    retrieved = vs.similarity_search(clean_q or question.strip(), k=10)
    
    if not retrieved:
        return "No relevant passages found.", "<i>—</i>"

    context = format_context(retrieved)
    answer = (ANSWER_PROMPT | llm).invoke({"question": question.strip(), "context": context}).content.strip()

    highlights_html = build_highlight_blocks(question.strip(), retrieved, max_blocks=5)
    return answer, highlights_html


# UI
with gr.Blocks(title="PDF RAG Chatbot") as rag_pdf_app:
    gr.Markdown("### PDF RAG Chatbot\nUpload a PDF, ask a question, get answers with quotes and page references.")
    
    with gr.Row():
        file = gr.File(label="PDF Upload", file_types=[".pdf"], file_count="multiple")
        
    with gr.Row():
        ingest_btn = gr.Button("Index")
        session_id = gr.Textbox(label="Session ID", interactive=False)
        status = gr.Textbox(label="Status", interactive=False)
        
    with gr.Row():
        question = gr.Textbox(label="Your Question", placeholder="Enter your question here...", lines=2)
        ask_btn = gr.Button("Ask", variant="primary")
        
    answer = gr.Markdown(label="Answer")
    highlights = gr.HTML(label="Sources with highlights")
    
    ingest_btn.click(
        ingest_pdfs,
        inputs=[file],
        outputs=[session_id, status]
    )
    ask_btn.click(
        ask_question,
        inputs=[session_id, question],
        outputs=[answer, highlights],
        show_progress="full"
    )

if __name__ == "__main__":
    rag_pdf_app.launch()