import re
import nltk
from typing import List, Set
from nltk.corpus import stopwords

try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

STOPWORDS: Set[str] = set(stopwords.words("english")) | set(stopwords.words("dutch"))
WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+")

def preprocess_for_retrieval(query: str) -> str:
    if not query:
        return ""
    
    tokens = [token.lower() for token in WORD_RE.findall(query)]
    tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 1]
    return " ".join(tokens)

def extract_keywords_from_query(query: str, max_keywords: int = 8) -> List[str]:
    tokens = [token for token in WORD_RE.findall(query) if len(token) > 2]
    seen = set()
    keywords = []
    
    for token in tokens:
        token_lower = token.lower()
        if token_lower not in seen:
            seen.add(token_lower)
            keywords.append(token)
            if len(keywords) >= max_keywords:
                break
    
    return keywords

def highlight_text(
    text: str,
    terms: List[str],
    bg_color: str = "#FFD580"
) -> str:

    if not terms or not text:
        return text
    
    escaped_terms = [re.escape(term) for term in terms]
    pattern = re.compile(r"(" + "|".join(escaped_terms) + r")", flags=re.IGNORECASE)
    
    return pattern.sub(
        f"<mark style='background-color: {bg_color};'>\\1</mark>", 
        text
    )

def create_snippet(
    full_page: str,
    chunk_text: str, 
    terms: List[str], 
    window_sentences: int = 1
) -> str:

    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(chunk_text) if s.strip()]
    
    if not sentences:
        sentences = [chunk_text.strip()]
    
    term_lower = [term.lower() for term in terms]
    matching_snippets = []
    
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        
        if any(term in sentence_lower for term in term_lower):
            start = max(0, i - window_sentences)
            end = min(len(sentences), i + 1 + window_sentences)
            snippet = " ".join(sentences[start:end]).strip()
            matching_snippets.append(snippet)
    
    final_snippet = (
        matching_snippets[0] if matching_snippets 
        else " ".join(sentences[:min(3, len(sentences))])
    )
    
    return highlight_text(final_snippet, terms)
