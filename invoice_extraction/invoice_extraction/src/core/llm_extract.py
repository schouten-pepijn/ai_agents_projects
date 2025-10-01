import json
from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough

from core.invoice_prompt import INVOICE_PROMPT
from core.schema import FieldSpec, DocumentPayload, Candidate
from core.vectorstore import query_field_shots
from configs.env_settings import env_settings

def build_schema(fields: List[FieldSpec]) -> Dict[str, Any]:
    properties = {f.name: {"type": f.type_} for f in fields}

    return {"type": "object", "properties": properties, "additionalProperties": False}
        
def render_few_shots(fields: List[FieldSpec], k: int = 2) -> str:
    parts = []
    for f in fields:
        docs = query_field_shots(f.name, k=k)
        
        for d in docs:
            value = d["meta"].get("value")
            
            if value:
                parts.append(f'Input snippet:\n{d["text"]}\nOutput:\n{{"{f.name}": "{value}"}}\n')

    return "\n".join(parts[:8])

def llm_chain():
    llm = ChatOllama(
        model=env_settings.ollama_model,
        base_url=env_settings.ollama_base_url,
        temperature=0,
    )
    
    chain = (
        {
            "schema_json": RunnablePassthrough(),
            "few_shots": RunnablePassthrough(),
            "text": RunnablePassthrough()
        } | INVOICE_PROMPT | llm
    )
    
    return chain


def extract_with_llm(fields: List[FieldSpec], doc: DocumentPayload, k_shots: int = 2) -> Dict[str, Candidate]:
    schema_json = build_schema(fields)
    few_shots = render_few_shots(fields, k=k_shots)
    
    chain = llm_chain()
    
    response = chain.invoke({
        "schema_json": schema_json,
        "few_shots": few_shots,
        "text": doc.text
    })
    
    content = response.content if hasattr(response, "content") else str(response)
    
    start, end = content.find("{"), content.rfind("}")
    
    data = {}
    if start != -1 and end > start:
        try:
            data = json.loads(content[start:end+1])
            
        except Exception:
            data = {}
            
    out = {}
    for f in fields:
        v = data.get(f.name)
        
        if v:
            out[f.name] = Candidate(value=v, confidence=0.72, source="llm", evidence={"model": env_settings.ollama_model})
            
    return out
    
    
    