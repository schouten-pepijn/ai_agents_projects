from langchain_core.prompts import PromptTemplate

INVOICE_PROMPT = PromptTemplate("""
You extract fields from an invoice. Output ONLY a compact JSON per the schema.
No extra keys. No commentary.

Schema (JSON Schema semantics, additionalProperties=false):
{schema_json}

Few-shots (may be noisy; infer style, not exact values):
{few_shots}

Text:
{text}

Return JSON only."""
)