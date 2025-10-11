from langchain_core.prompts import PromptTemplate

EXTRACTION_PROMPT = PromptTemplate.from_template("""You are an expert at extracting structured data from invoice documents.

TASK: Extract the field "{field_name}" from the document context below.

FIELD DETAILS:
- Description: {field_description}
- Type: {field_type}
- Suggested tools: {suggested_tools}

AVAILABLE TOOLS:
{tools_desc}

DOCUMENT CONTEXT:
{context}

INSTRUCTIONS:
1. Carefully analyze the context above to locate the exact value for "{field_name}"
2. Look for labels like: {field_name}, {field_description}
3. If you need to use a tool (e.g., for date parsing, currency extraction, or pattern matching), respond with ONLY a JSON object in this exact format:
   {{"tool": "tool_name", "parameters": {{"param_name": "value"}}}}

4. If you don't need tools OR after receiving tool results, respond with ONLY the extracted value
   - Just the value, nothing else
   - No explanations, no labels, no extra text
   - If the field cannot be found, respond with exactly: NOT_FOUND

EXAMPLES:
- For invoice_number, respond with: INV-12345
- For total_amount, respond with: $1,234.56
- For invoice_date, respond with: 2024-01-15
- For vendor_name, respond with: Acme Corporation

CRITICAL RULES:
- Extract ONLY the value, no extra text
- If uncertain, still provide your best answer (don't say "I'm not sure")
- If truly not found in context, say: NOT_FOUND

RESPONSE (either JSON for tool call OR the final extracted value):""")

