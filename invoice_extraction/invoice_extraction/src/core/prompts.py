from langchain_core.prompts import PromptTemplate

EXTRACTION_PROMPT = PromptTemplate(
   template="""
You are extracting the field "{field_name}" from a document.

Field Description: {field_description}
Field Type: {field_type}

You have access to the following tools:
{tools_desc}

Suggested tools for this field: {suggested_tools}

Document Context:
{context}

Instructions:
1. Analyze the context to find the field value
2. If needed, use tools by responding with JSON in this format:
   {{"tool": "tool_name", "parameters": {{"param": "value"}}}}
3. After using tools (or if not needed), provide the final extracted value
4. If the field cannot be found, respond with "NOT_FOUND"

Your response:""",
   input_variables=[
      "field_name", 
      "field_description", 
      "field_type",
      "tools_desc", 
      "suggested_tools",
      "context"
   ]
)

