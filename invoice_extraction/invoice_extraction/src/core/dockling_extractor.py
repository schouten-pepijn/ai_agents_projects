import io
import re
import logging
import uuid
import json
from typing import Dict, Any, List, Optional
from docling.datamodel.base_models import DocumentStream
from langchain.schema import SystemMessage, HumanMessage, AIMessage

from core.prompts import EXTRACTION_PROMPT

logger = logging.getLogger()


class DoclingRAGExtractor:
    def __init__(
        self,
        llm_client,
        emb_client,
        converter,
        vector_store,
        tools,
        chunker,
        max_chunk_tokens
    ):
        self.llm_client = llm_client
        self.emb_client = emb_client
        self.converter = converter
        self.chunker = chunker
        self.max_chunk_tokens = max_chunk_tokens
        
        self.tools = tools
        self.vector_store = vector_store
        
        self.current_doc_id = None
        
        # Fallback regex patterns for common invoice fields
        self.fallback_patterns = {
            "invoice_number": [
                r'(?:invoice|inv)[\s#:]*([A-Z0-9-]+)',
                r'#[\s]?([0-9]+)',
                r'invoice[\s]?number[\s:]*([A-Z0-9-]+)'
            ],
            "invoice_date": [
                r'(?:date|dated)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'(?:date|dated)[\s:]*(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
            ],
            "due_date": [
                r'(?:due|payment due)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            ],
            "total_amount": [
                r'(?:total|amount due|grand total)[\s:]*([£$€¥]?[\s]?\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?)',
            ],
            "subtotal": [
                r'(?:subtotal|sub-total)[\s:]*([£$€¥]?[\s]?\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?)',
            ],
            "tax_amount": [
                r'(?:tax|vat|gst)[\s:]*([£$€¥]?[\s]?\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?)',
            ],
        }

    def process_document(self, path_or_bytes) -> List[Dict[str, Any]]:

        if isinstance(path_or_bytes, (bytes, bytearray)):
            logger.info("Converting file into documentstream")
            path_or_bytes = DocumentStream(name="file", stream=io.BytesIO(path_or_bytes))
        
        logger.info("Converting document with Docling")
        result = self.converter.convert(path_or_bytes)
        
        logger.info("Chunking document")
        chunk_iter = self.chunker.chunk(
            dl_doc=result.document,
            # max_tokens=self.max_chunk_tokens,
        )
        
        chunks = []
        for i, chunk in enumerate(chunk_iter):
            chunk_id = str(uuid.uuid4())
            
            if chunks and len(chunk.text.split()) < 10:   
                chunks[-1]["metadata"]["chunk_index"] = i
                chunks[-1]["text"] += f"\n{chunk.text}"
            else:
                # txt_ctx = self.chunker.contextualize(chunk)
                chunks.append({
                    "id": chunk_id,
                    # "text": txt_ctx,
                    "text": chunk.text,
                    "metadata": {
                        "headings": chunk.meta.headings,
                        "chunk_index": i,
                        "has_table": '|' in chunk.text,
                    }
                })
        
        return chunks
    
    def retrieve_relevant_chunks(
        self,
        field_name: str, 
        field_description: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        
        query = f"{field_name}: {field_description}"  
        logger.info(f" Searching for: {query}")
        
        results = self.vector_store.search(query, top_k=top_k)
        logger.info(f" Retrieved {len(results)} relevant chunks")
        
        return results
    
    def extract_fields(
        self,
        field_schema: Dict[str, Dict[str, str]],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Extract fields with confidence scoring and validation.
        
        Returns:
            Dict with field names and their extracted values with metadata:
            {
                "field_name": {
                    "value": extracted_value,
                    "confidence": float (0-1),
                    "source": "chunk text where found",
                    "status": "success" | "not_found" | "low_confidence" | "error"
                }
            }
        """
        
        extracted_fields = {}
        
        for field_name, field_config in field_schema.items():
            logger.info(f"Processing field: {field_name}")
            
            try:
                relevant_chunks = self.retrieve_relevant_chunks(
                    field_name=field_name,
                    field_description=field_config.get("description", ""),
                    top_k=top_k
                ) 
                logger.debug(f"Relevant chunks: {relevant_chunks}")
                
                if not relevant_chunks:
                    logger.warning(f"No relevant chunks found for field: {field_name}")
                    extracted_fields[field_name] = {
                        "value": None,
                        "confidence": 0.0,
                        "source": None,
                        "status": "not_found"
                    }
                    continue
                
                # Calculate confidence from similarity scores
                avg_similarity = sum(c.get("similarity_score", 0) for c in relevant_chunks) / len(relevant_chunks)
                
                context = self._build_context(relevant_chunks)
                logger.debug(f"Context for field {field_name}:\n{context}")
                
                prompt = self._create_extraction_prompt(
                   field_name,
                   field_config,
                   context
                )
                logger.debug(f"Extraction prompt for field {field_name}:\n{prompt}")
                
                result = self._call_llm_with_tools(
                    prompt,
                    field_config,
                    relevant_chunks
                )
                
                logger.info(f"Extracted value for {field_name}: {result}")
                
                # Try fallback extraction if LLM failed
                if result is None:
                    logger.info(f"Attempting fallback extraction for {field_name}")
                    fallback_result = self._try_fallback_extraction(field_name, context)
                    if fallback_result:
                        result = fallback_result
                        logger.info(f"Fallback extraction successful: {result}")
                
                # Validate and score the result
                if result is None:
                    status = "not_found"
                    confidence = 0.0
                elif avg_similarity < 0.3:
                    status = "low_confidence"
                    confidence = avg_similarity
                else:
                    status = "success"
                    confidence = min(avg_similarity + 0.2, 1.0)  # Boost confidence if extracted
                
                extracted_fields[field_name] = {
                    "value": result,
                    "confidence": round(confidence, 2),
                    "source": relevant_chunks[0]["text"][:200] if relevant_chunks else None,
                    "status": status
                }
                
            except Exception as e:
                logger.error(f"Error extracting field {field_name}: {e}")
                extracted_fields[field_name] = {
                    "value": None,
                    "confidence": 0.0,
                    "source": None,
                    "status": "error",
                    "error": str(e)
                }
        
        return extracted_fields
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        context_parts = []
        for chunk in chunks:
            heading = " > ".join(chunk["metadata"]["headings"]) if chunk["metadata"]["headings"] else "Content"
            context_parts.append(f"[{heading}]\n{chunk['text']}")
        
        return "\n---\n".join(context_parts)
    
    def _create_extraction_prompt(
        self, 
        field_name: str,
        field_config: Dict[str, str], 
        context: str
    ) -> str:
        tool_desc = self.tools.get_tools_description()
        
        suggested_tools = field_config.get("suggested_tools", [])
        suggested_tools_str = ", ".join(suggested_tools) if suggested_tools else "None specified"
        
        return EXTRACTION_PROMPT.format(
            field_name=field_name,
            field_description=field_config.get('description', ''),
            field_type=field_config.get('type', 'string'),
            tools_desc=tool_desc,
            suggested_tools=suggested_tools_str,
            context=context
        )
        
    def _call_llm_with_tools(
        self,
        prompt: str,
        field_config: Dict[str, str],
        relevant_chunks: List[Dict[str, Any]],
        max_iter: int = 3
    ) -> Any:
        """Call LLM with tool support and robust error handling"""
        
        conversation_history = [{"role": "user", "content": prompt}]
        
        for iteration in range(max_iter):
            try:
                response = self._get_llm_response(conversation_history)
                logger.debug(f"LLM response (iteration {iteration + 1}): {response}")
                
                # Check if response indicates not found
                if response.strip().upper() == "NOT_FOUND":
                    logger.info("LLM explicitly marked field as NOT_FOUND")
                    return None
                
                tool_call = self._parse_tool_call(response)
                
                if tool_call:
                    tool_name = tool_call["tool"]
                    params = tool_call.get("parameters", {})
                    
                    logger.info(f"Invoking tool: {tool_name} with params: {params}")
                    
                    try:
                        tool_result = self.tools.invoke_tool(tool_name, **params)
                        
                        conversation_history.append({
                            "role": "assistant",
                            "content": json.dumps(tool_call)
                        })
                        conversation_history.append({
                            "role": "user",
                            "content": f"Tool result: {json.dumps(tool_result)}\n\nNow provide the final extracted value (just the value, no explanation)."
                        })
                        
                    except Exception as e:
                        logger.error(f"Error invoking tool {tool_name}: {e}")
                        conversation_history.append({
                            "role": "user",
                            "content": f"Tool error: {e}\n\nTry extracting the value directly from the context without tools."
                        })
                    
                else:
                    # No tool call - this should be the final value
                    parsed_value = self._parse_final_value(response, field_config)
                    
                    # Validate the extraction
                    if parsed_value is not None:
                        return parsed_value
                    
                    # If still None and we have iterations left, try direct extraction
                    if iteration < max_iter - 1:
                        logger.warning("Failed to parse value, retrying with simpler prompt")
                        conversation_history.append({
                            "role": "user",
                            "content": "Please extract just the value, nothing else. If not found, say NOT_FOUND."
                        })
                    else:
                        return None
                        
            except Exception as e:
                logger.error(f"Error in LLM tool call iteration {iteration + 1}: {e}")
                if iteration == max_iter - 1:
                    return None
            
        logger.warning(f"Max iterations ({max_iter}) reached without successful extraction")
        return None
        
    def _get_llm_response(
        self,
        conversation_history: List[Dict]
    ) -> str:
        try:
            messages = [
                SystemMessage(content="You are a precise invoice data extractor. Only return the requested information without explanation.")
            ]
            
            for msg in conversation_history:
                if msg["role"] == "system":
                    messages.append(SystemMessage(content=msg["content"]))
                    
                elif msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                    
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        
            response =  self.llm_client.invoke(messages)
            
            return response.content
        
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            return f"ERROR: {str(e)}"
        
    def _parse_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse potential tool call from LLM response with robust type checking"""
        try:
            parsed = json.loads(response.strip())
            
            # Ensure parsed is a dictionary before checking keys
            if isinstance(parsed, dict) and "tool" in parsed and "parameters" in parsed:
                return parsed
        except json.JSONDecodeError:
            logger.debug("Response is not a tool call JSON")
        except Exception as e:
            logger.error(f"Unexpected error parsing tool call: {e}")
        
        return None
    
    def _parse_final_value(self, response: str, field_config: Dict[str, str]) -> Any:
        """Parse final value with improved validation and type handling"""
        response = response.strip()
        
        # Handle explicit not found responses
        if response.upper() in ["NOT_FOUND", "NOT FOUND", "N/A", "NONE", "NULL", ""]:
            return None
        
        # Clean common LLM artifacts
        response = re.sub(r'^(Answer:|Result:|Value:|Extracted:)\s*', '', response, flags=re.IGNORECASE)
        response = response.strip('"\'\'`')
        
        field_type = field_config.get("type", "string").lower()
        
        if field_type == 'currency':
            # Extract currency with symbol - improved pattern
            match = re.search(r'([£$€¥₹])\s*([\d,]+\.?\d*)', response)
            if match:
                return f"{match.group(1)}{match.group(2).replace(',', '')}"
            # Pattern without symbol but with decimal
            match = re.search(r'([\d,]+\.\d{2})', response)
            if match:
                return match.group(1).replace(',', '')
            # Fallback: just number
            match = re.search(r'[\d,]+\.?\d*', response)
            if match:
                return match.group().replace(',', '')
            
        elif field_type == 'integer':
            match = re.search(r'\d+', response)
            if match:
                try:
                    return int(match.group())
                except ValueError:
                    logger.warning(f"Could not convert {match.group()} to integer")
                    return None
            
        elif field_type == 'date':
            # Clean and return date
            date_match = re.search(
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|'
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b|'
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b',
                response,
                re.IGNORECASE
            )
            if date_match:
                return date_match.group()
            return response if len(response) > 0 else None
        
        elif field_type == 'email':
            email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', response)
            if email_match:
                return email_match.group()
            return None
        
        elif field_type == 'phone':
            phone_match = re.search(r'[\d\s()+-]{10,}', response)
            if phone_match:
                return phone_match.group().strip()
            return response if len(response) > 0 else None
        
        # Default: return cleaned response
        return response if len(response) > 0 else None
    
    def _try_fallback_extraction(self, field_name: str, context: str) -> Optional[str]:
        """Try pattern-based extraction as fallback when LLM fails"""
        if field_name not in self.fallback_patterns:
            return None
        
        patterns = self.fallback_patterns[field_name]
        
        for pattern in patterns:
            try:
                match = re.search(pattern, context, re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1) if match.lastindex else match.group(0)
                    logger.info(f"Fallback extraction successful for {field_name}: {value}")
                    return value.strip()
            except Exception as e:
                logger.debug(f"Fallback pattern failed for {field_name}: {e}")
                continue
        
        return None
