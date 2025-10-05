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
            
        self.vector_store.add_chunks(chunks)
        
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
    ) -> Dict[str, Any]:
        """
        Format:
        {
            "field_name": {
                "description": "Description of the field",
                "type": "string",
                "suggested_tools": ["tool1"]
            }
        }
        """
        
        extracted_fields = {}
        
        for field_name, field_config in field_schema.items():
            logger.info(f"Processing field: {field_name}")
            
            relevant_chunks = self.retrieve_relevant_chunks(
                field_name=field_name,
                field_description=field_config.get("description", "")
            ) 
            logger.debug(f"Relevant chunks: {relevant_chunks}")
            
            if not relevant_chunks:
                logger.warning(f"No relevant chunks found for field: {field_name}")
                extracted_fields[field_name] = None
                continue
            
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
                field_config
            )
            logger.info(f"Extracted value for {field_name}: {result}")
    
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
        max_iter: int = 3
    ) -> Any:
        
        conversation_history= [{"role": "user", "content": prompt}]
        
        for _ in range(max_iter):
            response = self._get_llm_response(conversation_history)
            logger.debug(f"LLM response: {response}")
            
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
                        "content": f"Tool result: {json.dumps(tool_result)}\n\nNow provide the final extracted value."
                    })
                    
                except Exception as e:
                    logger.error(f"Error invoking tool {tool_name}: {e}")
                    conversation_history.append({
                        "role": "user",
                        "content": f"Tool error: {e}\n\nTry a different approach or provide the value directly."
                    })
                
            else:
                return self._parse_final_value(response, field_config)
            
        return None
        
    def _get_llm_response(
        self,
        conversation_history: List[Dict]
    ) -> str:
        try:
            messages = []
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
        try:
            parsed = json.loads(response.strip())
            if "tool" in parsed and 'parameters' in parsed:
                return parsed
        
        except json.JSONDecodeError:
            logger.warning("Failed to parse tool call from response")
            pass
        
        return None
    
    def _parse_final_value(self, response: str, field_config: Dict[str, str]) -> Any:
        response = response.strip()
        
        if response == "NOT_FOUND":
            return None
        
        field_type = field_config.get("type", "string").lower()
        
        if field_type == 'currency':
            match = re.search(r'[\d,]+\.?\d*', response)
            
            if match:
                return float(match.group().replace(',', ''))
            
        elif field_type == 'integer':
            match = re.search(r'\d+', response)
            
            if match:
                return int(match.group())
            
        elif field_type == 'date':
            return response
        
        return response
