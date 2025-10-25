from core.schema import Tool
import re
from typing import Optional, Any, Dict, List


class ToolRegistry:
    """Simplified registry of essential tools for invoice extraction"""
    
    def __init__(self):
        self.tools = {}
        self._register_essential_tools()
    
    def _register_essential_tools(self):
        """Register only essential tools for invoice processing"""
        # Date parsing tool
        self.register(Tool(
            name="date_parse",
            description="Parse dates in various formats (DD/MM/YYYY, YYYY-MM-DD, Month DD YYYY, etc.)",
            parameters={
                "text": "string - text containing date"
            },
            function=self._date_parse
        ))
        
        # Currency parsing tool
        self.register(Tool(
            name="currency_parse",
            description="Extract currency amounts with symbols ($, €, £) and handle thousands separators",
            parameters={
                "text": "string - text containing currency amount"
            },
            function=self._currency_parse
        ))
    
    def register(self, tool: Tool):
        """Register a new tool"""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def get_tools_description(self) -> str:
        """Get formatted description of all tools for LLM prompt"""
        descriptions = []
        for tool in self.tools.values():
            params = "\n  ".join([f"- {k}: {v}" for k, v in tool.parameters.items()])
            descriptions.append(
                f"Tool: {tool.name}\n"
                f"Description: {tool.description}\n"
                f"Parameters:\n  {params}"
            )
            
        return "\n\n".join(descriptions)
    
    def invoke_tool(self, tool_name: str, **kwargs) -> Any:
        tool = self.get_tool(tool_name)
        
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        return tool.function(**kwargs)
    
    """Essential tool implementations"""
    @staticmethod
    def _date_parse(text: str) -> List[str]:
        """Parse dates from text"""
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # DD/MM/YYYY or MM/DD/YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY-MM-DD
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b',  # Month DD, YYYY
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b',  # DD Month YYYY
        ]
        
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text, re.IGNORECASE))
        
        return list(set(dates))
    
    @staticmethod
    def _currency_parse(text: str) -> List[Dict[str, Any]]:
        """Extract currency amounts"""
        # Pattern for currency with optional thousands separators
        pattern = r'([£$€¥])\s*(\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?)'
        
        matches = re.finditer(pattern, text)
        results = []
        
        for match in matches:
            symbol = match.group(1)
            amount_str = match.group(2).replace(',', '').replace(' ', '')
            
            try:
                amount = float(amount_str)
                results.append({
                    'symbol': symbol,
                    'amount': amount,
                    'formatted': match.group(0)
                })
                
            except ValueError:
                continue
        
        return results