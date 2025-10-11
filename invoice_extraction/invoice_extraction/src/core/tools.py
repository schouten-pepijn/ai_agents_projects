from core.schema import Tool
import re
from typing import Optional, Any, Dict, List


class ToolRegistry:
    """Registry of tools the LLM can use for extraction"""
    
    def __init__(self):
        self.tools = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        # Regex extraction tool
        self.register(Tool(
            name="regex_extract",
            description="Extract text matching a regex pattern. Returns all matches or specific groups.",
            parameters={
                "pattern": "string - regex pattern to match",
                "text": "string - text to search in",
                "group": "integer (optional) - capture group to extract"
            },
            function=self._regex_extract
        ))
        
        # Date parsing tool
        self.register(Tool(
            name="date_parse",
            description="Parse dates in various formats (DD/MM/YYYY, YYYY-MM-DD, Month DD YYYY, etc.)",
            parameters={
                "text": "string - text containing date",
                "format_hint": "string (optional) - expected format"
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
           
        # Fuzzy matching tool
        self.register(Tool(
            name="fuzzy_match",
            description="Find text similar to a target string (handles typos, case differences)",
            parameters={
                "text": "string - text to search in",
                "target": "string - target text to find",
                "threshold": "float (optional) - similarity threshold (0-1)"
            },
            function=self._fuzzy_match
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
    
    """Tool implementations"""
    @staticmethod
    def _regex_extract(pattern: str, text: str, group: Optional[int] = None) -> List[str]:
        """Extract text using regex"""
        try:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            if group is not None:
                return [m.group(group) for m in matches if m.group(group)]
            
            return [m.group(0) for m in matches]
        
        except re.error as e:
            return [f"Error: Invalid regex pattern - {str(e)}"]
    
    @staticmethod
    def _date_parse(text: str, format_hint: Optional[str] = None) -> List[str]:
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
    
    @staticmethod
    def _table_extract(table_text: str, row_pattern: Optional[str] = None, 
                      column: Optional[str] = None) -> List[Dict[str, str]]:
        """Extract data from markdown table"""
        lines = [line.strip() for line in table_text.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return []
        
        # Parse header
        header = [cell.strip() for cell in lines[0].split('|') if cell.strip()]
        
        # Skip separator line
        data_lines = [data_line for data_line in lines[2:] if not set(data_line.replace('|', '').strip()) == {'-', ' ', ''}]
        
        results = []
        for line in data_lines:
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            
            if len(cells) != len(header):
                continue
            
            row_dict = dict(zip(header, cells))
            
            # Filter by row pattern if provided
            if row_pattern and not any(re.search(row_pattern, cell, re.IGNORECASE) for cell in cells):
                continue
            
            # Filter by column if provided
            if column:
                if column in row_dict:
                    results.append({column: row_dict[column]})
                    
            else:
                results.append(row_dict)
        
        return results
    
    @staticmethod
    def _fuzzy_match(text: str, target: str, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find fuzzy matches using simple similarity"""
        # Simple implementation - in production use rapidfuzz or similar
        words = re.findall(r'\b\w+\b', text.lower())
        target_lower = target.lower()
        
        matches = []
        for i, word in enumerate(words):
            # Simple similarity: shared characters / max length
            shared = set(word) & set(target_lower)
            similarity = len(shared) / max(len(word), len(target_lower))
            
            if similarity >= threshold:
                # Get context around match
                start = max(0, i - 3)
                end = min(len(words), i + 4)
                context = ' '.join(words[start:end])
                
                matches.append({
                    'match': word,
                    'similarity': similarity,
                    'context': context
                })
        
        return sorted(matches, key=lambda x: x['similarity'], reverse=True)