from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from pydantic import BaseModel, Field
from enum import Enum


class ToolType(Enum):
    REGEX_EXTRACT = "regex_extract"
    DATE_PARSE = "date_parse"
    CURRENCY_PARSE = "currency_parse"
    TABLE_EXTRACT = "table_extract"
    FUZZY_MATCH = "fuzzy_match"
    
    
@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    
    
field_schema = {
    "invoice_number": {
        "description": "The invoice number, usually starts with INV or #",
        "type": "string",
        "suggested_tools": ["regex_extract"]
    },
    "invoice_date": {
        "description": "The date when the invoice was issued",
        "type": "date",
        "suggested_tools": ["date_parse"]
    },
    "total_amount": {
        "description": "The total amount to be paid, including currency symbol",
        "type": "currency",
        "suggested_tools": ["currency_parse"]
    },
    "vendor_name": {
        "description": "The name of the vendor or company issuing the invoice",
        "type": "string",
        "suggested_tools": ["fuzzy_match"]
    }
}