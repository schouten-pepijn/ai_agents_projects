from typing import Any, Dict, Callable
from dataclasses import dataclass
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
        "description": "The invoice number, usually starts with INV, #, or is labeled as 'Invoice No', 'Invoice Number'",
        "type": "string",
        "suggested_tools": ["regex_extract"]
    },
    "invoice_date": {
        "description": "The date when the invoice was issued, labeled as 'Date', 'Invoice Date', or 'Issue Date'",
        "type": "date",
        "suggested_tools": ["date_parse"]
    },
    "due_date": {
        "description": "The payment due date, labeled as 'Due Date', 'Payment Due', or 'Date Due'",
        "type": "date",
        "suggested_tools": ["date_parse"]
    },
    "vendor_name": {
        "description": "The name of the vendor or company issuing the invoice, usually at the top or in a 'From' section",
        "type": "string",
        "suggested_tools": ["fuzzy_match"]
    },
    "vendor_address": {
        "description": "The full address of the vendor or company, including street, city, state, and zip code",
        "type": "string",
        "suggested_tools": []
    },
    "customer_name": {
        "description": "The name of the customer or client being billed, labeled as 'Bill To', 'Customer', or 'Client'",
        "type": "string",
        "suggested_tools": []
    },
    "subtotal": {
        "description": "The subtotal amount before taxes, labeled as 'Subtotal' or 'Amount Before Tax'",
        "type": "currency",
        "suggested_tools": ["currency_parse"]
    },
    "tax_amount": {
        "description": "The tax amount, labeled as 'Tax', 'VAT', 'GST', or 'Sales Tax'",
        "type": "currency",
        "suggested_tools": ["currency_parse"]
    },
    "total_amount": {
        "description": "The total amount to be paid including all taxes and fees, labeled as 'Total', 'Amount Due', or 'Grand Total'",
        "type": "currency",
        "suggested_tools": ["currency_parse"]
    },
    "payment_terms": {
        "description": "Payment terms or conditions, such as 'Net 30', 'Due on Receipt', or payment instructions",
        "type": "string",
        "suggested_tools": []
    },
}