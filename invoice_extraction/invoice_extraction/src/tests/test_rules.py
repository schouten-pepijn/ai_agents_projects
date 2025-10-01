from core.dockling_adapter import DocklingAdapter
from core.schema import DocumentPayload
import logging

logger = logging.getLogger()

def test_invoice_number_regex():

    doc = DocumentPayload(text="Invoice Number: INV-12345\n", structure={})
    
    converter = DocklingAdapter(ocr=False)
    parsed = converter.parse_to_json(doc.text)
    
    logger.debug(f"Parsed output: {parsed}")