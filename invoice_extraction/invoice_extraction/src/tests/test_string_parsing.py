import logging
from io import BytesIO
from docling.datamodel.base_models import DocumentStream
from core.dockling_adapter import DocklingAdapter

logger = logging.getLogger()

def test_string_parsing():
    """Test parsing strings with DocklingAdapter"""
    
    # Sample invoice text
    invoice_text = """
    INVOICE
    
    Invoice Number: INV-2024-001
    Date: 2024-01-15
    
    Bill To:
    John Doe
    123 Main St
    Anytown, ST 12345
    
    Description       Qty    Price    Total
    Web Development    10    $100.00  $1000.00
    Consulting         5     $150.00  $750.00
    
    Subtotal: $1750.00
    Tax: $175.00
    Total: $1925.00
    """
    
    logger.debug("Testing with OCR disabled (mock mode)")
    adapter_no_ocr = DocklingAdapter(ocr=False)
    
    text_bytes = invoice_text.encode('utf-8')
    result = adapter_no_ocr.parse_to_json(text_bytes)
    logger.debug(f"Mock result: {result}")
    
    logger.debug("Testing with OCR enabled")
    adapter_with_ocr = DocklingAdapter(ocr=True)
    
    buffer = BytesIO(text_bytes)
    input_text = DocumentStream(name="invoice.html", stream=buffer)
    result_ocr = adapter_with_ocr.parse_to_json(input_text)
    logger.debug(f"OCR result: {result_ocr}")