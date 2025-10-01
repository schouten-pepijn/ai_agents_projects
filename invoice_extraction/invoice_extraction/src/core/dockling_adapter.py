from typing import Dict, Any
from docling.document_converter import DocumentConverter
from .preprocessing import load_bytes


class DocklingAdapter:
    def __init__(self, ocr: bool = True):
        self.ocr = ocr
        
    def parse_to_json(self, path_or_bytes) -> Dict[str, Any]:
        if self.ocr:
            converter = DocumentConverter()
            result = converter.convert(path_or_bytes)
            
            return result.document.export_to_dict()
        
        return self._mock(path_or_bytes)
        
    def _mock(self, x) -> Dict[str, Any]:
        text = load_bytes(x).decode(errors="ignore") if isinstance(x, (bytes, bytearray)) else ""
        
        return {"pages": [], "fulltext": text, "kv": []}