from pathlib import Path

def load_bytes(path_or_bytes) -> bytes:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        return bytes(path_or_bytes)
    
    p = Path(path_or_bytes)
    
    return p.read_bytes()