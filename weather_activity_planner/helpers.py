from datetime import datetime, timezone
from typing import Optional
import re
from models import City


def to_zulu(dt_iso: str) -> str:
    """Normalize any ISO string to UTC Z with seconds."""
    dt = datetime.fromisoformat(
        dt_iso.replace("Z", "+00:00")
    ).astimezone(timezone.utc)
    
    # 'YYYY-MM-DDTHH:MM:SSZ'
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

def safe_get(d: dict, *path, default=None):
    cur = d
    
    for k in path:
        if not isinstance(cur, dict):
            return default
        
        cur = cur.get(k, default)
        
        if cur is default:
            return default
        
    return cur


def safe_parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    fmts = ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d")
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None
    
    
def clean_city_label(city: City) -> str:
    # Prefer the nice label; fallback to raw query
    label = city.label or city.query or ""
    
    # Remove overly specific region codes and commas which may confuse the engine
    label = re.sub(r",?\s*(NH|Noord[-\s]?Holland)\b", "", label, flags=re.IGNORECASE)
    label = re.sub(r"\s*,\s*", " ", label).strip()
    
    # Collapse multiple spaces
    label = re.sub(r"\s{2,}", " ", label)
    
    return label or "Amsterdam"
