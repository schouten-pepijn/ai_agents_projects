from datetime import datetime, timezone


def to_zulu(dt_iso: str) -> str:
    """Normalize any ISO string to UTC Z with seconds."""
    dt = datetime.fromisoformat(
        dt_iso.replace("Z", "+00:00")
    ).astimezone(timezone.utc)
    
    # Ticketmaster prefers 'YYYY-MM-DDTHH:MM:SSZ'
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