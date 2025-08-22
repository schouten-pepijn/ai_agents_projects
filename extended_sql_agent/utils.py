import re

SQL_READ_ONLY_RE = re.compile(r"^\s*(--.*\n\s*)*(SELECT|WITH)\b", re.IGNORECASE | re.DOTALL)

def ensure_select_only(query: str) -> None:
    forbidden = re.findall(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|MERGE)\b", query, flags=re.I)
    if forbidden:
        raise ValueError(f"Forbidden SQL verb(s): {', '.join(set(forbidden))}")
    if not SQL_READ_ONLY_RE.match(query):
        raise ValueError("Only SELECT/WITH queries are allowed.")