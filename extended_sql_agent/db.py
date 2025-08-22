from langchain_community.utilities import SQLDatabase
import pathlib
import requests


def get_db():
    if not pathlib.Path("Chinook.db").exists():
        url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        open("Chinook.db","wb").write(r.content)
    return SQLDatabase.from_uri("sqlite:///Chinook.db")


def get_dialect(db: SQLDatabase):
    return db.dialect