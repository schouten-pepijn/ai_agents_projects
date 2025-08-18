from dotenv import load_dotenv
import os
load_dotenv(".env")

# DB configuration for SQL agent
DB_PATH = os.getenv("SQLITE_PATH")
MODEL_URL = os.environ["MODEL_URL"]
MODEL = os.environ["MODEL_EXT"]