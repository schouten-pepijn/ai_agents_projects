from dotenv import load_dotenv
import os
load_dotenv(".env")


DB_PATH = os.getenv("SQLITE_PATH")
BASE_URL = os.environ["BASE_URL"]
MODEL = os.environ["MODEL_EXT"]