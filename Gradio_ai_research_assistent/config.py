from dotenv import load_dotenv
import os
from dataclasses import dataclass
from langchain_ollama import ChatOllama
from tavily import TavilyClient

load_dotenv(".env")

@dataclass(frozen=True)
class Config:
    """
    Config class is responsible for managing configuration settings and providing utility methods 
    to initialize various components of the application.
    Attributes:
        model (str): The name of the model to be used, fetched from the environment variable `MODEL_SMALL`.
        base_url (str): The base URL for the model, fetched from the environment variable `BASE_URL`.
        tavily_api_key (str): The API key for Tavily, fetched from the environment variable `TAVILY_API_KEY`.
        use_tavily (bool): A flag indicating whether to use Tavily, fetched from the environment variable `USE_TAVILY`. Defaults to False if not set.
    Methods:
        get_llm():
            Initializes and returns a ChatOllama instance configured with the model, base URL, and a fixed temperature.
        get_web_search_client():
            Returns a TavilyClient instance if `use_tavily` is True, otherwise returns None.
        get_graph():
            Builds and returns a graph instance using the `build_graph` function from the `graph` module.
    """
    model: str = os.environ["MODEL_SMALL"]
    base_url: str = os.environ["BASE_URL"]
    tavily_api_key: str = os.environ["TAVILY_API_KEY"]
    use_tavily: bool = os.environ.get("USE_TAVILY", False)
    
    def get_llm(self):
        return ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=0.1
        )

    def get_web_search_client(self):
        if self.use_tavily:
            return TavilyClient(
                api_key=self.tavily_api_key
            )
        else:
            return None

    def get_graph(self):
        from graph import build_graph
        return build_graph()

CONFIG = Config()

llm = CONFIG.get_llm()
tavily = CONFIG.get_web_search_client()
graph = CONFIG.get_graph()
