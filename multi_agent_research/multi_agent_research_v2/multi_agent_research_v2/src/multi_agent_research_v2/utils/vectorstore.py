import logging
import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from multi_agent_research_v2.config.config import WorkflowConfig
from multi_agent_research_v2.utils.llm import initialize_embeddings

logger = logging.getLogger("multi_agent_research")


class VectorStoreManager:
    """A class for managing FAISS vector stores with embedding integration."""

    def __init__(self, config: WorkflowConfig):
        """Initialize the VectorStoreManager with configuration."""
        self.config = config
        self.embeddings = None
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Initialize embeddings using the llm.py module."""
        self.embeddings = initialize_embeddings(
            model_name=self.config.embed_model, base_url=self.config.base_url
        )

    def prepare_vector_store(self, documents: List[Document]) -> FAISS:
        """Initialize FAISS vector store with documents."""
        logger.info(f"Preparing vector store with {len(documents)} documents")

        if not self.embeddings:
            self._initialize_embeddings()

        vector_store = FAISS.from_documents(documents, self.embeddings)
        return vector_store

    def prepare_persistent_vector_store(
        self,
        documents: List[Document],
        persist_directory: str,
        update_on_new_documents: bool = True,
    ) -> FAISS:
        """Initialize or load a persistent FAISS vector store."""
        logger.info(
            f"Preparing persistent vector store at '{persist_directory}' with {len(documents)} documents"
        )

        if not self.embeddings:
            self._initialize_embeddings()

        # Ensure the directory exists
        os.makedirs(persist_directory, exist_ok=True)

        try:
            # Attempt to load an existing index
            vector_store = FAISS.load_local(persist_directory, self.embeddings)
            logger.info("Loaded existing vector store from disk")

            # Optionally update with new documents
            if update_on_new_documents and documents:
                try:
                    vector_store.add_documents(documents)
                    logger.info("Added new documents to existing vector store")

                    vector_store.save_local(persist_directory)
                    logger.info("Re-saved updated vector store to disk")

                except Exception as e:
                    logger.warning(
                        f"Failed to add new documents to existing vector store: {e}"
                    )

        except Exception as e:
            # No existing index or failed to load; create a new one
            if isinstance(e, FileNotFoundError) or "No such file or directory" in str(
                e
            ):
                logger.info("No existing vector store found; creating a new one")

            else:
                logger.warning(
                    f"Error loading existing vector store ({e}); rebuilding index"
                )

            vector_store = FAISS.from_documents(documents, self.embeddings)

            vector_store.save_local(persist_directory)
            logger.info("Saved new vector store to disk")

        return vector_store

    def add_documents(self, vector_store: FAISS, documents: List[Document]):
        """Add documents to an existing vector store."""
        logger.info(f"Adding {len(documents)} documents to existing vector store")

        vector_store.add_documents(documents)

    def similarity_search(
        self, vector_store: FAISS, query: str, k: int = 4
    ) -> List[Document]:
        """Perform similarity search on the vector store."""
        logger.info(f"Performing similarity search for query: '{query}' with k={k}")

        return vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self, vector_store: FAISS, query: str, k: int = 4
    ) -> List[tuple]:
        """Perform similarity search with scores on the vector store."""
        logger.info(
            f"Performing similarity search with scores for query: '{query}' with k={k}"
        )

        return vector_store.similarity_search_with_score(query, k=k)
