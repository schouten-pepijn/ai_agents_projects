import os
from typing import List
import logging
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config

"""
    Kijken of docling een goede toevoeging kan zijn
    Mogelijk ook orc toevoegen
"""
logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )

    def load_pdf_documents(self, file_path: str, file_label: str) -> List[Document]:
        try:
            reader = PdfReader(file_path)
            documents: List[Document] = []

            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                text = " ".join(text.split())

                if not text.strip():
                    logger.warning(f"Empty text on page {page_num} of {file_label}")
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "page": page_num,
                            "full_page": text,
                            "file": file_label,
                            "source": file_path,
                        },
                    )
                )

            logger.info(f"Loaded {len(documents)} pages from {file_label}")
            return documents

        except Exception as e:
            logger.error(f"Error processing PDF {file_label}: {str(e)}")
            raise

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        try:
            chunks = self.text_splitter.split_documents(documents)

            for chunk in chunks:
                if "full_page" not in chunk.metadata:
                    chunk.metadata["full_page"] = ""
                if "file" not in chunk.metadata:
                    chunk.metadata["file"] = "unknown"

            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            return chunks

        except Exception as e:
            logger.error(f"Error chunking documents: {str(e)}")
            raise

    def process_files(self, file_objects) -> tuple[List[Document], List[str]]:
        all_chunks: List[Document] = []
        processed_files: List[str] = []

        for file_obj in file_objects:
            try:
                file_name = os.path.basename(
                    getattr(
                        file_obj, "name", getattr(file_obj, "orig_name", "upload.pdf")
                    )
                )

                pages = self.load_pdf_documents(file_obj.name, file_name)
                if not pages:
                    logger.warning(f"No content extracted from {file_name}")
                    continue

                chunks = self.chunk_documents(pages)
                all_chunks.extend(chunks)
                processed_files.append(file_name)

            except Exception as e:
                logger.error(f"Failed to process file: {str(e)}")
                continue

        return all_chunks, processed_files
