"""
Document loader and processor for RAG Agent
"""
from pathlib import Path
from typing import List

from langchain_community.document_loaders.pdf import PyPDFLoader as PDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.logger import logger
from src.config import settings


class DocumentProcessor:
    """Process and split documents for RAG"""

    def __init__(
        self,
        chunk_size: int = settings.max_chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def load_documents(self, documents_path: str = None) -> List[Document]:
        """
        Load documents from directory
        
        Args:
            documents_path: Path to documents directory
            
        Returns:
            List of loaded documents
        """
        if documents_path is None:
            documents_path = str(settings.documents_dir)

        docs_path = Path(documents_path)
        if not docs_path.exists():
            logger.warning(f"Documents directory not found: {documents_path}")
            return []

        documents = []

        # Load PDF files
        pdf_loader = DirectoryLoader(
            str(docs_path),
            glob="**/*.pdf",
            loader_cls=PDFLoader,
        )
        try:
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
            logger.info(f"Loaded {len(pdf_docs)} PDF documents")
        except Exception as e:
            logger.warning(f"Error loading PDF documents: {e}")

        # Load TXT files
        txt_loader = DirectoryLoader(
            str(docs_path),
            glob="**/*.txt",
            loader_cls=TextLoader,
        )
        try:
            txt_docs = txt_loader.load()
            documents.extend(txt_docs)
            logger.info(f"Loaded {len(txt_docs)} TXT documents")
        except Exception as e:
            logger.warning(f"Error loading TXT documents: {e}")

        if not documents:
            logger.warning("No documents found to load")

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of split documents
        """
        if not documents:
            logger.warning("No documents to split")
            return []

        split_docs = self.text_splitter.split_documents(documents)

        for i, chunk in enumerate(split_docs):
            source = chunk.metadata.get("source", "")
            if source:
                chunk.metadata["filename"] = Path(source).name
            chunk.metadata["chunk_index"] = i

        logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")

        return split_docs

    def process_documents(self, documents_path: str = None) -> List[Document]:
        """
        Load and split documents in one step
        
        Args:
            documents_path: Path to documents directory
            
        Returns:
            List of processed documents
        """
        logger.info(f"Processing documents from {documents_path or settings.documents_dir}")
        
        documents = self.load_documents(documents_path)
        split_docs = self.split_documents(documents)
        
        logger.info(f"Document processing complete: {len(split_docs)} chunks created")
        return split_docs
