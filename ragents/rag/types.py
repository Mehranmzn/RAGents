"""Types for RAG system."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported document types."""
    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    CSV = "csv"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    IMAGE = "image"


class ChunkType(str, Enum):
    """Types of content chunks."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    EQUATION = "equation"
    CODE = "code"
    METADATA = "metadata"


class ProcessingStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


class Document(BaseModel):
    """Document representation."""
    id: str
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    doc_type: DocumentType
    file_path: Optional[str] = None
    created_at: float
    updated_at: float
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    file_size: Optional[int] = None
    checksum: Optional[str] = None


class ContentChunk(BaseModel):
    """A chunk of processed content."""
    id: str
    document_id: str
    content: str
    chunk_type: ChunkType = ChunkType.TEXT
    metadata: Dict[str, Any] = Field(default_factory=dict)
    start_index: int = 0
    end_index: int = 0
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    embedding: Optional[List[float]] = None
    created_at: float


class RetrievalResult(BaseModel):
    """Result from retrieval operation."""
    chunk: ContentChunk
    score: float
    rank: int
    retrieval_method: str


class QueryContext(BaseModel):
    """Context for query processing."""
    original_query: str
    expanded_queries: List[str] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    retrieval_strategy: str = "similarity"
    top_k: int = 5
    similarity_threshold: float = 0.7
    include_metadata: bool = True


class RAGResponse(BaseModel):
    """Response from RAG system."""
    query: str
    answer: str
    sources: List[RetrievalResult]
    context_chunks: List[ContentChunk]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    processing_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)