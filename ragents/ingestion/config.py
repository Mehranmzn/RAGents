"""Configuration classes for document ingestion."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IngestionConfig(BaseModel):
    """Configuration for document ingestion pipeline."""

    # Processing settings
    max_file_size_mb: int = 100
    supported_formats: List[str] = Field(
        default_factory=lambda: [
            ".txt", ".md", ".pdf", ".docx", ".xlsx", ".csv", ".parquet",
            ".json", ".html", ".xml", ".png", ".jpg", ".jpeg", ".gif", ".bmp"
        ]
    )

    # Quality settings
    min_content_length: int = 10
    max_content_length: int = 1000000
    enable_content_validation: bool = True
    enable_metadata_extraction: bool = True

    # Processing options
    enable_ocr: bool = True
    ocr_language: str = "en"
    extract_tables: bool = True
    extract_images: bool = True
    analyze_structure: bool = True

    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    preserve_structure: bool = True

    # Parallel processing
    max_concurrent_files: int = 5
    batch_size: int = 10
    enable_progress_tracking: bool = True

    # Output settings
    generate_summaries: bool = False
    extract_keywords: bool = False
    detect_language: bool = True

    # Error handling
    skip_errors: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


class IngestionResult(BaseModel):
    """Result of document ingestion."""

    file_path: str
    document: Optional[Any] = None  # Will be Document from rag.types
    success: bool
    error_message: Optional[str] = None
    processing_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    validation_result: Optional[Any] = None  # Will be ValidationResult
    chunks_created: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)