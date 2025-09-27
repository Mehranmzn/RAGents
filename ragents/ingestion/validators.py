"""Document validation utilities for the ingestion pipeline."""

import os
import mimetypes
from pathlib import Path
from typing import List, Optional, Set
from dataclasses import dataclass
from datetime import datetime

from ..rag.types import Document


@dataclass
class ValidationResult:
    """Result of document validation."""
    is_valid: bool
    error_message: Optional[str] = None
    warnings: List[str] = None
    file_size: int = 0
    mime_type: Optional[str] = None
    validation_time: float = 0.0

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class DocumentValidator:
    """Validator for documents before processing."""

    def __init__(
        self,
        max_file_size_mb: int = 100,
        allowed_mime_types: Optional[Set[str]] = None,
        blocked_extensions: Optional[Set[str]] = None
    ):
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.allowed_mime_types = allowed_mime_types or {
            'text/plain', 'text/markdown', 'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.ms-excel', 'text/csv', 'application/json',
            'image/png', 'image/jpeg', 'image/gif', 'image/bmp',
            'application/octet-stream'  # For parquet files
        }
        self.blocked_extensions = blocked_extensions or {
            '.exe', '.bat', '.cmd', '.scr', '.com', '.pif', '.vbs', '.js'
        }

    async def validate_file(self, file_path: Path) -> ValidationResult:
        """Validate a single file."""
        start_time = datetime.now()

        try:
            # Check if file exists
            if not file_path.exists():
                return ValidationResult(
                    is_valid=False,
                    error_message=f"File does not exist: {file_path}",
                    validation_time=(datetime.now() - start_time).total_seconds()
                )

            # Check if it's a file (not directory)
            if not file_path.is_file():
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Path is not a file: {file_path}",
                    validation_time=(datetime.now() - start_time).total_seconds()
                )

            # Get file info
            file_stat = file_path.stat()
            file_size = file_stat.st_size
            mime_type, _ = mimetypes.guess_type(str(file_path))

            warnings = []

            # Check file size
            if file_size > self.max_file_size_bytes:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"File too large: {file_size / (1024*1024):.1f}MB exceeds limit of {self.max_file_size_bytes / (1024*1024):.1f}MB",
                    file_size=file_size,
                    mime_type=mime_type,
                    validation_time=(datetime.now() - start_time).total_seconds()
                )

            # Check file extension
            file_extension = file_path.suffix.lower()
            if file_extension in self.blocked_extensions:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"File extension blocked for security: {file_extension}",
                    file_size=file_size,
                    mime_type=mime_type,
                    validation_time=(datetime.now() - start_time).total_seconds()
                )

            # Check MIME type
            if mime_type and mime_type not in self.allowed_mime_types:
                warnings.append(f"MIME type {mime_type} not in allowed list")

            # Check for empty files
            if file_size == 0:
                warnings.append("File is empty")

            # Check file permissions
            if not os.access(file_path, os.R_OK):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"File not readable: {file_path}",
                    file_size=file_size,
                    mime_type=mime_type,
                    validation_time=(datetime.now() - start_time).total_seconds()
                )

            # Additional checks for specific file types
            if file_extension == '.pdf':
                pdf_validation = await self._validate_pdf(file_path)
                if not pdf_validation.is_valid:
                    return pdf_validation
                warnings.extend(pdf_validation.warnings)

            elif file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                image_validation = await self._validate_image(file_path)
                if not image_validation.is_valid:
                    return image_validation
                warnings.extend(image_validation.warnings)

            return ValidationResult(
                is_valid=True,
                warnings=warnings,
                file_size=file_size,
                mime_type=mime_type,
                validation_time=(datetime.now() - start_time).total_seconds()
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation error: {str(e)}",
                validation_time=(datetime.now() - start_time).total_seconds()
            )

    async def _validate_pdf(self, file_path: Path) -> ValidationResult:
        """Validate PDF-specific requirements."""
        try:
            # Check if PDF is password protected
            try:
                import pymupdf as fitz
            except ImportError:
                try:
                    import fitz
                except ImportError:
                    return ValidationResult(
                        is_valid=True,
                        warnings=["PyMuPDF not available for PDF validation"]
                    )

            doc = fitz.open(file_path)

            warnings = []

            # Check if encrypted
            if doc.needs_pass:
                doc.close()
                return ValidationResult(
                    is_valid=False,
                    error_message="PDF is password protected"
                )

            # Check page count
            page_count = len(doc)
            if page_count > 1000:
                warnings.append(f"PDF has many pages ({page_count}), processing may be slow")

            # Check for corrupt pages
            corrupt_pages = []
            for page_num in range(min(10, page_count)):  # Check first 10 pages
                try:
                    page = doc[page_num]
                    page.get_text()
                except Exception:
                    corrupt_pages.append(page_num + 1)

            if corrupt_pages:
                warnings.append(f"Potentially corrupt pages: {corrupt_pages}")

            doc.close()

            return ValidationResult(
                is_valid=True,
                warnings=warnings
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"PDF validation error: {str(e)}"
            )

    async def _validate_image(self, file_path: Path) -> ValidationResult:
        """Validate image-specific requirements."""
        try:
            from PIL import Image
        except ImportError:
            return ValidationResult(
                is_valid=True,
                warnings=["Pillow not available for image validation"]
            )

        try:
            warnings = []

            with Image.open(file_path) as img:
                width, height = img.size

                # Check image dimensions
                if width > 10000 or height > 10000:
                    warnings.append(f"Very large image dimensions: {width}x{height}")

                # Check for extremely small images
                if width < 10 or height < 10:
                    warnings.append(f"Very small image dimensions: {width}x{height}")

                # Verify image can be loaded
                img.verify()

            return ValidationResult(
                is_valid=True,
                warnings=warnings
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Image validation error: {str(e)}"
            )

    async def validate_content(self, document: Document) -> ValidationResult:
        """Validate document content after processing."""
        start_time = datetime.now()

        try:
            warnings = []

            # Check content length
            content_length = len(document.content)
            if content_length < 10:
                warnings.append("Very short content extracted")
            elif content_length > 1000000:  # 1MB of text
                warnings.append("Very long content extracted")

            # Check for suspicious content patterns
            suspicious_patterns = [
                'javascript:', 'data:', 'vbscript:', '<script', 'eval(',
                'document.cookie', 'window.location'
            ]

            for pattern in suspicious_patterns:
                if pattern.lower() in document.content.lower():
                    warnings.append(f"Potentially suspicious content pattern found: {pattern}")

            # Check character encoding issues
            if '�' in document.content:
                warnings.append("Possible character encoding issues detected")

            # Check for binary data in text content
            try:
                document.content.encode('utf-8')
            except UnicodeEncodeError:
                warnings.append("Content contains non-UTF-8 characters")

            return ValidationResult(
                is_valid=True,
                warnings=warnings,
                validation_time=(datetime.now() - start_time).total_seconds()
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Content validation error: {str(e)}",
                validation_time=(datetime.now() - start_time).total_seconds()
            )

    def update_settings(
        self,
        max_file_size_mb: Optional[int] = None,
        allowed_mime_types: Optional[Set[str]] = None,
        blocked_extensions: Optional[Set[str]] = None
    ):
        """Update validation settings."""
        if max_file_size_mb is not None:
            self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

        if allowed_mime_types is not None:
            self.allowed_mime_types = allowed_mime_types

        if blocked_extensions is not None:
            self.blocked_extensions = blocked_extensions

    def get_validation_summary(self, results: List[ValidationResult]) -> dict:
        """Get summary of validation results."""
        if not results:
            return {"total": 0}

        valid_count = sum(1 for r in results if r.is_valid)
        total_size = sum(r.file_size for r in results if r.file_size > 0)
        total_warnings = sum(len(r.warnings) for r in results)

        return {
            "total": len(results),
            "valid": valid_count,
            "invalid": len(results) - valid_count,
            "success_rate": valid_count / len(results),
            "total_size_mb": total_size / (1024 * 1024),
            "total_warnings": total_warnings,
            "avg_validation_time": sum(r.validation_time for r in results) / len(results),
            "common_errors": self._get_common_errors(results),
            "common_warnings": self._get_common_warnings(results)
        }

    def _get_common_errors(self, results: List[ValidationResult]) -> dict:
        """Get most common error messages."""
        error_counts = {}
        for result in results:
            if not result.is_valid and result.error_message:
                # Generalize error messages
                error_type = self._generalize_error(result.error_message)
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

        return dict(sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5])

    def _get_common_warnings(self, results: List[ValidationResult]) -> dict:
        """Get most common warning messages."""
        warning_counts = {}
        for result in results:
            for warning in result.warnings:
                warning_counts[warning] = warning_counts.get(warning, 0) + 1

        return dict(sorted(warning_counts.items(), key=lambda x: x[1], reverse=True)[:5])

    def _generalize_error(self, error_message: str) -> str:
        """Generalize error message for categorization."""
        if "too large" in error_message.lower():
            return "File too large"
        elif "does not exist" in error_message.lower():
            return "File not found"
        elif "not readable" in error_message.lower():
            return "Permission denied"
        elif "password protected" in error_message.lower():
            return "Password protected"
        elif "extension blocked" in error_message.lower():
            return "Blocked file type"
        else:
            return "Other error"