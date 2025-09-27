"""Main ingestion pipeline for processing multiple document formats."""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

from pydantic import BaseModel, Field

from .config import IngestionConfig, IngestionResult
from .processors import DocumentProcessor, get_processor_for_file
from .validators import DocumentValidator, ValidationResult
from .extractors import MetadataExtractor, ContentAnalyzer
from .monitoring import IngestionMonitor, IngestionStats
from ..rag.types import Document





class IngestionPipeline:
    """Main document ingestion pipeline."""

    def __init__(
        self,
        config: IngestionConfig = IngestionConfig(),
        validator: Optional[DocumentValidator] = None,
        monitor: Optional[IngestionMonitor] = None
    ):
        self.config = config
        self.validator = validator or DocumentValidator()
        self.monitor = monitor or IngestionMonitor()
        self.metadata_extractor = MetadataExtractor()
        self.content_analyzer = ContentAnalyzer()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Processing history
        self.processing_history: List[IngestionResult] = []

    async def ingest_file(self, file_path: Union[str, Path]) -> IngestionResult:
        """Ingest a single file."""
        start_time = datetime.now()
        file_path = Path(file_path)

        try:
            # Validate file
            if self.config.enable_content_validation:
                validation_result = await self.validator.validate_file(file_path)
                if not validation_result.is_valid:
                    return IngestionResult(
                        file_path=str(file_path),
                        success=False,
                        error_message=f"Validation failed: {validation_result.error_message}",
                        processing_time=0.0,
                        validation_result=validation_result
                    )
            else:
                validation_result = None

            # Get appropriate processor
            processor = get_processor_for_file(file_path)
            if not processor:
                return IngestionResult(
                    file_path=str(file_path),
                    success=False,
                    error_message=f"No processor available for file type: {file_path.suffix}",
                    processing_time=0.0
                )

            # Process document
            document = await processor.process(file_path, self.config)

            # Extract metadata if enabled
            if self.config.enable_metadata_extraction:
                metadata = await self.metadata_extractor.extract(file_path, document)
                document.metadata.update(metadata)

            # Analyze content
            analysis = await self.content_analyzer.analyze(document)
            document.metadata.update(analysis)

            # Create chunks if needed
            chunks_created = 0
            if hasattr(processor, 'create_chunks'):
                chunks = await processor.create_chunks(document, self.config)
                chunks_created = len(chunks)

            processing_time = (datetime.now() - start_time).total_seconds()

            result = IngestionResult(
                file_path=str(file_path),
                document=document,
                success=True,
                processing_time=processing_time,
                validation_result=validation_result,
                chunks_created=chunks_created,
                metadata={
                    "file_size": file_path.stat().st_size,
                    "processor_type": processor.__class__.__name__,
                    "content_length": len(document.content)
                }
            )

            # Update monitoring
            self.monitor.record_success(result)
            self.processing_history.append(result)

            return result

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_result = IngestionResult(
                file_path=str(file_path),
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )

            self.monitor.record_error(error_result)
            self.processing_history.append(error_result)

            if not self.config.skip_errors:
                raise e

            return error_result

    async def ingest_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        pattern: str = "*"
    ) -> List[IngestionResult]:
        """Ingest all files in a directory."""
        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")

        # Find all files
        if recursive:
            files = list(directory_path.rglob(pattern))
        else:
            files = list(directory_path.glob(pattern))

        # Filter by supported formats
        supported_files = [
            f for f in files
            if f.is_file() and f.suffix.lower() in self.config.supported_formats
        ]

        self.logger.info(f"Found {len(supported_files)} supported files in {directory_path}")

        # Process files in batches
        results = []
        for i in range(0, len(supported_files), self.config.batch_size):
            batch = supported_files[i:i + self.config.batch_size]
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)

        return results

    async def ingest_multiple_paths(
        self,
        paths: List[Union[str, Path]],
        max_concurrent: Optional[int] = None
    ) -> List[IngestionResult]:
        """Ingest multiple files or directories."""
        max_concurrent = max_concurrent or self.config.max_concurrent_files

        # Separate files and directories
        files = []
        directories = []

        for path in paths:
            path = Path(path)
            if path.is_file():
                files.append(path)
            elif path.is_dir():
                directories.append(path)
            else:
                self.logger.warning(f"Path does not exist: {path}")

        results = []

        # Process directories first
        for directory in directories:
            dir_results = await self.ingest_directory(directory)
            results.extend(dir_results)

        # Process individual files
        if files:
            file_results = await self._process_batch(files, max_concurrent)
            results.extend(file_results)

        return results

    async def _process_batch(
        self,
        files: List[Path],
        max_concurrent: Optional[int] = None
    ) -> List[IngestionResult]:
        """Process a batch of files concurrently."""
        max_concurrent = max_concurrent or self.config.max_concurrent_files

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(file_path: Path) -> IngestionResult:
            async with semaphore:
                return await self.ingest_file(file_path)

        # Create tasks
        tasks = [process_with_semaphore(file_path) for file_path in files]

        # Process with progress tracking if enabled
        if self.config.enable_progress_tracking:
            results = []
            for i, task in enumerate(asyncio.as_completed(tasks), 1):
                result = await task
                self.logger.info(f"Processed {i}/{len(tasks)}: {result.file_path}")
                results.append(result)
            return results
        else:
            return await asyncio.gather(*tasks)

    async def ingest_from_config(self, config_file: Union[str, Path]) -> List[IngestionResult]:
        """Ingest files based on configuration file."""
        import json

        config_path = Path(config_file)
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        paths = config_data.get('paths', [])
        if not paths:
            raise ValueError("No paths specified in configuration file")

        return await self.ingest_multiple_paths(paths)

    def get_stats(self) -> IngestionStats:
        """Get ingestion statistics."""
        return self.monitor.get_stats()

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing results."""
        if not self.processing_history:
            return {"total_files": 0}

        successful = [r for r in self.processing_history if r.success]
        failed = [r for r in self.processing_history if not r.success]

        return {
            "total_files": len(self.processing_history),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.processing_history) if self.processing_history else 0,
            "avg_processing_time": sum(r.processing_time for r in successful) / len(successful) if successful else 0,
            "total_chunks_created": sum(r.chunks_created for r in successful),
            "file_types_processed": list(set(
                Path(r.file_path).suffix.lower() for r in successful
            )),
            "errors": [{"file": r.file_path, "error": r.error_message} for r in failed]
        }

    async def cleanup_failed_ingestions(self) -> int:
        """Clean up any temporary files from failed ingestions."""
        cleanup_count = 0

        for result in self.processing_history:
            if not result.success and result.document:
                try:
                    # Clean up any temporary files
                    temp_dir = Path("/tmp/ragents_ingestion")
                    if temp_dir.exists():
                        for temp_file in temp_dir.glob(f"*{Path(result.file_path).stem}*"):
                            temp_file.unlink()
                            cleanup_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup temp files: {e}")

        return cleanup_count

    def reset_history(self):
        """Reset processing history."""
        self.processing_history.clear()
        self.monitor.reset()

    async def validate_pipeline_health(self) -> Dict[str, Any]:
        """Validate that the ingestion pipeline is healthy."""
        health_status = {
            "status": "healthy",
            "issues": [],
            "recommendations": []
        }

        # Check configuration
        if self.config.max_file_size_mb > 500:
            health_status["issues"].append("Very large max file size may cause memory issues")
            health_status["recommendations"].append("Consider reducing max_file_size_mb")

        # Check processing history
        if self.processing_history:
            recent_results = self.processing_history[-10:]
            failure_rate = sum(1 for r in recent_results if not r.success) / len(recent_results)

            if failure_rate > 0.5:
                health_status["status"] = "degraded"
                health_status["issues"].append(f"High failure rate: {failure_rate:.1%}")
                health_status["recommendations"].append("Check file formats and validation settings")

        # Check resource usage
        stats = self.get_stats()
        if hasattr(stats, 'avg_processing_time') and stats.avg_processing_time > 10.0:
            health_status["issues"].append("Slow processing times detected")
            health_status["recommendations"].append("Consider optimizing processing settings")

        return health_status