"""Structured logging for RAG operations with tamper-proof reasoning summaries."""

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ReasoningSummary(BaseModel):
    """Tamper-proof reasoning summary with integrity checking."""
    reasoning_id: str
    timestamp: float
    operation: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    reasoning_steps: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    integrity_hash: Optional[str] = None

    def calculate_hash(self) -> str:
        """Calculate integrity hash for tamper detection."""
        # Create deterministic content for hashing
        content = {
            "reasoning_id": self.reasoning_id,
            "timestamp": self.timestamp,
            "operation": self.operation,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "reasoning_steps": self.reasoning_steps,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

        # Sort keys for deterministic hashing
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def seal(self) -> "ReasoningSummary":
        """Seal the reasoning summary with integrity hash."""
        self.integrity_hash = self.calculate_hash()
        return self

    def verify_integrity(self) -> bool:
        """Verify the integrity of the reasoning summary."""
        if not self.integrity_hash:
            return False
        return self.calculate_hash() == self.integrity_hash


class StructuredLogEntry(BaseModel):
    """Structured log entry with metadata."""
    timestamp: float
    level: str
    message: str
    operation: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    reasoning_summary: Optional[ReasoningSummary] = None


class StructuredLogger:
    """Enhanced structured logger for RAG operations."""

    def __init__(
        self,
        logger_name: str = "ragents",
        level: str = "INFO",
        enable_reasoning_capture: bool = True,
    ):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Configure JSON formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.enable_reasoning_capture = enable_reasoning_capture
        self.reasoning_summaries: List[ReasoningSummary] = []

    def _create_log_entry(
        self,
        level: str,
        message: str,
        operation: Optional[str] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        reasoning_summary: Optional[ReasoningSummary] = None,
        **metadata
    ) -> StructuredLogEntry:
        """Create a structured log entry."""
        return StructuredLogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            operation=operation,
            trace_id=trace_id,
            span_id=span_id,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            reasoning_summary=reasoning_summary,
        )

    def _log_entry(self, entry: StructuredLogEntry):
        """Log a structured entry."""
        log_data = {
            "timestamp": entry.timestamp,
            "level": entry.level,
            "message": entry.message,
            "operation": entry.operation,
            "trace_id": entry.trace_id,
            "span_id": entry.span_id,
            "user_id": entry.user_id,
            "session_id": entry.session_id,
            "metadata": entry.metadata,
        }

        if entry.reasoning_summary:
            log_data["reasoning_summary"] = entry.reasoning_summary.model_dump()

        # Log as JSON for structured logging systems
        json_message = json.dumps(log_data)

        log_level = getattr(logging, entry.level.upper())
        self.logger.log(log_level, json_message)

    def info(self, message: str, **kwargs):
        """Log info message."""
        entry = self._create_log_entry("INFO", message, **kwargs)
        self._log_entry(entry)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        entry = self._create_log_entry("WARNING", message, **kwargs)
        self._log_entry(entry)

    def error(self, message: str, **kwargs):
        """Log error message."""
        entry = self._create_log_entry("ERROR", message, **kwargs)
        self._log_entry(entry)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        entry = self._create_log_entry("DEBUG", message, **kwargs)
        self._log_entry(entry)

    def log_agent_decision(
        self,
        agent_name: str,
        decision: str,
        reasoning_steps: List[str],
        confidence: float,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        trace_id: Optional[str] = None,
        **metadata
    ):
        """Log agent decision with reasoning summary."""
        if self.enable_reasoning_capture:
            reasoning_summary = ReasoningSummary(
                reasoning_id=f"{agent_name}_{int(time.time() * 1000)}",
                timestamp=time.time(),
                operation=f"agent_decision_{agent_name}",
                inputs=inputs,
                outputs=outputs,
                reasoning_steps=reasoning_steps,
                confidence=confidence,
                metadata=metadata,
            ).seal()

            self.reasoning_summaries.append(reasoning_summary)
        else:
            reasoning_summary = None

        self.info(
            f"Agent {agent_name} decision: {decision}",
            operation="agent_decision",
            trace_id=trace_id,
            agent_name=agent_name,
            decision=decision,
            confidence=confidence,
            reasoning_summary=reasoning_summary,
            **metadata
        )

    def log_llm_interaction(
        self,
        model: str,
        prompt: str,
        response: str,
        usage: Optional[Dict[str, int]] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        **metadata
    ):
        """Log LLM interaction."""
        self.info(
            f"LLM interaction with {model}",
            operation="llm_interaction",
            trace_id=trace_id,
            span_id=span_id,
            model=model,
            prompt_length=len(prompt),
            response_length=len(response),
            usage=usage,
            **metadata
        )

    def log_rag_query(
        self,
        query: str,
        answer: str,
        sources_count: int,
        confidence: float,
        processing_time: float,
        trace_id: Optional[str] = None,
        **metadata
    ):
        """Log RAG query operation."""
        self.info(
            f"RAG query processed: '{query[:100]}...'",
            operation="rag_query",
            trace_id=trace_id,
            query_length=len(query),
            answer_length=len(answer),
            sources_count=sources_count,
            confidence=confidence,
            processing_time=processing_time,
            **metadata
        )

    def log_vector_search(
        self,
        query: str,
        collection: str,
        top_k: int,
        results_count: int,
        search_time: float,
        trace_id: Optional[str] = None,
        **metadata
    ):
        """Log vector search operation."""
        self.info(
            f"Vector search in {collection}: {results_count} results",
            operation="vector_search",
            trace_id=trace_id,
            collection=collection,
            top_k=top_k,
            results_count=results_count,
            search_time=search_time,
            **metadata
        )

    def log_document_processing(
        self,
        document_id: str,
        document_type: str,
        chunks_created: int,
        processing_time: float,
        trace_id: Optional[str] = None,
        **metadata
    ):
        """Log document processing operation."""
        self.info(
            f"Document processed: {document_id} ({document_type})",
            operation="document_processing",
            trace_id=trace_id,
            document_id=document_id,
            document_type=document_type,
            chunks_created=chunks_created,
            processing_time=processing_time,
            **metadata
        )

    def log_evaluation_result(
        self,
        evaluation_type: str,
        score: float,
        details: Dict[str, Any],
        trace_id: Optional[str] = None,
        **metadata
    ):
        """Log evaluation results."""
        self.info(
            f"Evaluation {evaluation_type}: score {score:.3f}",
            operation="evaluation",
            trace_id=trace_id,
            evaluation_type=evaluation_type,
            score=score,
            details=details,
            **metadata
        )

    def log_tool_execution(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        outputs: Any,
        execution_time: float,
        success: bool,
        trace_id: Optional[str] = None,
        **metadata
    ):
        """Log tool execution."""
        level = "INFO" if success else "WARNING"
        message = f"Tool {tool_name} {'succeeded' if success else 'failed'}"

        entry = self._create_log_entry(
            level,
            message,
            operation="tool_execution",
            trace_id=trace_id,
            tool_name=tool_name,
            execution_time=execution_time,
            success=success,
            inputs_keys=list(inputs.keys()),
            **metadata
        )
        self._log_entry(entry)

    def get_reasoning_summaries(
        self,
        operation_filter: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[ReasoningSummary]:
        """Retrieve reasoning summaries with optional filters."""
        filtered_summaries = self.reasoning_summaries

        if operation_filter:
            filtered_summaries = [
                s for s in filtered_summaries
                if operation_filter in s.operation
            ]

        if start_time:
            filtered_summaries = [
                s for s in filtered_summaries
                if s.timestamp >= start_time
            ]

        if end_time:
            filtered_summaries = [
                s for s in filtered_summaries
                if s.timestamp <= end_time
            ]

        return filtered_summaries

    def verify_reasoning_integrity(self) -> Dict[str, Any]:
        """Verify integrity of all reasoning summaries."""
        total_summaries = len(self.reasoning_summaries)
        valid_summaries = 0
        corrupted_summaries = []

        for summary in self.reasoning_summaries:
            if summary.verify_integrity():
                valid_summaries += 1
            else:
                corrupted_summaries.append(summary.reasoning_id)

        return {
            "total_summaries": total_summaries,
            "valid_summaries": valid_summaries,
            "corrupted_summaries": corrupted_summaries,
            "integrity_rate": valid_summaries / total_summaries if total_summaries > 0 else 1.0,
        }

    def export_reasoning_summaries(self, format: str = "json") -> str:
        """Export reasoning summaries in specified format."""
        if format == "json":
            return json.dumps([
                summary.model_dump() for summary in self.reasoning_summaries
            ], indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def clear_reasoning_summaries(self):
        """Clear all stored reasoning summaries."""
        self.reasoning_summaries.clear()


# Global logger instance
_global_logger = StructuredLogger()


def get_logger() -> StructuredLogger:
    """Get the global structured logger instance."""
    return _global_logger


def setup_structured_logging(
    level: str = "INFO",
    enable_reasoning_capture: bool = True,
    logger_name: str = "ragents"
) -> StructuredLogger:
    """Set up structured logging for RAG operations."""
    global _global_logger
    _global_logger = StructuredLogger(
        logger_name=logger_name,
        level=level,
        enable_reasoning_capture=enable_reasoning_capture
    )
    return _global_logger