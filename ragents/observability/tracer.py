"""Distributed tracing for RAG operations."""

import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SpanType(str, Enum):
    """Types of spans in RAG operations."""
    AGENT = "agent"
    RAG_QUERY = "rag_query"
    LLM_CALL = "llm_call"
    VECTOR_SEARCH = "vector_search"
    DOCUMENT_PROCESSING = "document_processing"
    TOOL_EXECUTION = "tool_execution"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    EVALUATION = "evaluation"


class SpanStatus(str, Enum):
    """Status of a span."""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"


class Span(BaseModel):
    """A trace span representing an operation."""
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str
    parent_span_id: Optional[str] = None
    operation_name: str
    span_type: SpanType
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: SpanStatus = SpanStatus.OK
    tags: Dict[str, Any] = Field(default_factory=dict)
    logs: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None

    def finish(self, status: SpanStatus = SpanStatus.OK, error: Optional[str] = None):
        """Finish the span."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = status
        if error:
            self.error = error

    def add_tag(self, key: str, value: Any):
        """Add a tag to the span."""
        self.tags[key] = value

    def add_log(self, message: str, level: str = "info", **kwargs):
        """Add a log entry to the span."""
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)


class Trace(BaseModel):
    """A complete trace containing multiple spans."""
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    spans: List[Span] = Field(default_factory=list)
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    root_operation: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_span(self, span: Span):
        """Add a span to the trace."""
        span.trace_id = self.trace_id
        self.spans.append(span)

    def finish(self):
        """Finish the trace."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

    def get_root_span(self) -> Optional[Span]:
        """Get the root span of the trace."""
        for span in self.spans:
            if span.parent_span_id is None:
                return span
        return None

    def get_span_tree(self) -> Dict[str, List[Span]]:
        """Get spans organized by parent-child relationships."""
        tree = {}
        for span in self.spans:
            parent_id = span.parent_span_id or "root"
            if parent_id not in tree:
                tree[parent_id] = []
            tree[parent_id].append(span)
        return tree


class RAGTracer:
    """Main tracer for RAG operations with OpenInference compatibility."""

    def __init__(self):
        self.current_trace: Optional[Trace] = None
        self.current_span: Optional[Span] = None
        self.traces: List[Trace] = []
        self.enabled = True
        self.collectors = []

    def start_trace(self, operation_name: str, **metadata) -> Trace:
        """Start a new trace."""
        if not self.enabled:
            return None

        trace = Trace(
            start_time=time.time(),
            root_operation=operation_name,
            metadata=metadata
        )
        self.current_trace = trace
        self.traces.append(trace)
        return trace

    def start_span(
        self,
        operation_name: str,
        span_type: SpanType,
        parent_span: Optional[Span] = None,
        **tags
    ) -> Optional[Span]:
        """Start a new span."""
        if not self.enabled or not self.current_trace:
            return None

        parent_id = None
        if parent_span:
            parent_id = parent_span.span_id
        elif self.current_span:
            parent_id = self.current_span.span_id

        span = Span(
            trace_id=self.current_trace.trace_id,
            parent_span_id=parent_id,
            operation_name=operation_name,
            span_type=span_type,
            start_time=time.time(),
            tags=tags
        )

        self.current_trace.add_span(span)
        self.current_span = span
        return span

    def finish_span(self, span: Optional[Span] = None, status: SpanStatus = SpanStatus.OK, error: Optional[str] = None):
        """Finish a span."""
        if not self.enabled:
            return

        target_span = span or self.current_span
        if target_span:
            target_span.finish(status, error)

            # Notify collectors
            for collector in self.collectors:
                collector.on_span_finish(target_span)

            # Update current span to parent
            if target_span == self.current_span:
                parent_span = self._find_parent_span(target_span)
                self.current_span = parent_span

    def finish_trace(self):
        """Finish the current trace."""
        if not self.enabled or not self.current_trace:
            return

        self.current_trace.finish()

        # Notify collectors
        for collector in self.collectors:
            collector.on_trace_finish(self.current_trace)

        self.current_trace = None
        self.current_span = None

    def _find_parent_span(self, span: Span) -> Optional[Span]:
        """Find the parent span of a given span."""
        if not span.parent_span_id or not self.current_trace:
            return None

        for s in self.current_trace.spans:
            if s.span_id == span.parent_span_id:
                return s
        return None

    def add_collector(self, collector):
        """Add a trace collector."""
        self.collectors.append(collector)

    def enable(self):
        """Enable tracing."""
        self.enabled = True

    def disable(self):
        """Disable tracing."""
        self.enabled = False

    @contextmanager
    def trace(self, operation_name: str, **metadata):
        """Context manager for tracing an operation."""
        trace = self.start_trace(operation_name, **metadata)
        try:
            yield trace
        finally:
            self.finish_trace()

    @contextmanager
    def span(self, operation_name: str, span_type: SpanType, **tags):
        """Context manager for creating a span."""
        span = self.start_span(operation_name, span_type, **tags)
        try:
            yield span
        except Exception as e:
            if span:
                span.add_log(f"Error: {str(e)}", level="error")
            self.finish_span(span, SpanStatus.ERROR, str(e))
            raise
        else:
            self.finish_span(span, SpanStatus.OK)

    @asynccontextmanager
    async def async_span(self, operation_name: str, span_type: SpanType, **tags):
        """Async context manager for creating a span."""
        span = self.start_span(operation_name, span_type, **tags)
        try:
            yield span
        except Exception as e:
            if span:
                span.add_log(f"Error: {str(e)}", level="error")
            self.finish_span(span, SpanStatus.ERROR, str(e))
            raise
        else:
            self.finish_span(span, SpanStatus.OK)

    def get_trace_summary(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a trace."""
        trace = None
        for t in self.traces:
            if t.trace_id == trace_id:
                trace = t
                break

        if not trace:
            return None

        summary = {
            "trace_id": trace.trace_id,
            "duration": trace.duration,
            "span_count": len(trace.spans),
            "error_count": sum(1 for span in trace.spans if span.status == SpanStatus.ERROR),
            "operation_counts": {},
            "avg_duration_by_type": {}
        }

        # Count operations and calculate averages
        type_durations = {}
        for span in trace.spans:
            op_type = span.span_type.value
            summary["operation_counts"][op_type] = summary["operation_counts"].get(op_type, 0) + 1

            if span.duration:
                if op_type not in type_durations:
                    type_durations[op_type] = []
                type_durations[op_type].append(span.duration)

        for op_type, durations in type_durations.items():
            summary["avg_duration_by_type"][op_type] = sum(durations) / len(durations)

        return summary

    def clear_traces(self):
        """Clear all stored traces."""
        self.traces.clear()

    def export_traces(self, format: str = "json") -> str:
        """Export traces in specified format."""
        if format == "json":
            import json
            return json.dumps([trace.model_dump() for trace in self.traces], indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global tracer instance
_global_tracer = RAGTracer()


def get_tracer() -> RAGTracer:
    """Get the global tracer instance."""
    return _global_tracer


def trace_operation(operation_name: str, span_type: SpanType, **tags):
    """Decorator for tracing operations."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                tracer = get_tracer()
                async with tracer.async_span(operation_name, span_type, **tags) as span:
                    if span:
                        span.add_tag("function", func.__name__)
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                tracer = get_tracer()
                with tracer.span(operation_name, span_type, **tags) as span:
                    if span:
                        span.add_tag("function", func.__name__)
                    return func(*args, **kwargs)
            return sync_wrapper
    return decorator


# Import asyncio here to avoid circular imports
import asyncio