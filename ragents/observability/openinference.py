"""OpenInference integration for RAG tracing."""

import json
from typing import Any, Dict, List, Optional

from .tracer import RAGTracer, Span, SpanType, Trace


class OpenInferenceIntegration:
    """Integration with OpenInference tracing standards."""

    def __init__(self, tracer: RAGTracer):
        self.tracer = tracer
        self.tracer.add_collector(self)

    def on_span_finish(self, span: Span):
        """Handle span completion for OpenInference."""
        # Convert to OpenInference format
        otel_span = self._convert_to_otel_format(span)
        self._export_span(otel_span)

    def on_trace_finish(self, trace: Trace):
        """Handle trace completion."""
        # Export complete trace
        otel_trace = self._convert_trace_to_otel_format(trace)
        self._export_trace(otel_trace)

    def _convert_to_otel_format(self, span: Span) -> Dict[str, Any]:
        """Convert RAGents span to OpenTelemetry format."""
        otel_span = {
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "parent_span_id": span.parent_span_id,
            "operation_name": span.operation_name,
            "start_time": int(span.start_time * 1_000_000_000),  # Convert to nanoseconds
            "end_time": int(span.end_time * 1_000_000_000) if span.end_time else None,
            "duration": int(span.duration * 1_000_000_000) if span.duration else None,
            "status": {
                "code": "OK" if span.status.value == "ok" else "ERROR",
                "message": span.error or ""
            },
            "attributes": self._create_openinference_attributes(span),
            "events": self._convert_logs_to_events(span.logs)
        }

        return otel_span

    def _create_openinference_attributes(self, span: Span) -> Dict[str, Any]:
        """Create OpenInference-compatible attributes."""
        attributes = {
            # OpenInference standard attributes
            "openinference.span.kind": self._map_span_type_to_openinference(span.span_type),
        }

        # Add span-type specific attributes
        if span.span_type == SpanType.LLM_CALL:
            attributes.update({
                "llm.request.model": span.tags.get("model"),
                "llm.request.temperature": span.tags.get("temperature"),
                "llm.request.max_tokens": span.tags.get("max_tokens"),
                "llm.response.model": span.tags.get("response_model"),
                "llm.usage.prompt_tokens": span.tags.get("prompt_tokens"),
                "llm.usage.completion_tokens": span.tags.get("completion_tokens"),
                "llm.usage.total_tokens": span.tags.get("total_tokens"),
            })

        elif span.span_type == SpanType.RETRIEVAL:
            attributes.update({
                "retrieval.query": span.tags.get("query"),
                "retrieval.top_k": span.tags.get("top_k"),
                "retrieval.similarity_threshold": span.tags.get("similarity_threshold"),
                "retrieval.results_count": span.tags.get("results_count"),
            })

        elif span.span_type == SpanType.VECTOR_SEARCH:
            attributes.update({
                "vector_search.query_vector_dimension": span.tags.get("vector_dimension"),
                "vector_search.collection_name": span.tags.get("collection_name"),
                "vector_search.similarity_metric": span.tags.get("similarity_metric"),
                "vector_search.results_returned": span.tags.get("results_count"),
            })

        elif span.span_type == SpanType.DOCUMENT_PROCESSING:
            attributes.update({
                "document.id": span.tags.get("document_id"),
                "document.type": span.tags.get("document_type"),
                "document.size": span.tags.get("document_size"),
                "document.chunks_created": span.tags.get("chunks_created"),
            })

        # Add custom tags
        for key, value in span.tags.items():
            if key not in attributes:
                attributes[f"custom.{key}"] = value

        # Remove None values
        return {k: v for k, v in attributes.items() if v is not None}

    def _map_span_type_to_openinference(self, span_type: SpanType) -> str:
        """Map RAGents span types to OpenInference span kinds."""
        mapping = {
            SpanType.LLM_CALL: "LLM",
            SpanType.RETRIEVAL: "RETRIEVER",
            SpanType.VECTOR_SEARCH: "RETRIEVER",
            SpanType.DOCUMENT_PROCESSING: "CHAIN",
            SpanType.AGENT: "AGENT",
            SpanType.RAG_QUERY: "CHAIN",
            SpanType.TOOL_EXECUTION: "TOOL",
            SpanType.GENERATION: "LLM",
            SpanType.EVALUATION: "EVALUATOR",
        }
        return mapping.get(span_type, "UNKNOWN")

    def _convert_logs_to_events(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert span logs to OpenTelemetry events."""
        events = []
        for log in logs:
            event = {
                "timestamp": int(log["timestamp"] * 1_000_000_000),
                "name": log.get("level", "info"),
                "attributes": {
                    "message": log.get("message", ""),
                    **{k: v for k, v in log.items() if k not in ["timestamp", "level", "message"]}
                }
            }
            events.append(event)
        return events

    def _convert_trace_to_otel_format(self, trace: Trace) -> Dict[str, Any]:
        """Convert complete trace to OpenTelemetry format."""
        otel_trace = {
            "trace_id": trace.trace_id,
            "spans": [self._convert_to_otel_format(span) for span in trace.spans],
            "start_time": int(trace.start_time * 1_000_000_000),
            "end_time": int(trace.end_time * 1_000_000_000) if trace.end_time else None,
            "duration": int(trace.duration * 1_000_000_000) if trace.duration else None,
            "root_operation": trace.root_operation,
            "attributes": trace.metadata
        }
        return otel_trace

    def _export_span(self, otel_span: Dict[str, Any]):
        """Export span to OpenInference-compatible endpoint."""
        # This would typically send to an OpenInference collector
        # For now, we'll just log it
        print(f"OpenInference Span: {json.dumps(otel_span, indent=2)}")

    def _export_trace(self, otel_trace: Dict[str, Any]):
        """Export complete trace."""
        # This would typically send to an OpenInference collector
        print(f"OpenInference Trace: {json.dumps(otel_trace, indent=2)}")

    def create_rag_session_attributes(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create session-level attributes for RAG operations."""
        attributes = {
            "session.id": session_id,
            "openinference.span.kind": "CHAIN",
        }

        if user_id:
            attributes["user.id"] = user_id
        if conversation_id:
            attributes["conversation.id"] = conversation_id

        return attributes

    def create_llm_attributes(
        self,
        model_name: str,
        prompt: str,
        response: str,
        usage: Optional[Dict[str, int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create LLM-specific attributes."""
        attributes = {
            "llm.request.model": model_name,
            "llm.input_messages": [{"role": "user", "content": prompt}],
            "llm.output_messages": [{"role": "assistant", "content": response}],
            "openinference.span.kind": "LLM",
        }

        if usage:
            attributes.update({
                "llm.usage.prompt_tokens": usage.get("prompt_tokens"),
                "llm.usage.completion_tokens": usage.get("completion_tokens"),
                "llm.usage.total_tokens": usage.get("total_tokens"),
            })

        # Add any additional parameters
        for key, value in kwargs.items():
            attributes[f"llm.request.{key}"] = value

        return attributes

    def create_retrieval_attributes(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Create retrieval-specific attributes."""
        attributes = {
            "retrieval.query": query,
            "retrieval.documents": documents,
            "retrieval.top_k": top_k,
            "openinference.span.kind": "RETRIEVER",
        }

        if similarity_threshold:
            attributes["retrieval.similarity_threshold"] = similarity_threshold

        return attributes

    def instrument_llm_client(self, llm_client):
        """Instrument LLM client for automatic tracing."""
        original_complete = llm_client.acomplete

        async def traced_complete(*args, **kwargs):
            with self.tracer.span("llm_completion", SpanType.LLM_CALL) as span:
                if span:
                    span.add_tag("model", llm_client.config.model_name)
                    span.add_tag("provider", llm_client.config.provider.value)
                    span.add_tag("temperature", llm_client.config.temperature)

                try:
                    result = await original_complete(*args, **kwargs)
                    if span and hasattr(result, 'usage') and result.usage:
                        span.add_tag("prompt_tokens", result.usage.get("prompt_tokens"))
                        span.add_tag("completion_tokens", result.usage.get("completion_tokens"))
                        span.add_tag("total_tokens", result.usage.get("total_tokens"))
                    return result
                except Exception as e:
                    if span:
                        span.add_log(f"LLM error: {str(e)}", level="error")
                    raise

        llm_client.acomplete = traced_complete
        return llm_client

    def instrument_rag_engine(self, rag_engine):
        """Instrument RAG engine for automatic tracing."""
        original_query = rag_engine.query

        async def traced_query(*args, **kwargs):
            with self.tracer.span("rag_query", SpanType.RAG_QUERY) as span:
                if span and args:
                    span.add_tag("query", args[0])

                try:
                    result = await original_query(*args, **kwargs)
                    if span:
                        span.add_tag("answer_length", len(result.answer) if result.answer else 0)
                        span.add_tag("sources_count", len(result.sources))
                        span.add_tag("confidence", result.confidence)
                    return result
                except Exception as e:
                    if span:
                        span.add_log(f"RAG query error: {str(e)}", level="error")
                    raise

        rag_engine.query = traced_query
        return rag_engine


def setup_openinference_tracing(
    tracer: Optional[RAGTracer] = None,
    endpoint: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None
) -> OpenInferenceIntegration:
    """Set up OpenInference tracing integration."""
    if tracer is None:
        from .tracer import get_tracer
        tracer = get_tracer()

    integration = OpenInferenceIntegration(tracer)

    # Configure endpoint if provided
    if endpoint:
        integration.endpoint = endpoint

    if headers:
        integration.headers = headers

    return integration