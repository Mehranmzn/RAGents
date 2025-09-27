"""Metrics collection for RAG operations."""

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class MetricValue(BaseModel):
    """A single metric value with timestamp."""
    value: float
    timestamp: float
    labels: Dict[str, str] = {}


class MetricSeries(BaseModel):
    """A series of metric values."""
    name: str
    values: List[MetricValue] = []
    metric_type: str = "gauge"  # gauge, counter, histogram

    def add_value(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Add a value to the metric series."""
        self.values.append(MetricValue(
            value=value,
            timestamp=time.time(),
            labels=labels or {}
        ))

    def get_latest(self) -> Optional[MetricValue]:
        """Get the latest metric value."""
        return self.values[-1] if self.values else None

    def get_average(self, window_seconds: Optional[float] = None) -> float:
        """Get average value, optionally within a time window."""
        if not self.values:
            return 0.0

        if window_seconds:
            cutoff_time = time.time() - window_seconds
            relevant_values = [v.value for v in self.values if v.timestamp >= cutoff_time]
        else:
            relevant_values = [v.value for v in self.values]

        return sum(relevant_values) / len(relevant_values) if relevant_values else 0.0


class MetricsCollector:
    """Collector for RAG operation metrics."""

    def __init__(self):
        self.metrics: Dict[str, MetricSeries] = {}
        self.start_time = time.time()

    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a gauge metric."""
        if name not in self.metrics:
            self.metrics[name] = MetricSeries(name=name, metric_type="gauge")
        self.metrics[name].add_value(value, labels)

    def record_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Record a counter metric."""
        if name not in self.metrics:
            self.metrics[name] = MetricSeries(name=name, metric_type="counter")

        # For counters, add to the last value
        last_value = 0.0
        if self.metrics[name].values:
            last_value = self.metrics[name].values[-1].value

        self.metrics[name].add_value(last_value + value, labels)

    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        if name not in self.metrics:
            self.metrics[name] = MetricSeries(name=name, metric_type="histogram")
        self.metrics[name].add_value(value, labels)

    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Increment a counter by 1."""
        self.record_counter(name, 1.0, labels)

    def time_operation(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager to time an operation."""
        class Timer:
            def __init__(self, collector, metric_name, metric_labels):
                self.collector = collector
                self.metric_name = metric_name
                self.metric_labels = metric_labels
                self.start_time = None

            def __enter__(self):
                self.start_time = time.time()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.collector.record_histogram(self.metric_name, duration, self.metric_labels)

        return Timer(self, name, labels)

    def get_metric(self, name: str) -> Optional[MetricSeries]:
        """Get a metric series by name."""
        return self.metrics.get(name)

    def get_all_metrics(self) -> Dict[str, MetricSeries]:
        """Get all metrics."""
        return self.metrics.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        summary = {
            "uptime_seconds": time.time() - self.start_time,
            "metrics_count": len(self.metrics),
            "metrics": {}
        }

        for name, series in self.metrics.items():
            latest = series.get_latest()
            summary["metrics"][name] = {
                "type": series.metric_type,
                "latest_value": latest.value if latest else None,
                "latest_timestamp": latest.timestamp if latest else None,
                "total_samples": len(series.values),
                "average_1m": series.get_average(60),
                "average_5m": series.get_average(300),
                "average_1h": series.get_average(3600),
            }

        return summary

    # Pre-defined RAG metrics
    def record_llm_request(self, model: str, duration: float, tokens: int, success: bool = True):
        """Record LLM request metrics."""
        labels = {"model": model, "status": "success" if success else "error"}
        self.record_histogram("llm_request_duration_seconds", duration, labels)
        self.record_histogram("llm_request_tokens", tokens, labels)
        self.increment_counter("llm_requests_total", labels)

    def record_vector_search(self, collection: str, duration: float, results_count: int):
        """Record vector search metrics."""
        labels = {"collection": collection}
        self.record_histogram("vector_search_duration_seconds", duration, labels)
        self.record_histogram("vector_search_results_count", results_count, labels)
        self.increment_counter("vector_searches_total", labels)

    def record_rag_query(self, duration: float, sources_count: int, confidence: float):
        """Record RAG query metrics."""
        self.record_histogram("rag_query_duration_seconds", duration)
        self.record_histogram("rag_query_sources_count", sources_count)
        self.record_histogram("rag_query_confidence", confidence)
        self.increment_counter("rag_queries_total")

    def record_document_processing(self, doc_type: str, duration: float, chunks_count: int):
        """Record document processing metrics."""
        labels = {"document_type": doc_type}
        self.record_histogram("document_processing_duration_seconds", duration, labels)
        self.record_histogram("document_chunks_created", chunks_count, labels)
        self.increment_counter("documents_processed_total", labels)

    def record_agent_decision(self, agent_type: str, duration: float, confidence: float):
        """Record agent decision metrics."""
        labels = {"agent_type": agent_type}
        self.record_histogram("agent_decision_duration_seconds", duration, labels)
        self.record_histogram("agent_decision_confidence", confidence, labels)
        self.increment_counter("agent_decisions_total", labels)

    def record_tool_execution(self, tool_name: str, duration: float, success: bool):
        """Record tool execution metrics."""
        labels = {"tool": tool_name, "status": "success" if success else "error"}
        self.record_histogram("tool_execution_duration_seconds", duration, labels)
        self.increment_counter("tool_executions_total", labels)

    def record_evaluation_metric(self, metric_name: str, score: float):
        """Record evaluation metrics."""
        labels = {"metric": metric_name}
        self.record_histogram("evaluation_scores", score, labels)
        self.increment_counter("evaluations_total", labels)

    # System metrics
    def record_memory_usage(self, usage_mb: float):
        """Record memory usage."""
        self.record_gauge("memory_usage_mb", usage_mb)

    def record_cpu_usage(self, usage_percent: float):
        """Record CPU usage."""
        self.record_gauge("cpu_usage_percent", usage_percent)

    def record_active_connections(self, count: int):
        """Record active connections."""
        self.record_gauge("active_connections", count)

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        for name, series in self.metrics.items():
            # Add metric help and type
            lines.append(f"# HELP {name} {name} metric")
            lines.append(f"# TYPE {name} {series.metric_type}")

            # Add values
            latest = series.get_latest()
            if latest:
                labels_str = ""
                if latest.labels:
                    label_pairs = [f'{k}="{v}"' for k, v in latest.labels.items()]
                    labels_str = "{" + ",".join(label_pairs) + "}"

                lines.append(f"{name}{labels_str} {latest.value} {int(latest.timestamp * 1000)}")

        return "\n".join(lines)

    def clear_metrics(self):
        """Clear all metrics."""
        self.metrics.clear()

    def on_span_finish(self, span):
        """Handle span completion for metrics collection."""
        if span.duration:
            # Record operation duration
            self.record_histogram(
                f"{span.span_type.value}_duration_seconds",
                span.duration,
                {"operation": span.operation_name}
            )

            # Increment operation counter
            self.increment_counter(
                f"{span.span_type.value}_operations_total",
                {"operation": span.operation_name, "status": span.status.value}
            )

    def on_trace_finish(self, trace):
        """Handle trace completion for metrics collection."""
        if trace.duration:
            self.record_histogram("trace_duration_seconds", trace.duration)
            self.record_histogram("trace_spans_count", len(trace.spans))
            self.increment_counter("traces_total")


# Global metrics collector
_global_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    return _global_collector