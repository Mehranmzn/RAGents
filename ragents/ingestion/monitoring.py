"""Monitoring and statistics for ingestion pipeline."""

import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from .config import IngestionResult


@dataclass
class IngestionStats:
    """Statistics for ingestion operations."""
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_size_bytes: int = 0
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    success_rate: float = 0.0
    files_per_second: float = 0.0
    bytes_per_second: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    file_types: Dict[str, int] = field(default_factory=dict)
    error_types: Dict[str, int] = field(default_factory=dict)
    processing_times_by_type: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    hourly_throughput: Dict[str, int] = field(default_factory=dict)


class IngestionMonitor:
    """Monitor ingestion pipeline performance and statistics."""

    def __init__(self):
        self.results: List[IngestionResult] = []
        self.session_start_time = datetime.now()
        self.hourly_stats: Dict[str, int] = defaultdict(int)
        self.error_patterns: Dict[str, int] = defaultdict(int)

    def record_success(self, result: IngestionResult):
        """Record a successful ingestion."""
        self.results.append(result)
        self._update_hourly_stats(result.timestamp)

    def record_error(self, result: IngestionResult):
        """Record a failed ingestion."""
        self.results.append(result)
        self._update_hourly_stats(result.timestamp)

        # Track error patterns
        if result.error_message:
            error_type = self._categorize_error(result.error_message)
            self.error_patterns[error_type] += 1

    def get_stats(self, time_window_hours: Optional[int] = None) -> IngestionStats:
        """Get comprehensive ingestion statistics."""
        results = self.results

        # Filter by time window if specified
        if time_window_hours:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            results = [r for r in results if r.timestamp >= cutoff_time]

        if not results:
            return IngestionStats()

        # Calculate basic stats
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        total_files = len(results)
        successful_files = len(successful)
        failed_files = len(failed)

        total_size = sum(
            r.metadata.get('file_size', 0) for r in successful
            if isinstance(r.metadata, dict)
        )

        total_processing_time = sum(r.processing_time for r in results)
        avg_processing_time = total_processing_time / total_files if total_files > 0 else 0

        success_rate = successful_files / total_files if total_files > 0 else 0

        # Calculate throughput
        start_time = min(r.timestamp for r in results)
        end_time = max(r.timestamp for r in results)
        duration = (end_time - start_time).total_seconds()

        files_per_second = total_files / duration if duration > 0 else 0
        bytes_per_second = total_size / duration if duration > 0 else 0

        # File type distribution
        file_types = defaultdict(int)
        for result in results:
            if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
                file_ext = result.metadata.get('file_extension', 'unknown')
                file_types[file_ext] += 1

        # Error type distribution
        error_types = defaultdict(int)
        for result in failed:
            if result.error_message:
                error_type = self._categorize_error(result.error_message)
                error_types[error_type] += 1

        # Processing times by file type
        processing_times_by_type = defaultdict(list)
        for result in successful:
            if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
                file_ext = result.metadata.get('file_extension', 'unknown')
                processing_times_by_type[file_ext].append(result.processing_time)

        # Hourly throughput
        hourly_throughput = self._calculate_hourly_throughput(results)

        return IngestionStats(
            total_files=total_files,
            successful_files=successful_files,
            failed_files=failed_files,
            total_size_bytes=total_size,
            total_processing_time=total_processing_time,
            avg_processing_time=avg_processing_time,
            success_rate=success_rate,
            files_per_second=files_per_second,
            bytes_per_second=bytes_per_second,
            start_time=start_time,
            end_time=end_time,
            file_types=dict(file_types),
            error_types=dict(error_types),
            processing_times_by_type=dict(processing_times_by_type),
            hourly_throughput=hourly_throughput
        )

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get key performance metrics."""
        stats = self.get_stats()

        return {
            "success_rate": stats.success_rate,
            "avg_processing_time": stats.avg_processing_time,
            "files_per_second": stats.files_per_second,
            "mb_per_second": stats.bytes_per_second / (1024 * 1024),
            "error_rate": 1 - stats.success_rate,
            "total_throughput": stats.total_files
        }

    def get_bottlenecks(self) -> Dict[str, any]:
        """Identify performance bottlenecks."""
        stats = self.get_stats()
        bottlenecks = {}

        # Check success rate
        if stats.success_rate < 0.8:
            bottlenecks["low_success_rate"] = {
                "current": stats.success_rate,
                "threshold": 0.8,
                "top_errors": dict(list(stats.error_types.items())[:3])
            }

        # Check processing speed
        if stats.avg_processing_time > 5.0:
            bottlenecks["slow_processing"] = {
                "current": stats.avg_processing_time,
                "threshold": 5.0,
                "slowest_file_types": self._get_slowest_file_types(stats)
            }

        # Check throughput
        if stats.files_per_second < 1.0:
            bottlenecks["low_throughput"] = {
                "current": stats.files_per_second,
                "threshold": 1.0,
                "suggested_actions": ["Increase parallel processing", "Optimize file type processors"]
            }

        return bottlenecks

    def get_recommendations(self) -> List[str]:
        """Get optimization recommendations."""
        recommendations = []
        stats = self.get_stats()
        bottlenecks = self.get_bottlenecks()

        if "low_success_rate" in bottlenecks:
            recommendations.append("Review and fix common error patterns")
            recommendations.append("Implement better file validation")

        if "slow_processing" in bottlenecks:
            recommendations.append("Optimize processors for large files")
            recommendations.append("Consider chunked processing for large documents")

        if "low_throughput" in bottlenecks:
            recommendations.append("Increase max_concurrent_files setting")
            recommendations.append("Implement batch processing optimizations")

        # File type specific recommendations
        if stats.processing_times_by_type:
            slowest_type = max(
                stats.processing_times_by_type.items(),
                key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0
            )
            if slowest_type[1] and sum(slowest_type[1]) / len(slowest_type[1]) > 10:
                recommendations.append(f"Optimize processing for {slowest_type[0]} files")

        return recommendations

    def export_stats(self, format: str = "dict") -> Dict[str, any]:
        """Export statistics in specified format."""
        stats = self.get_stats()
        performance = self.get_performance_metrics()
        bottlenecks = self.get_bottlenecks()
        recommendations = self.get_recommendations()

        export_data = {
            "summary": {
                "total_files": stats.total_files,
                "success_rate": stats.success_rate,
                "avg_processing_time": stats.avg_processing_time,
                "total_size_mb": stats.total_size_bytes / (1024 * 1024),
                "session_duration": (datetime.now() - self.session_start_time).total_seconds(),
            },
            "performance": performance,
            "file_types": stats.file_types,
            "error_analysis": {
                "error_types": stats.error_types,
                "error_patterns": dict(self.error_patterns)
            },
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "hourly_throughput": stats.hourly_throughput,
            "timestamp": datetime.now().isoformat()
        }

        if format == "json":
            import json
            return json.dumps(export_data, indent=2, default=str)

        return export_data

    def reset(self):
        """Reset monitoring data."""
        self.results.clear()
        self.hourly_stats.clear()
        self.error_patterns.clear()
        self.session_start_time = datetime.now()

    def _update_hourly_stats(self, timestamp: datetime):
        """Update hourly statistics."""
        hour_key = timestamp.strftime("%Y-%m-%d %H:00")
        self.hourly_stats[hour_key] += 1

    def _categorize_error(self, error_message: str) -> str:
        """Categorize error message for tracking."""
        error_lower = error_message.lower()

        if "too large" in error_lower or "size" in error_lower:
            return "file_size"
        elif "not found" in error_lower or "does not exist" in error_lower:
            return "file_not_found"
        elif "permission" in error_lower or "access" in error_lower:
            return "permission_denied"
        elif "password" in error_lower or "encrypted" in error_lower:
            return "password_protected"
        elif "corrupt" in error_lower or "invalid" in error_lower:
            return "file_corruption"
        elif "timeout" in error_lower:
            return "processing_timeout"
        elif "memory" in error_lower:
            return "memory_error"
        elif "processor" in error_lower or "format" in error_lower:
            return "unsupported_format"
        else:
            return "other"

    def _get_slowest_file_types(self, stats: IngestionStats) -> Dict[str, float]:
        """Get file types with slowest average processing times."""
        avg_times = {}
        for file_type, times in stats.processing_times_by_type.items():
            if times:
                avg_times[file_type] = sum(times) / len(times)

        return dict(sorted(avg_times.items(), key=lambda x: x[1], reverse=True)[:5])

    def _calculate_hourly_throughput(self, results: List[IngestionResult]) -> Dict[str, int]:
        """Calculate hourly throughput."""
        hourly_counts = defaultdict(int)
        for result in results:
            hour_key = result.timestamp.strftime("%Y-%m-%d %H:00")
            hourly_counts[hour_key] += 1

        return dict(hourly_counts)

    def get_real_time_metrics(self) -> Dict[str, any]:
        """Get real-time metrics for monitoring dashboard."""
        recent_results = self.results[-100:]  # Last 100 operations

        if not recent_results:
            return {"status": "idle"}

        recent_successful = [r for r in recent_results if r.success]
        recent_failed = [r for r in recent_results if not r.success]

        # Calculate recent performance
        recent_success_rate = len(recent_successful) / len(recent_results)
        recent_avg_time = sum(r.processing_time for r in recent_results) / len(recent_results)

        # Current processing rate
        if len(recent_results) >= 2:
            time_span = (recent_results[-1].timestamp - recent_results[0].timestamp).total_seconds()
            current_rate = len(recent_results) / time_span if time_span > 0 else 0
        else:
            current_rate = 0

        return {
            "status": "active",
            "recent_success_rate": recent_success_rate,
            "recent_avg_processing_time": recent_avg_time,
            "current_processing_rate": current_rate,
            "recent_file_count": len(recent_results),
            "recent_error_count": len(recent_failed),
            "last_activity": recent_results[-1].timestamp.isoformat() if recent_results else None
        }