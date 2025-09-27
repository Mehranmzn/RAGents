"""Base classes for guardrails system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class GuardrailSeverity(Enum):
    """Severity levels for guardrail violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GuardrailAction(Enum):
    """Actions to take when guardrail is triggered."""
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    MODIFY = "modify"
    ESCALATE = "escalate"


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    passed: bool
    guardrail_name: str
    severity: GuardrailSeverity
    action: GuardrailAction
    message: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    processing_time: float = 0.0
    modified_content: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()


class GuardrailConfig(BaseModel):
    """Configuration for a guardrail."""
    name: str
    enabled: bool = True
    severity: GuardrailSeverity = GuardrailSeverity.MEDIUM
    action: GuardrailAction = GuardrailAction.WARN
    confidence_threshold: float = 0.7
    custom_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class Guardrail(ABC):
    """Abstract base class for all guardrails."""

    def __init__(self, config: GuardrailConfig):
        self.config = config
        self.name = config.name
        self.enabled = config.enabled

    @abstractmethod
    async def check(self, content: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
        """Check content against this guardrail."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get human-readable description of this guardrail."""
        pass

    def is_enabled(self) -> bool:
        """Check if this guardrail is enabled."""
        return self.enabled

    def enable(self):
        """Enable this guardrail."""
        self.enabled = True
        self.config.enabled = True

    def disable(self):
        """Disable this guardrail."""
        self.enabled = False
        self.config.enabled = False

    def update_config(self, **kwargs):
        """Update guardrail configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def _create_result(
        self,
        passed: bool,
        message: str,
        severity: Optional[GuardrailSeverity] = None,
        action: Optional[GuardrailAction] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        modified_content: Optional[str] = None
    ) -> GuardrailResult:
        """Helper method to create guardrail results."""
        return GuardrailResult(
            passed=passed,
            guardrail_name=self.name,
            severity=severity or self.config.severity,
            action=action or self.config.action,
            message=self.config.custom_message or message,
            confidence=confidence,
            metadata=metadata or {},
            modified_content=modified_content
        )


class GuardrailManager:
    """Manager for multiple guardrails."""

    def __init__(self):
        self.guardrails: List[Guardrail] = []
        self.results_history: List[GuardrailResult] = []

    def add_guardrail(self, guardrail: Guardrail):
        """Add a guardrail to the manager."""
        self.guardrails.append(guardrail)

    def remove_guardrail(self, name: str) -> bool:
        """Remove a guardrail by name."""
        for i, guardrail in enumerate(self.guardrails):
            if guardrail.name == name:
                del self.guardrails[i]
                return True
        return False

    def get_guardrail(self, name: str) -> Optional[Guardrail]:
        """Get a guardrail by name."""
        for guardrail in self.guardrails:
            if guardrail.name == name:
                return guardrail
        return None

    async def check_all(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        stop_on_first_failure: bool = False
    ) -> List[GuardrailResult]:
        """Check content against all enabled guardrails."""
        results = []

        for guardrail in self.guardrails:
            if not guardrail.is_enabled():
                continue

            try:
                result = await guardrail.check(content, context)
                results.append(result)
                self.results_history.append(result)

                # Stop on first failure if requested
                if stop_on_first_failure and not result.passed:
                    break

            except Exception as e:
                # Create error result
                error_result = GuardrailResult(
                    passed=False,
                    guardrail_name=guardrail.name,
                    severity=GuardrailSeverity.HIGH,
                    action=GuardrailAction.BLOCK,
                    message=f"Guardrail error: {str(e)}",
                    confidence=0.0,
                    metadata={"error": str(e)}
                )
                results.append(error_result)
                self.results_history.append(error_result)

        return results

    def get_active_guardrails(self) -> List[Guardrail]:
        """Get list of active (enabled) guardrails."""
        return [g for g in self.guardrails if g.is_enabled()]

    def get_guardrail_summary(self) -> Dict[str, Any]:
        """Get summary of all guardrails."""
        total = len(self.guardrails)
        enabled = len(self.get_active_guardrails())

        severity_counts = {}
        action_counts = {}

        for guardrail in self.guardrails:
            severity = guardrail.config.severity.value
            action = guardrail.config.action.value

            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            action_counts[action] = action_counts.get(action, 0) + 1

        return {
            "total_guardrails": total,
            "enabled_guardrails": enabled,
            "disabled_guardrails": total - enabled,
            "severity_distribution": severity_counts,
            "action_distribution": action_counts,
            "guardrail_names": [g.name for g in self.guardrails]
        }

    def get_violation_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get violation statistics for the specified time period."""
        cutoff_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if hours < 24:
            cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_results = [
            r for r in self.results_history
            if r.timestamp >= cutoff_time
        ]

        if not recent_results:
            return {"total_checks": 0, "violations": 0}

        violations = [r for r in recent_results if not r.passed]

        violation_by_guardrail = {}
        violation_by_severity = {}
        violation_by_action = {}

        for violation in violations:
            # By guardrail
            name = violation.guardrail_name
            violation_by_guardrail[name] = violation_by_guardrail.get(name, 0) + 1

            # By severity
            severity = violation.severity.value
            violation_by_severity[severity] = violation_by_severity.get(severity, 0) + 1

            # By action
            action = violation.action.value
            violation_by_action[action] = violation_by_action.get(action, 0) + 1

        return {
            "total_checks": len(recent_results),
            "violations": len(violations),
            "violation_rate": len(violations) / len(recent_results),
            "violations_by_guardrail": violation_by_guardrail,
            "violations_by_severity": violation_by_severity,
            "violations_by_action": violation_by_action,
            "time_period_hours": hours
        }

    def clear_history(self):
        """Clear the results history."""
        self.results_history.clear()

    def export_config(self) -> Dict[str, Any]:
        """Export guardrail configurations."""
        return {
            "guardrails": [
                {
                    "name": g.name,
                    "type": g.__class__.__name__,
                    "config": g.config.dict(),
                    "description": g.get_description()
                }
                for g in self.guardrails
            ],
            "export_timestamp": datetime.now().isoformat()
        }

    def import_config(self, config_data: Dict[str, Any]):
        """Import guardrail configurations."""
        # This would need to be implemented based on specific guardrail types
        # For now, just update existing guardrail configs
        guardrail_configs = {g["name"]: g["config"] for g in config_data.get("guardrails", [])}

        for guardrail in self.guardrails:
            if guardrail.name in guardrail_configs:
                new_config = guardrail_configs[guardrail.name]
                guardrail.update_config(**new_config)


# Helper functions
def create_simple_guardrail(
    name: str,
    check_function: callable,
    description: str,
    config: Optional[GuardrailConfig] = None
) -> Guardrail:
    """Create a simple guardrail from a function."""

    if config is None:
        config = GuardrailConfig(name=name)

    class SimpleGuardrail(Guardrail):
        def __init__(self, config: GuardrailConfig, check_func: callable, desc: str):
            super().__init__(config)
            self.check_function = check_func
            self.description = desc

        async def check(self, content: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
            try:
                result = await self.check_function(content, context)
                if isinstance(result, bool):
                    return self._create_result(
                        passed=result,
                        message=f"Simple check {'passed' if result else 'failed'}"
                    )
                elif isinstance(result, GuardrailResult):
                    return result
                else:
                    return self._create_result(
                        passed=False,
                        message="Invalid result from check function"
                    )
            except Exception as e:
                return self._create_result(
                    passed=False,
                    message=f"Check function error: {str(e)}",
                    severity=GuardrailSeverity.HIGH,
                    metadata={"error": str(e)}
                )

        def get_description(self) -> str:
            return self.description

    return SimpleGuardrail(config, check_function, description)