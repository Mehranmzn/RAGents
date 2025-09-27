"""Base configuration classes."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel


class BaseConfig(BaseModel):
    """Base configuration class with environment variable support."""

    class Config:
        extra = "forbid"
        validate_assignment = True

    @classmethod
    def from_env(cls, prefix: str = "") -> "BaseConfig":
        """Create config from environment variables."""
        env_vars = {}
        for key, field_info in cls.model_fields.items():
            env_key = f"{prefix}{key.upper()}" if prefix else key.upper()
            if env_key in os.environ:
                env_vars[key] = os.environ[env_key]
        return cls(**env_vars)


@dataclass
class WorkingDirectories:
    """Working directory configuration."""

    base_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("RAGENTS_WORKING_DIR", "./output"))
    )
    documents_dir: Path = field(default=None)
    cache_dir: Path = field(default=None)
    logs_dir: Path = field(default=None)
    temp_dir: Path = field(default=None)

    def __post_init__(self):
        """Initialize derived directories."""
        if self.documents_dir is None:
            self.documents_dir = self.base_dir / "documents"
        if self.cache_dir is None:
            self.cache_dir = self.base_dir / "cache"
        if self.logs_dir is None:
            self.logs_dir = self.base_dir / "logs"
        if self.temp_dir is None:
            self.temp_dir = self.base_dir / "temp"

        # Create directories if they don't exist
        for directory in [
            self.base_dir,
            self.documents_dir,
            self.cache_dir,
            self.logs_dir,
            self.temp_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)