"""
Policy definition — parsed once at startup, not per-request.

The Java-OOP instinct here is a singleton class with a static
`getInstance()`. Instead: parse the YAML into a Pydantic model once (in
main.py's lifespan) and pass the resulting plain value through FastAPI's
`Depends()`. No singleton machinery, no hidden global mutable state, and
it's trivially fakeable in tests — just construct a `Policy(...)` directly.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class CheckAction(str, Enum):
    REDACT = "redact"
    BLOCK = "block"
    ALLOW = "allow"


class PIICheckConfig(BaseModel):
    action: CheckAction = CheckAction.REDACT


class TopicDenialConfig(BaseModel):
    action: CheckAction = CheckAction.BLOCK
    topics: list[str] = Field(default_factory=list)
    threshold: float = 0.75

    @field_validator("threshold")
    @classmethod
    def _threshold_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {v}")
        return v


class ChecksConfig(BaseModel):
    pii: PIICheckConfig = Field(default_factory=PIICheckConfig)
    topic_denial: TopicDenialConfig = Field(default_factory=TopicDenialConfig)


class Policy(BaseModel):
    checks: ChecksConfig = Field(default_factory=ChecksConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Policy":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)
