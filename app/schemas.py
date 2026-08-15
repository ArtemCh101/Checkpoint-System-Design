from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DecisionEnum(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    MANUAL_REVIEW = "manual_review"


class Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    network: Literal["online", "offline"] = "online"
    cache_age_minutes: int = Field(default=0, ge=0)
    lighting: Literal["normal", "dim", "backlight"] = "normal"
    occlusion_hint: Literal["mask", "partial"] | None = None
    spoofing_suspected: bool = False


class AccessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1, max_length=64)
    trace_id: str | None = Field(default=None, min_length=1, max_length=128)
    gate_id: str = Field(default="gate-1", min_length=1, max_length=64)
    camera_id: str = Field(default="camera-1", min_length=1, max_length=64)
    image_path: str = Field(min_length=1, max_length=1024)
    occurred_at: datetime
    metadata: Metadata = Field(default_factory=Metadata)


class QualityMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quality_score: float = Field(ge=0.0, le=1.0)
    liveness_score: float | None = Field(default=None, ge=0.0, le=1.0)


class FaceObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    face_detected: bool
    face_count: int = Field(ge=0)
    aligned: bool
    bbox: list[float] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
    )


class PipelineStageLatency(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quality_gate_ms: float = Field(ge=0.0)
    detection_ms: float = Field(ge=0.0)
    liveness_ms: float = Field(ge=0.0)
    embedding_ms: float = Field(ge=0.0)
    vector_search_ms: float = Field(ge=0.0)
    total_ms: float = Field(ge=0.0)


class AccessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    decision: DecisionEnum
    turnstile_command: Literal["open", "keep_closed"]
    requires_human_review: bool
    reasons: list[str]
    matched_employee_id: str | None = None
    face_observation: FaceObservation
    quality_metrics: QualityMetrics
    execution_time_ms: PipelineStageLatency
    match_score: float | None = Field(default=None, ge=0.0, le=1.0)
    second_best_score: float | None = Field(default=None, ge=0.0, le=1.0)
    margin_to_second_best: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
