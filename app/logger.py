import json
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from app.schemas import AccessRequest, AccessResponse


class AuditLogger:
    def __init__(self, log_path: str | Path = "access_events.log") -> None:
        self.log_path = Path(log_path)
        self._lock = Lock()

    def log(self, request: AccessRequest, response: AccessResponse) -> None:
        record: dict[str, Any] = {
            "trace_id": request.trace_id or request.event_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "event_id": request.event_id,
            "gate_id": request.gate_id,
            "camera_id": request.camera_id,
            "occurred_at": request.occurred_at.isoformat(),
            "image_path": request.image_path,
            "person_id_matched": response.matched_employee_id,
            "quality_score": response.quality_metrics.quality_score,
            "liveness_score": response.quality_metrics.liveness_score,
            "top1_similarity_score": response.match_score,
            "decision": response.decision.value,
            "decision_reason": response.reasons[0],
            "decision_reasons": response.reasons,
            "execution_time_ms": response.execution_time_ms.model_dump(),
            "face_observation": response.face_observation.model_dump(),
            "turnstile_command": response.turnstile_command,
        }
        line = json.dumps(record, separators=(",", ":"), sort_keys=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"{line}\n")
