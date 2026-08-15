import json
from datetime import UTC, datetime
from pathlib import Path

from app.pipeline import AccessDecisionEngine
from app.schemas import AccessRequest, Metadata


def build_scenarios() -> list[AccessRequest]:
    repository_root = Path(__file__).resolve().parent
    occurred_at = datetime(2026, 8, 15, 9, 0, tzinfo=UTC)
    return [
        AccessRequest(
            event_id="e-1001",
            trace_id="trace-e-1001",
            gate_id="gate-1",
            camera_id="camera-1",
            image_path=str(repository_root / "e-1001.jpg"),
            occurred_at=occurred_at,
        ),
        AccessRequest(
            event_id="e-1002",
            trace_id="trace-e-1002",
            gate_id="gate-1",
            camera_id="camera-1",
            image_path=str(repository_root / "e-1002.jpg"),
            occurred_at=occurred_at,
            metadata=Metadata(occlusion_hint="mask"),
        ),
        AccessRequest(
            event_id="e-1003",
            trace_id="trace-e-1003",
            gate_id="gate-1",
            camera_id="camera-1",
            image_path=str(repository_root / "e-1003.jpg"),
            occurred_at=occurred_at,
            metadata=Metadata(spoofing_suspected=True),
        ),
        AccessRequest(
            event_id="e-1004",
            trace_id="trace-e-1004",
            gate_id="gate-1",
            camera_id="camera-1",
            image_path=str(repository_root / "e-1001.jpg"),
            occurred_at=occurred_at,
        ),
        AccessRequest(
            event_id="e-1005",
            trace_id="trace-e-1005",
            gate_id="gate-1",
            camera_id="camera-1",
            image_path=str(repository_root / "e-1001.jpg"),
            occurred_at=occurred_at,
            metadata=Metadata(network="offline", cache_age_minutes=125),
        ),
    ]


def main() -> None:
    expected = {
        "e-1001": (
            "allow",
            [
                "quality_ok",
                "liveness_ok",
                "match_above_allow_threshold",
            ],
        ),
        "e-1002": (
            "manual_review",
            ["low_quality_score", "face_occluded"],
        ),
        "e-1003": ("manual_review", ["liveness_failed"]),
        "e-1004": (
            "manual_review",
            ["low_margin_to_second_best"],
        ),
        "e-1005": (
            "manual_review",
            ["stale_edge_cache_offline"],
        ),
    }
    engine = AccessDecisionEngine()
    for scenario in build_scenarios():
        response = engine.verify(scenario)
        expected_decision, expected_reasons = expected[scenario.event_id]
        assert response.decision == expected_decision
        assert response.reasons == expected_reasons
        print(json.dumps(response.model_dump(mode="json"), sort_keys=True))
    print(json.dumps({"scenarios": 5, "status": "success"}, sort_keys=True))


if __name__ == "__main__":
    main()
