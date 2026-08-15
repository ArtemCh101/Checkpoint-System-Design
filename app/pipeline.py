from threading import Lock
from time import perf_counter

import numpy as np

from app.logger import AuditLogger
from app.mock_cv import ImageLoadError, MatchResult, MockCVEngine
from app.schemas import (
    AccessRequest,
    AccessResponse,
    DecisionEnum,
    FaceObservation,
    PipelineStageLatency,
    QualityMetrics,
)


MOCK_EMPLOYEE_RIGHTS: dict[str, dict[str, str | list[str]]] = {
    "emp-4821": {
        "status": "active",
        "allowed_gates": ["gate-1"],
    },
    "emp-9999": {
        "status": "terminated",
        "allowed_gates": [],
    },
}


class AccessDecisionEngine:
    quality_threshold = 0.50
    liveness_threshold = 0.80
    deny_match_threshold = 0.50
    match_threshold = 0.75
    margin_threshold = 0.10
    maximum_offline_cache_age_minutes = 120
    cooldown_seconds = 5.0

    def __init__(
        self,
        cv_engine: MockCVEngine | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.cv_engine = cv_engine or MockCVEngine()
        self.audit_logger = audit_logger or AuditLogger()
        self._event_results: dict[str, AccessResponse] = {}
        self._last_allowed_at: dict[tuple[str, str], float] = {}
        self._state_lock = Lock()

    def verify(self, request: AccessRequest) -> AccessResponse:
        with self._state_lock:
            cached = self._event_results.get(request.event_id)
            if cached is not None:
                return cached.model_copy(deep=True)
            response = self._verify_once(request)
            self._event_results[request.event_id] = response.model_copy(
                deep=True
            )
            return response

    def _verify_once(self, request: AccessRequest) -> AccessResponse:
        pipeline_started = perf_counter()
        stage_times = self._empty_stage_times()
        empty_observation = FaceObservation(
            face_detected=False,
            face_count=0,
            aligned=False,
            bbox=None,
        )

        stage_started = perf_counter()
        try:
            image = self.cv_engine.load_image(request.image_path)
            quality = self.cv_engine.assess_quality_laplacian(
                image,
                request.image_path,
                request.event_id,
                request.metadata,
            )
        except ImageLoadError:
            stage_times["quality_gate_ms"] = self._elapsed_ms(stage_started)
            return self._manual_review(
                request,
                ["face_not_detected"],
                empty_observation,
                QualityMetrics(quality_score=0.0),
                stage_times,
                pipeline_started,
            )
        stage_times["quality_gate_ms"] = self._elapsed_ms(stage_started)

        stage_started = perf_counter()
        detection = self.cv_engine.detect_and_align_scrfd(image)
        stage_times["detection_ms"] = self._elapsed_ms(stage_started)

        if quality.score < self.quality_threshold:
            reasons = ["low_quality_score"]
            if self.cv_engine.is_occluded(
                request.image_path,
                request.event_id,
                request.metadata,
            ):
                reasons.append("face_occluded")
            return self._manual_review(
                request,
                reasons,
                detection.observation,
                QualityMetrics(quality_score=quality.score),
                stage_times,
                pipeline_started,
            )
        if not detection.observation.face_detected:
            return self._manual_review(
                request,
                ["face_not_detected"],
                detection.observation,
                QualityMetrics(quality_score=quality.score),
                stage_times,
                pipeline_started,
            )
        if detection.observation.face_count != 1:
            return self._manual_review(
                request,
                ["invalid_face_count"],
                detection.observation,
                QualityMetrics(quality_score=quality.score),
                stage_times,
                pipeline_started,
            )
        if detection.aligned_face is None or not detection.observation.aligned:
            return self._manual_review(
                request,
                ["face_alignment_failed"],
                detection.observation,
                QualityMetrics(quality_score=quality.score),
                stage_times,
                pipeline_started,
            )

        stage_started = perf_counter()
        liveness_score = self.cv_engine.assess_liveness_minifasnet(
            request.image_path,
            request.event_id,
            request.metadata,
        )
        stage_times["liveness_ms"] = self._elapsed_ms(stage_started)
        metrics = QualityMetrics(
            quality_score=quality.score,
            liveness_score=liveness_score,
        )
        if liveness_score < self.liveness_threshold:
            return self._manual_review(
                request,
                ["liveness_failed"],
                detection.observation,
                metrics,
                stage_times,
                pipeline_started,
            )

        stage_started = perf_counter()
        candidate = self.cv_engine.extract_embedding_arcface(
            detection.aligned_face,
            request.image_path,
            request.event_id,
        )
        stage_times["embedding_ms"] = self._elapsed_ms(stage_started)

        stage_started = perf_counter()
        match = self._search_faiss_index_flat_ip(candidate)
        stage_times["vector_search_ms"] = self._elapsed_ms(stage_started)

        if match.match_score < self.deny_match_threshold:
            return self._deny(
                request,
                ["no_match_found"],
                detection.observation,
                metrics,
                stage_times,
                pipeline_started,
                match,
            )
        if match.match_score < self.match_threshold:
            return self._manual_review(
                request,
                ["low_match_score"],
                detection.observation,
                metrics,
                stage_times,
                pipeline_started,
                match,
            )
        if match.margin_to_second_best < self.margin_threshold:
            return self._manual_review(
                request,
                ["low_margin_to_second_best"],
                detection.observation,
                metrics,
                stage_times,
                pipeline_started,
                match,
            )
        if not self._is_authorized(match.employee_id, request.gate_id):
            return self._deny(
                request,
                ["employee_access_revoked"],
                detection.observation,
                metrics,
                stage_times,
                pipeline_started,
                match,
            )
        if (
            request.metadata.network == "offline"
            and request.metadata.cache_age_minutes
            > self.maximum_offline_cache_age_minutes
        ):
            return self._manual_review(
                request,
                ["stale_edge_cache_offline"],
                detection.observation,
                metrics,
                stage_times,
                pipeline_started,
                match,
            )

        cooldown_key = (request.gate_id, match.employee_id)
        event_timestamp = request.occurred_at.timestamp()
        last_allowed_at = self._last_allowed_at.get(cooldown_key)
        if (
            last_allowed_at is not None
            and 0.0 <= event_timestamp - last_allowed_at < self.cooldown_seconds
        ):
            return self._manual_review(
                request,
                ["cooldown_active"],
                detection.observation,
                metrics,
                stage_times,
                pipeline_started,
                match,
            )

        response = AccessResponse(
            event_id=request.event_id,
            decision=DecisionEnum.ALLOW,
            turnstile_command="open",
            requires_human_review=False,
            reasons=[
                "quality_ok",
                "liveness_ok",
                "match_above_allow_threshold",
            ],
            matched_employee_id=match.employee_id,
            face_observation=detection.observation,
            quality_metrics=metrics,
            execution_time_ms=self._latencies(
                stage_times,
                pipeline_started,
            ),
            match_score=match.match_score,
            second_best_score=match.second_best_score,
            margin_to_second_best=match.margin_to_second_best,
        )
        self._last_allowed_at[cooldown_key] = event_timestamp
        self.audit_logger.log(request, response)
        return response

    def _search_faiss_index_flat_ip(
        self,
        candidate: np.ndarray,
    ) -> MatchResult:
        employee_ids, index = self.cv_engine.index_matrix()
        scores = np.dot(index, candidate)
        ranked_indices = np.argsort(scores)[::-1]
        best_index = int(ranked_indices[0])
        second_index = int(ranked_indices[1])
        best_score = round(float(np.clip(scores[best_index], 0.0, 1.0)), 6)
        second_score = round(
            float(np.clip(scores[second_index], 0.0, 1.0)),
            6,
        )
        return MatchResult(
            employee_id=employee_ids[best_index],
            match_score=best_score,
            second_best_score=second_score,
            margin_to_second_best=round(
                max(0.0, best_score - second_score),
                6,
            ),
        )

    @staticmethod
    def _is_authorized(employee_id: str, gate_id: str) -> bool:
        rights = MOCK_EMPLOYEE_RIGHTS.get(employee_id)
        if rights is None or rights["status"] != "active":
            return False
        allowed_gates = rights["allowed_gates"]
        return isinstance(allowed_gates, list) and gate_id in allowed_gates

    def _deny(
        self,
        request: AccessRequest,
        reasons: list[str],
        observation: FaceObservation,
        metrics: QualityMetrics,
        stage_times: dict[str, float],
        pipeline_started: float,
        match: MatchResult | None = None,
    ) -> AccessResponse:
        return self._closed_response(
            request,
            DecisionEnum.DENY,
            False,
            reasons,
            observation,
            metrics,
            stage_times,
            pipeline_started,
            match,
        )

    def _manual_review(
        self,
        request: AccessRequest,
        reasons: list[str],
        observation: FaceObservation,
        metrics: QualityMetrics,
        stage_times: dict[str, float],
        pipeline_started: float,
        match: MatchResult | None = None,
    ) -> AccessResponse:
        return self._closed_response(
            request,
            DecisionEnum.MANUAL_REVIEW,
            True,
            reasons,
            observation,
            metrics,
            stage_times,
            pipeline_started,
            match,
        )

    def _closed_response(
        self,
        request: AccessRequest,
        decision: DecisionEnum,
        requires_human_review: bool,
        reasons: list[str],
        observation: FaceObservation,
        metrics: QualityMetrics,
        stage_times: dict[str, float],
        pipeline_started: float,
        match: MatchResult | None,
    ) -> AccessResponse:
        response = AccessResponse(
            event_id=request.event_id,
            decision=decision,
            turnstile_command="keep_closed",
            requires_human_review=requires_human_review,
            reasons=reasons,
            matched_employee_id=match.employee_id if match else None,
            face_observation=observation,
            quality_metrics=metrics,
            execution_time_ms=self._latencies(
                stage_times,
                pipeline_started,
            ),
            match_score=match.match_score if match else None,
            second_best_score=match.second_best_score if match else None,
            margin_to_second_best=(
                match.margin_to_second_best if match else None
            ),
        )
        self.audit_logger.log(request, response)
        return response

    @staticmethod
    def _empty_stage_times() -> dict[str, float]:
        return {
            "quality_gate_ms": 0.0,
            "detection_ms": 0.0,
            "liveness_ms": 0.0,
            "embedding_ms": 0.0,
            "vector_search_ms": 0.0,
        }

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((perf_counter() - started) * 1000.0, 3)

    def _latencies(
        self,
        stage_times: dict[str, float],
        pipeline_started: float,
    ) -> PipelineStageLatency:
        return PipelineStageLatency(
            **stage_times,
            total_ms=self._elapsed_ms(pipeline_started),
        )
