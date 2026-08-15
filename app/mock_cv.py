from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app.schemas import FaceObservation, Metadata


FloatVector = NDArray[np.float32]
UInt8Image = NDArray[np.uint8]


def _seeded_unit_vector(key: str) -> FloatVector:
    digest = sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    generator = np.random.default_rng(seed)
    vector = generator.standard_normal(512).astype(np.float32)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def _orthogonal_direction(
    basis: list[FloatVector],
    key: str,
) -> FloatVector:
    vector = _seeded_unit_vector(key).astype(np.float64)
    for base in basis:
        normalized_base = base.astype(np.float64)
        vector = vector - np.dot(vector, normalized_base) * normalized_base
    return (vector / np.linalg.norm(vector)).astype(np.float32)


@dataclass(frozen=True)
class LoadedImage:
    bgr: UInt8Image
    gray: UInt8Image


@dataclass(frozen=True)
class QualityResult:
    score: float
    laplacian_variance: float


@dataclass(frozen=True)
class DetectionResult:
    observation: FaceObservation
    aligned_face: UInt8Image | None


@dataclass(frozen=True)
class MatchResult:
    employee_id: str
    match_score: float
    second_best_score: float
    margin_to_second_best: float


class ImageLoadError(ValueError):
    pass


class MockCVEngine:
    quality_stage_name = "Laplacian Quality Gate"
    detection_stage_name = "SCRFD"
    liveness_stage_name = "MiniFASNetV2"
    embedding_stage_name = "ArcFace ResNet-50"
    vector_search_stage_name = "FAISS IndexFlatIP"
    canonical_face_size = (112, 112)
    simulated_scenarios = {
        "e-1001",
        "e-1002",
        "e-1003",
        "e-1004",
        "e-1005",
    }

    def __init__(self) -> None:
        cascade_path = (
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError("OpenCV Haar cascade could not be loaded")
        self.reference_embeddings = self._build_reference_index()

    def load_image(self, image_path: str) -> LoadedImage:
        path = Path(image_path)
        if not path.is_file():
            raise ImageLoadError(f"Image does not exist: {image_path}")
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError) as error:
            message = f"Image integrity check failed: {image_path}"
            raise ImageLoadError(message) from error
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ImageLoadError(f"OpenCV could not decode image: {image_path}")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return LoadedImage(bgr=bgr, gray=gray)

    def assess_quality_laplacian(
        self,
        image: LoadedImage,
        image_path: str,
        event_id: str,
        metadata: Metadata,
    ) -> QualityResult:
        variance = float(
            cv2.Laplacian(image.gray, cv2.CV_64F).var()
        )
        normalized = variance / (variance + 100.0)
        scenario_id = self.scenario_id(image_path, event_id)
        if scenario_id == "e-1002" or metadata.occlusion_hint is not None:
            normalized = 0.35
        elif metadata.lighting in {"dim", "backlight"}:
            normalized = min(normalized, 0.42)
        return QualityResult(
            score=round(max(0.0, min(1.0, normalized)), 6),
            laplacian_variance=round(variance, 6),
        )

    def detect_and_align_scrfd(self, image: LoadedImage) -> DetectionResult:
        detected = self.detector.detectMultiScale(
            image.gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(40, 40),
        )
        boxes = [tuple(int(value) for value in box) for box in detected]
        if not boxes:
            return DetectionResult(
                observation=FaceObservation(
                    face_detected=False,
                    face_count=0,
                    aligned=False,
                    bbox=None,
                ),
                aligned_face=None,
            )
        x, y, width, height = max(
            boxes,
            key=lambda box: box[2] * box[3],
        )
        face_crop = image.bgr[y : y + height, x : x + width]
        aligned_face = cv2.resize(
            face_crop,
            self.canonical_face_size,
            interpolation=cv2.INTER_AREA,
        )
        return DetectionResult(
            observation=FaceObservation(
                face_detected=True,
                face_count=len(boxes),
                aligned=True,
                bbox=[float(x), float(y), float(width), float(height)],
            ),
            aligned_face=aligned_face,
        )

    def assess_liveness_minifasnet(
        self,
        image_path: str,
        event_id: str,
        metadata: Metadata,
    ) -> float:
        scenario_id = self.scenario_id(image_path, event_id)
        if scenario_id == "e-1003" or metadata.spoofing_suspected:
            return 0.20
        if metadata.occlusion_hint == "mask":
            return 0.20
        return 0.95

    def extract_embedding_arcface(
        self,
        aligned_face: UInt8Image,
        image_path: str,
        event_id: str,
    ) -> FloatVector:
        raw_embedding = self._histogram_embedding(aligned_face)
        scenario_id = self.scenario_id(image_path, event_id)
        if scenario_id in {"e-1001", "e-1005"}:
            return self._candidate_with_scores(0.94, 0.61, scenario_id)
        if scenario_id == "e-1004":
            return self._candidate_with_scores(
                0.910193,
                0.906378,
                scenario_id,
            )
        if "emp-9999" in image_path.lower() or "emp-9999" in event_id.lower():
            reference = self.reference_embeddings["emp-9999"]
            direction = _orthogonal_direction(
                [reference],
                f"{image_path}:{event_id}",
            )
            candidate = 0.94 * reference
            candidate = candidate + np.sqrt(1.0 - 0.94**2) * direction
            return (candidate / np.linalg.norm(candidate)).astype(np.float32)
        return raw_embedding

    def is_occluded(
        self,
        image_path: str,
        event_id: str,
        metadata: Metadata,
    ) -> bool:
        return (
            self.scenario_id(image_path, event_id) == "e-1002"
            or metadata.occlusion_hint is not None
        )

    def index_matrix(self) -> tuple[list[str], NDArray[np.float32]]:
        employee_ids = list(self.reference_embeddings)
        matrix = np.vstack(
            [self.reference_embeddings[employee_id] for employee_id in employee_ids]
        ).astype(np.float32)
        return employee_ids, matrix

    @classmethod
    def scenario_id(cls, image_path: str, event_id: str) -> str:
        normalized_event_id = event_id.lower()
        if normalized_event_id in cls.simulated_scenarios:
            return normalized_event_id
        return Path(image_path).stem.lower()

    def _build_reference_index(self) -> dict[str, FloatVector]:
        reference_path = Path(__file__).resolve().parent.parent / "e-1001.jpg"
        reference_image = self.load_image(str(reference_path))
        detection = self.detect_and_align_scrfd(reference_image)
        if detection.aligned_face is None:
            raise RuntimeError("Reference face could not be detected")
        employee_vector = self._histogram_embedding(detection.aligned_face)
        nearby_direction = _orthogonal_direction(
            [employee_vector],
            "emp-7310-reference-direction",
        )
        nearby_vector = 0.65 * employee_vector
        nearby_vector = nearby_vector + (
            np.sqrt(1.0 - 0.65**2) * nearby_direction
        )
        terminated_vector = _orthogonal_direction(
            [employee_vector, nearby_direction],
            "emp-9999-reference",
        )
        return {
            "emp-4821": employee_vector,
            "emp-7310": (
                nearby_vector / np.linalg.norm(nearby_vector)
            ).astype(np.float32),
            "emp-9999": terminated_vector,
        }

    def _candidate_with_scores(
        self,
        employee_score: float,
        second_best_score: float,
        key: str,
    ) -> FloatVector:
        employee = self.reference_embeddings["emp-4821"]
        second = self.reference_embeddings["emp-7310"]
        correlation = float(np.dot(employee, second))
        denominator = np.sqrt(1.0 - correlation**2)
        nearby_direction = (
            second - correlation * employee
        ) / denominator
        third_direction = _orthogonal_direction(
            [employee, nearby_direction.astype(np.float32)],
            f"{key}-candidate-direction",
        )
        nearby_component = (
            second_best_score - correlation * employee_score
        ) / denominator
        remaining = max(
            0.0,
            1.0 - employee_score**2 - nearby_component**2,
        )
        candidate = employee_score * employee
        candidate = candidate + nearby_component * nearby_direction
        candidate = candidate + np.sqrt(remaining) * third_direction
        return (candidate / np.linalg.norm(candidate)).astype(np.float32)

    @staticmethod
    def _histogram_embedding(aligned_face: UInt8Image) -> FloatVector:
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        gray_histogram = cv2.calcHist(
            [gray],
            [0],
            None,
            [256],
            [0, 256],
        ).reshape(-1)
        channel_histograms = [
            cv2.calcHist(
                [aligned_face],
                [channel],
                None,
                [64],
                [0, 256],
            ).reshape(-1)
            for channel in range(3)
        ]
        spatial_statistics = cv2.resize(
            gray,
            (8, 8),
            interpolation=cv2.INTER_AREA,
        ).reshape(-1)
        vector = np.concatenate(
            [gray_histogram, *channel_histograms, spatial_statistics]
        ).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return np.zeros(512, dtype=np.float32)
        return (vector / norm).astype(np.float32)
