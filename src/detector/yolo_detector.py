import logging
import time
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from src.config import config


logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single object detection result."""

    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class YoloDetector:
    """Real-time object detector using YOLOv8."""

    TRACKED_CLASSES = {"person", "car", "truck", "bus", "motorcycle"}

    def __init__(self, model_path: str = config.model_path) -> None:
        """Initialise YOLOv8 detector and load model onto device.

        Args:
            model_path: Path or name of the YOLO model weights file.
        """
        logger.info("Loading YOLOv8 model: %s on %s", model_path, config.device)
        try:
            self.model = YOLO(model_path)
            self.model.to(config.device)
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise

        logger.info("Model loaded successfully")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a single frame and return filtered detections.

        Args:
            frame: BGR image array from OpenCV.

        Returns:
            List of Detection objects for tracked classes only.
        """
        if frame is None:
            logger.warning("detect called with None frame")
            return []

        start = time.time()
        results = self.model(frame, verbose=False)
        elapsed_ms = (time.time() - start) * 1000

        detections = self._parse_results(results)
        logger.debug("Inference: %.1f ms, %d detections", elapsed_ms, len(detections))

        return detections

    def _parse_results(self, results) -> list[Detection]:
        """Parse raw YOLO results into Detection objects.

        Args:
            results: Raw output from YOLO model inference.

        Returns:
            Filtered list of Detection objects for tracked classes only.
        """
        detections = []
        names = results[0].names

        for box in results[0].boxes:
            class_name = names[int(box.cls)]
            if class_name not in self.TRACKED_CLASSES:
                continue

            confidence = float(box.conf)
            if confidence < config.confidence:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        return detections
