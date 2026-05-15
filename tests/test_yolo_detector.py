from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detector.yolo_detector import YoloDetector


@pytest.fixture
def detector():
    """Create a YoloDetector instance with mocked model."""
    with patch("src.detector.yolo_detector.YOLO") as mock_yolo:
        mock_yolo.return_value = MagicMock()
        yield YoloDetector()


def test_detect_returns_empty_list_for_none_frame(detector):
    """detect should return empty list when frame is None."""
    result = detector.detect(None)
    assert result == []


def test_detect_filters_untracked_classes(detector):
    """detect should ignore detections outside TRACKED_CLASSES."""
    mock_box = MagicMock()
    mock_box.cls = MagicMock()
    mock_box.cls.__int__ = lambda self: 73  # 73 = 'book', not tracked
    mock_box.conf = MagicMock()
    float(mock_box.conf)

    mock_results = [MagicMock()]
    mock_results[0].names = {73: "book"}
    mock_results[0].boxes = [mock_box]

    detector.model.return_value = mock_results
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = detector.detect(frame)
    assert result == []


def test_detect_filters_low_confidence(detector):
    """detect should ignore detections below confidence threshold."""
    mock_box = MagicMock()
    mock_box.cls = MagicMock()
    mock_box.cls.__int__ = lambda self: 0  # 0 = 'person', tracked
    mock_box.conf = 0.1  # below default threshold of 0.5

    mock_results = [MagicMock()]
    mock_results[0].names = {0: "person"}
    mock_results[0].boxes = [mock_box]

    detector.model.return_value = mock_results
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = detector.detect(frame)
    assert result == []
