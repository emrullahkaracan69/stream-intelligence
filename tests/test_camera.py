from unittest.mock import MagicMock, patch

import pytest

from src.detector.camera import CameraSource


def test_camera_opens_successfully():
    """Camera should open and return valid frame data."""
    with patch("src.detector.camera.cv2.VideoCapture") as mock_cap_class:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, MagicMock())
        mock_cap_class.return_value = mock_cap

        with CameraSource(0) as cam:
            ret, frame, fps = cam.read_frame()

        assert ret is True
        assert frame is not None
        assert fps >= 0.0


def test_camera_raises_on_invalid_index():
    """RuntimeError should be raised when camera cannot be opened."""
    with patch("src.detector.camera.cv2.VideoCapture") as mock_cap_class:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_class.return_value = mock_cap

        with pytest.raises(RuntimeError):
            with CameraSource(99) as cam:
                pass