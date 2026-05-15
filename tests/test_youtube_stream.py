from unittest.mock import MagicMock, patch
import pytest
import subprocess
from src.stream.youtube_stream import YouTubeStream


@pytest.fixture
def stream():
    """Create YouTubeStream with mocked yt-dlp and OpenCV."""
    with patch("src.stream.youtube_stream.subprocess.check_output") as mock_cmd, patch(
        "src.stream.youtube_stream.cv2.VideoCapture"
    ) as mock_cap_class:
        mock_cmd.return_value = b"https://fake-stream-url.m3u8\n"
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, MagicMock())
        mock_cap_class.return_value = mock_cap

        with YouTubeStream("https://youtube.com/watch?v=test") as s:
            yield s


def test_stream_connects_successfully(stream):
    """YouTubeStream should connect and return valid frame data."""
    ret, frame, fps = stream.read_frame()
    assert ret is True
    assert frame is not None
    assert fps >= 0.0


def test_stream_raises_when_ytdlp_fails():
    """RuntimeError should be raised when yt-dlp cannot extract URL."""
    with patch("src.stream.youtube_stream.subprocess.check_output") as mock_cmd:
        mock_cmd.side_effect = subprocess.CalledProcessError(1, "yt-dlp")
        with pytest.raises(RuntimeError):
            with YouTubeStream("https://youtube.com/watch?v=test") as _:
                pass


def test_stream_raises_when_cap_fails_to_open():
    """RuntimeError should be raised when OpenCV cannot open stream URL."""
    with patch("src.stream.youtube_stream.subprocess.check_output") as mock_cmd, patch(
        "src.stream.youtube_stream.cv2.VideoCapture"
    ) as mock_cap_class:
        mock_cmd.return_value = b"https://fake-url.m3u8\n"
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_class.return_value = mock_cap
        with pytest.raises(RuntimeError):
            with YouTubeStream("https://youtube.com/watch?v=test") as _:
                pass
