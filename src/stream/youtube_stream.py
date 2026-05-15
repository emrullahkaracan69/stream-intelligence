import logging
import subprocess
import time

import cv2


logger = logging.getLogger(__name__)


class YouTubeStream:
    """Live stream reader for YouTube URLs using yt-dlp and OpenCV."""

    def __init__(self, url: str) -> None:
        """Initialise YouTube stream reader.

        Args:
            url: YouTube live stream URL.
        """
        self.url = url
        self.cap = None
        self.start_time = None
        self.frame_count = 0
        self._stream_url = None

    def _get_stream_url(self) -> str:
        """Extract direct stream URL from YouTube using yt-dlp.

        Returns:
            Direct stream URL suitable for OpenCV VideoCapture.

        Raises:
            RuntimeError: If yt-dlp fails to extract the stream URL.
        """
        logger.info("Extracting stream URL for: %s", self.url)
        try:
            result = subprocess.check_output(
                ["yt-dlp", "-g", "-f", "best[ext=mp4]/best", self.url],
                stderr=subprocess.DEVNULL,  # mute output from yt-dlp
                timeout=30,
            )
            stream_url = result.decode().strip()
            logger.info("Stream URL extracted successfully")
            return stream_url
        except subprocess.CalledProcessError as e:
            logger.error("yt-dlp failed to extract stream URL: %s", e)
            raise RuntimeError(f"Could not extract stream URL from: {self.url}") from e
        except subprocess.TimeoutExpired:
            logger.error("yt-dlp timed out after 30 seconds")
            raise RuntimeError("Stream URL extraction timed out") from None

    def __enter__(self):
        """Connect to YouTube stream and initialise frame timing.

        Returns:
            self: The YouTubeStream instance for use in with-blocks.

        Raises:
            RuntimeError: If the stream URL cannot be extracted or opened.
        """
        logger.info("Connecting to the stream %s", self.url)

        self._stream_url = self._get_stream_url()
        self.cap = cv2.VideoCapture(self._stream_url)

        if not self.cap.isOpened():
            logger.error("Failed to connect stream at url: %s", self.url)
            raise RuntimeError(
                f"Stream url {self.url} could not be opened. "
                "Check that the device is connected and not in use."
            )

        self.start_time = time.time()
        logger.info("Connected camera %s successfully", self.url)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the stream resource on exit."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            logger.info("Stream %s released", self.url)

    def read_frame(self):
        """Read a single frame and return it with current FPS.

        Returns:
            tuple[bool, np.ndarray | None, float]:
                ret: True if frame was read successfully.
                frame: The captured frame, or None on failure.
                fps: Current frames-per-second rate.
        """
        if self.start_time is None:
            logger.warning("read_frame called before camera was opened")
            return False, None, 0.0

        ret, frame = self.cap.read()
        self.frame_count += 1
        fps = self.frame_count / (time.time() - self.start_time)

        return ret, frame, fps
