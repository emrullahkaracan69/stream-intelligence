import logging
import time

import cv2

logger = logging.getLogger(__name__)


class CameraSource:
    def __init__(self, index):
        self.index = index
        self.cap = None
        self.frame_count = 0
        self.start_time = None

    def __enter__(self):
        """Open the camera and initialise frame timing.

        Returns:
            self: The CameraSource instance for use in with-blocks.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        logger.info("Opening camera at index %d", self.index)

        self.cap = cv2.VideoCapture(self.index)

        if not self.cap.isOpened():
            logger.error("Failed to open camera at index %d", self.index)
            raise RuntimeError(
                f"Camera index {self.index} could not be opened. "
                "Check that the device is connected and not in use."
            )

        self.start_time = time.time()
        logger.info("Camera %d opened successfully", self.index)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the camera resource on exit."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            logger.info("Camera %d released", self.index)

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
