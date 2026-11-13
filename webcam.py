
import cv2
import time
import sys
from typing import Tuple, Optional
import numpy as np


class Webcam:
    """
    Professional Cross-Platform Webcam Handler

    Features:
    - Cross-platform backend fallback
    - Automatic recovery
    - Safe initialization
    - FPS estimation
    - Frame validation
    - Warmup stabilization
    - Safe cleanup
    - Retry handling
    - Optional frame flipping
    """

    def __init__(
        self,
        cam_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        backend: Optional[int] = None,
        warmup_frames: int = 10,
        retry_delay: float = 0.5,
        max_retries: int = 3,
        flip: bool = True,
        buffer_size: int = 1
    ):

        self.cam_id = cam_id
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend

        self.warmup_frames = warmup_frames
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.flip = flip
        self.buffer_size = buffer_size

        self.cap: Optional[cv2.VideoCapture] = None

        # FPS calculation
        self.prev_time = time.time()
        self.current_fps = 0.0

        # Camera state
        self.backend_used = None

        # Initialize
        self._init_camera_with_fallback()
        self._warmup()

    # ---------------------------------------------------
    # Backend Selection
    # ---------------------------------------------------
    def _get_backends(self):

        # User-specified backend
        if self.backend is not None:
            return [self.backend]

        # Windows
        if sys.platform.startswith("win"):
            return [
                cv2.CAP_DSHOW,
                cv2.CAP_MSMF,
                cv2.CAP_ANY
            ]

        # Linux
        elif sys.platform.startswith("linux"):
            return [
                cv2.CAP_V4L2,
                cv2.CAP_ANY
            ]

        # macOS
        else:
            return [
                cv2.CAP_AVFOUNDATION,
                cv2.CAP_ANY
            ]

    # ---------------------------------------------------
    # Camera Initialization
    # ---------------------------------------------------
    def _init_camera_with_fallback(self):

        self.release()

        for backend in self._get_backends():

            try:

                cap = cv2.VideoCapture(self.cam_id, backend)

                if not cap.isOpened():
                    continue

                self._apply_settings(cap)

                # Test frame read
                ret, frame = cap.read()

                if not ret or frame is None:
                    cap.release()
                    continue

                self.cap = cap
                self.backend_used = backend

                print(f"[INFO] Webcam opened successfully")
                print(f"[INFO] Backend: {backend}")

                return

            except Exception as e:
                print(f"[WARNING] Backend failed: {backend} -> {e}")

        raise RuntimeError(
            "❌ Failed to initialize webcam with all backends"
        )

    # ---------------------------------------------------
    # Apply Camera Settings
    # ---------------------------------------------------
    def _apply_settings(self, cap: cv2.VideoCapture):

        try:

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        except Exception as e:
            print(f"[WARNING] Failed applying settings: {e}")

    # ---------------------------------------------------
    # Warmup
    # ---------------------------------------------------
    def _warmup(self):

        if self.cap is None:
            return

        print("[INFO] Warming up camera...")

        for _ in range(self.warmup_frames):
            self.cap.read()
            time.sleep(0.02)

    # ---------------------------------------------------
    # Read Frame
    # ---------------------------------------------------
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:

        if self.cap is None:
            return False, None

        for attempt in range(self.max_retries + 1):

            try:

                ret, frame = self.cap.read()

                # Validation
                if not ret or frame is None:
                    raise ValueError("Empty frame")

                # Validate shape
                if frame.size == 0:
                    raise ValueError("Invalid frame size")

                # Optional mirror flip
                if self.flip:
                    frame = cv2.flip(frame, 1)

                # FPS update
                self._update_fps()

                return True, frame

            except Exception as e:

                print(
                    f"[WARNING] Frame read failed "
                    f"(attempt {attempt + 1}): {e}"
                )

                time.sleep(self.retry_delay)

        # Recovery
        print("[INFO] Attempting camera recovery...")

        if self._recover():

            try:

                ret, frame = self.cap.read()

                if ret and frame is not None:

                    if self.flip:
                        frame = cv2.flip(frame, 1)

                    self._update_fps()

                    return True, frame

            except Exception:
                pass

        return False, None

    # ---------------------------------------------------
    # FPS Calculation
    # ---------------------------------------------------
    def _update_fps(self):

        current_time = time.time()

        delta = current_time - self.prev_time

        if delta > 0:
            self.current_fps = 1.0 / delta

        self.prev_time = current_time

    # ---------------------------------------------------
    # Get FPS
    # ---------------------------------------------------
    def get_fps(self) -> float:
        return round(self.current_fps, 2)

    # ---------------------------------------------------
    # Camera Recovery
    # ---------------------------------------------------
    def _recover(self) -> bool:

        try:

            self.release()

            time.sleep(self.retry_delay)

            self._init_camera_with_fallback()

            print("[INFO] Camera recovery successful")

            return True

        except Exception as e:

            print(f"[ERROR] Camera recovery failed: {e}")

            return False

    # ---------------------------------------------------
    # Status Check
    # ---------------------------------------------------
    def is_opened(self) -> bool:

        return (
            self.cap is not None and
            self.cap.isOpened()
        )

    # ---------------------------------------------------
    # Get Resolution
    # ---------------------------------------------------
    def get_resolution(self) -> Tuple[int, int]:
        return self.width, self.height

    # ---------------------------------------------------
    # Release Resources
    # ---------------------------------------------------
    def release(self):

        try:

            if self.cap is not None:

                if self.cap.isOpened():
                    self.cap.release()

                self.cap = None

            cv2.destroyAllWindows()

        except Exception as e:
            print(f"[WARNING] Cleanup failed: {e}")

    # ---------------------------------------------------
    # Context Manager
    # ---------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    # ---------------------------------------------------
    # Destructor Safety
    # ---------------------------------------------------
    def __del__(self):

        try:
            self.release()
        except Exception:
            pass

