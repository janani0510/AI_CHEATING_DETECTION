
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


# ---------------------------------------------------
# Base Config
# ---------------------------------------------------
@dataclass(frozen=True)
class BaseConfig:
    """
    Base configuration class with validation helpers.
    """

    def __post_init__(self):
        import dataclasses
        for f in dataclasses.fields(self):
            if getattr(self, f.name)is None:
                raise ValueError(
                    f"{f.name} cannot be None"
                )


# ---------------------------------------------------
# YOLO Configuration
# ---------------------------------------------------
@dataclass(frozen=True)
class YOLOConfig(BaseConfig):

    # COCO class IDs
    PERSON_ID: int = 0
    PHONE_ID: int = 67

    # Model
    MODEL_PATH: str = "yolov8n.pt"

    # Detection
    CONF_THRESHOLD: float = 0.4

    # Performance
    DEVICE: str = "cpu"      # "cpu" or "cuda"
    IMG_SIZE: int = 640

    def __post_init__(self):

        super().__post_init__()

        # Confidence validation
        if not (0.0 < self.CONF_THRESHOLD <= 1.0):
            raise ValueError(
                "CONF_THRESHOLD must be in (0,1]"
            )

        # Image size validation
        if self.IMG_SIZE <= 0:
            raise ValueError(
                "IMG_SIZE must be > 0"
            )

        # Device validation
        if self.DEVICE not in ("cpu", "cuda"):
            raise ValueError(
                "DEVICE must be 'cpu' or 'cuda'"
            )

        # YOLO auto-download support
        if not Path(self.MODEL_PATH).exists():

            if not self.MODEL_PATH.endswith(".pt"):
                raise ValueError(
                    "MODEL_PATH must point to a .pt file"
                )

            print(
                f"[INFO] YOLO model will be auto-downloaded: "
                f"{self.MODEL_PATH}"
            )


# ---------------------------------------------------
# Tracking Configuration
# ---------------------------------------------------
@dataclass(frozen=True)
class TrackingConfig(BaseConfig):

    MAX_DISTANCE: int = 80
    MAX_LOST: int = 10
    MIN_IOU: float = 0.1

    def __post_init__(self):

        super().__post_init__()

        if self.MAX_DISTANCE <= 0:
            raise ValueError(
                "MAX_DISTANCE must be > 0"
            )

        if self.MAX_LOST < 0:
            raise ValueError(
                "MAX_LOST must be >= 0"
            )

        if not (0.0 <= self.MIN_IOU <= 1.0):
            raise ValueError(
                "MIN_IOU must be in [0,1]"
            )


# ---------------------------------------------------
# Head Pose Configuration
# ---------------------------------------------------
@dataclass(frozen=True)
class HeadPoseConfig(BaseConfig):

    HISTORY_SIZE: int = 5

    # Head pose thresholds
    LEFT_THRESHOLD: float = -15.0
    RIGHT_THRESHOLD: float = 15.0
    UP_THRESHOLD: float = -10.0
    DOWN_THRESHOLD: float = 10.0

    def __post_init__(self):

        super().__post_init__()

        if self.HISTORY_SIZE <= 0:
            raise ValueError(
                "HISTORY_SIZE must be > 0"
            )


# ---------------------------------------------------
# Alert System Configuration
# ---------------------------------------------------
@dataclass(frozen=True)
class AlertConfig(BaseConfig):

    COOLDOWN: int = 5

    IOU_THRESHOLD: float = 0.3

    MIN_FRAMES: int = 5

    MIN_BOX_SIZE: int = 30

    SAVE_PATH: str = "data/outputs"

    PHONE_CONF_THRESHOLD: float = 0.5

    def __post_init__(self):

        super().__post_init__()

        if self.COOLDOWN < 0:
            raise ValueError(
                "COOLDOWN must be >= 0"
            )

        if not (0.0 <= self.IOU_THRESHOLD <= 1.0):
            raise ValueError(
                "IOU_THRESHOLD must be in [0,1]"
            )

        if self.MIN_FRAMES <= 0:
            raise ValueError(
                "MIN_FRAMES must be > 0"
            )

        if self.MIN_BOX_SIZE <= 0:
            raise ValueError(
                "MIN_BOX_SIZE must be > 0"
            )

        if not (0.0 <= self.PHONE_CONF_THRESHOLD <= 1.0):
            raise ValueError(
                "PHONE_CONF_THRESHOLD must be in [0,1]"
            )

        # Create save directory
        Path(self.SAVE_PATH).mkdir(
            parents=True,
            exist_ok=True
        )


# ---------------------------------------------------
# Webcam Configuration
# ---------------------------------------------------
@dataclass(frozen=True)
class WebcamConfig(BaseConfig):

    CAM_ID: int = 0

    WIDTH: int = 640
    HEIGHT: int = 480

    FPS: int = 30

    FLIP: bool = True

    BUFFER_SIZE: int = 1

    WARMUP_FRAMES: int = 10

    MAX_RETRIES: int = 3

    RETRY_DELAY: float = 0.5

    def __post_init__(self):

        super().__post_init__()

        if self.WIDTH <= 0 or self.HEIGHT <= 0:
            raise ValueError(
                "Resolution must be > 0"
            )

        if self.FPS <= 0:
            raise ValueError(
                "FPS must be > 0"
            )

        if self.BUFFER_SIZE <= 0:
            raise ValueError(
                "BUFFER_SIZE must be > 0"
            )


# ---------------------------------------------------
# Display Configuration
# ---------------------------------------------------
@dataclass(frozen=True)
class DisplayConfig(BaseConfig):

    SHOW_FPS: bool = True
    SHOW_ALERTS: bool = True
    SHOW_HEAD_POSE: bool = True
    SHOW_TRACKS: bool = True

    WINDOW_NAME: str = "AI Proctoring System"

    ALERT_PANEL_SIZE: Tuple[int, int] = (350, 200)


# ---------------------------------------------------
# Global Configuration
# ---------------------------------------------------
class Config:
    """
    Centralized configuration access.
    """

    YOLO = YOLOConfig()

    TRACKING = TrackingConfig()

    HEAD_POSE = HeadPoseConfig()

    ALERT = AlertConfig()

    WEBCAM = WebcamConfig()

    DISPLAY = DisplayConfig()


# ---------------------------------------------------
# Shortcuts
# ---------------------------------------------------

# YOLO
CONF_THRESHOLD = Config.YOLO.CONF_THRESHOLD
MODEL_PATH = Config.YOLO.MODEL_PATH
DEVICE = Config.YOLO.DEVICE

# Tracking
MAX_DISTANCE = Config.TRACKING.MAX_DISTANCE
MAX_LOST = Config.TRACKING.MAX_LOST
MIN_IOU = Config.TRACKING.MIN_IOU

# Alert
COOLDOWN = Config.ALERT.COOLDOWN
IOU_THRESHOLD = Config.ALERT.IOU_THRESHOLD
MIN_FRAMES = Config.ALERT.MIN_FRAMES

# Webcam
CAM_ID = Config.WEBCAM.CAM_ID
FRAME_WIDTH = Config.WEBCAM.WIDTH
FRAME_HEIGHT = Config.WEBCAM.HEIGHT

# Display
WINDOW_NAME = Config.DISPLAY.WINDOW_NAME


# ---------------------------------------------------
# Export
# ---------------------------------------------------
__all__ = [

    # Main
    "Config",

    # YOLO
    "CONF_THRESHOLD",
    "MODEL_PATH",
    "DEVICE",

    # Tracking
    "MAX_DISTANCE",
    "MAX_LOST",
    "MIN_IOU",

    # Alert
    "COOLDOWN",
    "IOU_THRESHOLD",
    "MIN_FRAMES",

    # Webcam
    "CAM_ID",
    "FRAME_WIDTH",
    "FRAME_HEIGHT",

    # Display
    "WINDOW_NAME",
]

