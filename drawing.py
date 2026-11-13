
import cv2
from typing import List, Tuple, Iterable, Union, Dict, Any

# ---------------------------------------------------
# Constants
# ---------------------------------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 2
PADDING = 5

# Colors (BGR)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
ORANGE = (0, 165, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_MAROON = (0, 0, 128)


# ---------------------------------------------------
# Clamp utility
# ---------------------------------------------------
def _clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))


# ---------------------------------------------------
# Safe bbox parser
# ---------------------------------------------------
def _safe_bbox(bbox):

    try:
        x1, y1, x2, y2 = map(int, bbox)
        return x1, y1, x2, y2
    except Exception:
        return None


# ---------------------------------------------------
# Draw text with transparent background
# ---------------------------------------------------
def draw_text_with_bg(
    frame,
    text: str,
    x: int,
    y: int,
    text_color: Tuple[int, int, int],
    bg_color: Tuple[int, int, int] = BLACK,
    alpha: float = 0.6
) -> None:

    if frame is None or text is None:
        return

    try:

        h, w = frame.shape[:2]

        text = str(text)

        (tw, th), _ = cv2.getTextSize(
            text,
            FONT,
            FONT_SCALE,
            THICKNESS
        )

        x = _clamp(x, 0, max(0, w - tw - PADDING))
        y = _clamp(y, th + PADDING, max(th + PADDING, h - PADDING))

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (x, y - th - PADDING),
            (x + tw + PADDING, y + PADDING),
            bg_color,
            -1
        )

        cv2.addWeighted(
            overlay,
            alpha,
            frame,
            1 - alpha,
            0,
            frame
        )

        cv2.putText(
            frame,
            text,
            (x + 2, y),
            FONT,
            FONT_SCALE,
            text_color,
            THICKNESS,
            cv2.LINE_AA
        )

    except Exception:
        pass


# ---------------------------------------------------
# Draw Bounding Box
# ---------------------------------------------------
def draw_box(
    frame,
    bbox: Iterable[int],
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 2
) -> None:

    if frame is None:
        return

    parsed = _safe_bbox(bbox)

    if parsed is None:
        return

    x1, y1, x2, y2 = parsed

    try:

        h, w = frame.shape[:2]

        x1 = _clamp(x1, 0, w - 1)
        y1 = _clamp(y1, 0, h - 1)
        x2 = _clamp(x2, 0, w - 1)
        y2 = _clamp(y2, 0, h - 1)

        if x2 <= x1 or y2 <= y1:
            return

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

        # Label
        if label:
            draw_text_with_bg(
                frame,
                label,
                x1,
                y1 - 10,
                color
            )

    except Exception:
        pass


# ---------------------------------------------------
# Draw Person
# ---------------------------------------------------
def draw_person(
    frame,
    bbox,
    pid: int,
    active: bool = True
) -> None:

    color = GREEN if active else ORANGE

    label = f"ID {pid}"

    if not active:
        label += " LOST"

    draw_box(
        frame,
        bbox,
        label,
        color
    )


# ---------------------------------------------------
# Draw Phone
# ---------------------------------------------------
def draw_phone(
    frame,
    bbox,
    confidence: float = None
) -> None:

    label = "Phone"

    if confidence is not None:
        label += f" {confidence:.2f}"

    draw_box(
        frame,
        bbox,
        label,
        RED
    )


# ---------------------------------------------------
# Draw Head Pose
# ---------------------------------------------------
def draw_head_pose(
    frame,
    bbox,
    direction: str
) -> None:

    parsed = _safe_bbox(bbox)

    if parsed is None:
        return

    x1, y1, _, _ = parsed

    direction = str(direction).title()

    # Dynamic colors
    if direction == "Forward":
        color = GREEN

    elif direction == "No Face":
        color = RED

    else:
        color = BLUE

    draw_text_with_bg(
        frame,
        direction,
        x1,
        y1 - 30,
        color
    )


# ---------------------------------------------------
# Draw Alerts Panel
# ---------------------------------------------------
def draw_alerts(
    frame,
    alerts: List[Union[str, Dict[str, Any]]],
    max_alerts: int = 5
) -> None:

    if frame is None:
        return

    try:

        h, w = frame.shape[:2]

        max_alerts = max(1, min(max_alerts, h // 35))

        visible_alerts = alerts[-max_alerts:]

        panel_width = 350
        panel_height = 35 * len(visible_alerts) + 10

        overlay = frame.copy()

        # Semi-transparent panel
        cv2.rectangle(
            overlay,
            (5, 5),
            (panel_width, panel_height),
            BLACK,
            -1
        )

        cv2.addWeighted(
            overlay,
            0.4,
            frame,
            0.6,
            0,
            frame
        )

        y = 35

        for alert in visible_alerts:

            # ---------------------------------------
            # Dictionary alerts
            # ---------------------------------------
            if isinstance(alert, dict):

                msg = str(
                    alert.get("message", "Unknown Alert")
                )

                level = str(
                    alert.get("level", "MEDIUM")
                ).upper()

                pid = alert.get("id", "?")

                msg = f"[ID {pid}] {msg}"

                # Dynamic colors
                if level == "HIGH":
                    color = RED

                elif level == "MEDIUM":
                    color = ORANGE

                else:
                    color = WHITE

            # ---------------------------------------
            # String alerts
            # ---------------------------------------
            else:
                msg = str(alert)
                color = ORANGE

            draw_text_with_bg(
                frame,
                msg,
                15,
                y,
                color
            )

            y += 30

    except Exception:
        pass


# ---------------------------------------------------
# Draw FPS
# ---------------------------------------------------
def draw_fps(
    frame,
    fps: float
) -> None:

    if frame is None:
        return

    try:

        text = f"FPS: {fps:.2f}"

        # FPS color logic
        if fps >= 25:
            color = GREEN

        elif fps >= 15:
            color = ORANGE

        else:
            color = RED

        draw_text_with_bg(
            frame,
            text,
            10,
            frame.shape[0] - 15,
            color
        )

    except Exception:
        pass

