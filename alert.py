
import cv2
import os
import re
import time
from datetime import datetime
from typing import List, Dict, Any


class AlertSystem:
    """
    AI Exam Cheating Detection Alert System

    Features:
    - Head pose monitoring
    - Eye gaze monitoring
    - Phone usage detection
    - Face visibility alerts
    - Temporal smoothing
    - Safe bounding box handling
    - Full-frame screenshot evidence
    - Person cropped evidence
    - Date-wise evidence storage
    - Cooldown system
    - False positive reduction
    """

    EPS = 1e-6

    def __init__(
        self,
        save_path: str = "data/outputs",
        cooldown: int = 5,
        iou_threshold: float = 0.3,
        min_frames: int = 5,
        min_box_size: int = 30
    ):

        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        self.cooldown = cooldown
        self.iou_threshold = iou_threshold
        self.min_frames = min_frames
        self.min_box_size = min_box_size

        self.last_saved: Dict[int, float] = {}
        self.violation_frames: Dict[tuple, int] = {}

    # ---------------------------------------------------
    # IoU
    # ---------------------------------------------------
    @staticmethod
    def iou(boxA, boxB) -> float:

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)

        inter_area = inter_w * inter_h

        if inter_area <= 0:
            return 0.0

        areaA = max(0, boxA[2] - boxA[0]) * max(
            0, boxA[3] - boxA[1]
        )

        areaB = max(0, boxB[2] - boxB[0]) * max(
            0, boxB[3] - boxB[1]
        )

        union = areaA + areaB - inter_area

        return inter_area / (union + AlertSystem.EPS)

    # ---------------------------------------------------
    # Main Alert Logic
    # ---------------------------------------------------
    def check_alerts(
        self,
        tracked_persons: List[Dict[str, Any]],
        phones: List[Dict[str, Any]],
        frame
    ) -> List[Dict[str, Any]]:

        if frame is None:
            return []

        try:
            h, w = frame.shape[:2]
        except Exception:
            return []

        alerts = []
        current_time = time.time()

        for person in tracked_persons:

            try:

                if not person.get("active", True):
                    continue

                pid = person.get("id")
                bbox = person.get("bbox")

                if pid is None or bbox is None:
                    continue

                try:
                    x1, y1, x2, y2 = map(int, bbox)
                except Exception:
                    continue

                # Safe bbox clipping
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w - 1, x2)
                y2 = min(h - 1, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                if (
                    (x2 - x1) < self.min_box_size or
                    (y2 - y1) < self.min_box_size
                ):
                    continue

                pbox = (x1, y1, x2, y2)

                # ---------------------------------------
                # Head Pose Detection
                # ---------------------------------------
                direction = str(
                    person.get("head_pose", "forward")
                ).strip().lower()

                # ---------------------------------------
                # Eye Gaze Detection
                # ---------------------------------------
                gaze_direction = str(
                    person.get("eye_gaze", "center")
                ).strip().lower()

                # Normalize gaze labels
                gaze_direction = gaze_direction.replace(
                    "looking ", ""
                )

                gaze_direction = gaze_direction.replace(
                    "eye ", ""
                )

                # ---------------------------------------
                # Phone Detection
                # ---------------------------------------
                phone_detected = False

                for phone in phones:

                    ph_box = phone.get("bbox")

                    if ph_box is None or len(ph_box) != 4:
                        continue

                    try:
                        px1, py1, px2, py2 = map(int, ph_box)
                    except Exception:
                        continue

                    if px2 <= px1 or py2 <= py1:
                        continue

                    ph_box = (px1, py1, px2, py2)

                    overlap = self.iou(pbox, ph_box)

                    inside_person = (
                        px1 >= x1 and py1 >= y1 and
                        px2 <= x2 and py2 <= y2
                    )

                    if (
                        overlap > self.iou_threshold
                        or inside_person
                    ):
                        phone_detected = True
                        break

                # ---------------------------------------
                # Alert Decision
                # ---------------------------------------
                alert_msg = None
                level = None

                # Highest priority: Phone + head movement
                if phone_detected and direction != "forward":

                    alert_msg = (
                        f"Phone + {direction.title()}"
                    )

                    level = "HIGH"

                # Phone detected
                elif phone_detected:

                    alert_msg = "Phone Detected"
                    level = "HIGH"

                # Face not visible
                elif direction == "no face":

                    alert_msg = "Face Not Visible"
                    level = "MEDIUM"

                # Head pose violation
                elif direction != "forward":

                    alert_msg = direction.title()
                    level = "MEDIUM"

                # Eye gaze violation
                elif gaze_direction not in (
                    "center",
                    "forward",
                    "no face",
                    "n/a",
                    ""
                ):

                    alert_msg = (
                        f"Eye Looking {gaze_direction.title()}"
                    )

                    level = "MEDIUM"

                # No violation
                else:

                    # Clear previous violation counters
                    for key in list(
                        self.violation_frames.keys()
                    ):

                        if key[0] == pid:
                            del self.violation_frames[key]

                    continue

                # ---------------------------------------
                # Temporal Smoothing
                # ---------------------------------------
                key = (pid, alert_msg)

                # Remove old counters for different alerts
                for old_key in list(
                    self.violation_frames.keys()
                ):

                    if old_key[0] == pid and old_key != key:
                        del self.violation_frames[old_key]

                self.violation_frames[key] = (
                    self.violation_frames.get(key, 0) + 1
                )

                if (
                    self.violation_frames[key]
                    < self.min_frames
                ):
                    continue

                # ---------------------------------------
                # Alert Object
                # ---------------------------------------
                alert = {
                    "id": pid,
                    "message": alert_msg,
                    "level": level,
                    "direction": direction,
                    "eye_gaze": gaze_direction,
                    "phone": phone_detected,
                    "timestamp": current_time
                }

                # ---------------------------------------
                # Save Evidence After Cooldown
                # ---------------------------------------
                last_time = self.last_saved.get(
                    pid, 0.0
                )

                if (
                    current_time - last_time
                    > self.cooldown
                ):

                    evidence = self.save_evidence(
                        frame=frame,
                        bbox=pbox,
                        pid=pid,
                        reason=alert_msg
                    )

                    if evidence:

                        alert["evidence"] = evidence

                        self.last_saved[pid] = (
                            current_time
                        )

                alerts.append(alert)

            except Exception as e:

                print(
                    f"[AlertSystem ERROR] {e}"
                )

        return alerts

    # ---------------------------------------------------
    # Save Full Screenshot + Person Crop
    # ---------------------------------------------------
    def save_evidence(
        self,
        frame,
        bbox,
        pid,
        reason
    ):

        try:

            if frame is None:
                return None

            h, w = frame.shape[:2]

            x1, y1, x2, y2 = map(int, bbox)

            # Safe bbox clipping
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return None

            # Person crop
            crop = frame[
                y1:y2,
                x1:x2
            ].copy()

            if crop.size == 0:
                return None

            # Date-wise folders
            now = datetime.now()

            date_folder = now.strftime(
                "%Y-%m-%d"
            )

            full_folder = os.path.join(
                self.save_path,
                date_folder,
                "full_frames"
            )

            crop_folder = os.path.join(
                self.save_path,
                date_folder,
                "person_crops"
            )

            os.makedirs(
                full_folder,
                exist_ok=True
            )

            os.makedirs(
                crop_folder,
                exist_ok=True
            )

            # Safe filename
            safe_reason = re.sub(
                r"[^a-zA-Z0-9_]",
                "",
                reason.replace(" ", "_")
            )

            timestamp = now.strftime(
                "%Y%m%d_%H%M%S_%f"
            )

            filename = (
                f"ID{pid}_{safe_reason}_{timestamp}.jpg"
            )

            # -------------------------------------------
            # Save Full Webcam Screenshot
            # -------------------------------------------
            full_frame = frame.copy()

            cv2.rectangle(
                full_frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            full_path = os.path.join(
                full_folder,
                filename
            )

            full_saved = cv2.imwrite(
                full_path,
                full_frame
            )

            # -------------------------------------------
            # Save Person Crop
            # -------------------------------------------
            crop_h, crop_w = crop.shape[:2]

            cv2.rectangle(
                crop,
                (0, 0),
                (crop_w - 1, crop_h - 1),
                (0, 0, 255),
                2
            )

            crop_path = os.path.join(
                crop_folder,
                filename
            )

            crop_saved = cv2.imwrite(
                crop_path,
                crop
            )

            if not full_saved or not crop_saved:

                print(
                    "[SaveEvidence ERROR] Image save failed"
                )

                return None

            print("[EVIDENCE SAVED]")

            print(
                f"Full screenshot: {full_path}"
            )

            print(
                f"Person crop: {crop_path}"
            )

            return {
                "full_frame": full_path,
                "person_crop": crop_path
            }

        except Exception as e:

            print(
                f"[SaveEvidence ERROR] {e}"
            )

            return None
