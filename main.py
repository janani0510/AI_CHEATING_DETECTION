
import cv2
import time
import math
from typing import Tuple

from person_phone_detector import PersonPhoneDetector
from tracking import Tracker
from head_pose import HeadPoseEstimator
from alert import AlertSystem
from webcam import Webcam
from drawing import draw_person, draw_phone, draw_head_pose, draw_alerts, DARK_MAROON
from config import Config


# ---------------------------
# Safe crop
# ---------------------------
def safe_crop(frame, bbox):

    if frame is None or bbox is None:
        return None

    try:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
    except Exception:
        return None

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


# ---------------------------
# FPS calculation
# ---------------------------
def update_fps(prev_time: float, fps: float) -> Tuple[float, float]:

    curr_time = time.time()
    dt = curr_time - prev_time

    if dt > 0:
        instant_fps = 1.0 / dt
        fps = 0.9 * fps + 0.1 * instant_fps

    return curr_time, fps


# ---------------------------
# Screen Gaze Direction
# ---------------------------
def get_screen_direction(x, y, threshold=1.5):

    if x is None or y is None:
        return "Waiting..."

    horizontal = ""
    vertical = ""

    if x < -threshold:
        horizontal = "RIGHT"
    elif x > threshold:
        horizontal = "LEFT"

    if y < -threshold:
        vertical = "TOP"
    elif y > threshold:
        vertical = "BOTTOM"

    if vertical and horizontal:
        return f"{vertical} {horizontal}"

    if vertical:
        return "UP" if vertical == "TOP" else "DOWN"

    if horizontal:
        return horizontal

    return "CENTER"


# ---------------------------
# Screen Gaze Display
# ---------------------------
def draw_screen_gaze(frame, x, y, person_id):

    if x is None or y is None:
        return

    direction = get_screen_direction(x, y)

    # Distance from screen center
    total_distance = math.sqrt((x ** 2) + (y ** 2))

    # Maroon color (BGR)
    MAROON = (0, 0, 128)

    # Display position
    start_x = 10
    start_y = 175 + (person_id * 95)

    line_gap = 25

    display_lines = [
        f"ID {person_id} - {direction}",
        f"X: {x:+.2f} cm",
        f"Y: {y:+.2f} cm",
        f"Distance: {total_distance:.2f} cm"
    ]

    for index, text in enumerate(display_lines):

        cv2.putText(
            frame,
            text,
            (start_x, start_y + (index * line_gap)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            MAROON,
            2,
            cv2.LINE_AA
        )


# ---------------------------
# Main
# ---------------------------
def main():

    cam = None
    headpose = None

    calibration_person_ids = set()
    screen_calibration_person_ids = set()

    try:

        cam = Webcam()

        detector = PersonPhoneDetector()

        tracker = Tracker(
            max_distance=Config.TRACKING.MAX_DISTANCE,
            max_lost=Config.TRACKING.MAX_LOST,
            min_iou=Config.TRACKING.MIN_IOU
        )

        headpose = HeadPoseEstimator()

        alert_system = AlertSystem(
            save_path=Config.ALERT.SAVE_PATH,
            cooldown=Config.ALERT.COOLDOWN,
            iou_threshold=Config.ALERT.IOU_THRESHOLD,
            min_frames=Config.ALERT.MIN_FRAMES,
            min_box_size=Config.ALERT.MIN_BOX_SIZE
        )

    except Exception as e:

        print(f"[ERROR] Initialization failed: {e}")

        if headpose is not None:
            headpose.close()

        if cam is not None:
            cam.release()

        cv2.destroyAllWindows()
        return

    prev_time = time.time()
    fps = 30.0

    try:

        while True:

            ret, frame = cam.read()

            if not ret or frame is None:
                print("[ERROR] Camera error. Exiting...")
                break

            frame = cv2.resize(frame, (640, 480))

            # -------- DETECTION --------

            detections = detector.detect(frame)

            persons = detections.get("persons", [])
            phones = detections.get("phones", [])

            # -------- TRACKING --------

            tracked_persons = tracker.update(persons)

            # -------- HEAD POSE + EYE GAZE --------

            for person in tracked_persons:

                bbox = person.get("bbox")

                if not bbox:

                    person["head_pose"] = "No Face"
                    person["eye_gaze"] = "No Face"
                    person["gaze_x_cm"] = None
                    person["gaze_y_cm"] = None

                    continue

                crop = safe_crop(frame, bbox)

                if crop is None:

                    person["head_pose"] = "No Face"
                    person["eye_gaze"] = "No Face"
                    person["gaze_x_cm"] = None
                    person["gaze_y_cm"] = None

                    continue

                h = crop.shape[0]

                face_region = crop[0:int(h * 0.6), :]

                if face_region is None or face_region.size == 0:

                    person["head_pose"] = "No Face"
                    person["eye_gaze"] = "No Face"
                    person["gaze_x_cm"] = None
                    person["gaze_y_cm"] = None

                    continue

                result = headpose.analyze(
                    face_region,
                    person_id=person["id"]
                )

                print(
                    f"[SCREEN DEBUG] ID: {person['id']} | "
                    f"Screen Gaze: {result.get('screen_gaze')} | "
                    f"X: {result.get('gaze_x_cm')} | "
                    f"Y: {result.get('gaze_y_cm')}"
                )

                person["head_pose"] = result["head_pose"]
                person["head_confidence"] = result["head_confidence"]

                person["eye_gaze"] = result["eye_gaze"]
                person["gaze_confidence"] = result["gaze_confidence"]

                person["gaze_x_cm"] = result.get("gaze_x_cm")
                person["gaze_y_cm"] = result.get("gaze_y_cm")

            # -------- ALERTS --------

            alerts = alert_system.check_alerts(
                tracked_persons,
                phones,
                frame
            )

            # -------- DRAW --------

            for person in tracked_persons:

                if "bbox" not in person:
                    continue

                bbox = person["bbox"]
                pid = person["id"]

                head_direction = person.get(
                    "head_pose", "N/A"
                )

                gaze_direction = person.get(
                    "eye_gaze", "N/A"
                )

                draw_person(
                    frame,
                    bbox,
                    pid,
                    person.get("active", True)
                )

                draw_head_pose(
                    frame,
                    bbox,
                    head_direction
                )

                x1, y1, x2, y2 = map(int, bbox)

                # Eye gaze display
                cv2.putText(
                    frame,
                    f"Eye: {gaze_direction}",
                    (x1 + 5, max(20, y2 - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                # Direction + X/Y + Total distance
                gaze_x = person.get("gaze_x_cm")
                gaze_y = person.get("gaze_y_cm")

                draw_screen_gaze(
                    frame,
                    gaze_x,
                    gaze_y,
                    pid
                )

            # -------- PHONE DISPLAY --------

            for phone in phones:

                if "bbox" in phone:
                    draw_phone(frame, phone["bbox"])

            # -------- ALERT DISPLAY --------

            draw_alerts(frame, alerts)

            # -------- CALIBRATION INSTRUCTIONS --------

            cv2.putText(
                frame,
                "Eye: L/C/R | Eye Save: S | Screen: 1-5 | Save: P",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"Eye IDs: {sorted(calibration_person_ids)}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"Screen IDs: {sorted(screen_calibration_person_ids)}",
                (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

            # -------- FPS --------

            prev_time, fps = update_fps(prev_time, fps)

            if Config.DISPLAY.SHOW_FPS:

                cv2.putText(
                    frame,
                    f"FPS: {int(fps)}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

            # -------- DISPLAY --------

            cv2.imshow(
                "AI Exam Cheating Detection",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            # -------- EYE CALIBRATION CONTROLS --------

            if key in (
                ord("l"), ord("L"),
                ord("c"), ord("C"),
                ord("r"), ord("R")
            ):

                target_map = {
                    ord("l"): -10,
                    ord("L"): -10,
                    ord("c"): 0,
                    ord("C"): 0,
                    ord("r"): 10,
                    ord("R"): 10
                }

                target_angle = target_map[key]

                eligible_ids = []

                for person in tracked_persons:

                    pid = person.get("id")

                    if pid is None:
                        continue

                    if person.get("head_pose") != "Forward":
                        continue

                    if person.get("eye_gaze") == "No Face":
                        continue

                    eligible_ids.append(pid)

                if not eligible_ids:

                    print(
                        "[EYE CALIBRATION] No forward-facing "
                        "person detected. Try again."
                    )

                else:

                    for pid in eligible_ids:

                        headpose.set_eye_calibration_target(
                            pid,
                            target_angle
                        )

                        calibration_person_ids.add(pid)

            # -------- SCREEN CALIBRATION TARGETS --------

            elif key in (
                ord("1"), ord("2"), ord("3"),
                ord("4"), ord("5")
            ):

                screen_target_map = {
                    ord("1"): "TL",
                    ord("2"): "TR",
                    ord("3"): "CENTER",
                    ord("4"): "BL",
                    ord("5"): "BR"
                }

                target_name = screen_target_map[key]

                eligible_ids = []

                for person in tracked_persons:

                    pid = person.get("id")

                    if pid is None:
                        continue

                    if person.get("head_pose") != "Forward":
                        continue

                    if person.get("eye_gaze") == "No Face":
                        continue

                    eligible_ids.append(pid)

                if not eligible_ids:

                    print(
                        "[SCREEN CALIBRATION] No forward-facing "
                        "person detected. Try again."
                    )

                else:

                    for pid in eligible_ids:

                        headpose.set_screen_calibration_target(
                            pid,
                            target_name
                        )

                        screen_calibration_person_ids.add(pid)

                    print(
                        f"[SCREEN CALIBRATION] Target {target_name} "
                        f"selected for IDs: {eligible_ids}"
                    )

            # -------- FINALIZE EYE CALIBRATION --------

            elif key in (ord("s"), ord("S")):

                if not calibration_person_ids:

                    print(
                        "[EYE CALIBRATION] No calibration IDs found."
                    )

                else:

                    for pid in sorted(calibration_person_ids):

                        success = headpose.finalize_eye_calibration(
                            pid
                        )

                        if success:
                            print(
                                f"[INFO] Eye calibration saved for ID {pid}"
                            )

            # -------- FINALIZE SCREEN CALIBRATION --------

            elif key in (ord("p"), ord("P")):

                if not screen_calibration_person_ids:

                    print(
                        "[SCREEN CALIBRATION] No calibration IDs found."
                    )

                else:

                    for pid in sorted(screen_calibration_person_ids):

                        success = headpose.finalize_screen_calibration(
                            pid
                        )

                        if success:
                            print(
                                f"[INFO] Screen calibration saved for ID {pid}"
                            )

            # -------- EXIT --------

            elif key in (27, ord("q"), ord("Q")):

                print("[INFO] Exiting...")
                break

    except KeyboardInterrupt:

        print("[INFO] Interrupted")

    except Exception:

        import traceback
        traceback.print_exc()

    finally:

        if headpose is not None:
            headpose.close()

        if cam is not None:
            cam.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
