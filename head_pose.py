
import json
import os
import cv2
import mediapipe as mp
import numpy as np

from collections import Counter, deque
from typing import Optional


class HeadPoseEstimator:

    def __init__(
        self,
        history_size=5,
        calibration_frames=20,
        yaw_threshold=12.0,
        pitch_threshold=12.0,
        roll_threshold=12.0
    ):

        self.mp_face = mp.solutions.face_mesh

        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.history_size = history_size
        self.calibration_frames = calibration_frames

        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.roll_threshold = roll_threshold

        self.head_history = {}
        self.gaze_history = {}

        self.calibration_data = {}
        self.baselines = {}

        # Existing eye gaze calibration
        self.eye_calibration = {}
        self.eye_calibration_target = {}
        self.eye_gaze_history = {}
        self.eye_calibration_results = {}

        # ---------------------------------------
        # Laptop screen dimensions
        # ---------------------------------------

        self.screen_width_cm = 30.0
        self.screen_height_cm = 19.0

        self.screen_center_y = self.screen_height_cm / 2.0

        # Center-origin coordinate system
        # X: left negative, right positive
        # Y: up negative, down positive
        # Center: X=0, Y=0

        self.screen_calibration_points = {
            "TL": (-15.0, -9.5),
            "TR": (15.0, -9.5),
            "CENTER": (0.0, 0.0),
            "BL": (-15.0, 9.5),
            "BR": (15.0, 9.5)
        }

        self.screen_calibration_data = {}
        self.screen_calibration_target = {}
        self.screen_calibration_models = {}
        self.latest_eye_position = {}

        # ---------------------------------------
        # Coordinate stability settings
        # ---------------------------------------

        self.screen_gaze_history = {}

        # Lower alpha = smoother output
        self.screen_smoothing_alpha = 0.15

        # Small general dead zone
        self.screen_dead_zone_cm = 0.3

        # Center lock threshold
        # Both X and Y must be inside this radius
        # to lock the coordinate at exactly zero.
        self.screen_center_lock_cm = 1.5

        # Screen calibration save file
        self.screen_calibration_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "screen_calibration.json"
        )

        # Automatically load compatible calibration
        self.load_screen_calibration()

        self.model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1)
        ], dtype=np.float64)

    # ---------------------------------------
    # Circular angle difference
    # ---------------------------------------

    @staticmethod
    def _angle_difference(angle, baseline):

        return (
            (angle - baseline + 180.0) % 360.0
        ) - 180.0

    # ---------------------------------------
    # Temporal smoothing
    # ---------------------------------------

    def _smooth(self, history_dict, person_id, direction):

        if person_id not in history_dict:
            history_dict[person_id] = deque(
                maxlen=self.history_size
            )

        if direction == "No Face":
            return direction

        history_dict[person_id].append(direction)

        return Counter(
            history_dict[person_id]
        ).most_common(1)[0][0]

    # ---------------------------------------
    # Get FaceMesh landmarks
    # ---------------------------------------

    def _get_landmarks(self, frame):

        if frame is None or frame.size == 0:
            return None

        try:
            img_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = self.face_mesh.process(img_rgb)

            if not results.multi_face_landmarks:
                return None

            landmarks = results.multi_face_landmarks[0].landmark

            if len(landmarks) < 478:
                return None

            return landmarks

        except Exception as e:
            print(f"[FaceMesh ERROR] {e}")
            return None

    # ---------------------------------------
    # Head Pose Estimation
    # ---------------------------------------

    def _estimate_head_angles(self, landmarks, frame_shape):

        height, width = frame_shape[:2]

        landmark_ids = [1, 152, 33, 263, 61, 291]

        image_points = np.array([
            (
                landmarks[index].x * width,
                landmarks[index].y * height
            )
            for index in landmark_ids
        ], dtype=np.float64)

        focal_length = float(max(width, height))

        camera_matrix = np.array([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        distortion_coefficients = np.zeros(
            (4, 1),
            dtype=np.float64
        )

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            distortion_coefficients,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(
            rotation_vector
        )

        angles, _, _, _, _, _ = cv2.RQDecomp3x3(
            rotation_matrix
        )

        return {
            "pitch": float(angles[0]),
            "yaw": float(angles[1]),
            "roll": float(angles[2])
        }

    # ---------------------------------------
    # Head Calibration
    # ---------------------------------------

    @staticmethod
    def _average_angle(values):

        radians = np.radians(values)

        sin_mean = np.mean(np.sin(radians))
        cos_mean = np.mean(np.cos(radians))

        return float(
            np.degrees(
                np.arctan2(sin_mean, cos_mean)
            )
        )

    def _calibrate(self, person_id, angles):

        if person_id in self.baselines:
            return True

        if person_id not in self.calibration_data:
            self.calibration_data[person_id] = []

        self.calibration_data[person_id].append(angles)

        samples = self.calibration_data[person_id]

        if len(samples) >= self.calibration_frames:

            self.baselines[person_id] = {
                "pitch": self._average_angle([
                    item["pitch"] for item in samples
                ]),

                "yaw": self._average_angle([
                    item["yaw"] for item in samples
                ]),

                "roll": self._average_angle([
                    item["roll"] for item in samples
                ])
            }

            print(
                f"[INFO] Face calibration completed for ID {person_id}"
            )

            return True

        return False

    # ---------------------------------------
    # Head Pose Classification
    # ---------------------------------------

    def _calculate_head_pose(
        self,
        landmarks,
        frame_shape,
        person_id
    ):

        angles = self._estimate_head_angles(
            landmarks,
            frame_shape
        )

        if angles is None:
            return "No Face", 0.0

        calibrated = self._calibrate(
            person_id,
            angles
        )

        if not calibrated:
            return "Forward", 0.0

        baseline = self.baselines[person_id]

        pitch_diff = self._angle_difference(
            angles["pitch"],
            baseline["pitch"]
        )

        yaw_diff = self._angle_difference(
            angles["yaw"],
            baseline["yaw"]
        )

        roll_diff = self._angle_difference(
            angles["roll"],
            baseline["roll"]
        )

        print(
            f"[HEAD DEBUG] "
            f"Pitch Diff: {pitch_diff:.2f} | "
            f"Yaw Diff: {yaw_diff:.2f} | "
            f"Roll Diff: {roll_diff:.2f}"
        )

        if abs(roll_diff) >= self.roll_threshold:

            direction = (
                "Tilted Right"
                if roll_diff > 0
                else "Tilted Left"
            )

            amount = abs(roll_diff) / 30.0

        elif abs(yaw_diff) >= self.yaw_threshold:

            direction = (
                "Looking Right"
                if yaw_diff > 0
                else "Looking Left"
            )

            amount = abs(yaw_diff) / 35.0

        elif abs(pitch_diff) >= self.pitch_threshold:

            direction = (
                "Looking Down"
                if pitch_diff > 0
                else "Looking Up"
            )

            amount = abs(pitch_diff) / 35.0

        else:

            direction = "Forward"

            amount = max(
                abs(yaw_diff) / self.yaw_threshold,
                abs(pitch_diff) / self.pitch_threshold,
                abs(roll_diff) / self.roll_threshold
            )

        confidence = min(
            1.0,
            max(0.0, amount)
        )

        return direction, round(confidence, 2)

    # ---------------------------------------
    # Existing Eye Calibration
    # ---------------------------------------

    def set_eye_calibration_target(self, person_id, target_angle):

        if target_angle not in (-10, 0, 10):
            raise ValueError("Target must be -10, 0 or 10")

        if person_id not in self.eye_calibration:
            self.eye_calibration[person_id] = {
                -10: [],
                0: [],
                10: []
            }

        self.eye_calibration_target[person_id] = target_angle

        print(
            f"[EYE CALIBRATION] ID {person_id}: "
            f"Target {target_angle} degrees"
        )

    def _collect_eye_calibration_sample(
        self,
        person_id,
        horizontal_ratio
    ):

        if person_id not in self.eye_calibration:
            return

        if person_id in self.eye_calibration_results:
            return

        target = self.eye_calibration_target.get(person_id)

        if target not in (-10, 0, 10):
            return

        samples = self.eye_calibration[person_id][target]

        if len(samples) >= 60:
            return

        samples.append(horizontal_ratio)

        if len(samples) % 10 == 0:
            print(
                f"[EYE CALIBRATION] ID {person_id} | "
                f"Target: {target}° | "
                f"Samples: {len(samples)}/60"
            )

    def finalize_eye_calibration(self, person_id):

        if person_id not in self.eye_calibration:
            print(
                f"[EYE CALIBRATION ERROR] "
                f"No calibration data for ID {person_id}"
            )
            return False

        samples = self.eye_calibration[person_id]

        for target in (-10, 0, 10):

            if len(samples[target]) < 30:
                print(
                    f"[EYE CALIBRATION] ID {person_id} | "
                    f"Target {target}° needs 30 samples. "
                    f"Current: {len(samples[target])}"
                )
                return False

        medians = {
            target: float(np.median(samples[target]))
            for target in (-10, 0, 10)
        }

        left_value = medians[-10]
        center_value = medians[0]
        right_value = medians[10]

        is_valid_order = (
            left_value < center_value < right_value
            or
            left_value > center_value > right_value
        )

        if not is_valid_order:
            print(
                f"[EYE CALIBRATION ERROR] ID {person_id}: "
                f"Invalid calibration order. "
                f"Please repeat calibration."
            )
            return False

        if min(
            abs(left_value - center_value),
            abs(right_value - center_value)
        ) < 0.005:
            print(
                f"[EYE CALIBRATION ERROR] ID {person_id}: "
                f"Eye positions are too close. "
                f"Please repeat calibration."
            )
            return False

        self.eye_calibration_results[person_id] = medians

        self.eye_calibration_target.pop(person_id, None)

        print(
            f"[EYE CALIBRATION COMPLETE] ID {person_id}"
        )
        print(f"Left -10° : {left_value:.4f}")
        print(f"Center 0° : {center_value:.4f}")
        print(f"Right +10°: {right_value:.4f}")

        return True

    # ---------------------------------------
    # Save Screen Calibration
    # ---------------------------------------

    def save_screen_calibration(self):

        try:
            saved_data = {
                "coordinate_system": "center_origin_v1",
                "screen_width_cm": self.screen_width_cm,
                "screen_height_cm": self.screen_height_cm,
                "models": {
                    str(person_id): coefficients.tolist()
                    for person_id, coefficients
                    in self.screen_calibration_models.items()
                }
            }

            temp_file = self.screen_calibration_file + ".tmp"

            with open(temp_file, "w") as file:
                json.dump(saved_data, file, indent=4)

            os.replace(temp_file, self.screen_calibration_file)

            print("[INFO] Screen calibration saved successfully.")

            return True

        except Exception as e:

            print(f"[SAVE ERROR] {e}")

            return False

    # ---------------------------------------
    # Load Screen Calibration
    # ---------------------------------------

    def load_screen_calibration(self):

        if not os.path.exists(self.screen_calibration_file):

            print("[INFO] No saved screen calibration found.")
            return False

        try:

            with open(self.screen_calibration_file, "r") as file:
                saved_data = json.load(file)

            if saved_data.get("coordinate_system") != "center_origin_v1":

                print(
                    "[WARNING] Old screen calibration format detected. "
                    "Please perform screen calibration again."
                )
                return False

            if (
                float(saved_data["screen_width_cm"]) != self.screen_width_cm
                or
                float(saved_data["screen_height_cm"]) != self.screen_height_cm
            ):

                print(
                    "[WARNING] Screen dimensions changed. "
                    "Recalibration required."
                )
                return False

            loaded_models = {}

            for person_id, coefficients in saved_data["models"].items():

                model = np.asarray(
                    coefficients,
                    dtype=np.float64
                )

                if model.shape != (3, 2) or not np.all(np.isfinite(model)):
                    print(
                        f"[WARNING] Invalid calibration model for ID {person_id}"
                    )
                    continue

                loaded_models[int(person_id)] = model

            self.screen_calibration_models.update(loaded_models)

            print(
                f"[INFO] Loaded screen calibration for IDs: "
                f"{list(loaded_models.keys())}"
            )

            return bool(loaded_models)

        except Exception as e:

            print(f"[LOAD ERROR] {e}")

            return False

    # ---------------------------------------
    # Screen Gaze Calibration
    # ---------------------------------------

    def set_screen_calibration_target(self, person_id, target_name):

        if target_name not in self.screen_calibration_points:
            raise ValueError("Invalid screen calibration target")

        if person_id not in self.screen_calibration_data:
            self.screen_calibration_data[person_id] = {
                name: [] for name in self.screen_calibration_points
            }

        self.screen_calibration_target[person_id] = target_name

        print(
            f"[SCREEN CALIBRATION] ID {person_id}: "
            f"Look at {target_name}"
        )

    def _collect_screen_calibration_sample(
        self,
        person_id,
        horizontal_ratio,
        vertical_offset
    ):

        if person_id not in self.screen_calibration_data:
            return

        target = self.screen_calibration_target.get(person_id)

        if target is None:
            return

        samples = self.screen_calibration_data[person_id][target]

        if len(samples) >= 60:
            return

        samples.append([
            horizontal_ratio,
            vertical_offset
        ])

        if len(samples) % 10 == 0:
            print(
                f"[SCREEN CALIBRATION] ID {person_id} | "
                f"{target}: {len(samples)}/60 samples"
            )

    def finalize_screen_calibration(self, person_id):

        if person_id not in self.screen_calibration_data:
            print("[SCREEN CALIBRATION ERROR] No calibration data")
            return False

        data = self.screen_calibration_data[person_id]

        features = []
        screen_positions = []

        for target_name, (screen_x, screen_y) in self.screen_calibration_points.items():

            samples = data[target_name]

            if len(samples) < 30:
                print(
                    f"[SCREEN CALIBRATION] {target_name} needs "
                    f"30 samples. Current: {len(samples)}"
                )
                return False

            median_values = np.median(
                np.asarray(samples, dtype=np.float64),
                axis=0
            )

            features.append([
                median_values[0],
                median_values[1],
                1.0
            ])

            screen_positions.append([
                screen_x,
                screen_y
            ])

        features = np.asarray(features, dtype=np.float64)

        screen_positions = np.asarray(
            screen_positions,
            dtype=np.float64
        )

        if np.linalg.matrix_rank(features) < 3:
            print(
                "[SCREEN CALIBRATION ERROR] "
                "Calibration points are not sufficiently distinct."
            )
            return False

        coefficients, _, _, _ = np.linalg.lstsq(
            features,
            screen_positions,
            rcond=None
        )

        # Force the calibrated center median to map exactly to (0, 0)
        center_samples = np.asarray(
            data["CENTER"],
            dtype=np.float64
        )

        center_feature = np.array([
            np.median(center_samples[:, 0]),
            np.median(center_samples[:, 1]),
            1.0
        ])

        center_prediction = center_feature @ coefficients

        coefficients[2, 0] -= center_prediction[0]
        coefficients[2, 1] -= center_prediction[1]

        self.screen_calibration_models[person_id] = coefficients

        # Reset coordinate history after calibration
        self.screen_gaze_history.pop(person_id, None)

        saved = self.save_screen_calibration()

        self.screen_calibration_target.pop(person_id, None)

        if not saved:
            print("[WARNING] Calibration active, but save failed.")
        else:
            print(
                f"[SCREEN CALIBRATION COMPLETE] ID {person_id}"
            )

        return True

    # ---------------------------------------
    # Calculate Screen Gaze Position
    # ---------------------------------------

    def calculate_screen_gaze_position(
        self,
        person_id,
        horizontal_ratio,
        vertical_offset
    ):

        if person_id not in self.screen_calibration_models:
            return None

        coefficients = self.screen_calibration_models[person_id]

        input_values = np.array([
            horizontal_ratio,
            vertical_offset,
            1.0
        ], dtype=np.float64)

        position = input_values @ coefficients

        # Raw center-origin coordinates
        raw_x = float(np.clip(
            position[0],
            -self.screen_width_cm / 2.0,
            self.screen_width_cm / 2.0
        ))

        raw_y = float(np.clip(
            position[1],
            -self.screen_height_cm / 2.0,
            self.screen_height_cm / 2.0
        ))

        # -----------------------------------
        # Center lock
        # -----------------------------------
        # If both coordinates are close to
        # the calibrated center, return exact zero.

        if (
            abs(raw_x) <= self.screen_center_lock_cm
            and
            abs(raw_y) <= self.screen_center_lock_cm
        ):

            self.screen_gaze_history[person_id] = (0.0, 0.0)

            return {
                "x_cm": 0.0,
                "y_cm": 0.0
            }

        # -----------------------------------
        # Exponential moving average
        # -----------------------------------

        if person_id not in self.screen_gaze_history:

            smoothed_x = raw_x
            smoothed_y = raw_y

        else:

            previous_x, previous_y = (
                self.screen_gaze_history[person_id]
            )

            alpha = self.screen_smoothing_alpha

            smoothed_x = (
                alpha * raw_x
                + (1.0 - alpha) * previous_x
            )

            smoothed_y = (
                alpha * raw_y
                + (1.0 - alpha) * previous_y
            )

        # Small dead zone around zero
        if abs(smoothed_x) <= self.screen_dead_zone_cm:
            smoothed_x = 0.0

        if abs(smoothed_y) <= self.screen_dead_zone_cm:
            smoothed_y = 0.0

        self.screen_gaze_history[person_id] = (
            smoothed_x,
            smoothed_y
        )

        return {
            "x_cm": round(smoothed_x, 2),
            "y_cm": round(smoothed_y, 2)
        }

    # ---------------------------------------
    # Eye Gaze Detection
    # ---------------------------------------

    def _calculate_eye_gaze(
        self,
        landmarks,
        person_id=0,
        head_direction="Forward"
    ):

        left_iris = landmarks[468]
        right_iris = landmarks[473]

        left_outer = landmarks[33]
        left_inner = landmarks[133]

        right_inner = landmarks[362]
        right_outer = landmarks[263]

        # Horizontal eye position

        left_min_x = min(left_outer.x, left_inner.x)
        left_max_x = max(left_outer.x, left_inner.x)

        right_min_x = min(right_inner.x, right_outer.x)
        right_max_x = max(right_inner.x, right_outer.x)

        left_width = left_max_x - left_min_x
        right_width = right_max_x - right_min_x

        if left_width <= 1e-6 or right_width <= 1e-6:
            self.latest_eye_position[person_id] = None
            return "No Face", 0.0

        left_ratio_x = (
            left_iris.x - left_min_x
        ) / left_width

        right_ratio_x = (
            right_iris.x - right_min_x
        ) / right_width

        horizontal_ratio = (
            left_ratio_x + right_ratio_x
        ) / 2.0

        # Vertical eye position

        left_corner_y = (
            left_outer.y + left_inner.y
        ) / 2.0

        right_corner_y = (
            right_outer.y + right_inner.y
        ) / 2.0

        left_vertical_offset = (
            left_iris.y - left_corner_y
        ) / left_width

        right_vertical_offset = (
            right_iris.y - right_corner_y
        ) / right_width

        vertical_offset = (
            left_vertical_offset + right_vertical_offset
        ) / 2.0

        # Existing eye calibration collection

        if (
            head_direction == "Forward"
            and person_id in self.baselines
        ):

            self._collect_eye_calibration_sample(
                person_id,
                horizontal_ratio
            )

            self._collect_screen_calibration_sample(
                person_id,
                horizontal_ratio,
                vertical_offset
            )

        # Debug information

        print(
            f"[EYE DEBUG] "
            f"ID: {person_id} | "
            f"Horizontal: {horizontal_ratio:.3f} | "
            f"Vertical: {vertical_offset:.3f}"
        )

        # Existing gaze classification

        if horizontal_ratio < 0.40:

            direction = "Looking Left"

        elif horizontal_ratio > 0.60:

            direction = "Looking Right"

        elif vertical_offset < -0.28:

            direction = "Looking Up"

        elif vertical_offset > 0.00:

            direction = "Looking Down"

        else:

            direction = "Center"

        # Existing confidence

        if direction == "Looking Up":

            confidence = min(
                1.0,
                abs(vertical_offset) / 0.50
            )

        elif direction == "Looking Down":

            confidence = min(
                1.0,
                vertical_offset / 0.20
            )

        elif direction in ("Looking Left", "Looking Right"):

            confidence = min(
                1.0,
                abs(horizontal_ratio - 0.50) * 2.0
            )

        else:

            confidence = 0.0

        # Calculate screen position if calibrated

        self.latest_eye_position[person_id] = (
            self.calculate_screen_gaze_position(
                person_id,
                horizontal_ratio,
                vertical_offset
            )
        )

        return direction, round(confidence, 2)

    # ---------------------------------------
    # Combined Analysis
    # ---------------------------------------

    def analyze(self, frame, person_id=0):

        landmarks = self._get_landmarks(frame)

        if landmarks is None:

            self.latest_eye_position[person_id] = None

            return {
                "head_pose": "No Face",
                "head_confidence": 0.0,
                "eye_gaze": "No Face",
                "gaze_confidence": 0.0,
                "screen_gaze": None,
                "gaze_x_cm": None,
                "gaze_y_cm": None
            }

        head, head_confidence = self._calculate_head_pose(
            landmarks,
            frame.shape,
            person_id
        )

        gaze, gaze_confidence = self._calculate_eye_gaze(
            landmarks,
            person_id,
            head
        )

        head = self._smooth(
            self.head_history,
            person_id,
            head
        )

        gaze = self._smooth(
            self.gaze_history,
            person_id,
            gaze
        )

        screen_position = self.latest_eye_position.get(person_id)

        gaze_x_cm = None
        gaze_y_cm = None

        if screen_position is not None:
            gaze_x_cm = screen_position["x_cm"]
            gaze_y_cm = screen_position["y_cm"]

        return {
            "head_pose": head,
            "head_confidence": head_confidence,
            "eye_gaze": gaze,
            "gaze_confidence": gaze_confidence,
            "screen_gaze": screen_position,
            "gaze_x_cm": gaze_x_cm,
            "gaze_y_cm": gaze_y_cm
        }

    # ---------------------------------------
    # Backward Compatible Function
    # ---------------------------------------

    def get_head_direction(
        self,
        frame,
        return_confidence=False,
        person_id=0
    ):

        result = self.analyze(
            frame,
            person_id
        )

        direction = result["head_pose"]
        confidence = result["head_confidence"]

        if return_confidence:
            return direction, confidence

        return direction

    # ---------------------------------------
    # Draw Direction
    # ---------------------------------------

    @staticmethod
    def draw_direction(
        frame,
        direction,
        confidence: Optional[float] = None
    ):

        if confidence is None:
            text = direction
        else:
            text = f"{direction} ({confidence:.2f})"

        cv2.putText(
            frame,
            text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    # ---------------------------------------
    # Cleanup
    # ---------------------------------------

    def close(self):

        if self.face_mesh:
            self.face_mesh.close()
