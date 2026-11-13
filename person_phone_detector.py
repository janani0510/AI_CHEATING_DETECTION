
import torch
from ultralytics import YOLO
from config import Config


class PersonPhoneDetector:
    """
    Detects multiple persons and selected objects using YOLOv8.
    Includes object detection debugging.
    """

    def __init__(self, model_path="yolov8n.pt", img_size=640):

        self.model = YOLO(model_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.class_names = self.model.names
        self.img_size = img_size

        # Existing confidence threshold
        self.default_conf = float(Config.YOLO.CONF_THRESHOLD)

        # Lower threshold for object detection
        self.object_conf = 0.12

        # Lower YOLO inference threshold to avoid early filtering
        self.inference_conf = min(self.default_conf, 0.10)

        # COCO object classes
        self.target_objects = {
            "cell phone",
            "laptop",
            "book",
            "bottle",
            "backpack",
            "handbag",
            "suitcase"
        }

        print(f"[INFO] Using device: {self.device}")
        print("[INFO] YOLO model loaded successfully")
        print("[INFO] Multiple person and object detection enabled")
        print(f"[INFO] Person/Phone confidence: {self.default_conf}")
        print(f"[INFO] Object confidence: {self.object_conf}")
        print(f"[INFO] YOLO inference confidence: {self.inference_conf}")

    def detect(self, frame):

        if frame is None or frame.size == 0:
            return self._empty_result()

        try:
            with torch.no_grad():

                results = self.model(
                    frame,
                    conf=self.inference_conf,
                    imgsz=self.img_size,
                    verbose=False
                )[0]

        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            return self._empty_result()

        if results.boxes is None or len(results.boxes) == 0:
            return self._empty_result()

        persons = []
        phones = []
        objects = []

        h, w = frame.shape[:2]

        for box in results.boxes:

            cls_id = int(box.cls.item())
            conf = float(box.conf.item())

            class_name = self.class_names[cls_id]

            # Debug output for selected objects
            if class_name in self.target_objects:
                print(
                    f"[OBJECT DEBUG] {class_name} | "
                    f"Confidence: {conf:.3f}"
                )

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            # Clip coordinates
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            detection = {
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 3),
                "class_id": cls_id,
                "class_name": class_name
            }

            # Multiple person detection
            if cls_id == Config.YOLO.PERSON_ID:

                if conf >= self.default_conf:
                    persons.append(detection)

            # Phone detection
            elif class_name == "cell phone":

                if conf >= self.default_conf:
                    phones.append(detection)
                    objects.append(detection)

            # Other object detection
            elif class_name in self.target_objects:

                if conf >= self.object_conf:
                    objects.append(detection)

        return {
            "persons": persons,
            "phones": phones,
            "objects": objects,

            "counts": {
                "persons": len(persons),
                "phones": len(phones),
                "objects": len(objects)
            }
        }

    def _empty_result(self):

        return {
            "persons": [],
            "phones": [],
            "objects": [],

            "counts": {
                "persons": 0,
                "phones": 0,
                "objects": 0
            }
        }
