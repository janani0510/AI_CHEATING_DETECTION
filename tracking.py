class Tracker:

    def __init__(self, max_distance=80, max_lost=10, min_iou=0.1):

        if max_distance <= 0:
            raise ValueError("max_distance must be > 0")

        if max_lost < 0:
            raise ValueError("max_lost must be >= 0")

        if not (0 <= min_iou <= 1):
            raise ValueError("min_iou must be between 0 and 1")
        self.next_id = 0
        self.objects = {}   # id -> {center, bbox, class_id, age}
        self.lost = {}      # id -> lost frames

        self.max_distance_sq = max_distance ** 2
        self.max_lost = max_lost
        self.min_iou = min_iou

    # -----------------------
    # Utilities
    # -----------------------
    def _get_center(self, bbox):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return (0, 0)
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _dist_sq(self, p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy

    def _iou(self, a, b):
        xA = max(a[0], b[0])
        yA = max(a[1], b[1])
        xB = min(a[2], b[2])
        yB = min(a[3], b[3])

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter = inter_w * inter_h

        if inter == 0:
            return 0.0

        areaA = (a[2] - a[0]) * (a[3] - a[1])
        areaB = (b[2] - b[0]) * (b[3] - b[1])

        return inter / (areaA + areaB - inter + 1e-6)

    # -----------------------
    # MAIN UPDATE
    # -----------------------
    def update(self, detections):

        # 🔥 Remove weak detections (noise reduction)
        detections = [
            d for d in detections 
            if d.get("confidence", 0) > 0.3
            and "bbox" in d
        ]

        new_objects = {}
        new_lost = {}
        tracked = []

        # ---- Case 0: First frame
        if not self.objects:
            for det in detections:

                obj_id = self.next_id
                self.next_id += 1

                bbox = det["bbox"]

                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue

                center = self._get_center(bbox)

                new_objects[obj_id] = {
                    "center": center,
                    "bbox": bbox,
                    "class_id": det.get("class_id"),
                    "age": 1
                }
                new_lost[obj_id] = 0

                tracked.append({
                    "id": obj_id,
                    "bbox": bbox,
                    "confidence": det.get("confidence", 0.0),
                    "class_id": det.get("class_id"),
                    "lost": False,
                    "lost_frames": 0,
                    "active": True
                })

            self.objects, self.lost = new_objects, new_lost
            return tracked

        # ---- Case 1: No detections
        if not detections:
            for obj_id, obj in self.objects.items():
                lost_count = self.lost.get(obj_id, 0) + 1

                if lost_count <= self.max_lost:
                    new_objects[obj_id] = obj
                    new_lost[obj_id] = lost_count

                    tracked.append({
                        "id": obj_id,
                        "bbox": obj["bbox"],
                        "confidence": 0.0,
                        "class_id": obj.get("class_id"),
                        "lost": True,
                        "lost_frames": lost_count,
                        "active": False
                    })

            self.objects, self.lost = new_objects, new_lost
            return tracked

        # ---- Case 2: Matching detections
        used_ids = set()

        # Sort detections by confidence (high first)
        detections = sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True)

        for det in detections:
        
            det_bbox = det["bbox"]

            if not isinstance(det_bbox, (list, tuple)) or len(det_bbox) != 4:
                continue

            det_center = self._get_center(det_bbox)
            det_cls = det.get("class_id")

            best_id = None
            best_score = float("inf")

            # 🔥 prioritize recently tracked objects
            for obj_id, obj in sorted(self.objects.items(),
                                      key=lambda x: self.lost.get(x[0], 0)):

                if obj_id in used_ids:
                    continue

                obj_bbox = obj["bbox"]
                obj_center = obj["center"]
                obj_cls = obj.get("class_id")

                # Class consistency
                if obj_cls is not None and det_cls is not None and obj_cls != det_cls:
                    continue

                dist_sq = self._dist_sq(det_center, obj_center)

                # Relaxed distance constraint
                if dist_sq > self.max_distance_sq * 1.5:
                    continue

                iou = self._iou(det_bbox, obj_bbox)

                if iou < self.min_iou:
                    continue

                # 🔥 FIXED scoring formula
                score = (dist_sq / self.max_distance_sq) + (1 - iou)

                if score < best_score:
                    best_score = score
                    best_id = obj_id

            # Assign ID
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1

            used_ids.add(best_id)

            new_objects[best_id] = {
                "center": det_center,
                "bbox": det_bbox,
                "class_id": det_cls,
                "age": self.objects.get(best_id, {}).get("age", 0) + 1
            }
            new_lost[best_id] = 0

            tracked.append({
                "id": best_id,
                "bbox": det_bbox,
                "confidence": det.get("confidence", 0.0),
                "class_id": det_cls,
                "lost": False,
                "lost_frames": 0,
                "active": True
            })

        # ---- Handle lost objects
        for obj_id, obj in self.objects.items():
            if obj_id not in used_ids:
                lost_count = self.lost.get(obj_id, 0) + 1

                if lost_count <= self.max_lost:
                    new_objects[obj_id] = obj
                    new_lost[obj_id] = lost_count

                    tracked.append({
                        "id": obj_id,
                        "bbox": obj["bbox"],
                        "confidence": 0.0,
                        "class_id": obj.get("class_id"),
                        "lost": True,
                        "lost_frames": lost_count,
                        "active": False
                    })

        self.objects, self.lost = new_objects, new_lost
        return tracked