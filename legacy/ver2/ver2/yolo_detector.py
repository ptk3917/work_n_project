import cv2
import numpy as np
import torch
from ultralytics import YOLOWorld
from config import YOLO_MODEL, DETECT_CONF, CAM_W, CAM_H


class YOLODetector:
    def __init__(self):
        print(f"🔄 YOLO-World 로딩 중... ({YOLO_MODEL})")
        self.model           = YOLOWorld(YOLO_MODEL)
        self.current_classes = []

        # ── 모델 전체 GPU 이동 ─────────────────────
        self.model.to("cuda")
        print("✅ YOLO-World 로딩 완료 (CUDA)")

    # ── 탐지 클래스 설정 ───────────────────────────
    def set_targets(self, classes: list):
        """찾을 물체 이름 목록 설정"""
        self.current_classes = classes
        self.model.set_classes(classes)

        # set_classes 후 텍스트 임베딩이 CPU로 돌아오는 문제 방지
        self.model.to("cuda")

        # CLIP txt_feats 명시적 GPU 이동
        if hasattr(self.model, "model"):
            inner = self.model.model
            if hasattr(inner, "txt_feats") and inner.txt_feats is not None:
                inner.txt_feats = inner.txt_feats.cuda()

        print(f"🎯 YOLO 탐지 대상: {classes}")

    # ── 감지 실행 ──────────────────────────────────
    def detect(self, frame: np.ndarray,
               conf: float = DETECT_CONF) -> list:
        """
        프레임에서 물체 감지
        반환: [{"label", "conf", "center", "bbox", "area", "area_ratio"}, ...]
        """
        if not self.current_classes:
            return []

        results    = self.model.predict(frame, conf=conf, verbose=False)
        h, w       = frame.shape[:2]
        frame_area = w * h
        detections = []

        for box in results[0].boxes:
            xyxy   = box.xyxy[0].tolist()
            xywh   = box.xywh[0].tolist()
            conf_  = float(box.conf[0])
            cls_id = int(box.cls[0])
            label  = self.model.names[cls_id]

            x1, y1, x2, y2 = map(int, xyxy)
            cx, cy          = xywh[0], xywh[1]
            area            = (x2 - x1) * (y2 - y1)

            detections.append({
                "label":      label,
                "conf":       conf_,
                "center":     (cx, cy),
                "bbox":       (x1, y1, x2, y2),
                "area":       area,
                "area_ratio": area / frame_area,
            })

        return detections

    # ── 화면에 그리기 ──────────────────────────────
    def draw(self, frame: np.ndarray,
             detections: list,
             highlight: dict = None) -> np.ndarray:
        """감지 결과를 프레임에 시각화"""
        out = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy          = det["center"]
            label           = det["label"]
            conf            = det["conf"]

            is_highlight = (highlight and det == highlight)
            color         = (0, 215, 255) if is_highlight else (137, 180, 250)
            thickness     = 3 if is_highlight else 2

            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(out, (int(cx), int(cy)), 6, (243, 139, 168), -1)
            cv2.putText(out,
                        f"{label} {conf:.0%}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, color, 2)

        return out

    # ── 유틸 ───────────────────────────────────────
    def is_close_enough(self, detections: list,
                        ratio: float = 0.25) -> bool:
        """bbox가 화면의 ratio 이상이면 충분히 가까운 것으로 판단"""
        if not detections:
            return False
        largest = max(detections, key=lambda d: d["area"])
        return largest["area_ratio"] >= ratio

    def get_largest(self, detections: list) -> dict:
        if not detections:
            return None
        return max(detections, key=lambda d: d["area"])

    def get_nearest_to_center(self, detections: list,
                               frame_w: int = CAM_W,
                               frame_h: int = CAM_H) -> dict:
        if not detections:
            return None
        cx_f, cy_f = frame_w / 2, frame_h / 2
        return min(detections, key=lambda d:
                   (d["center"][0] - cx_f) ** 2 +
                   (d["center"][1] - cy_f) ** 2)