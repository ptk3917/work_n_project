#!/usr/bin/env python3
"""
YOLO11 OBB 학습 스크립트
========================

[1단계] 데이터셋 구조 준비:
  obb_dataset/
  ├── images/
  │   ├── train/          ← 학습 이미지 (.jpg)
  │   └── val/            ← 검증 이미지 (.jpg)
  ├── labels/
  │   ├── train/          ← 학습 라벨 (.txt, YOLO OBB 포맷)
  │   └── val/            ← 검증 라벨 (.txt)
  └── data.yaml           ← 이 스크립트가 자동 생성

[2단계] 라벨 포맷 (YOLO OBB):
  각 .txt 파일에 한 줄 = 한 물체:
    <class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
  좌표는 0~1 정규화. 4개 코너를 시계방향으로.

  예) 클래스 0, 회전된 박스:
    0 0.45 0.30 0.55 0.28 0.56 0.42 0.46 0.44

[3단계] 학습:
  python obb_train.py --classes "my_product" --epochs 100

[4단계] 추론 테스트:
  python obb_train.py --test --weights runs/obb/train/weights/best.pt
"""

import os
import argparse
import yaml
from ultralytics import YOLO


def create_yaml(dataset_dir, classes):
    """data.yaml 자동 생성"""
    data = {
        "path": os.path.abspath(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(classes)},
    }
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"✅ data.yaml 생성: {yaml_path}")
    print(f"   클래스: {classes}")
    return yaml_path


def train(args):
    """YOLO11 OBB 학습"""
    classes = [c.strip() for c in args.classes.split(",")]

    # data.yaml 생성
    yaml_path = create_yaml(args.dataset, classes)

    # 데이터셋 검증
    train_img_dir = os.path.join(args.dataset, "images", "train")
    val_img_dir   = os.path.join(args.dataset, "images", "val")
    train_lbl_dir = os.path.join(args.dataset, "labels", "train")

    if not os.path.exists(train_img_dir):
        print(f"[오류] 학습 이미지 디렉토리 없음: {train_img_dir}")
        return
    if not os.path.exists(train_lbl_dir):
        print(f"[오류] 학습 라벨 디렉토리 없음: {train_lbl_dir}")
        return

    n_train = len([f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png'))])
    n_val   = len([f for f in os.listdir(val_img_dir) if f.endswith(('.jpg', '.png'))]) if os.path.exists(val_img_dir) else 0
    n_label = len([f for f in os.listdir(train_lbl_dir) if f.endswith('.txt')])

    print(f"\n데이터셋 현황:")
    print(f"  학습 이미지: {n_train}장")
    print(f"  학습 라벨:   {n_label}개")
    print(f"  검증 이미지: {n_val}장")
    print(f"  클래스:      {classes}")

    if n_train == 0:
        print("\n[오류] 학습 이미지가 없습니다. 먼저 데이터를 수집하세요.")
        return
    if n_label == 0:
        print("\n[오류] 라벨 파일이 없습니다. 라벨링을 먼저 하세요.")
        return

    # ★ 핵심: yolo11n-obb.pt (OBB 전용 pretrained 모델)
    model_name = f"yolo11{args.size}-obb.pt"
    print(f"\n모델: {model_name}")
    print(f"에폭: {args.epochs}")
    print(f"이미지 크기: {args.imgsz}")
    print(f"배치 크기: {args.batch}")

    model = YOLO(model_name)

    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,             # GPU
        workers=4,
        patience=20,          # early stopping
        save=True,
        plots=True,
        augment=True,
        degrees=180,          # ★ OBB: 회전 증강 (360도 전체)
        flipud=0.5,           # 상하 반전
        fliplr=0.5,           # 좌우 반전
        mosaic=1.0,           # 모자이크 증강
    )

    print(f"\n✅ 학습 완료!")
    print(f"   best.pt: runs/obb/train/weights/best.pt")
    print(f"   last.pt: runs/obb/train/weights/last.pt")
    return results


def test(args):
    """학습된 모델로 추론 테스트"""
    if not os.path.exists(args.weights):
        print(f"[오류] 가중치 파일 없음: {args.weights}")
        return

    import cv2
    import pyrealsense2 as rs
    import numpy as np

    model = YOLO(args.weights)

    # D455 실시간 테스트
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    for _ in range(30):
        pipeline.wait_for_frames()

    print("실시간 OBB 감지 테스트 | [Q] 종료")

    while True:
        frames = pipeline.wait_for_frames()
        color = np.asanyarray(frames.get_color_frame().get_data())

        results = model.predict(color, conf=0.5, verbose=False)

        vis = color.copy()
        for r in results:
            if r.obb is not None:
                for box in r.obb:
                    # OBB: xywhr (center_x, center_y, width, height, rotation)
                    xywhr = box.xywhr[0].cpu().numpy()
                    cx, cy, w, h, angle = xywhr
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]

                    # 4코너 좌표 (xyxyxyxy)
                    corners = box.xyxyxyxy[0].cpu().numpy().astype(int)
                    cv2.polylines(vis, [corners], True, (0, 255, 0), 2)

                    # 라벨 + 각도
                    angle_deg = np.degrees(angle) if angle <= np.pi else np.degrees(angle)
                    text = f"{label} {conf:.2f} {angle_deg:.1f}deg"
                    cv2.putText(vis, text, (int(cx), int(cy) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # 중심점
                    cv2.circle(vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        cv2.imshow("OBB Test", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipeline.stop()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="YOLO11 OBB 학습")
    parser.add_argument("--test", action="store_true", help="학습된 모델 테스트")
    parser.add_argument("--weights", type=str,
                        default="runs/obb/train/weights/best.pt",
                        help="테스트용 가중치 경로")
    parser.add_argument("--dataset", type=str, default="./obb_dataset",
                        help="데이터셋 디렉토리")
    parser.add_argument("--classes", type=str, default="product",
                        help="클래스 이름 (콤마 구분)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--size", type=str, default="n",
                        choices=["n", "s", "m", "l", "x"],
                        help="모델 크기 (n=nano, s=small, m=medium, l=large, x=xlarge)")
    args = parser.parse_args()

    if args.test:
        test(args)
    else:
        train(args)


if __name__ == "__main__":
    main()