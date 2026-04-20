#!/usr/bin/env python3
"""
YOLO11 OBB 학습 데이터 수집 도구
=================================
D455 카메라에서 스캔 위치 이미지를 촬영하여 학습 데이터를 수집합니다.

사용법:
  python obb_data_capture.py --output ./obb_dataset/images/train

조작:
  [SPACE] 이미지 캡처 및 저장
  [Q]     종료

캡처 후 라벨링:
  - Roboflow (추천): OBB 라벨링 지원, 웹 기반
    https://app.roboflow.com → 프로젝트 생성 → Oriented Bounding Box 선택
  - CVAT: https://app.cvat.ai → OBB 라벨링 지원
  - Label Studio: OBB polygon 방식 라벨링 가능

라벨 포맷 (YOLO OBB):
  <class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
  좌표는 이미지 크기로 정규화 (0~1)
"""

import os
import sys
import time
import argparse
import cv2
import pyrealsense2 as rs
import numpy as np
from datetime import datetime
from fairino import Robot

# ── 설정 (picking_pipeline과 동일) ──
ROBOT_IP = "192.168.58.3"
TOOL_NO  = 1
USER_NO  = 0
SPEED_J  = 30

HOME_J  = [10.0, -98.0, 100.0, -94.0, -84.0, -111.0]
SCAN1_J = [9.22, -123.75, 134.18, -148.74, -80.03, -100.25]


def wait_done(robot, timeout=30.0):
    time.sleep(0.4)
    start = time.time()
    while time.time() - start < timeout:
        ret = robot.GetRobotMotionDone()
        done = ret[1] if isinstance(ret, (list, tuple)) else ret
        if done == 1:
            time.sleep(0.1)
            return
        time.sleep(0.05)


def move_j(robot, joints, label=""):
    if label:
        print(f"  {label}...")
    ret = robot.MoveJ(joints, TOOL_NO, USER_NO, vel=SPEED_J, acc=SPEED_J, blendT=0)
    err = ret[0] if isinstance(ret, (list, tuple)) else ret
    if err != 0:
        raise RuntimeError(f"{label} 실패 (err={err})")
    wait_done(robot)


def main():
    parser = argparse.ArgumentParser(description="OBB 학습 데이터 캡처")
    parser.add_argument("--output", type=str, default="./obb_dataset/images/train",
                        help="이미지 저장 디렉토리")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    existing = len([f for f in os.listdir(args.output) if f.endswith(('.jpg', '.png'))])
    print(f"저장 디렉토리: {args.output}")
    print(f"기존 이미지: {existing}장")

    # ── FR5 초기화 + 스캔 위치 이동 ──
    print(f"\nFR5 연결 중... ({ROBOT_IP})")
    robot = Robot.RPC(ROBOT_IP)
    ret, error = robot.GetRobotErrorCode()
    if ret == 0 and error != 0:
        robot.ResetAllError()
        time.sleep(0.5)
    print("✅ FR5 연결 완료")

    move_j(robot, HOME_J,  "홈")
    move_j(robot, SCAN1_J, "스캔 위치")

    # ── D455 초기화 ──
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    print("카메라 안정화 중...")
    for _ in range(30):
        pipeline.wait_for_frames()
    print("✅ 준비 완료")

    print("\n[SPACE] 캡처 | [Q] 종료\n")

    count = existing
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        vis = img.copy()

        cv2.putText(vis, f"Captured: {count} | [SPACE] Save | [Q] Quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("OBB Data Capture", vis)

        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):
            count += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"img_{ts}_{count:04d}.jpg"
            fpath = os.path.join(args.output, fname)
            cv2.imwrite(fpath, img)
            print(f"  ✅ #{count} 저장: {fname}")

        elif key == ord('q'):
            break

    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"\n총 {count}장 캡처 완료.")
    move_j(robot, HOME_J, "홈 복귀")
    print(f"\n이제 라벨링하세요!")
    print(f"  Roboflow: https://app.roboflow.com")
    print(f"  CVAT:     https://app.cvat.ai")


if __name__ == "__main__":
    main()