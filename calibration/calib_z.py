#!/usr/bin/env python3
"""
Z축(높이) 캘리브레이션 도구 — YOLO-World bbox 감지 버전
========================================================
1) YOLO-World로 물체 bbox 표시
2) 사용자가 bbox 클릭 → 해당 물체 선택
3) 로봇이 물체 위로 이동 (카메라 중심에 물체 정렬)
4) 재촬영 + depth 재측정 (정면에서 더 정확)
5) 예측 Z + 120mm 상공으로 이동
6) 사용자가 조그로 하강 → Enter → TCP Z 기록

※ picking_pipeline.py와 동일한 로봇/카메라/좌표 변환 코드

사용법:
  1) 수집:  python z_calibration.py --collect --classes "cup,bottle,box"
  2) 피팅:  python z_calibration.py --fit
  3) 검증:  python z_calibration.py --verify --classes "cup,bottle,box"

설치 (YOLO-World):
  pip install ultralytics --break-system-packages
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import cv2
import pyrealsense2 as rs
from datetime import datetime
from scipy.spatial.transform import Rotation

# ── Fairino SDK ──
from fairino import Robot

# ── YOLO-World ──
from ultralytics import YOLO


# =====================================================================
# 설정 (picking_pipeline과 동일)
# =====================================================================
ROBOT_IP = "192.168.58.3"
TOOL_NO  = 1
USER_NO  = 0

SPEED    = 20
SPEED_J  = 30

# ── Hand-Eye 캘리브레이션 매트릭스 ──
CALIB_PATH = os.path.expanduser("~/vlm_agent/qwen_ver1/T_cam_to_ee.npy")
T_CAM_TO_EE = np.load(CALIB_PATH) if os.path.exists(CALIB_PATH) else np.eye(4)
if os.path.exists(CALIB_PATH):
    print(f"✅ Hand-Eye 매트릭스 로드: {CALIB_PATH}")
else:
    print(f"⚠ Hand-Eye 매트릭스 없음 — 단위 행렬 사용")

# ── 카메라 내부 파라미터 (캘리브레이션 값) ──
FX = 386.4894
FY = 386.6551
CX = 325.7376
CY = 246.5258

IMG_W, IMG_H = 640, 480
IMG_CENTER_X, IMG_CENTER_Y = IMG_W // 2, IMG_H // 2  # 320, 240

# ── 로봇 포즈 ──
HOME_J  = [10.0, -98.0, 100.0, -94.0, -84.0, -111.0]
SCAN1_J = [9.22, -123.75, 134.18, -148.74, -80.03, -100.25]

# ── Z 캘리브레이션 파일 ──
Z_CALIB_DATA_PATH  = "z_calib_data.json"
Z_CALIB_MODEL_PATH = "z_correction.json"

# ── 파라미터 ──
APPROACH_HEIGHT_OFFSET = 120  # 예측 Z 위에서 접근 (mm)
MIN_POINTS = 4
YOLO_CONFIDENCE = 0.3         # YOLO 감지 최소 신뢰도

# ── YOLO-World 모델 ──
yolo_model = None


# =====================================================================
# D455 초기화 (640×480)
# =====================================================================
rs_pipeline = rs.pipeline()
rs_config   = rs.config()
rs_config.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.bgr8, 30)
rs_config.enable_stream(rs.stream.depth, IMG_W, IMG_H, rs.format.z16, 30)
rs_pipeline.start(rs_config)
align = rs.align(rs.stream.color)

print("카메라 안정화 대기 중...")
for _ in range(30):
    rs_pipeline.wait_for_frames()
print(f"✅ D455 초기화 완료 ({IMG_W}×{IMG_H})")


# =====================================================================
# FR5 + 그리퍼 초기화
# =====================================================================
print(f"FR5 연결 중... ({ROBOT_IP})")
robot = Robot.RPC(ROBOT_IP)

ret, error = robot.GetRobotErrorCode()
if ret == 0 and error != 0:
    print(f"  ⚠ 기존 에러 감지 (code={error}), 자동 클리어 중...")
    robot.ResetAllError()
    time.sleep(0.5)
elif ret != 0:
    print(f"⚠ 로봇 통신 에러: {ret}")
    sys.exit(1)
print("✅ FR5 연결 완료")

print("그리퍼 초기화 중...")
robot.SetAxleCommunicationParam(
    baudRate=7, dataBit=8, stopBit=1, verify=0,
    timeout=50, timeoutTimes=3, period=10
)
time.sleep(1)
robot.SetGripperConfig(company=4, device=0)
time.sleep(1)
robot.ActGripper(index=1, action=1)
time.sleep(2)
robot.MoveGripper(index=1, pos=100, vel=50, force=50,
                  maxtime=5000, block=1, type=0,
                  rotNum=0, rotVel=0, rotTorque=0)
print("✅ 그리퍼 초기화 완료")
print("=" * 55)


# =====================================================================
# 로봇 이동 함수 (picking_pipeline과 동일)
# =====================================================================
def wait_done(timeout: float = 30.0):
    time.sleep(0.4)
    start = time.time()
    while time.time() - start < timeout:
        ret  = robot.GetRobotMotionDone()
        done = ret[1] if isinstance(ret, (list, tuple)) else ret
        if done == 1:
            time.sleep(0.1)
            return
        time.sleep(0.05)
    raise RuntimeError(f"이동 타임아웃 ({timeout}s 초과)")


def get_stable_pose(samples: int = 5, interval: float = 0.05) -> list:
    poses = []
    for _ in range(samples):
        _, p = robot.GetActualTCPPose(1)
        poses.append(p)
        time.sleep(interval)
    return np.array(poses).mean(axis=0).tolist()


def move_j(joints, label=""):
    if label:
        print(f"  {label}...")
    ret = robot.MoveJ(joints, TOOL_NO, USER_NO, vel=SPEED_J, acc=SPEED_J, blendT=0)
    err = ret[0] if isinstance(ret, (list, tuple)) else ret
    if err != 0:
        raise RuntimeError(f"{label} 실패 (err={err})")
    wait_done()


def move_l(pose, label=""):
    if label:
        print(f"  {label}...")
    ret = robot.MoveL(pose, TOOL_NO, USER_NO, vel=SPEED, acc=SPEED)
    err = ret[0] if isinstance(ret, (list, tuple)) else ret
    if err != 0:
        raise RuntimeError(f"{label} 실패 (err={err})")
    wait_done()


# =====================================================================
# 카메라 / 좌표 함수
# =====================================================================
def capture_frame():
    """컬러 + 뎁스 프레임 캡처"""
    frames  = rs_pipeline.wait_for_frames()
    aligned = align.process(frames)
    return np.asanyarray(aligned.get_color_frame().get_data()), aligned.get_depth_frame()


def get_depth_at(depth_frame, cx, cy, window=5):
    """윈도우 중앙값 depth (mm)"""
    depth_np = np.asanyarray(depth_frame.get_data())
    h, w = depth_np.shape
    x0, x1 = max(0, int(cx) - window), min(w, int(cx) + window + 1)
    y0, y1 = max(0, int(cy) - window), min(h, int(cy) + window + 1)
    roi = depth_np[y0:y1, x0:x1].astype(float)
    valid = roi[roi > 0]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


def pixel_to_base(px, py, depth_mm):
    """픽셀 → 베이스 좌표 (picking_pipeline 동일)"""
    z = depth_mm
    x = (px - CX) * z / FX
    y = (py - CY) * z / FY
    p_cam = np.array([x, y, z, 1.0])

    ee_pose = get_stable_pose(samples=3, interval=0.05)
    ex, ey, ez, erx, ery, erz = ee_pose

    R = Rotation.from_euler('xyz', [erx, ery, erz], degrees=True).as_matrix()
    T_ee_to_base = np.eye(4)
    T_ee_to_base[:3, :3] = R
    T_ee_to_base[:3,  3] = [ex, ey, ez]

    p_ee   = T_CAM_TO_EE @ p_cam
    p_base = T_ee_to_base @ p_ee
    return p_base[:3]


# =====================================================================
# YOLO-World 감지
# =====================================================================
def init_yolo(classes: list):
    """YOLO-World 모델 로드 및 클래스 설정"""
    global yolo_model
    print(f"YOLO-World 로드 중... (classes: {classes})")
    yolo_model = YOLO("yolov8s-worldv2.pt")
    yolo_model.set_classes(classes)
    print("✅ YOLO-World 준비 완료")


def detect_objects(color_np):
    """
    현재 프레임에서 YOLO-World 감지.
    Returns:
        list of dict: [{"bbox": [x1,y1,x2,y2], "label": str, "conf": float, "center": (cx,cy)}, ...]
    """
    results = yolo_model.predict(color_np, conf=YOLO_CONFIDENCE, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = yolo_model.names[cls_id] if cls_id in yolo_model.names else f"cls_{cls_id}"
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "label": label,
                "conf": conf,
                "center": (cx, cy),
            })
    return detections


def draw_detections(vis, detections, selected_idx=None):
    """bbox 시각화"""
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        color = (0, 255, 0) if i != selected_idx else (0, 255, 255)
        thickness = 2 if i != selected_idx else 3
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        label = f"[{i}] {det['label']} {det['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(vis, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return vis


def find_clicked_bbox(detections, click_x, click_y):
    """클릭 좌표가 어떤 bbox 안에 있는지 확인"""
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        if x1 <= click_x <= x2 and y1 <= click_y <= y2:
            return i
    return None


# =====================================================================
# Z 캘리브레이션 수집
# =====================================================================
def collect_data():
    """
    [Phase 1 — 카메라+YOLO] 스캔 위치에서:
      - YOLO-World bbox 표시
      - 사용자가 bbox 클릭 → 물체 선택
      - bbox 중심 depth + pixel_to_base로 예측 좌표 계산

    [Phase 2 — 조그] 예측 Z + 120mm 상공에서:
      - 스캔 자세(Rx/Ry/Rz) 유지하면서 물체 위 상공으로 MoveL
      - 카메라 사용 안 함
      - 사용자가 조그로 하강 → Enter → TCP Z 기록
    """
    print("\n" + "=" * 60)
    print("  Z축 캘리브레이션 — YOLO-World bbox 감지")
    print("  ─────────────────────────────")
    print("  [카메라] bbox 클릭 → 센터링 이동 → 재측정")
    print("  [터미널] 조그 하강 → Enter 기록")
    print("  [터미널] 's'=저장 | 'q'=종료 | 'u'=undo")
    print("=" * 60)

    # 기존 데이터 로드
    data_points = []
    if os.path.exists(Z_CALIB_DATA_PATH):
        with open(Z_CALIB_DATA_PATH, "r") as f:
            data_points = json.load(f).get("points", [])
        print(f"\n[안내] 기존 {len(data_points)}포인트 로드됨.")

    # 스캔 위치로 이동
    move_j(HOME_J,  "홈")
    move_j(SCAN1_J, "스캔 위치")

    while True:
        # ──────────────────────────────────
        # Phase 1: YOLO bbox 클릭
        # ──────────────────────────────────
        print(f"\n{'─'*45}")
        print(f"  포인트: {len(data_points)}개 | bbox를 클릭하세요")
        print(f"  (카메라 창: q=종료 s=저장 u=undo)")
        print(f"{'─'*45}")

        click_result = _yolo_click_phase(data_points)

        if click_result == "quit":
            move_j(HOME_J, "홈 복귀")
            break
        elif click_result == "save":
            if len(data_points) < MIN_POINTS:
                print(f"[경고] 최소 {MIN_POINTS}포인트 필요 (현재: {len(data_points)})")
                continue
            _save_data(data_points)
            move_j(HOME_J, "홈 복귀")
            break
        elif click_result == "undo":
            if data_points:
                removed = data_points.pop()
                print(f"[Undo] #{removed['index']} 제거 (남은: {len(data_points)})")
            continue
        elif not isinstance(click_result, dict):
            continue

        # 1차 측정 결과
        det        = click_result["detection"]
        depth_mm_1 = click_result["depth_mm"]
        base_xyz_1 = click_result["base_xyz"]

        print(f"\n[측정] {det['label']} (conf={det['conf']:.2f})")
        print(f"  bbox 중심: {det['center']}")
        print(f"  depth: {depth_mm_1:.1f}mm")
        print(f"  베이스: X={base_xyz_1[0]:.1f} Y={base_xyz_1[1]:.1f} Z={base_xyz_1[2]:.1f}")

        predicted_z = base_xyz_1[2]

        # ──────────────────────────────────
        # Phase 2: 스캔 자세 유지하면서 상공 접근
        # ──────────────────────────────────
        # ★ 스캔 위치의 Rx/Ry/Rz를 그대로 유지 → MoveL 자세 안정
        scan_pose = get_stable_pose(samples=3, interval=0.05)
        rx, ry, rz = scan_pose[3], scan_pose[4], scan_pose[5]

        approach_pose = [
            base_xyz_1[0], base_xyz_1[1],
            predicted_z + APPROACH_HEIGHT_OFFSET,
            rx, ry, rz
        ]

        print(f"\n[이동] Z={predicted_z:.1f} + {APPROACH_HEIGHT_OFFSET}mm 상공으로")
        move_l(approach_pose, "상공 접근")

        # 터미널에서 조그 대기
        print(f"\n{'─'*45}")
        print(f"  상공 도착. TCP Z = {predicted_z + APPROACH_HEIGHT_OFFSET:.1f}mm")
        print(f"  예측 물체 Z = {predicted_z:.1f}mm")
        print(f"")
        print(f"  ▶ 티칭펜던트로 아래로 내리세요")
        print(f"  ▶ 물체에 닿으면 Enter")
        print(f"  ▶ 'c' = 이 포인트 취소")
        print(f"{'─'*45}")

        user_input = input("  >> ").strip().lower()

        if user_input == 'c':
            print("[취소] 건너뜀.")
            move_j(SCAN1_J, "스캔 복귀")
            continue

        # TCP Z 읽기 (카메라 무관)
        tcp_actual = get_stable_pose(samples=5, interval=0.05)
        actual_z = tcp_actual[2]
        error = actual_z - predicted_z

        point = {
            "index": len(data_points) + 1,
            "predicted_z": round(predicted_z, 2),
            "actual_z": round(actual_z, 2),
            "error": round(error, 2),
            "base_x": round(base_xyz_1[0], 2),
            "base_y": round(base_xyz_1[1], 2),
            "depth_mm": round(depth_mm_1, 1),
            "label": det["label"],
            "timestamp": datetime.now().isoformat(),
        }
        data_points.append(point)

        print(f"\n  ✅ #{len(data_points)} 기록")
        print(f"     물체    = {det['label']}")
        print(f"     예측 Z  = {predicted_z:.2f}mm")
        print(f"     실측 Z  = {actual_z:.2f}mm")
        print(f"     오차    = {error:+.2f}mm")

        # 스캔 위치로 복귀
        move_j(SCAN1_J, "스캔 복귀")


def _yolo_click_phase(data_points):
    """
    카메라 피드 + YOLO bbox 표시. bbox 클릭 시 해당 물체 정보 반환.
    """
    click_point = [None]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_point[0] = (x, y)

    cv2.namedWindow("Z Calibration - YOLO")
    cv2.setMouseCallback("Z Calibration - YOLO", on_mouse)

    detections = []
    last_detect_time = 0
    detect_interval = 0.3  # YOLO 추론 간격 (초)

    while True:
        color_np, depth_frame = capture_frame()
        vis = color_np.copy()

        # 주기적으로 YOLO 감지 (매 프레임은 무거움)
        now = time.time()
        if now - last_detect_time > detect_interval:
            detections = detect_objects(color_np)
            last_detect_time = now

        # bbox 그리기
        draw_detections(vis, detections)

        # 상태 표시
        cv2.putText(vis, f"Pts:{len(data_points)} | Click bbox | q/s/u",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(vis, f"Detected: {len(detections)} objects",
                    (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Z Calibration - YOLO", vis)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            cv2.destroyAllWindows()
            return "quit"
        elif key == ord('s'):
            cv2.destroyAllWindows()
            return "save"
        elif key == ord('u'):
            cv2.destroyAllWindows()
            return "undo"

        # 클릭 처리
        if click_point[0] is not None:
            cx, cy = click_point[0]
            click_point[0] = None

            idx = find_clicked_bbox(detections, cx, cy)
            if idx is None:
                print(f"[경고] ({cx},{cy})에 bbox 없음. bbox 안을 클릭하세요.")
                continue

            det = detections[idx]
            obj_cx, obj_cy = det["center"]

            # 선택된 bbox의 depth
            depth_mm = get_depth_at(depth_frame, obj_cx, obj_cy)
            if depth_mm <= 0:
                print(f"[경고] depth=0 at {det['label']}. 다시 시도.")
                continue

            # 1차 베이스 좌표 계산
            base_xyz = pixel_to_base(obj_cx, obj_cy, depth_mm)

            cv2.destroyAllWindows()
            return {
                "detection": det,
                "depth_mm": depth_mm,
                "base_xyz": base_xyz,
            }


def _save_data(data_points):
    """수집 데이터 저장"""
    save_data = {
        "created": datetime.now().isoformat(),
        "total_points": len(data_points),
        "points": data_points,
    }
    with open(Z_CALIB_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    errors = [p["error"] for p in data_points]
    print(f"\n  ── 수집 요약 ──")
    print(f"  포인트 수 : {len(data_points)}")
    print(f"  평균 오차 : {np.mean(errors):+.2f}mm")
    print(f"  표준편차   : {np.std(errors):.2f}mm")
    print(f"  최소 오차 : {np.min(errors):+.2f}mm")
    print(f"  최대 오차 : {np.max(errors):+.2f}mm")
    print(f"  저장: {Z_CALIB_DATA_PATH}")


# =====================================================================
# 보정 모델 피팅
# =====================================================================
def fit_model():
    """actual_z = scale * predicted_z + offset (1차 선형 회귀)"""
    if not os.path.exists(Z_CALIB_DATA_PATH):
        print(f"[오류] 수집 데이터 없음: {Z_CALIB_DATA_PATH}")
        return

    with open(Z_CALIB_DATA_PATH, "r") as f:
        points = json.load(f)["points"]

    if len(points) < MIN_POINTS:
        print(f"[오류] 최소 {MIN_POINTS}포인트 필요 (현재: {len(points)})")
        return

    pred   = np.array([p["predicted_z"] for p in points])
    actual = np.array([p["actual_z"] for p in points])

    A = np.vstack([pred, np.ones(len(pred))]).T
    scale, offset = np.linalg.lstsq(A, actual, rcond=None)[0]

    fitted    = scale * pred + offset
    residuals = actual - fitted
    rmse      = np.sqrt(np.mean(residuals ** 2))
    max_err   = np.max(np.abs(residuals))

    model = {
        "type": "linear",
        "scale": round(float(scale), 6),
        "offset": round(float(offset), 2),
        "rmse": round(float(rmse), 2),
        "max_residual": round(float(max_err), 2),
        "num_points": len(points),
        "created": datetime.now().isoformat(),
        "formula": f"corrected_z = {scale:.6f} * predicted_z + ({offset:+.2f})",
    }

    with open(Z_CALIB_MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("  Z 보정 모델 피팅 완료")
    print("=" * 60)
    print(f"\n  모델: corrected_z = {scale:.6f} × predicted_z + ({offset:+.2f})")
    print(f"\n  scale  = {scale:.6f}")
    print(f"  offset = {offset:+.2f}mm")
    print(f"  RMSE   = {rmse:.2f}mm")
    print(f"  최대잔차 = {max_err:.2f}mm")
    print(f"  포인트 = {len(points)}")

    print(f"\n  {'#':>3}  {'예측Z':>10}  {'실측Z':>10}  {'보정Z':>10}  {'잔차':>8}")
    print(f"  {'─'*3}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}")
    for i, p in enumerate(points):
        corrected = scale * p["predicted_z"] + offset
        res = p["actual_z"] - corrected
        print(f"  {i+1:>3}  {p['predicted_z']:>10.2f}  {p['actual_z']:>10.2f}  "
              f"{corrected:>10.2f}  {res:>+8.2f}")

    print(f"\n  저장: {Z_CALIB_MODEL_PATH}")


# =====================================================================
# 검증
# =====================================================================
def verify():
    """보정 모델 적용 후 검증"""
    if not os.path.exists(Z_CALIB_MODEL_PATH):
        print(f"[오류] 보정 모델 없음: {Z_CALIB_MODEL_PATH}")
        return

    with open(Z_CALIB_MODEL_PATH, "r") as f:
        model = json.load(f)
    scale, offset = model["scale"], model["offset"]
    print(f"\n[보정 모델] corrected_z = {scale:.6f} × z + ({offset:+.2f})")

    move_j(HOME_J,  "홈")
    move_j(SCAN1_J, "스캔 위치")

    click_point = [None]
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_point[0] = (x, y)

    cv2.namedWindow("Z Verify")
    cv2.setMouseCallback("Z Verify", on_mouse)

    detections = []
    last_t = 0

    while True:
        color_np, depth_frame = capture_frame()
        vis = color_np.copy()

        if time.time() - last_t > 0.3:
            detections = detect_objects(color_np)
            last_t = time.time()

        draw_detections(vis, detections)
        cv2.putText(vis, "Click bbox to verify | q=quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow("Z Verify", vis)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            move_j(HOME_J, "홈 복귀")
            break

        if click_point[0] is not None:
            cx, cy = click_point[0]
            click_point[0] = None

            idx = find_clicked_bbox(detections, cx, cy)
            if idx is None:
                continue

            det = detections[idx]
            obj_cx, obj_cy = det["center"]
            depth_mm = get_depth_at(depth_frame, obj_cx, obj_cy)
            if depth_mm <= 0:
                continue

            base_xyz = pixel_to_base(obj_cx, obj_cy, depth_mm)

            predicted_z = base_xyz[2]
            corrected_z = scale * predicted_z + offset

            print(f"[검증] {det['label']}: 예측Z={predicted_z:.1f} → 보정Z={corrected_z:.1f}")

            scan_pose = get_stable_pose(samples=3, interval=0.05)
            move_l([base_xyz[0], base_xyz[1], corrected_z + 30,
                    scan_pose[3], scan_pose[4], scan_pose[5]], "검증 접근")
            time.sleep(2.0)
            move_j(SCAN1_J, "스캔 복귀")

    cv2.destroyAllWindows()


# =====================================================================
# Z 보정 헬퍼 (import용)
# =====================================================================
def load_z_correction(path=Z_CALIB_MODEL_PATH):
    if not os.path.exists(path):
        print(f"[Z보정] 모델 없음 → 보정 없이 사용")
        return lambda z: z
    with open(path, "r") as f:
        m = json.load(f)
    s, o = m["scale"], m["offset"]
    print(f"[Z보정] corrected_z = {s:.6f} × z + ({o:+.2f})  RMSE={m['rmse']}mm")
    return lambda z: s * z + o


# =====================================================================
# 엔트리 포인트
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Z축 캘리브레이션 (YOLO-World)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--collect", action="store_true", help="수집 모드")
    group.add_argument("--fit",     action="store_true", help="보정 모델 피팅")
    group.add_argument("--verify",  action="store_true", help="검증 모드")

    parser.add_argument("--classes", type=str, default="cup,bottle,box,part",
                        help="YOLO-World 감지 클래스 (콤마 구분)")
    args = parser.parse_args()

    if args.fit:
        fit_model()
        return

    # YOLO-World 초기화
    classes = [c.strip() for c in args.classes.split(",")]
    init_yolo(classes)

    try:
        if args.collect:
            collect_data()
        elif args.verify:
            verify()
    except KeyboardInterrupt:
        print("\n[중단]")
        move_j(HOME_J, "홈 복귀")
    finally:
        rs_pipeline.stop()
        print("[종료] 카메라 해제.")


if __name__ == "__main__":
    main()