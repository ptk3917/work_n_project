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

# ── XYZ 캘리브레이션 파일 ──
Z_CALIB_DATA_PATH  = "xyz_calib_data.json"
Z_CALIB_MODEL_PATH = "xyz_correction.json"
DOWN_POSE_PATH     = "down_pose.json"       # 수직 하강 자세 (Rx, Ry, Rz)
SCAN_DOWN_PATH     = "scan_down_pose.json"  # 수직 스캔 위치 (관절 각도)

# ── 파라미터 ──
APPROACH_HEIGHT_OFFSET = 120  # 예측 Z 위에서 접근 (mm)
Z_FLOOR = 150                 # 바닥 Z 좌표 (mm) — 이 아래로 절대 안 내려감
MIN_POINTS = 4
YOLO_CONFIDENCE = 0.5         # YOLO 감지 최소 신뢰도 (낮추면 더 많이 감지, 높이면 안정)

# ── 클래스별 색상 (BGR) ──
CLASS_COLORS = {
    "ohyes":      (0, 255, 0),    # 초록
    "freetime_y": (0, 255, 255),  # 노랑
    "freetime_b": (255, 150, 0),  # 파랑
}

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
# 수직 하강 자세 (지면과 90도)
# =====================================================================
def teach_scan_pose():
    """수직 스캔 위치 티칭 — 카메라가 지면과 수직."""
    print("\n" + "=" * 60)
    print("  수직 스캔 위치 티칭 (카메라 수직)")
    print("  ─────────────────────────────")
    print("  카메라가 지면을 수직으로 내려다보는 위치.")
    print("  작업 영역 전체가 카메라에 보이는 높이로.")
    print("  [SPACE] 저장 | [Q] 취소")
    print("=" * 60)

    while True:
        color_np, _ = capture_frame()
        vis = color_np.copy()

        cv2.putText(vis, "SCAN: Camera vertical (looking down)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(vis, "[SPACE] Save | [Q] Cancel",
                    (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        h, w = vis.shape[:2]
        cv2.line(vis, (w//2 - 30, h//2), (w//2 + 30, h//2), (0, 0, 255), 1)
        cv2.line(vis, (w//2, h//2 - 30), (w//2, h//2 + 30), (0, 0, 255), 1)

        cv2.imshow("Teach Scan Pose", vis)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(' '):
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            print("[취소]")
            return

    cv2.destroyAllWindows()

    # 관절 각도 저장
    tcp = get_stable_pose(samples=5, interval=0.05)
    _, joints = robot.GetActualJointPosDegree(0)
    if isinstance(joints, (list, tuple)):
        joints = list(joints)
    else:
        joints = [0, 0, 0, 0, 0, 0]

    scan_pose = {
        "joints": [round(j, 4) for j in joints],
        "tcp": [round(v, 4) for v in tcp],
        "created": datetime.now().isoformat(),
    }
    with open(SCAN_DOWN_PATH, "w", encoding="utf-8") as f:
        json.dump(scan_pose, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ 스캔 위치 저장 완료")
    print(f"     관절: {[round(j,1) for j in joints]}")
    print(f"     TCP:  X={tcp[0]:.1f} Y={tcp[1]:.1f} Z={tcp[2]:.1f}")
    print(f"     Rx={tcp[3]:.2f} Ry={tcp[4]:.2f} Rz={tcp[5]:.2f}")
    print(f"     파일: {SCAN_DOWN_PATH}")

    move_j(HOME_J, "홈 복귀")


def teach_down_pose():
    """수직 접근 자세 티칭 — 그리퍼가 지면과 수직."""
    print("\n" + "=" * 60)
    print("  수직 접근 자세 티칭 (그리퍼 수직)")
    print("  ─────────────────────────────")
    print("  그리퍼가 지면과 수직(90도)으로 물체를 잡을 자세.")
    print("  XY, Z 위치는 상관없고 Rx/Ry/Rz만 중요합니다.")
    print("  [SPACE] 저장 | [Q] 취소")
    print("=" * 60)

    while True:
        color_np, _ = capture_frame()
        vis = color_np.copy()

        cv2.putText(vis, "PICK: Gripper vertical (pointing down)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(vis, "[SPACE] Save | [Q] Cancel",
                    (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.imshow("Teach Down Pose", vis)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(' '):
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            print("[취소]")
            return

    cv2.destroyAllWindows()

    tcp = get_stable_pose(samples=5, interval=0.05)
    rx, ry, rz = tcp[3], tcp[4], tcp[5]

    # J6 관절 각도 저장 (범위 체크용)
    _, joints = robot.GetActualJointPosDegree(0)
    j6 = joints[5] if isinstance(joints, (list, tuple)) else 0.0

    down_pose = {
        "rx": round(rx, 4),
        "ry": round(ry, 4),
        "rz": round(rz, 4),
        "j6": round(float(j6), 4),
        "created": datetime.now().isoformat(),
    }
    with open(DOWN_POSE_PATH, "w", encoding="utf-8") as f:
        json.dump(down_pose, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ 접근 자세 저장 완료")
    print(f"     Rx={rx:.2f}, Ry={ry:.2f}, Rz={rz:.2f}")
    print(f"     J6={j6:.2f}deg")
    print(f"     파일: {DOWN_POSE_PATH}")

    move_j(HOME_J, "홈 복귀")


def load_down_pose():
    """저장된 수직 자세 로드. Returns (rx, ry, rz, j6) or None."""
    if not os.path.exists(DOWN_POSE_PATH):
        return None
    with open(DOWN_POSE_PATH, "r") as f:
        dp = json.load(f)
    j6 = dp.get("j6", 0.0)
    print(f"[접근 자세] Rx={dp['rx']:.2f}, Ry={dp['ry']:.2f}, Rz={dp['rz']:.2f}, J6={j6:.2f}")
    return (dp["rx"], dp["ry"], dp["rz"], j6)


def load_scan_down():
    """저장된 수직 스캔 관절 각도 로드."""
    if not os.path.exists(SCAN_DOWN_PATH):
        return None
    with open(SCAN_DOWN_PATH, "r") as f:
        sp = json.load(f)
    joints = sp["joints"]
    print(f"[수직 스캔] joints={[round(j,1) for j in joints]}")
    return joints


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
# YOLO 감지 (YOLO-World / 커스텀 OBB 모두 지원)
# =====================================================================
USE_OBB = False  # OBB 모델 사용 여부 (--weights 지정 시 True)


def init_yolo(classes: list, weights: str = None):
    """
    YOLO 모델 로드.
    - weights 없으면: YOLO-World (open-vocab)
    - weights 있으면: 커스텀 OBB 모델
    """
    global yolo_model, USE_OBB

    if weights and os.path.exists(weights):
        print(f"커스텀 OBB 모델 로드: {weights}")
        yolo_model = YOLO(weights)
        USE_OBB = True
        print(f"✅ OBB 모델 준비 완료 (classes: {list(yolo_model.names.values())})")
    else:
        print(f"YOLO-World 로드 중... (classes: {classes})")
        yolo_model = YOLO("yolov8s-worldv2.pt")
        yolo_model.set_classes(classes)
        USE_OBB = False
        print("✅ YOLO-World 준비 완료")


def detect_objects(color_np):
    """
    현재 프레임에서 감지. OBB/일반 모델 자동 분기.
    Returns:
        list of dict: [{
            "bbox": [x1,y1,x2,y2],     # axis-aligned bbox (클릭 판정용)
            "label": str,
            "conf": float,
            "center": (cx, cy),
            "angle": float,              # OBB일 때 회전 각도 (rad), 일반은 0
            "corners": np.array or None, # OBB 4코너 (int), 일반은 None
        }, ...]
    """
    results = yolo_model.predict(color_np, conf=YOLO_CONFIDENCE, verbose=False)
    detections = []

    for r in results:
        # ── OBB 모델 ──
        if USE_OBB and r.obb is not None and len(r.obb) > 0:
            for box in r.obb:
                xywhr = box.xywhr[0].cpu().numpy()
                cx, cy, w, h, angle = xywhr
                conf   = float(box.conf[0])
                cls_id = int(box.cls[0])
                label  = yolo_model.names.get(cls_id, f"cls_{cls_id}")

                # 4코너 좌표
                corners = box.xyxyxyxy[0].cpu().numpy().astype(int)

                # axis-aligned bbox (클릭 판정용)
                xs, ys = corners[:, 0], corners[:, 1]
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "label": label,
                    "conf": conf,
                    "center": (int(cx), int(cy)),
                    "angle": float(angle),
                    "corners": corners,
                })

        # ── 일반 모델 (YOLO-World 등) ──
        elif r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                label  = yolo_model.names.get(cls_id, f"cls_{cls_id}")
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "label": label,
                    "conf": conf,
                    "center": (cx, cy),
                    "angle": 0.0,
                    "corners": None,
                })

    return detections


def draw_detections(vis, detections, selected_idx=None):
    """bbox/OBB 시각화 — 클래스별 색상, 꼭짓점 라벨"""
    for i, det in enumerate(detections):
        base_color = CLASS_COLORS.get(det["label"], (0, 255, 0))
        color = (0, 255, 255) if i == selected_idx else base_color
        thickness = 3 if i == selected_idx else 2

        if det["corners"] is not None:
            cv2.polylines(vis, [det["corners"]], True, color, thickness)
            angle_deg = np.degrees(det["angle"])
            label = f"[{i}] {det['label']} {det['conf']:.2f} {angle_deg:.0f}deg"
            # 라벨 위치: OBB 가장 위쪽 꼭짓점 기준
            top_idx = det["corners"][:, 1].argmin()
            lx = int(det["corners"][top_idx, 0])
            ly = int(det["corners"][top_idx, 1]) - 8
        else:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            label = f"[{i}] {det['label']} {det['conf']:.2f}"
            lx, ly = x1, y1 - 8

        # 중심점
        cx, cy = det["center"]
        cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)

        # 라벨 배경 + 텍스트
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        # 화면 밖 방지
        lx = max(0, min(lx, vis.shape[1] - tw))
        ly = max(th + 4, ly)
        cv2.rectangle(vis, (lx, ly - th - 4), (lx + tw + 2, ly), color, -1)
        cv2.putText(vis, label, (lx + 1, ly - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return vis


def find_clicked_bbox(detections, click_x, click_y):
    """클릭 좌표가 어떤 bbox/OBB 안에 있는지 확인"""
    for i, det in enumerate(detections):
        if det["corners"] is not None:
            # OBB: point-in-polygon
            result = cv2.pointPolygonTest(
                det["corners"].astype(np.float32),
                (float(click_x), float(click_y)), False)
            if result >= 0:
                return i
        else:
            # 일반 bbox
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
    print("  XYZ 캘리브레이션 — 수직 스캔 + OBB 감지")
    print("  ─────────────────────────────")
    print("  [카메라] 수직 위에서 bbox 클릭 → 상공 접근")
    print("  [터미널] 조그 XYZ 조정 → Enter 기록")
    print("  [터미널] 's'=저장 | 'q'=종료 | 'u'=undo")
    print("=" * 60)

    # 수직 자세/스캔 위치 로드
    down_data = load_down_pose()
    if down_data is None:
        print("\n[오류] 접근 자세 미등록! 먼저 --teach-down 실행하세요.")
        return
    down_rx, down_ry, down_rz, base_j6 = down_data

    scan_down_j = load_scan_down()
    if scan_down_j is None:
        print("\n[오류] 스캔 위치 미등록! (scan_down_pose.json 없음)")
        return

    # 기존 데이터 로드
    data_points = []
    if os.path.exists(Z_CALIB_DATA_PATH):
        with open(Z_CALIB_DATA_PATH, "r") as f:
            data_points = json.load(f).get("points", [])
        print(f"\n[안내] 기존 {len(data_points)}포인트 로드됨.")

    # 스캔 위치로 이동 (수직)
    move_j(HOME_J,      "홈")
    move_j(scan_down_j, "스캔 위치")

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
        depth_mm   = click_result["depth_mm"]
        base_xyz   = click_result["base_xyz"]

        print(f"\n[측정] {det['label']} (conf={det['conf']:.2f}) — 수직 스캔")
        print(f"  depth: {depth_mm:.1f}mm")
        print(f"  베이스: X={base_xyz[0]:.1f} Y={base_xyz[1]:.1f} Z={base_xyz[2]:.1f}")

        predicted_z = base_xyz[2]
        rx, ry, rz = down_rx, down_ry, down_rz

        # 접근 Z (바닥 안전 제한)
        approach_z = predicted_z + APPROACH_HEIGHT_OFFSET
        if approach_z < Z_FLOOR + APPROACH_HEIGHT_OFFSET:
            print(f"  ⚠ 예측 Z={predicted_z:.1f} < 바닥({Z_FLOOR}) → 안전 높이")
            approach_z = Z_FLOOR + APPROACH_HEIGHT_OFFSET

        approach_pose = [
            base_xyz[0], base_xyz[1], approach_z,
            rx, ry, rz
        ]

        print(f"[이동] 접근 Z={approach_z:.1f}mm")
        move_l(approach_pose, "상공 접근")

        # 터미널에서 조그 대기
        print(f"\n{'─'*45}")
        print(f"  상공 도착.")
        print(f"  예측: X={base_xyz[0]:.1f} Y={base_xyz[1]:.1f} Z={predicted_z:.1f}")
        print(f"  접근: Z={approach_z:.1f}mm")
        print(f"")
        print(f"  ▶ 티칭펜던트로 XY + Z 조정")
        print(f"    - 아래로 내려서 물체에 닿게")
        print(f"    - 좌우/앞뒤도 맞춰서 정중앙에")
        print(f"  ▶ 다 맞추면 Enter")
        print(f"  ▶ 'c' = 이 포인트 취소")
        print(f"{'─'*45}")

        user_input = input("  >> ").strip().lower()

        if user_input == 'c':
            print("[취소] 건너뜀.")
            move_j(scan_down_j, "스캔 복귀")
            continue

        # TCP XYZ 읽기 (카메라 무관)
        tcp_actual = get_stable_pose(samples=5, interval=0.05)
        actual_x, actual_y, actual_z = tcp_actual[0], tcp_actual[1], tcp_actual[2]
        error_x = actual_x - base_xyz[0]
        error_y = actual_y - base_xyz[1]
        error_z = actual_z - predicted_z

        point = {
            "index": len(data_points) + 1,
            "predicted_x": round(base_xyz[0], 2),
            "predicted_y": round(base_xyz[1], 2),
            "predicted_z": round(predicted_z, 2),
            "actual_x": round(actual_x, 2),
            "actual_y": round(actual_y, 2),
            "actual_z": round(actual_z, 2),
            "error_x": round(error_x, 2),
            "error_y": round(error_y, 2),
            "error_z": round(error_z, 2),
            "depth_mm": round(depth_mm, 1),
            "label": det["label"],
            "angle_rad": round(det.get("angle", 0.0), 4),
            "timestamp": datetime.now().isoformat(),
        }
        data_points.append(point)

        print(f"\n  ✅ #{len(data_points)} 기록")
        print(f"     물체    = {det['label']}")
        print(f"     예측 X={base_xyz[0]:.1f}  Y={base_xyz[1]:.1f}  Z={predicted_z:.1f}")
        print(f"     실측 X={actual_x:.1f}  Y={actual_y:.1f}  Z={actual_z:.1f}")
        print(f"     오차 X={error_x:+.1f}  Y={error_y:+.1f}  Z={error_z:+.1f}mm")

        # 스캔 위치로 복귀
        move_j(scan_down_j, "스캔 복귀")


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
    detect_interval = 0.3

    while True:
        color_np, depth_frame = capture_frame()
        vis = color_np.copy()

        now = time.time()
        if now - last_detect_time > detect_interval:
            detections = detect_objects(color_np)
            last_detect_time = now

        draw_detections(vis, detections)

        cv2.putText(vis, f"Pts:{len(data_points)} | Click bbox | q/s/u/d",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 클래스별 색상 범례
        legend_y = vis.shape[0] - 15
        for cls_name, cls_color in CLASS_COLORS.items():
            cv2.circle(vis, (15, legend_y), 5, cls_color, -1)
            cv2.putText(vis, cls_name, (25, legend_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, cls_color, 1)
            legend_y -= 20

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
        elif key == ord('d'):
            # 현재 감지 목록 출력
            if detections:
                print(f"\n  감지 {len(detections)}개:")
                for i, d in enumerate(detections):
                    angle_str = f" {np.degrees(d['angle']):.0f}deg" if d['angle'] != 0 else ""
                    print(f"    [{i}] {d['label']} (conf={d['conf']:.2f}{angle_str})")
            else:
                print("  감지 없음")

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

    errors_x = [p["error_x"] for p in data_points]
    errors_y = [p["error_y"] for p in data_points]
    errors_z = [p["error_z"] for p in data_points]
    print(f"\n  ── 수집 요약 ──")
    print(f"  포인트 수: {len(data_points)}")
    print(f"  평균 오차: X={np.mean(errors_x):+.1f}  Y={np.mean(errors_y):+.1f}  Z={np.mean(errors_z):+.1f}mm")
    print(f"  표준편차:  X={np.std(errors_x):.1f}  Y={np.std(errors_y):.1f}  Z={np.std(errors_z):.1f}mm")
    print(f"  저장: {Z_CALIB_DATA_PATH}")


# =====================================================================
# 보정 모델 피팅
# =====================================================================
def fit_model():
    """XYZ 각 축별 1차 선형 회귀: actual = scale * predicted + offset"""
    if not os.path.exists(Z_CALIB_DATA_PATH):
        print(f"[오류] 수집 데이터 없음: {Z_CALIB_DATA_PATH}")
        return

    with open(Z_CALIB_DATA_PATH, "r") as f:
        points = json.load(f)["points"]

    if len(points) < MIN_POINTS:
        print(f"[오류] 최소 {MIN_POINTS}포인트 필요 (현재: {len(points)})")
        return

    # 기존 z_calib_data.json 호환 (predicted_x가 없는 경우)
    has_xy = "predicted_x" in points[0]

    model = {
        "type": "linear_xyz",
        "num_points": len(points),
        "created": datetime.now().isoformat(),
        "axes": {},
    }

    print("\n" + "=" * 60)
    print("  XYZ 보정 모델 피팅")
    print("=" * 60)

    axes = ["x", "y", "z"] if has_xy else ["z"]

    for axis in axes:
        pred_key = f"predicted_{axis}"
        act_key  = f"actual_{axis}"

        pred   = np.array([p[pred_key] for p in points])
        actual = np.array([p[act_key] for p in points])

        A = np.vstack([pred, np.ones(len(pred))]).T
        scale, offset = np.linalg.lstsq(A, actual, rcond=None)[0]

        fitted    = scale * pred + offset
        residuals = actual - fitted
        rmse      = np.sqrt(np.mean(residuals ** 2))
        max_err   = np.max(np.abs(residuals))

        model["axes"][axis] = {
            "scale": round(float(scale), 6),
            "offset": round(float(offset), 2),
            "rmse": round(float(rmse), 2),
            "max_residual": round(float(max_err), 2),
        }

        print(f"\n  [{axis.upper()}축]")
        print(f"    corrected_{axis} = {scale:.6f} * predicted_{axis} + ({offset:+.2f})")
        print(f"    RMSE = {rmse:.2f}mm | 최대잔차 = {max_err:.2f}mm")

    with open(Z_CALIB_MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)

    # 포인트별 상세 결과
    print(f"\n  ── 포인트별 결과 ──")
    if has_xy:
        print(f"  {'#':>3}  {'errX':>7} {'errY':>7} {'errZ':>7}  →  {'resX':>7} {'resY':>7} {'resZ':>7}")
        print(f"  {'─'*3}  {'─'*7} {'─'*7} {'─'*7}     {'─'*7} {'─'*7} {'─'*7}")
        for i, p in enumerate(points):
            res = {}
            for ax in axes:
                m = model["axes"][ax]
                corrected = m["scale"] * p[f"predicted_{ax}"] + m["offset"]
                res[ax] = p[f"actual_{ax}"] - corrected
            raw_err = f"X{p['error_x']:+.1f} Y{p['error_y']:+.1f} Z{p['error_z']:+.1f}"
            fit_res = f"X{res['x']:+.1f} Y{res['y']:+.1f} Z{res['z']:+.1f}"
            print(f"  {i+1:>3}  {raw_err}  →  {fit_res}")
    else:
        print(f"  {'#':>3}  {'예측Z':>10}  {'실측Z':>10}  {'보정Z':>10}  {'잔차':>8}")
        m = model["axes"]["z"]
        for i, p in enumerate(points):
            corrected = m["scale"] * p["predicted_z"] + m["offset"]
            res = p["actual_z"] - corrected
            print(f"  {i+1:>3}  {p['predicted_z']:>10.2f}  {p['actual_z']:>10.2f}  "
                  f"{corrected:>10.2f}  {res:>+8.2f}")

    print(f"\n  저장: {Z_CALIB_MODEL_PATH}")
    print(f"  포인트: {len(points)}개")


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

    axes_model = model["axes"]
    print(f"\n[보정 모델]")
    for ax, m in axes_model.items():
        print(f"  {ax.upper()}: scale={m['scale']:.6f}, offset={m['offset']:+.2f}, RMSE={m['rmse']}mm")

    down_data = load_down_pose()
    if down_data is None:
        print("[오류] 접근 자세 미등록! --teach-down 먼저 실행.")
        return
    down_rx, down_ry, down_rz, base_j6 = down_data

    scan_down_j = load_scan_down()
    if scan_down_j is None:
        print("[오류] 스캔 위치 미등록! (scan_down_pose.json 없음)")
        return

    move_j(HOME_J,      "홈")
    move_j(scan_down_j, "스캔 위치")

    click_point = [None]
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_point[0] = (x, y)

    cv2.namedWindow("XYZ Verify")
    cv2.setMouseCallback("XYZ Verify", on_mouse)

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
        cv2.imshow("XYZ Verify", vis)
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

            # XYZ 보정 적용
            corrected = list(base_xyz)
            for i, ax in enumerate(["x", "y", "z"]):
                if ax in axes_model:
                    m = axes_model[ax]
                    corrected[i] = m["scale"] * base_xyz[i] + m["offset"]

            print(f"[검증] {det['label']}:")
            print(f"  예측: X={base_xyz[0]:.1f} Y={base_xyz[1]:.1f} Z={base_xyz[2]:.1f}")
            print(f"  보정: X={corrected[0]:.1f} Y={corrected[1]:.1f} Z={corrected[2]:.1f}")

            scan_pose = get_stable_pose(samples=3, interval=0.05)
            move_l([corrected[0], corrected[1], corrected[2] + 30,
                    down_rx, down_ry, down_rz], "검증 접근")
            time.sleep(2.0)
            move_j(scan_down_j, "스캔 복귀")

    cv2.destroyAllWindows()


# =====================================================================
# XYZ 보정 헬퍼 (다른 파이프라인에서 import용)
# =====================================================================
def load_xyz_correction(path=Z_CALIB_MODEL_PATH):
    """
    XYZ 보정 모델 로드.
    Returns:
        callable: (x, y, z) → (corrected_x, corrected_y, corrected_z)
    """
    if not os.path.exists(path):
        print(f"[XYZ보정] 모델 없음 → 보정 없이 사용")
        return lambda x, y, z: (x, y, z)

    with open(path, "r") as f:
        model = json.load(f)

    axes = model["axes"]
    print(f"[XYZ보정] 로드 완료 ({len(axes)}축)")
    for ax, m in axes.items():
        print(f"  {ax.upper()}: scale={m['scale']:.4f}, offset={m['offset']:+.1f}, RMSE={m['rmse']}mm")

    def correct(x, y, z):
        cx = axes["x"]["scale"] * x + axes["x"]["offset"] if "x" in axes else x
        cy = axes["y"]["scale"] * y + axes["y"]["offset"] if "y" in axes else y
        cz = axes["z"]["scale"] * z + axes["z"]["offset"] if "z" in axes else z
        return (cx, cy, cz)

    return correct


# =====================================================================
# 엔트리 포인트
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="XYZ 캘리브레이션 (YOLO OBB + 수직 접근)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--collect",    action="store_true", help="수집 모드")
    group.add_argument("--fit",        action="store_true", help="보정 모델 피팅")
    group.add_argument("--verify",     action="store_true", help="검증 모드")
    group.add_argument("--teach-down", action="store_true", help="접근 자세 티칭 (그리퍼 수직)")

    parser.add_argument("--classes", type=str, default="ohyes,freetime_y,freetime_b",
                        help="감지 클래스 (콤마 구분, --weights 미사용 시)")
    parser.add_argument("--weights", type=str,
                        default=os.path.expanduser("~/vlm_agent/qwen_ver1/obb_dataset/train/weights/best.pt"),
                        help="커스텀 OBB 모델 가중치 경로")
    args = parser.parse_args()

    if args.fit:
        fit_model()
        return

    if args.teach_down:
        teach_down_pose()
        return

    # YOLO 초기화 (collect, verify에서 사용)
    classes = [c.strip() for c in args.classes.split(",")]
    init_yolo(classes, weights=args.weights)

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