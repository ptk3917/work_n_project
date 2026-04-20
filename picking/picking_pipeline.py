"""
FR5 + D455 + Qwen2.5-VL 통합 픽킹 파이프라인
스캔 → 감지 → depth 기반 Z 자동 계산 → 픽킹

실행: python picking_pipeline.py
"""
import sys
import os
sys.path.insert(0, os.path.expanduser("~/vlm_agent/qwen_ver1"))

import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from fairino import Robot
import torch
import re
import time
import threading


# ══════════════════════════════════════════════════════
# 설정값
# ══════════════════════════════════════════════════════
ROBOT_IP      = "192.168.58.3"
TOOL_NO       = 1
USER_NO       = 0
SPEED         = 20
SPEED_J       = 30

# 그리퍼 오프셋: 물체 Z 좌표에서 얼마나 더 내려갈지 (mm)
# 물체 크기나 그리퍼 길이에 따라 조정
GRIPPER_OFFSET = 30

# 접근/복귀 높이
APPROACH_DIST = 80    # 물체 위 접근 높이 (mm)
LIFT_DIST     = 120   # 픽킹 후 들어올리기 (mm)

FX = 386.4894
FY = 386.6551
CX = 325.7376
CY = 246.5258

T_CAM_TO_EE = np.load(os.path.expanduser("~/vlm_agent/qwen_ver1/T_cam_to_ee.npy"))

# ── 티칭 포인트 ───────────────────────────────────────
HOME_J = [2.0,  -98.0,  100.0,  -94.0,  -84.0, -111.0]
WAY_J  = [2.69, -128.26, 129.33, -94.03, -84.34, -111.73]

SCAN_POINTS_J = [
    [-10.26, -169.9,  145.79, -111.88,  -76.53, -111.69],  # 스캔 1
    [ 65.05,  -51.08,  64.13,  -34.35, -139.49,  -10.39],  # 스캔 2
    [ 19.89,  -46.15,  35.04,  -37.62,  -95.72,  -82.49],  # 스캔 3
    [-43.46,  -58.49,  82.47,  -53.5,   -53.0,  -155.16],  # 스캔 4
]

CONFIRM_COUNT = 1   # 재확인 최소 횟수

# VLM Lock: 스레드 간 GPU 동시접근 방지
vlm_lock = threading.Lock()


# ══════════════════════════════════════════════════════
# 1. 모델 로드
# ══════════════════════════════════════════════════════
print("=" * 50)
print("모델 로드 중...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="auto",
    max_memory={0: "14GiB", "cpu": "16GiB"}
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
print(f"✅ 모델 로드 완료 | VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")


# ══════════════════════════════════════════════════════
# 2. D455 초기화
# ══════════════════════════════════════════════════════
rs_pipeline = rs.pipeline()
config       = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
rs_pipeline.start(config)
align = rs.align(rs.stream.color)

print("카메라 안정화 대기 중...")
for _ in range(30):
    rs_pipeline.wait_for_frames()
print("✅ D455 초기화 완료")


# ══════════════════════════════════════════════════════
# 3. FR5 + 그리퍼 초기화
# ══════════════════════════════════════════════════════
print(f"FR5 연결 중... ({ROBOT_IP})")
robot = Robot.RPC(ROBOT_IP)
ret, error = robot.GetRobotErrorCode()
if ret != 0:
    print(f"⚠ 로봇 에러: {ret}")
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
print("=" * 50)


# ══════════════════════════════════════════════════════
# 로봇 이동 함수
# ══════════════════════════════════════════════════════
def move_j(joints, label=""):
    if label:
        print(f"  {label}...")
    ret = robot.MoveJ(joints, TOOL_NO, USER_NO, vel=SPEED_J, acc=SPEED_J, blendT=0)
    err = ret[0] if isinstance(ret, (list, tuple)) else ret
    if err != 0:
        raise RuntimeError(f"{label} 실패: {ret}")
    time.sleep(0.8)


def move_l(pose, label=""):
    if label:
        print(f"  {label}...")
    ret = robot.MoveL(pose, TOOL_NO, USER_NO, vel=SPEED, acc=SPEED)
    err = ret[0] if isinstance(ret, (list, tuple)) else ret
    if err != 0:
        raise RuntimeError(f"{label} 실패: {ret}")
    time.sleep(0.5)


def gripper_open():
    robot.MoveGripper(index=1, pos=100, vel=50, force=50,
                      maxtime=5000, block=1, type=0,
                      rotNum=0, rotVel=0, rotTorque=0)
    time.sleep(0.5)


def gripper_close():
    robot.MoveGripper(index=1, pos=30, vel=50, force=50,
                      maxtime=5000, block=1, type=0,
                      rotNum=0, rotVel=0, rotTorque=0)
    time.sleep(0.5)


def go_home():
    move_j(WAY_J,  "경유점 복귀")
    move_j(HOME_J, "홈 복귀")


# ══════════════════════════════════════════════════════
# VLM 함수
# ══════════════════════════════════════════════════════
def capture_frame():
    frames      = rs_pipeline.wait_for_frames()
    aligned     = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    color_np    = np.asanyarray(color_frame.get_data())
    rgb_pil     = Image.fromarray(cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB))
    return rgb_pil, color_np, depth_frame


def extract_bbox(image: Image.Image, target: str) -> dict:
    w, h = image.size
    prompt = f"""Detect '{target}' in the image and return bounding box as JSON.
Image size: {w}x{h} pixels.

Output ONLY this JSON, no other text:
{{"found": true, "label": "object name", "bbox": [x_min, y_min, x_max, y_max], "confidence": "high/medium/low"}}

If not found:
{{"found": false, "label": "", "bbox": [], "confidence": "none"}}"""

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": prompt}
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, return_tensors="pt"
    ).to("cuda")

    with vlm_lock:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    generated = out[:, inputs.input_ids.shape[1]:]
    raw = processor.batch_decode(generated, skip_special_tokens=True)[0]

    raw_clean   = re.sub(r"```json|```", "", raw).strip()
    bbox_match  = re.search(r'"bbox(?:_2d)?"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', raw_clean)
    label_match = re.search(r'"label"\s*:\s*"([^"]+)"', raw_clean)
    conf_match  = re.search(r'"confidence"\s*:\s*"([^"]+)"', raw_clean)

    if bbox_match:
        bbox  = [int(bbox_match.group(i)) for i in range(1, 5)]
        label = label_match.group(1) if label_match else target
        conf  = conf_match.group(1)  if conf_match  else "high"
        return {"found": True, "label": label, "bbox": bbox, "confidence": conf}

    return {"found": False, "bbox": [], "confidence": "none", "label": "parse_error"}


def get_depth_mm(depth_frame, bbox):
    """bbox 중심 5x5 영역 중앙값 depth (mm)"""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    depths = []
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            px = max(0, min(cx + dx, 639))
            py = max(0, min(cy + dy, 479))
            d  = depth_frame.get_distance(px, py) * 1000
            if d > 0:
                depths.append(d)
    return float(np.median(depths)) if depths else 0.0


def pixel_to_camera_coords(bbox, depth_mm):
    cx_px = (bbox[0] + bbox[2]) / 2
    cy_px = (bbox[1] + bbox[3]) / 2
    Z = depth_mm
    X = (cx_px - CX) * Z / FX
    Y = (cy_px - CY) * Z / FY
    return np.array([X, Y, Z, 1.0])


def camera_to_robot_coords(p_cam, ee_pose):
    from scipy.spatial.transform import Rotation
    x, y, z, rx, ry, rz = ee_pose
    R = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
    T_ee_to_base = np.eye(4)
    T_ee_to_base[:3, :3] = R
    T_ee_to_base[:3,  3] = [x, y, z]
    p_ee   = T_CAM_TO_EE @ p_cam
    p_base = T_ee_to_base @ p_ee
    return p_base[:3]


# ══════════════════════════════════════════════════════
# 픽킹 시퀀스
# depth로 계산된 Z 좌표 + GRIPPER_OFFSET 만큼 내려감
# ══════════════════════════════════════════════════════
def do_pick(target_xyz):
    tx, ty, tz = target_xyz
    _, current = robot.GetActualTCPPose(1)
    rx, ry, rz = current[3], current[4], current[5]

    # depth로 계산된 tz = 물체 표면까지의 로봇 좌표 Z
    # GRIPPER_OFFSET만큼 더 내려가서 확실히 잡기
    pick_z = tz + GRIPPER_OFFSET

    print(f"  목표: X={tx:.1f} Y={ty:.1f} Z={tz:.1f} → 실제 픽 Z={pick_z:.1f}")
    print(f"  (depth 기반 자동 Z 계산, GRIPPER_OFFSET={GRIPPER_OFFSET}mm)")

    gripper_open()
    move_l([tx, ty, pick_z + APPROACH_DIST, rx, ry, rz], "[1/5] 물체 위 접근")
    move_l([tx, ty, pick_z,                 rx, ry, rz], "[2/5] 내려가기")
    print("  [3/5] 그립...")
    gripper_close()
    move_l([tx, ty, pick_z + LIFT_DIST,     rx, ry, rz], "[4/5] 들어올리기")
    move_j(WAY_J,  "[5/5] 경유점 복귀")
    move_j(HOME_J, "       홈 복귀")
    print("  ✅ 픽킹 완료!")


# ══════════════════════════════════════════════════════
# 스캔 시퀀스 (별도 스레드)
# ══════════════════════════════════════════════════════
scan_running = False
scan_status  = "대기 중"


def scan_and_pick(target):
    global scan_running, scan_status, last_result, last_depth, last_robot_xyz

    try:
        move_j(HOME_J, "홈 이동")
        move_j(WAY_J,  "경유점 이동")

        idx = 0
        while scan_running:
            scan_pt = idx % len(SCAN_POINTS_J)
            scan_no = scan_pt + 1
            scan_j  = SCAN_POINTS_J[scan_pt]
            idx    += 1

            scan_status = f"스캔 {scan_no}/{len(SCAN_POINTS_J)} (#{idx}회차)"
            print(f"\n[스캔] 포인트 {scan_no} 이동 ({idx}번째)")
            move_j(scan_j, f"스캔 포인트 {scan_no}")

            if not scan_running:
                break

            time.sleep(0.5)

            # 1차 감지
            print("  1차 감지...")
            rgb_pil, _, depth_frame = capture_frame()
            result = extract_bbox(rgb_pil, target)

            if not result.get("found") or not result.get("bbox"):
                print("  → 미감지, 다음으로")
                last_result = result
                continue

            depth_mm = get_depth_mm(depth_frame, result["bbox"])
            last_result = result
            last_depth  = depth_mm
            print(f"  → 감지! bbox={result['bbox']} depth={depth_mm:.0f}mm")

            # depth 0이면 재시도
            if depth_mm == 0:
                print("  → depth=0 읽기 실패, 재시도")
                continue

            # 재확인 (한 번만)
            scan_status = f"재확인 중..."
            rgb_pil, _, df2 = capture_frame()
            r2 = extract_bbox(rgb_pil, target)
            if not r2.get("found") or not r2.get("bbox"):
                print(f"  → 재확인 실패, 다음으로")
                continue
            depth_mm    = get_depth_mm(df2, r2["bbox"])
            last_result = r2
            last_depth  = depth_mm
            print(f"  → 재확인 성공! depth={depth_mm:.0f}mm")

            if depth_mm == 0:
                print("  → 재확인 depth=0, 다음으로")
                continue

            # 좌표 계산 (트래킹 없이 바로)
            _, ee_pose = robot.GetActualTCPPose(1)
            p_cam      = pixel_to_camera_coords(last_result["bbox"], last_depth)
            robot_xyz  = camera_to_robot_coords(p_cam, ee_pose)
            last_robot_xyz = robot_xyz
            print(f"  로봇좌표: X={robot_xyz[0]:.1f} Y={robot_xyz[1]:.1f} Z={robot_xyz[2]:.1f}")

            # 픽킹
            scan_status = "픽킹 중"
            do_pick(robot_xyz)
            scan_status  = "픽킹 완료 ✅"
            scan_running = False
            return

    except RuntimeError as e:
        print(f"\n[에러] {e}")
        scan_status = f"에러: {e}"
        try:
            go_home()
        except:
            pass
    finally:
        scan_running = False


# ══════════════════════════════════════════════════════
# GUI
# ══════════════════════════════════════════════════════
WINDOW_NAME = "Picking Pipeline"
CAM_H       = 480
CAM_W       = 640
INPUT_H     = 80
TOTAL_H     = CAM_H + INPUT_H

input_text   = ""
TARGET       = "cup or mug"
input_active = True
action_queue = []

last_result    = {"found": False, "bbox": [], "confidence": "none", "label": ""}
last_depth     = 0.0
last_robot_xyz = None


def draw_ui(canvas, target, text, active, status):
    cv2.rectangle(canvas, (0, 0), (CAM_W, INPUT_H), (30, 30, 30), -1)
    cv2.line(canvas, (0, INPUT_H), (CAM_W, INPUT_H), (80, 80, 80), 1)
    cv2.putText(canvas, f"Target: {target}",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 220, 180), 1)
    status_color = (0, 200, 255) if any(k in status for k in ["스캔","재확인"]) else \
                   (0, 255, 100) if "완료" in status else \
                   (0, 100, 255) if "픽킹" in status else \
                   (100, 100, 100)
    cv2.putText(canvas, f"Status: {status}",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 1)
    box_color = (0, 200, 255) if active else (80, 80, 80)
    cv2.rectangle(canvas, (8, 46), (CAM_W - 195, 70), box_color, 1)
    display_text = text + ("|" if active else "")
    cv2.putText(canvas, display_text,
                (14, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    if not text and not active:
        cv2.putText(canvas, "type target here...",
                    (14, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 80, 80), 1)
    scan_btn_color = (0, 60, 180) if scan_running else (180, 100, 0)
    scan_btn_text  = "STOP" if scan_running else "SCAN"
    cv2.rectangle(canvas, (CAM_W - 185, 46), (CAM_W - 130, 70), scan_btn_color, -1)
    cv2.putText(canvas, scan_btn_text,
                (CAM_W - 180, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    cv2.rectangle(canvas, (CAM_W - 125, 46), (CAM_W - 65, 70), (50, 150, 50), -1)
    cv2.putText(canvas, "DETECT",
                (CAM_W - 123, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    cv2.rectangle(canvas, (CAM_W - 60, 46), (CAM_W - 5, 70), (50, 50, 200), -1)
    cv2.putText(canvas, "PICK",
                (CAM_W - 55, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return canvas


def on_mouse(event, x, y, flags, param):
    global input_active, action_queue
    if event == cv2.EVENT_LBUTTONDOWN:
        if 8 <= x <= CAM_W - 195 and 46 <= y <= 70:
            input_active = True
        elif CAM_W - 185 <= x <= CAM_W - 130 and 46 <= y <= 70:
            input_active = False
            action_queue.append("scan")
        elif CAM_W - 125 <= x <= CAM_W - 65 and 46 <= y <= 70:
            input_active = False
            action_queue.append("detect")
        elif CAM_W - 60 <= x <= CAM_W - 5 and 46 <= y <= 70:
            input_active = False
            action_queue.append("pick")
        else:
            input_active = False


def draw_result(frame_bgr, result, depth_mm, robot_coords=None):
    vis = frame_bgr.copy()
    cv2.line(vis, (305, 240), (335, 240), (0, 200, 255), 1)
    cv2.line(vis, (320, 225), (320, 255), (0, 200, 255), 1)

    if result.get("found") and len(result.get("bbox", [])) == 4:
        x1, y1, x2, y2 = result["bbox"]
        color = {"high": (0,255,0), "medium": (0,255,255), "low": (0,0,255)}.get(
            result.get("confidence", "high"), (0,255,0)
        )
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{result['label']} {depth_mm:.0f}mm"
        if robot_coords is not None:
            label += f" R:({robot_coords[0]:.0f},{robot_coords[1]:.0f},{robot_coords[2]:.0f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        cv2.rectangle(vis, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(vis, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,0,0), 2)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.circle(vis, (cx, cy), 5, color, -1)
        cv2.line(vis, (cx, cy), (320, 240), (255, 100, 0), 1)
    else:
        cv2.putText(vis, "NOT FOUND", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    return vis


# ══════════════════════════════════════════════════════
# 메인 루프
# ══════════════════════════════════════════════════════
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

print(f"\nSCAN: 스캔 시작 | DETECT: 감지만 | PICK: 감지+픽킹 | q: 종료")
print(f"GRIPPER_OFFSET={GRIPPER_OFFSET}mm (물체 잡을 깊이 조정)\n")

try:
    while True:
        rgb_pil, color_np, depth_frame = capture_frame()

        canvas   = np.zeros((TOTAL_H, CAM_W, 3), dtype=np.uint8)
        cam_view = draw_result(color_np, last_result, last_depth, last_robot_xyz)
        canvas[INPUT_H:, :] = cam_view
        canvas = draw_ui(canvas, TARGET, input_text, input_active, scan_status)
        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            scan_running = False
            break
        elif key == 27:
            input_text   = ""
            input_active = False
        elif input_active:
            if key == 13:
                if input_text.strip():
                    TARGET = input_text.strip()
                    print(f"타겟 변경: '{TARGET}'")
                    last_result    = {"found": False, "bbox": [], "confidence": "none", "label": ""}
                    last_depth     = 0.0
                    last_robot_xyz = None
                input_text   = ""
                input_active = False
            elif key == 8:
                input_text = input_text[:-1]
            elif 32 <= key <= 126:
                input_text += chr(key)
        else:
            if key == ord('s'):
                action_queue.append("scan")
            elif key == ord(' '):
                action_queue.append("detect")
            elif key == 13:
                action_queue.append("pick")

        if action_queue:
            action = action_queue.pop(0)

            if action == "scan":
                if scan_running:
                    print("스캔 중지")
                    scan_running = False
                    scan_status  = "중지됨"
                else:
                    print(f"\n[스캔 시작] 타겟: '{TARGET}'")
                    scan_running = True
                    scan_status  = "스캔 시작"
                    threading.Thread(
                        target=scan_and_pick, args=(TARGET,), daemon=True
                    ).start()

            elif action == "detect":
                if scan_running:
                    continue
                print(f"\n[감지] '{TARGET}'...")
                result = extract_bbox(rgb_pil, TARGET)
                if result.get("found") and len(result.get("bbox", [])) == 4:
                    depth_mm = get_depth_mm(depth_frame, result["bbox"])
                    if depth_mm > 0:
                        _, ee_pose = robot.GetActualTCPPose(1)
                        p_cam      = pixel_to_camera_coords(result["bbox"], depth_mm)
                        robot_xyz  = camera_to_robot_coords(p_cam, ee_pose)
                        print(f"  bbox={result['bbox']} depth={depth_mm:.0f}mm")
                        print(f"  로봇좌표: X={robot_xyz[0]:.1f} Y={robot_xyz[1]:.1f} Z={robot_xyz[2]:.1f}")
                        last_result    = result
                        last_depth     = depth_mm
                        last_robot_xyz = robot_xyz
                    else:
                        print("  ⚠ depth=0 읽기 실패")
                else:
                    print("  ❌ 감지 실패")
                    last_result = result

            elif action == "pick":
                if scan_running:
                    continue
                print(f"\n[픽킹] '{TARGET}'...")
                result = extract_bbox(rgb_pil, TARGET)
                if result.get("found") and len(result.get("bbox", [])) == 4:
                    depth_mm = get_depth_mm(depth_frame, result["bbox"])
                    if depth_mm > 0:
                        _, ee_pose = robot.GetActualTCPPose(1)
                        p_cam      = pixel_to_camera_coords(result["bbox"], depth_mm)
                        robot_xyz  = camera_to_robot_coords(p_cam, ee_pose)
                        last_result    = result
                        last_depth     = depth_mm
                        last_robot_xyz = robot_xyz
                        try:
                            do_pick(robot_xyz)
                        except RuntimeError as e:
                            print(f"  ❌ 로봇 오류: {e}")
                    else:
                        print("  ⚠ depth=0 읽기 실패")
                else:
                    print("  ❌ 감지 실패")

finally:
    scan_running = False
    rs_pipeline.stop()
    cv2.destroyAllWindows()
    print("종료")