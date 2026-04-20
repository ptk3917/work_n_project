"""
FR5 + D455 + GPT-4o API 통합 픽킹 파이프라인
【OpenAI API 버전】 — 로컬 VLM → GPT-4o API 교체

장점:
- GPU VRAM 0 사용 (카메라/로봇만)
- 최강 비전 모델 → "큰 박스", "검정 뚜껑 병" 등 문맥 완벽 이해
- 로컬 모델 로드 시간 0초

단점:
- API 호출당 1~3초 지연 (로컬 Qwen2.5-VL과 비슷하거나 더 빠름)
- 비용: 이미지당 ~$0.01
- 인터넷 연결 필요

【변경된 흐름】
  "컵 잡아줘" 입력 + 엔터
      ↓
  타겟 파싱 → 스캔 포인트 1 이동 → 1차 감지 (GPT-4o API)
      ↓ 감지됨
  재확인 1회 (bbox 2번 이하 확정)
      ↓ 확정
  픽킹 실행 → 홈 복귀

실행 전:
  1) pip install openai
  2) export OPENAI_API_KEY="sk-..."
     또는 아래 OPENAI_API_KEY 변수에 직접 입력

실행: python picking_pipeline_gpt4o.py
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~/vlm_agent/qwen_ver1"))

import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image
from fairino import Robot
import torch
import re
import json
import time
import threading
import base64
import io

# ── OpenAI API ────────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    print("⚠ openai 미설치!  pip install openai")
    sys.exit(1)

# API 키: 환경변수 또는 직접 입력
OPENAI_API_KEY = "API"
if not OPENAI_API_KEY:
    print("⚠ OPENAI_API_KEY가 설정되지 않았습니다!")
    print("  방법 1: export OPENAI_API_KEY='sk-...'")
    print("  방법 2: 스크립트 상단 OPENAI_API_KEY 변수에 직접 입력")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)
GPT_MODEL = "gpt-4o"          # gpt-4o / gpt-4o-mini (저렴)
GPT_DETAIL = "low"             # low: 빠르고 저렴 / high: 정밀 (느림)


# ══════════════════════════════════════════════════════
# 설정값
# ══════════════════════════════════════════════════════
ROBOT_IP       = "192.168.58.3"
TOOL_NO        = 1
USER_NO        = 0
SPEED          = 20
SPEED_J        = 30
GRIPPER_OFFSET = 30
APPROACH_DIST  = 80
LIFT_DIST      = 120

PICK_OFFSET_X =   0
PICK_OFFSET_Y =   0

FX = 386.4894
FY = 386.6551
CX = 325.7376
CY = 246.5258

T_CAM_TO_EE = np.load(os.path.expanduser("~/vlm_agent/qwen_ver1/T_cam_to_ee.npy"))

HOME_J  = [10.0,  -98.0,  100.0,  -94.0,  -84.0, -111.0]
WAY_J   = [10.69, -128.26, 129.33, -94.03, -84.34, -111.73]
SCAN1_J = [9.22, -123.75, 134.18, -148.74, -80.03, -100.25]
PLACE_J = [86.3, -56.45, 37.47, -94.04, -76.52, -111.0]

SCAN_Z_OFFSETS_MM = [0, 50, -50]

vlm_lock = threading.Lock()


# ══════════════════════════════════════════════════════
# 자연어 타겟 파싱
# ══════════════════════════════════════════════════════
KO_TO_EN = {
    "컵": "cup or mug", "머그컵": "mug", "병": "bottle",
    "박스": "box", "상자": "box", "사과": "apple",
    "책": "book", "가위": "scissors", "드라이버": "screwdriver",
    "공": "ball",
}

def parse_target(command: str) -> str:
    cmd = command.strip()
    for suffix in ["잡아줘", "집어줘", "가져와", "집어", "잡아", "줘", "해줘"]:
        cmd = cmd.replace(suffix, "").strip()
    for ko, en in KO_TO_EN.items():
        if ko in cmd:
            cmd = cmd.replace(ko, en)
            return cmd.strip()
    return cmd if cmd else command.strip()


# ══════════════════════════════════════════════════════
# 1. 모델 로드 → 없음! (API 사용)
# ══════════════════════════════════════════════════════
print("=" * 55)
print(f"VLM: OpenAI {GPT_MODEL} API (로컬 모델 없음)")
print(f"  → GPU VRAM 0 사용, API 키 확인 완료")


# ══════════════════════════════════════════════════════
# 2. D455 초기화
# ══════════════════════════════════════════════════════
rs_pipeline = rs.pipeline()
rs_config   = rs.config()
rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
rs_pipeline.start(rs_config)
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

# 기존 에러 자동 클리어
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


# ══════════════════════════════════════════════════════
# 로봇 이동 함수
# ══════════════════════════════════════════════════════
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
    arr = np.array(poses)
    return arr.mean(axis=0).tolist()

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
# VLM 함수  ★ GPT-4o API ★
# ══════════════════════════════════════════════════════
def capture_frame():
    frames      = rs_pipeline.wait_for_frames()
    aligned     = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    color_np    = np.asanyarray(color_frame.get_data())
    rgb_pil     = Image.fromarray(cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB))
    return rgb_pil, color_np, depth_frame


def _pil_to_base64(image: Image.Image, quality: int = 85) -> str:
    """PIL 이미지 → base64 JPEG 문자열"""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_bbox(image: Image.Image, target: str) -> dict:
    """
    GPT-4o API로 물체 감지.
    - 이미지를 base64로 인코딩하여 전송
    - JSON 응답 파싱
    - 문맥 이해 완벽: "큰 박스", "검정 뚜껑 병", "왼쪽 컵" 등
    """
    w, h = image.size

    prompt = f"""You are a robot vision system. Detect '{target}' in the image.
Image size: {w}x{h} pixels.

Return ONLY this JSON (no markdown, no explanation):
{{"found": true, "label": "object name", "bbox": [x_min, y_min, x_max, y_max], "confidence": "high/medium/low"}}

If not found:
{{"found": false, "label": "", "bbox": [], "confidence": "none"}}

Rules:
- bbox coordinates must be in pixels (0 to {w} for x, 0 to {h} for y)
- If multiple objects match, return ONLY the one that best matches '{target}'
- Understand context: "biggest box" = largest box, "black cap bottle" = bottle with black cap
- Detect even if partially visible or at an angle"""

    img_b64 = _pil_to_base64(image)

    t0 = time.time()
    try:
        with vlm_lock:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}",
                                "detail": GPT_DETAIL,
                            }
                        },
                        {"type": "text", "text": prompt},
                    ]
                }],
                max_tokens=256,
                temperature=0,
            )

        elapsed = time.time() - t0
        raw = response.choices[0].message.content.strip()
        print(f"  [GPT-4o] ({elapsed:.1f}s) {raw}")

    except Exception as e:
        print(f"  [GPT-4o 에러] {e}")
        return {"found": False, "bbox": [], "confidence": "none", "label": "api_error"}

    # ── JSON 파싱 ──────────────────────────────────────
    raw_clean = re.sub(r"```json|```", "", raw).strip()

    # 딕셔너리 형식
    dict_match = re.search(r'\{[^{}]*\}', raw_clean, re.DOTALL)
    if dict_match:
        try:
            item = json.loads(dict_match.group())
            bbox = item.get("bbox", [])
            if bbox and len(bbox) == 4 and item.get("found", True):
                bbox_int = [int(v) for v in bbox]
                # 클램핑
                bbox_int[0] = max(0, min(bbox_int[0], w - 1))
                bbox_int[1] = max(0, min(bbox_int[1], h - 1))
                bbox_int[2] = max(0, min(bbox_int[2], w - 1))
                bbox_int[3] = max(0, min(bbox_int[3], h - 1))
                return {
                    "found": True,
                    "label": item.get("label", target),
                    "bbox":  bbox_int,
                    "confidence": item.get("confidence", "high")
                }
        except (json.JSONDecodeError, Exception):
            pass

    # 정규식 fallback
    all_bboxes = re.findall(
        r'"bbox"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', raw_clean
    )
    if all_bboxes:
        bb = all_bboxes[0]
        return {
            "found": True,
            "label": target,
            "bbox":  [int(v) for v in bb],
            "confidence": "high"
        }

    return {"found": False, "bbox": [], "confidence": "none", "label": "parse_error"}


def get_depth_mm(depth_frame, bbox) -> float:
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


def bbox_iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter)


# ══════════════════════════════════════════════════════
# 픽킹 시퀀스
# ══════════════════════════════════════════════════════
def do_pick(target_xyz):
    tx, ty, tz = target_xyz
    tx += PICK_OFFSET_X
    ty += PICK_OFFSET_Y

    current_pose = get_stable_pose(samples=3, interval=0.05)
    rx, ry, rz   = current_pose[3], current_pose[4], current_pose[5]
    pick_z       = tz + GRIPPER_OFFSET

    print(f"  목표: X={tx:.1f} Y={ty:.1f} pick_z={pick_z:.1f}")
    if PICK_OFFSET_X != 0 or PICK_OFFSET_Y != 0:
        print(f"  (오프셋 적용: X{PICK_OFFSET_X:+.0f}mm Y{PICK_OFFSET_Y:+.0f}mm)")

    gripper_open()
    move_l([tx, ty, pick_z + APPROACH_DIST, rx, ry, rz], "[1/4] 접근")
    print("  [2/4] 그립...")
    gripper_close()
    move_l([tx, ty, pick_z + LIFT_DIST, rx, ry, rz], "[3/4] 들어올리기")

    if PLACE_J is not None:
        move_j(WAY_J,   "[4/4] 경유점 이동")
        move_j(PLACE_J, "      Place 위치 이동")
        print("  그리퍼 열기 (Place)...")
        gripper_open()
        time.sleep(0.5)
        move_j(WAY_J,  "      경유점 복귀")
        move_j(HOME_J, "      홈 복귀")
    else:
        print("  ⚠ PLACE_J 미설정 — 경유점 → 홈 복귀")
        move_j(WAY_J,  "[4/4] 경유점 복귀")
        move_j(HOME_J, "       홈 복귀")

    print("  ✅ 픽앤플레이스 완료!")


POINT_OFFSET_X = 50

def do_point(target_xyz):
    tx, ty, tz = target_xyz
    current_pose = get_stable_pose(samples=3, interval=0.05)
    rx, ry, rz   = current_pose[3], current_pose[4], current_pose[5]
    point_z      = tz + APPROACH_DIST
    point_x = tx + POINT_OFFSET_X

    print(f"  지목 위치: X={point_x:.1f} Y={ty:.1f} Z={point_z:.1f}")
    move_l([point_x, ty, point_z, rx, ry, rz], "[1/3] 지목 위치 이동")
    print("  [2/3] 지목 중... (1.5초)")
    time.sleep(1.5)
    move_j(WAY_J,  "[3/3] 경유점 복귀")
    move_j(HOME_J, "       홈 복귀")
    print("  ✅ 지목 완료!")


# ══════════════════════════════════════════════════════
# 핵심 시퀀스
# ══════════════════════════════════════════════════════
task_running   = False
task_status    = "대기 중"
current_target = ""

last_result    = {"found": False, "bbox": [], "confidence": "none", "label": ""}
last_depth     = 0.0
last_robot_xyz = None


def scan_and_detect(target: str) -> tuple:
    global task_status, last_result, last_depth, last_robot_xyz

    base_pose = get_stable_pose(samples=5, interval=0.05)
    base_x, base_y, base_z = base_pose[0], base_pose[1], base_pose[2]
    rx, ry, rz              = base_pose[3], base_pose[4], base_pose[5]
    print(f"  스캔 1 기준: X={base_x:.1f} Y={base_y:.1f} Z={base_z:.1f}")

    for z_offset in SCAN_Z_OFFSETS_MM:
        if not task_running:
            return None, None, None

        scan_z = base_z + z_offset
        if z_offset != 0:
            dir_str = f"Z{'+' if z_offset > 0 else ''}{z_offset}mm"
            task_status = f"스캔 {dir_str}"
            print(f"\n  [{dir_str} 이동]")
            move_l([base_x, base_y, scan_z, rx, ry, rz], dir_str)
        else:
            task_status = "스캔 원위치 감지"
            print(f"\n  [원위치]")

        time.sleep(0.3)

        task_status = f"GPT-4o 감지 중..."
        print(f"  [1차 감지] '{target}' 탐색...")
        rgb_pil, _, depth_frame = capture_frame()
        r1 = extract_bbox(rgb_pil, target)

        if not r1.get("found") or not r1.get("bbox"):
            print(f"  → 미감지")
            last_result = r1
            continue

        d1 = get_depth_mm(depth_frame, r1["bbox"])
        print(f"  → 감지 ✓  bbox={r1['bbox']}  depth={d1:.0f}mm")
        last_result = r1
        last_depth  = d1

        if d1 == 0:
            print(f"  → depth=0, 다음 높이로")
            continue

        task_status = "재감지 중 (GPT-4o)"
        print(f"  [재감지]...")
        time.sleep(0.2)
        rgb_pil2, _, depth_frame2 = capture_frame()
        r2 = extract_bbox(rgb_pil2, target)

        if not r2.get("found") or not r2.get("bbox"):
            print(f"  → 재감지 미감지, 다음 높이로")
            continue

        d2 = get_depth_mm(depth_frame2, r2["bbox"])
        print(f"  → 재감지 ✓  bbox={r2['bbox']}  depth={d2:.0f}mm")

        if d2 == 0:
            print(f"  → 재감지 depth=0, 다음 높이로")
            continue

        avg_bbox  = [int((r1["bbox"][i] + r2["bbox"][i]) / 2) for i in range(4)]
        avg_depth = (d1 + d2) / 2

        _, ee_pose = robot.GetActualTCPPose(1)
        p_cam      = pixel_to_camera_coords(avg_bbox, avg_depth)
        robot_xyz  = camera_to_robot_coords(p_cam, ee_pose)

        last_result    = r2
        last_depth     = avg_depth
        last_robot_xyz = robot_xyz

        print(f"\n  ✅ 좌표 확정!")
        print(f"     bbox 평균  : {avg_bbox}")
        print(f"     depth 평균 : {avg_depth:.0f}mm")
        print(f"     로봇 좌표  : X={robot_xyz[0]:.1f}  Y={robot_xyz[1]:.1f}  Z={robot_xyz[2]:.1f}")
        return robot_xyz, avg_bbox, avg_depth

    print(f"\n  ❌ '{target}' 를 찾지 못했습니다.")
    return None, None, None


def _go_scan1_and_detect(target: str) -> tuple:
    global task_status
    task_status = "홈 이동 중"
    move_j(HOME_J, "홈 이동")
    move_j(WAY_J,  "경유점 이동")
    task_status = "스캔 포인트 1 이동"
    print("\n[스캔 포인트 1] 이동...")
    move_j(SCAN1_J, "스캔 1")
    return scan_and_detect(target)


def run_detect_only(command: str):
    global task_running, task_status, current_target
    global last_result, last_depth, last_robot_xyz

    target = parse_target(command)
    current_target = target

    last_result    = {"found": False, "bbox": [], "confidence": "none", "label": ""}
    last_depth     = 0.0
    last_robot_xyz = None

    print(f"\n{'='*55}")
    print(f"[DETECT] '{command}'  →  타겟: '{target}'")
    print(f"{'='*55}")

    try:
        robot_xyz, _, _ = _go_scan1_and_detect(target)
        task_status = "감지 완료 ✅" if robot_xyz is not None else "미감지 — 실패"
        if robot_xyz is not None:
            print("  (DETECT 모드: 픽킹 생략, 홈 복귀)")
        go_home()

    except RuntimeError as e:
        print(f"\n[에러] {e}")
        task_status = f"에러: {e}"
        try:
            go_home()
        except:
            pass
    finally:
        task_running = False


def run_full_sequence(command: str):
    global task_running, task_status, current_target
    global last_result, last_depth, last_robot_xyz

    target = parse_target(command)
    current_target = target

    last_result    = {"found": False, "bbox": [], "confidence": "none", "label": ""}
    last_depth     = 0.0
    last_robot_xyz = None

    print(f"\n{'='*55}")
    print(f"명령: '{command}'  →  타겟: '{target}'")
    print(f"{'='*55}")

    try:
        robot_xyz, _, _ = _go_scan1_and_detect(target)

        if robot_xyz is None:
            task_status = "미감지 — 실패"
            go_home()
            return

        task_status = "픽킹 중"
        print(f"\n[픽킹 시작]")

        last_result    = {"found": False, "bbox": [], "confidence": "none", "label": ""}
        last_depth     = 0.0
        last_robot_xyz = None

        do_pick(robot_xyz)
        task_status = f"픽킹 완료 ✅  ({target})"
        print(f"\n{'='*55}")
        print(f"  '{command}' 완료!")
        print(f"{'='*55}\n")

    except RuntimeError as e:
        print(f"\n[에러] {e}")
        task_status = f"에러: {e}"
        try:
            go_home()
        except:
            pass
    finally:
        task_running = False


def run_point_sequence(command: str):
    global task_running, task_status, current_target
    global last_result, last_depth, last_robot_xyz

    target = parse_target(command)
    current_target = target

    last_result    = {"found": False, "bbox": [], "confidence": "none", "label": ""}
    last_depth     = 0.0
    last_robot_xyz = None

    print(f"\n{'='*55}")
    print(f"[지목] '{command}'  →  타겟: '{target}'")
    print(f"{'='*55}")

    try:
        robot_xyz, _, _ = _go_scan1_and_detect(target)

        if robot_xyz is None:
            task_status = "미감지 — 실패"
            go_home()
            return

        task_status = "지목 중"
        print(f"\n[지목 시작]")

        last_result    = {"found": False, "bbox": [], "confidence": "none", "label": ""}
        last_depth     = 0.0
        last_robot_xyz = None

        do_point(robot_xyz)
        task_status = f"지목 완료 ✅  ({target})"
        print(f"\n{'='*55}")
        print(f"  '{command}' 지목 완료!")
        print(f"{'='*55}\n")

    except RuntimeError as e:
        print(f"\n[에러] {e}")
        task_status = f"에러: {e}"
        try:
            go_home()
        except:
            pass
    finally:
        task_running = False


# ══════════════════════════════════════════════════════
# GUI
# ══════════════════════════════════════════════════════
from PIL import ImageFont, ImageDraw as PilDraw

WINDOW_NAME = "Picking Pipeline (GPT-4o API)"
CAM_H   = 480
CAM_W   = 640
INFO_H  = 95
TOTAL_H = CAM_H + INFO_H

input_text   = ""
input_active = True

_FONT_PATHS = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
]

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_PATHS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()

FONT_SM  = _load_font(14)
FONT_MD  = _load_font(16)
FONT_LG  = _load_font(18)

def put_kr(img_bgr, text, xy, font, color_rgb=(255,255,255)):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = PilDraw.Draw(pil)
    draw.text(xy, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def draw_ui(canvas):
    cv2.rectangle(canvas, (0, 0), (CAM_W, INFO_H), (25, 25, 25), -1)
    cv2.line(canvas, (0, INFO_H), (CAM_W, INFO_H), (70, 70, 70), 1)

    box_color = (0, 200, 255) if input_active else (70, 70, 70)
    cv2.rectangle(canvas, (8, 6), (CAM_W - 12, 32), box_color, 1)

    if input_active or input_text:
        display = input_text + ("|" if input_active else "")
        canvas = put_kr(canvas, display, (14, 9), FONT_MD, (255, 255, 255))
    else:
        canvas = put_kr(canvas, "명령 입력  예) 컵 잡아줘 / biggest box / black cap bottle",
                        (14, 9), FONT_SM, (80, 80, 80))

    if current_target:
        tgt_str = f"타겟: {current_target}"
        tgt_col = (120, 220, 120)
    else:
        tgt_str = "타겟: (없음)"
        tgt_col = (120, 120, 120)
    canvas = put_kr(canvas, tgt_str, (8, 38), FONT_SM, tgt_col)

    if   "이동" in task_status or "감지" in task_status or "GPT" in task_status:
        sc = (0, 200, 255)
    elif "완료" in task_status:
        sc = (80, 255, 120)
    elif "픽킹" in task_status:
        sc = (80, 140, 255)
    elif "에러" in task_status or "실패" in task_status:
        sc = (80, 80, 255)
    else:
        sc = (130, 130, 130)
    canvas = put_kr(canvas, f"상태: {task_status}", (8, 58), FONT_SM, sc)

    if task_running:
        cv2.rectangle(canvas, (CAM_W - 78, 38), (CAM_W - 10, 70), (180, 40, 40), -1)
        canvas = put_kr(canvas, "STOP", (CAM_W - 68, 45), FONT_MD, (255, 255, 255))
    else:
        cv2.rectangle(canvas, (CAM_W - 168, 38), (CAM_W - 90, 70), (40, 130, 40), -1)
        canvas = put_kr(canvas, "DETECT", (CAM_W - 164, 45), FONT_MD, (255, 255, 255))

        cv2.rectangle(canvas, (CAM_W - 85, 38), (CAM_W - 10, 70), (180, 120, 20), -1)
        canvas = put_kr(canvas, "지목", (CAM_W - 72, 45), FONT_MD, (255, 255, 255))

    canvas = put_kr(canvas, "Enter: 실행    d: 감지    p: 지목    ESC: 취소    q: 종료",
                    (8, 78), FONT_SM, (70, 70, 70))
    return canvas

def on_mouse(event, x, y, flags, param):
    global input_active, task_running, action_queue
    if event == cv2.EVENT_LBUTTONDOWN:
        if 8 <= x <= CAM_W - 12 and 6 <= y <= 32:
            input_active = True
        elif task_running and CAM_W - 78 <= x <= CAM_W - 10 and 38 <= y <= 70:
            print("\n[STOP] 작업 중지 요청")
            task_running = False
        elif not task_running and CAM_W - 168 <= x <= CAM_W - 90 and 38 <= y <= 70:
            action_queue.append("detect")
            input_active = False
        elif not task_running and CAM_W - 85 <= x <= CAM_W - 10 and 38 <= y <= 70:
            action_queue.append("point")
            input_active = False
        else:
            input_active = False

def draw_result(frame_bgr):
    vis = frame_bgr.copy()
    cv2.line(vis, (305, 240), (335, 240), (0, 160, 255), 1)
    cv2.line(vis, (320, 225), (320, 255), (0, 160, 255), 1)

    if last_result.get("found") and len(last_result.get("bbox", [])) == 4:
        x1, y1, x2, y2 = last_result["bbox"]
        color_bgr = {"high": (0,255,0), "medium": (0,255,200), "low": (0,100,255)}.get(
            last_result.get("confidence", "high"), (0, 255, 0)
        )
        cv2.rectangle(vis, (x1, y1), (x2, y2), color_bgr, 2)
        label = f"{last_result['label']}  {last_depth:.0f}mm"
        if last_robot_xyz is not None:
            rx, ry, rz = last_robot_xyz
            label += f"  ({rx:.0f},{ry:.0f},{rz:.0f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 6, y1), color_bgr, -1)
        cv2.putText(vis, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(vis, (cx, cy), 5, color_bgr, -1)
        cv2.line(vis, (cx, cy), (320, 240), (255, 120, 0), 1)
    elif task_running:
        vis = put_kr(vis, "GPT-4o 응답 대기 중...", (16, 446), FONT_LG, (0, 200, 255))
    return vis


# ══════════════════════════════════════════════════════
# 메인 루프
# ══════════════════════════════════════════════════════
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

action_queue: list = []

print("\n사용법:")
print("  명령 입력 후 Enter → 스캔 → 감지 → 픽킹 전체 실행")
print("  DETECT 버튼 또는 d키 → 현재 위치에서 감지만")
print("  q: 종료\n")

try:
    while True:
        rgb_pil, color_np, depth_frame = capture_frame()

        canvas   = np.zeros((TOTAL_H, CAM_W, 3), dtype=np.uint8)
        cam_view = draw_result(color_np)
        canvas[INFO_H:, :] = cam_view
        canvas = draw_ui(canvas)

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            task_running = False
            break
        elif key == 27:
            input_text   = ""
            input_active = False
        elif key == ord('d') and not task_running and not input_active:
            action_queue.append("detect")
        elif key == ord('p') and not task_running and not input_active:
            action_queue.append("point")
        elif input_active:
            if key == 13:
                cmd = input_text.strip()
                input_text   = ""
                input_active = False
                if not cmd:
                    continue
                if task_running:
                    print("⚠ 현재 작업 진행 중입니다.")
                    continue
                task_running   = True
                task_status    = "시작 중"
                current_target = parse_target(cmd)
                threading.Thread(target=run_full_sequence, args=(cmd,), daemon=True).start()
            elif key == 8:
                input_text = input_text[:-1]
            elif 32 <= key <= 126:
                input_text += chr(key)
        else:
            if key == 13 and not task_running:
                input_active = True

        if action_queue:
            action = action_queue.pop(0)
            if action == "detect" and not task_running:
                cmd = input_text.strip() or (current_target if current_target else "")
                if not cmd:
                    print("⚠ 타겟을 먼저 입력하세요.")
                else:
                    task_running   = True
                    task_status    = "감지 시작 (GPT-4o)"
                    current_target = parse_target(cmd)
                    threading.Thread(target=run_detect_only, args=(cmd,), daemon=True).start()
            elif action == "point" and not task_running:
                cmd = input_text.strip() or (current_target if current_target else "")
                if not cmd:
                    print("⚠ 타겟을 먼저 입력하세요.")
                else:
                    task_running   = True
                    task_status    = "지목 시작 (GPT-4o)"
                    current_target = parse_target(cmd)
                    threading.Thread(target=run_point_sequence, args=(cmd,), daemon=True).start()

finally:
    task_running = False
    rs_pipeline.stop()
    cv2.destroyAllWindows()
    print("종료")
