"""
FR5 + D455 + Qwen2.5-VL 속도 최적화 픽킹 파이프라인 + Z축 보정
【속도 최적화 3종 적용 + Z 캘리브레이션】

1) Qwen2.5-VL-3B (기본) — 7B 대비 추론 2배 빠름, VRAM ~6GB
   → MODEL_SIZE = "3B" / "7B" 로 전환 가능
2) 이미지 축소 — 640x480 → 320x240 으로 VLM 전달 (토큰 75% 절감)
   → VLM_IMG_SIZE = (320, 240)
3) 토큰 절감 — max_new_tokens 256→128, 프롬프트 간결화

예상 속도: 7B ~6초 → 3B+최적화 ~1.5~2초

실행: python picking_pipeline_fast.py
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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
from qwen_vl_utils import process_vision_info
from fairino import Robot
import torch
import re
import json
import time
import threading
from dist_z import load_z_correction


# ══════════════════════════════════════════════════════
# ★ 속도 설정 ★
# ══════════════════════════════════════════════════════
MODEL_SIZE   = "3B"            # "3B" (빠름) / "7B" (정확)
VLM_IMG_SIZE = (640, 480)      # VLM 입력 해상도 (작을수록 빠름)
                               # (320,240): 가장 빠름 / (480,360): 균형 / (640,480): 원본
MAX_TOKENS   = 128             # 생성 토큰 수 (128이면 JSON 충분)


# ══════════════════════════════════════════════════════
# 설정값
# ══════════════════════════════════════════════════════
ROBOT_IP       = "192.168.58.3"
TOOL_NO        = 1
USER_NO        = 0
SPEED          = 30            # 직선 이동 속도 (원본 20 → 30)
SPEED_J        = 50            # 관절 이동 속도 (원본 30 → 50)
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

HOME_J  = [10.0, -98.0, 100.0, -94.0, -84.0, -111.0]
WAY_J   = [10.0, -98.0, 100.0, -94.0, -84.0, -111.0]
SCAN1_J = [9.22, -123.75, 134.18, -148.74, -80.03, -100.25]
PLACE_J = [86.3, -56.45, 37.47, -94.04, -76.52, -111.0]

SCAN_Z_OFFSETS_MM = [0, 50, -50]

vlm_lock = threading.Lock()

# Z축 보정 모델 로드
correct_z = load_z_correction()


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
# 1. 모델 로드  ★ 3B / 7B 선택 ★
# ══════════════════════════════════════════════════════
MODEL_MAP = {
    "3B": "Qwen/Qwen2.5-VL-3B-Instruct",
    "7B": "Qwen/Qwen2.5-VL-7B-Instruct",
}
MODEL_ID = MODEL_MAP.get(MODEL_SIZE, MODEL_MAP["3B"])

print("=" * 55)
print(f"모델 로드 중... ({MODEL_ID})")
print(f"  VLM 입력: {VLM_IMG_SIZE[0]}x{VLM_IMG_SIZE[1]}  토큰: {MAX_TOKENS}")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="auto",
    max_memory={0: "14GiB", "cpu": "16GiB"}
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

vram = torch.cuda.memory_allocated() / 1024**3
print(f"✅ {MODEL_SIZE} 모델 로드 완료 | VRAM: {vram:.1f} GB")


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

try:
    result = robot.GetRobotErrorCode()
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        if result[1] != 0:
            print(f"  ⚠ 에러 감지 (code={result[1]}), 클리어 중...")
            robot.ResetAllError()
            time.sleep(0.5)
except Exception as e:
    print(f"  ⚠ {e}")
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
    time.sleep(0.15)          # 0.4→0.15: 모션 시작 대기 (최소한)
    start = time.time()
    while time.time() - start < timeout:
        ret  = robot.GetRobotMotionDone()
        done = ret[1] if isinstance(ret, (list, tuple)) else ret
        if done == 1:
            time.sleep(0.05)  # 0.1→0.05: 정착 대기
            return
        time.sleep(0.05)
    raise RuntimeError(f"이동 타임아웃 ({timeout}s 초과)")

def get_stable_pose(samples: int = 3, interval: float = 0.03) -> list:
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

def gripper_open():
    robot.MoveGripper(index=1, pos=100, vel=50, force=50,
                      maxtime=5000, block=1, type=0,
                      rotNum=0, rotVel=0, rotTorque=0)
    time.sleep(0.3)           # 0.5→0.3

def gripper_close():
    robot.MoveGripper(index=1, pos=30, vel=50, force=50,
                      maxtime=5000, block=1, type=0,
                      rotNum=0, rotVel=0, rotTorque=0)
    time.sleep(0.3)           # 0.5→0.3

def go_home():
    if WAY_J != HOME_J:
        move_j(WAY_J,  "경유점 복귀")
    move_j(HOME_J, "홈 복귀")


# ══════════════════════════════════════════════════════
# VLM 함수  ★ 속도 최적화 ★
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
    """
    ★ 최적화된 VLM 감지 ★
    1) 이미지를 VLM_IMG_SIZE로 축소 → 토큰 대폭 절감
    2) 간결한 프롬프트 → 파싱 빠름
    3) bbox를 원본 해상도(640x480)로 역변환
    """
    orig_w, orig_h = image.size   # 640, 480
    vlm_w, vlm_h   = VLM_IMG_SIZE

    # ── 이미지 축소 (VLM 입력용) ─────────────────────
    if (vlm_w, vlm_h) != (orig_w, orig_h):
        vlm_image = image.resize((vlm_w, vlm_h), Image.LANCZOS)
    else:
        vlm_image = image

    # ── 간결한 프롬프트 ──────────────────────────────
    prompt = (
        f"Detect '{target}' in this {vlm_w}x{vlm_h} image. "
        f"Return JSON only: "
        f'{{"found":true,"label":"name","bbox":[x1,y1,x2,y2]}} '
        f"or "
        f'{{"found":false,"label":"","bbox":[]}} '
        f"Important: bbox must tightly fit the object. Return best match only."
    )

    messages = [{"role": "user", "content": [
        {"type": "image", "image": vlm_image},
        {"type": "text",  "text": prompt}
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, return_tensors="pt"
    ).to("cuda")

    t0 = time.time()
    with vlm_lock:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)

    generated = out[:, inputs.input_ids.shape[1]:]
    raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
    elapsed = time.time() - t0
    print(f"  [VLM {MODEL_SIZE}] ({elapsed:.1f}s) {raw.strip()}")

    # ── JSON 파싱 ────────────────────────────────────
    raw_clean    = re.sub(r"```json|```", "", raw).strip()
    target_words = set(target.lower().replace("or", " ").split())

    # ── 스케일 팩터 (VLM 해상도 → 원본 해상도) ──────
    sx = orig_w / vlm_w   # 예: 640/320 = 2.0
    sy = orig_h / vlm_h   # 예: 480/240 = 2.0

    def _scale_bbox(bbox):
        """VLM 해상도 bbox → 원본 해상도로 변환"""
        return [
            int(bbox[0] * sx), int(bbox[1] * sy),
            int(bbox[2] * sx), int(bbox[3] * sy),
        ]

    # 케이스 1: 리스트
    list_match = re.search(r'\[.*?\]', raw_clean, re.DOTALL)
    if list_match:
        try:
            items = json.loads(list_match.group())
            if isinstance(items, list) and items:
                best = None
                for item in items:
                    lbl  = item.get("label", "").lower()
                    bbox = item.get("bbox_2d") or item.get("bbox", [])
                    if not bbox or len(bbox) < 4:
                        continue
                    if any(w in lbl for w in target_words):
                        best = item
                        break
                if best is None:
                    for item in items:
                        bbox = item.get("bbox_2d") or item.get("bbox", [])
                        if bbox and len(bbox) == 4:
                            best = item
                            break
                if best:
                    bbox = best.get("bbox_2d") or best.get("bbox", [])
                    if bbox and len(bbox) == 4:
                        return {
                            "found": True,
                            "label": best.get("label", target),
                            "bbox":  _scale_bbox([int(v) for v in bbox]),
                            "confidence": "high"
                        }
        except (json.JSONDecodeError, Exception):
            pass

    # 케이스 2: 딕셔너리
    dict_match = re.search(r'\{[^{}]*\}', raw_clean, re.DOTALL)
    if dict_match:
        try:
            item = json.loads(dict_match.group())
            bbox = item.get("bbox_2d") or item.get("bbox", [])
            if bbox and len(bbox) == 4 and item.get("found", True):
                return {
                    "found": True,
                    "label": item.get("label", target),
                    "bbox":  _scale_bbox([int(v) for v in bbox]),
                    "confidence": "high"
                }
        except (json.JSONDecodeError, Exception):
            pass

    # 케이스 3: 정규식
    all_bboxes = re.findall(
        r'"bbox(?:_2d)?"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', raw_clean
    )
    all_labels = re.findall(r'"label"\s*:\s*"([^"]*)"', raw_clean)

    if all_bboxes:
        pairs = list(zip(all_labels, all_bboxes)) if all_labels else \
                [(target, b) for b in all_bboxes]
        chosen_label, chosen_bbox = pairs[0]
        for lbl, bb in pairs:
            if any(w in lbl.lower() for w in target_words):
                chosen_label, chosen_bbox = lbl, bb
                break
        return {
            "found": True,
            "label": chosen_label or target,
            "bbox":  _scale_bbox([int(v) for v in chosen_bbox]),
            "confidence": "high"
        }

    return {"found": False, "bbox": [], "confidence": "none", "label": "parse_error"}


def get_depth_mm(depth_frame, bbox) -> float:
    """bbox 중심 5x5 영역 중앙값 depth (mm) — 원본 해상도(640x480) 기준"""
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
    from scipy.spatial.transform import Rotation   # 캐시됨 (첫 호출만 느림)
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

    current_pose = get_stable_pose()
    rx, ry, rz   = current_pose[3], current_pose[4], current_pose[5]
    pick_z       = tz + GRIPPER_OFFSET

    print(f"  목표: X={tx:.1f} Y={ty:.1f} pick_z={pick_z:.1f}")

    gripper_open()
    move_l([tx, ty, pick_z + APPROACH_DIST, rx, ry, rz], "[1/4] 접근")
    print("  [2/4] 그립...")
    gripper_close()
    move_l([tx, ty, pick_z + LIFT_DIST, rx, ry, rz], "[3/4] 들어올리기")

    if PLACE_J is not None:
        if WAY_J != HOME_J:
            move_j(WAY_J, "[4/4] 경유점 이동")
        move_j(PLACE_J, "      Place 위치 이동")
        print("  그리퍼 열기 (Place)...")
        gripper_open()
        time.sleep(0.3)     # 0.5→0.3
        go_home()
    else:
        go_home()

    print("  ✅ 픽앤플레이스 완료!")


POINT_OFFSET_X = 50

def do_point(target_xyz):
    tx, ty, tz = target_xyz
    current_pose = get_stable_pose()
    rx, ry, rz   = current_pose[3], current_pose[4], current_pose[5]
    point_z      = tz + APPROACH_DIST
    point_x = tx + POINT_OFFSET_X

    print(f"  지목 위치: X={point_x:.1f} Y={ty:.1f} Z={point_z:.1f}")
    move_l([point_x, ty, point_z, rx, ry, rz], "[1/3] 지목 위치 이동")
    print("  [2/3] 지목 중... (1초)")
    time.sleep(1.0)           # 1.5→1.0
    go_home()
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

    base_pose = get_stable_pose()       # 5→3 샘플 (기본값 변경됨)
    base_x, base_y, base_z = base_pose[0], base_pose[1], base_pose[2]
    rx, ry, rz              = base_pose[3], base_pose[4], base_pose[5]
    print(f"  스캔 기준: X={base_x:.1f} Y={base_y:.1f} Z={base_z:.1f}")

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

        time.sleep(0.15)          # 0.3→0.15: 카메라 안정화 최소

        task_status = f"감지 중 ({MODEL_SIZE})"
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

        task_status = "재감지 중"
        print(f"  [재감지]...")
        time.sleep(0.1)           # 0.2→0.1
        rgb_pil2, _, depth_frame2 = capture_frame()
        r2 = extract_bbox(rgb_pil2, target)

        if not r2.get("found") or not r2.get("bbox"):
            print(f"  → 재감지 미감지, 다음 높이로")
            continue

        d2 = get_depth_mm(depth_frame2, r2["bbox"])
        print(f"  → 재감지 ✓  bbox={r2['bbox']}  depth={d2:.0f}mm")

        if d2 == 0:
            continue

        avg_bbox  = [int((r1["bbox"][i] + r2["bbox"][i]) / 2) for i in range(4)]
        avg_depth = (d1 + d2) / 2

        _, ee_pose = robot.GetActualTCPPose(1)
        p_cam      = pixel_to_camera_coords(avg_bbox, avg_depth)
        robot_xyz  = camera_to_robot_coords(p_cam, ee_pose)

        raw_z = robot_xyz[2]
        robot_xyz[2] = correct_z(robot_xyz[2])

        last_result    = r2
        last_depth     = avg_depth
        last_robot_xyz = robot_xyz

        print(f"\n  ✅ 좌표 확정!")
        print(f"     bbox 평균  : {avg_bbox}")
        print(f"     depth 평균 : {avg_depth:.0f}mm")
        print(f"     로봇 좌표  : X={robot_xyz[0]:.1f}  Y={robot_xyz[1]:.1f}  Z={robot_xyz[2]:.1f}")
        print(f"     Z 보정     : {raw_z:.1f} → {robot_xyz[2]:.1f} (차이={robot_xyz[2]-raw_z:+.1f})")
        return robot_xyz, avg_bbox, avg_depth

    print(f"\n  ❌ '{target}' 를 찾지 못했습니다.")
    return None, None, None


def _go_scan1_and_detect(target: str) -> tuple:
    global task_status
    task_status = "홈 이동 중"
    move_j(HOME_J, "홈 이동")
    if WAY_J != HOME_J:
        move_j(WAY_J, "경유점 이동")
    task_status = "스캔 포인트 이동"
    print("\n[스캔 포인트] 이동...")
    move_j(SCAN1_J, "스캔")
    return scan_and_detect(target)


def run_detect_only(command: str):
    global task_running, task_status, current_target
    global last_result, last_depth, last_robot_xyz

    target = parse_target(command)
    current_target = target
    last_result = {"found": False, "bbox": [], "confidence": "none", "label": ""}
    last_depth = 0.0
    last_robot_xyz = None

    print(f"\n{'='*55}")
    print(f"[DETECT] '{command}'  →  타겟: '{target}'")
    print(f"{'='*55}")

    try:
        robot_xyz, _, _ = _go_scan1_and_detect(target)
        task_status = "감지 완료 ✅" if robot_xyz is not None else "미감지 — 실패"
        go_home()
    except RuntimeError as e:
        print(f"\n[에러] {e}")
        task_status = f"에러: {e}"
        try: go_home()
        except: pass
    finally:
        task_running = False


def run_full_sequence(command: str):
    global task_running, task_status, current_target
    global last_result, last_depth, last_robot_xyz

    target = parse_target(command)
    current_target = target
    last_result = {"found": False, "bbox": [], "confidence": "none", "label": ""}
    last_depth = 0.0
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
        last_result = {"found": False, "bbox": [], "confidence": "none", "label": ""}
        last_depth = 0.0
        last_robot_xyz = None

        do_pick(robot_xyz)
        task_status = f"픽킹 완료 ✅  ({target})"
    except RuntimeError as e:
        print(f"\n[에러] {e}")
        task_status = f"에러: {e}"
        try: go_home()
        except: pass
    finally:
        task_running = False


def run_point_sequence(command: str):
    global task_running, task_status, current_target
    global last_result, last_depth, last_robot_xyz

    target = parse_target(command)
    current_target = target
    last_result = {"found": False, "bbox": [], "confidence": "none", "label": ""}
    last_depth = 0.0
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
        last_result = {"found": False, "bbox": [], "confidence": "none", "label": ""}
        last_depth = 0.0
        last_robot_xyz = None

        do_point(robot_xyz)
        task_status = f"지목 완료 ✅  ({target})"
    except RuntimeError as e:
        print(f"\n[에러] {e}")
        task_status = f"에러: {e}"
        try: go_home()
        except: pass
    finally:
        task_running = False


# ══════════════════════════════════════════════════════
# GUI
# ══════════════════════════════════════════════════════
from PIL import ImageFont, ImageDraw as PilDraw

WINDOW_NAME = f"Picking Pipeline (Qwen2.5-VL-{MODEL_SIZE} FAST)"
CAM_H, CAM_W = 480, 640
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

def _load_font(size):
    for path in _FONT_PATHS:
        if os.path.exists(path):
            try: return ImageFont.truetype(path, size)
            except: continue
    return ImageFont.load_default()

FONT_SM = _load_font(14)
FONT_MD = _load_font(16)
FONT_LG = _load_font(18)

def put_kr(img, text, xy, font, color=(255,255,255)):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = PilDraw.Draw(pil)
    draw.text(xy, text, font=font, fill=color)
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
        canvas = put_kr(canvas, "명령 입력  예) 컵 잡아줘 / bottle / red cup",
                        (14, 9), FONT_SM, (80, 80, 80))

    tgt_str = f"타겟: {current_target}" if current_target else "타겟: (없음)"
    tgt_col = (120, 220, 120) if current_target else (120, 120, 120)
    canvas = put_kr(canvas, tgt_str, (8, 38), FONT_SM, tgt_col)

    if   "이동" in task_status or "감지" in task_status: sc = (0, 200, 255)
    elif "완료" in task_status: sc = (80, 255, 120)
    elif "픽킹" in task_status: sc = (80, 140, 255)
    elif "에러" in task_status or "실패" in task_status: sc = (80, 80, 255)
    else: sc = (130, 130, 130)
    canvas = put_kr(canvas, f"상태: {task_status}", (8, 58), FONT_SM, sc)

    if task_running:
        cv2.rectangle(canvas, (CAM_W-78, 38), (CAM_W-10, 70), (180,40,40), -1)
        canvas = put_kr(canvas, "STOP", (CAM_W-68, 45), FONT_MD, (255,255,255))
    else:
        cv2.rectangle(canvas, (CAM_W-168, 38), (CAM_W-90, 70), (40,130,40), -1)
        canvas = put_kr(canvas, "DETECT", (CAM_W-164, 45), FONT_MD, (255,255,255))
        cv2.rectangle(canvas, (CAM_W-85, 38), (CAM_W-10, 70), (180,120,20), -1)
        canvas = put_kr(canvas, "지목", (CAM_W-72, 45), FONT_MD, (255,255,255))

    canvas = put_kr(canvas, "Enter: 실행  d: 감지  p: 지목  ESC: 취소  q: 종료",
                    (8, 78), FONT_SM, (70, 70, 70))
    return canvas

def on_mouse(event, x, y, flags, param):
    global input_active, task_running, action_queue
    if event == cv2.EVENT_LBUTTONDOWN:
        if 8 <= x <= CAM_W-12 and 6 <= y <= 32:
            input_active = True
        elif task_running and CAM_W-78 <= x <= CAM_W-10 and 38 <= y <= 70:
            task_running = False
        elif not task_running and CAM_W-168 <= x <= CAM_W-90 and 38 <= y <= 70:
            action_queue.append("detect"); input_active = False
        elif not task_running and CAM_W-85 <= x <= CAM_W-10 and 38 <= y <= 70:
            action_queue.append("point"); input_active = False
        else:
            input_active = False

def draw_result(frame_bgr):
    vis = frame_bgr.copy()
    cv2.line(vis, (305,240), (335,240), (0,160,255), 1)
    cv2.line(vis, (320,225), (320,255), (0,160,255), 1)

    if last_result.get("found") and len(last_result.get("bbox",[])) == 4:
        x1,y1,x2,y2 = last_result["bbox"]
        c = (0,255,0)
        cv2.rectangle(vis, (x1,y1), (x2,y2), c, 2)
        label = f"{last_result['label']}  {last_depth:.0f}mm"
        if last_robot_xyz is not None:
            label += f"  ({last_robot_xyz[0]:.0f},{last_robot_xyz[1]:.0f},{last_robot_xyz[2]:.0f})"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(vis, (x1,y1-th-8), (x1+tw+6,y1), c, -1)
        cv2.putText(vis, label, (x1+3,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,0,0), 1)
        cx,cy = (x1+x2)//2, (y1+y2)//2
        cv2.circle(vis, (cx,cy), 5, c, -1)
    elif task_running:
        vis = put_kr(vis, "탐색 중...", (16, 446), FONT_LG, (0, 200, 255))
    return vis


# ══════════════════════════════════════════════════════
# 메인 루프
# ══════════════════════════════════════════════════════
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)
action_queue: list = []

print(f"\n사용법: Enter→픽킹  d→감지  p→지목  q→종료")
print(f"모델: {MODEL_SIZE} | 입력: {VLM_IMG_SIZE[0]}x{VLM_IMG_SIZE[1]} | 토큰: {MAX_TOKENS}\n")

try:
    while True:
        rgb_pil, color_np, depth_frame = capture_frame()
        canvas = np.zeros((TOTAL_H, CAM_W, 3), dtype=np.uint8)
        canvas[INFO_H:, :] = draw_result(color_np)
        canvas = draw_ui(canvas)
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            task_running = False; break
        elif key == 27:
            input_text = ""; input_active = False
        elif key == ord('d') and not task_running and not input_active:
            action_queue.append("detect")
        elif key == ord('p') and not task_running and not input_active:
            action_queue.append("point")
        elif input_active:
            if key == 13:
                cmd = input_text.strip(); input_text = ""; input_active = False
                if cmd and not task_running:
                    task_running = True; task_status = "시작 중"
                    current_target = parse_target(cmd)
                    threading.Thread(target=run_full_sequence, args=(cmd,), daemon=True).start()
            elif key == 8: input_text = input_text[:-1]
            elif 32 <= key <= 126: input_text += chr(key)
        else:
            if key == 13 and not task_running: input_active = True

        if action_queue:
            action = action_queue.pop(0)
            cmd = input_text.strip() or current_target
            if not cmd:
                print("⚠ 타겟을 먼저 입력하세요.")
            elif not task_running:
                task_running = True
                current_target = parse_target(cmd)
                if action == "detect":
                    task_status = "감지 시작"
                    threading.Thread(target=run_detect_only, args=(cmd,), daemon=True).start()
                elif action == "point":
                    task_status = "지목 시작"
                    threading.Thread(target=run_point_sequence, args=(cmd,), daemon=True).start()

finally:
    task_running = False
    rs_pipeline.stop()
    cv2.destroyAllWindows()
    print("종료")