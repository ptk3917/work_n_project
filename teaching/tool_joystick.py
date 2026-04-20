#!/usr/bin/env python3
"""
=== FR5 Joystick Controller ===
로봇 팔을 키보드로 실시간 조작하는 컨트롤러

조작법:
  WASD    = XY 이동 (좌우/전후)
  I / K   = Z 이동 (위/아래)
  ↑↓←→    = 회전 (Ry/Rz)
  Space   = 그리퍼 잡기/펴기 토글
  H       = 홈 포지션 복귀
  + / -   = 이동 스텝 크기 조절
  1~9     = 현재 위치 저장 (슬롯)
  P       = 저장된 위치 전체 출력
  G       = 저장된 위치로 이동 (번호 입력)
  ESC     = 종료
"""

import time
import json
import numpy as np
import cv2
import pyrealsense2 as rs
from fairino import Robot
from PIL import Image, ImageDraw, ImageFont

# ============================================================
#  CONFIG
# ============================================================
ROBOT_IP = "192.168.58.3"
TOOL_NO = 1

# 이동 스텝 (mm / deg) — 고정값, 부드러움 유지
POS_STEP = 2.0
ROT_STEP = 1.0

# JOG 속도 (X/Z로 조절)
JOG_VEL_DEFAULT = 20.0
JOG_VEL_MIN = 5.0
JOG_VEL_MAX = 100.0
JOG_VEL_STEP = 5.0

# 속도
SPEED = 15
SPEED_J = 30

# 홈 포지션
HOME_J = [10.0, -98.0, 100.0, -94.0, -84.0, -111.0]

# 그리퍼 설정
GRIPPER_ID = 1
GRIPPER_OPEN_POS = 100
GRIPPER_CLOSE_POS = 0
GRIPPER_SPEED = 50
GRIPPER_FORCE = 30
GRIPPER_MAX_TIME = 10000
GRIPPER_BLOCK = 0

# 카메라 해상도
CAM_W, CAM_H = 640, 480

# 방향 반전 (카메라 화면 기준으로 직관적이게 조정)
# 로봇 좌표계와 화면이 안 맞으면 여기서 부호 바꾸면 됨
DIR_X = 1.0    # A(-) / D(+)
DIR_Y = -1.0   # W(-) / S(+)  화면 위=로봇 앞쪽이면 -1
DIR_Z = 1.0    # I(+) / K(-)

# 저장 파일
SAVE_FILE = "saved_positions.json"
DOWN_POSE_PATH = "down_pose.json"   # 수직 하강 자세 (Rx, Ry, Rz)

# 한글 폰트
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"


# ============================================================
#  UTILS
# ============================================================
def get_tcp_pose(robot):
    """현재 TCP 포즈 읽기"""
    ret = robot.GetActualTCPPose(0)
    if ret[0] == 0:
        return list(ret[1])
    return None


def get_joint_deg(robot):
    """현재 조인트 각도 읽기 (TCP 기반 fallback 없이 시도)"""
    ret = robot.GetActualJointPosDegree(0)
    if ret[0] == 0:
        return list(ret[1])
    return None


def move_tcp_delta(robot, dx=0, dy=0, dz=0, drx=0, dry=0, drz=0):
    """TCP 델타 이동 → 역기구학 → ServoJ(UDP) 비동기 전송"""
    pose = get_tcp_pose(robot)
    if pose is None:
        return False
    target_tcp = [
        pose[0] + dx, pose[1] + dy, pose[2] + dz,
        pose[3] + drx, pose[4] + dry, pose[5] + drz,
    ]
    ret = robot.GetInverseKin(0, target_tcp)
    if ret[0] != 0:
        return False
    target_joints = ret[1]
    robot.ServoJ(target_joints, [0.0, 0.0, 0.0, 0.0],
                 cmdT=0.033, filterT=0.05, cmdType=1)
    return True


def toggle_gripper(robot, close: bool):
    """그리퍼 열기/닫기"""
    pos = GRIPPER_CLOSE_POS if close else GRIPPER_OPEN_POS
    robot.MoveGripper(
        GRIPPER_ID, pos, GRIPPER_SPEED,
        GRIPPER_FORCE, GRIPPER_MAX_TIME, GRIPPER_BLOCK,
        0, 0, 0, 0
    )


def load_saved_positions():
    """저장된 포지션 로드"""
    try:
        with open(SAVE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_positions(positions):
    """포지션 저장"""
    with open(SAVE_FILE, "w") as f:
        json.dump(positions, f, indent=2, ensure_ascii=False)


def put_text_kr(img, text, pos, font_size=18, color=(255, 255, 255)):
    """한글 텍스트 렌더링 (PIL 기반)"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except OSError:
        font = ImageFont.load_default()
    # BGR -> RGB 색상 변환
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ============================================================
#  HUD DRAWING
# ============================================================
def draw_crosshair(frame):
    """중앙 크로스헤어"""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 25, cy), (cx + 25, cy), (0, 255, 0), 1)
    cv2.line(frame, (cx, cy - 25), (cx, cy + 25), (0, 255, 0), 1)
    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)


def draw_hud(frame, pose, gripper_closed, jog_vel, _unused=0, msg=""):
    """화면 HUD 오버레이"""
    h, w = frame.shape[:2]

    # 반투명 상단 바
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # TCP 좌표
    if pose:
        pos_text = f"TCP  X:{pose[0]:+8.2f}  Y:{pose[1]:+8.2f}  Z:{pose[2]:+8.2f}"
        rot_text = f"ROT  Rx:{pose[3]:+7.2f}  Ry:{pose[4]:+7.2f}  Rz:{pose[5]:+7.2f}"
        cv2.putText(frame, pos_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, rot_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    # 그리퍼 상태
    grip_text = "GRIP: CLOSED" if gripper_closed else "GRIP: OPEN"
    grip_color = (0, 0, 255) if gripper_closed else (0, 255, 0)
    cv2.putText(frame, grip_text, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, grip_color, 2, cv2.LINE_AA)

    # 스텝 크기
    step_text = f"Speed: {jog_vel:.0f}  [X+/Z-]"
    cv2.putText(frame, step_text, (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

    # 하단 조작 가이드
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 30), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)
    guide = "WASD=XY  I/K=Z  Arrows=Rot  SPACE=Grip  H=Home  1-9=Save  P=List  ESC=Quit"
    cv2.putText(frame, guide, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    # 알림 메시지 (중앙)
    if msg:
        text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        tx = (w - text_size[0]) // 2
        ty = h // 2 + 50
        cv2.putText(frame, msg, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    return frame


# ============================================================
#  MAIN
# ============================================================
def main():
    print("=" * 50)
    print("  FR5 Joystick Controller")
    print("=" * 50)

    # ---- Robot 연결 ----
    print(f"[*] 로봇 연결 중... ({ROBOT_IP})")
    robot = Robot.RPC(ROBOT_IP)
    robot.SetSpeed(SPEED)
    print("[+] 로봇 연결 완료")

    # ---- Camera 시작 ----
    print("[*] 카메라 시작 중...")
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, CAM_W, CAM_H, rs.format.bgr8, 30)
    pipeline.start(rs_config)
    print("[+] 카메라 시작 완료")

    # ---- 상태 초기화 ----
    gripper_closed = False
    jog_vel = JOG_VEL_DEFAULT
    saved_positions = load_saved_positions()
    hud_msg = ""
    msg_expire = 0

    # ---- 홈으로 이동 ----
    print("[*] 홈 포지션 이동 중...")
    robot.MoveJ(HOME_J, 0, 0)
    time.sleep(3)
    print("[+] 준비 완료!\n")

    # 그리퍼 초기화 (활성화 + 열기)
    print("[*] 그리퍼 초기화 중...")
    robot.ActGripper(GRIPPER_ID, 1)
    time.sleep(1)
    toggle_gripper(robot, False)
    time.sleep(1)
    print("[+] 그리퍼 초기화 완료")

    cv2.namedWindow("FR5 Joystick", cv2.WINDOW_AUTOSIZE)

    # 서보 모드 시작
    robot.ServoMoveStart()
    time.sleep(0.5)

    # 활성 이동 방향 (키를 누르는 동안 유지)
    active_move = [0, 0, 0, 0, 0, 0]  # dx, dy, dz, drx, dry, drz
    last_key_time = 0
    KEY_HOLD_TIMEOUT = 0.15  # 이 시간 안에 키 반복이 오면 계속 이동

    try:
        while True:
            loop_start = time.time()

            # ---- 카메라 프레임 ----
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())

            # ---- TCP 포즈 읽기 ----
            pose = get_tcp_pose(robot)

            # ---- HUD 메시지 만료 ----
            if time.time() > msg_expire:
                hud_msg = ""

            # ---- 화면 그리기 ----
            draw_crosshair(frame)
            frame = draw_hud(frame, pose, gripper_closed, jog_vel, 0, hud_msg)
            cv2.imshow("FR5 Joystick", frame)

            # ---- 키 입력 ----
            raw_key = cv2.waitKeyEx(1)

            step = jog_vel * 0.05
            rstep = jog_vel * 0.02

            if raw_key != -1:
                key = raw_key & 0xFF

                # --- 이동 키 → 활성 방향 갱신 ---
                if key == ord('w'):
                    active_move = [-step, 0, 0, 0, 0, 0]
                    last_key_time = time.time()
                elif key == ord('s'):
                    active_move = [step, 0, 0, 0, 0, 0]
                    last_key_time = time.time()
                elif key == ord('d'):
                    active_move = [0, step, 0, 0, 0, 0]
                    last_key_time = time.time()
                elif key == ord('a'):
                    active_move = [0, -step, 0, 0, 0, 0]
                    last_key_time = time.time()
                elif key == ord('i'):
                    active_move = [0, 0, step, 0, 0, 0]
                    last_key_time = time.time()
                elif key == ord('k'):
                    active_move = [0, 0, -step, 0, 0, 0]
                    last_key_time = time.time()
                elif raw_key == 65362:  # ↑
                    active_move = [0, 0, 0, 0, rstep, 0]
                    last_key_time = time.time()
                elif raw_key == 65364:  # ↓
                    active_move = [0, 0, 0, 0, -rstep, 0]
                    last_key_time = time.time()
                elif raw_key == 65361:  # ←
                    active_move = [0, 0, 0, 0, 0, rstep]
                    last_key_time = time.time()
                elif raw_key == 65363:  # →
                    active_move = [0, 0, 0, 0, 0, -rstep]
                    last_key_time = time.time()

                # --- 그리퍼 토글 ---
                elif key == ord(' '):
                    active_move = [0, 0, 0, 0, 0, 0]
                    gripper_closed = not gripper_closed
                    toggle_gripper(robot, gripper_closed)
                    state = "CLOSED" if gripper_closed else "OPEN"
                    hud_msg = f"Gripper {state}"
                    msg_expire = time.time() + 1.5
                    print(f"[GRIP] {state}")

                # --- 홈 복귀 ---
                elif key == ord('h'):
                    active_move = [0, 0, 0, 0, 0, 0]
                    robot.ServoMoveEnd()
                    hud_msg = "Moving HOME..."
                    msg_expire = time.time() + 3
                    print("[*] 홈 복귀 중...")
                    robot.MoveJ(HOME_J, 0, 0)
                    time.sleep(3)
                    robot.ServoMoveStart()
                    time.sleep(0.5)
                    hud_msg = "HOME"
                    msg_expire = time.time() + 1.5

                # --- 속도 조절 ---
                elif key == ord('x'):
                    jog_vel = min(jog_vel + JOG_VEL_STEP, JOG_VEL_MAX)
                    hud_msg = f"Speed: {jog_vel:.0f}"
                    msg_expire = time.time() + 1.5
                elif key == ord('z'):
                    jog_vel = max(jog_vel - JOG_VEL_STEP, JOG_VEL_MIN)
                    hud_msg = f"Speed: {jog_vel:.0f}"
                    msg_expire = time.time() + 1.5

                # --- 포지션 저장 (1~9) ---
                elif ord('1') <= key <= ord('9'):
                    slot = chr(key)
                    tcp = get_tcp_pose(robot)
                    joints = get_joint_deg(robot)
                    if tcp:
                        saved_positions[slot] = {
                            "tcp": tcp,
                            "joints": joints,
                            "label": f"pos_{slot}",
                            "timestamp": time.strftime("%H:%M:%S"),
                        }
                        save_positions(saved_positions)
                        hud_msg = f"Slot [{slot}] SAVED!"
                        msg_expire = time.time() + 2
                        print(f"\n[SAVE] Slot {slot}:")
                        print(f"  TCP:    {[round(v,2) for v in tcp]}")
                        if joints:
                            print(f"  Joint:  {[round(v,2) for v in joints]}")
                        print()

                # --- 저장 목록 출력 ---
                elif key == ord('p'):
                    print("\n" + "=" * 55)
                    print("  Saved Positions")
                    print("=" * 55)
                    if not saved_positions:
                        print("  (없음)")
                    for slot, data in sorted(saved_positions.items()):
                        tcp = data["tcp"]
                        label = data.get("label", "")
                        ts = data.get("timestamp", "")
                        print(f"  [{slot}] {label} ({ts})")
                        print(f"      TCP: [{tcp[0]:.2f}, {tcp[1]:.2f}, {tcp[2]:.2f}, "
                              f"{tcp[3]:.2f}, {tcp[4]:.2f}, {tcp[5]:.2f}]")
                        if data.get("joints"):
                            j = data["joints"]
                            print(f"      J:   [{j[0]:.2f}, {j[1]:.2f}, {j[2]:.2f}, "
                                  f"{j[3]:.2f}, {j[4]:.2f}, {j[5]:.2f}]")
                    print("=" * 55)
                    print()
                    print("# --- Copy-paste for pipeline ---")
                    for slot, data in sorted(saved_positions.items()):
                        tcp = data["tcp"]
                        label = data.get("label", f"POS_{slot}")
                        print(f"{label.upper()}_TCP = {[round(v,2) for v in tcp]}")
                        if data.get("joints"):
                            j = data["joints"]
                            print(f"{label.upper()}_J = {[round(v,2) for v in j]}")
                    print()
                    hud_msg = f"{len(saved_positions)} positions (see terminal)"
                    msg_expire = time.time() + 2

                # --- 저장 위치로 이동 ---
                elif key == ord('g'):
                    active_move = [0, 0, 0, 0, 0, 0]
                    robot.ServoMoveEnd()
                    hud_msg = "Press 1-9 to GO..."
                    msg_expire = time.time() + 3
                    next_key = cv2.waitKey(3000) & 0xFF
                    slot = chr(next_key) if 0 <= next_key < 256 else ""
                    if slot in saved_positions:
                        target = saved_positions[slot]
                        if target.get("joints"):
                            hud_msg = f"Moving to [{slot}]..."
                            msg_expire = time.time() + 3
                            print(f"[GO] Slot {slot} (MoveJ)")
                            robot.MoveJ(target["joints"], 0, 0)
                            time.sleep(3)
                        else:
                            hud_msg = f"Moving to [{slot}]..."
                            msg_expire = time.time() + 3
                            print(f"[GO] Slot {slot} (MoveL)")
                            robot.MoveL(target["tcp"], 0, 0)
                            time.sleep(2)
                        hud_msg = f"Arrived [{slot}]"
                        msg_expire = time.time() + 1.5
                    else:
                        hud_msg = f"Slot [{slot}] empty"
                        msg_expire = time.time() + 1.5
                    robot.ServoMoveStart()
                    time.sleep(0.5)

                # --- 종료 ---
                elif key == 27:  # ESC
                    print("[*] 종료 중...")
                    break

            # ---- 키 홀드 타임아웃: 키를 뗐으면 이동 중지 ----
            if time.time() - last_key_time > KEY_HOLD_TIMEOUT:
                active_move = [0, 0, 0, 0, 0, 0]

            # ---- 활성 이동이 있으면 매 루프마다 계속 이동 ----
            if any(v != 0 for v in active_move):
                move_tcp_delta(robot, *active_move)

            # ---- 루프 타이밍 유지 (~30Hz) ----
            elapsed = time.time() - loop_start
            if elapsed < 0.033:
                time.sleep(0.033 - elapsed)

    except KeyboardInterrupt:
        print("\n[*] Ctrl+C 종료")
    finally:
        robot.ServoMoveEnd()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[+] 종료 완료")


if __name__ == "__main__":
    main()