"""
FR5 로봇 위치 조정 유틸리티 (카메라 라이브뷰)
- D455 카메라 실시간 영상 + 십자선 + depth 표시
- 티칭펜던트로 위치 잡으면서 카메라 화면 확인
- 키보드로 위치 저장, 코드 출력

사용법:
  1) 티칭펜던트로 로봇을 원하는 위치로 이동
  2) 카메라 화면 보면서 시야 확인
  3) 키보드로 위치 저장
  4) c 누르면 코드 형태로 출력

실행: python robot_teach.py
"""
import sys
import os
import time
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw as PilDraw
import pyrealsense2 as rs
from fairino import Robot

ROBOT_IP = "192.168.58.3"
TOOL_NO  = 1

# ══════════════════════════════════════════════════════
# 초기화
# ══════════════════════════════════════════════════════
print("=" * 55)
print("FR5 위치 조정 유틸리티 (카메라 라이브뷰)")
print("=" * 55)

# ── D455 카메라 ──────────────────────────────────────
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

# ── FR5 로봇 ─────────────────────────────────────────
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
print("=" * 55)

# ── 한글 폰트 ────────────────────────────────────────
_FONT_PATHS = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
]

def _load_font(size):
    for path in _FONT_PATHS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
    return ImageFont.load_default()

FONT_SM = _load_font(13)
FONT_MD = _load_font(15)

def put_kr(img, text, xy, font, color=(255,255,255)):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = PilDraw.Draw(pil)
    draw.text(xy, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ══════════════════════════════════════════════════════
# 포즈 읽기
# ══════════════════════════════════════════════════════
def get_tcp():
    try:
        _, p = robot.GetActualTCPPose(TOOL_NO)
        if isinstance(p, (list, tuple)) and len(p) == 6:
            return [round(v, 2) for v in p]
    except:
        pass
    return None

def get_joint():
    try:
        _, j = robot.GetActualJointPosDegree(0)
        if isinstance(j, (list, tuple)) and len(j) == 6:
            return [round(v, 2) for v in j]
    except:
        pass
    return None


# ══════════════════════════════════════════════════════
# 저장 슬롯
# ══════════════════════════════════════════════════════
slots = {
    "1": {"name": "HOME_J",  "data": None},
    "2": {"name": "WAY_J",   "data": None},
    "3": {"name": "SCAN1_J", "data": None},
    "4": {"name": "PLACE_J", "data": None},
    "5": {"name": "CUSTOM1", "data": None},
    "6": {"name": "CUSTOM2", "data": None},
}

last_msg      = ""
last_msg_time = 0

def set_msg(text):
    global last_msg, last_msg_time
    last_msg = text
    last_msg_time = time.time()
    print(f"  {text}")


# ══════════════════════════════════════════════════════
# 화면 그리기
# ══════════════════════════════════════════════════════
CAM_W, CAM_H = 640, 480
INFO_H = 155
TOTAL_H = CAM_H + INFO_H

WINDOW = "Robot Teach"
cv2.namedWindow(WINDOW)


def draw_frame(color_bgr, depth_frame):
    canvas = np.zeros((TOTAL_H, CAM_W, 3), dtype=np.uint8)
    vis = color_bgr.copy()

    # ── 십자선 + 중심 depth ──────────────────────────
    cx, cy = 320, 240
    cv2.line(vis, (cx - 25, cy), (cx + 25, cy), (0, 255, 255), 1)
    cv2.line(vis, (cx, cy - 25), (cx, cy + 25), (0, 255, 255), 1)
    cv2.circle(vis, (cx, cy), 3, (0, 255, 255), -1)

    center_d = depth_frame.get_distance(cx, cy) * 1000
    cv2.putText(vis, f"{center_d:.0f}mm", (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # ── 그리드 ───────────────────────────────────────
    for x in range(0, CAM_W, 160):
        cv2.line(vis, (x, 0), (x, CAM_H), (40, 40, 40), 1)
    for y in range(0, CAM_H, 120):
        cv2.line(vis, (0, y), (CAM_W, y), (40, 40, 40), 1)

    canvas[INFO_H:, :] = vis

    # ── 정보 패널 ────────────────────────────────────
    cv2.rectangle(canvas, (0, 0), (CAM_W, INFO_H), (25, 25, 25), -1)
    cv2.line(canvas, (0, INFO_H), (CAM_W, INFO_H), (70, 70, 70), 1)

    tcp = get_tcp()
    j   = get_joint()

    y0 = 6
    if tcp:
        canvas = put_kr(canvas,
            f"TCP  X={tcp[0]:>8.1f}  Y={tcp[1]:>8.1f}  Z={tcp[2]:>8.1f}",
            (8, y0), FONT_SM, (0, 220, 255))
        canvas = put_kr(canvas,
            f"     Rx={tcp[3]:>7.1f}  Ry={tcp[4]:>7.1f}  Rz={tcp[5]:>7.1f}",
            (8, y0 + 16), FONT_SM, (0, 180, 220))
    else:
        canvas = put_kr(canvas, "TCP: 읽기 실패", (8, y0), FONT_SM, (100, 100, 255))

    y0 += 36
    if j:
        canvas = put_kr(canvas,
            f"Joint  {j[0]:>7.1f}  {j[1]:>7.1f}  {j[2]:>7.1f}  {j[3]:>7.1f}  {j[4]:>7.1f}  {j[5]:>7.1f}",
            (8, y0), FONT_SM, (180, 220, 180))
    else:
        canvas = put_kr(canvas, "Joint: 읽기 실패", (8, y0), FONT_SM, (100, 100, 255))

    # ── 저장 슬롯 ────────────────────────────────────
    y0 += 22
    for row_keys in [["1","2","3"], ["4","5","6"]]:
        x0 = 8
        for k in row_keys:
            s = slots[k]
            if s["data"]:
                txt = f"{k}:{s['name']} ✓"
                col = (120, 255, 120)
            else:
                txt = f"{k}:{s['name']}"
                col = (80, 80, 80)
            canvas = put_kr(canvas, txt, (x0, y0), FONT_SM, col)
            x0 += 210
        y0 += 17

    # ── 메시지 ───────────────────────────────────────
    y0 += 4
    if last_msg and time.time() - last_msg_time < 5:
        canvas = put_kr(canvas, last_msg, (8, y0), FONT_MD, (80, 255, 120))

    # ── 단축키 ───────────────────────────────────────
    canvas = put_kr(canvas, "1~6: 저장  c: 코드출력  p: 포즈출력  r: 에러리셋  q: 종료",
                    (8, INFO_H - 18), FONT_SM, (70, 70, 70))

    return canvas


def print_code():
    print("\n" + "=" * 55)
    print("# ── picking_pipeline.py에 복사 ──")
    print("=" * 55)
    for key in sorted(slots.keys()):
        s = slots[key]
        if s["data"]:
            j = s["data"].get("joint")
            t = s["data"].get("tcp")
            j_str = str(j) if j else "None"
            t_str = f"[{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}]" if t else ""
            print(f'{s["name"]:<10} = {j_str:<58} # TCP: {t_str}')
    print("=" * 55 + "\n")


# ══════════════════════════════════════════════════════
# 메인 루프
# ══════════════════════════════════════════════════════
print("\n단축키: 1~6 저장 | c 코드출력 | p 포즈출력 | r 에러리셋 | q 종료\n")

try:
    while True:
        frames  = rs_pipeline.wait_for_frames()
        aligned = align.process(frames)
        color   = aligned.get_color_frame()
        depth   = aligned.get_depth_frame()
        color_bgr = np.asanyarray(color.get_data())

        canvas = draw_frame(color_bgr, depth)
        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break

        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
            k = chr(key)
            tcp = get_tcp()
            j   = get_joint()
            if tcp:
                slots[k]["data"] = {"tcp": tcp, "joint": j}
                set_msg(f"✅ {slots[k]['name']} 저장!  TCP=[{tcp[0]:.1f}, {tcp[1]:.1f}, {tcp[2]:.1f}]")
            else:
                set_msg("⚠ 포즈 읽기 실패")

        elif key == ord('c'):
            print_code()
            set_msg("코드 출력 완료 (터미널 확인)")

        elif key == ord('p'):
            tcp = get_tcp()
            j   = get_joint()
            print(f"\n  TCP:   {tcp}")
            print(f"  Joint: {j}\n")

        elif key == ord('r'):
            robot.ResetAllError()
            time.sleep(0.3)
            set_msg("✅ 에러 리셋 완료")

finally:
    rs_pipeline.stop()
    cv2.destroyAllWindows()
    print("종료")