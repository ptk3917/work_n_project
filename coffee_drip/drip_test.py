"""
커피 드립 파이프라인 - 단계별 테스트
잡기: 옆접근 → 그립(✊) → 위로 들기 → tool -Z 후퇴
놓기: 위접근 → 내려놓기(🖐) → tool -Z 후퇴

사용법: python coffee_drip_test.py
"""

import time
import math
import sys
import numpy as np
from fairino import Robot

# ============================================================
# 설정
# ============================================================
ROBOT_IP = "192.168.58.2"
TOOL_NO = 1

SPEED_SLOW = 10
SPEED_NORMAL = 30
SPEED_J_SLOW = 30
SPEED_J_NORMAL = 40
SPEED_J_FAST = 60

GRIPPER_OPEN = 100
GRIPPER_CLOSE = 0

LIFT_HEIGHT = 50       # 잡은 후 들어올리기 (mm)
RETRACT = 100          # tool -Z 후퇴 거리 (mm)

# 드립 파라미터 (Circle 기반)
DDUM_TIME_WAIT = 20         # 뜸뜰이기
DRIP_RADIUS = 10.0         # 원 반지름 (mm)
DRIP_ROUNDS = 5            # 나선형 시작 바퀴 수 (= 반지름/shrink)
DRIP_VEL = 20.0            # Circle 속도
SPIRAL_SHRINK = 2.0        # 매 바퀴 반지름 줄이는 양 (mm) → 30/6=5바퀴
SPIRAL_WAIT = 10.0          # 바퀴 사이 대기 시간 (초)

# ============================================================
# 포지션 (robot_teach_drip.py → c → 여기에 붙여넣기)
# ============================================================

# --- 공통 ---
HOME_J                     = [-46.28, -123.5, 110.89, 14.36, 115.72, 5.19]

# --- 1단계: 드리퍼→서버 ---
DRIPPER_APPROACH_J         = [-30.77, -74.43, 106.83, -34.89, 133.39, -4.19]
DRIPPER_PICK_J             = [-33.69, -72.0, 103.32, -33.7, 130.47, -4.02]
SERVER_ABOVE_J             = [11.47, -86.94, 123.06, -16.16, -6.99, -21.15]  # ✊ 잡기
SERVER_PLACE_J             = [12.24, -79.78, 126.14, -24.03, -6.28, -23.55]  # ✊ 잡기

# --- 2단계: 커피가루 붓기 ---
CUP_APPROACH_J             = [-49.6, -86.25, 130.6, -42.88, 108.8, 5.24]  # 🖐 놓기
CUP_PICK_J                 = [-53.98, -76.29, 118.86, -41.13, 104.42, 5.12]  # 🖐 놓기
POUR_ABOVE_J               = [12.9, -101.12, 131.93, -37.53, 7.33, 5.24]  # 🖐 놓기
POUR_J                     = [13.72, -99.84, 135.33, -43.43, 4.52, -142.26]  # ✊ 잡기
CUP_RETURN_ABOVE_J         = [-57.3, -86.61, 117.62, -30.35, 95.62, -2.21]  # ✊ 잡기
CUP_RETURN_J               = [-57.99, -77.74, 121.34, -42.93, 94.94, -2.22]  # ✊ 잡기

# --- 3단계: 드립 ---
POT_APPROACH_J             = [-79.61, -96.04, 72.88, 25.54, 79.72, 4.4]
POT_PICK_J                 = [-77.6, -84.0, 61.19, 27.11, 82.55, -2.61]
DRIP_ABOVE_J               = [81.74, -105.35, 117.91, -10.36, 92.87, -9.12]
DRIP_ABOVE_TCP             = [18.28, -507.06, 215.31, 92.23, 9.22, -10.77]
DRIP_CENTER_TCP            = [-38.18, -507.04, 354.25, 105.44, -33.83, -16.13] 
POT_RETURN_ABOVE_J         = [-79.81, -73.53, 31.3, 44.63, 78.69, -4.82]  # ✊ 잡기
POT_RETURN_J               = [-77.60, -84.00, 61.19, 27.11, 82.55, -2.61] # ✊ 잡기
# --- 4단계: 찌꺼기 버리기 ---
SVR_DRIP_APPROACH_J        = [1.19, -84.12, 130.56, -38.28, -17.03, -9.15]  # 🖐 놓기
SVR_DRIP_PICK_J            = [12.24, -79.78, 126.14, -24.03, -6.28, -23.55]  # 🖐 놓기
TRASH_ABOVE_J              = [95.59, -101.43, 123.82, -23.37, 77.17, -6.2]  # ✊ 잡기
TRASH_FLIP_J               = [90.66, -73.77, 139.45, -66.7, 69.95, -178.89]  # ✊ 잡기
DRIP_BACK_ABOVE_J          = [-21.5, -91.71, 105.31, -9.21, 140.66, 5.16]  # ✊ 잡기
DRIP_BACK_PLACE_J          = [-33.69, -72.44, 102.88, -32.82, 130.47, -4.02]  # 🖐 놓기

# --- 5단계: 서빙 ---
SERVER_APPROACH_J          = [0, 0, 0, 0, 0, 0]  # TODO
SERVER_PICK_J              = [0, 0, 0, 0, 0, 0]  # TODO
CUP_POUR_ABOVE_J           = [0, 0, 0, 0, 0, 0]  # TODO
CUP_POUR_J                 = [0, 0, 0, 0, 0, 0]  # TODO
SVR_RETURN_ABOVE_J         = [0, 0, 0, 0, 0, 0]  # TODO
SVR_RETURN_J               = [0, 0, 0, 0, 0, 0]  # TODO


# ============================================================
# 유틸리티
# ============================================================
robot = None

def connect():
    global robot
    robot = Robot.RPC(ROBOT_IP)
    robot.SetSpeed(SPEED_NORMAL)
    print(f"[OK] 로봇 연결됨 ({ROBOT_IP})")
    print("  그리퍼 초기화 중...")
    robot.ActGripper(1, 1)
    time.sleep(2.0)
    robot.MoveGripper(1, GRIPPER_OPEN, 50, 10, 10000, 0, 0, 0, 0, 0)
    time.sleep(1.0)
    print("  그리퍼 OK (열림)")


def wait_done(pre_delay=0.4, timeout=30):
    time.sleep(pre_delay)
    t0 = time.time()
    while time.time() - t0 < timeout:
        ret = robot.GetRobotMotionDone()
        if isinstance(ret, (list, tuple)):
            done = ret[1]
        else:
            done = ret
        if done == 1:
            return True
        time.sleep(0.1)
    print("[WARN] timeout!")
    return False


def grip_open(width=GRIPPER_OPEN):
    robot.MoveGripper(1, width, 50, 10, 10000, 0, 0, 0, 0, 0)
    time.sleep(1.0)

def grip_close(width=GRIPPER_CLOSE, force=50):
    robot.MoveGripper(1, width, 50, force, 10000, 0, 0, 0, 0, 0)
    time.sleep(1.0)

def mj(joints, speed=SPEED_J_NORMAL):
    robot.SetSpeed(speed)
    robot.MoveJ(joints, TOOL_NO, 0, [0,0,0,0,0,0])
    wait_done()

def ml(tcp, speed=SPEED_SLOW):
    robot.SetSpeed(speed)
    robot.MoveL(tcp, TOOL_NO, 0, [0,0,0,0,0,0])
    wait_done()

def get_tcp():
    time.sleep(0.5)
    poses = []
    for _ in range(5):
        ret = robot.GetActualTCPPose(0)
        if ret[0] == 0:
            poses.append(ret[1])
        time.sleep(0.1)
    if not poses:
        return None
    return [sum(x) / len(x) for x in zip(*poses)]

def home():
    print("  → 홈")
    mj(HOME_J, SPEED_J_FAST)

def lift_up(height=LIFT_HEIGHT):
    """현재 위치에서 base Z만 올리기"""
    tcp = get_tcp()
    if tcp:
        lift = tcp.copy()
        lift[2] += height
        ml(lift, SPEED_SLOW)
    else:
        print("  [WARN] TCP 읽기 실패")

def move_tool(dx=0, dy=0, dz=0):
    """tool 좌표계 기준으로 이동 (mm)"""
    tcp = get_tcp()
    if not tcp:
        print("  [WARN] TCP 읽기 실패")
        return
    rx, ry, rz = np.radians(tcp[3]), np.radians(tcp[4]), np.radians(tcp[5])
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    R = Rz @ Ry @ Rx
    offset = dx * R[:, 0] + dy * R[:, 1] + dz * R[:, 2]
    new_tcp = tcp.copy()
    new_tcp[0] += offset[0]
    new_tcp[1] += offset[1]
    new_tcp[2] += offset[2]
    ml(new_tcp, SPEED_SLOW)

def pause(msg="계속하려면 Enter..."):
    input(f"\n  ⏸  {msg}")
    print()

def check_pos(name, joints):
    if joints == [0, 0, 0, 0, 0, 0]:
        print(f"  [!] {name} 미티칭!")
        return False
    return True


# ============================================================
# 공통 패턴
# ============================================================

def pick(approach_j, pick_j, grip_width=GRIPPER_CLOSE, grip_force=50, label="물체"):
    """옆접근 → 잡기(✊) → 위로 들기 → tool -Z 후퇴"""
    print(f"  [{label}] 🖐 열고 옆에서 접근")
    grip_open()
    mj(approach_j, SPEED_J_NORMAL)
    pause(f"[{label}] 옆에서 들어감")

    mj(pick_j, SPEED_J_SLOW)
    pause(f"[{label}] ✊ 그리퍼 닫기")

    grip_close(width=grip_width, force=grip_force)
    time.sleep(0.5)

    print(f"  [{label}] 위로 들기 (+{LIFT_HEIGHT}mm)")
    lift_up()

    print(f"  [{label}] tool -Z 후퇴 ({RETRACT}mm)")
    move_tool(dz=-RETRACT)

    print(f"  [{label}] ✊ 잡기 완료")


def place(above_j, place_j, label="물체"):
    """위접근 → 놓기(🖐) → tool -Z 후퇴"""
    print(f"  [{label}] ✊ 잡고 위에서 접근")
    mj(above_j, SPEED_J_NORMAL)
    pause(f"[{label}] 내려놓기")

    mj(place_j, SPEED_J_SLOW)
    pause(f"[{label}] 🖐 그리퍼 열기")

    grip_open()
    time.sleep(0.3)

    print(f"  [{label}] tool -Z 후퇴 ({RETRACT}mm)")
    move_tool(dz=-RETRACT)

    print(f"  [{label}] 🖐 놓기 완료")


def pour_shake(duration=3.0, shake_z=5, shake_interval=0.3):
    """붓는 동안 base Z 위아래로 흔들기"""
    tcp = get_tcp()
    if not tcp:
        print("  [WARN] TCP 읽기 실패, 흔들기 스킵")
        time.sleep(duration)
        return
    center = tcp.copy()
    t0 = time.time()
    up = True
    while time.time() - t0 < duration:
        shake = center.copy()
        shake[2] += shake_z if up else -shake_z
        ml(shake, SPEED_NORMAL)
        time.sleep(shake_interval)
        up = not up
    ml(center, SPEED_SLOW)


# ============================================================
# Circle 기반 드립 함수
# ============================================================

def do_circle(radius, vel=DRIP_VEL):
    """DRIP_CENTER_TCP 기준 한 바퀴 (Circle 명령)"""
    cx, cy, cz = DRIP_CENTER_TCP[0], DRIP_CENTER_TCP[1], DRIP_CENTER_TCP[2]
    rx, ry, rz = DRIP_CENTER_TCP[3], DRIP_CENTER_TCP[4], DRIP_CENTER_TCP[5]

    def pt(angle):
        rad = math.radians(angle)
        return [cx + radius * math.cos(rad), cy + radius * math.sin(rad), cz, rx, ry, rz]

    ml(pt(0), SPEED_SLOW)

    ret = robot.Circle(
        desc_pos_p=pt(90), tool_p=TOOL_NO, user_p=0,
        desc_pos_t=pt(180), tool_t=TOOL_NO, user_t=0,
        vel_p=vel, vel_t=vel,
    )
    print(f"    Circle ret={ret} (R={radius:.1f}mm)")
    if ret == 0:
        wait_done(pre_delay=1.0, timeout=60)
    return ret


def do_spiral_drip(start_radius=DRIP_RADIUS, shrink=SPIRAL_SHRINK, vel=DRIP_VEL, wait=SPIRAL_WAIT):
    """나선형: 원 → 세우기 → 대기 → 기울이기 → 다음 원 (반지름 줄여가며)"""
    radius = start_radius
    round_num = 1
    while radius > 0.5:
        print(f"  라운드 {round_num} (R={radius:.1f}mm)")
        do_circle(radius, vel)

        radius -= shrink
        if radius > 0.5:
            print(f"  세우기 → {wait:.0f}초 대기")
            ml(DRIP_ABOVE_TCP, SPEED_J_FAST)
            time.sleep(wait)
            print(f"  기울이기 (다음 라운드)")
            ml(DRIP_CENTER_TCP, SPEED_J_SLOW)

        round_num += 1
    print("  중심으로 복귀")
    ml(DRIP_CENTER_TCP, SPEED_SLOW)


# ============================================================
# 개별 테스트
# ============================================================

def test_home():
    print("\n--- 홈 ---")
    home()

def test_gripper():
    print("\n--- 그리퍼 ---")
    grip_open(); pause("닫기?")
    grip_close(); pause("열기?")
    grip_open()

def test_tcp():
    print("\n--- TCP ---")
    tcp = get_tcp()
    if tcp:
        print(f"  [{tcp[0]:.2f}, {tcp[1]:.2f}, {tcp[2]:.2f}, "
              f"{tcp[3]:.2f}, {tcp[4]:.2f}, {tcp[5]:.2f}]")

# ----- 1단계: 드리퍼 → 서버 -----
def test_1a():
    print("\n--- 1-A: 드리퍼 잡기 ---")
    if not check_pos("DRIPPER_APPROACH_J", DRIPPER_APPROACH_J): return
    if not check_pos("DRIPPER_PICK_J", DRIPPER_PICK_J): return
    home()
    pick(DRIPPER_APPROACH_J, DRIPPER_PICK_J, label="드리퍼")
    pause("홈?")
    home()

def test_1b():
    print("\n--- 1-B: 서버에 놓기 (✊ 들고있는 상태) ---")
    if not check_pos("SERVER_ABOVE_J", SERVER_ABOVE_J): return
    if not check_pos("SERVER_PLACE_J", SERVER_PLACE_J): return
    place(SERVER_ABOVE_J, SERVER_PLACE_J, label="드리퍼→서버")

def test_1():
    print("\n=== 1단계 전체 ===")
    if not check_pos("DRIPPER_APPROACH_J", DRIPPER_APPROACH_J): return
    if not check_pos("SERVER_PLACE_J", SERVER_PLACE_J): return
    home()
    pick(DRIPPER_APPROACH_J, DRIPPER_PICK_J, label="드리퍼")
    pause("서버에 놓기?")
    place(SERVER_ABOVE_J, SERVER_PLACE_J, label="드리퍼→서버")
    home()
    print("  [OK] 1단계 완료")

# ----- 2단계: 커피가루 붓기 -----
def test_2a():
    print("\n--- 2-A: 커피 컵 잡기 ---")
    if not check_pos("CUP_APPROACH_J", CUP_APPROACH_J): return
    if not check_pos("CUP_PICK_J", CUP_PICK_J): return
    home()
    pick(CUP_APPROACH_J, CUP_PICK_J, label="커피컵")
    pause("홈?")
    home()

def test_2b():
    print("\n--- 2-B: 붓기 (✊ 들고있는 상태) ---")
    if not check_pos("POUR_ABOVE_J", POUR_ABOVE_J): return
    if not check_pos("POUR_J", POUR_J): return
    pause("드리퍼 위로 이동 (컵 세운 상태)")
    mj(POUR_ABOVE_J, SPEED_J_NORMAL)
    pause("기울여서 붓기")
    mj(POUR_J, SPEED_J_FAST)
    print("  붓는 중 (흔들기 3초)...")
    pour_shake(duration=3.0, shake_z=5)
    mj(POUR_ABOVE_J, SPEED_J_FAST)

def test_2c():
    print("\n--- 2-C: 빈 컵 놓기 ---")
    if not check_pos("CUP_RETURN_ABOVE_J", CUP_RETURN_ABOVE_J): return
    if not check_pos("CUP_RETURN_J", CUP_RETURN_J): return
    place(CUP_RETURN_ABOVE_J, CUP_RETURN_J, label="빈컵")

def test_2():
    print("\n=== 2단계 전체 ===")
    home()
    pick(CUP_APPROACH_J, CUP_PICK_J, label="커피컵")
    pause("붓기?")
    mj(POUR_ABOVE_J, SPEED_J_NORMAL)
    mj(POUR_J, SPEED_J_FAST)
    print("  붓는 중 (흔들기 3초)...")
    pour_shake(duration=3.0, shake_z=5)
    mj(POUR_ABOVE_J, SPEED_J_FAST)
    pause("컵 놓기?")
    place(CUP_RETURN_ABOVE_J, CUP_RETURN_J, label="빈컵")
    home()
    print("  [OK] 2단계 완료")

# ----- 3단계: 드립 -----
def test_3a():
    print("\n--- 3-A: 드립포트 잡기 ---")
    if not check_pos("POT_APPROACH_J", POT_APPROACH_J): return
    if not check_pos("POT_PICK_J", POT_PICK_J): return
    home()
    pick(POT_APPROACH_J, POT_PICK_J, grip_width=GRIPPER_CLOSE, label="포트")
    pause("홈?")
    home()

def test_3b():
    """3-B 전체: 세움 → 뜸들이기 → 나선형 드립(Circle) → 세움"""
    print("\n--- 3-B: 드립 전체 (✊ 들고있는 상태) ---")
    if not check_pos("DRIP_ABOVE_TCP", DRIP_ABOVE_TCP): return
    if not check_pos("DRIP_CENTER_TCP", DRIP_CENTER_TCP): return

    print("  홈 경유")
    mj(HOME_J, SPEED_J_NORMAL)
    mj(DRIP_ABOVE_J, SPEED_J_FAST)
    pause("DRIP_ABOVE_TCP로 이동")
    ml(DRIP_ABOVE_TCP, SPEED_J_SLOW)

    pause("기울이기 (뜸들이기 시작)")
    ml(DRIP_CENTER_TCP, SPEED_J_SLOW)
    time.sleep(2.0)
    ml(DRIP_ABOVE_TCP, SPEED_J_FAST)
    print("  뜸들이기 대기 ",DDUM_TIME_WAIT,"초...")
    time.sleep(DDUM_TIME_WAIT)

    pause("다시 기울이기 (나선형 드립 시작)")
    ml(DRIP_CENTER_TCP, SPEED_J_SLOW)

    do_spiral_drip()

    print("  세우기 (물 멈춤)")
    ml(DRIP_ABOVE_TCP, SPEED_SLOW)
    mj(DRIP_ABOVE_J, SPEED_J_FAST)

def test_3b1():
    """드리퍼 위로 이동 (홈→DRIP_ABOVE_J 경유)"""
    print("\n--- 3-B1: 드리퍼 위로 (세운 상태) ---")
    if not check_pos("DRIP_ABOVE_TCP", DRIP_ABOVE_TCP): return
    print("  홈 경유")
    mj(HOME_J, SPEED_J_NORMAL)
    mj(DRIP_ABOVE_J, SPEED_J_FAST)
    ml(DRIP_ABOVE_TCP, SPEED_J_SLOW)
    print("  [OK]")

def test_3b2():
    """기울이기 테스트"""
    print("\n--- 3-B2: 기울이기 테스트 ---")
    if not check_pos("DRIP_CENTER_TCP", DRIP_CENTER_TCP): return
    pause("기울이기 (물 나옴)")
    ml(DRIP_CENTER_TCP, SPEED_J_SLOW)
    pause("세우기?")
    ml(DRIP_ABOVE_TCP, SPEED_J_SLOW)
    print("  [OK]")

def test_3b3():
    """원형 1바퀴 (Circle, 기울어진 상태에서)"""
    print(f"\n--- 3-B3: 원형 1바퀴 (R={DRIP_RADIUS}mm) ---")
    if not check_pos("DRIP_CENTER_TCP", DRIP_CENTER_TCP): return
    pause("1바퀴")
    do_circle(DRIP_RADIUS)
    ml(DRIP_CENTER_TCP, SPEED_SLOW)
    print("  [OK]")

def test_3c():
    print("\n--- 3-C: 드립포트 놓기 ---")
    if not check_pos("POT_RETURN_ABOVE_J", POT_RETURN_ABOVE_J): return
    if not check_pos("POT_RETURN_J", POT_RETURN_J): return
    place(POT_RETURN_ABOVE_J, POT_RETURN_J, label="포트")

def test_3():
    print("\n=== 3단계 전체 ===")
    home()
    pick(POT_APPROACH_J, POT_PICK_J, grip_width=GRIPPER_CLOSE, label="포트")
    pause("드립?")
    test_3b()
    pause("포트 놓기?")
    home()
    place(POT_RETURN_ABOVE_J, POT_RETURN_J, label="포트")
    home()
    print("  [OK] 3단계 완료")

# ----- 4단계: 찌꺼기 버리기 -----
def test_4a():
    print("\n--- 4-A: 서버에서 드리퍼 잡기 ---")
    if not check_pos("SVR_DRIP_APPROACH_J", SVR_DRIP_APPROACH_J): return
    if not check_pos("SVR_DRIP_PICK_J", SVR_DRIP_PICK_J): return
    home()
    pick(SVR_DRIP_APPROACH_J, SVR_DRIP_PICK_J, label="서버위드리퍼")
    pause("홈?")
    home()

def test_4b():
    print("\n--- 4-B: 뒤집어서 버리기 (✊ 들고있는 상태) ---")
    if not check_pos("TRASH_ABOVE_J", TRASH_ABOVE_J): return
    if not check_pos("TRASH_FLIP_J", TRASH_FLIP_J): return
    pause("쓰레기통 위로")
    mj(TRASH_ABOVE_J, SPEED_J_NORMAL)
    pause("뒤집기")
    mj(TRASH_FLIP_J, SPEED_J_SLOW)
    time.sleep(2.0)
    pause("흔들기?")
    current = get_tcp()
    if current:
        for _ in range(3):
            sl = current.copy(); sl[0] -= 10
            ml(sl, SPEED_NORMAL)
            sr = current.copy(); sr[0] += 10
            ml(sr, SPEED_NORMAL)
        ml(current, SPEED_SLOW)
    mj(TRASH_ABOVE_J, SPEED_J_SLOW)

def test_4c():
    print("\n--- 4-C: 드리퍼 원위치 ---")
    if not check_pos("DRIP_BACK_ABOVE_J", DRIP_BACK_ABOVE_J): return
    if not check_pos("DRIP_BACK_PLACE_J", DRIP_BACK_PLACE_J): return
    place(DRIP_BACK_ABOVE_J, DRIP_BACK_PLACE_J, label="드리퍼원위치")

def test_4():
    print("\n=== 4단계 전체 ===")
    home()
    pick(SVR_DRIP_APPROACH_J, SVR_DRIP_PICK_J, label="서버위드리퍼")
    pause("쓰레기통?")
    test_4b()
    pause("드리퍼 원위치?")
    place(DRIP_BACK_ABOVE_J, DRIP_BACK_PLACE_J, label="드리퍼원위치")
    home()
    print("  [OK] 4단계 완료")

# ----- 5단계: 서빙 -----
def test_5a():
    print("\n--- 5-A: 서버 잡기 ---")
    if not check_pos("SERVER_APPROACH_J", SERVER_APPROACH_J): return
    if not check_pos("SERVER_PICK_J", SERVER_PICK_J): return
    home()
    pick(SERVER_APPROACH_J, SERVER_PICK_J, grip_force=30, label="서버")
    pause("홈?")
    home()

def test_5b():
    print("\n--- 5-B: 컵에 따르기 (✊ 들고있는 상태) ---")
    if not check_pos("CUP_POUR_ABOVE_J", CUP_POUR_ABOVE_J): return
    if not check_pos("CUP_POUR_J", CUP_POUR_J): return
    pause("컵 위로 이동 (서버 세운 상태)")
    mj(CUP_POUR_ABOVE_J, SPEED_J_NORMAL)
    pause("기울여서 따르기")
    mj(CUP_POUR_J, SPEED_J_SLOW)
    print("  따르는 중... (5초)")
    time.sleep(5.0)
    mj(CUP_POUR_ABOVE_J, SPEED_J_SLOW)

def test_5c():
    print("\n--- 5-C: 서버 놓기 ---")
    if not check_pos("SVR_RETURN_ABOVE_J", SVR_RETURN_ABOVE_J): return
    if not check_pos("SVR_RETURN_J", SVR_RETURN_J): return
    place(SVR_RETURN_ABOVE_J, SVR_RETURN_J, label="서버")

def test_5():
    print("\n=== 5단계 전체 ===")
    home()
    pick(SERVER_APPROACH_J, SERVER_PICK_J, grip_force=30, label="서버")
    pause("따르기?")
    mj(CUP_POUR_ABOVE_J, SPEED_J_NORMAL)
    mj(CUP_POUR_J, SPEED_J_SLOW)
    print("  따르는 중... (5초)")
    time.sleep(5.0)
    mj(CUP_POUR_ABOVE_J, SPEED_J_SLOW)
    pause("서버 놓기?")
    place(SVR_RETURN_ABOVE_J, SVR_RETURN_J, label="서버")
    home()
    print("  [OK] 5단계 완료")

# ----- 전체 -----
def test_all():
    print("\n========== 전체 시퀀스 ==========")
    home()
    pause("1단계: 🖐→드리퍼잡기(✊)→서버놓기(🖐)")
    test_1()
    pause("2단계: 🖐→컵잡기(✊)→붓기→컵놓기(🖐)")
    test_2()
    pause("3단계: 🖐→포트잡기(✊)→드립→포트놓기(🖐)")
    test_3()
    pause("물 빠질때까지 대기 → Enter")
    pause("4단계: 🖐→드리퍼잡기(✊)→뒤집기→원위치놓기(🖐)")
    test_4()
    pause("5단계: 🖐→서버잡기(✊)→따르기→서버놓기(🖐)")
    test_5()
    print("\n  ☕ 커피 완성!")


# ============================================================
# 메뉴
# ============================================================

MENU = """
╔════════════════════════════════════════════════════════╗
║              ☕ 커피 드립 테스트 메뉴                  ║
║  잡기: 옆접근 → ✊잡기 → 위로 들기 → tool -Z 후퇴    ║
║  놓기: 위접근 → 🖐놓기 → tool -Z 후퇴                ║
╠════════════════════════════════════════════════════════╣
║  [공통]                                                ║
║    h  - 홈          g - 그리퍼       t - TCP           ║
║                                                        ║
║  [1] 🖐→드리퍼(✊)→서버(🖐)                            ║
║    1a - 드리퍼 잡기       1b - 서버에 놓기             ║
║    1  - 1단계 전체                                     ║
║                                                        ║
║  [2] 🖐→컵(✊)→붓기(흔들기)→컵놓기(🖐)                 ║
║    2a - 컵 잡기   2b - 붓기   2c - 컵 놓기            ║
║    2  - 2단계 전체                                     ║
║                                                        ║
║  [3] 🖐→포트(✊)→뜸들이기→나선형드립(Circle)→포트(🖐)  ║
║    3a  - 포트 잡기                                     ║
║    3b  - 드립 전체 (뜸들이기+나선형)                   ║
║    3b1 - 드리퍼 위로 (세운 상태)                       ║
║    3b2 - 기울이기 테스트                               ║
║    3b3 - 원형 1바퀴 (Circle)                           ║
║    3c  - 포트 놓기                                     ║
║    3   - 3단계 전체                                    ║
║                                                        ║
║  [4] 🖐→드리퍼(✊)→뒤집기→원위치(🖐)                   ║
║    4a - 드리퍼 잡기  4b - 뒤집기  4c - 원위치          ║
║    4  - 4단계 전체                                     ║
║                                                        ║
║  [5] 🖐→서버(✊)→세움→기울임→따르기→세움→서버(🖐)      ║
║    5a - 서버 잡기  5b - 따르기  5c - 서버 놓기         ║
║    5  - 5단계 전체                                     ║
║                                                        ║
║  all - 전체 실행                  q - 종료             ║
╚════════════════════════════════════════════════════════╝
"""

COMMANDS = {
    'h': test_home, 'g': test_gripper, 't': test_tcp,
    '1a': test_1a, '1b': test_1b, '1': test_1,
    '2a': test_2a, '2b': test_2b, '2c': test_2c, '2': test_2,
    '3a': test_3a, '3b': test_3b, '3b1': test_3b1, '3b2': test_3b2, '3b3': test_3b3, '3c': test_3c, '3': test_3,
    '4a': test_4a, '4b': test_4b, '4c': test_4c, '4': test_4,
    '5a': test_5a, '5b': test_5b, '5c': test_5c, '5': test_5,
    'all': test_all,
}

def main():
    connect()
    if len(sys.argv) > 1:
        cmd = sys.argv[1].replace('--step', '').replace('-', '').strip()
        if cmd in COMMANDS:
            COMMANDS[cmd]()
        return

    print(MENU)
    while True:
        try:
            cmd = input("\n명령 > ").strip().lower()
            if cmd == 'q':
                home(); break
            elif cmd == 'm':
                print(MENU)
            elif cmd in COMMANDS:
                try:
                    COMMANDS[cmd]()
                except Exception as e:
                    print(f"\n  [ERROR] {e}")
                    try: robot.StopMotion()
                    except: pass
                    time.sleep(1)
                    pause("홈?")
                    home()
            else:
                print(f"  ? '{cmd}' (m=메뉴)")
        except KeyboardInterrupt:
            print("\n  [Ctrl+C]")
            try: robot.StopMotion()
            except: pass
            pause("홈?")
            home()
            break

if __name__ == "__main__":
    main()