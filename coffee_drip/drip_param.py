"""
3단계 드립 프로세스 단독 테스트
Circle 명령으로 부드러운 원형 드립 + 나선형(안쪽으로 좁혀짐)

사용법: python drip_only_test.py
"""

import time
import math
import sys
from fairino import Robot

# ============================================================
# 설정
# ============================================================


ROBOT_IP = "192.168.58.2"
TOOL_NO = 1

SPEED_SLOW = 10
SPEED_NORMAL = 20
SPEED_J_SLOW = 20
SPEED_J_NORMAL = 30
SPEED_J_FAST = 50

# 드립 파라미터
DRIP_RADIUS = 20.0      # 원 반지름 (mm)
DRIP_ROUNDS = 5         # 바퀴 수
DRIP_VEL = 20.0         # Circle 속도
SPIRAL_SHRINK = 3.0     # 나선형: 매 바퀴 반지름 줄이는 양 (mm)

# ============================================================
# 포지션
# ============================================================


HOME_J = [111,-105.35,118,-10.35,93,-9,12]

# 홈 → DRIP_ABOVE 사이 경유점 (HOME에서 J1만 변경)
WAYPOINT_J1 = 81  # J1 값

# 드리퍼 위 (포트 세운 상태, TCP) - 물 안 나옴
DRIP_ABOVE_TCP = [18.28, -507.06, 215.31, 92.23, 9.22, -10.77]  # TODO: 여기에 입력

# 드리퍼 중심 (기울어진 상태, TCP) - 물 나옴
DRIP_CENTER_TCP = [-38.18, -507.04, 354.25, 105.44, -33.83, -16.13]  # TODO: 여기에 입력


# ============================================================
# 유틸리티
# ============================================================
robot = None

def connect():
    global robot
    robot = Robot.RPC(ROBOT_IP)
    robot.SetSpeed(SPEED_NORMAL)
    print(f"[OK] 로봇 연결됨 ({ROBOT_IP})")

def wait_done(pre_delay=0.4, timeout=120):
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

def pause(msg="Enter..."):
    input(f"\n  ⏸  {msg}")
    print()

def check():
    if DRIP_ABOVE_TCP == [0,0,0,0,0,0]:
        print("  [!] DRIP_ABOVE_TCP 미입력!")
        return False
    if DRIP_CENTER_TCP == [0,0,0,0,0,0]:
        print("  [!] DRIP_CENTER_TCP 미입력!")
        return False
    return True

def go_drip_position():
    """홈 → J1경유 → DRIP_ABOVE_TCP"""
    print("  홈 경유")
    mj(HOME_J, SPEED_J_NORMAL)
    waypoint = HOME_J.copy()
    waypoint[0] = WAYPOINT_J1
    print(f"  J1→{WAYPOINT_J1} 경유")
    mj(waypoint, SPEED_J_SLOW)
    print("  DRIP_ABOVE_TCP로 이동")
    ml(DRIP_ABOVE_TCP, SPEED_SLOW)


# ============================================================
# Circle 기반 원형 드립
# ============================================================

def circle_points(cx, cy, cz, rx, ry, rz, radius, angle_deg):
    """중심에서 angle 방향으로 radius만큼 떨어진 TCP 좌표"""
    rad = math.radians(angle_deg)
    x = cx + radius * math.cos(rad)
    y = cy + radius * math.sin(rad)
    return [x, y, cz, rx, ry, rz]


def do_circle(radius, vel=DRIP_VEL):
    """
    DRIP_CENTER_TCP 기준 한 바퀴 (Circle = 전체 원)
    3점: 현재(0°), 경유(90°), 끝(180°) → 전체 원
    """
    cx, cy, cz = DRIP_CENTER_TCP[0], DRIP_CENTER_TCP[1], DRIP_CENTER_TCP[2]
    rx, ry, rz = DRIP_CENTER_TCP[3], DRIP_CENTER_TCP[4], DRIP_CENTER_TCP[5]

    def pt(angle):
        rad = math.radians(angle)
        return [cx + radius * math.cos(rad), cy + radius * math.sin(rad), cz, rx, ry, rz]

    # 0도 시작점으로 이동
    ml(pt(0), SPEED_SLOW)

    # Circle: 현재(0°) → 90°(경유) → 180°(끝) → 전체 원
    ret = robot.Circle(
        desc_pos_p=pt(90), tool_p=TOOL_NO, user_p=0,
        desc_pos_t=pt(180), tool_t=TOOL_NO, user_t=0,
        vel_p=vel, vel_t=vel,
    )
    print(f"    Circle ret={ret} (R={radius:.1f}mm)")
    if ret == 0:
        wait_done(pre_delay=1.0, timeout=60)
    return ret


def do_drip_circles(rounds=DRIP_ROUNDS, radius=DRIP_RADIUS, vel=DRIP_VEL):
    """같은 반지름으로 여러 바퀴"""
    for r in range(rounds):
        print(f"  라운드 {r+1}/{rounds} (R={radius:.1f}mm)")
        do_circle(radius, vel)
    # 중심 복귀
    ml(DRIP_CENTER_TCP, SPEED_SLOW)


def do_spiral_drip(start_radius=DRIP_RADIUS, shrink=SPIRAL_SHRINK, vel=DRIP_VEL):
    """나선형: 바깥에서 안쪽으로 반지름 줄여가며 원형"""
    radius = start_radius
    round_num = 1
    while radius > 0.5:  # 0.5mm 이하면 멈춤
        print(f"  라운드 {round_num} (R={radius:.1f}mm)")
        do_circle(radius, vel)
        radius -= shrink
        round_num += 1
    # 중심 복귀
    print("  중심으로 복귀")
    ml(DRIP_CENTER_TCP, SPEED_SLOW)


# ============================================================
# 테스트 메뉴
# ============================================================

MENU = f"""
╔══════════════════════════════════════════════════╗
║       ☕ 드립 프로세스 테스트 (Circle)           ║
╠══════════════════════════════════════════════════╣
║  1 - DRIP_ABOVE로 이동 (홈→J1경유)             ║
║  2 - 기울이기 테스트 (↔세우기)                  ║
║  3 - 원형 1바퀴 (R={DRIP_RADIUS}mm)                      ║
║  4 - 원형 {DRIP_ROUNDS}바퀴 (같은 반지름)                  ║
║  5 - 나선형 (바깥→안, 매바퀴 -{SPIRAL_SHRINK}mm)           ║
║  6 - 전체 (이동→뜸들이기→나선형→세우기)        ║
║  7 - 세우기 (DRIP_ABOVE로)                     ║
║                                                  ║
║  h - 홈        t - TCP 읽기                     ║
║  p - 현재 포즈  q - 종료                        ║
╚══════════════════════════════════════════════════╝
"""


def test_1():
    print("\n--- DRIP_ABOVE로 이동 ---")
    if not check(): return
    go_drip_position()
    print("  [OK]")

def test_2():
    print("\n--- 기울이기 테스트 ---")
    if not check(): return
    pause("기울이기")
    ml(DRIP_CENTER_TCP, SPEED_SLOW)
    pause("세우기")
    ml(DRIP_ABOVE_TCP, SPEED_SLOW)

def test_3():
    print(f"\n--- 원형 1바퀴 (R={DRIP_RADIUS}mm) ---")
    if not check(): return
    pause("1바퀴")
    do_circle(DRIP_RADIUS)
    ml(DRIP_CENTER_TCP, SPEED_SLOW)
    print("  [OK]")

def test_4():
    print(f"\n--- 원형 {DRIP_ROUNDS}바퀴 ---")
    if not check(): return
    pause(f"{DRIP_ROUNDS}바퀴 시작")
    do_drip_circles()
    print("  [OK]")

def test_5():
    print(f"\n--- 나선형 (R={DRIP_RADIUS}→0, 매바퀴 -{SPIRAL_SHRINK}mm) ---")
    if not check(): return
    pause("나선형 시작")
    do_spiral_drip()
    print("  [OK]")

def test_6():
    print("\n--- 드립 전체 프로세스 ---")
    if not check(): return

    go_drip_position()

    pause("기울이기 (뜸들이기)")
    ml(DRIP_CENTER_TCP, SPEED_SLOW)
    time.sleep(2.0)
    ml(DRIP_ABOVE_TCP, SPEED_NORMAL)
    print("  뜸들이기 10초...")
    time.sleep(10.0)

    pause("다시 기울이기 (나선형 드립)")
    ml(DRIP_CENTER_TCP, SPEED_SLOW)

    do_spiral_drip()

    print("  세우기")
    ml(DRIP_ABOVE_TCP, SPEED_SLOW)
    print("  [OK] 드립 완료")

def test_7():
    print("\n--- 세우기 ---")
    if not check(): return
    ml(DRIP_ABOVE_TCP, SPEED_SLOW)


COMMANDS = {
    '1': test_1, '2': test_2, '3': test_3,
    '4': test_4, '5': test_5, '6': test_6, '7': test_7,
}

def main():
    connect()
    print(MENU)

    while True:
        try:
            cmd = input("\n명령 > ").strip().lower()
            if cmd == 'q':
                mj(HOME_J, SPEED_J_FAST); break
            elif cmd == 'h':
                mj(HOME_J, SPEED_J_FAST)
            elif cmd == 't':
                tcp = get_tcp()
                if tcp:
                    print(f"  TCP = [{', '.join(f'{v:.2f}' for v in tcp)}]")
            elif cmd == 'p':
                tcp = get_tcp()
                ret = robot.GetActualJointPosDegree(0)
                joint = ret[1] if ret[0] == 0 else None
                if tcp: print(f"  TCP:   [{', '.join(f'{v:.2f}' for v in tcp)}]")
                if joint: print(f"  Joint: [{', '.join(f'{v:.2f}' for v in joint)}]")
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
                    mj(HOME_J, SPEED_J_FAST)
            else:
                print(f"  ? '{cmd}' (m=메뉴)")
        except KeyboardInterrupt:
            print("\n  [Ctrl+C]")
            try: robot.StopMotion()
            except: pass
            pause("홈?")
            mj(HOME_J, SPEED_J_FAST)
            break

if __name__ == "__main__":
    main()