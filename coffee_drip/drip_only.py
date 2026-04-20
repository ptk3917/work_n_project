"""
3단계 드립 프로세스 단독 테스트
포트 들고 있는 상태에서 실행

사용법: python drip_only_test.py
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
SPEED_NORMAL = 20
SPEED_J_SLOW = 20
SPEED_J_NORMAL = 30
SPEED_J_FAST = 50

DRIP_CIRCLE_RADIUS = 25   # 반지름 mm (지름 1cm)
DRIP_CIRCLE_POINTS = 8    # 원 등분
DRIP_ROUNDS = 5            # 바퀴 수
DRIP_POINT_DELAY = 0.3     # 포인트 간 대기(초)

# ============================================================
# 포지션 (여기에 직접 입력)
# ============================================================

HOME_J = [111,-105.35,118,-10.35,93,-9,12]

# 홈 → DRIP_ABOVE 사이 경유점 (HOME에서 J1만 변경)
WAYPOINT_J1 = 81  # J1 값

# 드리퍼 위 (포트 세운 상태, TCP) - 물 안 나옴
DRIP_ABOVE_TCP = [18.28, -507.06, 215.31, 92.23, 9.22, -10.77]  # TODO: 여기에 입력

# 드리퍼 중심 (기울어진 상태, TCP) - 물 나옴
DRIP_CENTER_TCP = [-29.35, -507.63, 354.26, 105.44, -33.83, -15.13]  # TODO: 여기에 입력


# ============================================================
# 유틸리티
# ============================================================
robot = None

def connect():
    global robot
    robot = Robot.RPC(ROBOT_IP)
    robot.SetSpeed(SPEED_NORMAL)
    print(f"[OK] 로봇 연결됨 ({ROBOT_IP})")

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
    """홈 → J1경유 → DRIP_ABOVE_TCP 안전 이동"""
    print("  홈 경유")
    mj(HOME_J, SPEED_J_NORMAL)
    waypoint = HOME_J.copy()
    waypoint[0] = WAYPOINT_J1
    print(f"  J1→{WAYPOINT_J1} 경유")
    mj(waypoint, SPEED_J_SLOW)
    print("  DRIP_ABOVE_TCP로 이동")
    ml(DRIP_ABOVE_TCP, SPEED_SLOW)


# ============================================================
# 테스트 메뉴
# ============================================================

MENU = """
╔════════════════════════════════════════════╗
║       ☕ 드립 프로세스 테스트              ║
╠════════════════════════════════════════════╣
║  1 - DRIP_ABOVE로 이동 (홈→J1경유)       ║
║  2 - 기울이기 테스트 (↔세우기)            ║
║  3 - 원형 1바퀴 (기울어진 상태에서)       ║
║  4 - 원형 풀 ({rounds}바퀴)              ║
║  5 - 뜸들이기 + 원형 풀 (전체)           ║
║  6 - 세우기 (DRIP_ABOVE로)               ║
║                                            ║
║  h - 홈        t - TCP 읽기               ║
║  p - 현재 포즈  q - 종료                  ║
╚════════════════════════════════════════════╝
""".replace("{rounds}", str(DRIP_ROUNDS))


def test_1():
    """DRIP_ABOVE로 이동"""
    print("\n--- DRIP_ABOVE로 이동 ---")
    if not check(): return
    go_drip_position()
    print("  [OK] 도착 (포트 세운 상태)")

def test_2():
    """기울이기 ↔ 세우기 테스트"""
    print("\n--- 기울이기 테스트 ---")
    if not check(): return
    pause("기울이기 (물 나옴)")
    ml(DRIP_CENTER_TCP, SPEED_SLOW)
    pause("세우기 (물 멈춤)")
    ml(DRIP_ABOVE_TCP, SPEED_SLOW)
    print("  [OK]")

def test_3():
    """원형 1바퀴"""
    print("\n--- 원형 1바퀴 (기울어진 상태에서) ---")
    if not check(): return
    cx, cy, cz = DRIP_CENTER_TCP[0], DRIP_CENTER_TCP[1], DRIP_CENTER_TCP[2]
    rx, ry, rz = DRIP_CENTER_TCP[3], DRIP_CENTER_TCP[4], DRIP_CENTER_TCP[5]
    pause(f"원형 1바퀴 (R={DRIP_CIRCLE_RADIUS}mm)")
    for i in range(DRIP_CIRCLE_POINTS):
        angle = 2 * math.pi * i / DRIP_CIRCLE_POINTS
        x = cx + DRIP_CIRCLE_RADIUS * math.cos(angle)
        y = cy + DRIP_CIRCLE_RADIUS * math.sin(angle)
        ml([x, y, cz, rx, ry, rz], SPEED_J_NORMAL)
        time.sleep(DRIP_POINT_DELAY)
    ml(DRIP_CENTER_TCP, SPEED_SLOW)
    print("  [OK]")

def test_4():
    """원형 풀"""
    print(f"\n--- 원형 {DRIP_ROUNDS}바퀴 (기울어진 상태에서) ---")
    if not check(): return
    cx, cy, cz = DRIP_CENTER_TCP[0], DRIP_CENTER_TCP[1], DRIP_CENTER_TCP[2]
    rx, ry, rz = DRIP_CENTER_TCP[3], DRIP_CENTER_TCP[4], DRIP_CENTER_TCP[5]
    pause(f"원형 {DRIP_ROUNDS}바퀴 시작")
    for r in range(DRIP_ROUNDS):
        print(f"  라운드 {r+1}/{DRIP_ROUNDS}")
        for i in range(DRIP_CIRCLE_POINTS):
            angle = 2 * math.pi * i / DRIP_CIRCLE_POINTS
            x = cx + DRIP_CIRCLE_RADIUS * math.cos(angle)
            y = cy + DRIP_CIRCLE_RADIUS * math.sin(angle)
            ml([x, y, cz, rx, ry, rz], SPEED_SLOW)
            time.sleep(DRIP_POINT_DELAY)
    ml(DRIP_CENTER_TCP, SPEED_SLOW)
    print("  [OK]")

def test_5():
    """전체: 이동 → 뜸들이기 → 원형 풀 → 세우기"""
    print("\n--- 드립 전체 프로세스 ---")
    if not check(): return

    go_drip_position()

    pause("기울이기 (뜸들이기 시작)")
    ml(DRIP_CENTER_TCP, SPEED_SLOW)
    time.sleep(2.0)
    ml(DRIP_ABOVE_TCP, SPEED_NORMAL)
    print("  뜸들이기 대기 10초...")
    time.sleep(10.0)

    pause("다시 기울이기 (원형 드립 시작)")
    ml(DRIP_CENTER_TCP, SPEED_SLOW)

    cx, cy, cz = DRIP_CENTER_TCP[0], DRIP_CENTER_TCP[1], DRIP_CENTER_TCP[2]
    rx, ry, rz = DRIP_CENTER_TCP[3], DRIP_CENTER_TCP[4], DRIP_CENTER_TCP[5]
    for r in range(DRIP_ROUNDS):
        print(f"  라운드 {r+1}/{DRIP_ROUNDS}")
        for i in range(DRIP_CIRCLE_POINTS):
            angle = 2 * math.pi * i / DRIP_CIRCLE_POINTS
            x = cx + DRIP_CIRCLE_RADIUS * math.cos(angle)
            y = cy + DRIP_CIRCLE_RADIUS * math.sin(angle)
            ml([x, y, cz, rx, ry, rz], SPEED_SLOW)
            time.sleep(DRIP_POINT_DELAY)
    ml(DRIP_CENTER_TCP, SPEED_SLOW)

    print("  세우기 (물 멈춤)")
    ml(DRIP_ABOVE_TCP, SPEED_SLOW)
    print("  [OK] 드립 완료")

def test_6():
    """세우기"""
    print("\n--- 세우기 ---")
    if not check(): return
    ml(DRIP_ABOVE_TCP, SPEED_SLOW)
    print("  [OK]")


COMMANDS = {
    '1': test_1,
    '2': test_2,
    '3': test_3,
    '4': test_4,
    '5': test_5,
    '6': test_6,
}

def main():
    connect()
    print(MENU)

    while True:
        try:
            cmd = input("\n명령 > ").strip().lower()
            if cmd == 'q':
                mj(HOME_J, SPEED_J_FAST)
                break
            elif cmd == 'h':
                print("  → 홈")
                mj(HOME_J, SPEED_J_FAST)
            elif cmd == 't':
                tcp = get_tcp()
                if tcp:
                    print(f"  TCP = [{', '.join(f'{v:.2f}' for v in tcp)}]")
            elif cmd == 'p':
                tcp = get_tcp()
                ret = robot.GetActualJointPosDegree(0)
                joint = ret[1] if ret[0] == 0 else None
                if tcp:
                    print(f"  TCP:   [{', '.join(f'{v:.2f}' for v in tcp)}]")
                if joint:
                    print(f"  Joint: [{', '.join(f'{v:.2f}' for v in joint)}]")
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