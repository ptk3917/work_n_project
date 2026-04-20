"""
FR5 커피 드립 위치 티칭 유틸리티
- 스페이스바: 그리퍼 열기/닫기
- 슬롯 번호: 현재 위치 + 그리퍼 상태 저장
- g번호: 이미 저장된 슬롯의 그리퍼 상태만 변경
- c: 코드 출력

실행: python robot_teach_drip.py
"""
import os
import sys
import time
import json
import tty
import termios
from fairino import Robot

ROBOT_IP = "192.168.58.2"
TOOL_NO = 1
SAVE_FILE = "drip_positions.json"

GRIPPER_OPEN = 100
GRIPPER_CLOSE = 0

# ══════════════════════════════════════════════════════
# 키 입력
# ══════════════════════════════════════════════════════

def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

def input_normal(prompt=""):
    return input(prompt).strip()

# ══════════════════════════════════════════════════════
# 슬롯 정의: (키, 변수명, 좌표타입, 설명)
# ══════════════════════════════════════════════════════

SLOTS = [
    ("home",                "HOME_J",                "J",   "홈 위치"),

    ("dripper_approach",    "DRIPPER_APPROACH_J",    "J",   "드리퍼 옆 접근"),
    ("dripper_pick",        "DRIPPER_PICK_J",        "J",   "드리퍼 잡는 위치"),
    ("server_above",        "SERVER_ABOVE_J",        "J",   "서버 위 접근"),
    ("server_place",        "SERVER_PLACE_J",        "J",   "서버에 놓는 위치"),

    ("cup_approach",        "CUP_APPROACH_J",        "J",   "커피 컵 옆 접근"),
    ("cup_pick",            "CUP_PICK_J",            "J",   "커피 컵 잡는 위치"),
    ("pour_above",          "POUR_ABOVE_J",          "J",   "드리퍼 위 (붓기 전, 컵 세운 상태)"),
    ("pour",                "POUR_J",                "J",   "붓는 자세 (기울어짐)"),
    ("cup_return_above",    "CUP_RETURN_ABOVE_J",    "J",   "빈 컵 놓기 위"),
    ("cup_return",          "CUP_RETURN_J",          "J",   "빈 컵 놓는 위치"),

    ("pot_approach",        "POT_APPROACH_J",        "J",   "드립포트 옆 접근"),
    ("pot_pick",            "POT_PICK_J",            "J",   "드립포트 잡는 위치"),
    ("drip_above",          "DRIP_ABOVE_TCP",        "TCP", "드리퍼 위 (포트 세운 상태, TCP)"),
    ("drip_center",         "DRIP_CENTER_TCP",       "TCP", "드리퍼 중심 (기울어진, TCP)"),
    ("pot_return_above",    "POT_RETURN_ABOVE_J",    "J",   "포트 놓기 위"),
    ("pot_return",          "POT_RETURN_J",          "J",   "포트 놓는 위치"),

    ("svr_drip_approach",   "SVR_DRIP_APPROACH_J",   "J",   "서버 위 드리퍼 옆 접근"),
    ("svr_drip_pick",       "SVR_DRIP_PICK_J",       "J",   "서버 위 드리퍼 잡기"),
    ("trash_above",         "TRASH_ABOVE_J",         "J",   "쓰레기통 위"),
    ("trash_flip",          "TRASH_FLIP_J",          "J",   "뒤집는 자세"),
    ("drip_back_above",     "DRIP_BACK_ABOVE_J",     "J",   "드리퍼 원위치 위"),
    ("drip_back_place",     "DRIP_BACK_PLACE_J",     "J",   "드리퍼 원위치"),

    ("server_approach",     "SERVER_APPROACH_J",     "J",   "서버 옆 접근"),
    ("server_pick",         "SERVER_PICK_J",         "J",   "서버 잡는 위치"),
    ("cup_pour_above",      "CUP_POUR_ABOVE_J",      "J",   "컵 위 (따르기 전, 서버 세운 상태)"),
    ("cup_pour",            "CUP_POUR_J",            "J",   "컵에 따르는 자세 (기울어짐)"),
    ("svr_return_above",    "SVR_RETURN_ABOVE_J",    "J",   "서버 놓기 위"),
    ("svr_return",          "SVR_RETURN_J",          "J",   "서버 원위치"),
]

GROUPS = [
    ("공통",                      [0]),
    ("1단계: 드리퍼→서버",         [1, 2, 3, 4]),
    ("2단계: 커피가루 붓기",       [5, 6, 7, 8, 9, 10]),
    ("3단계: 드립",                [11, 12, 13, 14, 15, 16]),
    ("4단계: 찌꺼기 버리기",       [17, 18, 19, 20, 21, 22]),
    ("5단계: 서빙",                [23, 24, 25, 26, 27, 28]),
]

# ══════════════════════════════════════════════════════
# 초기화
# ══════════════════════════════════════════════════════
print("=" * 55)
print("  ☕ FR5 커피 드립 티칭 유틸리티")
print("=" * 55)

print(f"FR5 연결 중... ({ROBOT_IP})")
robot = Robot.RPC(ROBOT_IP)
try:
    result = robot.GetRobotErrorCode()
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        if result[1] != 0:
            robot.ResetAllError()
            time.sleep(0.5)
except:
    pass
print("  FR5 OK")

print("  그리퍼 초기화 중...")
robot.ActGripper(1, 1)
time.sleep(2.0)
robot.MoveGripper(1, GRIPPER_OPEN, 50, 10, 10000, 0, 0, 0, 0, 0)
time.sleep(1.0)
print("  그리퍼 OK (열림)\n")

gripper_closed = False

# ══════════════════════════════════════════════════════
# 저장 관리
# ══════════════════════════════════════════════════════
saved = {}

def load_saved():
    global saved
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r") as f:
            saved = json.load(f)
        print(f"  기존 데이터 로드: {SAVE_FILE} ({len(saved)}개)")

def save_to_file():
    with open(SAVE_FILE, "w") as f:
        json.dump(saved, f, indent=2, ensure_ascii=False)

load_saved()

# ══════════════════════════════════════════════════════
# 로봇 유틸
# ══════════════════════════════════════════════════════

def get_tcp():
    time.sleep(0.3)
    poses = []
    for _ in range(5):
        ret = robot.GetActualTCPPose(0)
        if ret[0] == 0:
            poses.append(ret[1])
        time.sleep(0.05)
    if not poses:
        return None
    return [round(sum(x)/len(x), 2) for x in zip(*poses)]

def get_joint():
    time.sleep(0.3)
    poses = []
    for _ in range(5):
        ret = robot.GetActualJointPosDegree(0)
        if ret[0] == 0:
            poses.append(ret[1])
        time.sleep(0.05)
    if not poses:
        return None
    return [round(sum(x)/len(x), 2) for x in zip(*poses)]

def toggle_gripper():
    global gripper_closed
    if gripper_closed:
        robot.MoveGripper(1, GRIPPER_OPEN, 50, 10, 10000, 0, 0, 0, 0, 0)
        gripper_closed = False
        print("  🖐  그리퍼 열림")
    else:
        robot.MoveGripper(1, GRIPPER_CLOSE, 50, 50, 10000, 0, 0, 0, 0, 0)
        gripper_closed = True
        print("  ✊ 그리퍼 닫힘")
    time.sleep(0.8)


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
    print("  [WARN] timeout!")
    return False


def go_slot(idx):
    """저장된 슬롯 위치로 로봇 이동"""
    if idx < 0 or idx >= len(SLOTS):
        print(f"  ⚠ 잘못된 번호: {idx}")
        return
    key, var_name, coord_type, desc = SLOTS[idx]
    if key not in saved:
        print(f"  ⚠ [{idx}] {var_name} 아직 저장 안됨!")
        return

    data = saved[key]
    gi = grip_icon(key)
    print(f"  → [{idx}] {var_name} ({desc}) 로 이동 중...")

    robot.SetSpeed(20)
    if coord_type == "TCP":
        tcp = data["tcp"]
        ret = robot.MoveL(tcp, TOOL_NO, 0, [0,0,0,0,0,0])
        print(f"    MoveL ret={ret}")
    else:
        joint = data["joint"]
        ret = robot.MoveJ(joint, TOOL_NO, 0, [0,0,0,0,0,0])
        print(f"    MoveJ ret={ret}")
    wait_done()

    print(f"  ✅ 도착! 그리퍼: {gi} {data.get('grip', '미설정')}")

# ══════════════════════════════════════════════════════
# 출력
# ══════════════════════════════════════════════════════

def grip_icon(key):
    """저장된 슬롯의 그리퍼 아이콘"""
    if key not in saved:
        return "  "
    g = saved[key].get("grip")
    if g == "close":
        return "✊"
    elif g == "open":
        return "🖐"
    else:
        return "—"  # 미설정


def print_menu():
    print("\n" + "=" * 70)
    print("  ☕ 슬롯 목록")
    print("  잡기: APPROACH(옆) → PICK(✊잡기) → 위로 들기")
    print("  놓기: ABOVE(위) → PLACE(🖐놓기) → 위로 빠지기")
    print("=" * 70)
    for group_name, indices in GROUPS:
        print(f"\n  [{group_name}]")
        for idx in indices:
            key, var_name, coord_type, desc = SLOTS[idx]
            pos_ok = "✓" if key in saved else " "
            gi = grip_icon(key)
            print(f"    {idx:2d}. {pos_ok} {gi} {var_name:<26s} [{coord_type}]  {desc}")

    done = sum(1 for key, _, _, _ in SLOTS if key in saved)
    print(f"\n  저장: {done}/{len(SLOTS)}")
    print()
    print("  ─── 바로 눌러서 동작 ───")
    print("  SPACE = 그리퍼 열기/닫기")
    print("  p     = 현재 포즈 출력")
    print("  r     = 에러 리셋")
    print()
    print("  ─── 번호 입력 (Enter) ───")
    print("  번호     = 위치 + 현재 그리퍼 상태 저장")
    print("  v번호    = 저장된 위치로 이동")
    print("  g번호    = 그리퍼 상태만 변경 (close↔open↔미설정)")
    print("  c = 코드출력  s = 현황  m = 메뉴  q = 종료")
    print("=" * 70)


def print_saved_status():
    total = len(SLOTS)
    done = sum(1 for key, _, _, _ in SLOTS if key in saved)
    print(f"\n  저장 현황: {done}/{total}")
    for group_name, indices in GROUPS:
        print(f"\n  [{group_name}]")
        for i in indices:
            key, var_name, _, desc = SLOTS[i]
            if key in saved:
                gi = grip_icon(key)
                g = saved[key].get("grip", "미설정")
                print(f"    ✅ {var_name:<26s} 그리퍼: {gi} {g}")
            else:
                print(f"    ⬜ {var_name:<26s} 미티칭")

    no_grip = []
    for key, var_name, _, _ in SLOTS:
        if key in saved and saved[key].get("grip") is None:
            no_grip.append(var_name)
    if no_grip:
        print(f"\n  ⚠ 그리퍼 미설정: {', '.join(no_grip)}")
        print(f"    → g번호 로 설정 가능")


def print_code():
    print("\n" + "#" * 60)
    print("# coffee_drip_test.py 포지션 (복사해서 붙여넣기)")
    print("#" * 60)
    for group_name, indices in GROUPS:
        print(f"\n# --- {group_name} ---")
        for idx in indices:
            key, var_name, coord_type, desc = SLOTS[idx]
            if key in saved:
                data = saved[key]
                vals = data["joint"] if coord_type == "J" else data["tcp"]
                vals_str = "[" + ", ".join(f"{v}" for v in vals) + "]"
                pad = max(0, 26 - len(var_name))
                g = data.get("grip")
                grip_comment = ""
                if g == "close":
                    grip_comment = "  # ✊ 잡기"
                elif g == "open":
                    grip_comment = "  # 🖐 놓기"
                print(f"{var_name}{' ' * pad} = {vals_str}{grip_comment}")
            else:
                pad = max(0, 26 - len(var_name))
                print(f"{var_name}{' ' * pad} = [0, 0, 0, 0, 0, 0]  # TODO")

    # 그리퍼 동작 맵 출력
    print(f"\n# --- 그리퍼 동작 맵 ---")
    print("GRIP_ACTIONS = {")
    for idx in range(len(SLOTS)):
        key, var_name, _, _ = SLOTS[idx]
        if key in saved:
            g = saved[key].get("grip")
            if g is not None:
                pad = max(0, 26 - len(var_name))
                print(f'    "{var_name}"{" " * pad}: "{g}",')
    print("}")

    print("\n" + "#" * 60)


def save_slot(idx):
    """위치 + 현재 그리퍼 상태 저장"""
    if idx < 0 or idx >= len(SLOTS):
        print(f"  ⚠ 잘못된 번호: {idx} (0~{len(SLOTS)-1})")
        return
    key, var_name, coord_type, desc = SLOTS[idx]
    tcp = get_tcp()
    joint = get_joint()
    if tcp is None or joint is None:
        print(f"  ⚠ 포즈 읽기 실패!")
        return

    grip_state = "close" if gripper_closed else "open"

    saved[key] = {
        "joint": joint,
        "tcp": tcp,
        "type": coord_type,
        "desc": desc,
        "grip": grip_state,
    }
    save_to_file()

    gi = "✊" if gripper_closed else "🖐"
    print(f"  ✅ [{idx}] {var_name} 저장!  그리퍼: {gi} {grip_state}")
    print(f"     Joint = [{', '.join(f'{v:.2f}' for v in joint)}]")
    print(f"     TCP   = [{', '.join(f'{v:.1f}' for v in tcp)}]")


def toggle_slot_grip(idx):
    """이미 저장된 슬롯의 그리퍼 상태만 변경 (close → open → 미설정 → close)"""
    if idx < 0 or idx >= len(SLOTS):
        print(f"  ⚠ 잘못된 번호: {idx}")
        return
    key, var_name, _, _ = SLOTS[idx]
    if key not in saved:
        print(f"  ⚠ [{idx}] {var_name} 아직 위치가 저장 안됨!")
        return

    current = saved[key].get("grip")
    if current == "close":
        saved[key]["grip"] = "open"
        print(f"  [{idx}] {var_name} 그리퍼: ✊close → 🖐open")
    elif current == "open":
        saved[key]["grip"] = None
        print(f"  [{idx}] {var_name} 그리퍼: 🖐open → — 미설정(변화없음)")
    else:
        saved[key]["grip"] = "close"
        print(f"  [{idx}] {var_name} 그리퍼: — 미설정 → ✊close")
    save_to_file()


# ══════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════

def main():
    print_menu()
    grip_str = lambda: "✊닫힘" if gripper_closed else "🖐열림"
    print(f"\n  그리퍼: {grip_str()}")
    print(f"  대기 중... (SPACE=그리퍼, 숫자=저장, o숫자=이동, g숫자=그리퍼변경)\n")

    while True:
        try:
            ch = getch()

            if ch == ' ':
                toggle_gripper()
                print(f"  대기 중... [{grip_str()}]")

            elif ch == 'p':
                tcp = get_tcp()
                joint = get_joint()
                if tcp and joint:
                    print(f"\n  TCP:   [{', '.join(f'{v:.2f}' for v in tcp)}]")
                    print(f"  Joint: [{', '.join(f'{v:.2f}' for v in joint)}]")
                else:
                    print("  ⚠ 읽기 실패")

            elif ch == 'r':
                robot.ResetAllError()
                time.sleep(0.3)
                print("  ✅ 에러 리셋")

            elif ch == 'm':
                print_menu()

            elif ch == 'q':
                print("\n\n최종 저장 현황:")
                print_saved_status()
                print("종료")
                break

            elif ch == 'g':
                # g + 번호: 그리퍼 상태만 변경
                print(f"\n  그리퍼 변경 슬롯: g", end="", flush=True)
                rest = input_normal("")
                if rest.isdigit():
                    toggle_slot_grip(int(rest))
                else:
                    print(f"  ⚠ 'g{rest}'?  g번호 형식으로 입력")
                print(f"  대기 중... [{grip_str()}]")

            elif ch == 'v':
                # v + 번호: 저장된 위치로 이동
                print(f"\n  이동할 슬롯: v", end="", flush=True)
                rest = input_normal("")
                if rest.isdigit():
                    go_slot(int(rest))
                else:
                    print(f"  ⚠ 'v{rest}'?  v번호 형식으로 입력")
                print(f"  대기 중... [{grip_str()}]")

            elif ch.isdigit():
                print(f"\n  슬롯 번호: {ch}", end="", flush=True)
                rest = input_normal("")
                num_str = ch + rest
                if num_str.isdigit():
                    save_slot(int(num_str))
                else:
                    print(f"  ⚠ '{num_str}'?")
                print(f"  대기 중... [{grip_str()}]")

            elif ch == 'c':
                print_code()

            elif ch == 's':
                print_saved_status()

            elif ch == '\x03':
                print("\n\n종료")
                break

        except KeyboardInterrupt:
            print("\n\n종료")
            break

if __name__ == "__main__":
    main()