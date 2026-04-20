from config import (
    ROBOT_IP, ROBOT_VEL, ROBOT_TOOL, ROBOT_USER,
    PLACE_POSITIONS, CAM_W, CAM_H
)

# ── FR5 실제 연결 시 아래 주석 해제 ───────────────
# from fairino import Robot

class FR5Controller:
    """
    FR5 로봇팔 컨트롤러
    - MOCK_MODE = True  → 실제 로봇 없이 테스트 가능
    - MOCK_MODE = False → 실제 FR5에 연결
    """
    MOCK_MODE = True   # ← FR5 연결되면 False로 변경

    def __init__(self):
        if self.MOCK_MODE:
            print("⚠️  [Mock 모드] 실제 로봇 미연결 — 동작을 터미널에 출력합니다")
            self.robot = None
        else:
            print(f"🔗 FR5 연결 중... ({ROBOT_IP})")
            # self.robot = Robot.RPC(ROBOT_IP)
            # ret = self.robot.GetRobotState()
            # if ret[0] != 0:
            #     raise ConnectionError(f"FR5 연결 실패: {ret}")
            print("✅ FR5 연결 완료")

    # ── 조인트 이동 ────────────────────────────────
    def move_joint(self, joint_angles: list):
        """
        조인트 각도로 이동
        joint_angles: [j1, j2, j3, j4, j5, j6] (도 단위)
        """
        if self.MOCK_MODE:
            print(f"[Mock] move_joint({[f'{a}°' for a in joint_angles]})")
            return
        self.robot.MoveJ(
            joint_angles,
            tool=ROBOT_TOOL,
            user=ROBOT_USER,
            vel=ROBOT_VEL
        )

    # ── 직교 좌표 이동 ─────────────────────────────
    def move_to_pose(self, pose: list):
        """
        직교 좌표로 이동
        pose: [x, y, z, rx, ry, rz] (mm, 도 단위)
        """
        if self.MOCK_MODE:
            print(f"[Mock] move_to_pose({pose})")
            return
        self.robot.MoveL(
            pose,
            tool=ROBOT_TOOL,
            user=ROBOT_USER,
            vel=ROBOT_VEL
        )

    # ── 미세 이동 (Eye-in-Hand 접근) ──────────────
    def step_toward(self, cx: float, cy: float,
                    frame_w: int = CAM_W,
                    frame_h: int = CAM_H):
        """
        화면 중심과 물체 중심의 오차만큼 TCP 미세 이동
        cx, cy: 물체 중심 픽셀 좌표
        """
        dx = (cx - frame_w / 2) * 0.1   # 픽셀 → mm 스케일 (실측 후 조정)
        dy = (cy - frame_h / 2) * 0.1
        if self.MOCK_MODE:
            print(f"[Mock] step_toward → dx:{dx:.1f}mm  dy:{dy:.1f}mm  전진:5mm")
            return
        self.robot.MoveRelL(
            [dx, dy, 5, 0, 0, 0],       # 5mm 전진 + 중심 보정
            tool=ROBOT_TOOL,
            user=ROBOT_USER,
            vel=10
        )

    # ── 그리퍼 ────────────────────────────────────
    def grasp(self) -> bool:
        """그리퍼 닫기 (물체 잡기)"""
        if self.MOCK_MODE:
            print("[Mock] grasp() — 그리퍼 닫힘")
            return True
        ret = self.robot.SetDO(1, 1)    # DO 포트 번호는 실제 배선 확인 필요
        return ret == 0

    def release(self) -> bool:
        """그리퍼 열기 (물체 놓기)"""
        if self.MOCK_MODE:
            print("[Mock] release() — 그리퍼 열림")
            return True
        ret = self.robot.SetDO(1, 0)
        return ret == 0

    # ── 놓기 (place 위치로 이동 후 release) ────────
    def place(self, location: str = "default") -> bool:
        pos = PLACE_POSITIONS.get(location, PLACE_POSITIONS["default"])
        if self.MOCK_MODE:
            print(f"[Mock] place({location}) → 위치: {pos}")
            return True
        self.move_to_pose(pos)
        return self.release()

    # ── 홈 복귀 ───────────────────────────────────
    def go_home(self):
        """로봇 홈 위치로 복귀"""
        home = [0, -30, 60, 0, 30, 0]
        if self.MOCK_MODE:
            print(f"[Mock] go_home() → {home}")
            return
        self.move_joint(home)
