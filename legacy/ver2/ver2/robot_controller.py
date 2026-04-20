from config import (
    ROBOT_IP, ROBOT_VEL, ROBOT_TOOL, ROBOT_USER,
    PLACE_POSITIONS, CAM_W, CAM_H
)

# FR5 실제 연결 시 주석 해제
# from fairino import Robot


class FR5Controller:
    """
    FR5 로봇팔 컨트롤러
    MOCK_MODE = True  → 터미널 출력으로 시뮬레이션
    MOCK_MODE = False → 실제 FR5 연결
    """
    MOCK_MODE = True   # FR5 연결되면 False로 변경

    def __init__(self):
        if self.MOCK_MODE:
            print("⚠️  [Mock 모드] 실제 로봇 미연결")
            self.robot = None
        else:
            print(f"🔗 FR5 연결 중... ({ROBOT_IP})")
            # self.robot = Robot.RPC(ROBOT_IP)
            print("✅ FR5 연결 완료")

    # ── 조인트 이동 ────────────────────────────────
    def move_joint(self, joint_angles: list):
        if self.MOCK_MODE:
            print(f"[Mock] move_joint({[f'{a}°' for a in joint_angles]})")
            return
        self.robot.MoveJ(
            joint_angles, tool=ROBOT_TOOL,
            user=ROBOT_USER, vel=ROBOT_VEL
        )

    # ── 직교 좌표 이동 ─────────────────────────────
    def move_to_pose(self, pose: list):
        """pose: [x, y, z, rx, ry, rz] mm/도"""
        if self.MOCK_MODE:
            print(f"[Mock] move_to_pose({pose})")
            return
        self.robot.MoveL(
            pose, tool=ROBOT_TOOL,
            user=ROBOT_USER, vel=ROBOT_VEL
        )

    # ── 3D 좌표로 이동 (D455 연동 핵심) ───────────
    def move_to_3d(self, x: float, y: float, z: float,
                   offset_z: float = 50.0):
        """
        카메라 3D 좌표 → 로봇 베이스 좌표로 변환 후 이동
        offset_z: 물체 위쪽으로 얼마나 띄울지 (mm)
        → 핸드아이 캘리브레이션 후 T_cam_to_robot 행렬로 변환 필요
        """
        if self.MOCK_MODE:
            print(f"[Mock] move_to_3d → 카메라좌표: "
                  f"({x:.0f}, {y:.0f}, {z:.0f})mm")
            print(f"         (핸드아이 캘리브레이션 후 실제 변환 적용)")
            return

        # TODO: 핸드아이 캘리브레이션 행렬 적용
        # robot_x, robot_y, robot_z = self._cam_to_robot(x, y, z)
        # pose = [robot_x, robot_y, robot_z + offset_z, 0, 180, 0]
        # self.robot.MoveL(pose, tool=ROBOT_TOOL, user=ROBOT_USER, vel=ROBOT_VEL)
        pass

    # ── 3D 파지 ────────────────────────────────────
    def grasp_at_3d(self, x: float, y: float, z: float):
        """
        3D 좌표로 이동 후 파지
        1. 물체 위 50mm 이동
        2. 물체 위치로 하강
        3. 그리퍼 닫기
        """
        if self.MOCK_MODE:
            print(f"[Mock] grasp_at_3d({x:.0f}, {y:.0f}, {z:.0f})mm")
            print(f"  1. 물체 위 이동: z+50mm")
            print(f"  2. 하강: z")
            print(f"  3. 그리퍼 닫기")
            return True

        # TODO: 핸드아이 캘리브레이션 후 실제 구현
        return True

    # ── 미세 이동 (2D Eye-in-Hand) ─────────────────
    def step_toward(self, cx: float, cy: float,
                    frame_w: int = CAM_W,
                    frame_h: int = CAM_H):
        """화면 중심 오차만큼 TCP 미세 이동"""
        dx = (cx - frame_w / 2) * 0.1
        dy = (cy - frame_h / 2) * 0.1
        if self.MOCK_MODE:
            print(f"[Mock] step_toward → "
                  f"dx:{dx:.1f}mm  dy:{dy:.1f}mm  전진:5mm")
            return
        self.robot.MoveRelL(
            [dx, dy, 5, 0, 0, 0],
            tool=ROBOT_TOOL, user=ROBOT_USER, vel=10
        )

    # ── 그리퍼 ────────────────────────────────────
    def grasp(self) -> bool:
        if self.MOCK_MODE:
            print("[Mock] grasp() — 그리퍼 닫힘")
            return True
        ret = self.robot.SetDO(1, 1)
        return ret == 0

    def release(self) -> bool:
        if self.MOCK_MODE:
            print("[Mock] release() — 그리퍼 열림")
            return True
        ret = self.robot.SetDO(1, 0)
        return ret == 0

    # ── 배치 ──────────────────────────────────────
    def place(self, location: str = "default") -> bool:
        pos = PLACE_POSITIONS.get(location, PLACE_POSITIONS["default"])
        if self.MOCK_MODE:
            print(f"[Mock] place({location}) → {pos}")
            return True
        self.move_to_pose(pos)
        return self.release()

    # ── 홈 ────────────────────────────────────────
    def go_home(self):
        home = [0, -30, 60, 0, 30, 0]
        if self.MOCK_MODE:
            print(f"[Mock] go_home() → {home}")
            return
        self.move_joint(home)