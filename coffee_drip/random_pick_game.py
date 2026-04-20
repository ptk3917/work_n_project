"""
작업 영역 검증 스크립트 (Workspace Validation)
==============================================
FR5 로봇 + RealSense D455

사용법:
  1) 티칭: 판의 4꼭짓점 저장 + Z 고정 높이 설정
  2) 검증: Enter → 판 중앙(Z+150) 이동 → 랜덤 포인트(Z고정) 이동

키 조작:
  [1~4]  꼭짓점 1~4 저장
  [z]    Z 높이 저장 (현재 TCP Z)
  [Enter] 랜덤 포인트 이동
  [h]    HOME 복귀
  [s]    설정 저장 (JSON)
  [l]    설정 불러오기 (JSON)
  [q]    종료
"""

import time
import random
import json
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import pyrealsense2 as rs
from fairino import Robot

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
ROBOT_IP = "192.168.58.3"
TOOL_NO = 1

HOME_J = [10.0, -98.0, 100.0, -94.0, -84.0, -111.0]
SPEED = 20
SPEED_J = 30
HOVER_HEIGHT = 150   # 랜덤 이동 전 판 중앙 위 높이 (mm)

SAVE_FILE = "workspace_config.json"

# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────
def put_korean_text(img, text, pos, font_size=20, color=(0, 255, 0)):
    """OpenCV 이미지에 한글 텍스트 렌더링"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def get_stable_tcp(robot, n=5, delay=0.3):
    """TCP 포즈 평균"""
    poses = []
    for _ in range(n):
        ret, pose = robot.GetActualTCPPose(0)
        if ret == 0 and pose:
            poses.append(pose)
        time.sleep(delay)
    if not poses:
        return None
    return [sum(col) / len(col) for col in zip(*poses)]


def wait_done(robot, timeout=10):
    """동작 완료 대기"""
    time.sleep(0.4)
    t0 = time.time()
    while time.time() - t0 < timeout:
        ret, state = robot.GetRobotMotionDone()
        if ret == 0 and state == 1:
            return True
        time.sleep(0.1)
    print("[WARN] wait_done timeout")
    return False


def sort_corners_cw(corners):
    """4개 꼭짓점을 시계방향 정렬 (bilinear interpolation용)"""
    pts = np.array([[c[0], c[1]] for c in corners])
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    return [corners[i] for i in order]


class WorkspaceValidator:
    def __init__(self):
        # 로봇 초기화
        self.robot = Robot.RPC(ROBOT_IP)
        self.robot.SetSpeed(SPEED)

        # 그리퍼 초기화 & 닫기
        self.robot.ActGripper(1, 0)
        time.sleep(1)
        self.robot.ActGripper(1, 1)
        time.sleep(1)
        self.robot.MoveGripper(1, 0, 50, 30, 5000, 0, 0, 0, 0, 0)
        time.sleep(1)

        # RealSense 초기화
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        for _ in range(30):
            self.pipeline.wait_for_frames()

        # 작업 영역 — 4 꼭짓점
        self.corners = [None, None, None, None]  # 각각 [x,y,z,rx,ry,rz]
        self.z_fixed = None
        self.orientation = None  # [rx, ry, rz]

        # 검증 기록
        self.visited_points = []
        self.total_tests = 0
        self.success_count = 0

    # ── 설정 저장/불러오기 ──────────────────────
    def save_config(self):
        data = {
            "corners": self.corners,
            "z_fixed": self.z_fixed,
            "orientation": self.orientation,
            "visited_points": self.visited_points,
            "total_tests": self.total_tests,
            "success_count": self.success_count,
        }
        with open(SAVE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] 설정 저장됨: {SAVE_FILE} (포인트 {self.total_tests}개 포함)")

    def load_config(self):
        if not os.path.exists(SAVE_FILE):
            print(f"[WARN] 파일 없음: {SAVE_FILE}")
            return False
        with open(SAVE_FILE, 'r') as f:
            data = json.load(f)
        self.corners = data.get("corners", [None]*4)
        self.z_fixed = data.get("z_fixed")
        self.orientation = data.get("orientation")
        self.visited_points = [tuple(p) for p in data.get("visited_points", [])]
        self.total_tests = data.get("total_tests", len(self.visited_points))
        self.success_count = data.get("success_count", sum(1 for p in self.visited_points if p[2]))
        for i, c in enumerate(self.corners):
            if c:
                print(f"[LOAD] 꼭짓점{i+1}: X={c[0]:.1f} Y={c[1]:.1f}")
        print(f"[LOAD] Z={self.z_fixed}, orientation={self.orientation}")
        print(f"[LOAD] 포인트 기록: {self.success_count}/{self.total_tests}개")
        return True

    # ── 영역 계산 ───────────────────────────────
    def all_corners_set(self):
        return all(c is not None for c in self.corners)

    def is_ready(self):
        return self.all_corners_set() and self.z_fixed is not None and self.orientation is not None

    def get_center_xy(self):
        """4꼭짓점의 중심 XY"""
        xs = [c[0] for c in self.corners]
        ys = [c[1] for c in self.corners]
        return sum(xs) / 4.0, sum(ys) / 4.0

    def get_bounds(self):
        """4꼭짓점의 bounding box"""
        xs = [c[0] for c in self.corners]
        ys = [c[1] for c in self.corners]
        return min(xs), max(xs), min(ys), max(ys)

    def generate_random_point(self):
        """4꼭짓점 내 랜덤 포인트 (bilinear interpolation)"""
        if not self.is_ready():
            print("[ERROR] 작업 영역 미설정 (꼭짓점 4개 + Z + orientation)")
            return None

        sorted_c = sort_corners_cw(self.corners)
        p0 = np.array(sorted_c[0][:2])
        p1 = np.array(sorted_c[1][:2])
        p2 = np.array(sorted_c[2][:2])
        p3 = np.array(sorted_c[3][:2])

        # bilinear: P = (1-u)(1-v)*P0 + u*(1-v)*P1 + u*v*P2 + (1-u)*v*P3
        u = random.random()
        v = random.random()
        pt = (1-u)*(1-v)*p0 + u*(1-v)*p1 + u*v*p2 + (1-u)*v*p3

        rx, ry, rz = self.orientation
        return [pt[0], pt[1], self.z_fixed, rx, ry, rz]

    # ── 로봇 이동 ──────────────────────────────
    def move_to_center_hover(self):
        """판 중앙, Z_fixed + HOVER_HEIGHT 위치로 이동"""
        cx, cy = self.get_center_xy()
        rx, ry, rz = self.orientation
        hover_z = self.z_fixed + HOVER_HEIGHT
        hover_pose = [cx, cy, hover_z, rx, ry, rz]
        print(f"[HOVER] 판 중앙 상공: X={cx:.1f}, Y={cy:.1f}, Z={hover_z:.1f}")
        ret = self.robot.MoveL(hover_pose, TOOL_NO, 0)
        if ret != 0:
            print(f"[ERROR] MoveL hover 실패: ret={ret}")
            return False
        return wait_done(self.robot)

    def move_to_point(self, target):
        """MoveL로 타겟 포인트 이동"""
        print(f"[MOVE] 랜덤 포인트: X={target[0]:.1f}, Y={target[1]:.1f}, Z={target[2]:.1f}")
        ret = self.robot.MoveL(target, TOOL_NO, 0)
        if ret != 0:
            print(f"[ERROR] MoveL 실패: ret={ret}")
            return False
        done = wait_done(self.robot)
        if done:
            actual = get_stable_tcp(self.robot, n=3, delay=0.2)
            if actual:
                dx = actual[0] - target[0]
                dy = actual[1] - target[1]
                err = (dx**2 + dy**2) ** 0.5
                print(f"[OK] 도달: X={actual[0]:.1f}, Y={actual[1]:.1f} (오차: {err:.2f}mm)")
                return True
        return False

    def go_home(self):
        print("[HOME] 복귀 중...")
        self.robot.MoveJ(HOME_J, TOOL_NO, 0)
        wait_done(self.robot)
        print("[HOME] 도착")

    # ── UI 오버레이 ─────────────────────────────
    def draw_overlay(self, frame):
        lines = ["=== 작업 영역 검증 (4P) ==="]

        for i in range(4):
            c = self.corners[i]
            if c:
                lines.append(f"P{i+1}: X={c[0]:.1f} Y={c[1]:.1f}")
            else:
                lines.append(f"P{i+1}: [미설정] → [{i+1}]")

        if self.z_fixed is not None:
            lines.append(f"Z={self.z_fixed:.1f}  hover=Z+{HOVER_HEIGHT}")
        else:
            lines.append("Z: [미설정] → [z]")

        if self.all_corners_set():
            x_min, x_max, y_min, y_max = self.get_bounds()
            lines.append(f"X:[{x_min:.0f}~{x_max:.0f}] Y:[{y_min:.0f}~{y_max:.0f}]")

        ready = "준비완료" if self.is_ready() else "설정필요"
        lines.append(f"{ready} | 테스트: {self.success_count}/{self.total_tests}")
        lines.append("")
        lines.append("[1][2][3][4]꼭짓점 [z]Z고정")
        lines.append("[Enter]랜덤이동 [h]HOME [s]저장 [l]불러오기 [q]종료")

        for i, line in enumerate(lines):
            frame = put_korean_text(frame, line, (10, 10 + i * 25), font_size=18, color=(0, 255, 0))

        if self.all_corners_set() and self.visited_points:
            self._draw_minimap(frame)

        return frame

    def _draw_minimap(self, frame):
        x_min, x_max, y_min, y_max = self.get_bounds()
        map_w, map_h = 180, 180
        margin = 15
        h, w = frame.shape[:2]
        ox = w - map_w - margin
        oy = h - map_h - margin

        overlay = frame.copy()
        cv2.rectangle(overlay, (ox, oy), (ox + map_w, oy + map_h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (ox, oy), (ox + map_w, oy + map_h), (100, 100, 100), 1)

        # 4꼭짓점 (노란색)
        for c in self.corners:
            nx = int((c[0] - x_min) / (x_max - x_min + 1e-6) * (map_w - 20)) + ox + 10
            ny = int((c[1] - y_min) / (y_max - y_min + 1e-6) * (map_h - 20)) + oy + 10
            cv2.circle(frame, (nx, ny), 5, (255, 255, 0), -1)

        # 방문 포인트
        for (px, py, success) in self.visited_points:
            nx = int((px - x_min) / (x_max - x_min + 1e-6) * (map_w - 20)) + ox + 10
            ny = int((py - y_min) / (y_max - y_min + 1e-6) * (map_h - 20)) + oy + 10
            color = (0, 255, 0) if success else (0, 0, 255)
            cv2.circle(frame, (nx, ny), 4, color, -1)

    # ── 메인 루프 ──────────────────────────────
    def run(self):
        print("\n" + "="*50)
        print("  작업 영역 검증 (4-Point Workspace)")
        print("="*50)
        print("  [1~4] 꼭짓점 저장   [z] Z 높이 저장")
        print("  [Enter] 중앙상공→랜덤포인트")
        print("  [h] HOME  [s] 저장  [l] 불러오기  [q] 종료")
        print("="*50 + "\n")

        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())
                frame = self.draw_overlay(frame)

                cv2.imshow("Workspace Validation", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("[EXIT] 종료")
                    break

                elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                    idx = key - ord('1')
                    tcp = get_stable_tcp(self.robot)
                    if tcp:
                        self.corners[idx] = tcp[:6]
                        if self.orientation is None:
                            self.orientation = tcp[3:6]
                        print(f"[SET] 꼭짓점{idx+1}: X={tcp[0]:.1f} Y={tcp[1]:.1f} Z={tcp[2]:.1f}")
                    else:
                        print("[ERROR] TCP 읽기 실패")

                elif key == ord('z'):
                    tcp = get_stable_tcp(self.robot)
                    if tcp:
                        self.z_fixed = tcp[2]
                        print(f"[SET] Z 고정: {self.z_fixed:.1f}mm")
                    else:
                        print("[ERROR] TCP 읽기 실패")

                elif key == 13:  # Enter
                    if not self.is_ready():
                        print("[ERROR] 설정 미완료 (꼭짓점 4개 + Z 필요)")
                        continue

                    target = self.generate_random_point()
                    if target:
                        self.total_tests += 1
                        # 1) 판 중앙 상공(Z+150)으로 이동
                        hover_ok = self.move_to_center_hover()
                        if hover_ok:
                            # 2) 랜덤 포인트(Z고정)로 하강 이동
                            success = self.move_to_point(target)
                        else:
                            success = False
                        self.visited_points.append((target[0], target[1], success))
                        if success:
                            self.success_count += 1
                        print(f"[STAT] 성공률: {self.success_count}/{self.total_tests}")

                elif key == ord('h'):
                    self.go_home()

                elif key == ord('s'):
                    self.save_config()

                elif key == ord('l'):
                    self.load_config()

        except KeyboardInterrupt:
            print("\n[EXIT] Ctrl+C")
        finally:
            self.go_home()
            self.pipeline.stop()
            cv2.destroyAllWindows()
            self._print_report()

    def _print_report(self):
        print("\n" + "="*50)
        print("  검증 결과 리포트")
        print("="*50)
        print(f"  총 테스트:  {self.total_tests}")
        print(f"  성공:       {self.success_count}")
        print(f"  실패:       {self.total_tests - self.success_count}")
        if self.total_tests > 0:
            print(f"  성공률:     {self.success_count/self.total_tests*100:.1f}%")
        if self.all_corners_set():
            x_min, x_max, y_min, y_max = self.get_bounds()
            print(f"  X 범위:     [{x_min:.1f} ~ {x_max:.1f}] mm")
            print(f"  Y 범위:     [{y_min:.1f} ~ {y_max:.1f}] mm")
            if self.z_fixed:
                print(f"  Z 고정:     {self.z_fixed:.1f} mm")
        print("="*50)


if __name__ == "__main__":
    validator = WorkspaceValidator()
    validator.run()