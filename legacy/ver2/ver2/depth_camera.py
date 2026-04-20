import numpy as np
import cv2
import pyrealsense2 as rs


class DepthCamera:
    """
    Intel RealSense D455 깊이 카메라
    - RGB + Depth 동시 스트림
    - 픽셀 좌표 → 실제 3D 좌표 (mm) 변환
    """

    def __init__(self, width=640, height=480, fps=30):
        print("🔄 D455 초기화 중...")

        self.width  = width
        self.height = height

        self.pipeline = rs.pipeline()
        config        = rs.config()

        config.enable_stream(
            rs.stream.color, width, height, rs.format.bgr8, fps
        )
        config.enable_stream(
            rs.stream.depth, width, height, rs.format.z16, fps
        )

        profile = self.pipeline.start(config)

        # ── Depth → Color 정렬 (같은 픽셀이 같은 위치 가리키도록) ──
        self.align = rs.align(rs.stream.color)

        # ── 카메라 내부 파라미터 저장 ─────────────────
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()  # m 단위 스케일

        depth_stream = profile.get_stream(rs.stream.depth)
        self.intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

        print(f"✅ D455 초기화 완료")
        print(f"   해상도: {width}x{height} @ {fps}fps")
        print(f"   깊이 스케일: {self.depth_scale}")
        print(f"   fx:{self.intrinsics.fx:.1f}  fy:{self.intrinsics.fy:.1f}")
        print(f"   cx:{self.intrinsics.ppx:.1f}  cy:{self.intrinsics.ppy:.1f}")

    # ── 프레임 취득 ────────────────────────────────
    def get_frames(self):
        """
        RGB 프레임 + Depth 프레임 동시 반환
        반환: (color_frame: np.ndarray, depth_frame: rs.depth_frame)
        """
        frames        = self.pipeline.wait_for_frames()
        aligned       = self.align.process(frames)
        color_frame   = aligned.get_color_frame()
        depth_frame   = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_frame

    # ── 픽셀 → 3D 좌표 ────────────────────────────
    def get_3d_point(self, cx: int, cy: int,
                     depth_frame) -> tuple:
        """
        픽셀 좌표 + depth_frame → 실제 3D 좌표 (mm)
        반환: (x, y, z) mm 단위  |  z=0 이면 측정 실패
        """
        # 경계 체크
        cx = max(0, min(cx, self.width  - 1))
        cy = max(0, min(cy, self.height - 1))

        # 깊이값 (m 단위)
        dist_m = depth_frame.get_distance(int(cx), int(cy))

        if dist_m <= 0:
            return 0.0, 0.0, 0.0  # 측정 실패

        # m → mm
        z = dist_m * 1000.0

        # 핀홀 카메라 역투영
        x = (cx - self.intrinsics.ppx) * z / self.intrinsics.fx
        y = (cy - self.intrinsics.ppy) * z / self.intrinsics.fy

        return round(x, 1), round(y, 1), round(z, 1)

    # ── bbox 중심 3D 좌표 ──────────────────────────
    def bbox_to_3d(self, bbox: tuple, depth_frame) -> dict:
        """
        bbox (x1,y1,x2,y2) → 중심 3D 좌표 + 안정적 깊이 추정
        중심 한 픽셀 대신 3x3 영역 중앙값 사용 → 노이즈 감소
        """
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # 3x3 영역 깊이 중앙값 (노이즈 제거)
        depths = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx = max(0, min(cx + dx, self.width  - 1))
                ny = max(0, min(cy + dy, self.height - 1))
                d  = depth_frame.get_distance(nx, ny)
                if d > 0:
                    depths.append(d)

        if not depths:
            return {"x": 0, "y": 0, "z": 0,
                    "valid": False, "dist_cm": 0}

        dist_m = float(np.median(depths))
        z      = dist_m * 1000.0
        x      = (cx - self.intrinsics.ppx) * z / self.intrinsics.fx
        y      = (cy - self.intrinsics.ppy) * z / self.intrinsics.fy

        return {
            "x":       round(x, 1),
            "y":       round(y, 1),
            "z":       round(z, 1),
            "valid":   True,
            "dist_cm": round(dist_m * 100, 1),
            "px":      cx,
            "py":      cy,
        }

    # ── 깊이 시각화 ────────────────────────────────
    def colorize_depth(self, depth_frame) -> np.ndarray:
        depth_image = np.asanyarray(depth_frame.get_data())
    
        # D455 유효 범위로 클리핑 (600mm ~ 6000mm)
        MIN_DEPTH = 300
        MAX_DEPTH = 3000   # 3m로 제한 → 실내 작업 환경에 맞춤
    
        depth_clipped = np.clip(depth_image, MIN_DEPTH, MAX_DEPTH)
    
        # 측정 불가 픽셀(0) 마스킹
        valid_mask = depth_image > 0
    
        # 정규화 0~255
        depth_norm = ((depth_clipped - MIN_DEPTH) /
                      (MAX_DEPTH - MIN_DEPTH) * 255).astype(np.uint8)
    
        # 가까울수록 밝게 반전 (빨강=가까이)
        depth_norm = 255 - depth_norm
        depth_norm[~valid_mask] = 0  # 측정 불가 = 검정
    
        colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
        return cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    
    # ── 거리 오버레이 ──────────────────────────────
    def draw_distance(self, frame: np.ndarray,
                      point3d: dict) -> np.ndarray:
        """프레임에 3D 좌표 정보 오버레이"""
        out = frame.copy()
        if not point3d.get("valid"):
            return out

        px, py   = point3d["px"], point3d["py"]
        dist_cm  = point3d["dist_cm"]
        x, y, z  = point3d["x"], point3d["y"], point3d["z"]

        # 중심점
        cv2.circle(out, (px, py), 8, (0, 255, 0), -1)

        # 거리 텍스트
        cv2.putText(out,
                    f"{dist_cm:.1f}cm",
                    (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

        # 3D 좌표 텍스트
        cv2.putText(out,
                    f"({x:.0f}, {y:.0f}, {z:.0f})mm",
                    (px + 10, py + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1)

        return out

    # ── 종료 ───────────────────────────────────────
    def release(self):
        self.pipeline.stop()
        print("📷 D455 종료")