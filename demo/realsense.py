import pyrealsense2 as rs
import numpy as np
import cv2

# 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Depth → Color 정렬
align = rs.align(rs.stream.color)
profile = pipeline.start(config)

# Depth 스케일 (미터 변환용)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Intrinsics (3D 좌표 변환용)
depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# Colorizer (Depth 시각화용)
colorizer = rs.colorizer()

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # numpy 변환
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # 화면 중앙 좌표
        h, w = depth_image.shape
        cx, cy = w // 2, h // 2

        # 중앙 픽셀의 거리 (미터)
        dist = depth_frame.get_distance(cx, cy)

        # 2D 픽셀 → 3D 좌표 (카메라 기준, 미터)
        point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], dist)
        x, y, z = point_3d

        # 화면에 정보 표시
        text = f"3D: X={x:.3f} Y={y:.3f} Z={z:.3f} m"
        cv2.putText(color_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.circle(color_image, (cx, cy), 5, (0, 255, 0), -1)

        cv2.putText(depth_colormap, f"Depth: {dist:.3f} m", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # RGB-D 나란히 표시
        combined = np.hstack((color_image, depth_colormap))
        cv2.imshow("RGB-D Viewer", combined)

        key = cv2.waitKey(1)
        if key == 27:  # ESC 종료
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()