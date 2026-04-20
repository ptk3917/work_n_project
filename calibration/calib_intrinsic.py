"""
D455 내부 파라미터 + Hand-Eye 캘리브레이션
체커보드: 8x5 내부 코너, 25mm 한 칸

실행: python calibration.py

Step 1 - 내부 파라미터 캘리브레이션 (로봇 고정, 체커보드 움직임)
Step 2 - Hand-Eye 캘리브레이션 (체커보드 고정, 로봇 움직임)

조작:
  스페이스바: 현재 프레임 저장
  q: 저장 종료 후 계산 시작
"""
import sys
import os
sys.path.insert(0, os.path.expanduser("~/vlm_agent/qwen_ver1"))

import pyrealsense2 as rs
import numpy as np
import cv2
import pickle
from fairino import Robot


# ══════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════
BOARD_CORNERS  = (9, 6)    # 내부 코너 수 (가로, 세로)
SQUARE_SIZE_MM = 25.0      # 한 칸 크기 (mm)
ROBOT_IP       = "192.168.58.3"
MIN_SAMPLES    = 15        # 최소 수집 샘플 수


# ══════════════════════════════════════════════════════
# D455 초기화
# ══════════════════════════════════════════════════════
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

print("카메라 안정화 대기 중...")
for _ in range(30):
    pipeline.wait_for_frames()
print("✅ D455 준비 완료")


# ══════════════════════════════════════════════════════
# FR5 초기화
# ══════════════════════════════════════════════════════
print(f"FR5 연결 중... ({ROBOT_IP})")
robot = Robot.RPC(ROBOT_IP)
print("✅ FR5 준비 완료")


# ══════════════════════════════════════════════════════
# 체커보드 3D 기준점
# ══════════════════════════════════════════════════════
objp = np.zeros((BOARD_CORNERS[0] * BOARD_CORNERS[1], 3), np.float32)
objp[:, :2] = np.mgrid[
    0:BOARD_CORNERS[0], 0:BOARD_CORNERS[1]
].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def get_frame():
    frames = pipeline.wait_for_frames()
    color  = np.asanyarray(frames.get_color_frame().get_data())
    return color  # BGR


def detect_corners(frame_bgr):
    """체커보드 코너 검출 → (found, corners, gray)"""
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, BOARD_CORNERS, None)
    if found:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners, gray


# ══════════════════════════════════════════════════════
# STEP 1: 내부 파라미터 캘리브레이션
# ══════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("STEP 1: 내부 파라미터 캘리브레이션")
print("=" * 55)
print("방법: 로봇은 고정, 체커보드를 다양한 각도/거리로 움직이세요.")
print(f"목표: {MIN_SAMPLES}장 이상 수집")
print("스페이스바: 저장 | q: 완료\n")

objpoints_int = []   # 3D 기준점
imgpoints_int = []   # 2D 이미지 코너

collected = 0
while True:
    frame = get_frame()
    found, corners, gray = detect_corners(frame)

    display = frame.copy()
    if found:
        cv2.drawChessboardCorners(display, BOARD_CORNERS, corners, found)
        cv2.putText(display, f"FOUND | samples={collected} | SPACE to save",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    else:
        cv2.putText(display, f"NOT FOUND | samples={collected}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.putText(display, "q: finish collection",
                (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.imshow("Step1: Intrinsic Calibration", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        if collected < MIN_SAMPLES:
            print(f"⚠ 샘플 부족 ({collected}/{MIN_SAMPLES}). 더 수집하세요.")
            continue
        break

    elif key == ord(' ') and found:
        objpoints_int.append(objp)
        imgpoints_int.append(corners)
        collected += 1
        print(f"  ✅ #{collected} 저장")

cv2.destroyAllWindows()

# 내부 파라미터 계산
print("\n내부 파라미터 계산 중...")
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints_int, imgpoints_int, gray.shape[::-1], None, None
)

print(f"\n✅ 내부 파라미터 캘리브레이션 완료")
print(f"  재투영 오차: {ret:.4f} px  (목표: < 1.0)")
print(f"  FX={K[0,0]:.2f}  FY={K[1,1]:.2f}")
print(f"  CX={K[0,2]:.2f}  CY={K[1,2]:.2f}")
print(f"  왜곡계수: {dist.ravel()}")

# 저장
np.save(os.path.expanduser("~/vlm_agent/qwen_ver1/camera_matrix.npy"), K)
np.save(os.path.expanduser("~/vlm_agent/qwen_ver1/dist_coeffs.npy"), dist)
print("  💾 camera_matrix.npy, dist_coeffs.npy 저장 완료")


# ══════════════════════════════════════════════════════
# STEP 2: Hand-Eye 캘리브레이션
# ══════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("STEP 2: Hand-Eye 캘리브레이션 (Eye-in-Hand)")
print("=" * 55)
print("방법: 체커보드를 고정 위치에 두고")
print("      로봇을 20가지 이상 다른 자세로 이동시키세요.")
print("      각 자세에서 스페이스바를 누르세요.")
print(f"목표: {MIN_SAMPLES}장 이상 수집")
print("⚠ 주의: 매 자세마다 체커보드가 카메라에 보여야 해요!\n")

input("준비되면 엔터를 누르세요...")

R_gripper2base_list = []
t_gripper2base_list = []
R_target2cam_list   = []
t_target2cam_list   = []

collected = 0
while True:
    frame = get_frame()
    found, corners, gray = detect_corners(frame)

    display = frame.copy()
    if found:
        cv2.drawChessboardCorners(display, BOARD_CORNERS, corners, found)
        cv2.putText(display, f"FOUND | samples={collected} | SPACE to save",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    else:
        cv2.putText(display, f"NOT FOUND | samples={collected}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.putText(display, "Move robot to new pose, then SPACE | q: finish",
                (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
    cv2.imshow("Step2: Hand-Eye Calibration", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        if collected < MIN_SAMPLES:
            print(f"⚠ 샘플 부족 ({collected}/{MIN_SAMPLES}). 더 수집하세요.")
            continue
        break

    elif key == ord(' ') and found:
        # 체커보드 → 카메라 변환
        _, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
        R_t2c, _      = cv2.Rodrigues(rvec)

        # 로봇 FK → 그리퍼 → 베이스 변환
        _, pose = robot.GetActualTCPPose(1)
        x, y, z, rx, ry, rz = pose

        from scipy.spatial.transform import Rotation
        R_g2b = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
        t_g2b = np.array([[x], [y], [z]])

        R_gripper2base_list.append(R_g2b)
        t_gripper2base_list.append(t_g2b)
        R_target2cam_list.append(R_t2c)
        t_target2cam_list.append(tvec)

        collected += 1
        print(f"  ✅ #{collected} 저장 | 로봇 포즈: X={x:.1f} Y={y:.1f} Z={z:.1f}")

cv2.destroyAllWindows()

# Hand-Eye 계산 (4가지 방법 비교)
print("\nHand-Eye 캘리브레이션 계산 중...")

methods = {
    "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
    "PARK":       cv2.CALIB_HAND_EYE_PARK,
}

best_T      = None
best_error  = float("inf")
best_method = ""

for name, method in methods.items():
    try:
        R_cam2ee, t_cam2ee = cv2.calibrateHandEye(
            R_gripper2base_list, t_gripper2base_list,
            R_target2cam_list,   t_target2cam_list,
            method=method
        )
        T = np.eye(4)
        T[:3, :3] = R_cam2ee
        T[:3,  3] = t_cam2ee.flatten()

        # 변환 행렬 유효성 체크 (번역 벡터 크기)
        t_norm = np.linalg.norm(t_cam2ee)
        print(f"  {name:12s} | 번역 벡터 크기: {t_norm:.1f} mm")

        if t_norm < best_error:
            best_error  = t_norm
            best_method = name
            best_T      = T
    except Exception as e:
        print(f"  {name:12s} | 실패: {e}")

print(f"\n✅ Hand-Eye 캘리브레이션 완료")
print(f"  최적 방법: {best_method}")
print(f"  T_cam_to_ee =\n{np.round(best_T, 4)}")

# 저장
np.save(os.path.expanduser("~/vlm_agent/qwen_ver1/T_cam_to_ee.npy"), best_T)
print("  💾 T_cam_to_ee.npy 저장 완료")

pipeline.stop()

# ══════════════════════════════════════════════════════
# 최종 결과 출력
# ══════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("캘리브레이션 완료! picking_pipeline.py 상단을 이렇게 수정하세요:")
print("=" * 55)
print(f"FX = {K[0,0]:.4f}")
print(f"FY = {K[1,1]:.4f}")
print(f"CX = {K[0,2]:.4f}")
print(f"CY = {K[1,2]:.4f}")
print(f"\nT_CAM_TO_EE = np.load('T_cam_to_ee.npy')")
print("=" * 55)