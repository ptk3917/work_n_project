#!/usr/bin/env python3
"""
방향 정렬 파이프라인
====================
흩어진 제품을 잡아서 한 방향으로 정렬하여 일렬 배치.

사용법:
  1) 배치 슬롯 티칭:  python calib_ort.py --teach-slots
  2) 목표 각도 확인:  python calib_ort.py --check-angle
  3) 정렬 실행:       python calib_ort.py --run
  4) 정렬 실행(자동): python calib_ort.py --run --auto

사전 준비:
  - OBB 모델 (best.pt)
  - XYZ 보정 (xyz_correction.json)
  - 수직 스캔 위치 (scan_down_pose.json)
  - 접근 자세 (down_pose.json)
"""

import os, sys, time, json, argparse
import numpy as np
import cv2
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation
from ultralytics import YOLO
from fairino import Robot

# =====================================================================
# 설정
# =====================================================================
ROBOT_IP = "192.168.58.3"
TOOL_NO, USER_NO = 1, 0
SPEED, SPEED_J = 40, 70

FX, FY = 386.4894, 386.6551
CX, CY = 325.7376, 246.5258
IMG_W, IMG_H = 640, 480

CALIB_PATH = os.path.expanduser("~/vlm_agent/qwen_ver1/T_cam_to_ee.npy")
T_CAM_TO_EE = np.load(CALIB_PATH) if os.path.exists(CALIB_PATH) else np.eye(4)

HOME_J  = [10.0, -98.0, 100.0, -94.0, -84.0, -111.0]
WAY_J   = HOME_J

# ── 파라미터 ──
APPROACH_DIST  = 120
GRIPPER_OFFSET = -20
Z_FLOOR        = 150
J6_MIN, J6_MAX = -150.0, 150.0
J6_CCW_RANGE = 70
J6_CW_RANGE  = 240
OBB_TO_GRIPPER_OFFSET = -90
PLACE_RZ = -60.0
OBB_CONFIDENCE = 0.5
MAX_PER_ROW = 10

# ── 모델/캘리브레이션 경로 ──
OBB_WEIGHTS         = os.path.expanduser("~/vlm_agent/qwen_ver1/obb_dataset/train/weights/best.pt")
XYZ_CORRECTION_PATH = "xyz_correction.json"
SCAN_DOWN_PATH      = "scan_down_pose.json"
DOWN_POSE_PATH      = "down_pose.json"
SLOTS_PATH          = "place_slots.json"
TARGET_ANGLE_PATH   = "target_angle.json"

CLASS_COLORS = {
    "ohyes":      (0, 255, 0),
    "freetime_y": (0, 255, 255),
    "freetime_b": (255, 150, 0),
}

GRIPPER_POS = {
    "ohyes": 60,
    "freetime_y": 32,
    "freetime_b": 32,
}


# =====================================================================
# 초기화
# =====================================================================
print("=" * 55)
print("  방향 정렬 파이프라인")
print("=" * 55)

rs_pipeline = rs.pipeline()
rs_config = rs.config()
rs_config.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.bgr8, 30)
rs_config.enable_stream(rs.stream.depth, IMG_W, IMG_H, rs.format.z16, 30)
rs_pipeline.start(rs_config)
align = rs.align(rs.stream.color)
for _ in range(30): rs_pipeline.wait_for_frames()
print("✅ D455 초기화 완료")

robot = Robot.RPC(ROBOT_IP)
ret, error = robot.GetRobotErrorCode()
if ret == 0 and error != 0:
    robot.ResetAllError(); time.sleep(0.5)
print("✅ FR5 연결 완료")

robot.SetAxleCommunicationParam(baudRate=7, dataBit=8, stopBit=1, verify=0,
                                 timeout=50, timeoutTimes=3, period=10)
time.sleep(1)
robot.SetGripperConfig(company=4, device=0); time.sleep(1)
robot.ActGripper(index=1, action=1); time.sleep(2)
robot.MoveGripper(index=1, pos=100, vel=50, force=50,
                  maxtime=5000, block=1, type=0, rotNum=0, rotVel=0, rotTorque=0)
print("✅ 그리퍼 초기화 완료")

obb_model = YOLO(OBB_WEIGHTS)
print(f"✅ OBB 모델 준비 ({list(obb_model.names.values())})")

# XYZ 보정
xyz_correct = lambda x, y, z: (x, y, z)
if os.path.exists(XYZ_CORRECTION_PATH):
    with open(XYZ_CORRECTION_PATH) as f:
        _ax = json.load(f)["axes"]
    def xyz_correct(x, y, z):
        return (_ax["x"]["scale"]*x+_ax["x"]["offset"] if "x" in _ax else x,
                _ax["y"]["scale"]*y+_ax["y"]["offset"] if "y" in _ax else y,
                _ax["z"]["scale"]*z+_ax["z"]["offset"] if "z" in _ax else z)
    print("✅ XYZ 보정 로드")

# 스캔 위치
scan_j = HOME_J
if os.path.exists(SCAN_DOWN_PATH):
    with open(SCAN_DOWN_PATH) as f: scan_j = json.load(f)["joints"]
    print("✅ 스캔 위치 로드")

# 접근 자세
down_rx = down_ry = down_rz = base_j6 = 0
down_joints = None
if os.path.exists(DOWN_POSE_PATH):
    with open(DOWN_POSE_PATH) as f: dp = json.load(f)
    down_rx, down_ry, down_rz = dp["rx"], dp["ry"], dp["rz"]
    base_j6 = dp.get("j6", 0.0)
    down_joints = dp.get("joints", None)
    print(f"✅ 접근 자세 로드 (J6={base_j6:.1f})")
else:
    print("⚠ 접근 자세 없음!"); sys.exit(1)

print("=" * 55)


# =====================================================================
# 로봇 함수
# =====================================================================
def wait_done(timeout=30.0):
    time.sleep(0.4)
    t0 = time.time()
    while time.time()-t0 < timeout:
        r = robot.GetRobotMotionDone()
        if (r[1] if isinstance(r,(list,tuple)) else r) == 1:
            time.sleep(0.1); return
        time.sleep(0.05)

def get_stable_pose(samples=5, interval=0.05):
    poses = []
    for _ in range(samples):
        _, p = robot.GetActualTCPPose(1); poses.append(p); time.sleep(interval)
    return np.array(poses).mean(axis=0).tolist()

def move_j(joints, label=""):
    if label: print(f"  {label}...")
    r = robot.MoveJ(joints, TOOL_NO, USER_NO, vel=SPEED_J, acc=SPEED_J, blendT=0)
    e = r[0] if isinstance(r,(list,tuple)) else r
    if e != 0: raise RuntimeError(f"{label} 실패 (err={e})")
    wait_done()

def move_l(pose, label=""):
    if label: print(f"  {label}...")
    r = robot.MoveL(pose, TOOL_NO, USER_NO, vel=SPEED, acc=SPEED)
    e = r[0] if isinstance(r,(list,tuple)) else r
    if e != 0: raise RuntimeError(f"{label} 실패 (err={e})")
    wait_done()

def gripper_open():
    robot.MoveGripper(index=1, pos=100, vel=50, force=50,
                      maxtime=5000, block=1, type=0, rotNum=0, rotVel=0, rotTorque=0)
    time.sleep(0.5)

def gripper_close(pos=30):
    robot.MoveGripper(index=1, pos=pos, vel=50, force=50,
                      maxtime=5000, block=1, type=0, rotNum=0, rotVel=0, rotTorque=0)
    time.sleep(0.5)


# =====================================================================
# 카메라 / 좌표
# =====================================================================
def capture_frame():
    frames = rs_pipeline.wait_for_frames()
    aligned = align.process(frames)
    return np.asanyarray(aligned.get_color_frame().get_data()), aligned.get_depth_frame()

def get_depth_at(df, cx, cy, w=5):
    d = np.asanyarray(df.get_data())
    h, ww = d.shape
    roi = d[max(0,int(cy)-w):min(h,int(cy)+w+1), max(0,int(cx)-w):min(ww,int(cx)+w+1)].astype(float)
    v = roi[roi>0]
    return float(np.median(v)) if len(v)>0 else 0.0

def pixel_to_base(px, py, depth_mm):
    z = depth_mm
    p_cam = np.array([(px-CX)*z/FX, (py-CY)*z/FY, z, 1.0])
    ee = get_stable_pose(3, 0.05)
    R = Rotation.from_euler('xyz', ee[3:], degrees=True).as_matrix()
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=ee[:3]
    return (T @ (T_CAM_TO_EE @ p_cam))[:3]


# =====================================================================
# OBB 감지
# =====================================================================
def detect_all(color_np):
    results = obb_model.predict(color_np, conf=OBB_CONFIDENCE, verbose=False)
    dets = []
    for r in results:
        if r.obb is None: continue
        for box in r.obb:
            xywhr = box.xywhr[0].cpu().numpy()
            cx, cy, w, h, angle = xywhr
            corners = box.xyxyxyxy[0].cpu().numpy().astype(int)
            dets.append({
                "label": obb_model.names.get(int(box.cls[0]),"?"),
                "conf": float(box.conf[0]),
                "center": (int(cx), int(cy)),
                "angle_deg": float(np.degrees(angle)),
                "corners": corners,
            })
    return dets

def draw_dets(vis, dets):
    for i, d in enumerate(dets):
        c = CLASS_COLORS.get(d["label"], (0,255,0))
        cv2.polylines(vis, [d["corners"]], True, c, 2)
        cx, cy = d["center"]
        cv2.circle(vis, (cx,cy), 4, (0,0,255), -1)
        label = f"[{i}] {d['label']} {d['angle_deg']:.0f}deg"
        top = d["corners"][:,1].argmin()
        lx, ly = int(d["corners"][top,0]), int(d["corners"][top,1])-8
        lx = max(0, min(lx, vis.shape[1]-150)); ly = max(15, ly)
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(vis, (lx,ly-th-4), (lx+tw+2,ly), c, -1)
        cv2.putText(vis, label, (lx+1,ly-3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)
    return vis


# =====================================================================
# J6 계산
# =====================================================================
def calc_pick_j6(obj_angle_deg):
    """Pick J6: 물체 각도에 맞춰 그리퍼 정렬. J6 범위 체크."""
    j6_upper = min(base_j6 + J6_CCW_RANGE, J6_MAX)
    j6_lower = max(base_j6 - J6_CW_RANGE,  J6_MIN)

    t1 = base_j6 + obj_angle_deg + OBB_TO_GRIPPER_OFFSET
    if j6_lower <= t1 <= j6_upper:
        return t1

    t2 = t1 + 180
    if j6_lower <= t2 <= j6_upper:
        return t2

    t3 = t1 - 180
    if j6_lower <= t3 <= j6_upper:
        return t3

    clamped = max(j6_lower, min(j6_upper, t1))
    print(f"  ⚠ Pick J6 클램프: {t1:.1f} → {clamped:.1f}")
    return clamped


# =====================================================================
# 티칭: 배치 슬롯
# =====================================================================
def teach_slots():
    """
    클래스별 배치 슬롯 티칭.
    각 클래스: 첫/마지막 위치 저장.
    """
    classes = list(obb_model.names.values())
    print(f"\n  클래스별 슬롯 티칭")
    print(f"  클래스: {classes}")
    print(f"  ─────────────────")

    move_j(HOME_J, "홈")
    if down_joints:
        move_j(down_joints, "그리퍼 수직 위치")

    all_slots = {}

    for cls_name in classes:
        c = CLASS_COLORS.get(cls_name, (0,255,0))
        print(f"\n{'='*45}")
        print(f"  [{cls_name}] 줄 티칭")
        print(f"  아무 키=시작 | N=건너뛰기")

        while True:
            color_np, _ = capture_frame()
            vis = color_np.copy()
            cv2.putText(vis, f"{cls_name} | Any key=start, N=skip",
                        (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
            cv2.imshow("Teach Slots", vis)
            key = cv2.waitKey(0) & 0xFF
            break

        if key == ord('n'):
            print(f"  [{cls_name}] 건너뜀")
            continue

        tcp_list = []
        for label in ["첫 번째", "마지막"]:
            print(f"\n  [{cls_name} - {label}] 조그 → SPACE")
            while True:
                color_np, _ = capture_frame()
                vis = color_np.copy()
                cv2.putText(vis, f"{cls_name} - {label} | [SPACE] Save",
                            (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
                h, w = vis.shape[:2]
                cv2.line(vis, (w//2-30,h//2), (w//2+30,h//2), (0,0,255), 1)
                cv2.line(vis, (w//2,h//2-30), (w//2,h//2+30), (0,0,255), 1)
                cv2.imshow("Teach Slots", vis)
                k = cv2.waitKey(30) & 0xFF
                if k == ord(' '): break
                elif k == ord('q'):
                    cv2.destroyAllWindows(); return
            tcp = get_stable_pose(5, 0.05)
            tcp_list.append(tcp)
            print(f"    TCP: X={tcp[0]:.1f} Y={tcp[1]:.1f} Z={tcp[2]:.1f}")

        all_slots[cls_name] = {
            "first": [round(v,4) for v in tcp_list[0]],
            "last":  [round(v,4) for v in tcp_list[1]],
        }
        print(f"  ✅ [{cls_name}] 저장")

    cv2.destroyAllWindows()

    with open(SLOTS_PATH, "w") as f:
        json.dump(all_slots, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ 전체 슬롯 저장: {SLOTS_PATH}")
    for cls_name, s in all_slots.items():
        f_ = s["first"]
        l_ = s["last"]
        print(f"     {cls_name}: ({f_[0]:.0f},{f_[1]:.0f}) → ({l_[0]:.0f},{l_[1]:.0f})")

    move_j(HOME_J, "홈 복귀")


# =====================================================================
# 티칭: 목표 각도
# =====================================================================
def check_angle():
    """정위치 제품의 OBB 각도 확인 → 목표 각도 저장."""
    print(f"\n  목표 각도 확인")
    print(f"  제품을 정위치(올바른 방향)로 놓고 확인하세요.")
    print(f"  [SPACE] 현재 각도 저장 | [Q] 취소")

    move_j(HOME_J, "홈")
    move_j(scan_j,  "스캔 위치")

    while True:
        color_np, _ = capture_frame()
        vis = color_np.copy()
        dets = detect_all(color_np)
        draw_dets(vis, dets)

        if dets:
            best = max(dets, key=lambda d: d["conf"])
            cv2.putText(vis, f"Target angle: {best['angle_deg']:.1f}deg | [SPACE] Save",
                        (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
        else:
            cv2.putText(vis, "No detection | Place product correctly",
                        (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)

        cv2.imshow("Check Target Angle", vis)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(' ') and dets:
            best = max(dets, key=lambda d: d["conf"])
            target = best["angle_deg"]
            with open(TARGET_ANGLE_PATH, "w") as f:
                json.dump({"target_angle_deg": round(target, 2)}, f, indent=2)
            cv2.destroyAllWindows()
            print(f"\n  ✅ 목표 각도 저장: {target:.1f}deg → {TARGET_ANGLE_PATH}")
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            print("  [취소]")
            break

    move_j(HOME_J, "홈 복귀")


# =====================================================================
# 메인: 정렬 실행
# =====================================================================
def run_sort(auto=False, target_class=None):
    """
    전체 정렬: 클래스별 순차 처리, 물체 없을 때까지 반복.
    가려진 물체도 앞의 것을 치우면 다음 스캔에서 발견됨.
    """

    # 슬롯 로드
    if not os.path.exists(SLOTS_PATH):
        print(f"[오류] 슬롯 없음! --teach-slots 먼저 실행")
        return
    with open(SLOTS_PATH) as f:
        sd = json.load(f)
    if "first" in sd:
        sd = {"_all": sd}

    # 목표 각도 로드
    if not os.path.exists(TARGET_ANGLE_PATH):
        print(f"[오류] 목표 각도 없음! --check-angle 먼저 실행")
        return
    with open(TARGET_ANGLE_PATH) as f:
        target_angle = json.load(f)["target_angle_deg"]

    # 처리할 클래스 목록
    if target_class:
        class_order = [target_class]
    else:
        class_order = [c for c in sd.keys() if c != "_all"]
        if not class_order:
            class_order = ["_all"]

    # 슬롯 간격 사전 계산 (클래스별)
    slot_steps = {}
    for cls_name in class_order:
        slot_key = cls_name if cls_name in sd else "_all"
        if slot_key not in sd:
            continue
        first = np.array(sd[slot_key]["first"])
        last  = np.array(sd[slot_key]["last"])
        step  = (last - first) / max(MAX_PER_ROW - 1, 1)
        slot_steps[cls_name] = {"first": first, "step": step, "last": last}

    print(f"\n  목표 각도: {target_angle:.1f}deg")
    print(f"  Place Rz: {PLACE_RZ:.1f} (고정)")
    print(f"  처리 순서: {class_order}")
    print(f"  한 줄 최대: {MAX_PER_ROW}개")

    try:
        total_done = 0

        for cls_name in class_order:
            if cls_name not in slot_steps:
                print(f"\n  ⚠ [{cls_name}] 슬롯 미등록 → 건너뜀")
                continue

            ss = slot_steps[cls_name]
            slot_idx = 0

            print(f"\n{'='*50}")
            print(f"  [{cls_name}] 처리 시작 (Place Rz={PLACE_RZ:.1f})")

            while slot_idx < MAX_PER_ROW:
                # ── 스캔 ──
                move_j(HOME_J, "홈")
                move_j(scan_j, "스캔")
                time.sleep(0.7)
                for _ in range(7):
                    color_np, depth_frame = capture_frame()
                    time.sleep(0.05)
                dets = detect_all(color_np)

                # ── 감지 결과 표시 ──
                vis = color_np.copy()
                draw_dets(vis, dets)
                cls_dets = [d for d in dets if d["label"] == cls_name]
                info = f"{cls_name}: {len(cls_dets)} | done: {total_done}"
                if not auto:
                    info += " | SPACE=go Q=quit"
                cv2.putText(vis, info, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.imshow("Sort", vis)
                key = cv2.waitKey(1000 if auto else 0) & 0xFF
                if key == ord('q'):
                    print("\n  [중단]")
                    raise KeyboardInterrupt

                # ── 이 클래스 필터 ──
                if not cls_dets:
                    print(f"\n  [{cls_name}] 더 이상 없음. {slot_idx}개 완료.")
                    break

                det = max(cls_dets, key=lambda d: d["conf"])
                obj_angle = det["angle_deg"]
                print(f"\n  [{cls_name} #{slot_idx+1}] angle={obj_angle:.1f}deg")

                # ── 좌표 ──
                cx, cy = det["center"]
                depth_mm = get_depth_at(depth_frame, cx, cy)
                if depth_mm <= 0:
                    print("  ⚠ depth=0 → 건너뜀")
                    continue

                base_xyz = pixel_to_base(cx, cy, depth_mm)
                tx, ty, tz = xyz_correct(base_xyz[0], base_xyz[1], base_xyz[2])
                print(f"  좌표: X={tx:.1f} Y={ty:.1f} Z={tz:.1f}")

                pick_j6 = calc_pick_j6(obj_angle)
                print(f"  Pick J6={pick_j6:.1f}")

                pick_z     = max(tz + GRIPPER_OFFSET, Z_FLOOR)
                approach_z = max(tz + APPROACH_DIST, Z_FLOOR + APPROACH_DIST)

                # ── Pick ──
                gripper_open()
                move_l([tx, ty, approach_z, down_rx, down_ry, down_rz], "접근")

                _, cur_j = robot.GetActualJointPosDegree(0)
                cur_j = list(cur_j)
                cur_j[5] = pick_j6
                move_j(cur_j, f"Pick J6→{pick_j6:.1f}")

                tcp_now = get_stable_pose(3, 0.05)
                move_l([tx, ty, pick_z, tcp_now[3], tcp_now[4], tcp_now[5]], "하강")
                gripper_close(pos=GRIPPER_POS.get(cls_name, 30))
                move_l([tx, ty, approach_z, tcp_now[3], tcp_now[4], tcp_now[5]], "리프트")

                move_j(HOME_J, "안전 경유")

                # ── Place (고정 Rz) ──
                slot_tcp = (ss["first"] + ss["step"] * slot_idx).tolist()
                sx, sy, sz = slot_tcp[0], slot_tcp[1], slot_tcp[2]
                s_approach_z = max(sz + APPROACH_DIST, Z_FLOOR + APPROACH_DIST)

                print(f"  [Place → slot {slot_idx}] X={sx:.1f} Y={sy:.1f}")
                move_l([sx, sy, s_approach_z, down_rx, down_ry, PLACE_RZ], "슬롯 접근")
                move_l([sx, sy, sz, down_rx, down_ry, PLACE_RZ], "배치")
                gripper_open()
                move_l([sx, sy, s_approach_z, down_rx, down_ry, PLACE_RZ], "후퇴")

                move_j(HOME_J, "안전 경유")
                slot_idx += 1
                total_done += 1
                print(f"  ✅ 완료!")

        print(f"\n{'='*55}")
        print(f"  정렬 완료! 총 {total_done}개 배치")
        print(f"{'='*55}")

    except KeyboardInterrupt:
        print("\n[중단]")
    finally:
        move_j(HOME_J, "홈 복귀")
        cv2.destroyAllWindows()


# =====================================================================
# 엔트리 포인트
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="방향 정렬 파이프라인")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--teach-slots", action="store_true",
                       help="배치 슬롯 첫/마지막 위치 티칭")
    group.add_argument("--check-angle", action="store_true",
                       help="정위치 제품의 OBB 각도 확인 및 저장")
    group.add_argument("--run", action="store_true",
                       help="정렬 실행")

    parser.add_argument("--auto", action="store_true",
                       help="자동 모드 (확인 없이)")
    parser.add_argument("--target-class", type=str, default=None,
                       help="특정 클래스만 (예: freetime_b)")
    args = parser.parse_args()

    try:
        if args.teach_slots:
            teach_slots()
        elif args.check_angle:
            check_angle()
        elif args.run:
            run_sort(auto=args.auto, target_class=args.target_class)
    except Exception as e:
        print(f"\n[오류] {e}")
        move_j(HOME_J, "홈 복귀")
    finally:
        rs_pipeline.stop()
        print("[종료]")


if __name__ == "__main__":
    main()