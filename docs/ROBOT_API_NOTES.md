# Fairino FR5 SDK 주의사항

FR5 로봇 개발 과정에서 발견한 SDK 제약 사항과 해결 패턴을 정리한 문서입니다.

## MoveJ — 비블로킹 이동

`MoveJ`는 호출 즉시 리턴되며, 로봇이 목표 위치에 도달하기 전에 다음 코드가 실행됩니다.
TCP 좌표를 읽기 전에 반드시 동기화가 필요합니다.

```python
def wait_done(robot, timeout=10):
    """MoveJ 완료 대기 — 0.4s 프리폴 후 폴링"""
    time.sleep(0.4)
    start = time.time()
    while time.time() - start < timeout:
        ret = robot.GetRobotMotionDone()
        # 반환값이 int 또는 tuple일 수 있음
        done = ret[1] if isinstance(ret, tuple) else ret
        if done == 1:
            return True
        time.sleep(0.05)
    return False

def get_stable_pose(robot, samples=5, interval=0.05):
    """여러 번 읽어서 평균 — UDP 노이즈 보정"""
    poses = []
    for _ in range(samples):
        ret = robot.GetActualTCPPose(0)
        if isinstance(ret, tuple) and len(ret) >= 2:
            poses.append(ret[1])
        time.sleep(interval)
    return [sum(p[i] for p in poses) / len(poses) for i in range(6)]
```

## MoveL — 장거리 금지

장거리 `MoveL` 호출 시 "축1 관절 공간 내 명령 속도 초과" 에러가 발생합니다.

```
✅ 장거리 이동: MoveJ → HOME → MoveJ → 목표
❌ 장거리 이동: MoveL(현재 → 먼 목표)  # 관절 속도 초과 에러
✅ 단거리 정밀: MoveL (접근 → 피킹 등 짧은 거리)
```

## MoveGripper — 10개 인자 필수

```python
# MoveGripper(index, pos, speed, force, maxTime, block, type, rotNum, rotVel, rotTorque)
robot.MoveGripper(1, 100, 50, 30, 10000, 0, 0, 0, 0, 0)
#                 │   │    │   │    │      │  └─ 마지막 4개는 보통 0
#                 │   │    │   │    │      └─ block (0=비블로킹)
#                 │   │    │   │    └─ maxTime (ms)
#                 │   │    │   └─ force (N)
#                 │   │    └─ speed (%)
#                 │   └─ position (0=완전닫힘, 100=완전열림)
#                 └─ gripper index
```

초기화 순서: `ActGripper(1, 1)` → `time.sleep(2)` → `MoveGripper(...)`.

## MoveJ — desc_pos는 반드시 리스트

```python
✅ robot.MoveJ(joint_pos, tool=1, user=0, desc_pos=[0,0,0,0,0,0])
❌ robot.MoveJ(joint_pos, tool=1, user=0, desc_pos=0)  # 에러
```

## GetActualJointPosDegree — ctypes 에러

UDP 타이밍 이슈로 `GetActualJointPosDegree`가 ctypes 에러를 발생시킵니다.
대안으로 `GetActualTCPPose`에 `time.sleep(3)`을 사용합니다.

## J6 vs Rz 회전

피킹 각도 계산 시 **Rz 절대 사용 금지** — err=112 에러 발생.
반드시 J6 기반 회전을 사용해야 합니다.

```python
# J6 범위: base_j6 기준 CCW +70° / CW -240°
pick_j6 = base_j6 + obb_angle + OBB_TO_GRIPPER_OFFSET  # -90°
if pick_j6 > base_j6 + 70:
    pick_j6 -= 180  # 범위 초과 시 180° 플립
```

## RPC 모드 제한

수동 모드(티칭펜던트) 활성화 시 RPC 이동 명령이 차단됩니다 (에러 185).
프로그래밍 모드로 전환 후 실행해야 합니다.

## Tool Frame 오프셋

`move_tool(dx, dy, dz)` 구현 시 ZYX Euler 회전 행렬로 도구 좌표계 오프셋을 계산합니다.

```python
def move_tool(robot, dx=0, dy=0, dz=0, speed=20):
    """도구 좌표계 기준 이동"""
    tcp = get_stable_pose(robot)
    rx, ry, rz = [math.radians(a) for a in tcp[3:6]]
    # ZYX Euler 회전 행렬
    R = euler_to_rotation_matrix(rx, ry, rz)
    offset_base = R @ np.array([dx, dy, dz])
    target = tcp.copy()
    target[0] += offset_base[0]
    target[1] += offset_base[1]
    target[2] += offset_base[2]
    robot.MoveL(target, 1, 0, [0,0,0,0,0,0])
```
