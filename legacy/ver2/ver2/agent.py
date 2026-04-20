import time
import cv2
import numpy as np
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    user_input:   str
    frame:        Optional[np.ndarray]   # RGB 프레임
    depth_frame:  Optional[object]       # D455 depth_frame
    detections:   list                   # YOLO 감지 결과
    detections_3d: list                  # 3D 좌표 포함 감지 결과
    scene_info:   dict                   # LLaVA 장면 분석
    task_plan:    dict                   # 로봇 액션 플랜
    current_step: int
    status:       str
    log:          List[str]


def build_agent(vlm, yolo, robot, depth=None):
    """
    vlm   : LLaVAEngine
    yolo  : YOLODetector
    robot : FR5Controller
    depth : DepthCamera (None 이면 2D 모드)
    """

    use_depth = depth is not None
    mode_str  = "3D (D455)" if use_depth else "2D (웹캠)"
    print(f"🤖 에이전트 모드: {mode_str}")

    # ── 노드 1: 장면 분석 ────────────────────────
    def analyze_scene(state: AgentState) -> AgentState:
        frame       = state["frame"]
        depth_frame = state.get("depth_frame")
        user_input  = state["user_input"]

        # 질문이면 LLaVA 직접 답변
        if vlm.is_question(user_input):
            state["log"].append("💬 질문 모드")
            answer = vlm.ask(frame, f"한국어로 답해줘: {user_input}")
            state["log"].append(f"💬 {answer}")
            state["task_plan"] = {}
            return state

        state["log"].append(f"🔍 장면 분석 중... [{mode_str}]")

        # 타겟 물체 추출 → YOLO 미리 설정
        target_hint = vlm.extract_target(frame, user_input)
        state["log"].append(f"🔎 타겟 힌트: {target_hint}")
        if target_hint:
            yolo.set_targets([target_hint])

        # YOLO 2D 감지
        detections  = yolo.detect(frame)
        yolo_labels = [d["label"] for d in detections]
        state["detections"] = detections

        # D455 3D 좌표 추가
        detections_3d = []
        if use_depth and depth_frame:
            for det in detections:
                point3d = depth.bbox_to_3d(det["bbox"], depth_frame)
                det_3d  = {**det, "point3d": point3d}
                detections_3d.append(det_3d)
                if point3d["valid"]:
                    state["log"].append(
                        f"   📍 {det['label']}: "
                        f"({point3d['x']:.0f}, {point3d['y']:.0f}, "
                        f"{point3d['z']:.0f})mm  "
                        f"거리: {point3d['dist_cm']:.1f}cm"
                    )
        else:
            detections_3d = [{**d, "point3d": None} for d in detections]

        state["detections_3d"] = detections_3d

        # LLaVA 장면 분석
        scene = vlm.analyze_scene(frame)
        state["scene_info"] = scene

        state["log"].append(f"📸 장면: {scene.get('scene_description','')}")
        state["log"].append(f"🤖 YOLO: {yolo_labels}")
        state["log"].append(f"🧠 LLaVA: {scene.get('graspable',[])}")
        return state

    # ── 노드 2: 태스크 플랜 생성 ─────────────────
    def plan_task(state: AgentState) -> AgentState:
        # 질문이면 플랜 불필요
        if not state.get("scene_info"):
            return state

        frame       = state["frame"]
        user_input  = state["user_input"]
        detections  = state["detections"]
        yolo_labels = [d["label"] for d in detections]

        state["log"].append("📋 태스크 플랜 생성 중...")

        plan = vlm.generate_robot_command(frame, user_input, yolo_labels)
        state["task_plan"]    = plan
        state["current_step"] = 0

        state["log"].append(f"✅ 목표: {plan.get('task_understood','')}")
        state["log"].append(f"🎯 타겟: {plan.get('target_object','')}")
        state["log"].append(f"⚠️  안전: {plan.get('safety_check','')}")

        for i, s in enumerate(plan.get("steps", [])):
            obj = s.get("object", s.get("location", ""))
            state["log"].append(f"  {i+1}. {s.get('action','')} {obj}")

        return state

    # ── 노드 3: 로봇 실행 ─────────────────────────
    def execute_plan(state: AgentState) -> AgentState:
        steps        = state["task_plan"].get("steps", [])
        target       = state["task_plan"].get("target_object", "")
        detections_3d = state.get("detections_3d", [])

        if target:
            yolo.set_targets([target])

        for i, step in enumerate(steps):
            action = step.get("action", "")
            state["log"].append(f"▶ [{i+1}/{len(steps)}] {action}")

            # find
            if action == "find":
                obj    = step.get("object", target)
                result = vlm.find_target(state["frame"], obj)
                found  = result.get("found", False)
                loc    = result.get("location", "")
                state["log"].append(
                    f"  → {'✅ 발견' if found else '❌ 못 찾음'} | {loc}"
                )

            # approach — 3D 좌표 사용
            elif action == "approach":
                dets = yolo.detect(state["frame"])

                if dets:
                    cx, cy = dets[0]["center"]

                    if use_depth and state.get("depth_frame"):
                        # 3D 접근: 실제 거리 기반
                        point3d = depth.bbox_to_3d(
                            dets[0]["bbox"], state["depth_frame"]
                        )
                        if point3d["valid"]:
                            robot.move_to_3d(
                                point3d["x"],
                                point3d["y"],
                                point3d["z"]
                            )
                            state["log"].append(
                                f"  → 3D 접근: "
                                f"({point3d['x']:.0f}, "
                                f"{point3d['y']:.0f}, "
                                f"{point3d['z']:.0f})mm  "
                                f"거리:{point3d['dist_cm']:.1f}cm"
                            )
                        else:
                            state["log"].append("  → 깊이 측정 실패 → 2D 접근")
                            robot.step_toward(cx, cy)
                    else:
                        # 2D 접근: 픽셀 오차 기반
                        robot.step_toward(cx, cy)
                        state["log"].append(
                            f"  → 2D 접근 (cx:{cx:.0f}, cy:{cy:.0f})"
                        )
                else:
                    obj    = step.get("object", target)
                    result = vlm.find_target(state["frame"], obj)
                    state["log"].append(
                        f"  → YOLO 실패, LLaVA: {result.get('location','불명')}"
                    )

            # confirm
            elif action == "confirm":
                secs = step.get("seconds", 2)
                state["log"].append(f"  → {secs}초 확인 중...")
                time.sleep(secs)
                state["log"].append("  → 확인 완료")

            # grasp
            elif action == "grasp":
                # 3D 좌표 있으면 정밀 파지
                if use_depth and detections_3d:
                    best = max(
                        [d for d in detections_3d if d.get("point3d", {}).get("valid")],
                        key=lambda d: d["conf"],
                        default=None
                    )
                    if best:
                        p = best["point3d"]
                        robot.grasp_at_3d(p["x"], p["y"], p["z"])
                        state["log"].append(
                            f"  → 3D 파지: "
                            f"({p['x']:.0f}, {p['y']:.0f}, {p['z']:.0f})mm"
                        )
                    else:
                        ok = robot.grasp()
                        state["log"].append(
                            f"  → {'✅ 파지 완료' if ok else '❌ 파지 실패'}"
                        )
                else:
                    ok = robot.grasp()
                    state["log"].append(
                        f"  → {'✅ 파지 완료' if ok else '❌ 파지 실패'}"
                    )

            # place
            elif action == "place":
                loc = step.get("location", "default")
                ok  = robot.place(loc)
                state["log"].append(
                    f"  → {'✅ 배치 완료' if ok else '❌ 배치 실패'} ({loc})"
                )

            # move_to
            elif action == "move_to":
                from config import PLACE_POSITIONS
                loc = step.get("location", "default")
                pos = PLACE_POSITIONS.get(loc, PLACE_POSITIONS["default"])
                robot.move_to_pose(pos)
                state["log"].append(f"  → {loc}으로 이동")

            # wait
            elif action == "wait":
                secs = step.get("seconds", 1)
                time.sleep(secs)
                state["log"].append(f"  → {secs}초 대기")

            else:
                state["log"].append(f"  → ⚠️ 알 수 없는 액션: {action}")

        state["status"] = "done"
        state["log"].append("🎉 전체 태스크 완료!")
        return state

    # ── 라우터 ────────────────────────────────────
    def router(state: AgentState) -> str:
        if state["task_plan"].get("steps"):
            return "execute"
        return "end"

    # ── 그래프 빌드 ───────────────────────────────
    graph = StateGraph(AgentState)
    graph.add_node("analyze", analyze_scene)
    graph.add_node("plan",    plan_task)
    graph.add_node("execute", execute_plan)

    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "plan")
    graph.add_conditional_edges(
        "plan", router,
        {"execute": "execute", "end": END}
    )
    graph.add_edge("execute", END)

    return graph.compile()