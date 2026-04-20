import time
import numpy as np
import cv2
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END


# ── 에이전트 상태 정의 ─────────────────────────────
class AgentState(TypedDict):
    user_input:   str
    frame:        Optional[np.ndarray]  # 현재 카메라 프레임
    detections:   list                  # YOLO 감지 결과
    scene_info:   dict                  # LLaVA 장면 분석 결과
    task_plan:    dict                  # 로봇 액션 플랜
    current_step: int
    status:       str                   # idle / running / done / error
    log:          List[str]


# ── 에이전트 그래프 빌드 ───────────────────────────
def build_agent(vlm, yolo, robot):
    """
    vlm   : LLaVAEngine
    yolo  : YOLODetector
    robot : FR5Controller
    """

    # ── 노드 1: 장면 분석 ────────────────────────
    def analyze_scene(state: AgentState) -> AgentState:
        frame      = state["frame"]
        user_input = state["user_input"]
    
        state["log"].append("🔍 장면 분석 중...")
    
        # ── 타겟 물체 이름 먼저 뽑기 ──────────────────
        # LLaVA한테 "어떤 물체를 찾아야 해?" 먼저 물어봄
        target_q = f"""
    What specific object does the user want to find or pick up?
    User said: "{user_input}"
    Reply with just the object name in English, one word or short phrase.
    Example: "cup" or "red bottle" or "blue box"
    """
        target_hint = vlm.ask(frame, target_q).strip().lower()
        # 따옴표, 마침표 제거
        target_hint = target_hint.strip('."\'').split('\n')[0]
        state["log"].append(f"🔎 타겟 힌트: {target_hint}")
    
        # YOLO 타겟 미리 설정 후 감지
        if target_hint:
            yolo.set_targets([target_hint])
    
        detections  = yolo.detect(frame)
        yolo_labels = [d["label"] for d in detections]
        state["detections"] = detections
    
        # LLaVA 장면 분석
        scene = vlm.analyze_scene(frame)
        state["scene_info"] = scene
    
        state["log"].append(f"📸 장면: {scene.get('scene_description', '')}")
        state["log"].append(f"🤖 YOLO 감지: {yolo_labels}")
        state["log"].append(f"🧠 LLaVA 파악: {scene.get('graspable', [])}")
        return state

    # ── 노드 2: 태스크 플랜 생성 ─────────────────
    def plan_task(state: AgentState) -> AgentState:
        frame       = state["frame"]
        user_input  = state["user_input"]
        detections  = state["detections"]
        yolo_labels = [d["label"] for d in detections]

        state["log"].append("📋 태스크 플랜 생성 중...")

        plan = vlm.generate_robot_command(frame, user_input, yolo_labels)
        state["task_plan"]    = plan
        state["current_step"] = 0

        state["log"].append(f"✅ 목표: {plan.get('task_understood', '')}")
        state["log"].append(f"🎯 타겟: {plan.get('target_object', '')}")
        state["log"].append(f"⚠️  안전: {plan.get('safety_check', '')}")

        steps = plan.get("steps", [])
        for i, s in enumerate(steps):
            obj = s.get("object", s.get("location", ""))
            state["log"].append(
                f"  {i+1}. {s.get('action', '')} {obj}"
            )

        return state

    # ── 노드 3: 로봇 실행 ─────────────────────────
    def execute_plan(state: AgentState) -> AgentState:
        steps  = state["task_plan"].get("steps", [])
        target = state["task_plan"].get("target_object", "")

        # YOLO 탐지 대상 설정
        if target:
            yolo.set_targets([target])

        for i, step in enumerate(steps):
            action = step.get("action", "")
            state["log"].append(
                f"▶ [{i+1}/{len(steps)}] {action}"
            )

            # find: LLaVA로 물체 위치 확인
            if action == "find":
                obj    = step.get("object", target)
                result = vlm.find_target(state["frame"], obj)
                found  = result.get("found", False)
                loc    = result.get("location", "")
                state["log"].append(
                    f"  → {'✅ 발견' if found else '❌ 못 찾음'} | {loc}"
                )

            # approach: 물체 중심으로 미세 이동
            elif action == "approach":
                # 저장 프레임 말고 실시간으로 다시 감지
                dets = yolo.detect(state["frame"])

                if dets:
                    cx, cy = dets[0]["center"]
                    robot.step_toward(cx, cy)
                    state["log"].append(
                        f"  → 접근 중 (cx:{cx:.0f}, cy:{cy:.0f})"
                    )
                else:
                    # YOLO 실패 시 LLaVA 위치 정보로 대체
                    obj    = step.get("object", target)
                    result = vlm.find_target(state["frame"], obj)
                    state["log"].append(
                        f"  → YOLO 실패, LLaVA 위치: {result.get('location', '불명')}"
                    )

            # confirm: N초 동안 물체 확인
            elif action == "confirm":
                secs = step.get("seconds", 2)
                state["log"].append(f"  → {secs}초 확인 중...")
                time.sleep(secs)
                state["log"].append("  → 확인 완료")

            # grasp: 그리퍼 닫기
            elif action == "grasp":
                ok = robot.grasp()
                state["log"].append(
                    f"  → {'✅ 파지 완료' if ok else '❌ 파지 실패'}"
                )

            # place: 지정 위치에 놓기
            elif action == "place":
                loc = step.get("location", "default")
                ok  = robot.place(loc)
                state["log"].append(
                    f"  → {'✅ 배치 완료' if ok else '❌ 배치 실패'} ({loc})"
                )

            # move_to: 특정 위치로 이동
            elif action == "move_to":
                from config import PLACE_POSITIONS
                loc = step.get("location", "default")
                pos = PLACE_POSITIONS.get(loc, PLACE_POSITIONS["default"])
                robot.move_to_pose(pos)
                state["log"].append(f"  → {loc}으로 이동 완료")

            # wait: 대기
            elif action == "wait":
                secs = step.get("seconds", 1)
                time.sleep(secs)
                state["log"].append(f"  → {secs}초 대기 완료")

            else:
                state["log"].append(f"  → ⚠️ 알 수 없는 액션: {action} 스킵")

        state["status"] = "done"
        state["log"].append("🎉 전체 태스크 완료!")
        return state

    # ── 라우터: 플랜 있으면 실행, 없으면 종료 ────
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
        "plan",
        router,
        {"execute": "execute", "end": END}
    )
    graph.add_edge("execute", END)

    return graph.compile()
