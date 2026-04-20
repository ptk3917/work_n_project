import tkinter as tk
from tkinter import ttk
import cv2
import threading
import numpy as np
from PIL import Image, ImageTk

from config           import CAM_W, CAM_H
from vlm_engine       import LLaVAEngine
from yolo_detector    import YOLODetector
from agent            import build_agent, AgentState
from robot_controller import FR5Controller

# ────────────────────────────────────────────────────
#  모델 초기화 (순서 중요 — LLaVA가 가장 오래 걸림)
# ────────────────────────────────────────────────────
print("=" * 50)
print("  VLM Robot Agent 시작")
print("=" * 50)

yolo  = YOLODetector()      # 약 1~2초
robot = FR5Controller()     # Mock 모드: 즉시 / 실제: IP 연결
vlm   = LLaVAEngine()       # 약 3~5분 (첫 실행 시 다운로드 포함)
agent = build_agent(vlm, yolo, robot)

print("\n✅ 모든 모델 로딩 완료 — GUI 시작")

# ────────────────────────────────────────────────────
#  공유 상태
# ────────────────────────────────────────────────────
running      = True
latest_frame = None   # 카메라 스레드 → GUI / 에이전트가 공유

# ────────────────────────────────────────────────────
#  GUI 설정
# ────────────────────────────────────────────────────
root = tk.Tk()
root.title("🧠 VLM Robot Agent — LLaVA-1.6 + YOLO-World")
root.configure(bg="#1e1e2e")
root.geometry("1200x720")

FONT_TITLE  = ("Helvetica", 14, "bold")
FONT_NORMAL = ("Helvetica", 11)
FONT_LOG    = ("Courier", 10)

# ── 왼쪽: 웹캠 화면 ───────────────────────────────
left = tk.Frame(root, bg="#1e1e2e")
left.pack(side=tk.LEFT, padx=10, pady=10)

cam_label = tk.Label(left, bg="#1e1e2e")
cam_label.pack()

# 상태 표시
state_var = tk.StringVar(value="● IDLE")
tk.Label(left, textvariable=state_var,
         font=FONT_TITLE, fg="#a6e3a1", bg="#1e1e2e").pack(pady=4)

# ── 빠른 질문 버튼 (LLaVA 직접 질의) ────────────
btn_frame = tk.Frame(left, bg="#1e1e2e")
btn_frame.pack(pady=4)

QUICK_QUESTIONS = [
    ("📸 장면 분석",     "Describe everything you see in this image in detail."),
    ("🤏 집을 수 있는 것", "What objects can a robot arm pick up? List them all."),
    ("📏 공간 관계",     "Describe the spatial relationships between all objects."),
]

def quick_ask(question: str):
    if latest_frame is None:
        log_add("⚠️ 카메라 프레임 없음")
        return

    def run():
        state_var.set("● VLM 분석 중...")
        log_add(f"❓ {question[:40]}...")
        answer = vlm.ask(latest_frame.copy(), question)
        log_add(f"💬 {answer}")
        state_var.set("● IDLE")

    threading.Thread(target=run, daemon=True).start()

for label, q in QUICK_QUESTIONS:
    tk.Button(btn_frame, text=label,
              font=("Helvetica", 10),
              bg="#313244", fg="#cdd6f4",
              relief="flat", cursor="hand2",
              command=lambda q=q: quick_ask(q)
              ).pack(side=tk.LEFT, padx=3)

# ── 오른쪽: 컨트롤 패널 ───────────────────────────
right = tk.Frame(root, bg="#2a2a3e", width=390)
right.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
right.pack_propagate(False)

tk.Label(right, text="🧠  VLM Robot Agent",
         font=FONT_TITLE, fg="#cdd6f4", bg="#2a2a3e").pack(pady=(20, 5))

tk.Label(right,
         text=(
             "예시:\n"
             "  컵을 가져다줘\n"
             "  책상 위에 뭐가 있어?\n"
             "  가장 큰 물체를 집어줘\n"
             "  커피 한잔 타줘"
         ),
         font=("Courier", 9), fg="#6c7086", bg="#2a2a3e",
         justify=tk.LEFT).pack(anchor="w", padx=15, pady=(0, 8))

# 입력창
entry_var = tk.StringVar()
entry = tk.Entry(right, textvariable=entry_var,
                 font=FONT_NORMAL, bg="#313244", fg="#cdd6f4",
                 insertbackground="white", relief="flat", width=30)
entry.pack(padx=15, pady=(0, 8), ipady=8)
entry.focus()

# ── 실행 버튼 ─────────────────────────────────────
def on_execute():
    cmd = entry_var.get().strip()
    if not cmd:
        return
    if latest_frame is None:
        log_add("⚠️ 카메라 프레임 없음")
        return

    log_add(f"🗣️ 입력: {cmd}")
    state_var.set("● 처리 중...")

    def run():
        state = AgentState(
            user_input=cmd,
            frame=latest_frame.copy(),
            detections=[],
            scene_info={},
            task_plan={},
            current_step=0,
            status="running",
            log=[]
        )
        try:
            result = agent.invoke(state)
            for msg in result["log"]:
                log_add(msg)
        except Exception as e:
            log_add(f"❌ 에러: {e}")
        finally:
            state_var.set("● IDLE")

    threading.Thread(target=run, daemon=True).start()

tk.Button(right, text="▶  실행  (Enter)",
          font=FONT_TITLE, bg="#a6e3a1", fg="#1e1e2e",
          relief="flat", cursor="hand2",
          command=on_execute).pack(padx=15, pady=(0, 5), fill=tk.X)

entry.bind("<Return>", lambda e: on_execute())

# ── 중단 버튼 ─────────────────────────────────────
def on_stop():
    state_var.set("● IDLE")
    log_add("⏹ 중단 요청 (현재 스텝 완료 후 정지)")

tk.Button(right, text="⏹  중단",
          font=FONT_NORMAL, bg="#f38ba8", fg="#1e1e2e",
          relief="flat", cursor="hand2",
          command=on_stop).pack(padx=15, pady=(0, 15), fill=tk.X)

ttk.Separator(right, orient="horizontal").pack(fill=tk.X, padx=15, pady=5)

# ── 로그창 ────────────────────────────────────────
tk.Label(right, text="📋  실행 로그",
         font=FONT_TITLE, fg="#cdd6f4", bg="#2a2a3e").pack(
    anchor="w", padx=15, pady=(8, 4))

log_frame = tk.Frame(right, bg="#2a2a3e")
log_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 10))

log_text = tk.Text(log_frame, font=FONT_LOG,
                   bg="#1e1e2e", fg="#a6e3a1",
                   relief="flat", state=tk.DISABLED, wrap=tk.WORD)
sb = tk.Scrollbar(log_frame, command=log_text.yview)
log_text.configure(yscrollcommand=sb.set)
sb.pack(side=tk.RIGHT, fill=tk.Y)
log_text.pack(fill=tk.BOTH, expand=True)

def log_add(msg: str):
    """스레드 안전 로그 추가"""
    def _update():
        log_text.configure(state=tk.NORMAL)
        log_text.insert(tk.END, msg + "\n")
        log_text.see(tk.END)
        log_text.configure(state=tk.DISABLED)
    root.after(0, _update)

# ────────────────────────────────────────────────────
#  카메라 루프 (별도 스레드)
# ────────────────────────────────────────────────────
def camera_loop():
    global latest_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    if not cap.isOpened():
        log_add("❌ 카메라를 열 수 없습니다")
        return

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        latest_frame = frame.copy()

        # GUI에 표시
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img   = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        root.after(0, lambda i=imgtk:
                   cam_label.configure(image=i) or
                   setattr(cam_label, 'image', i))

    cap.release()

threading.Thread(target=camera_loop, daemon=True).start()

# ────────────────────────────────────────────────────
#  종료 처리
# ────────────────────────────────────────────────────
def on_close():
    global running
    running = False
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
log_add("✅ 시스템 준비 완료 — 명령을 입력하세요")
root.mainloop()
