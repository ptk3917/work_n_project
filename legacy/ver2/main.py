import tkinter as tk
from tkinter import ttk
import cv2
import threading
import numpy as np
from PIL import Image, ImageTk

from config           import CAM_W, CAM_H
from depth_camera     import DepthCamera
from vlm_engine       import LLaVAEngine
from yolo_detector    import YOLODetector
from agent            import build_agent, AgentState
from robot_controller import FR5Controller

# ────────────────────────────────────────────────────
#  모델 초기화
# ────────────────────────────────────────────────────
print("=" * 50)
print("  VLM Robot Agent v2 — D455 3D Vision")
print("=" * 50)

yolo  = YOLODetector()
robot = FR5Controller()
depth = DepthCamera()
vlm   = LLaVAEngine()
agent = build_agent(vlm, yolo, robot, depth=depth)

print("\n✅ 모든 모듈 로딩 완료 — GUI 시작")

# ────────────────────────────────────────────────────
#  공유 상태
# ────────────────────────────────────────────────────
running      = True
latest_frame = None
latest_depth = None

# 카메라 표시 크기
CAM_DISP_W = 640
CAM_DISP_H = 480

# ────────────────────────────────────────────────────
#  GUI
# ────────────────────────────────────────────────────
root = tk.Tk()
root.title("🧠 VLM Robot Agent v2 — D455 3D Vision")
root.configure(bg="#1e1e2e")
root.geometry("1700x600")
root.resizable(True, True)

FONT_TITLE  = ("Helvetica", 14, "bold")
FONT_NORMAL = ("Helvetica", 11)
FONT_LOG    = ("Courier", 10)

# ────────────────────────────────────────────────────
#  왼쪽: 카메라 2개 (양옆 배치)
# ────────────────────────────────────────────────────
cam_area = tk.Frame(root, bg="#1e1e2e")
cam_area.pack(side=tk.LEFT, padx=10, pady=10)

# RGB (왼쪽)
rgb_wrap = tk.Frame(cam_area, bg="#1e1e2e")
rgb_wrap.pack(side=tk.LEFT, padx=(0, 8))

tk.Label(rgb_wrap, text="📷  RGB",
         font=("Helvetica", 11, "bold"),
         fg="#89b4fa", bg="#1e1e2e").pack(anchor="w", pady=(0, 4))
cam_label = tk.Label(rgb_wrap, bg="#000000",
                     width=CAM_DISP_W, height=CAM_DISP_H)
cam_label.pack()

# Depth (오른쪽)
depth_wrap = tk.Frame(cam_area, bg="#1e1e2e")
depth_wrap.pack(side=tk.LEFT, padx=(8, 0))

tk.Label(depth_wrap, text="🌊  Depth Map",
         font=("Helvetica", 11, "bold"),
         fg="#f9e2af", bg="#1e1e2e").pack(anchor="w", pady=(0, 4))
depth_label = tk.Label(depth_wrap, bg="#000000",
                       width=CAM_DISP_W, height=CAM_DISP_H)
depth_label.pack()

# ────────────────────────────────────────────────────
#  오른쪽: 컨트롤 패널
# ────────────────────────────────────────────────────
right = tk.Frame(root, bg="#2a2a3e", width=350)
right.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
right.pack_propagate(False)

tk.Label(right, text="🧠  VLM Robot Agent v2",
         font=FONT_TITLE, fg="#cdd6f4", bg="#2a2a3e").pack(pady=(16, 4))

# 상태 표시
state_var = tk.StringVar(value="● IDLE")
tk.Label(right, textvariable=state_var,
         font=("Helvetica", 12, "bold"),
         fg="#a6e3a1", bg="#2a2a3e").pack(pady=(0, 8))

# 힌트 텍스트
tk.Label(right,
         text=(
             "예시:\n"
             "  컵을 가져다줘\n"
             "  신발 어디있어?\n"
             "  가장 큰 물체 집어줘\n"
             "  책상 위에 뭐가 있어?"
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

# ── 빠른 질문 버튼 ────────────────────────────────
QUICK_QS = [
    ("📸 장면",    "Describe everything you see in detail in Korean."),
    ("🤏 집기",    "What objects can a robot arm pick up? List in Korean."),
    ("📏 공간",    "Describe spatial relationships between objects in Korean."),
]

def quick_ask(question):
    if latest_frame is None:
        return
    def run():
        state_var.set("● VLM 분석 중...")
        ans = vlm.ask(latest_frame.copy(), question)
        log_add(f"💬 {ans}")
        state_var.set("● IDLE")
    threading.Thread(target=run, daemon=True).start()

q_frame = tk.Frame(right, bg="#2a2a3e")
q_frame.pack(padx=15, pady=(0, 8), fill=tk.X)
for lbl, q in QUICK_QS:
    tk.Button(q_frame, text=lbl,
              font=("Helvetica", 9),
              bg="#313244", fg="#cdd6f4",
              relief="flat", cursor="hand2",
              command=lambda q=q: quick_ask(q)
              ).pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

# ── 실행/중단 버튼 ────────────────────────────────
def on_execute():
    cmd = entry_var.get().strip()
    if not cmd or latest_frame is None:
        return
    log_add(f"🗣️ 입력: {cmd}")
    state_var.set("● 처리 중...")

    def run():
        state = AgentState(
            user_input=cmd,
            frame=latest_frame.copy(),
            depth_frame=latest_depth,
            detections=[],
            detections_3d=[],
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
            import traceback
            log_add(traceback.format_exc())
        finally:
            state_var.set("● IDLE")

    threading.Thread(target=run, daemon=True).start()

tk.Button(right, text="▶  실행  (Enter)",
          font=FONT_TITLE, bg="#a6e3a1", fg="#1e1e2e",
          relief="flat", cursor="hand2",
          command=on_execute).pack(padx=15, pady=(0, 5), fill=tk.X)

entry.bind("<Return>", lambda e: on_execute())

tk.Button(right, text="⏹  중단",
          font=FONT_NORMAL, bg="#f38ba8", fg="#1e1e2e",
          relief="flat", cursor="hand2",
          command=lambda: state_var.set("● IDLE")
          ).pack(padx=15, pady=(0, 12), fill=tk.X)

ttk.Separator(right, orient="horizontal").pack(fill=tk.X, padx=15, pady=4)

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
    def _f():
        log_text.configure(state=tk.NORMAL)
        log_text.insert(tk.END, msg + "\n")
        log_text.see(tk.END)
        log_text.configure(state=tk.DISABLED)
    root.after(0, _f)

# ────────────────────────────────────────────────────
#  카메라 루프
# ────────────────────────────────────────────────────
def camera_loop():
    global latest_frame, latest_depth

    while running:
        color_frame, depth_frame = depth.get_frames()
        if color_frame is None:
            continue

        latest_frame = color_frame.copy()
        latest_depth = depth_frame

        # RGB → 640x480
        rgb_resized = cv2.resize(color_frame, (CAM_DISP_W, CAM_DISP_H))
        rgb_pil     = Image.fromarray(cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2RGB))
        rgb_imgtk   = ImageTk.PhotoImage(image=rgb_pil)

        # Depth 컬러맵 → 640x480
        depth_color   = depth.colorize_depth(depth_frame)
        depth_resized = cv2.resize(depth_color, (CAM_DISP_W, CAM_DISP_H))
        depth_pil     = Image.fromarray(cv2.cvtColor(depth_resized, cv2.COLOR_BGR2RGB))
        depth_imgtk   = ImageTk.PhotoImage(image=depth_pil)

        root.after(0, lambda r=rgb_imgtk: (
            cam_label.configure(image=r) or
            setattr(cam_label, 'image', r)
        ))
        root.after(0, lambda d=depth_imgtk: (
            depth_label.configure(image=d) or
            setattr(depth_label, 'image', d)
        ))

threading.Thread(target=camera_loop, daemon=True).start()

# ────────────────────────────────────────────────────
#  종료
# ────────────────────────────────────────────────────
def on_close():
    global running
    running = False
    depth.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
log_add("✅ D455 3D Vision 시스템 준비 완료")
log_add("📍 물체 감지 시 3D 좌표(mm)가 함께 표시됩니다")
root.mainloop()