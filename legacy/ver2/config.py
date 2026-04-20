# ── 로봇 설정 ──────────────────────────────────────
ROBOT_IP   = "192.168.58.2"   # FR5 컨트롤러 IP (티치펜던트에서 확인)
ROBOT_VEL  = 20               # 이동 속도 (%)
ROBOT_TOOL = 0
ROBOT_USER = 0

# ── 카메라 설정 ────────────────────────────────────
CAM_INDEX    = 0
CAM_W, CAM_H = 640, 480

# ── YOLO 설정 ──────────────────────────────────────
YOLO_MODEL   = "yolov8m-worldv2.pt"
DETECT_CONF  = 0.3

# ── LLaVA 설정 ────────────────────────────────────
LLM_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
LLM_4BIT     = False     # True = 4bit 양자화 (VRAM 절약 약 6GB)
                         # False = fp16 풀 정밀도 (약 14GB)
LLM_MAX_TOKENS = 512

# ── 감지 설정 ──────────────────────────────────────
CONFIRM_SECS        = 2.0    # 물체 확인 대기 시간 (초)
APPROACH_BBOX_RATIO = 0.25   # 화면 면적의 25% 이상이면 충분히 가까운 것

# ── 탐색 웨이포인트 (FR5 조인트 각도) ────────────
SEARCH_WAYPOINTS = [
    [0,   -30, 60, 0, 30, 0],    # 정면
    [30,  -30, 60, 0, 30, 0],    # 오른쪽
    [-30, -30, 60, 0, 30, 0],    # 왼쪽
    [0,   -45, 80, 0, 10, 0],    # 아래쪽
]

# ── 물체 놓을 위치 (로봇 베이스 기준 mm, rx ry rz) ──
PLACE_POSITIONS = {
    "default": [300,   0, 200, 0, 180, 0],
    "table":   [350, 100, 150, 0, 180, 0],
}
