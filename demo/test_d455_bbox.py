"""
D455 실시간 캡처 + Qwen2.5-VL bbox 추출 테스트 (수정본)
실행: python test_d455_bbox.py
"""
import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image, ImageDraw
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import re
import os


# ── 1. 모델 로드 ─────────────────────────────────────────────────
print("모델 로드 중...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="auto",
    max_memory={0: "14GiB", "cpu": "16GiB"}
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
print(f"✅ 모델 로드 완료 | VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")


# ── 2. D455 초기화 ───────────────────────────────────────────────
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

print("✅ D455 초기화 완료")
print("카메라 안정화 대기 중...")
for _ in range(30):
    pipeline.wait_for_frames()
print("준비 완료!")


# ── 3. bbox 추출 함수 ────────────────────────────────────────────
def extract_bbox(image: Image.Image, target: str) -> dict:
    w, h = image.size

    prompt = f"""Detect '{target}' in the image and return bounding box as JSON.
Image size: {w}x{h} pixels.

Output ONLY this JSON, no other text:
{{"found": true, "label": "object name", "bbox": [x_min, y_min, x_max, y_max], "confidence": "high/medium/low"}}

If not found:
{{"found": false, "label": "", "bbox": [], "confidence": "none"}}

Important: detect any cup, mug, or drinking vessel even if partially visible."""

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": prompt}
    ]}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    generated = out[:, inputs.input_ids.shape[1]:]
    raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
    print(f"VLM 원본 출력: {raw}")

    # ── 파싱: bbox_2d 리스트 형식 + dict 형식 둘 다 처리 ──
    raw_clean = re.sub(r"```json|```", "", raw).strip()

    # JSON 불완전한 따옴표 보정
    raw_clean = re.sub(r'"(\w+)}', r'"\1"}', raw_clean)

    # bbox_2d 또는 bbox 값 직접 추출 (파싱 실패해도 동작)
    bbox_match = re.search(r'"bbox(?:_2d)?"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', raw_clean)
    label_match = re.search(r'"label"\s*:\s*"([^"]+)"', raw_clean)
    conf_match  = re.search(r'"confidence"\s*:\s*"([^"]+)"', raw_clean)

    if bbox_match:
        bbox  = [int(bbox_match.group(i)) for i in range(1, 5)]
        label = label_match.group(1) if label_match else target
        conf  = conf_match.group(1)  if conf_match  else "high"
        return {"found": True, "label": label, "bbox": bbox, "confidence": conf}

    return {"found": False, "bbox": [], "confidence": "none", "label": "parse_error"}
    

# ── 4. bbox 중심점 depth 읽기 ────────────────────────────────────
def get_depth_at_bbox(depth_frame, bbox: list) -> float:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    depths = []
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            px = max(0, min(cx + dx, 639))
            py = max(0, min(cy + dy, 479))
            d  = depth_frame.get_distance(px, py) * 1000
            if d > 0:
                depths.append(d)
    return float(np.median(depths)) if depths else 0.0


# ── 5. 결과 시각화 ───────────────────────────────────────────────
def draw_result(frame_bgr: np.ndarray, result: dict, depth_mm: float) -> np.ndarray:
    vis = frame_bgr.copy()
    if result.get("found") and len(result.get("bbox", [])) == 4:
        x1, y1, x2, y2 = result["bbox"]
        color_map = {"high": (0,255,0), "medium": (0,255,255), "low": (0,0,255)}
        color = color_map.get(result.get("confidence", "high"), (0,255,0))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label = f"{result['label']} [{result['confidence']}] {depth_mm:.0f}mm"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(vis, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.circle(vis, (cx, cy), 5, color, -1)
        cv2.putText(vis, f"({cx},{cy})", (cx+8, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    else:
        cv2.putText(vis, "NOT FOUND", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    return vis


# ── 6. 메인 루프 ─────────────────────────────────────────────────
TARGET = "cup or mug"

print(f"\n스페이스바: '{TARGET}' 탐색 | q: 종료")
print("카메라 프리뷰 창을 클릭 후 키를 누르세요.\n")

last_result = {"found": False, "bbox": [], "confidence": "none", "label": ""}
last_depth  = 0.0

try:
    while True:
        frames      = pipeline.wait_for_frames()
        aligned     = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_np = np.asanyarray(color_frame.get_data())
        rgb_pil  = Image.fromarray(cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB))

        vis = draw_result(color_np, last_result, last_depth)
        cv2.putText(vis, "SPACE: detect | q: quit",
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.imshow("D455 + Qwen2.5-VL", vis)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):
            rgb_pil.save(os.path.expanduser("~/d455_vlm_input.jpg"))
            print(f"'{TARGET}' 탐색 중...")

            result = extract_bbox(rgb_pil, TARGET)
            print(f"파싱 결과: {result}")

            if result.get("found") and len(result.get("bbox", [])) == 4:
                last_depth  = get_depth_at_bbox(depth_frame, result["bbox"])
                last_result = result
                print(f"✅ 감지 성공 | depth: {last_depth:.1f}mm")

                save_path = os.path.expanduser("~/d455_result.jpg")
                cv2.imwrite(save_path, draw_result(color_np, result, last_depth))
                print(f"결과 저장: {save_path}")
            else:
                last_result = result
                last_depth  = 0.0
                print("❌ 감지 실패")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("종료")