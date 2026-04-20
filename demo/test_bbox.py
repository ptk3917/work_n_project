"""
Qwen2.5-VL bbox 추출 테스트
실행: python test_bbox.py
"""
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image, ImageDraw, ImageFont
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


# ── 2. 테스트 이미지 준비 ────────────────────────────────────────
def make_dummy_image(save_path: str):
    """테스트용 이미지가 없을 때 더미 이미지 생성"""
    img = Image.new("RGB", (640, 480), color=(220, 220, 220))
    draw = ImageDraw.Draw(img)

    # 컵 모양 그리기
    draw.rectangle([200, 160, 400, 360], fill=(180, 100, 50), outline=(100, 60, 20), width=3)
    draw.ellipse([200, 145, 400, 185], fill=(200, 120, 60), outline=(100, 60, 20), width=3)
    draw.ellipse([200, 340, 400, 380], fill=(180, 100, 50), outline=(100, 60, 20), width=3)
    # 손잡이
    draw.arc([380, 220, 450, 310], start=270, end=90, fill=(160, 90, 40), width=8)
    # 라벨
    draw.text((260, 250), "CUP", fill=(255, 255, 255))

    img.save(save_path)
    print(f"더미 이미지 생성: {save_path}")
    return img


# 이미지 경로 설정 (본인 이미지 경로로 바꿔도 됩니다)
IMAGE_PATH = os.path.expanduser("~/test_image.jpg")

if os.path.exists(IMAGE_PATH):
    image = Image.open(IMAGE_PATH).convert("RGB")
    print(f"이미지 로드: {IMAGE_PATH}")
else:
    print(f"이미지 없음 → 더미 이미지 생성")
    image = make_dummy_image(IMAGE_PATH)

w, h = image.size
print(f"이미지 크기: {w}x{h}")


# ── 3. bbox 추출 함수 ────────────────────────────────────────────
def extract_bbox(image: Image.Image, target: str) -> dict:
    prompt = f"""이미지에서 '{target}'을 찾아서 bounding box를 JSON으로 반환하세요.
이미지 크기: {image.width}x{image.height} pixels.

반드시 아래 형식만 출력하세요 (다른 텍스트 없이):
{{"found": true, "label": "객체명", "bbox": [x_min, y_min, x_max, y_max], "confidence": "high/medium/low"}}

없으면:
{{"found": false, "label": "", "bbox": [], "confidence": "none"}}"""

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
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.05,
            do_sample=False
        )

    generated = out[:, inputs.input_ids.shape[1]:]
    raw = processor.batch_decode(generated, skip_special_tokens=True)[0]

    # JSON 파싱
    raw_clean = re.sub(r"```json|```", "", raw).strip()
    match = re.search(r"\{.*\}", raw_clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group()), raw
        except json.JSONDecodeError:
            pass
    return {"found": False, "bbox": [], "confidence": "none", "label": "parse_error"}, raw


# ── 4. 실행 ─────────────────────────────────────────────────────
TARGET = "cup"   # ← 찾고 싶은 물체 이름으로 변경

print(f"\n'{TARGET}' 탐색 중...")
result, raw_output = extract_bbox(image, TARGET)

print(f"\n{'='*40}")
print(f"원본 출력: {raw_output}")
print(f"파싱 결과: {result}")
print(f"{'='*40}")


# ── 5. 결과 시각화 저장 ──────────────────────────────────────────
def visualize(image: Image.Image, result: dict, save_path: str):
    vis = image.copy()
    draw = ImageDraw.Draw(vis)

    if result.get("found") and len(result.get("bbox", [])) == 4:
        x1, y1, x2, y2 = result["bbox"]
        color_map = {"high": "green", "medium": "yellow", "low": "red"}
        color = color_map.get(result.get("confidence", "low"), "red")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        label_text = f"{result['label']} [{result['confidence']}]"
        draw.rectangle([x1, y1-22, x1+len(label_text)*8, y1], fill=color)
        draw.text((x1+2, y1-20), label_text, fill="black")

        cx, cy = (x1+x2)//2, (y1+y2)//2
        draw.ellipse([cx-6, cy-6, cx+6, cy+6], fill=color)
        print(f"중심점: ({cx}, {cy})")
    else:
        draw.text((10, 10), "NOT FOUND", fill="red")
        print("물체를 찾지 못했습니다.")

    vis.save(save_path)
    print(f"결과 이미지 저장: {save_path}")


OUTPUT_PATH = os.path.expanduser("~/test_result.jpg")
visualize(image, result, OUTPUT_PATH)
print("\n✅ 테스트 완료!")