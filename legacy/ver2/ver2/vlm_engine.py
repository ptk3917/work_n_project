import json
import re
import base64
import numpy as np
import cv2
import requests
from config import LLM_MAX_TOKENS

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llava:7b"


class LLaVAEngine:
    def __init__(self):
        print("🔄 LLaVA Ollama 연결 중...")
        print(f"   모델  : {OLLAMA_MODEL}")
        print(f"   서버  : {OLLAMA_URL}")
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            print(f"   설치된 모델: {models}")
        except requests.exceptions.ConnectionError:
            print("\n❌ Ollama 서버 연결 실패 — ollama serve 실행 확인\n")
        print("✅ LLaVA Ollama 준비 완료")

    def _frame_to_base64(self, frame):
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode("utf-8")

    def _parse_json(self, text):
        try:
            clean = re.search(r'\{.*\}', text, re.DOTALL)
            if clean:
                return json.loads(clean.group())
        except json.JSONDecodeError:
            pass
        return {}

    def ask(self, frame, question):
        img_b64 = self._frame_to_base64(frame)
        payload = {
            "model":  OLLAMA_MODEL,
            "prompt": question,
            "images": [img_b64],
            "stream": False,
            "options": {"num_predict": LLM_MAX_TOKENS, "temperature": 0}
        }
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            return "❌ Ollama 서버 연결 실패"
        except Exception as e:
            return f"❌ 에러: {e}"

    def analyze_scene(self, frame):
        question = """Analyze this scene and answer ONLY in JSON format:
{
  "objects": ["list of all visible objects"],
  "graspable": ["objects a robot arm can pick up"],
  "scene_description": "one sentence description",
  "suggested_action": "what a robot should do next"
}"""
        response = self.ask(frame, question)
        result   = self._parse_json(response)
        if not result:
            result = {"objects": [], "graspable": [], "scene_description": response, "suggested_action": "unknown"}
        return result

    def find_target(self, frame, target):
        question = f"""Look for '{target}' in this image.
Answer ONLY in JSON format:
{{
  "found": true or false,
  "location": "where it is (left/center/right + near/far)",
  "confidence": "high or medium or low",
  "description": "brief description"
}}"""
        response = self.ask(frame, question)
        result   = self._parse_json(response)
        if not result:
            result = {"found": False, "location": "unknown", "confidence": "low", "description": ""}
        return result

    def extract_target(self, frame, user_input):
        question = f"""What object does the user want to find?
User said: "{user_input}"
Reply with ONLY the object name in English. One word only.
Examples: cup / bottle / box / shoe"""
        response = self.ask(frame, question)
        return response.strip().strip('."\'').split('\n')[0].lower()

    def is_question(self, user_input):
        keywords = ["뭐", "뭐가", "있어", "있나", "알려줘", "설명",
                    "어디", "어떤", "몇", "언제", "왜",
                    "what", "where", "which", "describe", "tell me", "is there"]
        return any(kw in user_input for kw in keywords)

    def spatial_reasoning(self, frame, query):
        question = f"""Answer this spatial question: {query}
Be specific about positions. Answer in 2-3 sentences in Korean if possible."""
        return self.ask(frame, question)

    def generate_robot_command(self, frame, user_instruction, detected_objects):
        objects_str = ", ".join(detected_objects) if detected_objects else "none"
        question = f"""User instruction: "{user_instruction}"
Objects in scene: {objects_str}

Reply ONLY with JSON. Fill ALL fields with real values:
{{
  "task_understood": "pick up the cup and place it on the table",
  "target_object": "cup",
  "steps": [
    {{"action": "find", "object": "cup"}},
    {{"action": "approach", "object": "cup"}},
    {{"action": "confirm", "seconds": 2}},
    {{"action": "grasp"}},
    {{"action": "place", "location": "table"}}
  ],
  "safety_check": "none"
}}
Now generate for: "{user_instruction}"
Available actions: find, approach, confirm, grasp, place, move_to, wait"""
        response = self.ask(frame, question)
        result   = self._parse_json(response)
        if not result:
            result = {"task_understood": user_instruction, "target_object": "unknown", "steps": [], "safety_check": "parse failed"}
        return result
