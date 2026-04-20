# 🤖 AIXERA — Natural Language Vision Picking System

> **FR5 로봇팔 + VLM 기반 자연어 명령 픽킹 · 커피 드립 자동화 · OBB 방향 교정**

<p align="center">
  <img src="docs/images/system_overview.png" alt="System Overview" width="720"/>
</p>

<!-- 위 이미지는 실제 시스템 사진으로 교체하세요 -->

## 📋 프로젝트 개요

Fairino FR5 로봇팔과 Intel RealSense D455 Eye-in-Hand 카메라를 활용한 **자연어 기반 비전 픽킹 시스템**입니다.

한국어 음성/텍스트 명령("컵 잡아줘", "오른쪽 병 집어")을 입력하면, VLM(Qwen2.5-VL-7B)이 카메라 영상에서 대상 물체를 감지하고 depth 좌표 변환을 거쳐 로봇이 자율적으로 피킹합니다.

### 핵심 특징

- **자연어 제어** — 복합 수식어("뚜껑이 검정인 병", "왼쪽에서 두 번째") 이해
- **로컬 VLM 추론** — Qwen2.5-VL-7B, 외부 API 의존 없이 RTX 5070 Ti에서 실시간 구동
- **YOLO11 OBB** — 커스텀 학습으로 제품 방향 감지 및 자동 정렬
- **커피 드립 자동화** — 룰베이스 5단계 핸드드립 시퀀스
- **모듈형 아키텍처** — 픽킹, 방향 교정, 드립 자동화를 독립 모듈로 구성

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    사용자 입력 (한국어)                      │
│                  "컵 잡아줘" / "과자 정리해"                  │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  VLM Engine (Qwen2.5-VL-7B-Instruct)                    │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ 프롬프트    │→ │ bbox 감지     │→ │ 3단계 JSON 파싱  │ │
│  │ 엔지니어링  │  │ + 라벨 매칭   │  │ 폴백 처리       │ │
│  └────────────┘  └──────────────┘  └──────────────────┘ │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  좌표 변환 파이프라인                                      │
│  D455 Depth → 카메라 좌표 → Hand-Eye 변환 → 로봇 좌표     │
│  (T_cam_to_ee.npy)         (Z축 선형 보정 적용)           │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  FR5 로봇 제어                                            │
│  MoveJ(HOME) → MoveJ(SCAN) → MoveJ(APPROACH) → 피킹     │
│  wait_done() + get_stable_pose() 로 비블로킹 동기화       │
└──────────────────────────────────────────────────────────┘
```

---

## 📁 프로젝트 구조

```
aixera-vlm-picking/
├── README.md
├── requirements.txt
├── .gitignore
│
├── picking/                     # VLM 픽킹 파이프라인
│   ├── picking_pipeline.py      # 메인 파이프라인 (Qwen2.5-VL)
│   ├── picking_pipeline_fast.py # 속도 최적화 버전
│   └── config.py                # 로봇/카메라 상수 정의
│
├── orientation/                 # OBB 방향 교정
│   ├── calib_ort.py             # 방향 교정 메인 파이프라인
│   ├── obb_data_capture.py      # 학습 데이터 수집 도구
│   ├── obb_train.py             # YOLO11 OBB 학습 스크립트
│   └── obb_fitting.py           # XYZ 캘리브레이션 도구
│
├── coffee_drip/                 # 커피 드립 자동화
│   ├── coffee_drip_test.py      # 드립 시퀀스 테스트 러너
│   ├── drip_only_test.py        # 드립 모션 단독 테스트
│   └── drip_positions.json      # 티칭 포지션 데이터
│
├── teaching/                    # 로봇 티칭 유틸리티
│   ├── robot_teach.py           # 범용 포지션 티칭
│   ├── robot_teach_drip.py      # 드립 전용 티칭
│   └── robot_joystick.py        # 키보드 조이스틱 컨트롤러
│
├── calibration/                 # 캘리브레이션
│   ├── z_calibration.py         # Z축 depth 보정 도구
│   ├── T_cam_to_ee.npy          # Hand-Eye 변환 행렬
│   └── calibration_data.json    # Z 캘리브레이션 데이터
│
├── demo/                        # 데모 & 프로토타입
│   └── barista_demo.py          # AIXERA CAFÉ 바리스타 데모
│
└── docs/                        # 문서 & 미디어
    ├── images/                  # 시스템 사진, 다이어그램
    ├── ARCHITECTURE.md          # 상세 아키텍처 문서
    ├── ROBOT_API_NOTES.md       # Fairino SDK 주의사항
    └── WEEKLY_REPORTS.md        # 주간보고 아카이브
```

---

## 🔧 하드웨어 구성

| 구성요소 | 사양 |
|---------|------|
| **로봇팔** | Fairino FR5 (6-DOF, TCP/IP RPC 제어) |
| **카메라** | Intel RealSense D455 (Eye-in-Hand 장착) |
| **GPU** | NVIDIA RTX 5070 Ti (16GB VRAM) |
| **VLM** | Qwen2.5-VL-7B-Instruct (로컬 추론) |
| **객체감지** | YOLO11 OBB (커스텀 학습) |

---

## 🚀 설치 및 실행

### 사전 요구사항

- Python 3.10+
- NVIDIA GPU (VRAM 12GB 이상)
- Fairino FR5 로봇 + Fairino Python SDK
- Intel RealSense D455 카메라

### 설치

```bash
git clone https://github.com/<YOUR_USERNAME>/aixera-vlm-picking.git
cd aixera-vlm-picking
pip install -r requirements.txt

# 한국어 폰트 (OpenCV GUI용)
sudo apt install fonts-nanum
```

### 실행

```bash
# 1. VLM 픽킹 파이프라인
python picking/picking_pipeline.py

# 2. OBB 방향 교정 (자동 모드)
python orientation/calib_ort.py --auto

# 3. 커피 드립 테스트
python coffee_drip/coffee_drip_test.py

# 4. 바리스타 데모
python demo/barista_demo.py
```

---

## 📊 모듈별 상세

### 1. VLM 픽킹 파이프라인

자연어 명령을 받아 VLM이 카메라 영상에서 물체를 감지하고 로봇이 피킹합니다.

**핵심 구현:**
- 스캔 로직: `SCAN1_J` 포지션에서 Z축 오프셋 `[0, +50, -50]mm`로 다중 촬영
- VLM 출력 파싱: list → dict → regex 3단계 폴백으로 불안정한 JSON 대응
- 비블로킹 동기화: `MoveJ` 후 `wait_done()`(0.4s 프리폴) + `get_stable_pose()`(5회 평균)
- 그리퍼 오프셋 보정: `PICK_OFFSET_X/Y`로 체계적 위치 오차 보정

**VLM 모델 비교 결과:**

| 모델 | VRAM | 자연어 이해 | bbox 정확도 | 비고 |
|------|------|-----------|-----------|------|
| **Qwen2.5-VL-7B** ✅ | 12.7GB | 완전 | 높음 | 현재 채택 |
| Qwen3.5-9B (4-bit) | 4.3GB | 완전 | 미검증 | CPU 오프로드 시 출력 불안정 |
| Florence-2-large | ~1.5GB | 제한적 | 높음 | 복합 수식어 처리 불가 |
| AIXERA API | - | 완전 | 낮음 | 텍스트 기반 좌표 추정 한계 |

### 2. YOLO11 OBB 방향 교정

VLM이 감지하지 못하는 특정 제품을 커스텀 학습된 YOLO11 OBB로 감지하고, 방향을 정렬하여 정해진 슬롯에 배치합니다.

**핵심 구현:**
- J6 회전 기반 피킹 (Rz 사용 금지 — err=112 발생)
- OBB → 그리퍼 각도 오프셋: -90°, 범위 초과 시 180° 플립
- 클래스별 그리퍼 폭 자동 설정 (ohyes=60, freetime=32)
- first/last 슬롯 티칭 후 감지 개수에 따라 자동 보간 배치
- 매 피킹 후 재스캔 (가려진 물체 재감지)

### 3. 커피 드립 자동화

카메라 없이 사전 티칭된 포지션으로 핸드드립 커피를 자동 제조합니다.

**5단계 시퀀스:**

```
1. 드리퍼 배치     → 그리퍼로 드리퍼를 서버 위에 장착
2. 원두 투입       → 원두통에서 드리퍼로 원두 투하 (±5mm 진동)
3. 원형 드립 모션  → Circle SDK 명령으로 반경 축소 반복 드립
4. 드리퍼 제거     → 사용한 드리퍼를 쓰레기통으로 이동
5. 서빙            → 서버에서 컵으로 커피 따르기
```

---

## ⚠️ 주요 기술적 제약 & 해결

| 문제 | 원인 | 해결 |
|------|------|------|
| TCP 좌표 오독 | `MoveJ` 비블로킹 | `wait_done()` + `get_stable_pose()` |
| 장거리 `MoveL` 에러 | 관절 속도 초과 | 장거리는 `MoveJ`, `MoveL`은 단거리만 |
| 그리퍼 미동작 | 초기화 누락 | `ActGripper(1,1)` → 2초 대기 → `MoveGripper` |
| 피킹 각도 err=112 | Rz 기반 계산 | J6 기반 회전으로 전환 |
| Qwen3.5 출력 깨짐 | CPU/GPU dtype 불일치 | 전체 GPU 배치 or Qwen2.5-VL 유지 |
| ctypes 에러 | UDP 타이밍 | `GetActualTCPPose` + `sleep(3)` 대체 |

---

## 🗺️ 로드맵

- [x] VLM 픽킹 파이프라인 구축
- [x] 다중 VLM 모델 비교 분석
- [x] YOLO11 OBB 커스텀 학습 파이프라인
- [x] 제품 방향 교정 자동화
- [x] 커피 드립 Stage 1-2 포지션 티칭
- [x] 키보드 조이스틱 컨트롤러
- [x] Z축 캘리브레이션 도구
- [ ] 커피 드립 Stage 3+ 완성
- [ ] ROS2 + 디지털 저울 연동 (적응형 드립)
- [ ] Isaac Sim 디지털 트윈 구축
- [ ] OpenVLA 파인튜닝 (QLoRA, 100+ 에피소드)
- [ ] 자성 공구 교환기 (MagBot) 통합

---

## 📸 데모

<!-- 실제 영상/GIF로 교체하세요 -->
<!-- ![VLM Picking Demo](docs/images/picking_demo.gif) -->
<!-- ![Coffee Drip Demo](docs/images/drip_demo.gif) -->
<!-- ![OBB Sorting Demo](docs/images/obb_demo.gif) -->

> 🎬 데모 영상은 [YouTube 링크]()에서 확인할 수 있습니다.

---

## 🛠️ 기술 스택

**로봇 제어:** Fairino SDK (Python) · TCP/IP RPC  
**비전:** Intel RealSense D455 · pyrealsense2 · OpenCV  
**AI/ML:** Qwen2.5-VL-7B · YOLO11 OBB · transformers · accelerate  
**GUI:** OpenCV + PIL (NanumGothic 한국어 렌더링)  
**예정:** Isaac Sim · OpenVLA · ROS2

---

## 📄 라이선스

이 프로젝트는 개인 포트폴리오 목적으로 공개되었습니다.  
상업적 사용 시 별도 문의 바랍니다.

---

<p align="center">
  <b>AIXERA</b> — 자연어로 제어하는 로봇 자동화
</p>
