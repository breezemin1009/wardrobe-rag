# 보유 옷장을 고려한 RAG 기반 의류 구매 지원 시스템

동국대학교 컴퓨터공학과 조현민의 개인 졸업연구 프로젝트입니다. 새로 살 옷이 **내 옷과 얼마나 조합되는지(C)**, **기존 옷장에 어떤 활용을 더하는지(M)**를 나누어 계산하고 공개 코디 근거와 함께 보여줍니다.

## 직접 구현한 부분

- React Native 앱의 사진 등록·속성 확인 화면과 FastAPI의 분석·평가·설명 API 연결
- ASOS·POG 정제, 양방향 코디 검색 특징, 조합 모형 학습과 옷장 기준 점수 계산
- SQLite에 저장한 평가 결과를 설명 요청과 연결하고, LLM 응답의 의류 정보·비교 내용·인용 출처 검사

## 처리 흐름

`의류 사진 → AI 속성 제안 → 사용자 확인 → 유사 상품·코디 검색 → 조합 모형 → C·M 계산 → 결과 저장 → LLM 근거 설명`

수치 계산은 학습한 모형과 서버 계산식이 수행합니다. RAG는 검색한 상품·코디 기록을 LLM 입력에 연결합니다. 서버가 반환된 비교 내용과 출처를 검사합니다.

## 기술 스택

Python · FastAPI · SQLite · PyTorch · scikit-learn · Sentence Transformers · React Native(Expo) · TypeScript · NVIDIA LLM/VLM API

## 코드 바로 보기

| 내용 | 파일 |
|---|---|
| API와 평가 결과 저장·조회 | [backend/app.py](backend/app.py) |
| 확인한 의류 정보 → 검색·추론 → C·M 집계 | [backend/photo_case_assessment.py](backend/photo_case_assessment.py) |
| 비교 내용·인용 번호 검사 | [backend/generation.py](backend/generation.py) |
| 코디 원문과 의류 속성 대조 | [backend/outfit_facts.py](backend/outfit_facts.py) |
| 양방향 검색 특징 | [backend/thesis_repair_stable_kernel_20261026.py](backend/thesis_repair_stable_kernel_20261026.py) |
| ASOS 신경 모형 학습 | [backend/thesis_multilink_train_20261027.py](backend/thesis_multilink_train_20261027.py) |
| POG 모형 학습 | [backend/thesis_pog_disjoint_train_20261027.py](backend/thesis_pog_disjoint_train_20261027.py) |
| 앱 API 요청과 응답 타입 | [src/services/rag.ts](src/services/rag.ts) |

## 빠른 실행: 데이터 없이 계산 확인

Python 3.12 이상에서 가상환경을 만든 뒤 실행합니다.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-demo.txt
python -m demo.score_demo
python -m unittest discover -s backend/tests -v
```

출력은 후보 A `C=75, M=0`, 후보 B `C=50, M=25`입니다. 공개본의 설명용 입력값을 **원래 앱의 집계 코드**에 전달합니다. 모형 예측·논문 실험을 재현하는 명령과 구분됩니다.

## 앱 화면 실행

Node.js 22 이상이 필요합니다. 원본 앱은 Expo 57을 사용합니다.

```bash
npm ci
# .env.example을 .env로 복사
npm run web
```

보유 옷·구매 후보의 등록과 수정 화면을 확인할 수 있습니다. 실제 검색·평가에는 별도 원자료, 검색 인덱스와 학습 모형이 필요합니다. 서버 의존성은 `requirements.txt`, 서버 진입점은 `backend.app:app`입니다. [데이터 준비 경계](docs/data-and-running.md)를 참고하세요.

## 검증

```bash
npm run typecheck
npm run lint
npm test
python -m unittest discover -s backend/tests -v
```

공개본에는 출처·의류 속성 불일치, 비교 내용 뒤집힘, 같은 코디의 중복 집계, 계산 성질을 확인하는 원본 단위 테스트를 포함했습니다. [구현과 검증 위치](docs/implementation.md)에 연결 지점을 정리했습니다.

## 공개 범위

앱·서버·학습 코드와 직접 생성한 표시용 의류 이미지를 포함합니다. ASOS·POG 원본 자료, 검색 벡터, 학습 가중치, 사용자 사진·저장 데이터, API 키, 학번·서명·논문 스캔은 포함하지 않습니다. 데이터는 원 배포처의 이용 조건에 따라 별도로 준비합니다.

Expo 템플릿의 라이선스 고지는 [licenses/EXPO-TEMPLATE-LICENSE](licenses/EXPO-TEMPLATE-LICENSE)에 보존했습니다. 표시용 이미지의 생성 출처는 [assets/demo/README.md](assets/demo/README.md)에 있습니다.
