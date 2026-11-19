# 구현과 검증 위치

## 평가와 설명의 연결

`backend/app.py`의 `/assess`는 확인한 의류 정보를 받아 점수를 계산한 뒤 결과와 입력을 SQLite의 `runs`에 저장합니다. `/explain`은 같은 `runId`를 조회해 저장된 점수·검색 근거를 사용합니다. 인덱스 버전과 자료 해시도 확인합니다.

`backend/generation.py`는 근거마다 인용 번호를 배정하고, 반환된 의류 정보와 비교 관계가 계산 결과와 일치하는지 검사합니다. LLM 호출에는 시간 제한과 제한된 재시도가 있습니다.

## 검색과 모형

`backend/photo_case_assessment.py`는 사용자가 확정한 종류·색상·무늬를 검색 표현으로 만듭니다. 인코더는 외부 사전학습 모델을 사용하고, 조합 지원 모형은 공개 코디의 상하의 관계로 학습했습니다. 원자료 상품 ID와 사진에서 만든 임시 질의를 구분합니다.

`thesis_repair_stable_kernel_20261026.py`의 검색 특징과 `thesis_multilink_train_20261027.py`, `thesis_pog_disjoint_train_20261027.py`의 학습 코드는 원본 그대로 보존했습니다. 서버는 두 의류의 조합 지원값을 옷장별로 모아 C와 M을 계산합니다.

## 원본 테스트

- `test_outfit_facts.py`: 다른 코디의 출처나 원문에 없는 색상·무늬를 사용하는 응답 거부
- `test_explanation_plans.py`: 반대로 설명한 문장·후보 혼동 검사와 구조화된 응답 처리
- `test_photo_case_assessment.py`: 확인 속성의 표현, 상하의 구분, 코디 중복 제거와 점수 성질
- `tests/*.test.mjs`: 의류 상태 변경, 저장·등록 정보와 계산 입력 구성

원래 연구의 데이터 기반 검증과 이 저장소의 데이터 없는 단위 테스트는 실행 입력이 다릅니다. 공개 실행은 README의 명령으로 확인할 수 있습니다.
