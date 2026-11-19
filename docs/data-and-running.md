# 데이터와 서버 실행

## 코드 공개와 데이터 취득

이 저장소는 원래 연구의 소스 코드 공개본입니다. 자료를 다시 배포할 권한을 별도로 확인하지 않은 원본 상품·코디·사진, 검색 벡터와 학습 가중치는 올리지 않았습니다.

- POG: <https://github.com/wenyuer/POG>
- ASOS: 연구에 사용한 원자료와 전처리 경로는 `backend/thesis_multilink_source_20261027.py`에서 확인할 수 있습니다. 원 자료 취득과 인덱스 생성에는 해당 배포처의 입력 파일이 필요합니다.
- 문장 인코더: `backend/pog_retrieval.py`의 모델 ID와 고정 리비전을 사용합니다.

배포처의 사용 조건을 확인하고 자료를 확보한 환경에서 `backend/data`와 각 연구 모듈의 경로 상수를 연결합니다. 원래 서버는 코디 기록, 검색 인덱스와 선택 모형의 파일 해시를 확인한 뒤 시작합니다. 빈 자료로 실제 점수를 반환하지 않습니다.

## 서버

```bash
python -m pip install -r requirements.txt
# 필요한 원자료·전처리 파일·검색 인덱스·모형을 backend/data에 준비
# 외부 설명 API를 쓸 때만 서버 환경 변수 NVIDIA_API_KEY 설정
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8787
```

`backend/config.py`가 서버 환경 변수를 읽습니다. API 키를 앱의 `EXPO_PUBLIC_*` 변수에 넣지 않습니다. 원래 연구 환경에서는 외부 인코더를 별도로 확보한 후 로컬 캐시에서 불러옵니다.

데이터가 없는 환경에서는 `requirements-demo.txt`와 `demo.score_demo`로 집계·단위 테스트를 실행합니다. 원래 평가 자료와 학습 결과를 사용한 전체 실험 재실행에는 해당 입력 파일이 추가로 필요합니다.
