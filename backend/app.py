import json
import sqlite3
import time
import uuid
from functools import lru_cache
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI, HTTPException, Query as QueryParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field, ConfigDict, model_validator
from backend.config import DATA, VERSION, WORKSPACE
from backend.retrieval import Retriever
from backend.scoring import evaluate
from backend.generation import explain, generation_status
from backend.vision import propose, VisionFailure
from backend.foreground import extract
from backend.design import DesignProfile,legacy_colour
from backend.design_vision import propose_design
from backend.design_reasoning import rulebook
from backend.robustness import audit_evaluation
from backend.audience import validate_audiences, SCOPE

retriever = None

@asynccontextmanager
async def lifespan(app):
    global retriever
    from backend.index_profile import active_index_path
    retriever = Retriever(active_index_path())
    retriever.model()
    from backend.photo_case_assessment import service
    service().encoder=retriever.model()
    from backend.menswear_catalog import catalog
    catalog()  # Validate the photo-reviewed publication list before serving.
    with sqlite3.connect(DATA / 'runs.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created REAL, kind TEXT, payload TEXT)')
    yield

app = FastAPI(title='Wardrobe Research API', version=VERSION, lifespan=lifespan)
@app.middleware('http')
async def retire_unverified_catalog(request, call_next):
    if request.url.path.startswith('/catalog/'):
        return JSONResponse(status_code=410, content={'detail':'이전 공개 코디 자료는 남성복 범위가 확인되지 않아 현재 추천에서 분리했습니다. /menswear/catalog에서 검증한 상품을 확인해주세요.'})
    return await call_next(request)

@app.get('/menswear/catalog')
def menswear_catalog(category: Literal['상의','하의']|None=None,
                     audience: Literal['mens','unisex']|None=None,
                     query:str=QueryParam(default='',max_length=80),
                     offset:int=QueryParam(default=0,ge=0,le=10000),
                     limit:int=QueryParam(default=12,ge=1,le=24)):
    from backend.menswear_catalog import list_catalog
    return list_catalog(category,audience,query,offset,limit)

@app.get('/menswear/photos/{product_id}')
def menswear_photo(product_id:str):
    from backend.menswear_catalog import photo
    try:return Response(photo(product_id),media_type='image/jpeg',headers={'Cache-Control':'public, max-age=86400'})
    except (ValueError,FileNotFoundError):raise HTTPException(404,'확인된 남성복·공용 상품 사진을 찾지 못했어요.') from None

@app.get('/menswear/items/{product_id}')
def menswear_item(product_id:str):
    import base64, hashlib
    from backend.menswear_catalog import catalog, photo, public_item
    try:
        data=photo(product_id)
        return {**public_item(catalog()[product_id]),'image':'data:image/jpeg;base64,'+base64.b64encode(data).decode(),
                'imageHash':hashlib.sha256(data).hexdigest()}
    except (ValueError,FileNotFoundError):raise HTTPException(404,'확인된 남성복·공용 상품을 찾지 못했어요.') from None

@app.get('/menswear/outfits')
def menswear_outfits(offset:int=QueryParam(default=0,ge=0,le=10000),limit:int=QueryParam(default=6,ge=1,le=12)):
    from backend.menswear_catalog import list_outfits
    return list_outfits(offset,limit)

@lru_cache(maxsize=1)
def menswear_case_engine():
    from backend.evidence_history_case_utility import EvidenceHistoryCaseUtility
    return EvidenceHistoryCaseUtility()

class MenswearCaseQuery(BaseModel):
    model_config=ConfigDict(extra='forbid')
    candidates:list[str]=Field(min_length=1,max_length=12)
    contexts:list[str]=Field(min_length=1,max_length=64)
    owned:list[str]=Field(default_factory=list,max_length=64)
    @model_validator(mode='after')
    def verified_identifiers(self):
        import re
        if any(not re.fullmatch('[a-f0-9]{28,32}',p) for p in self.candidates+self.contexts+self.owned):
            raise ValueError('확인된 남성복 원자료의 상품을 선택해주세요.')
        return self

@app.get('/menswear/case-study')
def menswear_case_summary():
    try:return menswear_case_engine().summary()
    except (FileNotFoundError,ValueError):raise HTTPException(503,'검증한 남성복 코디 검색 자료를 준비해주세요.') from None

@app.post('/menswear/case-utility')
def menswear_case_assess(request:MenswearCaseQuery):
    try:return menswear_case_engine().assess(request.candidates,request.contexts,request.owned)
    except FileNotFoundError:raise HTTPException(503,'남성복 코디 검색 자료를 준비해주세요.') from None
    except ValueError as exc:raise HTTPException(422,str(exc)) from None

@app.get('/menswear/case-examples')
def menswear_case_examples():
    try:return menswear_case_engine().examples()
    except (FileNotFoundError,ValueError):raise HTTPException(503,'남성복 비교 예시를 준비해주세요.') from None

@app.post('/menswear/case-explain')
def menswear_case_explain(request:MenswearCaseQuery):
    from backend.menswear_case_explanation import explain_case
    try:return explain_case(menswear_case_engine(),request)
    except FileNotFoundError:raise HTTPException(503,'남성복 코디 검색 자료를 준비해주세요.') from None
    except ValueError as exc:raise HTTPException(422,str(exc)) from None

app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:8081', 'http://127.0.0.1:8081', 'http://localhost:8082', 'http://127.0.0.1:8082'],
                   allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])

class CatalogCoverageQuery(BaseModel):
    model_config = ConfigDict(extra='forbid')
    candidates: list[str] = Field(min_length=1,max_length=12)
    contexts: list[str] = Field(min_length=1,max_length=64)
    owned: list[str] = Field(default_factory=list,max_length=64)
    @model_validator(mode='after')
    def bounded_identifiers(self):
        if any(not pid.isascii() or not pid.isdigit() or len(pid)>20 for pid in [*self.candidates,*self.contexts,*self.owned]):
            raise ValueError('확인된 카탈로그 상품 ID를 입력해주세요.')
        return self

@lru_cache(maxsize=1)
def catalog_engine():
    from backend.catalog_coverage import CatalogCoverage
    return CatalogCoverage.from_source(WORKSPACE/'tmp/open-fashion-sources-20261020/iqon3000-gpbpr')

@app.post('/catalog/coverage')
def catalog_recorded_coverage(request: CatalogCoverageQuery):
    """Confirmed public-record accounting; separate from /assess photo attributes."""
    try:
        rows=catalog_engine().rank(request.candidates,request.contexts,request.owned)
    except (FileNotFoundError,sqlite3.Error):
        raise HTTPException(503,'공개 카탈로그 자료를 준비한 뒤 사용할 수 있습니다.') from None
    except ValueError:
        raise HTTPException(422,'카탈로그 상품 ID와 상·하의 구분을 확인해주세요. 보유 대안은 구매 후보와 같은 종류로 지정해주세요.') from None
    return {'mode':'identified-catalog-recorded-coverage-v1','results':rows,
            'meaning':'확인된 공개 코디 기록에서 새로 지원하는 조합 수를 우선하고, 전체 지원 조합 수로 동점을 구분합니다.',
            'inputRequirement':'공개 카탈로그에서 상품이 확인된 의류를 사용합니다.','aestheticProbability':False}

@lru_cache(maxsize=1)
def fashion_utility_engine():
    from backend.fashion_utility import FashionUtility
    return FashionUtility(DATA/'verified-utility-v30')

@app.post('/catalog/utility')
def catalog_purchase_utility(request: CatalogCoverageQuery):
    try:
        return fashion_utility_engine().assess(request.candidates,request.contexts,request.owned)
    except (FileNotFoundError,sqlite3.Error,OSError):
        raise HTTPException(503,'사진 코디 검색 자료를 준비한 뒤 사용할 수 있습니다.') from None
    except ValueError as exc:
        # Typed request errors and a changed frozen deployment have different causes.
        if str(exc).startswith(('Validated','Verified, frozen')):
            raise HTTPException(503,'검증한 사진 검색 자료의 버전을 확인해주세요.') from None
        raise HTTPException(422,'확인된 하의 후보와 같은 종류의 보유 하의, 비교할 보유 상의를 선택해주세요.') from None

@app.get('/catalog/examples')
def catalog_comparison_examples():
    from backend.catalog_media import examples
    try:return examples()
    except (FileNotFoundError,OSError,sqlite3.Error,ValueError):
        raise HTTPException(503,'공개 코디 예시 자료를 준비한 뒤 사용할 수 있습니다.') from None

@app.get('/catalog/items/{product_id}')
def catalog_product(product_id:str):
    from backend.catalog_media import item
    try:return item(product_id)
    except ValueError:raise HTTPException(404,'공개 상품을 찾지 못했습니다.') from None
    except (FileNotFoundError,OSError,sqlite3.Error):
        raise HTTPException(503,'공개 상품 자료를 준비한 뒤 사용할 수 있습니다.') from None

@app.get('/catalog/photos/{product_id}')
def catalog_product_photo(product_id:str):
    from backend.catalog_media import photo
    try:return Response(photo(product_id),media_type='image/jpeg',headers={'Cache-Control':'public, max-age=86400'})
    except ValueError:raise HTTPException(404,'공개 상품 사진을 찾지 못했습니다.') from None
    except (FileNotFoundError,OSError,sqlite3.Error):
        raise HTTPException(503,'공개 상품 사진을 준비한 뒤 사용할 수 있습니다.') from None

class Query(BaseModel):
    model_config = ConfigDict(extra='forbid')
    query: str = Field(min_length=2, max_length=500)
    collection: Literal['research', 'style'] = 'research'
    mode: Literal['keyword', 'vector', 'hybrid', 'conditioned'] = 'hybrid'
    limit: int = Field(default=5, ge=1, le=10)
    includeHistorical: bool = False
    @model_validator(mode='after')
    def nonblank(self):
        if len(self.query.strip()) < 2:
            raise ValueError('검색어를 입력해주세요.')
        return self

class Garment(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=60)
    category: Literal['상의', '하의']
    color: Literal['화이트', '크림', '블루', '블랙', '베이지', '그레이', '올리브', '기타']
    pattern: Literal['무지', '스트라이프', '체크', '프린트', '기타']
    occasion: Literal['데일리', '출근', '나들이']
    role: Literal['owned', 'candidate']
    audience: Literal['mens', 'unisex', 'unknown', 'womens'] = 'unknown'
    sourceProductId: str|None = Field(default=None,pattern=r'^\d{1,10}$')
    referenceProductId: str|None = Field(default=None,pattern=r'^[a-f0-9]{32}$')
    design: DesignProfile | None = None
    @model_validator(mode='after')
    def aligned_design(self):
        if self.sourceProductId:
            from backend.menswear_catalog import catalog
            product=catalog().get(self.sourceProductId)
            if not product or product['audience']!=self.audience:
                raise ValueError('원본 상품의 남성복·공용 표기를 확인해주세요.')
        if self.design and (legacy_colour(self.design.front.baseColor)!=self.color or self.design.front.pattern!=self.pattern):
            raise ValueError('기본 속성과 앞면 디자인을 함께 확인해주세요.')
        if self.design:
            kind=self.design.garmentType
            if (kind in {'tshirt','shirt','knit','jacket'} and self.category!='상의') or (kind in {'jeans','trousers','shorts','skirt'} and self.category!='하의'):
                raise ValueError('세부 종류와 상·하의 구분을 함께 확인해주세요.')
        return self

class Assessment(BaseModel):
    model_config = ConfigDict(extra='forbid')
    candidate: Garment
    wardrobe: list[Garment] = Field(max_length=64)
    occasion: Literal['데일리', '출근', '나들이'] = '데일리'
    mode: Literal['keyword', 'vector', 'hybrid'] = 'hybrid'
    @model_validator(mode='after')
    def validate_roles(self):
        validate_audiences(self.candidate.model_dump(), [g.model_dump() for g in self.wardrobe])
        if self.candidate.role != 'candidate' or any(g.role != 'owned' for g in self.wardrobe):
            raise ValueError('구매 후보와 보유 옷을 구분해주세요.')
        if len({g.id for g in self.wardrobe}) != len(self.wardrobe) or self.candidate.id in {g.id for g in self.wardrobe}:
            raise ValueError('의류 ID가 중복되었습니다.')
        return self

class Ranking(BaseModel):
    model_config = ConfigDict(extra='forbid')
    candidates: list[Garment] = Field(min_length=1, max_length=12)
    wardrobe: list[Garment] = Field(max_length=64)
    occasion: Literal['데일리', '출근', '나들이'] = '데일리'
    @model_validator(mode='after')
    def validate_roles(self):
        for candidate in self.candidates:
            validate_audiences(candidate.model_dump(), [g.model_dump() for g in self.wardrobe])
        ids = [g.id for g in self.wardrobe]
        candidates = [g.id for g in self.candidates]
        if any(g.role != 'owned' for g in self.wardrobe) or any(g.role != 'candidate' for g in self.candidates):
            raise ValueError('보유 의류와 구매 후보를 구분해주세요.')
        if len(set(ids)) != len(ids) or len(set(candidates)) != len(candidates) or set(ids) & set(candidates):
            raise ValueError('의류 ID가 중복되었습니다.')
        return self

class Explanation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    runId: str = Field(pattern=r'^[a-f0-9]{32}$')

class PhotoRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    image: str = Field(min_length=30, max_length=4_000_000)
    consent: Literal[True]

class DesignPhotoRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    front:str=Field(min_length=30,max_length=4_000_000)
    back:str|None=Field(default=None,min_length=30,max_length=4_000_000)
    consent:Literal[True]

@app.post('/photo-design')
def photo_design(request:DesignPhotoRequest):
    try:return propose_design(request.front,request.back)
    except VisionFailure as exc:raise HTTPException(exc.status,str(exc)) from None

@app.post('/photo-attributes')
def photo_attributes(request: PhotoRequest):
    try:
        return propose(request.image)
    except VisionFailure as exc:
        raise HTTPException(exc.status, str(exc)) from None

@app.post('/photo-cutout')
def photo_cutout(request: PhotoRequest):
    try:
        return {k:v for k,v in extract(request.image).items() if k!='analysisImage'}
    except VisionFailure as exc:
        raise HTTPException(exc.status,str(exc)) from None
    except (RuntimeError, OSError):
        raise HTTPException(503,'배경 분리를 완료하지 못했어요. 원본 사진을 유지하고 다시 시도해주세요.') from None

@app.get('/experiments')
def experiments():
    path = DATA / 'menswear-rebuild-v1/validation-report.json'
    if not path.exists():
        raise HTTPException(404, '아직 실행한 비교 검사가 없습니다.')
    report = json.loads(path.read_text(encoding='utf-8'))
    import hashlib
    if report.get('catalogReviewSha256')!=hashlib.sha256((DATA/'menswear-rebuild-v1/catalog-review.json').read_bytes()).hexdigest():
        raise HTTPException(409,'공개 상품 사진의 검토 목록이 바뀌었습니다. 현재 자료의 검사를 다시 실행해주세요.')
    if report['corpusSha256'] != retriever.index['chunksSha256'] or report['version'] != VERSION or report.get('designRulesSha256')!=rulebook()[1]:
        raise HTTPException(409, '자료 또는 계산 버전이 변경됐습니다. 비교 검사를 다시 실행해주세요.')
    return report

def public_evidence(evidence):
    return [{k: v for k, v in item.items() if k != 'path'} | {'text': item['text'][:1800]} for item in evidence]

def audit(kind, result):
    run_id = uuid.uuid4().hex
    snapshot = {**result, 'runId': run_id, 'version': VERSION, 'indexRevision': retriever.index['revision'],
                'corpusSha256': retriever.index['chunksSha256']}
    with sqlite3.connect(DATA / 'runs.sqlite3') as db:
        db.execute('INSERT INTO runs VALUES (?, ?, ?, ?)', (run_id, time.time(), kind, json.dumps(snapshot, ensure_ascii=False)))
    return snapshot

@app.get('/health')
def health():
    from backend.photo_case_assessment import service
    active=service()
    return {'status': 'ready', 'version': VERSION, 'sources': len(retriever.manifest['sources']), 'chunks': len(retriever.chunks),
            'corpusSha256': retriever.index['chunksSha256'],
            'assessmentModel':active.utility.version,'assessmentSource':'POG',
            'assessmentModelSha256':active.utility.profile['modelSha256'],
            'assessmentReferencePairs':len(active.utility.confirmed),
            'designRulesVersion':rulebook()[0]['version'],'designRulesSha256':rulebook()[1],
            'styleEvidence': sum(c['collection'] == 'style' for c in retriever.chunks),
            'encoder': retriever.index['model'], 'revision': retriever.index['revision'],
            **generation_status(), 'scope': SCOPE}

@app.get('/sources')
def sources():
    return {'sources': [{k: v for k, v in s.items() if k != 'path'} for s in retriever.manifest['sources']],
            'limitations': retriever.manifest['limitations'], 'skipped': [{'title': s.get('title', s.get('path', '').split('/')[-1]), 'error': s['error']} for s in retriever.manifest['skipped']]}

@app.post('/search')
def search(request: Query):
    start = time.perf_counter()
    records = public_evidence(retriever.search(request.query, request.collection, request.mode, request.limit, include_historical=request.includeHistorical))
    return {'query': request.query, 'mode': request.mode, 'collection': request.collection, 'evidence': records,
            'elapsedMs': round((time.perf_counter()-start)*1000), 'indexRevision': retriever.index['revision']}

@app.post('/ask')
def ask(request: Query):
    result = search(request)
    if request.collection == 'research':
        from backend.research_context import with_research_context
        result['evidence'] = public_evidence(with_research_context(result['evidence'],retriever.chunks,request.limit))
    return audit('question', {**result, 'answer': explain(request.query, result['evidence'])})

@app.post('/assess')
def assess(request: Assessment):
    from backend.photo_case_assessment import service
    try:
        if any(g.design and g.design.garmentType in {'skirt','other'} for g in [request.candidate]+request.wardrobe):
            raise ValueError('검색에 사용할 의류의 세부 종류를 확인해주세요.')
        utility=service()
        if utility.encoder is None:utility.encoder=retriever.model()
        result = utility.assess(request.candidate.model_dump(), [g.model_dump() for g in request.wardrobe], request.occasion, request.mode)
        candidate_ref=utility.utility.canonical.get(request.candidate.referenceProductId)
        if candidate_ref and any(utility.utility.canonical.get(g.referenceProductId)==candidate_ref for g in request.wardrobe):
            result['score']=result['contribution']=0.
            for group in result['groups']:
                group['gain']=0.;group['existingSupport']=max(group['existingSupport'],group['support'])
            result['alreadyOwned']=True
        result['rankingScore']=100*sum(g['gain'] for g in result['groups'])/len(result['groups'])
        result['method']['supportModelSha256']=utility.utility.profile['modelSha256']
        result['method']['selectionSha256']=utility.utility.report['selectionSha256']
        result['method']['referencePairs']=len(utility.utility.confirmed)
        result['method']['referenceSource']='POG'
        result['method']['inputEncoderRevision']=retriever.index['revision']
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from None
    result['evidence'] = public_evidence(result['evidence'])
    result['input'] = request.model_dump()
    return audit('assessment', result)

@app.get('/runs/{run_id}')
def run_snapshot(run_id: str):
    with sqlite3.connect(DATA / 'runs.sqlite3') as db:
        row = db.execute('SELECT payload FROM runs WHERE id = ?', (run_id,)).fetchone()
    if not row:
        raise HTTPException(404, '실행 기록을 찾지 못했습니다.')
    snapshot=json.loads(row[0])
    if snapshot.get('version') != VERSION:
        raise HTTPException(409, '현재 자료로 다시 평가해주세요. 이전 버전 기록은 보존되어 있어요.')
    return snapshot

@app.post('/explain')
def explain_assessment(request: Explanation):
    snapshot = run_snapshot(request.runId)
    if snapshot.get('version') != VERSION or snapshot.get('corpusSha256') != retriever.index['chunksSha256']:
        raise HTTPException(409, '현재 자료로 다시 평가한 후 설명을 요청해주세요.')
    if 'groups' not in snapshot:
        raise HTTPException(422, '의류 평가 기록을 선택해주세요.')
    candidate = snapshot['input']['candidate']
    occasion = snapshot['input'].get('occasion', '데일리')
    query = f"{candidate['color']} {candidate['pattern']} {candidate['category']} 구매 후보의 조합 지원과 기존 하의 대비 추가 기여를 고정 평가에 맞춰 설명해주세요."
    answer = explain(query, snapshot['evidence'], snapshot)
    return {'runId': request.runId, 'evaluationFingerprint': snapshot.get('evaluationFingerprint'), 'answer': answer}

@app.get('/runs/{run_id}/robustness')
def assessment_robustness(run_id: str):
    snapshot = run_snapshot(run_id)
    if snapshot.get('assessmentScope')=='public-case':
        raise HTTPException(422,'공개 코디 모형의 검증은 해당 버전의 실험 보고서에서 확인할 수 있습니다.')
    if 'groups' not in snapshot:
        raise HTTPException(422, '의류 평가 기록을 선택해주세요.')
    try:
        return audit_evaluation(snapshot, retriever)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None

@app.post('/rank')
def rank(request: Ranking):
    if len({g.id for g in request.candidates}) != len(request.candidates):
        raise HTTPException(422, '후보 ID가 중복되었습니다.')
    results = [assess(Assessment(candidate=c, wardrobe=request.wardrobe, occasion=request.occasion)) for c in request.candidates]
    # Compare scores only when the computed group set is identical. Partial results are not ranked.
    scored = [r for r in results if r['score'] is not None]
    masks = {(r['assessmentScope'],tuple(g['groupId'] for g in r['groups'] if g['gain'] is not None)) for r in scored}
    comparable = len(scored)==len(results) and len(masks)==1
    results.sort(key=lambda r: (r['score'] is None, -r.get('rankingScore',r['score'] or 0), r['candidateId']) if comparable else (0, 0, r['candidateId']))
    previous=None;current_rank=None;eligible=0
    for result in results:
        raw=result.get('rankingScore',result['score'])
        result['rank']=None
        if comparable and not result.get('alreadyOwned'):
            eligible+=1
            if previous is None or abs(raw-previous)>1e-8:current_rank=eligible
            result['rank']=current_rank;previous=raw
    return {'results': results, 'comparable': comparable, 'message': '모든 후보가 같은 근거·디자인 범위에서 계산됐을 때만 순위를 표시하며, 같은 점수는 같은 순위입니다.'}

@app.get('/runs')
def runs():
    with sqlite3.connect(DATA / 'runs.sqlite3') as db:
        rows = db.execute('SELECT id,created,kind,payload FROM runs WHERE json_extract(payload,\'$.version\') = ? AND json_extract(payload,\'$.corpusSha256\') = ? ORDER BY created DESC LIMIT 20', (VERSION,retriever.index['chunksSha256'])).fetchall()
    return {'runs': [{'id': i, 'created': t, 'kind': k, 'candidateId': (p := json.loads(payload)).get('candidateId'),
                     'candidateName': p.get('input', {}).get('candidate', {}).get('name'), 'query': p.get('query'),
                     'status': p.get('status'), 'score': p.get('score'), 'version': p['version']} for i, t, k, payload in rows]}
