"""User-confirmed descriptors -> frozen POG retrieval/support -> wardrobe utility.

Source identifiers are preserved. New garments get ephemeral query records,
never an invented source-product identity. Source files and model stay immutable.
"""
from functools import lru_cache
from copy import copy
import hashlib,json
import numpy as np
from scipy.sparse import vstack
from threadpoolctl import threadpool_limits
from backend import thesis_repair_stable_kernel_20261026 as kernel
from backend.thesis_repair_expanded_models_20261026 import feature_view
from backend.pog_retrieval import text,MODEL,REVISION

COLOURS={'화이트':'白色','크림':'米白色','블루':'蓝色','블랙':'黑色','베이지':'米色','그레이':'灰色','올리브':'军绿色','레드':'红色','핑크':'粉色','브라운':'棕色','옐로':'黄色','오렌지':'橙色','퍼플':'紫色','그린':'绿色'}
PATTERNS={'무지':'纯色','스트라이프':'条纹','체크':'格子','프린트':'印花'}
KINDS={'tshirt':'tee','shirt':'shirt','knit':'knit','jacket':'jacket','jeans':'jeans','trousers':'trousers','shorts':'shorts'}
KIND_TEXT={'tee':'T恤','shirt':'衬衫','knit':'针织衫','jacket':'外套','jeans':'牛仔裤','trousers':'长裤','shorts':'短裤'}
VERSION='public-case-photo-v2'

def outfit_evidence_id(record):
 # A source pair found for two wardrobe queries is still one public record.
 identity=dict(source=record['source'],top=record['top']['id'],bottom=record['bottom']['id'],
               sourceOutfits=sorted(set(record['sourceOutfits'])))
 return 'case:'+hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:16]

@lru_cache(maxsize=1)
def service():
 from backend.research_pog_disjoint_api import POGDisjointUtility
 return PhotoCaseAssessment(POGDisjointUtility())

class PhotoCaseAssessment:
 def __init__(self,utility):
  self.utility=utility;self.base=utility.engine;self.encoder=None

 def descriptor(self,g):
  d=g.get('design') or {};kind=KINDS.get(d.get('garmentType'))
  if kind is None:
   raise ValueError('의류 정보 수정에서 세부 종류(셔츠·니트·청바지 등)를 확인해주세요. 옷 이름만으로 종류를 추측하지 않습니다.')
  if (kind in ['jeans','shorts','trousers'])!=(g['category']=='하의'):raise ValueError('의류의 세부 종류와 상하의 구분을 확인해주세요.')
  front=d.get('front') or {};colour=front.get('baseColor',g['color']);pattern=front.get('pattern',g['pattern'])
  query=' '.join([COLOURS.get(colour,colour),PATTERNS.get(pattern,pattern),KIND_TEXT[kind]])
  return kind,query

 def prepare(self,garments,encoder=None):
  e=copy(self.base);e.items=dict(e.items);e.ix=dict(e.ix);e.ids=list(e.ids)
  virtual=[];ids={};maps={}
  for g in garments:
   sid=g.get('referenceProductId')
   if sid:
    p=self.utility.canonical.get(sid)
    if not p or self.base.items[p]['side']!=('top' if g['category']=='상의' else 'bottom'):raise ValueError('공개 상품 ID와 의류 종류를 확인해주세요.')
    ids[g['id']]=p;maps[g['id']]=dict(mode='exact',products=[dict(id=p,title=self.base.items[p]['title'],similarity=1.)]);continue
   kind,query=self.descriptor(g);p='user:'+hashlib.sha256((kind+'|'+query).encode()).hexdigest()[:24];ids[g['id']]=p
   if p not in e.ix:
    e.ix[p]=len(e.ids);e.ids.append(p);e.items[p]=dict(id=p,title=query,kind=kind,sourceCategory='POG '+kind,side='top' if g['category']=='상의' else 'bottom',aliases=[])
    virtual.append(p)
  if virtual:
   if encoder is None:
    if self.encoder is None:
     from sentence_transformers import SentenceTransformer
     self.encoder=SentenceTransformer(MODEL,revision=REVISION,local_files_only=True,trust_remote_code=False,device='cpu')
    encoder=self.encoder
   vectors=encoder.encode([text(e.items[p]) for p in virtual],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
   e.vectors=np.concatenate([e.vectors,np.asarray(vectors,dtype=np.float32)])
   e.lex=vstack([e.lex,e.vectorizer.transform([text(e.items[p]) for p in virtual])],format='csr')
  for g in garments:
   if g['id'] in maps:continue
   p=ids[g['id']];kind=e.items[p]['kind'];options=[x for x in self.base.ids if e.items[x]['kind']==kind]
   ix=np.asarray([e.ix[x] for x in options]);sim=e.vectors[ix]@e.vectors[e.ix[p]]
   order=np.argsort(-sim,kind='stable')[:3]
   maps[g['id']]=dict(mode='similar',query=e.items[p]['title'],products=[dict(id=options[i],title=e.items[options[i]]['title'],similarity=float(sim[i])) for i in order])
  return e,ids,maps

 def matrix(self,e,tops,bottoms):
  values=np.zeros((len(bottoms),len(tops)))
  for kind in sorted({e.items[p]['kind'] for p in bottoms}):
   indexes=[i for i,p in enumerate(bottoms) if e.items[p]['kind']==kind];cs=[bottoms[i] for i in indexes]
   X=kernel.base_features(e,tops,cs);F,R,_=kernel.support_features(e,tops,cs,exclude_self=False,exclude_near=True)
   z=feature_view(X,F,R,'RAG',self.utility.profile['view'])
   S=self.utility.model.predict_proba(z.reshape(-1,z.shape[-1]))[:,1].reshape(len(cs),len(tops))
   for j,c in enumerate(cs):
    for k,q in enumerate(tops):
     if (q,c) in self.utility.confirmed:S[j,k]=1.
   values[indexes]=S
  return values

 def assess(self,candidate,wardrobe,occasion='데일리',mode='hybrid'):
  if candidate['category']!='하의':raise ValueError('현재 코디 비교는 구매 후보 하의와 보유 상의·하의를 기준으로 합니다.')
  tops=sorted([g for g in wardrobe if g['category']=='상의'],key=lambda g:g['id'])
  owned=sorted([g for g in wardrobe if g['category']=='하의'],key=lambda g:g['id'])
  if not tops or not owned:raise ValueError('비교하려면 보유 상의와 기존 하의를 각각 한 벌 이상 등록해주세요.')
  unique=lambda gs:list({('id:'+self.utility.canonical.get(g['referenceProductId'],g['referenceProductId'])) if g.get('referenceProductId') else json.dumps(self.descriptor(g),ensure_ascii=False):g for g in reversed(gs)}.values())
  tops=unique(tops);owned=unique(owned)
  with self.utility.lock,threadpool_limits(limits=1):
   e,ids,maps=self.prepare([candidate]+tops+owned)
   tq=[ids[g['id']] for g in tops];bs=[ids[g['id']] for g in [candidate]+owned];S=self.matrix(e,tq,bs)
   # Exact source records preserve binary observed history. For user garments,
   # history is estimated by the same frozen support model as the candidate.
   exact=all(m['mode']=='exact' for m in maps.values())
   H=np.asarray([max(int((q,w) in self.utility.confirmed) for w in bs[1:]) for q in tq],float) if exact else S[1:].max(0)
   C=float(S[0].mean());gains=np.maximum(S[0]-H,0.);M=float(gains.mean())
   evidence=[];rows=[]
   for i,g in enumerate(tops):
    qp=maps[g['id']]['products'][0]['id'];cp=maps[candidate['id']]['products'][0]['id']
    related=self.utility.evidence(cp,[qp])[:2];eids=[]
    for r in related:
     eid=outfit_evidence_id(r);eids.append(eid)
     if not any(x['id']==eid for x in evidence):evidence.append(dict(id=eid,sourceId='POG',sourceFamily='POG',title='공개 코디 사례',text='유사 상품의 실제 상하의 코디 기록',locator=' · '.join(r['sourceOutfits']),url=r['source'],collection='style',kind='outfit',currentScope=True,retrieval=dict(bm25=0,cosine=maps[g['id']]['products'][0]['similarity'],rrf=0),publicPair=r))
    rows.append(dict(groupId=g['id'],garmentIds=[g['id']],label=g['color']+' · '+g['pattern'],support=float(S[0,i]),existingSupport=float(H[i]),gain=float(gains[i]),evidenceIds=eids,grounds=dict(familyCount=1 if eids else 0,directFamilyCount=0,families=[dict(family='POG',support=float(S[0,i]),evidenceIds=eids)] if eids else [])))
  n=len(tops);fingerprint=hashlib.sha256(json.dumps(dict(candidate=candidate,wardrobe=sorted(wardrobe,key=lambda x:x['id']),version=VERSION,model=self.utility.version),sort_keys=True,ensure_ascii=False).encode()).hexdigest()
  return dict(candidateId=candidate['id'],evaluationFingerprint=fingerprint,status='scored',score=round(100*M,1),combination=round(100*C,1),contribution=round(100*M,1),
   assessmentScope='public-case',counts=dict(totalGroups=n,supportedGroups=n,comparableGroups=n,baselineGroups=len(owned),uniqueFamilies=1),coverage=100.,gainCoverage=100.,
   partialAssessment=dict(frontSupportedGroups=n,baseSupportedGroups=n,totalGroups=n,frontCombination=None,label=''),
   evidenceQuality=dict(directGroups=0,ruleOnlyGroups=n,level='public-case-model',label='공개 코디 검색·학습 기반'),gaps=[],inputReviews=[],groups=rows,evidence=evidence,
   matches=maps,historyMode='observed' if exact else 'estimated',
   message='공개 상품의 코디 기록과 학습된 조합 지원을 이용해 계산했습니다.' if exact else '확인한 의류 정보로 공개 코디를 검색하고, 기존 옷과 후보를 같은 모형으로 비교했습니다.',
   method=dict(version=VERSION,alpha=0,minimumCoverage=1,grouping='동일한 검색 입력의 의류를 한 그룹으로 집계',formula='C=mean(S), M=mean(max(S−H,0)), U=M',retrieval='POG 양방향 코디 검색',aggregation='사진 입력은 기존 지원 H를 같은 모형으로 추정; 공개 상품 입력은 관측 관계 사용',supportModel=self.utility.version),
   limitations=['유사 상품은 입력한 의류와 동일 상품으로 간주하지 않습니다.','등록한 종류·색상·무늬를 검색에 사용하며 착용 상황이나 로고의 미세한 위치는 현재 점수에 반영하지 않습니다.'])
