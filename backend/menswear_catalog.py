"""Only metadata-approved catalog IDs are addressable through the active API."""
from functools import lru_cache
from pathlib import Path
import hashlib, json, re
from backend.config import DATA
from backend.audience import ALLOWED_AUDIENCES
from backend.acquire_menswear_catalog import title_scope_conflict, multi_garment_title

ROOT=DATA/'menswear-rebuild-v1'


@lru_cache(maxsize=1)
def eligible_catalog():
    rows=json.loads((ROOT/'catalog.json').read_text(encoding='utf8'))
    accepted={r['productId']:r for r in rows}
    if len(accepted)!=len(rows): raise ValueError('Duplicate catalog identity')
    for row in rows:
        if row['audience'] not in ALLOWED_AUDIENCES or row['gender'] not in {'Men','Unisex'}:
            raise ValueError('Out-of-scope catalog item')
        if row['audienceBasis'].get('type')!='source-metadata': raise ValueError('Missing audience provenance')
        if title_scope_conflict(row['productDisplayName']) or multi_garment_title(row['productDisplayName']):
            raise ValueError('Catalog title contradicts single-garment adult scope')
    return accepted


@lru_cache(maxsize=1)
def catalog():
    pool=eligible_catalog()
    review=json.loads((ROOT/'catalog-review.json').read_text(encoding='utf8'))
    originals=json.loads((ROOT/'original-photos.json').read_text(encoding='utf8'))['images']
    accepted={}
    for item in review['records']:
        pid=item['id'];row=pool.get(pid)
        if (not row or item['status']!='approved' or item['gender']!=row['gender']
                or item['title']!=row['productDisplayName'] or item['thumbnailSha256']!=row['photoSha256']
                or item['originalSha256']!=originals.get(pid,{}).get('sha256')):
            raise ValueError('Photo review no longer matches original product '+pid)
        accepted[pid]=row
    if len(accepted)!=review['accepted']:raise ValueError('Incomplete photo-reviewed catalog')
    return accepted


def public_item(row):
    original=original_photos().get(row['productId'])
    return dict(id=row['productId'],name=row['productDisplayName'],category=row['category'],
                audience=row['audience'],gender=row['gender'],articleType=row['articleType'],
                baseColour=row['baseColour'],photo='/menswear/photos/'+row['productId'],
                width=original['width'] if original else row['width'],height=original['height'] if original else row['height'],
                photoKind='original' if original else 'thumbnail',sourceUrl=row['sourceUrl'],
                audienceBasis=row['audienceBasis'])


def list_catalog(category=None,audience=None,query='',offset=0,limit=12):
    rows=[r for r in catalog().values() if (not category or r['category']==category)
          and (not audience or r['audience']==audience)
          and query.lower().strip() in (r['productDisplayName']+' '+r['baseColour']+' '+r['articleType']).lower()]
    rows.sort(key=lambda r:int(r['productId']))
    return dict(items=[public_item(r) for r in rows[offset:offset+limit]],total=len(rows),offset=offset,limit=limit,
                source='Fashion Product Images · Myntra / Kaggle',
                reviewVersion='catalog-photo-review-20261021-v1',metadataEligible=len(eligible_catalog()),
                use='상품 표기와 사진을 대조한 상품입니다. 코디 연결은 FashionStylist 자료에서 별도로 검색합니다.')


def photo(product_id):
    if not re.fullmatch(r'\d{1,10}',product_id) or product_id not in catalog():raise ValueError('Unknown verified catalog ID')
    original=original_photos().get(product_id)
    folder='catalog-images-original' if original else 'catalog-images'
    data=(ROOT/folder/(product_id+'.jpg')).read_bytes()
    expected=original['sha256'] if original else catalog()[product_id]['photoSha256']
    if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Changed catalog photograph')
    return data


@lru_cache(maxsize=1)
def original_photos():
    path=ROOT/'original-photos.json'
    images=json.loads(path.read_text(encoding='utf8'))['images'] if path.exists() else {}
    return {pid:row for pid,row in images.items() if pid in catalog()}


@lru_cache(maxsize=1)
def outfits():
    data=json.loads((ROOT/'fashionstylist.json').read_text(encoding='utf8'))
    if any(r['audience'] not in ALLOWED_AUDIENCES for r in data['items'].values()):raise ValueError('Invalid outfit audience')
    return data


def list_outfits(offset=0,limit=6):
    data=outfits();result=[]
    for outfit in data['outfits'][offset:offset+limit]:
        items=[]
        for pid in outfit['tops']+outfit['bottoms']:
            r=data['items'][pid]
            items.append(dict(id=pid,name=r['title'],category='하의' if r['category']=='bottom' else '상의',
                              color=r['normalizedColor'],pattern=r['normalizedPattern'],audience=r['audience'],
                              originalColor=r['color'],originalPattern=r['pattern']))
        result.append(dict(id=outfit['outfitID'],season=outfit['season'],occasion=outfit['occasion'],
                           description=outfit['look'],items=items,
                           sourceUrl=data['audit']['source']+'/blob/'+data['audit']['revision']+'/Dataset/Male/look_en.csv'))
    return dict(outfits=result,total=len(data['outfits']),audit=data['audit'])
