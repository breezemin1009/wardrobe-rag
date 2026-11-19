"""Bounded access to verified original catalog photographs for the app."""
from functools import lru_cache
from contextlib import closing
import json,sqlite3
from backend.config import WORKSPACE,DATA
from backend.iqon_catalog import LocalArchive
from backend.official_drive_archive import read_indexed_member

SOURCE=WORKSPACE/'tmp/open-fashion-sources-20261020/iqon3000-gpbpr'
KINDS={'ブラウス':'블라우스','ニット':'니트','Tシャツ':'티셔츠','タンクトップ':'민소매','カーディガン':'카디건',
    'トップス':'상의','パーカー':'후드 상의','ロングパンツ':'긴 바지','スカート':'스커트','ショートパンツ':'반바지','ロングスカート':'긴 스커트'}


def identifier(pid):
    value=str(pid)
    if not value.isascii() or not value.isdigit() or len(value)>20:raise ValueError('Invalid catalog ID')
    return value


@lru_cache(maxsize=1)
def image_entries():
    allowed=set(json.loads((DATA/'iqon-fashion-images-v24/itemIds.json').read_text(encoding='utf8')))
    with (SOURCE/'source-image-index.jsonl').open(encoding='utf8') as stream:
        return {str(r['itemId']):r for r in map(json.loads,stream) if str(r['itemId']) in allowed}


@lru_cache(maxsize=512)
def item(pid):
    pid=identifier(pid)
    if pid not in image_entries():raise ValueError('Verified catalog photograph required')
    with closing(sqlite3.connect(SOURCE/'catalog.sqlite')) as db:
        row=db.execute('SELECT name,category_color FROM products WHERE item_id=?',(pid,)).fetchone()
    if not row:raise ValueError('Unknown catalog item')
    categories=json.loads((DATA/'fashion-taxonomy-v28/categories.json').read_text(encoding='utf8'))
    category=categories.get(pid)
    if category not in KINDS:raise ValueError('Supported garment category required')
    return dict(id=pid,name=KINDS[category],originalName=row[0],fineCategory=category,photo=f'/catalog/photos/{pid}')


@lru_cache(maxsize=128)
def photo(pid):
    pid=identifier(pid)
    item(pid)
    archive=LocalArchive(SOURCE/'IQON3000.zip')
    try:return read_indexed_member(archive,image_entries()[pid])
    finally:archive.close()


def examples():
    # Fixed first three DEVELOPMENT examples; neither scores nor outcomes select them.
    rows=json.loads((DATA/'verified-utility-v30/development-scenarios.json').read_text(encoding='utf8'))[:3]
    result=[]
    for i,row in enumerate(rows,1):
        request={name:list(map(str,row[name])) for name in ['candidates','contexts','owned']}
        items={pid:item(pid) for ids in request.values() for pid in ids}
        result.append(dict(id=row['scenarioId'],title=f'옷장 사례 {i}',request=request,items=items))
    return dict(examples=result,source='IQON3000 · GP-BPR 저자 공개 자료',scope='공개 상품 사진으로 구성한 비교 예시')
