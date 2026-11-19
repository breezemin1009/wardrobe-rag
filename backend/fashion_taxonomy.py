"""Verify IQON category/color labels against the actual product category path.

The scraped category/color field sometimes refers to a different garment.
Only category-browser breadcrumbs (excluding brand and product names) determine
the canonical type. Disagreement in top/bottom role is quarantined.
"""
from pathlib import Path
import json,sqlite3,argparse
from collections import Counter
from backend.catalog_coverage import TOP_CATEGORIES,BOTTOM_CATEGORIES
from backend.utility_calibration import write,digest

TOP_MAP={value:value for value in TOP_CATEGORIES}
TOP_MAP.update({'シャツ':'ブラウス','キャミソール':'タンクトップ','ベスト':'トップス','プルオーバー':'トップス'})


def breadcrumb_category(raw):
    paths=json.loads(raw)
    categories=[p[2:-1] for p in paths if isinstance(p,list) and len(p)>=4 and p[1]=='カテゴリでさがす']
    if len(categories)!=1:return None
    labels=categories[0]
    if any(any(token in label for token in ['コート','ジャケット','アウター','ワンピース','ドレス']) for label in labels):return None
    bottom=any('パンツ' in label or 'スカート' in label for label in labels)
    top=any(label in TOP_MAP for label in labels)
    if bottom==top:return None
    if bottom:
        if any('ショートパンツ' in label for label in labels):return 'ショートパンツ'
        if any('ロングスカート' in label for label in labels):return 'ロングスカート'
        if any('スカート' in label for label in labels):return 'スカート'
        return 'ロングパンツ'
    return next((TOP_MAP[label] for label in reversed(labels) if label in TOP_MAP),None)


def create(source,cache,out):
    if out.exists():raise ValueError('Preserve the existing source-category audit')
    cached=set(map(int,json.loads((cache/'itemIds.json').read_text(encoding='utf8'))))
    accepted={};quarantined=[];counts=Counter()
    with sqlite3.connect(source/'catalog.sqlite') as db:
        for pid,name,declared,path in db.execute('SELECT item_id,name,category_color,category_path FROM products'):
            pid=int(pid)
            if pid not in cached:continue
            declared=declared.split(' × ')[0];canonical=breadcrumb_category(path)
            role=lambda c:'top' if c in TOP_CATEGORIES else 'bottom' if c in BOTTOM_CATEGORIES else None
            if not canonical or role(canonical)!=role(declared):
                quarantined.append(dict(itemId=pid,name=name,declared=declared,breadcrumb=canonical));continue
            accepted[str(pid)]=canonical;counts['accepted']+=1
            counts['fineTypeCorrected']+=canonical!=declared
    out.mkdir(parents=True);write(out/'categories.json',accepted);write(out/'quarantined.json',quarantined)
    write(out/'manifest.json',dict(**counts,quarantined=len(quarantined),
        policy='category-browser path; no brand/product-name tokens; top/bottom role agrees with declared field; other outerwear and dresses excluded',
        sourceCatalogSha256=digest(source/'catalog.sqlite'),cacheIdsSha256=digest(cache/'itemIds.json'),
        categoriesSha256=digest(out/'categories.json'),implementationSha256=digest(__file__)))
    print(json.dumps(dict(**counts,quarantined=len(quarantined))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();create(a.source,a.cache,a.output)
