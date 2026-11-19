"""Exact additional coverage of retrieved, identified public catalog garments.

This is a deterministic accounting layer, not an aesthetic predictor. Missing
records mean no confirmed catalog evidence. Photo-only recognition and inferred
compatibility scores cannot enter this layer as if they were confirmed facts.
"""
from pathlib import Path
from contextlib import closing
from collections import defaultdict
import json
import sqlite3
# The runtime counting layer has no torch/sklearn dependency.
TOP_CATEGORIES={'ブラウス','ニット','Tシャツ','タンクトップ','カーディガン','トップス','パーカー'}
BOTTOM_CATEGORIES={'ロングパンツ','スカート','ショートパンツ','ロングスカート'}


class CatalogCoverage:
    def __init__(self, roles, aliases, records):
        self.roles = roles
        self.aliases = aliases
        self.edges = defaultdict(set)
        for top,bottom,source in records:
            top,bottom = str(top),str(bottom)
            if roles.get(top) != 'top' or roles.get(bottom) != 'bottom':
                continue
            self.edges[(aliases[top],aliases[bottom])].add(str(source))

    @classmethod
    def from_source(cls, source):
        photos = {}
        with (source/'photo-hashes.jsonl').open(encoding='utf8') as stream:
            for line in stream:
                r=json.loads(line);photos[r['itemId']]=r['pixelSha256']
        roles={}
        from backend.fashion_taxonomy import breadcrumb_category
        with closing(sqlite3.connect(source/'catalog.sqlite')) as db:
            for pid,category,path in db.execute('SELECT item_id,category_color,category_path FROM products'):
                declared=category.split(' × ')[0];kind=breadcrumb_category(path)
                role=lambda value:'top' if value in TOP_CATEGORIES else 'bottom' if value in BOTTOM_CATEGORIES else None
                if pid in photos and kind and role(kind)==role(declared):
                    roles[pid]='top' if kind in TOP_CATEGORIES else 'bottom'
            records=[(t,b,'iqon-outfit:'+s) for t,b,s in db.execute('SELECT top_id,bottom_id,set_id FROM pair_evidence')]
        with (source/'additional-outfit-pairs.jsonl').open(encoding='utf8') as stream:
            for line in stream:
                r=json.loads(line);records.append((str(r['top']),str(r['bottom']),'iqon-outfit:'+r['sourceId']))
        # Alias equality is based on verified decoded source photos.
        aliases={pid:role+':'+photos[pid] for pid,role in roles.items()}
        return cls(roles,aliases,records)

    def pair(self, first, second):
        first,second=str(first),str(second)
        if first not in self.roles or second not in self.roles:
            raise ValueError('An identified catalog garment is required; an unknown photo is a separate input task')
        if self.roles[first]==self.roles[second]:
            raise ValueError('A top/bottom pair is required')
        top,bottom=(first,second) if self.roles[first]=='top' else (second,first)
        return sorted(self.edges.get((self.aliases[top],self.aliases[bottom]),set()))

    def evaluate(self, candidate, contexts, owned):
        candidate=str(candidate);contexts=list(map(str,contexts));owned=list(map(str,owned))
        for pid in [candidate,*contexts,*owned]:
            if pid not in self.roles or pid not in self.aliases:
                raise ValueError('Catalog identity has not been confirmed for every garment')
        if not contexts or any(self.roles[p]==self.roles[candidate] for p in contexts):
            raise ValueError('At least one opposite-category context is required')
        if any(self.roles[p]!=self.roles[candidate] for p in owned):
            raise ValueError('Owned alternatives must have the candidate category')
        context_groups={}
        for pid in sorted(contexts):context_groups.setdefault(self.aliases[pid],pid)
        owned_groups={}
        for pid in sorted(owned):owned_groups.setdefault(self.aliases[pid],pid)
        already_owned=self.aliases[candidate] in owned_groups
        result=[]
        for _,pid in sorted(context_groups.items()):
            evidence=self.pair(candidate,pid)
            prior=sorted({source for other in owned_groups.values() for source in self.pair(other,pid)})
            result.append({'contextProduct':pid,'recorded':bool(evidence),'alreadyCovered':bool(prior),
                'newRecordedCombination':bool(evidence) and not prior and not already_owned,
                'evidence':evidence,'existingEvidence':prior})
        c=sum(r['recorded'] for r in result);m=sum(r['newRecordedCombination'] for r in result);k=len(result)
        return {'candidateId':candidate,'mode':'identified-catalog-recorded-coverage-v1',
            'recordedCombinationCount':c,'newRecordedCombinationCount':m,'contextCount':k,
            'combinationCoverage':c/k,'additionalCoverage':m/k,'alreadyOwned':already_owned,
            'priorityValue':(k+1)*m+c,'priorityRule':'additional confirmed contexts first; total confirmed contexts breaks ties',
            'meaning':'counts of retrieved public co-outfit evidence; not aesthetic or purchase-satisfaction probabilities',
            'groups':result}

    def rank(self,candidates,contexts,owned):
        rows=[self.evaluate(pid,contexts,owned) for pid in sorted(set(map(str,candidates)))]
        rows.sort(key=lambda r:(-r['newRecordedCombinationCount'],-r['recordedCombinationCount'],r['candidateId']))
        ranks={}
        for index,row in enumerate(rows,1):
            key=(row['newRecordedCombinationCount'],row['recordedCombinationCount'])
            ranks.setdefault(key,index);row['rank']=ranks[key]
        return rows


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    engine=CatalogCoverage.from_source(args.source)
    case=json.loads(args.case.read_text(encoding='utf8'))
    result=engine.rank(case['candidates'],case['contexts'],case['owned'])
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'candidates':len(result),'rank1':result[0]['candidateId'],'newConfirmedContexts':result[0]['newRecordedCombinationCount']}))
