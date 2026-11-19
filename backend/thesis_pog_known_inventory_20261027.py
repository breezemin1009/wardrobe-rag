"""Source-feasibility inventory for known-context/fresh-candidate POG transfer."""
from collections import defaultdict,Counter
from pathlib import Path
from backend import thesis_repair_models_20261026 as b
ROOT=b.DATA/'pog';OUT=b.DATA/'research-repair-20261026/pog-known-source-20261027'

def main():
    assert not OUT.exists();OUT.mkdir();items=b.read(ROOT/'evidence-history-serving-20261026/items.json');canonical={a:p for p,x in items.items() for a in [p]+x['aliases']}
    rawio=b.read(ROOT/'raw-outfit-incidence-audit-20261023/item-outfits.json');io={p:set().union(*(set(rawio.get(a,[])) for a in [p]+x['aliases'])) for p,x in items.items()}
    prior=set();files={}
    def walk(node):
        if isinstance(node,dict):
            if 'contexts' in node and 'bottoms' in node:prior.update(canonical[p] for p in node['bottoms'] if isinstance(p,str) and p in canonical)
            if 'candidates' in node:
                for p in node['candidates']+node.get('owned',[]):
                    pid=p if isinstance(p,str) else p.get('id') if isinstance(p,dict) else None
                    if pid in canonical:prior.add(canonical[pid])
            for value in node.values():
                if isinstance(value,(dict,list)):walk(value)
        elif isinstance(node,list):
            for value in node:
                if isinstance(value,(dict,list)):walk(value)
    for file in ROOT.rglob('*.json'):
        if not (file.name.endswith('frames.json') or file.name.endswith('inputs.json') or file.name in ['frames.json','cohorts.json','cases.json']):continue
        try:value=b.read(file)
        except (ValueError,OSError):continue
        walk(value);files[str(file.relative_to(ROOT))]=b.sha(file)
    previously=b.read(ROOT/'history-transfer-eligible-20261026.json');eligible0={canonical[p] for p in previously['eligibleCandidateIds'] if p in canonical};oldouts=set().union(*(io[p] for p in prior))
    edges=[];g=defaultdict(set);rev=defaultdict(set)
    for key,origins in b.read(ROOT/'adversarial-audit-20261023/source-edge-outfits.json').items():
        q,c=key.split(':')
        if q not in items or c not in items:continue
        edges.append(dict(top=q,bottom=c,outfits=origins));g[q].add(c);rev[c].add(q)
    eligible={p for p in eligible0-prior if len(rev[p])>=2 and not io[p]&oldouts and items[p]['kind'] in ['jeans','trousers','shorts']}
    b.write(OUT/'eligible-candidate-ids.json',sorted(eligible));b.write(OUT/'items.json',items);b.write(OUT/'all-edges.json',edges);b.write(OUT/'item-outfits.json',{p:sorted(v) for p,v in io.items()})
    report=dict(createdAt=b.stamp(),codeSha256=b.sha(__file__),items=len(items),edges=len(edges),oldBottomRoleIDs=len(prior),freshAtLeastTwoTopCandidates=len(eligible),kinds=dict(Counter(items[p]['kind'] for p in eligible)),degreeCounts=dict(Counter(len(rev[p]) for p in eligible)),
      information='Known public tops/owned history may recur from older studies;candidate IDs and their full original source outfits cannot recur from any earlier bottom input role. This is the intended known-wardrobe/cold-candidate boundary,not globally unseen users. POG had earlier exposed outcomes and reference use;any later model/reference must respect full quarantine and disclose history. No model predictions or new experiment adoption yet.',priorFiles=files,
      sourceBindings={str(file.relative_to(ROOT)):b.sha(file) for file in [ROOT/'evidence-history-serving-20261026/items.json',ROOT/'adversarial-audit-20261023/source-edge-outfits.json',ROOT/'raw-outfit-incidence-audit-20261023/item-outfits.json',ROOT/'history-transfer-eligible-20261026.json']})
    b.write(OUT/'inventory.json',report);print('POG KNOWN-CONTEXT FRESH-CANDIDATE INVENTORY', {k:v for k,v in report.items() if k not in ['priorFiles','sourceBindings']},flush=True)

if __name__=='__main__':main()
