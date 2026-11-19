"""Source age-word correction; replay the already frozen recipe unchanged.

The correction is based solely on explicit secondary-school title terms, not
on outcomes. Original experiment files and all initial results are preserved.
"""
from pathlib import Path
from datetime import datetime,timezone
import json
from backend.pog_refined_study import OUT as ORIGINAL
from backend.pog_cold_study import read,sha
from backend.prepare_pog_menswear import classify
from backend.data_io import write_json_atomic as write

OUT=ORIGINAL.with_name(ORIGINAL.name+'-scope-audited')

def main():
    if OUT.exists():raise ValueError('Scope correction already prepared')
    items=read(ORIGINAL/'items.json');blocked={k for k,v in items.items() if classify(v['title'])[0] is None}
    reference=read(ORIGINAL/'reference.json');cohorts=read(ORIGINAL/'cohorts.json');filtered=[r for r in reference if r['top'] not in blocked and r['bottom'] not in blocked]
    if any(p in blocked for rows in cohorts.values() for c in rows for p in c['contexts']+c['candidates']+c['owned']):raise ValueError('A query requires explicit cohort correction; no silent dropping')
    OUT.mkdir()
    for name in ['items.json','cohorts.json','vectors.npy','itemIds.json','encoder.json','frozen-encoder-source.py','frozen-analysis.py','development-features.json','development-features-lock.json','frozen-refined-retrieval.py']:
        (OUT/name).write_bytes((ORIGINAL/name).read_bytes())
    write(OUT/'reference.json',filtered);protocol=read(ORIGINAL/'protocol.json');protocol['referencePairs']=len(filtered)
    protocol['scopeCorrection']=dict(createdAt=datetime.now(timezone.utc).isoformat(),parentProtocolSha256=sha(ORIGINAL/'protocol.json'),parentEvaluationSha256=sha(ORIGINAL/'evaluation-report.json'),reason='Explicit middle/high-school title terms excluded from adult menswear reference; no scored query is affected',excludedItems=[dict(id=p,title=items[p]['title']) for p in sorted(blocked)],removedReferencePairs=len(reference)-len(filtered),cohortsChanged=False,weightsRetuned=False,analysis='Corrected replay of same frozen choices and 511 cases; not a new independent trial')
    protocol['files']={n:sha(OUT/n) for n in ['items.json','reference.json','cohorts.json']};write(OUT/'protocol.json',protocol)
    selection=read(ORIGINAL/'selection.json');selection['scopeCorrection']=protocol['scopeCorrection'];selection['protocolSha256']=sha(OUT/'protocol.json');selection['originalSelectionSha256']=sha(ORIGINAL/'selection.json');write(OUT/'selection.json',selection)
    write(OUT/'final-lock.json',dict(selectionSha256=sha(OUT/'selection.json'),scopeCorrection=True,choiceUnchanged=True,originalSelectionSha256=sha(ORIGINAL/'selection.json')))
    print(json.dumps(dict(excluded=len(blocked),referencePairs=len(filtered),removedPairs=len(reference)-len(filtered),cohortCases=len(cohorts['evaluation'])),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
