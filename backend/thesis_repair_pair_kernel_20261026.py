"""Pointwise conservative forward/reverse co-outfit support features.

Each (query,candidate) pair uses its own fixed reference exclusions. Therefore
adding or reordering other candidates cannot alter this pair's support.
"""
import numpy as np
from scipy.special import logsumexp,expit
from backend.pog_retrieval import text

def base_features(e,queries,candidates):
    projection=np.random.default_rng(2026102621).normal(0,1/np.sqrt(128),(384,128)).astype(np.float32)
    qi=[e.ix[p] for p in queries];ci=[e.ix[p] for p in candidates]
    q=e.vectors[qi]@projection;c=e.vectors[ci]@projection
    qq=np.broadcast_to(q,(len(ci),*q.shape));cc=np.broadcast_to(c[:,None,:],qq.shape)
    dense=e.vectors[ci]@e.vectors[qi].T;lex=(e.lex[ci]@e.lex[qi].T).toarray()
    return np.concatenate([qq,cc,qq*cc,np.abs(qq-cc),dense[...,None],lex[...,None]],axis=-1).astype(np.float32)

def support_features(e,queries,candidates,exclude_self=True,exclude_near=True):
    ci=np.asarray([e.ix[p] for p in candidates]);fwd=[];rev=[];stats=[];cache={}
    for q in queries:
        source=e.items[q];target=e.items[candidates[0]];cat=(source['sourceCategory'],source['kind'],target['sourceCategory'],target['kind']);p=e.pools.get(cat)
        if p is None:
            fwd.append(np.full((len(ci),12),.5));rev.append(np.full((len(ci),12),.5));stats.append([dict(empty=True) for _ in ci]);continue
        qi=e.ix[q];ad=e.vectors[qi]@e.vectors[p['ti']].T;al=(e.lex[qi]@e.lex[p['ti']].T).toarray().ravel()
        if cat not in cache:
            bd=e.vectors[ci]@e.vectors[p['bi']].T;bl=(e.lex[ci]@e.lex[p['bi']].T).toarray()
            titles=[text(e.items[x]).casefold() for x in p['bottoms']]
            near=np.stack([((bd[i]>=.98)&(bl[i]>=.90))|np.asarray([t==text(e.items[c]).casefold() for t in titles]) for i,c in enumerate(candidates)])
            cache[cat]=(bd,bl,near)
        bd,bl,near=cache[cat];qf=[];qr=[];qs=[]
        for i,c in enumerate(candidates):
            kt=np.asarray([x!=q for x in p['tops']]) if exclude_self else np.ones(len(ad),bool)
            kb=~near[i] if exclude_near else np.ones(bd.shape[1],bool)
            available=kt[p['t']]&kb[p['b']];t=p['t'][available];b=p['b'][available]
            tc=np.bincount(t,minlength=len(ad));bc=np.bincount(b,minlength=bd.shape[1]);T=np.flatnonzero(tc);B=np.flatnonzero(bc)
            qs.append(dict(queryInReference=bool(q in p['tops']),nearBottomsRemoved=int((~kb).sum()),remainingEdges=len(t),empty=not len(t)))
            if not len(t):qf.append(np.full(12,.5));qr.append(np.full(12,.5));continue
            forward=[];reverse=[]
            for hybrid in [False,True]:
                a=(ad+al)/2 if hybrid else ad;v=(bd[i]+bl[i])/2 if hybrid else bd[i]
                to=T[np.argsort(-a[T],kind='stable')];bo=B[np.argsort(-v[B],kind='stable')]
                bg=logsumexp(v[b]/.1-np.log(tc[t])-np.log(len(T)));rbg=logsumexp(a[t]/.1-np.log(bc[b])-np.log(len(B)))
                conditional=[];density=[]
                for k in [1,8,32]:
                    mask=np.isin(t,to[:min(k,len(T))]);w=a[t[mask]]/.1-np.log(tc[t[mask]]);w-=logsumexp(w)
                    con=logsumexp(v[b[mask]]/.1+w);conditional.append(np.clip(con*.1,0,1));density.append(expit(con-bg))
                    rm=np.isin(b,bo[:min(k,len(B))]);rw=v[b[rm]]/.1-np.log(bc[b[rm]]);rw-=logsumexp(rw)
                    rcon=logsumexp(a[t[rm]]/.1+rw);reverse.extend([np.clip(rcon*.1,0,1),expit(rcon-rbg)])
                forward.extend(conditional+density)
            qf.append(forward);qr.append(reverse)
        fwd.append(qf);rev.append(qr);stats.append(qs)
    F=np.asarray(fwd,dtype=np.float32).transpose(1,0,2);R=np.asarray(rev,dtype=np.float32).transpose(1,0,2)
    assert F.shape==R.shape==(len(candidates),len(queries),12) and np.isfinite(F).all() and np.isfinite(R).all()
    return F,R,stats
