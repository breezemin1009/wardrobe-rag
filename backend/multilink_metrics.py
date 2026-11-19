"""Paired wardrobe metrics with the query subset applied to C as well as M."""
import numpy as np
COUNTS=np.asarray([q for q in [3,4,6,8] for w in [2,3,4,5]])
VIEWS=[f'T{q}B{w}' for q in [3,4,6,8] for w in [2,3,4,5]]

def arrays(frames):
    masks=[];gains=[];Y=[]
    for c in frames:
        nc=len(c['views'][VIEWS[0]]['c']);Y.append(c['observed'][:nc]);cm=[];cg=[]
        for name in VIEWS:
            v=c['views'][name];q=v['q'];w=v['w'];h=np.asarray(c['knownHistory'])[w][:,q].max(0);actual=np.asarray(c['observed'])[w][:,q].max(0)
            assert np.array_equal(h,actual)
            mask=np.zeros(8);mask[q]=1-h;cm.append(mask);cg.append((np.asarray(c['observed'])[:nc,q]*(1-actual)).sum(1))
        masks.append(cm);gains.append(cg)
    return dict(Y=np.asarray(Y),mask=np.asarray(masks),gain=np.asarray(gains),counts=COUNTS)

def ng_values(S,a,alpha):
    S=np.asarray(S);C=np.stack([S[...,:int(q)].mean(-1) for q in a['counts']],axis=1)
    M=np.einsum('ncq,nvq->nvc',S,a['mask'])/a['counts'][None,:,None]
    U=alpha*C+(1-alpha)*M;chosen=abs(U-U.max(-1,keepdims=True))<=1e-12
    gain=(a['gain']*chosen).sum(-1)/chosen.sum(-1);maximum=a['gain'].max(-1)
    return np.divide(gain,maximum,out=np.zeros_like(gain,dtype=float),where=maximum>0).mean(-1)

def condition(S,c,name,alpha):
    view=c['views'][name];q=view['q'];w=view['w'];nc=len(view['c']);S=np.asarray(S)[:,q]
    H=np.asarray(c['knownHistory'])[w][:,q].max(0);old=np.asarray(c['observed'])[w][:,q].max(0)
    G=(np.asarray(c['observed'])[:nc,q]*(1-old)).sum(1);C=S.mean(1);M=(S*(1-H)).mean(1);U=alpha*C+(1-alpha)*M
    chosen=np.flatnonzero(abs(U-U.max())<=1e-12);gain=float(G[chosen].mean());best=float(G.max())
    return dict(C=C.tolist(),M=M.tolist(),U=U.tolist(),gain=G.tolist(),chosen=chosen.tolist(),NG=gain/best if best else 0.,selectedGain=gain,selectedNewLinkFraction=gain/len(q),regret=best-gain,maxGain=best,knownCoverage=int(H.sum()))

def views(S,c,alpha):return {name:condition(S,c,name,alpha) for name in VIEWS}
def aggregate(v):return {key:float(np.mean([r[key] for r in v.values()])) for key in ['NG','selectedGain','selectedNewLinkFraction','regret','maxGain']}
