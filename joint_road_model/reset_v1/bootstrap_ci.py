"""Episode-level bootstrap 95% CIs for metrics in an evaluate_v2.py metrics.json (needs per-image records with det_tp/det_conf/n_gt
and lane_tol2_v1_640; i.e. evaluations run after 2026-10-31). Resamples whole EPISODES with replacement (images within an episode are correlated).
Optional --compare second metrics.json on the same images: paired bootstrap of the difference (B - A)."""
import sys,json,argparse
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ultralytics.utils.metrics import ap_per_class

def stats(recs):
    L=np.sum([r['lane_counts_v1_640'] for r in recs],0);T=np.sum([r['lane_tol2_v1_640'] for r in recs],0);D=np.sum([r['detection_counts'] for r in recs],0)
    P=T[1]/max(1,T[0]);R=T[3]/max(1,T[2])
    tp=np.concatenate([np.array(r['det_tp'],bool).reshape(-1,10) for r in recs]);cf=np.concatenate([np.array(r['det_conf']) for r in recs]);ng=sum(r['n_gt'] for r in recs)
    ap=ap_per_class(tp,cf,np.zeros(len(cf)),np.zeros(ng))[5] if ng and len(cf) else np.zeros((1,10))
    return {'lane_iou_v1_640':L[0]/max(1,L.sum()),'lane_f1_tol2':2*P*R/max(1e-12,P+R),'cw_precision':D[0]/max(1,D[0]+D[1]),'cw_recall':D[0]/max(1,D[0]+D[2]),'AP50':float(ap[:,0].mean())}

def main():
    p=argparse.ArgumentParser();p.add_argument('metrics',type=Path);p.add_argument('--compare',type=Path);p.add_argument('--n',type=int,default=1000);p.add_argument('--out',type=Path);a=p.parse_args()
    A=json.loads(a.metrics.read_text())['records'];B=json.loads(a.compare.read_text())['records'] if a.compare else None
    if B:assert [r['path'] for r in A]==[r['path'] for r in B]
    eps=sorted({r['path'].split('/')[0] for r in A});idx={e:[i for i,r in enumerate(A) if r['path'].startswith(e+'/')] for e in eps}
    rng=np.random.default_rng(0);point=stats(A);draws=[]
    for _ in range(a.n):
        sel=[i for e in rng.choice(eps,len(eps)) for i in idx[e]]
        sa=stats([A[i] for i in sel])
        draws.append({k:stats([B[i] for i in sel])[k]-v for k,v in sa.items()} if B else sa)
    out={'episodes':len(eps),'images':len(A),'resamples':a.n,'mode':'paired difference B-A' if B else 'single','point':stats(B) if B else point,
         'point_A':point if B else None,'ci95':{k:[float(np.percentile([d[k] for d in draws],2.5)),float(np.percentile([d[k] for d in draws],97.5))] for k in draws[0]}}
    s=json.dumps(out,indent=1,default=float);print(s)
    if a.out:a.out.write_text(s)
if __name__=='__main__':main()
