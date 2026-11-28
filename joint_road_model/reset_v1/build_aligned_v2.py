"""Build dataset_aligned_v2: frozen combined_v1 with lane-mask components rigidly snapped PERPENDICULAR to their axis onto paint evidence.

User-approved 2026-10-30 (train + val). AUTOMATED geometric alignment, NOT visual review.
- Never writes into dataset_combined_v1. Refuses to run if the output directory exists.
- Test rows, detection labels and the 132 user-approved repair versions (provenance human_*) are left unchanged.
- Ignore pixels (255) are never changed. A component is moved only if MIN_SHIFT <= |perp| <= MAX_SHIFT (search +/-R; larger optima near kerbs were ambiguous in gallery review), evidence gain >= MIN_GAIN,
  shape is elongated (line-like), and its shifted 1px-dilated footprint touches no other line (original or shifted).
  Draft1 (dataset_aligned_v2_rejected_draft1) lacked conflict checks: rejected after gallery review (double line merged).
  Draft2 (dataset_aligned_v2_superseded_draft2_R5) used R=5 and skipped all >=5px offsets; superseded.
Paint evidence = white top-hat(grey) + top-hat(Lab b*) with a 25px elliptical kernel, native resolution.
"""
import json,hashlib,argparse,collections
from pathlib import Path
import numpy as np,cv2

R=8;MAX_SHIFT=6.5;MIN_SHIFT=1.5;MIN_GAIN=1.10;MIN_AREA=60;MIN_ELONG=3.0
K=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))

def sha_bytes(b):return hashlib.sha256(b).hexdigest()

def evidence(im):
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);b=cv2.cvtColor(im,cv2.COLOR_BGR2LAB)[...,2]
    return cv2.morphologyEx(g,cv2.MORPH_TOPHAT,K).astype(np.float32)+cv2.morphologyEx(b,cv2.MORPH_TOPHAT,K).astype(np.float32)

def snap(im,m):
    """Return (new_mask, list of change records)."""
    ev=np.pad(evidence(im),R,mode='edge');out=m.copy();changes=[];proposals=[]
    lane=(m==1);nc,lab,st,_=cv2.connectedComponentsWithStats(lane.astype(np.uint8))
    for c in range(1,nc):
        area=int(st[c,4])
        if area<MIN_AREA:continue
        ys,xs=np.nonzero(lab==c);cov=np.cov(np.stack([xs,ys]).astype(np.float64));evals,evecs=np.linalg.eigh(cov)
        elong=float(np.sqrt(evals[1]/max(evals[0],1e-6)))
        if elong<MIN_ELONG:continue
        nx,ny=evecs[:,0];best=(-1.,0,0,0.);base=None
        for t in np.arange(-R,R+.5,.5):
            dx,dy=int(round(t*nx)),int(round(t*ny));v=float(ev[ys+R+dy,xs+R+dx].mean())
            if t==0:base=v
            if v>best[0]:best=(v,dx,dy,float(t))
        v,dx,dy,t=best;gain=v/max(base,1e-6)
        rec={'component':c,'area':area,'bbox_xywh':[int(x) for x in st[c,:4]],'elongation':round(elong,2),'perp_shift_px':t,'dx':dx,'dy':dy,'evidence_gain':round(gain,4)}
        if abs(t)<MIN_SHIFT or gain<MIN_GAIN:continue
        if abs(t)>MAX_SHIFT:rec['skipped']='shift_beyond_max_or_at_search_limit';changes.append(rec);continue
        proposals.append((c,rec))
    # Conflict resolution (draft1 failure: two labels of a double line converged on one stripe).
    # A proposal is rejected if its shifted, 1px-dilated footprint touches any OTHER component's original pixels
    # or any other proposal's shifted footprint. Rejected components stay where they were.
    H,Wd=m.shape;k3=np.ones((3,3),np.uint8);tg={}
    for c,rec in proposals:
        comp=(lab==c).astype(np.uint8);M=np.float32([[1,0,rec['dx']],[0,1,rec['dy']]])
        tg[c]=cv2.warpAffine(comp,M,(Wd,H),flags=cv2.INTER_NEAREST,borderValue=0).astype(bool)
    for c,rec in proposals:
        d=cv2.dilate(tg[c].astype(np.uint8),k3).astype(bool)
        if (d&lane&(lab!=c)).any():rec['skipped']='would_touch_other_original_line';changes.append(rec);continue
        if any((d&tg[o]).any() for o,_ in proposals if o!=c):rec['skipped']='conflicts_with_other_shift';changes.append(rec);continue
        comp=lab==c;out[comp&(out==1)]=0;out[tg[c]&(m!=255)]=1   # vacate; ignore pixels never touched
        rec['applied']=True;changes.append(rec)
    return out,changes

def main():
    p=argparse.ArgumentParser();p.add_argument('--src',type=Path,default=Path(__file__).resolve().parent/'dataset_combined_v1')
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'dataset_aligned_v2');a=p.parse_args()
    if a.out.exists():p.error('output exists; never overwrite a dataset version')
    srcm=a.src/'manifest.json';src_bytes=srcm.read_bytes();md=json.loads(src_bytes);a.out.mkdir(parents=True)
    items=[];log=[];stats=collections.Counter()
    for it in md['items']:
        r=dict(it);r['detection_labels']=str((a.src/it['detection_labels']).resolve())
        orig=a.src/it['lane_mask'];r['lane_mask']=str(orig.resolve());r['lane_mask_version']='combined_v1'
        eligible=it['split'] in('train','val') and not it['provenance'].startswith('human')
        if eligible:
            im=cv2.imread(it['image']);m=cv2.imread(str(orig),cv2.IMREAD_UNCHANGED)
            if hashlib.sha256(Path(it['image']).read_bytes()).hexdigest()!=it['source_sha256']:raise SystemExit(f"source hash mismatch {it['path']}")
            new,ch=snap(im,m);applied=[c for c in ch if c.get('applied')]
            stats['images_checked']+=1;stats['components_moved']+=len(applied);stats.update('skipped_'+c['skipped'] for c in ch if c.get('skipped'))
            if applied:
                assert new.shape==m.shape and set(np.unique(new))<= {0,1,255} and np.array_equal(new==255,m==255)
                dst=a.out/'lane_masks'/it['lane_mask'].split('lane_masks/',1)[-1];dst.parent.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(dst),new);back=cv2.imread(str(dst),cv2.IMREAD_UNCHANGED);assert np.array_equal(back,new)
                r['lane_mask']=str(dst.resolve());r['lane_mask_version']='aligned_v2';r['mask_sha256']=sha_bytes(dst.read_bytes())
                r['previous_mask_sha256']=it['mask_sha256'];stats['images_changed']+=1;stats[f"changed_{it['split']}_{it['provenance'][:6]}"]+=1
                log.append({'path':it['path'],'split':it['split'],'provenance':it['provenance'],'source_sha256':it['source_sha256'],
                            'original_mask':str(orig.resolve()),'original_mask_sha256':it['mask_sha256'],'new_mask':str(dst.resolve()),'new_mask_sha256':r['mask_sha256'],'components':ch})
        r['reviewed']=it.get('reviewed',False)  # automated alignment does not constitute review
        items.append(r)
    out={k:v for k,v in md.items() if k!='items'}
    out.update(version='aligned_v2',derived_from={'manifest':str(srcm.resolve()),'sha256':sha_bytes(src_bytes)},
        annotation_source=(md.get('annotation_source') or '')+' + automated perpendicular lane-line alignment (aligned_v2) on train/val; not visual review',
        alignment={'method':__doc__.strip(),'R':R,'MAX_SHIFT':MAX_SHIFT,'MIN_SHIFT':MIN_SHIFT,'MIN_GAIN':MIN_GAIN,'MIN_AREA':MIN_AREA,'MIN_ELONG':MIN_ELONG,
                   'excluded':'test split; provenance human_* (132 user-approved repair versions); detection labels','user_approval':'2026-10-30 chat: "Snap train + val"'},
        items=items)
    (a.out/'manifest.json').write_text(json.dumps(out,indent=1));(a.out/'change_manifest.json').write_text(json.dumps({'stats':stats,'changes':log},indent=1))
    print(json.dumps(stats,indent=1),'\nmanifest sha256',sha_bytes((a.out/'manifest.json').read_bytes()))

if __name__=='__main__':main()
