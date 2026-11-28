"""Evaluate ego-lane centre estimation on a split, using the deliverable inference path (road_perception.RoadPerception).

No vehicle-pose ground truth exists in the Mark episodes, so the REFERENCE is the same estimator applied to the supplied LABEL mask.
This measures whether the model's lane mask yields the same lane centre as the labels would, not absolute lane-keeping accuracy.
Also re-scores native lane IoU / F1@2px through the deliverable path as a consistency check against evaluate_v2.py.
--split test is allowed only with --final-test (reserved test is evaluated once, after development gates)."""
import sys,json,argparse,hashlib
from pathlib import Path
import numpy as np,cv2
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from road_perception import RoadPerception
from eval_common import lane_counts,rates

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--calibration',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--split',default='val');p.add_argument('--final-test',action='store_true');p.add_argument('--sheets',type=int,default=24);a=p.parse_args()
    if a.split=='test' and not a.final_test:p.error('reserved test requires --final-test')
    if a.out.exists():p.error('output exists')
    md=json.loads(Path(a.manifest).read_text());items=[it for it in md['items'] if it['split']==a.split]
    rp=RoadPerception(a.checkpoint,a.calibration,'cpu');a.out.mkdir(parents=True);rows=[];lane=np.zeros(3,np.int64);k5=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5));tol=np.zeros(4,np.int64)
    rng=np.random.default_rng(0);sheet_idx=set(rng.choice(len(items),min(a.sheets,len(items)),replace=False).tolist())
    for i,it in enumerate(items):
        frame=cv2.imread(it['image']);r=rp(frame);lab=np.asarray(Image.open(it['lane_mask']));ref=rp.center.estimate(lab==1);pr=r['lane_center']
        lane+=lane_counts(r['lane_mask'],lab)
        t=lab==1;v=lab!=255;pm=r['lane_mask']&v
        tol+=[pm.sum(),(pm&cv2.dilate(t.astype(np.uint8),k5).astype(bool)).sum(),t.sum(),(t&cv2.dilate(pm.astype(np.uint8),k5).astype(bool)).sum()]
        row={'path':it['path'],'ref_mode':ref['mode'],'pred_mode':pr['mode'],'ref_offset':ref['offset_norm'],'pred_offset':pr['offset_norm'],
             'ref_heading':ref['heading_deg'],'pred_heading':pr['heading_deg'],'ref_low_conf':ref.get('low_confidence'),'pred_low_conf':pr.get('low_confidence'),'n_crosswalks':len(r['crosswalks'])}
        rows.append(row)
        if i in sheet_idx:
            A=rp.draw(frame,r);B=frame.copy();B[t]=(0,255,0);rp.center.draw(B,ref,color=(255,0,0))
            cv2.putText(B,f"LABEL-derived: {ref['mode']} off={ref['offset_norm'] if ref['offset_norm'] is None else round(ref['offset_norm'],2)}",(15,35),0,1,(255,0,0),2)
            cv2.imwrite(str(a.out/(it['path'].replace('/','_')+'.jpg')),cv2.resize(np.hstack([B,A]),(1920,540)))
    both=[r for r in rows if r['ref_offset'] is not None and r['pred_offset'] is not None]
    d=np.array([abs(r['pred_offset']-r['ref_offset']) for r in both]);hd=np.array([abs(r['pred_heading']-r['ref_heading']) for r in both])
    hi=[r for r in both if not r['ref_low_conf'] and not r['pred_low_conf']];dh=np.array([abs(r['pred_offset']-r['ref_offset']) for r in hi])
    ref_avail=sum(r['ref_offset'] is not None for r in rows)
    P=tol[1]/max(1,tol[0]);R=tol[3]/max(1,tol[2])
    summary={'split':a.split,'images':len(rows),'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),'manifest':str(Path(a.manifest).resolve()),
        'reference':'same estimator on supplied label mask (no vehicle-pose ground truth)',
        'lane_native_deliverable_path':{**rates(lane),'f1_tol2px_native':float(2*P*R/max(1e-12,P+R))},
        'reference_available':ref_avail,'prediction_available_when_reference_available':sum(1 for r in rows if r['ref_offset'] is not None and r['pred_offset'] is not None),
        'prediction_available_when_reference_missing':sum(1 for r in rows if r['ref_offset'] is None and r['pred_offset'] is not None),
        'mode_agreement':float(np.mean([r['ref_mode']==r['pred_mode'] for r in rows])),
        'offset_abs_err_lane_widths':{'median':float(np.median(d)),'p90':float(np.percentile(d,90)),'within_0.05':float((d<=.05).mean()),'within_0.10':float((d<=.10).mean()),'n':len(d)} if len(d) else None,
        'offset_abs_err_high_confidence_only':{'median':float(np.median(dh)),'p90':float(np.percentile(dh,90)),'within_0.10':float((dh<=.10).mean()),'n':len(dh)} if len(dh) else None,
        'heading_abs_err_deg':{'median':float(np.median(hd)),'p90':float(np.percentile(hd,90))} if len(hd) else None}
    (a.out/'lane_center_metrics.json').write_text(json.dumps({'summary':summary,'rows':rows},indent=1,default=float));print(json.dumps(summary,indent=1,default=float))
if __name__=='__main__':main()
