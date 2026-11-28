"""Shared v2 evaluation: identical metric definitions to reset_v1/evaluate_combined.py, plus raster bookkeeping.

Lane agreement is reported on three rasters so runs at different input sizes stay comparable:
  v1_640   - prediction resampled into the v1 640x640 letterbox and scored against v1's 640 nearest-neighbour masks
             (this is the protocol behind the frozen laneIoU>=0.60 development gate);
  model    - the model's own letterboxed input raster;
  native   - prediction upsampled to the source resolution and scored against the native mask.
Detection matching/AP is copied from evaluate_combined.py (IoU is scale-invariant under uniform letterbox scaling).
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.metrics import ap_per_class

def match(pred,gt,threshold):
    used=set();ok=[]
    for b in pred:
        best=-1;score=threshold
        for j,g in enumerate(gt):
            if j in used:continue
            lo=np.maximum(b[:2],g[:2]);hi=np.minimum(b[2:4],g[2:4]);inter=np.maximum(0,hi-lo).prod()
            iou=inter/max(1e-9,(b[2]-b[0])*(b[3]-b[1])+(g[2]-g[0])*(g[3]-g[1])-inter)
            if iou>=score:score=iou;best=j
        ok.append(best>=0)
        if best>=0:used.add(best)
    return np.array(ok,dtype=bool)

def letterbox_geom(w,h,W,H):
    s=min(W/w,H/h);nw,nh=round(w*s),round(h*s);return (W-nw)//2,(H-nh)//2,nw,nh

def lane_counts(pred,mask):
    truth=mask==1;valid=mask!=255
    return np.array([(pred&truth&valid).sum(),(pred&~truth&valid).sum(),(~pred&truth&valid).sum()],dtype=np.int64)

TOL_KERNEL=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))  # radius-2 disk
def lane_tolerance_counts(pred,mask,kernel=TOL_KERNEL):
    """Boundary-tolerant agreement (2 px on the scored raster): [pred px, pred px within tol of label, label px, label px within tol of pred]; ignore excluded."""
    truth=mask==1;valid=mask!=255;p=pred&valid
    near_t=cv2.dilate(truth.astype(np.uint8),kernel).astype(bool);near_p=cv2.dilate(p.astype(np.uint8),kernel).astype(bool)
    return np.array([p.sum(),(p&near_t).sum(),truth.sum(),(truth&near_p).sum()],dtype=np.int64)

def tol_rates(c):
    P=c[1]/max(1,c[0]);R=c[3]/max(1,c[2]);return dict(precision=float(P),recall=float(R),f1=float(2*P*R/max(1e-12,P+R)),counts=[int(x) for x in c])

def rates(c):
    t,f,n=map(int,c);return dict(tp=t,fp=f,fn=n,precision=t/max(1,t+f),recall=t/max(1,t+n),iou=t/max(1,t+f+n))

class Evaluator:
    """Precomputes reference rasters once; call run(model) each epoch."""
    def __init__(self,dataset,v1_dataset=None,native=False,device='cpu'):
        self.ds=dataset;self.device=device;self.native=native
        self.sizes=[Image.open(dataset.root/r['image']).size for r in dataset.items]
        self.v1=[v1_dataset[i]['lane_mask'].numpy().astype(np.uint8) for i in range(len(v1_dataset))] if v1_dataset is not None else None
        self.v1_size=v1_dataset.out_w if v1_dataset is not None else None
    def _resample(self,prob,i,W,H,TW,TH):
        """prob on (H,W) letterbox -> (TH,TW) letterbox of the same source image, bilinear on the content region."""
        w,h=self.sizes[i];l,t,nw,nh=letterbox_geom(w,h,W,H);L,T,NW,NH=letterbox_geom(w,h,TW,TH)
        c=torch.from_numpy(prob[t:t+nh,l:l+nw])[None,None]
        c=F.interpolate(c,size=(NH,NW),mode='bilinear',align_corners=False)[0,0].numpy()
        out=np.zeros((TH,TW),np.float32);out[T:T+NH,L:L+NW]=c;return out
    @torch.no_grad()
    def run(self,model,lane_threshold=.5,conf=.25,keep=False,callback=None):
        model.eval();thresholds=np.linspace(.5,.95,10)
        lane={'model':np.zeros(3,np.int64),'v1_640':np.zeros(3,np.int64),'native':np.zeros(3,np.int64)}
        tol=np.zeros(4,np.int64);det=np.zeros(3,np.int64);tps=[];confs=[];targets=0;records=[];vloss=0.;n=0
        for i in range(len(self.ds)):
            r=self.ds[i];W,H=self.ds.out_w,self.ds.out_h
            out=model(r['img'][None].to(self.device))
            pred=non_max_suppression(out['detections'],conf_thres=.001,iou_thres=.6,nc=1)[0].cpu().numpy();pred=pred[np.argsort(-pred[:,4])]
            boxes=r['bboxes'].numpy()*np.array([W,H,W,H]);gt=np.concatenate((boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2),1)
            m=np.stack([match(pred,gt,t) for t in thresholds],1) if len(pred) else np.empty((0,10),bool)
            tps.append(m);confs.append(pred[:,4]);targets+=len(gt);sel=pred[:,4]>=conf;hit=int(m[sel,0].sum())
            dc=np.array([hit,int(sel.sum())-hit,len(gt)-hit]);det+=dc
            prob=out['lane_logits'][0,0].sigmoid().float().cpu().numpy();mask=r['lane_mask'].numpy()
            rec={'path':self.ds.items[i]['path'],'provenance':self.ds.items[i]['provenance'],'detection_counts':dc.tolist(),'det_tp':m.tolist(),'det_conf':pred[:,4].tolist(),'n_gt':int(len(gt))}
            c=lane_counts(prob>=lane_threshold,mask);lane['model']+=c;rec['lane_counts_model']=c.tolist()
            if self.v1 is not None:
                p1=prob if (W,H)==(self.v1_size,self.v1_size) else self._resample(prob,i,W,H,self.v1_size,self.v1_size)
                c=lane_counts(p1>=lane_threshold,self.v1[i]);lane['v1_640']+=c;rec['lane_counts_v1_640']=c.tolist();rec['lane_iou_v1_640']=float(c[0]/max(1,c.sum()))
                ct=lane_tolerance_counts(p1>=lane_threshold,self.v1[i]);tol+=ct;rec['lane_tol2_v1_640']=ct.tolist()
            if self.native:
                w,h=self.sizes[i];l,t,nw,nh=letterbox_geom(w,h,W,H)
                pn=F.interpolate(torch.from_numpy(prob[t:t+nh,l:l+nw])[None,None],size=(h,w),mode='bilinear',align_corners=False)[0,0].numpy()
                nm=np.asarray(Image.open(self.ds.root/self.ds.items[i]['lane_mask']));c=lane_counts(pn>=lane_threshold,nm);lane['native']+=c;rec['lane_counts_native']=c.tolist()
            if keep:rec['_prob']=prob;rec['_pred_boxes']=pred[sel,:5]
            records.append(rec)
            if callback:callback(i,r,out,prob,pred[sel,:4],gt)
        cf=np.concatenate(confs) if confs else np.zeros(0);correct=np.concatenate(tps) if tps else np.zeros((0,10),bool)
        ap=ap_per_class(correct,cf,np.zeros(len(cf)),np.zeros(targets))[5] if targets and len(cf) else np.zeros((1,10))
        res={'lane':{k:rates(v) for k,v in lane.items() if v.sum()},'crosswalk':{**rates(det),'AP50':float(ap[:,0].mean()),'AP50_95':float(ap.mean())},'lane_threshold':lane_threshold,'detection_confidence':conf}
        res['crosswalk'].pop('iou',None)
        if self.v1 is not None:res['lane']['v1_640_tol2']=tol_rates(tol)
        return res,records

def gate(res,lane_raster='v1_640'):
    """Revised 2026-10-31 (decided by Claude at the user's delegation, AFTER seeing v1-v2d results; disclosed): strict thin-line IoU
    is dominated by 1-2 px boundary placement (a label shifted 1 px scores IoU 0.62 against itself), so the primary lane gate is
    F1 with a 2 px tolerance on the v1 640 raster >= 0.80. Strict IoU>=0.60 is still reported ('all_strict_v1')."""
    l=res['lane'][lane_raster]['iou'];c=res['crosswalk'];f=res['lane'].get('v1_640_tol2',{}).get('f1',0)
    det=c['AP50']>=.75 and c['precision']>=.85 and c['recall']>=.8
    return {'laneF1@2px>=0.80':f>=.8,'laneIoU>=0.60':l>=.6,'AP50>=0.75':c['AP50']>=.75,'precision>=0.85':c['precision']>=.85,'recall>=0.80':c['recall']>=.8,
            'all':f>=.8 and det,'all_strict_v1':l>=.6 and det,'lane_raster':lane_raster}
