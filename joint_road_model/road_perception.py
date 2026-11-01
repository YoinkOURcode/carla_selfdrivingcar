"""Single-call inference for the joint road model: lane paint mask, pedestrian-crossing boxes, ego-lane centre.

    from road_perception import RoadPerception
    rp = RoadPerception('joint_final.pt', calibration='lane_center_calibration.json', device='cpu')
    out = rp(frame_bgr)            # numpy HxWx3 BGR (OpenCV / CARLA camera), or a file path
    out['lane_center']['offset_norm']   # 0 = centred; +0.5 = vehicle on right boundary; None if no lane found
    out['crosswalks']                   # [{'xyxy':[x1,y1,x2,y2], 'confidence':c}, ...] in input-image pixels
    out['lane_mask']                    # HxW bool, lane paint

The checkpoint is loaded with torch.load(weights_only=False): only load checkpoints you trust.
Preprocessing reproduces training exactly: RGB, bilinear resize into a (114,114,114) letterbox of the checkpoint's input size.
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from model import JointRoadModel
from lane_center import LaneCenterEstimator
from ultralytics.utils.nms import non_max_suppression

class RoadPerception:
    def __init__(self,checkpoint,calibration=None,device='cpu',conf=0.25,lane_threshold=0.5,nms_iou=0.6):
        self.model,self.meta=JointRoadModel.load(checkpoint,device);self.model.eval();self.device=device
        if not self.meta.get('trained'):raise ValueError('checkpoint lane head is not trained')
        lb=self.meta.get('input_letterbox') or [640,640];self.W,self.H=(lb,lb) if isinstance(lb,int) else lb
        self.conf=conf;self.lane_threshold=lane_threshold;self.nms_iou=nms_iou
        self.center=LaneCenterEstimator.from_file(calibration) if calibration else None

    def preprocess(self,frame):
        if isinstance(frame,(str,Path)):im=Image.open(frame).convert('RGB')
        else:im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        w,h=im.size;s=min(self.W/w,self.H/h);nw,nh=round(w*s),round(h*s);l,t=(self.W-nw)//2,(self.H-nh)//2
        canvas=Image.new('RGB',(self.W,self.H),(114,114,114));canvas.paste(im.resize((nw,nh),Image.Resampling.BILINEAR),(l,t))
        x=torch.from_numpy(np.array(canvas).transpose(2,0,1)).float()[None]/255
        return x,(w,h,nw,nh,l,t)

    @torch.no_grad()
    def __call__(self,frame):
        x,(w,h,nw,nh,l,t)=self.preprocess(frame);out=self.model(x.to(self.device))
        det=non_max_suppression(out['detections'],conf_thres=self.conf,iou_thres=self.nms_iou,nc=1)[0].cpu().numpy()
        sx,sy=w/nw,h/nh;boxes=[]
        for x1,y1,x2,y2,c,_ in det:
            boxes.append({'class':'crosswalk','class_id':7,'confidence':float(c),
                          'xyxy':[float(np.clip((x1-l)*sx,0,w)),float(np.clip((y1-t)*sy,0,h)),float(np.clip((x2-l)*sx,0,w)),float(np.clip((y2-t)*sy,0,h))]})
        prob=out['lane_logits'][0,0].sigmoid().float().cpu()[t:t+nh,l:l+nw]
        prob=F.interpolate(prob[None,None],size=(h,w),mode='bilinear',align_corners=False)[0,0].numpy()
        mask=prob>=self.lane_threshold
        res={'lane_mask':mask,'lane_prob':prob,'crosswalks':boxes,'image_size':[w,h]}
        res['lane_center']=self.center.estimate(mask) if self.center else None
        return res

    def draw(self,frame,res):
        vis=frame.copy();m=res['lane_mask'];vis[m]=(vis[m]*.4+np.array([0,255,0])*.6).astype(np.uint8)
        for b in res['crosswalks']:
            x1,y1,x2,y2=map(int,b['xyxy']);cv2.rectangle(vis,(x1,y1),(x2,y2),(255,0,255),3)
            cv2.putText(vis,f"crossing {b['confidence']:.2f}",(x1,max(20,y1-8)),0,.7,(255,0,255),2)
        lc=res.get('lane_center')
        if lc:
            self.center.draw(vis,lc)
            txt=f"lane: {lc['mode']}"+(f"  offset {lc['offset_norm']:+.2f} lane widths  heading {lc['heading_deg']:+.1f} deg" if lc['offset_norm'] is not None else '')
            if lc.get('low_confidence'):txt+='  (low confidence)'
            cv2.putText(vis,txt,(15,35),0,1,(0,0,255),2)
        return vis

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description='Run the joint road model on images and write overlays + JSON.')
    p.add_argument('--checkpoint',required=True);p.add_argument('--calibration');p.add_argument('--device',default='cpu')
    p.add_argument('--out',type=Path,required=True);p.add_argument('images',nargs='+');a=p.parse_args()
    rp=RoadPerception(a.checkpoint,a.calibration,a.device);a.out.mkdir(parents=True,exist_ok=True)
    for f in a.images:
        frame=cv2.imread(f);r=rp(frame);stem=Path(f).stem
        cv2.imwrite(str(a.out/f'{stem}_overlay.jpg'),rp.draw(frame,r))
        (a.out/f'{stem}.json').write_text(json.dumps({'image':f,'crosswalks':r['crosswalks'],'lane_center':r['lane_center']},indent=1))
        print(stem,r['lane_center']['mode'] if r['lane_center'] else '-',len(r['crosswalks']),'crossings')
