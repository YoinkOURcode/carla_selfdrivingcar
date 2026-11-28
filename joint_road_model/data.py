"""Strict image/mask alignment and joint labels. Missing masks are never negatives."""
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class RoadDataset(Dataset):
    def __init__(self,manifest,split,imgsz=640,require_reviewed=True,rect=None,augment=False,seed=0):
        # rect=(width,height) letterboxes to a rectangle (v2); default square imgsz reproduces v1 exactly.
        self.root=Path(manifest).resolve().parent;self.size=imgsz
        self.out_w,self.out_h=(int(rect[0]),int(rect[1])) if rect else (imgsz,imgsz)
        self.augment=augment;self.rng=np.random.default_rng(seed)
        manifest_data=json.loads(Path(manifest).read_text())
        self.source_crosswalk_id=manifest_data.get('crosswalk_class_id',4)
        self.items=[r for r in manifest_data['items'] if r['split']==split]
        if not self.items:raise ValueError(f'No {split} examples')
        for r in self.items:
            if require_reviewed and not r.get('reviewed',False):raise ValueError(f"Needs lane/box review: {r['image']}")
            for key in ('image','detection_labels','lane_mask'):
                if not (self.root/r[key]).is_file():raise ValueError(f"Missing {key}: {r[key]}")
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        row=self.items[i];im=Image.open(self.root/row['image']).convert('RGB');mask=Image.open(self.root/row['lane_mask'])
        if im.size!=mask.size:raise ValueError('Image and mask dimensions differ')
        a=np.asarray(mask)
        if a.ndim!=2 or not set(np.unique(a)).issubset({0,1,255}):raise ValueError('Masks require single-channel IDs 0,1,255')
        W,H=self.out_w,self.out_h
        w,h=im.size;s=min(W/w,H/h);nw,nh=round(w*s),round(h*s);left,top=(W-nw)//2,(H-nh)//2
        image=Image.new('RGB',(W,H),(114,114,114));image.paste(im.resize((nw,nh),Image.Resampling.BILINEAR),(left,top))
        padded=Image.new('L',(W,H),255);padded.paste(mask.resize((nw,nh),Image.Resampling.NEAREST),(left,top))
        boxes=[];classes=[]
        for line in (self.root/row['detection_labels']).read_text().splitlines():
            if not line.strip():continue
            c,x,y,bw,bh=map(float,line.split())
            if c!=int(c) or c<0:raise ValueError('Invalid class ID')
            if int(c)!=self.source_crosswalk_id:continue
            c=0  # focused detector only learns crosswalks; other objects are outside this task
            if not(0<bw<=1 and 0<bh<=1 and x-bw/2>=-1e-5 and y-bh/2>=-1e-5 and x+bw/2<=1.00001 and y+bh/2<=1.00001):raise ValueError('Invalid box')
            classes.append([c]);boxes.append([(x*nw+left)/W,(y*nh+top)/H,bw*nw/W,bh*nh/H])
        img=torch.from_numpy(np.array(image).transpose(2,0,1)).float()/255
        if self.augment:img=self._photometric(img)
        return {'img':img,'lane_mask':torch.from_numpy(np.array(padded).astype(np.int64)),'cls':torch.tensor(classes,dtype=torch.float32).reshape(-1,1),'bboxes':torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),'path':row['image']}
    def _photometric(self,img):
        """Label-preserving colour jitter only (no geometric change, so thin masks/boxes are untouched).
        Horizontal flip is deliberately not used: it would swap traffic-side paint conventions."""
        r=self.rng
        img=img*r.uniform(.7,1.3)                                   # brightness
        mean=img.mean();img=(img-mean)*r.uniform(.7,1.3)+mean       # contrast
        grey=img.mean(0,keepdim=True);img=(img-grey)*r.uniform(.6,1.4)+grey  # saturation
        img=img*torch.tensor(r.uniform(.95,1.05,3),dtype=img.dtype).view(3,1,1)  # mild colour cast
        if r.random()<.3:img=img+torch.randn_like(img)*r.uniform(0,.03)          # sensor noise
        return img.clamp(0,1)

def collate(rows):
    return {'img':torch.stack([r['img'] for r in rows]),'lane_mask':torch.stack([r['lane_mask'] for r in rows]),'cls':torch.cat([r['cls'] for r in rows]),'bboxes':torch.cat([r['bboxes'] for r in rows]),'batch_idx':torch.cat([torch.full((len(r['cls']),),i,dtype=torch.float32) for i,r in enumerate(rows)])}

def validate_splits(manifest):
    rows=json.loads(Path(manifest).read_text())['items'];seen={};images=set()
    for r in rows:
        if r['image'] in images:raise ValueError('Duplicate image in manifest')
        images.add(r['image']);episode=r['episode'];split=r['split']
        if split not in ('train','val','test'):raise ValueError('Invalid split')
        if episode in seen and seen[episode]!=split:raise ValueError(f'Episode leakage: {episode}')
        seen[episode]=split
