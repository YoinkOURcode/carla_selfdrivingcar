"""One YOLO11s encoder, the existing detector, and a new lane-boundary decoder.

The source checkpoint uses crosswalk=4. This focused detector uses crosswalk=0; public ID=7.
Lane masks use 0=background, 1=painted lane boundary, 255=ignore/unlabelled.
"""
from pathlib import Path
import copy
import os,sys
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'annotation_pilot/vendor').is_dir():sys.path.insert(0,str(ROOT/'annotation_pilot/vendor'))
if (ROOT/'annotation_pilot/config').is_dir():os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'annotation_pilot/config'))
import torch
from torch import nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.cfg import get_cfg, cfg2dict
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.nn.tasks import DetectionModel

class JointRoadModel(nn.Module):
    feature_indices=(2,4,6)  # YOLO11 features at strides 4, 8 and 16
    def __init__(self,weights=None,architecture=None,names=None,decoder_width=48,feature_indices=None,refine_index=None,refine_width=16):
        super().__init__()
        self.feature_indices=tuple(feature_indices) if feature_indices is not None else (2,4,6)
        if weights is not None:
            source=YOLO(str(weights)).model.float()
            self.detector=DetectionModel(copy.deepcopy(source.yaml),nc=1,verbose=False)
            self.detector.names={0:'crosswalk'}
            self.detector.load(source,verbose=False)
            self.detector.args=copy.deepcopy(source.args)
        elif architecture is not None:self.detector=DetectionModel(architecture,nc=len(names),verbose=False)
        else:raise ValueError('weights or architecture required')
        if names is not None:self.detector.names=names
        self.names=self.detector.names
        self.crosswalk_id=next(int(k) for k,v in self.names.items() if v=='crosswalk')
        self.detector.args=get_cfg({**DEFAULT_CFG_DICT, **cfg2dict(getattr(self.detector,'args',{}))})
        self._features={}
        # v2 option: refine_index (e.g. 0 = stride-2 YOLO11 stem) adds a higher-resolution skip refinement.
        self.refine_index=refine_index;self.refine_width=refine_width
        for index in self.feature_indices+((refine_index,) if refine_index is not None else ()):
            self.detector.model[index].register_forward_hook(self._capture(index))
        self.detector.eval()
        with torch.no_grad():self.detector(torch.zeros(1,3,64,64))
        channels=[self._features[i].shape[1] for i in self.feature_indices]
        refine_channels=self._features[refine_index].shape[1] if refine_index is not None else None
        self._features.clear();self.decoder_width=decoder_width
        self.projections=nn.ModuleList([nn.Conv2d(c,decoder_width,1) for c in channels])
        self.lane_head=nn.Sequential(nn.Conv2d(decoder_width*len(channels),decoder_width,3,padding=1),nn.GroupNorm(8,decoder_width),nn.SiLU(),nn.Conv2d(decoder_width,decoder_width,3,padding=1),nn.SiLU(),nn.Conv2d(decoder_width,1,1))
        if refine_index is not None:
            # Residual stride-2 refinement: final logits = upsampled coarse logits + correction; zero-init keeps v1 behaviour at start.
            self.refine_proj=nn.Conv2d(refine_channels,refine_width,1)
            self.refine=nn.Sequential(nn.Conv2d(decoder_width+refine_width,32,3,padding=1),nn.GroupNorm(8,32),nn.SiLU(),nn.Conv2d(32,1,3,padding=1))
            nn.init.zeros_(self.refine[-1].weight);nn.init.zeros_(self.refine[-1].bias)
        self.set_phase('joint')
    def lane_parameters(self):
        mods=[self.projections,self.lane_head]+([self.refine_proj,self.refine] if self.refine_index is not None else [])
        return [p for m in mods for p in m.parameters()]
    def _capture(self,index):
        def hook(module,args,out):self._features[index]=out
        return hook
    def set_phase(self,phase):
        if phase not in ('lane_warmup','joint'):raise ValueError(phase)
        self.phase=phase
        for name,p in self.detector.named_parameters():p.requires_grad_(phase=='joint' and 'dfl.conv.weight' not in name)
        for p in self.lane_parameters():p.requires_grad_(True)
    def train(self,mode=True):
        super().train(mode)
        if getattr(self,'phase',None)=='lane_warmup':self.detector.eval()
        return self
    def forward(self,images):
        self._features.clear();detections=self.detector(images)
        features=[self._features.pop(i) for i in self.feature_indices]
        size=features[0].shape[-2:]
        fused=torch.cat([F.interpolate(proj(feat),size=size,mode='bilinear',align_corners=False) for proj,feat in zip(self.projections,features)],dim=1)
        if self.refine_index is None:
            lane=F.interpolate(self.lane_head(fused),size=images.shape[-2:],mode='bilinear',align_corners=False)
        else:
            skip=self._features.pop(self.refine_index);hidden=self.lane_head[:-1](fused);coarse=self.lane_head[-1](hidden)
            up=lambda t:F.interpolate(t,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            lane=up(coarse)+self.refine(torch.cat([up(hidden),self.refine_proj(skip)],1))
            lane=F.interpolate(lane,size=images.shape[-2:],mode='bilinear',align_corners=False)
        return {'detections':detections,'lane_logits':lane}
    def loss(self,output,batch,lane_weight=1.0,lane_pos_weight=8.0):
        logits=output['lane_logits'][:,0];target=batch['lane_mask'];valid=target!=255
        if valid.any():
            pred=logits[valid];truth=target[valid].float()
            bce=F.binary_cross_entropy_with_logits(pred,truth,pos_weight=pred.new_tensor(lane_pos_weight))
            prob=pred.sigmoid();dice=1-(2*(prob*truth).sum()+1)/(prob.sum()+truth.sum()+1)
            lane=bce+dice
        else:lane=logits.sum()*0
        det=logits.sum()*0
        if self.phase=='joint':
            values,_=self.detector.loss(batch,output['detections']);det=values.sum()/batch['img'].shape[0]
        return det+lane_weight*lane,{'detection':float(det.detach()),'lane':float(lane.detach()),'labelled_lane_pixels':int(valid.sum())}
    def save(self,path,metadata):
        torch.save({'format':'joint-road-v1','architecture':self.detector.yaml,'names':self.names,'decoder_width':self.decoder_width,'feature_indices':self.feature_indices,'refine_index':self.refine_index,'refine_width':self.refine_width,'state_dict':self.state_dict(),'metadata':metadata},path)
    @classmethod
    def load(cls,path,device='cpu'):
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck.get('format')!='joint-road-v1':raise ValueError('not a joint-road checkpoint')
        model=cls(architecture=ck['architecture'],names=ck['names'],decoder_width=ck['decoder_width'],feature_indices=ck.get('feature_indices'),refine_index=ck.get('refine_index'),refine_width=ck.get('refine_width',16))
        model.load_state_dict(ck['state_dict']);model.to(device);return model,ck['metadata']
