"""Ego-lane centre estimation from a binary lane-paint mask (image space; no camera calibration or metric distances).

Assumes a forward-facing camera mounted on the vehicle's centreline (true for the CARLA "Mark" episodes), so image column
x0 = W/2 is the vehicle's lateral position.

Method
  1. Connected components of the lane mask below the horizon; keep line-like components and fit x = a*y + b to each.
  2. Merge collinear components (dashes of the same marking) by their x position at the bottom row and slope.
  3. Ego boundaries = nearest marking left of x0 and nearest right of x0 at the bottom row (extrapolated).
  4. Lane width in pixels is linear in the row for straight markings: w(y) = k*(y - y_h). k and y_h are calibrated once from
     training LABEL masks (calibrate()). The pair is rejected if its width is implausible, keeping the side closer to w/2 from x0.
  5. Centre line = midpoint of both boundaries, or one boundary +/- w(y)/2 when only one side is painted (e.g. kerb on the other side).

Output offset_norm = (x0 - centre_x)/w at the bottom row: 0 = centred, +0.5 = vehicle on the right boundary, -0.5 = on the left.
heading_deg = angle of the centre line from image vertical (positive = centre line leans right going up the image).
low_confidence = a used boundary had to be extrapolated over more than half the road region (e.g. intersections, occlusion).
"""
import json
from pathlib import Path
import numpy as np
import cv2

DEFAULT_CALIBRATION={'k':None,'y_h':None,'image_size':[1280,720]}

class LaneCenterEstimator:
    def __init__(self,calibration=None,scale=0.5,min_area=12,min_yspan=6,max_abs_slope=4.0,merge_frac=0.12):
        c=dict(DEFAULT_CALIBRATION,**(calibration or {}))
        self.k=c['k'];self.y_h=c['y_h'];self.cal_size=c['image_size'];self.s=scale
        self.min_area=min_area;self.min_yspan=min_yspan;self.max_abs_slope=max_abs_slope;self.merge_frac=merge_frac

    @classmethod
    def from_file(cls,path):return cls(json.loads(Path(path).read_text()))

    def width(self,y,H):
        """Expected ego-lane width (px, full resolution) at row y for an image of height H."""
        f=H/self.cal_size[1];return max(0.,self.k*(y/f-self.y_h))*f

    def lines(self,mask):
        """Fit straight lines to line-like components. Returns list of dicts in FULL-resolution coordinates."""
        H,W=mask.shape;s=self.s
        m=cv2.resize(mask.astype(np.uint8)*255,(round(W*s),round(H*s)),interpolation=cv2.INTER_AREA)>40
        if self.y_h is not None:m[:max(0,int((self.y_h*H/self.cal_size[1]+4)*s))]=False
        n,lab,st,_=cv2.connectedComponentsWithStats(m.astype(np.uint8),connectivity=8);out=[]
        for c in range(1,n):
            if st[c,4]<self.min_area or st[c,3]<self.min_yspan:continue
            ys,xs=np.nonzero(lab==c);ys=ys/s;xs=xs/s
            a,b=np.polyfit(ys,xs,1)
            if abs(a)>self.max_abs_slope:continue
            out.append({'a':float(a),'b':float(b),'y0':float(ys.min()),'y1':float(ys.max()),'n':int(len(ys)),'ys':ys,'xs':xs})
        # Merge pieces of the same marking (dashes). Short/far dashes have unreliable slopes, so a piece joins a group when its
        # pixels lie on the GROUP's line (median horizontal distance < merge_frac of the local lane width), largest pieces first.
        out.sort(key=lambda l:-l['n']);groups=[]
        for l in out:
            ym=float(np.median(l['ys']));tol=self.merge_frac*(self.width(ym,H) if self.k else 0.2*W*max(0.,ym/H-0.45)/0.55)+2
            best=None
            for g in groups:
                d=float(np.median(np.abs(l['xs']-(g['a']*l['ys']+g['b']))))
                if d<tol and (best is None or d<best[0]):best=(d,g)
            if best:
                g=best[1];ys=np.concatenate([g['ys'],l['ys']]);xs=np.concatenate([g['xs'],l['xs']]);a,b=np.polyfit(ys,xs,1)
                g.update(a=float(a),b=float(b),y0=float(ys.min()),y1=float(ys.max()),n=int(len(ys)),ys=ys,xs=xs)
            else:groups.append(dict(l))
        return groups

    def estimate(self,mask,x0=None):
        H,W=mask.shape;x0=W/2 if x0 is None else x0;yb=H-1
        G=self.lines(mask)
        for g in G:g['xb']=g['a']*yb+g['b']
        wb=self.width(yb,H) if self.k else None
        lim=wb*1.0 if wb else W/2
        left=[g for g in G if x0-lim<g['xb']<x0];right=[g for g in G if x0<g['xb']<x0+lim]
        L=max(left,key=lambda g:g['xb']) if left else None;R=min(right,key=lambda g:g['xb']) if right else None
        if L and R and wb:
            w=R['xb']-L['xb']
            if not 0.6*wb<w<1.6*wb:  # implausible pair: keep the side whose distance from x0 is closest to half a lane
                if abs((x0-L['xb'])-wb/2)<=abs((R['xb']-x0)-wb/2):R=None
                else:L=None
        mode='both' if L and R else 'left_only' if L else 'right_only' if R else 'none'
        res={'mode':mode,'x0':x0,'image_size':[W,H]}
        if mode=='none':return {**res,'offset_norm':None,'offset_px':None,'heading_deg':None,'center_line':[],'left':None,'right':None}
        if mode!='both' and not self.k:return {**res,'offset_norm':None,'offset_px':None,'heading_deg':None,'center_line':[],'left':_pub(L),'right':_pub(R)}
        ytop=min(g['y0'] for g in (L,R) if g);ys=np.linspace(ytop,yb,12)
        def cx(y):
            if L and R:return (L['a']*y+L['b']+R['a']*y+R['b'])/2
            if L:return L['a']*y+L['b']+self.width(y,H)/2
            return R['a']*y+R['b']-self.width(y,H)/2
        c=np.array([cx(y) for y in ys]);wref=(R['xb']-L['xb']) if (L and R) else wb
        slope=np.polyfit(ys,c,1)[0]  # dx/dy of centre line
        # extrapolation: share of the below-horizon height between a boundary's nearest painted pixel and the bottom row
        yh=self.y_h*H/self.cal_size[1] if self.y_h is not None else 0.
        ext=max((yb-g['y1'])/max(1.,yb-yh) for g in (L,R) if g)
        res['extrapolated_fraction']=float(ext);res['low_confidence']=bool(ext>0.5)
        return {**res,'offset_px':float(x0-c[-1]),'offset_norm':float((x0-c[-1])/wref) if wref else None,'lane_width_px_bottom':float(wref) if wref else None,
                'heading_deg':float(np.degrees(np.arctan(-slope))),'center_line':[[float(x),float(y)] for x,y in zip(c,ys)],'left':_pub(L),'right':_pub(R)}

    def draw(self,img,res,color=(0,0,255),thick=3):
        H=img.shape[0]
        for side in('left','right'):
            l=res.get(side)
            if l:cv2.line(img,(int(l['a']*l['y0']+l['b']),int(l['y0'])),(int(l['a']*(H-1)+l['b']),H-1),(255,200,0),2)
        pts=np.array(res.get('center_line') or [],np.int32)
        if len(pts):cv2.polylines(img,[pts],False,color,thick)
        cv2.line(img,(int(res['x0']),H-1),(int(res['x0']),H-40),(255,255,255),2)
        return img

def _pub(g):return None if g is None else {k:g[k] for k in('a','b','y0','y1','n','xb') if k in g}

def calibrate(masks,image_size=(1280,720)):
    """Estimate w(y)=k*(y-y_h) from LABEL masks: medians over frames whose two nearest markings straddle the image centre."""
    est=LaneCenterEstimator();ks=[];yhs=[];wbs=[]
    for m in masks:
        H,W=m.shape;r=est.estimate(m)
        if r['mode']!='both':continue
        L,R=r['left'],r['right'];k=R['a']-L['a']
        if k<=0.05:continue
        ks.append(k);yhs.append(-(R['b']-L['b'])/k);wbs.append(R['xb']-L['xb'])
    ks,yhs,wbs=map(np.array,(ks,yhs,wbs));wm=np.median(wbs);keep=np.abs(wbs-wm)<0.35*wm
    return {'k':float(np.median(ks[keep])),'y_h':float(np.median(yhs[keep])),'image_size':list(image_size),'frames_used':int(keep.sum()),'frames_with_pair':int(len(ks)),
            'median_width_bottom_px':float(np.median(wbs[keep])),'source':'training-split label masks (aligned_v2)'}
