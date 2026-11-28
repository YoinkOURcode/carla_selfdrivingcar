"""Evaluate a joint-road checkpoint on VALIDATION with v1 metric definitions plus multi-raster lane scores.
Agreement with supplied labels only; no claim of independent ground truth. Test split is refused unless --final-test (one-time, marker file):
reserved test evaluation happens once, only after development gates pass, with a separate explicit command."""
import sys,json,argparse,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import JointRoadModel
from data import RoadDataset,validate_splits
from eval_common import Evaluator,gate

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rect',type=int,nargs=2,metavar=('W','H'),help='letterbox size used in training; default square 640');p.add_argument('--device',default='cpu')
    p.add_argument('--sheets',type=int,default=0,help='write comparison sheets for all (1) or none (0)');p.add_argument('--native',type=int,default=1)
    p.add_argument('--final-test',action='store_true',help='ONE-TIME reserved test evaluation; refuses if training_v2/RESERVED_TEST_USED.json exists')
    a=p.parse_args();split='val';marker=Path(__file__).resolve().parent/'training_v2'/'RESERVED_TEST_USED.json'
    if a.final_test:
        if marker.exists():p.error(f'reserved test already used: {marker.read_text()}')
        split='test'
    if a.out.exists():p.error('output directory exists; use a fresh one')
    validate_splits(a.manifest);torch.set_num_threads(4)
    W,H=a.rect if a.rect else (640,640)
    ds=RoadDataset(a.manifest,split,W,require_reviewed=False,rect=(W,H));v1=RoadDataset(a.manifest,split,640,require_reviewed=False)
    if a.final_test:
        import datetime;marker.write_text(json.dumps({'checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),'manifest':str(Path(a.manifest).resolve()),'out':str(a.out.resolve()),'time':datetime.datetime.now().isoformat()},indent=1))
    assert all(r.get('training_authorized') for r in ds.items)
    model,meta=JointRoadModel.load(a.checkpoint,a.device);model.eval();a.out.mkdir(parents=True)
    def sheet(i,r,out,prob,boxes,gt):
        arr=(r['img'].numpy().transpose(1,2,0)*255).astype('uint8');truth=r['lane_mask'].numpy()==1;mask=prob>=.5
        S=Image.new('RGB',(2*W,H))
        for k,(mm,bb,title) in enumerate([(truth,gt,'Supplied labels'),(mask,boxes,'Predictions')]):
            vis=arr.copy();vis[mm]=(vis[mm]*.4+np.array([0,255,140])*.6).astype('uint8');im=Image.fromarray(vis);d=ImageDraw.Draw(im)
            for b in bb:d.rectangle(b.tolist(),outline='orange',width=2)
            d.text((10,10),title,fill='white');S.paste(im,(k*W,0))
        S.save(a.out/(ds.items[i]['path'].replace('/','_')+'.jpg'))
    ev=Evaluator(ds,v1,native=bool(a.native),device=a.device)
    res,records=ev.run(model,callback=sheet if a.sheets else None)
    for r in records:r['lane_iou']=r.get('lane_iou_v1_640')
    metrics=dict(split=split,images=len(ds),input_letterbox=[W,H],checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        manifest=str(Path(a.manifest).resolve()),manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),checkpoint_metadata={k:v for k,v in meta.items() if k!='record'},
        **res,development_gate_v1_protocol=gate(res,'v1_640'),
        limitations='Agreement with supplied labels. Episode held out, prior project exposure, unknown route overlap; not blind or deployment validation. Validation also used for checkpoint selection in v2 runs, so these figures are optimistic for the selected checkpoint.',records=records)
    (a.out/'metrics.json').write_text(json.dumps(metrics,indent=2,default=float))
    print(json.dumps({k:v for k,v in metrics.items() if k not in('records','checkpoint_metadata')},indent=1,default=float))
if __name__=='__main__':main()
