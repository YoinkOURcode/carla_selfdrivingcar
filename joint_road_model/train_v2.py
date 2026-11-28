"""v2 training: train.py plus rectangular letterbox, optional stride-2 lane refinement, photometric augmentation,
per-epoch validation lane IoU (v1 640 protocol) + crosswalk AP/P/R, and pre-registered checkpoint selection.
Writes only to a NEW --out directory. Never reads the test split."""
import argparse,json,random,hashlib,time,platform,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import JointRoadModel,ROOT
from data import RoadDataset,collate,validate_splits
from eval_common import Evaluator,gate

HERE=Path(__file__).resolve().parent
CODE=[HERE/'model.py',HERE/'data.py',HERE/'eval_common.py',HERE/'train_v2.py']

def winit(worker_id):
    # torch.initial_seed() inside a worker = generator-derived base seed + worker id, so augmentation is reproducible and distinct per worker.
    info=torch.utils.data.get_worker_info();info.dataset.rng=np.random.default_rng(torch.initial_seed()%2**32)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--warmup-epochs',type=int,default=3)
    p.add_argument('--rect',type=int,nargs=2,default=[960,544],metavar=('W','H'))
    p.add_argument('--batch',type=int,default=4);p.add_argument('--device',default='mps');p.add_argument('--workers',type=int,default=2)
    p.add_argument('--weights',type=Path,default=ROOT/'annotation_pilot/models/yolo11s.pt');p.add_argument('--init-checkpoint',type=Path)
    p.add_argument('--refine-index',type=int,default=None,help='YOLO layer index for stride-2 lane refinement (0=stem); omit for v1 decoder')
    p.add_argument('--augment',action='store_true');p.add_argument('--detector-lr',type=float,default=1e-4);p.add_argument('--lane-lr',type=float,default=1e-3)
    p.add_argument('--lane-pos-weight',type=float,default=8.0);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--allow-authorized-supplied-labels',action='store_true')
    p.add_argument('--lane-only',action='store_true',help='sequential training: detector (incl. backbone, BN stats) frozen for ALL epochs; only the lane decoder trains; selection over all epochs')
    a=p.parse_args()
    if a.out.exists():p.error('output directory exists; choose a new run')
    if any(v%32 for v in a.rect):p.error('rect dims must be divisible by 32')
    if a.lane_only:a.warmup_epochs=a.epochs
    elif a.epochs<=a.warmup_epochs:p.error('epochs must exceed warmup-epochs')
    validate_splits(a.manifest);md=json.loads(a.manifest.read_text())
    if a.allow_authorized_supplied_labels and not all(r.get('training_authorized') for r in md['items']):p.error('Every supplied row must explicitly authorize training')
    rr=not a.allow_authorized_supplied_labels
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    train=RoadDataset(a.manifest,'train',a.rect[0],require_reviewed=rr,rect=a.rect,augment=a.augment,seed=a.seed)
    val=RoadDataset(a.manifest,'val',a.rect[0],require_reviewed=rr,rect=a.rect);val640=RoadDataset(a.manifest,'val',640,require_reviewed=rr)
    if a.init_checkpoint:
        model,im=JointRoadModel.load(a.init_checkpoint,'cpu')
        if not im.get('trained'):p.error('initial joint checkpoint is not trained')
        if a.refine_index is not None and model.refine_index is None:
            fresh=JointRoadModel(a.weights,refine_index=a.refine_index);miss=fresh.load_state_dict(model.state_dict(),strict=False)
            assert all(k.startswith('refine') for k in miss.missing_keys) and not miss.unexpected_keys;model=fresh
    else:model=JointRoadModel(a.weights,refine_index=a.refine_index)
    model.to(a.device)
    g=torch.Generator();g.manual_seed(a.seed)
    dl=DataLoader(train,batch_size=a.batch,shuffle=True,collate_fn=collate,num_workers=a.workers,worker_init_fn=winit if a.workers else None,generator=g,persistent_workers=a.workers>0)
    vl=DataLoader(val,batch_size=a.batch,shuffle=False,collate_fn=collate,num_workers=0)
    ev=Evaluator(val,val640,native=False,device=a.device)
    a.out.mkdir(parents=True)
    import ultralytics,torchvision,PIL,cv2
    config={'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'manifest_sha256':sha(a.manifest),'annotation_source':md.get('annotation_source'),
        'manifest_version':md.get('version'),'train_images':len(train),'val_images':len(val),'test_split_read':False,
        'code_sha256':{str(c.relative_to(ROOT)):sha(c) for c in CODE},
        'environment':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'torchvision':torchvision.__version__,'ultralytics':ultralytics.__version__,'ultralytics_path':ultralytics.__file__,'numpy':np.__version__,'Pillow':PIL.__version__,'opencv':cv2.__version__},
        'determinism_note':'MPS kernels are not bitwise deterministic; seeds fix data order/augmentation/initialisation only.',
        'selection_rule':('lane-only: all epochs eligible; ' if a.lane_only else '')+'PRE-REGISTERED before training: best.pt = joint-phase epoch maximising mean(val lane IoU on v1 640 raster, val crosswalk AP50). best_val_loss.pt kept for the v1 criterion. last.pt = final epoch. Selection uses validation, so selected-checkpoint validation figures are optimistic.',
        'v1_comparison':'Lane metric on v1 640 raster reproduces v1 evaluate_combined.py exactly for 640 checkpoints (verified on joint_v1/best.pt).'}
    (a.out/'run_config.json').write_text(json.dumps(config,indent=2))
    optimizer=None;previous=None;best_loss=float('inf');best_score=-1;history=[]
    for epoch in range(a.epochs):
        t0=time.monotonic();phase='lane_warmup' if epoch<a.warmup_epochs else 'joint';model.set_phase(phase)
        if phase!=previous:
            optimizer=torch.optim.AdamW([{'params':[q for q in model.detector.parameters() if q.requires_grad],'lr':a.detector_lr},{'params':model.lane_parameters(),'lr':a.lane_lr}],weight_decay=1e-4);previous=phase
        model.train();tot=0
        for step,batch in enumerate(dl,1):
            batch={k:v.to(a.device) for k,v in batch.items() if k!='path'};optimizer.zero_grad()
            loss,parts=model.loss(model(batch['img']),batch,lane_pos_weight=a.lane_pos_weight)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            if parts['labelled_lane_pixels']==0:raise RuntimeError('Batch has no labelled lane pixels')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step();tot+=float(loss.detach())
            if step%100==0:print(json.dumps({'progress_epoch':epoch+1,'step':step,'steps':len(dl),'elapsed_seconds':round(time.monotonic()-t0,1)}),flush=True)
        model.eval();vtot=0
        with torch.no_grad():
            for batch in vl:
                batch={k:v.to(a.device) for k,v in batch.items() if k!='path'};l,_=model.loss(model(batch['img']),batch,lane_pos_weight=a.lane_pos_weight);vtot+=float(l)
        res,_=ev.run(model)
        lane=res['lane']['v1_640'];cw=res['crosswalk'];score=(lane['iou']+cw['AP50'])/2
        rec={'epoch':epoch+1,'phase':phase,'elapsed_seconds':round(time.monotonic()-t0,1),'train_loss':tot/len(dl),'val_loss':vtot/len(vl),
             'lane_iou_v1_640':lane['iou'],'lane_precision':lane['precision'],'lane_recall':lane['recall'],'lane_iou_model_raster':res['lane']['model']['iou'],
             'crosswalk_precision':cw['precision'],'crosswalk_recall':cw['recall'],'AP50':cw['AP50'],'AP50_95':cw['AP50_95'],'selection_score':score,'gate':gate(res)['all']}
        history.append(rec);print(json.dumps(rec),flush=True)
        meta={'trained':True,'lane_labels':md.get('annotation_source'),'manifest':str(a.manifest.resolve()),'manifest_sha256':config['manifest_sha256'],'record':rec,'phase':phase,
              'input_letterbox':a.rect,'initial_joint_checkpoint':str(a.init_checkpoint.resolve()) if a.init_checkpoint else None,'internal_crosswalk_id':model.crosswalk_id,'public_crosswalk_id':7,'requires_detection_evaluation':True}
        model.save(a.out/'last.pt',meta)
        if phase=='joint' or a.lane_only:
            if rec['val_loss']<best_loss:best_loss=rec['val_loss'];model.save(a.out/'best_val_loss.pt',meta)
            if score>best_score:best_score=score;model.save(a.out/'best.pt',meta)
        (a.out/'history.json').write_text(json.dumps(history,indent=2))
    print('done; evaluate best.pt with reset_v1/evaluate_v2.py (validation only)')

if __name__=='__main__':main()
