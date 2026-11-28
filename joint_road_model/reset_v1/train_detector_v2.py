"""Standalone crosswalk detector, standard Ultralytics recipe (SGD/auto optimiser, mosaic, flips, HSV/scale aug, close_mosaic),
YOLO11s COCO-pretrained, train/val only. best.pt is Ultralytics' val-fitness choice (validation used for selection; disclosed).
Purpose: (1) measure achievable detection with a standard recipe; (2) supply detector weights for a joint model whose lane head is
trained on top with the detector frozen (sequential training)."""
import sys,argparse
from pathlib import Path
from ultralytics import YOLO
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--name',required=True);p.add_argument('--imgsz',type=int,default=1280)
p.add_argument('--epochs',type=int,default=60);p.add_argument('--batch',type=int,default=8);p.add_argument('--device',default='mps');a=p.parse_args()
root=Path(__file__).resolve().parent/'training_v2'
if (root/a.name).exists():sys.exit('run exists')
YOLO(str(Path(__file__).resolve().parents[2]/'annotation_pilot/models/yolo11s.pt')).train(data=a.data,imgsz=a.imgsz,epochs=a.epochs,batch=a.batch,device=a.device,
    workers=4,patience=15,seed=0,deterministic=False,project=str(root),name=a.name,exist_ok=False,plots=True,amp=False)
