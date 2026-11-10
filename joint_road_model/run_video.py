"""Run the joint road model on a video file or a folder of frames; write an overlay video and a per-frame CSV.

    python run_video.py sample_frames/episode_0014 --out demo_out              # folder of .png/.jpg frames (sorted by name)
    python run_video.py drive.mp4 --out demo_out --device cuda                  # any video OpenCV can read

Outputs in --out:
  overlay.mp4   lane paint (green), crossing boxes (magenta), ego-lane boundaries (cyan) and centre line (red)
  frames.csv    frame, lane_mode, offset_norm, offset_px, heading_deg, low_confidence, n_crossings, max_crossing_conf, ms
CSV columns match RoadPerception outputs (see README). offset_norm: 0 = centred, +0.5 = car on right boundary, -0.5 = on left.
"""
import argparse,csv,sys,time
from pathlib import Path
import cv2

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from road_perception import RoadPerception

def frames(src):
    src=Path(src)
    if src.is_dir():
        for f in sorted(p for p in src.iterdir() if p.suffix.lower() in('.png','.jpg','.jpeg')):yield f.name,cv2.imread(str(f))
    else:
        cap=cv2.VideoCapture(str(src));i=0
        while True:
            ok,fr=cap.read()
            if not ok:break
            yield f'{i:06d}',fr;i+=1

def main():
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--fps',type=float,default=5.0,help='output video frame rate')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rp=RoadPerception(HERE/'joint_road_model.pt',HERE/'lane_center_calibration.json',a.device);vw=None
    with open(a.out/'frames.csv','w',newline='') as fh:
        w=csv.writer(fh);w.writerow(['frame','lane_mode','offset_norm','offset_px','heading_deg','low_confidence','n_crossings','max_crossing_conf','ms'])
        for name,fr in frames(a.source):
            t=time.perf_counter();r=rp(fr);ms=1000*(time.perf_counter()-t);lc=r['lane_center']
            vis=rp.draw(fr,r)
            if vw is None:vw=cv2.VideoWriter(str(a.out/'overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),a.fps,(vis.shape[1],vis.shape[0]))
            vw.write(vis)
            f=lambda v:'' if v is None else f'{v:.4f}'
            w.writerow([name,lc['mode'],f(lc['offset_norm']),f(lc['offset_px']),f(lc['heading_deg']),lc.get('low_confidence',''),len(r['crosswalks']),
                        f(max((b['confidence'] for b in r['crosswalks']),default=None)),f'{ms:.0f}'])
            print(f"{name}: lane={lc['mode']:10s} offset={f(lc['offset_norm']):>8s} crossings={len(r['crosswalks'])} {ms:.0f} ms")
    if vw:vw.release()
    print(f'wrote {a.out/"overlay.mp4"} and {a.out/"frames.csv"}')

if __name__=='__main__':main()
