"""Self-test for the joint road model package. Run from the package folder:

    python selftest.py            # CPU
    python selftest.py --device cuda   (or mps)

Checks, in order:
  1. Every file listed in SHA256SUMS is present and unmodified.
  2. The model loads and runs on the frames in sample_frames/.
  3. Its outputs match sample_frames/expected/ (produced by the same package on the machine that trained it).
     Small tolerances allow for different CPUs/GPUs and library versions.
  4. Reports inference speed.
Overlays are written to selftest_out/ so you can look at them. Exit code 0 = PASS, 1 = FAIL.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np,cv2

HERE=Path(__file__).resolve().parent

def box_iou(a,b):
    ix=max(0,min(a[2],b[2])-max(a[0],b[0]));iy=max(0,min(a[3],b[3])-max(a[1],b[1]));i=ix*iy
    u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i;return i/u if u>0 else 0.

def check_sums():
    bad=[]
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        h,f=line.split('  ',1);p=HERE/f
        if not p.is_file():bad.append(f'missing {f}')
        elif hashlib.sha256(p.read_bytes()).hexdigest()!=h:bad.append(f'changed {f}')
    return bad

def compare(name,r,exp,mask_exp):
    """Returns list of problems (empty = match)."""
    out=[]
    m=r['lane_mask'];inter=(m&mask_exp).sum();union=(m|mask_exp).sum();iou=inter/union if union else 1.
    if iou<0.97:out.append(f'lane mask IoU vs expected {iou:.3f} < 0.97')
    lc,le=r['lane_center'],exp['lane_center']
    if lc['mode']!=le['mode']:out.append(f"lane-centre mode {lc['mode']} != expected {le['mode']}")
    elif le['offset_norm'] is not None and abs(lc['offset_norm']-le['offset_norm'])>0.02:
        out.append(f"offset_norm {lc['offset_norm']:+.3f} vs expected {le['offset_norm']:+.3f}")
    # crossings: every confident expected box must be found (IoU >= .9); boxes near the 0.25 threshold may appear/disappear
    got=list(r['crosswalks'])
    for e in exp['crosswalks']:
        j=max(range(len(got)),key=lambda k:box_iou(got[k]['xyxy'],e['xyxy']),default=None)
        if j is not None and box_iou(got[j]['xyxy'],e['xyxy'])>=0.9:got.pop(j)
        elif e['confidence']>0.30:out.append(f"missing crossing conf {e['confidence']:.2f} at {[round(v) for v in e['xyxy']]}")
    out+= [f"extra crossing conf {g['confidence']:.2f}" for g in got if g['confidence']>0.30]
    return out,iou

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');a=p.parse_args()
    print('1) checksums ...',end=' ');bad=check_sums()
    print('OK' if not bad else 'FAIL');[print('   ',b) for b in bad]
    import torch,ultralytics
    print(f'   python {sys.version.split()[0]}, torch {torch.__version__}, ultralytics {ultralytics.__version__}, opencv {cv2.__version__}, numpy {np.__version__}')
    sys.path.insert(0,str(HERE))
    from road_perception import RoadPerception
    print('2) loading model ...',end=' ');rp=RoadPerception(HERE/'joint_road_model.pt',HERE/'lane_center_calibration.json',a.device);print('OK')
    exp_all=json.loads((HERE/'sample_frames/expected/expected_outputs.json').read_text())
    out_dir=HERE/'selftest_out';out_dir.mkdir(exist_ok=True);fails=0;times=[]
    print('3) comparing outputs with expected:')
    for name,exp in exp_all.items():
        frame=cv2.imread(str(HERE/'sample_frames'/name))
        t=time.perf_counter();r=rp(frame);times.append(time.perf_counter()-t)
        mask_exp=cv2.imread(str(HERE/'sample_frames/expected'/(Path(name).stem+'_lane.png')),cv2.IMREAD_GRAYSCALE)>0
        probs,iou=compare(name,r,exp,mask_exp);fails+=bool(probs)
        lc=r['lane_center'];off='-' if lc['offset_norm'] is None else f"{lc['offset_norm']:+.3f}"
        print(f"   {'PASS' if not probs else 'FAIL'}  {name:12s} crossings={len(r['crosswalks'])} lane-centre={lc['mode']:10s} offset={off:>7s} lane-mask-agreement={iou:.3f}")
        for q in probs:print('         ',q)
        cv2.imwrite(str(out_dir/(Path(name).stem+'_overlay.jpg')),rp.draw(frame,r))
    sp=times[1:] or times
    print(f'4) speed on {a.device}: {1000*np.mean(sp):.0f} ms/frame ({1/np.mean(sp):.1f} FPS), first frame {1000*times[0]:.0f} ms (warm-up)')
    ok=not bad and not fails
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}  ({len(exp_all)-fails}/{len(exp_all)} frames match; overlays in {out_dir.name}/)")
    sys.exit(0 if ok else 1)

if __name__=='__main__':main()
