"""Build a YOLO-format crosswalk dataset (train/val ONLY; test never written) from a joint manifest, for standalone
detector training with the standard Ultralytics recipe. Images are symlinks to the originals; labels are copied with
public class 7 remapped to internal class 0. Detection labels are identical in combined_v1 and aligned_v2."""
import json,argparse,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if a.out.exists():p.error('output exists')
md=json.loads(a.manifest.read_text());n={'train':0,'val':0,'boxes_train':0,'boxes_val':0}
for it in md['items']:
    s=it['split']
    if s not in('train','val'):continue
    name=it['path'].replace('/','_');img=a.out/'images'/s/name;lab=a.out/'labels'/s/(Path(name).stem+'.txt')
    img.parent.mkdir(parents=True,exist_ok=True);lab.parent.mkdir(parents=True,exist_ok=True)
    img.symlink_to(Path(it['image']).resolve())
    rows=[l.split() for l in Path(it['detection_labels']).read_text().splitlines() if l.strip()]
    assert all(r[0]=='7' and len(r)==5 for r in rows),it['path']
    lab.write_text(''.join('0 '+' '.join(r[1:])+'\n' for r in rows));n[s]+=1;n['boxes_'+s]+=len(rows)
(a.out/'data.yaml').write_text(f"path: {a.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: crosswalk\n")
(a.out/'build_info.json').write_text(json.dumps({'manifest':str(a.manifest.resolve()),'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'counts':n,'test_included':False},indent=1))
print(n)
