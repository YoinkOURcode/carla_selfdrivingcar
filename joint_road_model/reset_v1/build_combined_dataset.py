"""Freeze supplied labels without inventing human or independent approvals."""
from pathlib import Path
import json,hashlib,shutil,random
import numpy as np
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;D=B/'dataset_combined_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inv={r['path']:r for r in json.load(open(B/'inventory.json'))['items']}
published={r['path']:r for r in json.load(open(B/'review/items.json'))}
candidates={}
for p in sorted(B.rglob('output.json')):
 if 'claude_annotation' in p.parts:continue
 try:rows=json.load(open(p))['items']
 except (KeyError,TypeError):continue
 for r in rows:
  k=r.get('path');m=r.get('lane_mask')
  if k in published and m and Path(m).is_file() and sha(m)==published[k]['mask_sha256']:
   check=Image.new('L',(1280,720));cd=ImageDraw.Draw(check)
   for ck,cv in [('lane_polygons',1),('ignore_polygons',255)]:
    for poly in r.get(ck,[]):cd.polygon([tuple(t) for t in poly],fill=cv)
   if np.array_equal(np.asarray(check),np.asarray(Image.open(m))):candidates[k]=(r,p,'codex_published')
repairs=json.load(open(B/'full_annotation/user_relook_366_497/user_candidate_approvals.json'))['approvals'][-1]['items']
for a in repairs:
 p=Path(a['candidate_output']);assert sha(p)==a['output_sha256'],p
 r=next(r for r in json.load(open(p))['items'] if r['path']==a['path']);assert sha(r['lane_mask'])==a['mask_sha256']
 candidates[r['path']]=(r,p,'human_approved_repair')
assert set(candidates)==set(published),(len(candidates),set(published)-set(candidates))
reserved={r['path'] for r in json.load(open(B/'handoff/claude_1500_1999/assignment.json'))['items']}
claude=[]
for p in sorted((B/'claude_annotation').glob('episode_*/output.json')):
 for r in json.load(open(p))['items']:
  assert r['path'] in reserved and r['path'] not in candidates
  candidates[r['path']]=(r,p,'claude_as_supplied_no_codex_visual_review');claude.append(r['path'])
assert len(claude)==480 and set(candidates)==set(inv)
# Episodes are indivisible; evaluation is not claimed route-disjoint or blind.
calibration=set()
for p,r in published.items():
 if not r.get('batch','').startswith('Full folder'):calibration.add(p.split('/')[0])
episodes=sorted({r['episode'] for r in inv.values()});pool=[e for e in episodes if e not in calibration]
random.Random(20261030).shuffle(pool);test=set(pool[:10]);val=set(pool[10:25]);assert len(pool)>=25
D.mkdir(exist_ok=True);items=[];checks=[]
for path,(r,p,provenance) in sorted(candidates.items()):
 src=Path(inv[path]['source']);assert sha(src)==inv[path]['sha256']==r['source_sha256'],path
 arr=np.asarray(Image.open(r['lane_mask']));assert arr.shape==(720,1280) and set(np.unique(arr))<={0,1,255},path
 raster=Image.new('L',(1280,720));draw=ImageDraw.Draw(raster)
 for key,v in [('lane_polygons',1),('ignore_polygons',255)]:
  for poly in r.get(key,[]):
   assert len(poly)>=3 and all(0<=x<=1280 and 0<=y<=720 for x,y in poly),(path,key)
   draw.polygon([tuple(x) for x in poly],fill=v)
 assert np.array_equal(arr,np.asarray(raster)),path
 lines=[]
 for b in r['boxes']:
  x,y,X,Y=b['xyxy'];assert 0<=x<X<=1280 and 0<=y<Y<=720,(path,b)
  lines.append(f'7 {(x+X)/2560:.9f} {(y+Y)/1440:.9f} {(X-x)/1280:.9f} {(Y-y)/720:.9f}')
 mask=D/'lane_masks'/path;label=(D/'detection_labels'/path).with_suffix('.txt')
 mask.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(r['lane_mask'],mask);label.write_text('\n'.join(lines)+('\n' if lines else ''))
 ep=inv[path]['episode'];split='test' if ep in test else 'val' if ep in val else 'train'
 items.append(dict(path=path,image=str(src.resolve()),lane_mask=str(mask.relative_to(D)),detection_labels=str(label.relative_to(D)),episode=ep,split=split,reviewed=False,training_authorized=True,provenance=provenance,source_sha256=sha(src),mask_sha256=sha(mask),annotation_file=str(p),annotation_sha256=sha(p),uncertainties=r.get('uncertainties',r.get('uncertainty',[]))))
 checks.append(dict(path=path,source_sha256=sha(src),exact_raster=True,mask_sha256=sha(mask)))
assert len({r['source_sha256'] for r in items})==1999
manifest=dict(version=1,crosswalk_class_id=7,annotation_source='Codex published and approved repair versions plus Claude supplied labels; user authorized use without additional approval',review_policy='No new human or independent approval inferred. Training explicitly authorized on supplied labels.',evaluation_limitations='Episode-held-out reset evaluation with prior project exposure. Route overlap unknown; not a blind or scene-disjoint test.',split_seed=20261030,calibration_episodes_train_only=sorted(calibration),items=items)
for name,obj in [('manifest.json',manifest),('validation.json',dict(count=len(checks),claude_count=len(claude),approved_repairs=len(repairs),checks=checks)),('split.json',{s:sorted({r['episode'] for r in items if r['split']==s}) for s in ['train','val','test']})]:
 dest=D/name
 if dest.exists():assert json.load(open(dest))==obj,'Frozen artifact differs; choose a new version'
 else:dest.write_text(json.dumps(obj,indent=2))
print(json.dumps(dict(images=len(items),claude=len(claude),repairs=len(repairs),split_counts={s:sum(r['split']==s for r in items) for s in ['train','val','test']},manifest_sha256=sha(D/'manifest.json')),indent=2))
