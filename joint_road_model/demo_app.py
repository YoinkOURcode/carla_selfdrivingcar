"""Local browser demo of the joint road model: browse CARLA episodes or upload an image, see lanes, crossings and lane centre live.

    python demo_app.py                                   # sample episode, inside the package
    python demo_app.py --frames C:\\data\\Mark --device cuda   # your full Mark folder

Binds to 127.0.0.1 only. Requires flask (not in the package's requirements.txt). Viewing only: nothing is written and no setting here changes the model.
"""
import argparse,base64,json,sys,time
from pathlib import Path
import cv2,numpy as np
from flask import Flask,jsonify,request,Response

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--package',type=Path,default=HERE);p.add_argument('--frames',type=Path,default=HERE/'sample_frames')
p.add_argument('--manifest',type=Path);p.add_argument('--splits',type=Path,default=HERE/'episode_splits.json');p.add_argument('--device',default='cpu');p.add_argument('--port',type=int,default=8790);a=p.parse_args()
PKG=a.package.resolve();FRAMES=a.frames.resolve();sys.path.insert(0,str(PKG))
from road_perception import RoadPerception
rp=RoadPerception(PKG/'joint_road_model.pt',PKG/'lane_center_calibration.json',a.device)
split=json.loads(a.splits.read_text()) if a.splits.is_file() else {}
if a.manifest:
    for it in json.loads(a.manifest.read_text())['items']:split.setdefault(it['path'].split('/')[0],it['split'])
ORDER={'test':0,'val':1,'train':2}
episodes=sorted([d.name for d in FRAMES.iterdir() if d.is_dir() and d.name.startswith('episode_')],key=lambda e:(ORDER.get(split.get(e),3),e))
app=Flask(__name__);app.config['MAX_CONTENT_LENGTH']=20*1024*1024

def jpg(img,q=88):return base64.b64encode(cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,q])[1].tobytes()).decode()

def infer(frame,conf,lane):
    rp.conf=conf;rp.lane_threshold=lane
    t=time.perf_counter();r=rp(frame);ms=1000*(time.perf_counter()-t);lc=r['lane_center']
    h,w=frame.shape[:2]
    return {'overlay':jpg(rp.draw(frame,r)),'original':jpg(frame),'ms':round(ms),'device':a.device,'size':[w,h],'crosswalks':r['crosswalks'],'lane_pixels':int(r['lane_mask'].sum()),
            'lane_center':{k:lc.get(k) for k in('mode','offset_norm','offset_px','heading_deg','low_confidence','lane_width_px_bottom')}}

def params():return float(request.values.get('conf',.25)),float(request.values.get('lane',.5))

@app.get('/api/episodes')
def api_episodes():return jsonify([{'name':e,'split':split.get(e,'?')} for e in episodes])
@app.get('/api/frames/<ep>')
def api_frames(ep):
    if ep not in episodes:return jsonify(error='unknown episode'),404
    return jsonify(sorted(f.name for f in (FRAMES/ep).iterdir() if f.suffix.lower() in('.png','.jpg','.jpeg')))
@app.get('/api/infer/<ep>/<name>')
def api_infer(ep,name):
    f=FRAMES/ep/name
    if ep not in episodes or f.parent!=FRAMES/ep or not f.is_file():return jsonify(error='unknown frame'),404
    return jsonify(infer(cv2.imread(str(f)),*params()))
@app.post('/api/upload')
def api_upload():
    fs=request.files.get('image')
    img=cv2.imdecode(np.frombuffer(fs.read(),np.uint8),cv2.IMREAD_COLOR) if fs else None
    if img is None:return jsonify(error='could not read that file as an image'),400
    return jsonify(infer(img,*params()))
@app.get('/')
def index():return Response(PAGE,mimetype='text/html')

PAGE=r"""<!doctype html><html lang=en><meta charset=utf-8><title>Joint road model demo</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{--bg:#14171c;--panel:#1d2229;--line:#303741;--text:#e8eaed;--dim:#9aa3ae;--accent:#4c9ffe}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif}
header{padding:12px 20px;border-bottom:1px solid var(--line);display:flex;gap:14px;align-items:baseline}h1{font-size:17px;margin:0}header span{color:var(--dim)}
main{display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:16px;padding:16px 20px;max-width:1500px;margin:auto}
@media(max-width:950px){main{grid-template-columns:1fr}}
#view{position:relative;background:#000;border-radius:8px;overflow:hidden;aspect-ratio:16/9}#view img{width:100%;height:100%;object-fit:contain;display:block}
#err{position:absolute;inset:auto 0 0 0;background:#8c2f2f;padding:8px 12px;display:none}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:12px}
select,button{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:7px 11px;font:inherit;cursor:pointer}
button:hover{border-color:var(--accent)}button.on{background:var(--accent);border-color:var(--accent);color:#06121f}input[type=range]{accent-color:var(--accent)}#frame{flex:1;min-width:160px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin-bottom:12px}.card h2{font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:var(--dim);margin:0 0 8px;font-weight:600}
.big{font-size:26px;font-weight:600;font-variant-numeric:tabular-nums}.kv{display:flex;justify-content:space-between;gap:10px;padding:2px 0;font-variant-numeric:tabular-nums}.kv span:first-child{color:var(--dim)}
#gauge{position:relative;height:22px;margin:8px 0 2px;border-radius:4px;background:#2a3038}#gauge i{position:absolute;top:0;bottom:0;left:50%;width:1px;background:#5a6470}#mark{position:absolute;top:-2px;bottom:-2px;width:6px;margin-left:-3px;background:var(--accent);border-radius:3px;display:none}
.scale{display:flex;justify-content:space-between;color:var(--dim);font-size:11px}
.warn{color:#f0b35a;margin-top:6px;display:none}label.s{display:block;margin-top:8px;color:var(--dim)}label.s input{width:100%}ul{margin:0;padding:0;list-style:none}
.legend b{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}.legend div{padding:1px 0}.note{color:var(--dim);font-size:12px}
</style>
<header><h1>Joint road model</h1><span>lane paint · pedestrian crossings · lane centre (final model v2b)</span></header>
<main><section>
<div id=view><img id=img alt="Model output for the selected frame"><div id=err role=alert></div></div>
<div class=row><button id=play>▶ Play</button><button id=prev aria-label="Previous frame">◀</button><button id=next aria-label="Next frame">▶</button><input id=frame type=range min=0 value=0 aria-label="Frame"><span id=fname class=note></span></div>
<div class=row><label for=ep>Episode</label><select id=ep></select><button id=orig>Show original</button><button id=up>Upload image…</button><input id=file type=file accept="image/*" hidden></div>
</section><aside>
<div class=card><h2>Lane centre</h2><div class=big id=off>–</div><div class=note id=offt></div>
<div id=gauge><i></i><div id=mark></div></div><div class=scale><span>on left line</span><span>centred</span><span>on right line</span></div>
<div style="margin-top:8px"><div class=kv><span>Lines found</span><span id=mode>–</span></div><div class=kv><span>Heading</span><span id=head>–</span></div></div><div class=warn id=low>Low confidence: a lane line was extended a long way, so treat this value with caution.</div></div>
<div class=card><h2>Crossings</h2><div class=big id=nc>–</div><ul id=cl class=note></ul></div>
<div class=card><h2>Settings</h2><label class=s>Crossing confidence ≥ <span id=cv></span><input id=conf type=range min=.05 max=.95 step=.05 value=.25></label><label class=s>Lane threshold ≥ <span id=lv></span><input id=lane type=range min=.1 max=.9 step=.05 value=.5></label>
<div class=note style="margin-top:6px">Reported results use 0.25 and 0.50. <button id=reset style="padding:2px 8px">Reset</button></div></div>
<div class=card legend><h2>Key</h2><div class=legend><div><b style="background:#0f0"></b>Lane paint</div><div><b style="background:#f0f"></b>Crossing box</div><div><b style="background:#0cf"></b>Lane edges used</div><div><b style="background:#f00"></b>Lane centre</div><div><b style="background:#fff"></b>Car position</div></div></div>
<div class=card><div class=kv><span>Inference</span><span id=ms>–</span></div><div class=kv><span>Image</span><span id=size>–</span></div><div class=note id=split style="margin-top:6px"></div></div>
</aside></main>
<script>
const $=i=>document.getElementById(i);let frames=[],cur=0,playing=false,last=null,showOrig=false,upload=null,token=0;
const SPLIT={test:'Reserved test episode: not used for training or model selection.',val:'Validation episode: used to choose the model, not to train it.',train:'Training episode: the model learned from this, so expect flattering results.'};
function err(m){$('err').textContent=m||'';$('err').style.display=m?'block':'none'}
function q(){return `conf=${$('conf').value}&lane=${$('lane').value}`}
function paint(){if(last)$('img').src='data:image/jpeg;base64,'+(showOrig?last.original:last.overlay)}
function show(d){last=d;paint();const c=d.lane_center,o=c.offset_norm;
 $('off').textContent=o==null?'No lane found':(o>=0?'+':'')+o.toFixed(3);
 $('offt').textContent=o==null?'':Math.abs(o)<.02?'lane widths: centred':`lane widths: car is ${o>0?'right':'left'} of centre`;
 $('mark').style.display=o==null?'none':'block';if(o!=null)$('mark').style.left=Math.max(0,Math.min(100,50+o*100))+'%';
 $('mode').textContent={both:'left and right',left_only:'left only',right_only:'right only',none:'none'}[c.mode];
 $('head').textContent=c.heading_deg==null?'–':c.heading_deg.toFixed(1)+'°';$('low').style.display=c.low_confidence?'block':'none';
 $('nc').textContent=d.crosswalks.length;$('cl').innerHTML=d.crosswalks.map(b=>`<li>confidence ${b.confidence.toFixed(2)}, ${Math.round(b.xyxy[2]-b.xyxy[0])}×${Math.round(b.xyxy[3]-b.xyxy[1])} px</li>`).join('');
 $('ms').textContent=`${d.ms} ms (${d.device})`;$('size').textContent=d.size.join('×')+(d.size[0]==1280&&d.size[1]==720?'':' (lane width calibrated for 1280×720)');}
async function run(){const t=++token;try{let r;
  if(upload){const f=new FormData();f.append('image',upload);r=await fetch('/api/upload?'+q(),{method:'POST',body:f})}
  else r=await fetch(`/api/infer/${$('ep').value}/${frames[cur]}?${q()}`);
  const d=await r.json();if(t!=token)return;if(!r.ok)throw Error(d.error||r.statusText);err();show(d)}catch(e){if(t==token){err(e.message);stop()}}}
function go(i){upload=null;cur=(i+frames.length)%frames.length;$('frame').value=cur;$('fname').textContent=`${frames[cur]} (${cur+1}/${frames.length})`;return run()}
async function loadEp(){upload=null;const e=$('ep').selectedOptions[0];$('split').textContent=SPLIT[e.dataset.split]||'';frames=await (await fetch('/api/frames/'+e.value)).json();$('frame').max=frames.length-1;await go(0)}
function stop(){playing=false;$('play').textContent='▶ Play';$('play').classList.remove('on')}
async function loop(){while(playing){const t=performance.now();await go(cur+1);await new Promise(r=>setTimeout(r,Math.max(0,500-(performance.now()-t))))}}
$('play').onclick=()=>{if(playing)stop();else{if(upload)return;playing=true;$('play').textContent='❚❚ Pause';$('play').classList.add('on');loop()}};
$('prev').onclick=()=>{stop();go(cur-1)};$('next').onclick=()=>{stop();go(cur+1)};$('frame').oninput=e=>{stop();go(+e.target.value)};$('ep').onchange=()=>{stop();loadEp()};
$('orig').onclick=()=>{showOrig=!showOrig;$('orig').classList.toggle('on',showOrig);paint()};
$('up').onclick=()=>$('file').click();$('file').onchange=e=>{if(e.target.files[0]){stop();upload=e.target.files[0];$('fname').textContent=upload.name;$('split').textContent='Uploaded image. The model was only trained on CARLA frames from this camera.';run()}e.target.value=''};
function lbl(){$('cv').textContent=(+$('conf').value).toFixed(2);$('lv').textContent=(+$('lane').value).toFixed(2)}
$('conf').oninput=$('lane').oninput=()=>{lbl();run()};$('reset').onclick=()=>{$('conf').value=.25;$('lane').value=.5;lbl();run()};
(async()=>{lbl();const es=await (await fetch('/api/episodes')).json();$('ep').innerHTML=es.map(e=>`<option value="${e.name}" data-split="${e.split}">${e.name} (${e.split})</option>`).join('');await loadEp()})().catch(e=>err(e.message));
</script></html>"""

if __name__=='__main__':app.run('127.0.0.1',a.port,threaded=False)
