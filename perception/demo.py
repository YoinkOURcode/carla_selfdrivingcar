import argparse
import hashlib
import io
import json
import os
import re
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo_config"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

import torch
from PIL import Image, ImageOps
from ultralytics import YOLO, settings
from ultralytics.engine.results import Boxes

from dedupe import merge_duplicates

MAX_UPLOAD_BYTES = 25 * 1024 * 1024

PAGE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CARLA perception demo</title>
<style>
body{margin:0;background:#101722;color:#eef3fa;font:16px system-ui}main{max-width:1180px;margin:auto;padding:28px}
h1{font-size:28px;margin:0 0 6px}h2{font-size:22px;margin:0 0 4px}h3{font-size:17px;margin:22px 0 4px}.muted{color:#b3c1d3}button,select{font:inherit;padding:10px 16px;border-radius:8px;border:1px solid #4b5d76;background:#223149;color:white;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}#predict{background:#2363d9;border-color:#2363d9}.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:22px 0}
.scene{background:#080c12;border-radius:12px;text-align:center;min-height:250px}.scene img{width:100%;max-width:1067px;vertical-align:middle;border-radius:12px}
.panel{background:#192334;padding:16px;border-radius:10px;margin-top:16px}#status{min-height:24px;margin:12px 0}.tag{display:inline-block;margin:5px 8px 5px 0;padding:6px 10px;background:#293b54;border-radius:6px}
label{display:inline-flex;gap:8px;align-items:center}.error{color:#ffa6a6}footer{line-height:1.6;margin-top:20px}kbd{background:#293b54;padding:2px 5px;border-radius:4px}
.wrap{overflow-x:auto}table{border-collapse:collapse;margin:8px 0;font-variant-numeric:tabular-nums;min-width:100%}th,td{padding:6px 10px;border-bottom:1px solid #2c3d55;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:#b3c1d3;font-weight:600}tr.total td{font-weight:600;border-top:2px solid #4b5d76}
.low{color:#ffb86b;font-weight:600}.bad{background:#4a2530}.good{color:#8fe3a8}.chip{display:inline-block;padding:2px 8px;border-radius:5px;background:#293b54;font-size:14px}
.split-train{background:#2b4a3a}.split-val{background:#2b3f63}.split-test{background:#5a3d2a}.split-excluded,.split-unknown{background:#444}
ul.findings{margin:8px 0;padding-left:20px;line-height:1.55}ul.findings li{margin-bottom:6px}details>summary{cursor:pointer;font-size:22px;font-weight:600}
.figure{background:white;border-radius:8px;padding:8px;max-width:760px}.figure img{width:100%}
</style><main><h1>CARLA perception demo</h1>
<div class="muted">Vehicles, traffic lights, pedestrians and speed limits from a driving camera</div>
<div class="toolbar"><button id="previous">Previous</button><select id="frames" aria-label="Choose image"></select><button id="next">Next</button><button id="predict">Run inference</button><button id="upload">Upload image</button><input id="file" type="file" accept="image/png,image/jpeg,image/webp,image/bmp" hidden><label><input id="overlay" type="checkbox">Show prediction</label></div>
<div id="status" role="status">Loading examples...</div><div class="scene"><img id="image" alt="CARLA driving scene"></div>
<div class="panel"><strong>Detections</strong> <span id="frameSplit"></span><div id="detections">Run inference to detect objects in this image.</div><div id="compare" class="wrap"></div></div>
<details class="panel" id="statsPanel" open><summary>Dataset and model statistics</summary><div id="stats">Loading statistics...</div></details>
<footer class="muted">Provisional seven-class model, retrained on corrected label copies. Display confidence: 0.25. No clean test score yet; figures come from Round 2 validation, which also selected the model.<br>Yellow lights are still often missed. Conflicting light states can occur. Count checks cover the first four classes only. Duplicate boxes are merged for display; the statistics below are scored before merging.<br>Uploaded images are added to the list for this session only and are not part of any evaluation split.<br>Keyboard: <kbd>Left</kbd> / <kbd>Right</kbd> change image, <kbd>Space</kbd> runs inference, <kbd>T</kbd> toggles prediction.</footer>
</main><script>
let examples=[], index=0, busy=false, classNames=[];const results=new Map();const $=id=>document.getElementById(id);
const pct=v=>(100*v).toFixed(1)+'%';const f3=v=>Number(v).toFixed(3);const sum=a=>a.reduce((x,y)=>x+y,0);
const esc=v=>String(v).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function table(head,rows,totalLast){return '<div class="wrap"><table><tr>'+head.map(h=>'<th>'+h+'</th>').join('')+'</tr>'+rows.map((r,i)=>'<tr'+(totalLast&&i===rows.length-1?' class="total"':'')+'>'+r.map(c=>'<td>'+c+'</td>').join('')+'</tr>').join('')+'</table></div>';}
function splitChip(info){if(!info.split)return '<span class="chip split-unknown">not in dataset</span>';const label={train:'training split',val:'validation split',test:'held-out test split',excluded:'excluded buffer frame'}[info.split]||info.split;return '<span class="chip split-'+info.split+'">'+label+'</span>';}
function renderCompare(info,result){
if(!info.gt||!classNames.length){$('compare').innerHTML=result&&info.split===null?'<div class="muted">No ground truth for this image, so only predictions are shown.</div>':'';return;}
const pred=classNames.map(()=>0);if(result)for(const d of result.detections){const i=classNames.indexOf(d.class_name);if(i>=0)pred[i]++;}
const rows=classNames.map((n,i)=>{const miss=result&&pred[i]!==info.gt[i];return [n,info.gt[i],result?(miss?'<span class="low">'+pred[i]+'</span>':pred[i]):'-',result?(pred[i]===info.gt[i]?'<span class="good">match</span>':(pred[i]<info.gt[i]?'<span class="low">'+(info.gt[i]-pred[i])+' missed</span>':'<span class="low">'+(pred[i]-info.gt[i])+' extra</span>')):'-'];});
$('compare').innerHTML=table(['Class','Ground truth boxes','Predicted boxes','Count check'],rows)+'<div class="muted">Counts only. A matching count does not guarantee the boxes are on the right objects.</div>';}
function render(){const result=results.get(index);const info=examples[index]||{};$('frames').value=String(index);$('overlay').disabled=!result;if(!result)$('overlay').checked=false;
$('image').src=result&&$('overlay').checked?result.image:'/original/'+index;$('frameSplit').innerHTML=examples.length?splitChip(info):'';
$('status').className='';$('status').textContent=result?'Frame '+info.name+': '+result.elapsed_ms.toFixed(1)+' ms for the warmed inference call on '+result.device+'.':'Frame '+info.name+': original image.';
$('detections').replaceChildren();renderCompare(info,result);if(!result){$('detections').textContent='Run inference to detect objects in this image.';return;}
if(!result.detections.length)$('detections').textContent='No detections above confidence 0.25.';
for(const d of result.detections){const tag=document.createElement('span');tag.className='tag';tag.textContent=d.class_name+' '+d.confidence.toFixed(2);$('detections').append(tag);}}
function move(delta){index=(index+delta+examples.length)%examples.length;render();}
function setBusy(state){busy=state;$('predict').disabled=state;$('upload').disabled=state;}
function fillOptions(){$('frames').replaceChildren();examples.forEach((e,i)=>{const option=document.createElement('option');option.value=i;option.textContent=(i+1)+' / '+examples.length+' - '+e.name+(e.split?' ('+e.split+')':'');$('frames').append(option);});}
async function predict(){if(busy||!examples.length)return;setBusy(true);const requested=index;$('status').textContent='Running model inference...';
try{const response=await fetch('/predict/'+requested,{method:'POST'});const data=await response.json();if(!response.ok)throw Error(data.error);results.set(requested,data);if(index===requested){$('overlay').checked=true;render();}}
catch(error){$('status').className='error';$('status').textContent=error.message;}finally{setBusy(false);}}
async function upload(){const file=$('file').files[0];$('file').value='';if(!file||busy)return;setBusy(true);$('status').className='';$('status').textContent='Uploading '+file.name+'...';
try{const response=await fetch('/upload',{method:'POST',headers:{'Content-Type':file.type||'application/octet-stream','X-Filename':encodeURIComponent(file.name)},body:file});const data=await response.json();if(!response.ok)throw Error(data.error);
examples.push(data.example);fillOptions();index=data.index;render();}
catch(error){$('status').className='error';$('status').textContent=error.message;}finally{setBusy(false);}}
function renderCurrent(c){
const n=c.names, m=c.metrics, cell=v=>v<0.5?'<span class="low">'+f3(v)+'</span>':f3(v), sp=['train','val','test'], rr=c.rounds;
let h='<h3>Loaded model</h3><p class="muted">Provisional seven-class YOLO26n, input 1280. Retrained on 1 November 2026 on corrected label copies from the 1280 fine-tuned model, for about 2.3 epochs. Scored on Round 2 validation, which also selected the checkpoint. It has no clean test score: both test sets were used by earlier models, so a final figure needs a new CARLA capture.</p>';
h+='<h3>Results on Round 2 validation, corrected labels</h3><div class="muted">Values under 0.5 are highlighted. Precision and recall are at the operating point Ultralytics selects, not at the 0.25 display threshold.</div>'+table(['Class','Boxes','Precision','Recall','mAP50','mAP50-95'],n.map((k,i)=>[k,rr['Round 2'].val.boxes[i],...m.corrected_ultralytics[k].map(cell)]));
h+='<h3>Model comparison on Round 2 validation (AP50)</h3><div class="muted">The annotator B rows are the fairest check, because those labels were never edited. Corrected labels used scaling factors fitted to the previous model, so they are not an independent reference.</div>'+table(['Model and labels','Vehicle','Red','Yellow','Green','Adult','Child','Speed limit','Mean of 7'],m.r2_val.map(r=>[r[0],...r.slice(1).map(cell)]));
h+='<h3>Round 1 validation (AP50, labels unchanged)</h3><div class="muted">The current model is slightly lower here. Round 1 has no pedestrian or speed limit labels.</div>'+table(['Model','Vehicle','Red','Yellow','Green','Mean of 4'],m.r1_val.map(r=>[r[0],...r.slice(1).map(cell)]));
h+='<h3>Lights found with the right colour</h3><div class="muted">Round 2 validation, corrected labels, at confidence 0.25 and IoU 0.5.</div>'+table(['Class','Boxes','Previous fine-tuned model','Current model'],m.found.map(r=>[r[0],r[1],cell(r[2]),cell(r[3])]));
h+='<h3>Dataset</h3><div class="muted">Yellow training images appear three times per epoch. Round 1 frames are 800 x 600 and Round 2 frames 1280 x 720. Round 2 was split by whole episode: '+[70,10,20].join(', ')+' episodes.</div>'+table(['Round','Train images','Validation images','Test images'],Object.keys(rr).map(r=>[r,...sp.map(k=>rr[r][k].images)]));
h+=table(['Boxes by class','Round 1 train','Round 1 val','Round 1 test','Round 2 train','Round 2 val','Round 2 test'],n.map((k,i)=>[k,...Object.keys(rr).flatMap(r=>sp.map(s=>rr[r][s].boxes[i]))]));
h+='<h3>Object size at the model input</h3><div class="muted">Median box width x height, in pixels after scaling each frame to the 1280 input. Training split.</div>'+table(['Class','Round 1','Round 2'],n.map((k,i)=>[k,...Object.keys(rr).map(r=>rr[r].train.median_w[i]===null?'not labelled':rr[r].train.median_w[i]+' x '+rr[r].train.median_h[i])]));
const a=c.annotators;h+='<h3>Round 2 light labels by annotator</h3><div class="muted">Original labels. Found means the previous fine-tuned model found the light with the right colour on validation at confidence 0.25.</div>'+table(['Annotator','Frames','Light boxes','Median width x height','Found'],Object.keys(a).map(k=>[k,a[k].frames,a[k].light_boxes,a[k].median_w+' x '+a[k].median_h,pct(m.annotator_found[k])]));
const f=['Labels were a main limit on Round 2. On the same episodes, the previous model found 73% of lights in annotator B frames and 30% in annotator A frames. Sampled annotator A boxes were padded, and most sampled confident detections without a label in annotator B frames were real unlabelled lights.','Colour was not the main limit: a separate colour classifier was right 96 to 99.9% of the time on correct boxes, yet adding one changed yellow AP50 by only 0.001.','The gains are optimistic: one seed, about 2.3 epochs, 81 yellow validation boxes, and validation also selected the checkpoint. The 95% range of the yellow gain on corrected labels is -0.002 to +0.129.','Costs: speed limit on annotator B frames fell from 0.886 to 0.801, and Round 1 validation mean fell from 0.692 to 0.673.','Corrections were made on copies. Unlabelled lights in annotator B frames remain, and the team has not yet agreed one labelling rule.'];
return h+'<h3>What the numbers explain</h3><ul class="findings">'+f.map(x=>'<li>'+x+'</li>').join('')+'</ul>';}
function renderStats(s){
const n=s.class_names, o=s.overview, cov=s.coverage, sp=['train','val','test'], spName={train:'Train',val:'Validation',test:'Test'};
const excl=n.map((_,i)=>o.total_boxes[i]-sum(sp.map(k=>cov[k].class_boxes[i])));const allBoxes=sum(o.total_boxes);
let h='<p class="muted">'+o.images+' frames at '+o.width+' x '+o.height+' px from a single CARLA capture run. Split into ten temporal blocks (seed '+o.seed+'); '+o.excluded_images+' frames next to block edges were dropped as buffers so near-duplicate neighbouring frames cannot leak across splits (smallest frame-ID gap between splits: '+o.min_cross_split_id_gap+'). Class mapping status: '+esc(o.mapping_status).replace(/_/g,' ')+'.</p>';
h+='<h3>Images per split</h3>'+table(['','Train','Validation','Test','Excluded buffer','Total'],[['Images',cov.train.images,cov.val.images,cov.test.images,o.excluded_images,o.images],['Share',pct(cov.train.images/o.images),pct(cov.val.images/o.images),pct(cov.test.images/o.images),pct(o.excluded_images/o.images),'100%']]);
const boxRows=n.map((c,i)=>[c,...sp.map(k=>cov[k].class_boxes[i]+' <span class="muted">('+cov[k].class_images[i]+' img)</span>'),excl[i],o.total_boxes[i],pct(o.total_boxes[i]/allBoxes)]);
boxRows.push(['All classes',...sp.map(k=>sum(cov[k].class_boxes)),sum(excl),allBoxes,'100%']);
h+='<h3>Labelled boxes per class and split</h3><div class="muted">Boxes, with the number of images containing that class in brackets.</div>'+table(['Class','Train','Validation','Test','Excluded','Total','Share of boxes'],boxRows,true);
const mv=s.metrics.val, mt=s.metrics.test;const cell=v=>v<0.5?'<span class="low">'+f3(v)+'</span>':f3(v);
const mRows=n.map((c,i)=>{const a=mv.per_class[i],b=mt.per_class[i];return [c,b.Instances,cell(a['Box-P']),cell(a['Box-R']),cell(a.mAP50),cell(b['Box-P']),cell(b['Box-R']),cell(b.mAP50),cell(b['mAP50-95'])];});
const va=mv.aggregate, ta=mt.aggregate;mRows.push(['All classes',sum(mt.per_class.map(r=>r.Instances)),f3(va['metrics/precision(B)']),f3(va['metrics/recall(B)']),f3(va['metrics/mAP50(B)']),f3(ta['metrics/precision(B)']),f3(ta['metrics/recall(B)']),f3(ta['metrics/mAP50(B)']),f3(ta['metrics/mAP50-95(B)'])]);
h+='<h3>Per-class results: validation vs held-out test</h3><div class="muted">Values under 0.5 are highlighted. Precision and recall are at the operating point Ultralytics selects, not at the 0.25 display threshold.</div>'+table(['Class','Test boxes','Val P','Val R','Val mAP50','Test P','Test R','Test mAP50','Test mAP50-95'],mRows,true);
const bs=s.box_sizes;const sizeRows=n.map((c,i)=>[c,...sp.map(k=>bs[k][i].median_w+' x '+bs[k][i].median_h),...sp.map(k=>pct(bs[k][i].short_side_under_10px))]);
h+='<h3>Object size in pixels</h3><div class="muted">Median box width x height on the '+o.width+' x '+o.height+' frame, and the share of boxes whose shorter side is under 10 px.</div>'+table(['Class','Train','Validation','Test','Train under 10 px','Val under 10 px','Test under 10 px'],sizeRows);
const yi=n.indexOf('yellow_light');const blockRows=s.blocks.map(b=>{const big=yi>=0&&b.class_boxes[yi]>0.25*o.total_boxes[yi];return ['Block '+b.id,b.first+' to '+b.last,b.images,'<span class="chip split-'+b.split+'">'+spName[b.split]+'</span>',...b.class_boxes.map((v,i)=>i===yi&&big?'<span class="low">'+v+'</span>':v)];});
h+='<h3>Temporal blocks</h3><div class="muted">Each split is made of whole blocks of consecutive frames. Class mix differs sharply between blocks because traffic-light phases last many frames.</div>'+table(['Block','Frame IDs','Images','Split',...n],blockRows);
h+='<h3>Held-out test confusion matrix</h3><div class="muted">Ultralytics validator output. Each column is a true class; each row is what the model predicted. The background row counts missed objects; the background column counts false detections.</div><div class="figure"><img src="/figure/confusion" alt="Confusion matrix on the held-out test split"></div>';
const f=[];
if(yi>=0){const tr=cov.train.class_boxes[yi], peak=s.blocks.reduce((a,b)=>b.class_boxes[yi]>a.class_boxes[yi]?b:a);
f.push('Yellow light is the rarest class: '+o.total_boxes[yi]+' of '+allBoxes+' boxes ('+pct(o.total_boxes[yi]/allBoxes)+'). Block '+peak.id+' alone holds '+peak.class_boxes[yi]+' of them ('+pct(peak.class_boxes[yi]/o.total_boxes[yi])+'), so most training yellows come from one stretch of driving and look alike.');
f.push('The test set has only '+cov.test.class_boxes[yi]+' yellow boxes in '+cov.test.class_images[yi]+' images, so one missed yellow light moves yellow recall by about '+(100/cov.test.class_boxes[yi]).toFixed(1)+' points. Yellow test mAP50 of '+f3(mt.per_class[yi].mAP50)+' is a real weakness, but the exact figure is noisy.');}
const ri=n.indexOf('red_light'), gi=n.indexOf('green_light');
if(ri>=0)f.push('Red lights are tiny: median '+bs.test[ri].median_w+' x '+bs.test[ri].median_h+' px in test, and '+pct(bs.test[ri].short_side_under_10px)+' have a side under 10 px. Training red lights are larger (median '+bs.train[ri].median_w+' px wide), so distant lights sit at the edge of what the model saw during training.');
if(ri>=0&&gi>=0)f.push('Validation and test disagree on green light (mAP50 '+f3(mv.per_class[gi].mAP50)+' vs '+f3(mt.per_class[gi].mAP50)+') and red light ('+f3(mv.per_class[ri].mAP50)+' vs '+f3(mt.per_class[ri].mAP50)+'). With two blocks per split, results depend heavily on which scenes landed in each block.');
f.push('Light state is a colour judgement on a few pixels. The confusion matrix shows true red lights predicted as green, consistent with the conflicting red and green boxes that sometimes appear on the same light.');
f.push('All frames come from one CARLA capture run. Other towns, weather, night scenes and real camera footage are untested. Pedestrians and street signs are not classes. The class mapping was confirmed by visual inspection only.');
h+='<h3>What the numbers explain</h3><ul class="findings">'+f.map(x=>'<li>'+x+'</li>').join('')+'</ul>';
$('stats').innerHTML=(s.current?renderCurrent(s.current):'')+'<details><summary>Earlier Round 1 baseline: four classes, single capture run</summary><div class="muted">These tables describe the 25-epoch baseline that the demo used before, not the loaded model.</div>'+h+'</details>';}
$('previous').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('predict').onclick=predict;$('overlay').onchange=render;$('frames').onchange=()=>{index=Number($('frames').value);render();};
$('upload').onclick=()=>$('file').click();$('file').onchange=upload;
document.onkeydown=event=>{if(event.target.tagName==='SELECT')return;if(event.key==='ArrowLeft')move(-1);if(event.key==='ArrowRight')move(1);if(event.code==='Space'){event.preventDefault();predict();}if(event.key.toLowerCase()==='t'&&results.has(index)){$('overlay').checked=!$('overlay').checked;render();}};
fetch('/stats').then(r=>r.json()).then(s=>{if(s.error)throw Error(s.error);classNames=s.class_names;renderStats(s);render();}).catch(e=>{$('stats').textContent='Statistics unavailable: '+e.message;});
fetch('/examples').then(r=>r.json()).then(data=>{examples=data;fillOptions();render();}).catch(e=>{$('status').textContent=e.message;});
</script></html>'''


def main():
    # Keep the demonstration tied to the selected checkpoint
    parser = argparse.ArgumentParser(description="Local CARLA image inference demo")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--port", type=int, default=8773)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Choose a port between 1024 and 65535")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("Apple GPU inference is unavailable; use --device cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    bundle = ROOT / "carla_yolo26_demo_results"
    metadata = json.loads((bundle / "mac_demo/model_metadata.json").read_text())
    checkpoint = bundle / "mac_demo/best.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metadata["checkpoint_sha256"]:
        raise RuntimeError("The model checkpoint does not match its metadata")
    # Red-to-green queue sequence first, then the other Round 2 frames (ep...)
    examples = sorted((bundle / "mac_demo/examples").glob("*.png"),
                      key=lambda p: (not p.stem.startswith("ep070_0001"), not p.stem.startswith("ep"), p.stem))
    if not examples:
        raise RuntimeError("No bundled example images were found")
    settings.update({"sync": False, "wandb": False, "comet": False, "mlflow": False})
    model = YOLO(str(checkpoint))
    if model.names != {int(k): v for k, v in metadata["class_names"].items()}:
        raise RuntimeError("Model class names do not match the metadata")
    options = {"imgsz": metadata["imgsz"], "conf": metadata["display_confidence"],
               "device": args.device, "verbose": False}
    for _ in range(3):
        model.predict(str(examples[0]), **options)
    # Dataset statistics are optional so the demo still runs without them
    stats_path = ROOT / "dataset_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
    if stats is None:
        print("dataset_stats.json not found; run build_stats.py to show dataset tables", flush=True)
    frames = stats.pop("frames") if stats else {}
    current_stats = bundle / "mac_demo/model_stats.json"
    if stats and current_stats.exists():
        stats["current"] = json.loads(current_stats.read_text())
    # Round 2 example frames are not in the Round 1 statistics
    round2_frames = bundle / "mac_demo/round2_examples.json"
    if round2_frames.exists():
        frames.update(json.loads(round2_frames.read_text()))
    confusion = bundle / "held_out_test/confusion_matrix.png"
    sources = [p.stem for p in examples]

    def describe(i):
        info = frames.get(sources[i])
        return {"name": examples[i].stem, "split": info["split"] if info else None,
                "gt": info["gt"] if info else None}

    output = ROOT / "local_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    uploads = output / "uploads"
    inference_lock = threading.Lock()
    upload_lock = threading.Lock()
    upload_count = 0
    artifacts = {}

    class Handler(BaseHTTPRequestHandler):
        def send(self, content, mime, status=200):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def send_json(self, content, status=200):
            self.send(json.dumps(content).encode(), "application/json", status)

        def image_index(self, route):
            index = int(route.rsplit("/", 1)[-1])
            if index < 0 or index >= len(examples):
                raise ValueError("Unknown example image")
            return index

        def do_GET(self):
            # Serve only known examples and generated predictions
            route = urlsplit(self.path).path
            if route == "/":
                self.send(PAGE.encode(), "text/html; charset=utf-8")
            elif route == "/examples":
                self.send_json([describe(i) for i in range(len(examples))])
            elif route == "/stats":
                self.send_json(stats if stats else {"error": "Run build_stats.py first"}, 200 if stats else 404)
            elif route == "/figure/confusion" and confusion.exists():
                self.send(confusion.read_bytes(), "image/png")
            elif route.startswith("/original/"):
                try:
                    self.send(examples[self.image_index(route)].read_bytes(), "image/png")
                except (ValueError, OSError):
                    self.send_json({"error": "Unknown example image"}, 404)
            elif route in artifacts:
                self.send(artifacts[route].read_bytes(), "image/png")
            else:
                self.send_json({"error": "Not found"}, 404)

        def upload(self):
            nonlocal upload_count
            # Re-encode the upload as PNG and add it to this session's examples
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0:
                self.send_json({"error": "The uploaded file is empty"}, 400)
                return
            if length > MAX_UPLOAD_BYTES:
                self.send_json({"error": "Images must be 25 MB or smaller"}, 413)
                return
            data = self.rfile.read(length)
            try:
                image = Image.open(io.BytesIO(data))
                image.load()
                image = ImageOps.exif_transpose(image)
                if image.mode not in ("RGB", "RGBA"):
                    image = image.convert("RGB")
            except Exception:
                self.send_json({"error": "That file is not a readable image"}, 400)
                return
            stem = Path(unquote(self.headers.get("X-Filename", "image"))).stem
            stem = re.sub(r"[^A-Za-z0-9_-]", "_", stem)[:40] or "image"
            with upload_lock:
                uploads.mkdir(exist_ok=True)
                upload_count += 1
                path = uploads / f"upload{upload_count:02d}_{stem}.png"
                image.save(path)
                examples.append(path)
                sources.append(stem)
                index = len(examples) - 1
                example = describe(index)
            print(f"Uploaded {path.name} ({image.width}x{image.height})", flush=True)
            self.send_json({"index": index, "example": example})

        def do_POST(self):
            route = urlsplit(self.path).path
            if route == "/upload":
                self.upload()
                return
            if not route.startswith("/predict/"):
                self.send_json({"error": "Not found"}, 404)
                return
            try:
                index = self.image_index(route)
            except ValueError:
                self.send_json({"error": "Unknown example image"}, 400)
                return
            try:
                with inference_lock:
                    started = time.perf_counter()
                    result = model.predict(str(examples[index]), **options)[0]
                    merged, _ = merge_duplicates(result.boxes.data.cpu().numpy())
                    result.boxes = Boxes(torch.as_tensor(merged, dtype=torch.float32), result.orig_shape)
                    if args.device == "mps":
                        torch.mps.synchronize()
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    stamp = time.time_ns()
                    name = f"{examples[index].stem}_{stamp}.png"
                    result.save(filename=str(output / name))
                    artifacts["/prediction/" + name] = output / name
                    detections = [{"class_name": model.names[int(c)], "confidence": float(v)}
                                  for c, v in zip(result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist())]
                    payload = {"frame": examples[index].stem, "device": args.device,
                               "elapsed_ms": elapsed_ms, "model_ms": result.speed["inference"],
                               "detections": detections, "image": "/prediction/" + name}
                    (output / f"{examples[index].stem}_{stamp}.json").write_text(json.dumps(payload, indent=2))
                self.send_json(payload)
            except Exception as error:
                print(f"Inference failed: {error}", flush=True)
                self.send_json({"error": "Inference failed. Check the terminal for details."}, 500)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Model ready on {args.device}. Open http://127.0.0.1:{args.port}", flush=True)
    print(f"Predictions are saved to {output}. Press Control C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
