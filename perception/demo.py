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
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root{--ink:#000;--card:#1f1f1f;--card2:#2a2a2a;--line:#2e2e2e;--text:#f4f4f4;--dim:#9a9a9a;--light:#d9d9d9;--light2:#cfcfcf;--lightink:#111;--lightdim:#4a4a4a;--accent:#2f6bff;--warn:#ffb35c;--warnlight:#a14b00}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:#fff;color:var(--text);font:15px/1.5 Inter,system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
.frame{max-width:1440px;margin:0 auto;background:var(--ink);padding:0 20px 20px}
.topbar{display:flex;justify-content:space-between;gap:16px;padding:22px 36px 18px;font-size:12.5px;color:#d0d0d0;border-bottom:1px solid var(--line);margin:0 0 0}
.hero{position:relative;padding:34px 36px 56px;min-height:560px;display:grid;grid-template-columns:1fr auto;overflow:hidden}
.hero canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.hero>*{position:relative}
.brand{display:flex;align-items:center;gap:10px;font-weight:600;letter-spacing:.06em;font-size:15px;text-transform:uppercase}
.brand svg{width:26px;height:26px}
nav{display:flex;flex-direction:column;align-items:flex-end;gap:6px;font-size:14px}nav a{color:var(--text);text-decoration:none}nav a:hover{color:var(--dim)}
h1{font-weight:400;font-size:clamp(44px,6.2vw,84px);line-height:1.02;letter-spacing:-.045em;margin:56px 0 0;max-width:820px}
.lede{grid-column:1;align-self:end;max-width:400px;font-size:16px;line-height:1.6;color:#e6e6e6;margin-top:150px}
.cards{display:grid;grid-template-columns:repeat(3,1fr);background:var(--card);border-radius:14px;overflow:hidden}
.card{padding:36px 36px 34px;border-left:1px solid #0d0d0d;display:flex;flex-direction:column;min-height:520px}.card:first-child{border-left:0}
.card h2{font-weight:400;font-size:26px;line-height:1.25;letter-spacing:-.025em;margin:0;max-width:330px}
.card .icon{flex:1;display:flex;align-items:center;padding:30px 0}.card .icon svg{width:190px;height:190px}
.card p{color:var(--dim);font-size:14px;margin:0 0 16px;max-width:340px}
.pill{display:inline-block;font:inherit;font-size:14px;font-weight:500;padding:8px 14px;border-radius:999px;background:var(--card2);color:var(--text);border:0;text-decoration:none;cursor:pointer;margin:0 6px 6px 0}
.pill:hover{background:#363636}.pill.accent{background:var(--accent)}.pill.accent:hover{background:#1f58e6}.pill.dark{background:#111;color:#fff}
.pill:disabled{opacity:.5;cursor:wait}
.viewer{background:var(--card);border-radius:14px;padding:36px;margin:0 0 8px}
.viewer h2{font-weight:400;font-size:clamp(30px,3.4vw,46px);letter-spacing:-.04em;margin:0 0 6px;line-height:1.05}
.viewer .sub{color:var(--dim);margin:0 0 22px}
.toolbar{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:0 0 14px}.toolbar .pill{margin:0}
select{font:inherit;font-size:14px;padding:8px 14px;border-radius:999px;background:var(--card2);color:var(--text);border:0;max-width:100%;cursor:pointer}
label.toggle{display:inline-flex;gap:8px;align-items:center;font-size:14px;color:#ddd;padding:0 8px}
#status{min-height:22px;margin:4px 0 14px;color:#cfcfcf;font-size:14px}.error{color:#ff9d9d!important}
.scene{background:#0b0b0b;border-radius:10px;text-align:center;min-height:240px;overflow:hidden}.scene img{width:100%;display:block}
.det{margin-top:18px;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.3fr);gap:28px}
.det h3{font-weight:500;font-size:15px;margin:0 0 8px}
.tag{display:inline-block;margin:0 6px 6px 0;padding:5px 11px;background:var(--card2);border-radius:999px;font-size:13px}
.chip{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;background:#333;color:#eee;margin-left:6px;font-weight:500;vertical-align:1px}
.split-train{background:#24432f}.split-val{background:#1f3566}.split-test{background:#5a3a1e}.split-excluded,.split-unknown{background:#3a3a3a}
.light{background:var(--light);color:var(--lightink);border-radius:14px;padding:56px 36px 44px;margin-top:8px}
.light h2{font-weight:400;font-size:clamp(34px,4.4vw,60px);letter-spacing:-.045em;line-height:1.02;margin:0 0 18px}
.light .intro{max-width:560px;font-size:16px;line-height:1.65;color:#222;margin:0 0 26px}
.light h3{font-weight:500;font-size:17px;letter-spacing:-.01em;margin:34px 0 4px}
.muted{color:var(--dim)}.light .muted{color:var(--lightdim)}
.bignums{display:flex;gap:56px;flex-wrap:wrap;margin:6px 0 10px}.bignums div{font-size:13px;color:var(--lightdim)}.bignums b{display:block;font-weight:400;font-size:54px;letter-spacing:-.04em;color:#000;line-height:1.05}
.wrap{overflow-x:auto}table{border-collapse:collapse;margin:10px 0;font-variant-numeric:tabular-nums;min-width:100%;font-size:14px}
th,td{padding:8px 12px;border-bottom:1px solid #444;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left;padding-left:0}
th{font-weight:500;color:var(--dim)}.light th{color:var(--lightdim)}.light th,.light td{border-bottom-color:#b5b5b5}
tr.total td{font-weight:600;border-top:1.5px solid currentColor}
.low{color:var(--warn);font-weight:600}.light .low{color:var(--warnlight)}.good{color:#7fdc9c}
ul.findings{margin:10px 0;padding-left:20px;line-height:1.6;max-width:860px}ul.findings li{margin-bottom:8px}
details>summary{cursor:pointer;font-size:17px;font-weight:500;margin-top:30px}
.figure{background:white;border-radius:10px;padding:8px;max-width:760px}.figure img{width:100%}
footer{color:var(--dim);font-size:13px;line-height:1.7;padding:30px 36px 16px}kbd{background:#2a2a2a;padding:1px 6px;border-radius:4px;font:12px ui-monospace,monospace;color:#eee}
@media (max-width:900px){.cards{grid-template-columns:1fr}.card{border-left:0;border-top:1px solid #0d0d0d;min-height:0}.card:first-child{border-top:0}.det{grid-template-columns:1fr}}
@media (max-width:640px){.frame{padding:0 8px 8px}.topbar{padding:16px;flex-direction:column;gap:4px}.hero{padding:24px 16px 36px;min-height:460px;grid-template-columns:1fr}nav{display:none}.lede{margin-top:90px}.viewer,.card,.light{padding:28px 18px}.bignums{gap:28px}.bignums b{font-size:42px}footer{padding:24px 16px}}
</style>
<div class="frame">
<div class="topbar"><span>AAI3001 Computer Vision and Deep Learning, team P1-21</span><span>CARLA 0.9.16 &middot; YOLO26n &middot; seven classes</span></div>
<header class="hero"><canvas id="field" aria-hidden="true"></canvas>
<div><div class="brand"><svg viewBox="0 0 26 26" fill="none" stroke="#fff" stroke-width="2"><rect x="3" y="6" width="20" height="14" rx="2"/><circle cx="13" cy="13" r="3.2" fill="#fff" stroke="none"/></svg>CARLA perception</div>
<h1>Seeing the road<br>from one camera</h1></div>
<nav aria-label="Sections"><a href="#demo">Demo</a><a href="#test">Held-out test</a><a href="#results">Validation</a><a href="#dataset">Dataset</a><a href="#notes">Notes</a></nav>
<p class="lede">A seven-class detector trained on our own CARLA captures. It finds vehicles, traffic lights by colour, adult and child pedestrians, and speed-limit signs, and feeds them to the driving agent.</p>
</header>

<section class="viewer" id="demo"><h2>Run the detector</h2><p class="sub">All 50 frames come from the 20 held-out test episodes the model never trained on: day, night, rain and fog, city, suburb and highway. They were picked by a fixed seeded rule (every episode, with minimums for yellow lights, child pedestrians and speed-limit signs), not by how well the model does. Upload your own image to try anything else. Boxes below confidence 0.25 are hidden.</p>
<div class="toolbar"><button class="pill" id="previous">Previous</button><select id="frames" aria-label="Choose image"></select><button class="pill" id="next">Next</button><button class="pill accent" id="predict">Run inference</button><button class="pill" id="upload">Upload image</button><input id="file" type="file" accept="image/png,image/jpeg,image/webp,image/bmp" hidden><label class="toggle"><input id="overlay" type="checkbox">Show prediction</label></div>
<div id="status" role="status">Loading examples...</div><div class="scene"><img id="image" alt="CARLA driving scene"></div>
<div class="det"><div><h3>Detections <span id="frameSplit"></span></h3><div id="detections">Run inference to detect objects in this image.</div></div><div id="compare" class="wrap"></div></div>
</section>

<section class="cards" aria-label="Summary">
<div class="card"><h2>Seven classes from a single front camera</h2>
<div class="icon"><svg viewBox="0 0 190 190" fill="none" stroke="#fff" stroke-width="1.6"><path d="M20 150 L95 70 L170 150" stroke="#777"/><path d="M58 150 L95 70 L132 150" stroke="#555"/><rect x="62" y="44" width="66" height="54" rx="3"/><path d="M40 160 L95 104 L150 160 Z" fill="#fff" fill-opacity=".09" stroke="none"/><circle cx="95" cy="44" r="8" fill="#fff" stroke="none"/></svg></div>
<p>Vehicles, red, yellow and green lights, adult and child pedestrians, and speed-limit signs. YOLO26n at 1280 px, retrained on 1 November 2026.</p>
<div><a class="pill" href="#demo">Try the demo</a></div></div>
<div class="card"><h2>Scored once on 20 unseen episodes</h2>
<div class="icon"><svg viewBox="0 0 190 190" fill="none" stroke="#fff" stroke-width="1.6"><path d="M95 18 V172" stroke="#666"/><ellipse cx="95" cy="62" rx="62" ry="22"/><ellipse cx="95" cy="98" rx="52" ry="18" fill="#fff" fill-opacity=".08" stroke="#888"/><ellipse cx="95" cy="128" rx="34" ry="12" fill="#fff" fill-opacity=".08" stroke="#666"/><circle cx="95" cy="150" r="8" fill="#fff" stroke="none"/></svg></div>
<p>Held-out test mAP50 0.562 on Round 2 (seven classes) and 0.703 on Round 1 (four classes), scored on 22 November 2026.</p>
<div><a class="pill" href="#test">See the test score</a><a class="pill" href="#results">Validation</a></div></div>
<div class="card"><h2>Labelled for the hard cases</h2>
<div class="icon"><svg viewBox="0 0 190 190" fill="none" stroke="#fff" stroke-width="1.6"><ellipse cx="95" cy="95" rx="34" ry="78" transform="rotate(-40 95 95)" fill="#fff" fill-opacity=".1" stroke="none"/><ellipse cx="95" cy="95" rx="30" ry="74" transform="rotate(40 95 95)"/><path d="M18 95 H172" stroke="#777"/><circle cx="95" cy="95" r="8" fill="#fff" stroke="none"/></svg></div>
<p>9,060 of 20,000 captured frames hand-labelled, prioritising yellow lights and telling child from adult pedestrians.</p>
<div><a class="pill" href="#dataset">See the dataset</a></div></div>
</section>

<section class="light" id="test"><h2>Held-out test</h2>
<p class="intro">Scored once on 22 November 2026 with the loaded model. Test frames were never used for training or for choosing the checkpoint. Earlier models were also scored on these test sets in October.</p>
<div class="bignums"><div><b>0.562</b>mAP50, Round 2 test, 7 classes</div><div><b>0.607</b>with corrected light labels</div><div><b>0.703</b>mAP50, Round 1 test, 4 classes</div></div>
<h3>AP50 by class</h3><div class="muted">Round 2 test: 1,884 frames from 20 held-out episodes. Round 1 test: 661 frames. Corrected applies the training label fix (annotator A light boxes of 15 px or more scaled to 0.78 width, 0.85 height) to copies. Values under 0.5 are highlighted.</div>
<div id="testTable"></div></section>

<section class="light" id="results"><h2>Validation results</h2>
<p class="intro">Round 2 validation also selected the checkpoint, so these figures are optimistic. They explain what the label fix changed.</p>
<div id="stats">Loading statistics...</div></section>

<section class="light" id="dataset"><h2>Dataset</h2>
<p class="intro">Our own CARLA captures, labelled by the team. Round 2: 100 episodes captured at 200 frames each (20,000 frames). 9,060 were hand-labelled by two annotators, 25 to 119 per episode. Frames with no objects, or with only classes we already had plenty of, were skipped; labelling went to yellow lights and to adult and child pedestrians. Split by whole episode: 70 train, 10 validation, 20 test.</p>
<div id="datasetTables"></div></section>

<section class="light" id="notes"><h2>What the numbers explain</h2><div id="notesBody"></div></section>

<footer>Display confidence 0.25. Yellow lights are still often missed, and conflicting light states can occur. Count checks compare box counts with the hand labels for all seven classes; unlabelled lights in some frames show up as extra. Duplicate boxes are merged for display; the statistics are scored before merging.<br>Uploaded images are kept for this session only and are not part of any evaluation split.<br>Keyboard: <kbd>Left</kbd> / <kbd>Right</kbd> change image, <kbd>Space</kbd> runs inference, <kbd>T</kbd> toggles prediction.</footer>
</div>
<script>
let examples=[], index=0, busy=false, classNames=[];const results=new Map();const $=id=>document.getElementById(id);
const pct=v=>(100*v).toFixed(1)+'%';const f3=v=>Number(v).toFixed(3);const sum=a=>a.reduce((x,y)=>x+y,0);
const esc=v=>String(v).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const cell=v=>v===null?'-':(v<0.5?'<span class="low">'+f3(v)+'</span>':f3(v));
function table(head,rows,totalLast){return '<div class="wrap"><table><tr>'+head.map(h=>'<th>'+h+'</th>').join('')+'</tr>'+rows.map((r,i)=>'<tr'+(totalLast&&i===rows.length-1?' class="total"':'')+'>'+r.map(c=>'<td>'+c+'</td>').join('')+'</tr>').join('')+'</table></div>';}
// Held-out test, scored once on 22 November 2026 (team evaluation report). Columns: Round 1, Round 2, Round 2 corrected.
const TEST=[['Vehicle',0.841,0.832,0.832],['Red light',0.854,0.422,0.534],['Yellow light',0.202,0.402,0.497],['Green light',0.915,0.448,0.552],['Adult pedestrian',null,0.762,0.762],['Child pedestrian',null,0.556,0.556],['Speed limit',null,0.515,0.515]];
$('testTable').innerHTML=table(['Class','Round 1 test','Round 2 test','Round 2 test, corrected'],TEST.map(r=>[r[0],...r.slice(1).map(cell)]).concat([['mAP50','0.703 (4 classes)','0.562','0.607']]),true);
function drawField(){const c=$('field'),r=c.getBoundingClientRect(),d=window.devicePixelRatio||1;c.width=r.width*d;c.height=r.height*d;const g=c.getContext('2d');g.scale(d,d);
const vx=r.width*.62,vy=r.height*.30;let s=7;const rnd=()=>(s=(s*16807)%2147483647)/2147483647;
for(let k=-14;k<=14;k++){const ex=vx+k*r.width*.11;for(let i=0;i<70;i++){const t=Math.pow(rnd(),1.6);const x=vx+(ex-vx)*t+(rnd()-.5)*10*t,y=vy+(r.height*1.15-vy)*t+(rnd()-.5)*8*t;
const fade=Math.min(1,Math.max(.08,(x-r.width*.12)/(r.width*.38)));const a=Math.min(.85,.12+t*.9)*(Math.abs(k)<2?1:.75)*fade;g.fillStyle='rgba(255,255,255,'+a+')';g.beginPath();g.arc(x,y,.5+t*1.7*rnd()+.3,0,6.283);g.fill();}}}
drawField();window.addEventListener('resize',drawField);
function splitChip(info){if(!info.split)return '<span class="chip split-unknown">not in dataset</span>';const label={train:'training split',val:'validation split',test:'held-out test split',excluded:'excluded buffer frame'}[info.split]||info.split;return '<span class="chip split-'+info.split+'">'+label+'</span>';}
function renderCompare(info,result){
if(!info.gt||!classNames.length){$('compare').innerHTML=result&&info.split===null?'<div class="muted">No ground truth for this image, so only predictions are shown.</div>':'';return;}
const pred=classNames.map(()=>0);if(result)for(const d of result.detections){const i=classNames.indexOf(d.class_name);if(i>=0)pred[i]++;}
const rows=classNames.map((n,i)=>{const miss=result&&pred[i]!==info.gt[i];return [n,info.gt[i],result?(miss?'<span class="low">'+pred[i]+'</span>':pred[i]):'-',result?(pred[i]===info.gt[i]?'<span class="good">match</span>':(pred[i]<info.gt[i]?'<span class="low">'+(info.gt[i]-pred[i])+' missed</span>':'<span class="low">'+(pred[i]-info.gt[i])+' extra</span>')):'-'];});
$('compare').innerHTML=table(['Class','Ground truth','Predicted','Count check'],rows)+'<div class="muted">Counts only. A matching count does not guarantee the boxes are on the right objects.</div>';}
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
const n=c.names, m=c.metrics, sp=['train','val','test'], rr=c.rounds;
let v='<h3>Loaded model</h3><p class="muted">Seven-class YOLO26n, input 1280. Retrained on 1 November 2026 on corrected label copies, starting from the 1280 fine-tuned model, for about 2.3 epochs (100-minute limit). Round 2 validation selected the checkpoint. Held-out test, scored once on 22 November 2026: mAP50 0.562 on Round 2 and 0.703 on Round 1 (see above).</p>';
v+='<h3>Round 2 validation, corrected labels</h3><div class="muted">Values under 0.5 are highlighted. Precision and recall are at the operating point Ultralytics selects, not at the 0.25 display threshold.</div>'+table(['Class','Boxes','Precision','Recall','mAP50','mAP50-95'],n.map((k,i)=>[k,rr['Round 2'].val.boxes[i],...m.corrected_ultralytics[k].map(cell)]));
v+='<h3>Model comparison on Round 2 validation (AP50)</h3><div class="muted">The annotator B rows are the fairest check, because those labels were never edited. Corrected labels used scaling factors fitted to the previous model, so they are not an independent reference.</div>'+table(['Model and labels','Vehicle','Red','Yellow','Green','Adult','Child','Speed limit','Mean of 7'],m.r2_val.map(r=>[r[0],...r.slice(1).map(cell)]));
v+='<h3>Round 1 validation (AP50, labels unchanged)</h3><div class="muted">The current model is slightly lower here. Round 1 has no pedestrian or speed limit labels.</div>'+table(['Model','Vehicle','Red','Yellow','Green','Mean of 4'],m.r1_val.map(r=>[r[0],...r.slice(1).map(cell)]));
v+='<h3>Lights found with the right colour</h3><div class="muted">Round 2 validation, corrected labels, at confidence 0.25 and IoU 0.5.</div>'+table(['Class','Boxes','Previous fine-tuned model','Current model'],m.found.map(r=>[r[0],r[1],cell(r[2]),cell(r[3])]));
let d='<h3>Labelled images per split</h3><div class="muted">Yellow training images appear three times per epoch. Round 1 frames are 800 x 600 and Round 2 frames 1280 x 720. Round 2 split by whole episode: 70, 10 and 20 episodes.</div>'+table(['Round','Train images','Validation images','Test images'],Object.keys(rr).map(r=>[r,...sp.map(k=>rr[r][k].images)]));
d+='<h3>Boxes by class</h3>'+table(['Class','Round 1 train','Round 1 val','Round 1 test','Round 2 train','Round 2 val','Round 2 test'],n.map((k,i)=>[k,...Object.keys(rr).flatMap(r=>sp.map(s=>rr[r][s].boxes[i]))]));
d+='<h3>Object size at the model input</h3><div class="muted">Median box width x height, in pixels after scaling each frame to the 1280 input. Training split.</div>'+table(['Class','Round 1','Round 2'],n.map((k,i)=>[k,...Object.keys(rr).map(r=>rr[r].train.median_w[i]===null?'not labelled':rr[r].train.median_w[i]+' x '+rr[r].train.median_h[i])]));
const a=c.annotators;d+='<h3>Round 2 light labels by annotator</h3><div class="muted">Original labels. Found means the previous fine-tuned model found the light with the right colour on validation at confidence 0.25.</div>'+table(['Annotator','Frames','Light boxes','Median width x height','Found'],Object.keys(a).map(k=>[k,a[k].frames,a[k].light_boxes,a[k].median_w+' x '+a[k].median_h,pct(m.annotator_found[k])]));
const f=['Labels were a main limit on Round 2. On the same episodes, the previous model found 73% of lights in annotator B frames and 30% in annotator A frames. Sampled annotator A boxes were padded, and most sampled confident detections without a label in annotator B frames were real unlabelled lights.','Colour was not the main limit: a separate colour classifier was right 96 to 99.9% of the time on correct boxes, yet adding one changed yellow AP50 by only 0.001.','Yellow stays the weakest light. Selective labelling nearly doubled yellow boxes (502 in Round 1, 965 in Round 2), but a yellow phase lasts only a few seconds, so yellow is still 6.2% of Round 2 light boxes.','The validation gains are optimistic: one seed, about 2.3 epochs, 81 yellow validation boxes, and validation also selected the checkpoint. The 95% range of the yellow gain on corrected labels is -0.002 to +0.129.','Costs: speed limit on annotator B frames fell from 0.886 to 0.801, and Round 1 validation mean fell from 0.692 to 0.673.','Corrections were made on copies. Unlabelled lights in annotator B frames remain, and the team has not yet agreed one labelling rule.'];
return {v,d,notes:'<ul class="findings">'+f.map(x=>'<li>'+x+'</li>').join('')+'</ul>'};}
function renderStats(s){
const n=s.class_names, o=s.overview, cov=s.coverage, sp=['train','val','test'], spName={train:'Train',val:'Validation',test:'Test'};
const excl=n.map((_,i)=>o.total_boxes[i]-sum(sp.map(k=>cov[k].class_boxes[i])));const allBoxes=sum(o.total_boxes);
let h='<p class="muted">'+o.images+' frames at '+o.width+' x '+o.height+' px from a single CARLA capture run. Split into ten temporal blocks (seed '+o.seed+'); '+o.excluded_images+' frames next to block edges were dropped as buffers so near-duplicate neighbouring frames cannot leak across splits (smallest frame-ID gap between splits: '+o.min_cross_split_id_gap+'). Class mapping status: '+esc(o.mapping_status).replace(/_/g,' ')+'.</p>';
h+='<h3>Images per split</h3>'+table(['','Train','Validation','Test','Excluded buffer','Total'],[['Images',cov.train.images,cov.val.images,cov.test.images,o.excluded_images,o.images],['Share',pct(cov.train.images/o.images),pct(cov.val.images/o.images),pct(cov.test.images/o.images),pct(o.excluded_images/o.images),'100%']]);
const boxRows=n.map((c,i)=>[c,...sp.map(k=>cov[k].class_boxes[i]+' <span class="muted">('+cov[k].class_images[i]+' img)</span>'),excl[i],o.total_boxes[i],pct(o.total_boxes[i]/allBoxes)]);
boxRows.push(['All classes',...sp.map(k=>sum(cov[k].class_boxes)),sum(excl),allBoxes,'100%']);
h+='<h3>Labelled boxes per class and split</h3><div class="muted">Boxes, with the number of images containing that class in brackets.</div>'+table(['Class','Train','Validation','Test','Excluded','Total','Share of boxes'],boxRows,true);
const mv=s.metrics.val, mt=s.metrics.test;
const mRows=n.map((c,i)=>{const a=mv.per_class[i],b=mt.per_class[i];return [c,b.Instances,cell(a['Box-P']),cell(a['Box-R']),cell(a.mAP50),cell(b['Box-P']),cell(b['Box-R']),cell(b.mAP50),cell(b['mAP50-95'])];});
const va=mv.aggregate, ta=mt.aggregate;mRows.push(['All classes',sum(mt.per_class.map(r=>r.Instances)),f3(va['metrics/precision(B)']),f3(va['metrics/recall(B)']),f3(va['metrics/mAP50(B)']),f3(ta['metrics/precision(B)']),f3(ta['metrics/recall(B)']),f3(ta['metrics/mAP50(B)']),f3(ta['metrics/mAP50-95(B)'])]);
h+='<h3>Per-class results: validation vs held-out test</h3><div class="muted">Values under 0.5 are highlighted. Precision and recall are at the operating point Ultralytics selects, not at the 0.25 display threshold.</div>'+table(['Class','Test boxes','Val P','Val R','Val mAP50','Test P','Test R','Test mAP50','Test mAP50-95'],mRows,true);
const bs=s.box_sizes;const sizeRows=n.map((c,i)=>[c,...sp.map(k=>bs[k][i].median_w+' x '+bs[k][i].median_h),...sp.map(k=>pct(bs[k][i].short_side_under_10px))]);
h+='<h3>Object size in pixels</h3><div class="muted">Median box width x height on the '+o.width+' x '+o.height+' frame, and the share of boxes whose shorter side is under 10 px.</div>'+table(['Class','Train','Validation','Test','Train under 10 px','Val under 10 px','Test under 10 px'],sizeRows);
const yi=n.indexOf('yellow_light');const blockRows=s.blocks.map(b=>{const big=yi>=0&&b.class_boxes[yi]>0.25*o.total_boxes[yi];return ['Block '+b.id,b.first+' to '+b.last,b.images,'<span class="chip split-'+b.split+'">'+spName[b.split]+'</span>',...b.class_boxes.map((v,i)=>i===yi&&big?'<span class="low">'+v+'</span>':v)];});
h+='<h3>Temporal blocks</h3><div class="muted">Each split is made of whole blocks of consecutive frames. Class mix differs sharply between blocks because traffic-light phases last many frames.</div>'+table(['Block','Frame IDs','Images','Split',...n],blockRows);
h+='<h3>Held-out test confusion matrix</h3><div class="muted">Ultralytics validator output. Each column is a true class; each row is what the model predicted. The background row counts missed objects; the background column counts false detections.</div><div class="figure"><img src="/figure/confusion" alt="Confusion matrix on the held-out test split"></div>';
const f=[];
if(yi>=0){const peak=s.blocks.reduce((a,b)=>b.class_boxes[yi]>a.class_boxes[yi]?b:a);
f.push('Yellow light is the rarest class: '+o.total_boxes[yi]+' of '+allBoxes+' boxes ('+pct(o.total_boxes[yi]/allBoxes)+'). Block '+peak.id+' alone holds '+peak.class_boxes[yi]+' of them ('+pct(peak.class_boxes[yi]/o.total_boxes[yi])+'), so most training yellows come from one stretch of driving and look alike.');
f.push('The test set has only '+cov.test.class_boxes[yi]+' yellow boxes in '+cov.test.class_images[yi]+' images, so one missed yellow light moves yellow recall by about '+(100/cov.test.class_boxes[yi]).toFixed(1)+' points. Yellow test mAP50 of '+f3(mt.per_class[yi].mAP50)+' is a real weakness, but the exact figure is noisy.');}
const ri=n.indexOf('red_light'), gi=n.indexOf('green_light');
if(ri>=0)f.push('Red lights are tiny: median '+bs.test[ri].median_w+' x '+bs.test[ri].median_h+' px in test, and '+pct(bs.test[ri].short_side_under_10px)+' have a side under 10 px. Training red lights are larger (median '+bs.train[ri].median_w+' px wide), so distant lights sit at the edge of what the model saw during training.');
if(ri>=0&&gi>=0)f.push('Validation and test disagree on green light (mAP50 '+f3(mv.per_class[gi].mAP50)+' vs '+f3(mt.per_class[gi].mAP50)+') and red light ('+f3(mv.per_class[ri].mAP50)+' vs '+f3(mt.per_class[ri].mAP50)+'). With two blocks per split, results depend heavily on which scenes landed in each block.');
f.push('Light state is a colour judgement on a few pixels. The confusion matrix shows true red lights predicted as green, consistent with the conflicting red and green boxes that sometimes appear on the same light.');
f.push('All frames come from one CARLA capture run. Other towns, weather, night scenes and real camera footage are untested. Pedestrians and street signs are not classes. The class mapping was confirmed by visual inspection only.');
h+='<h3>What the numbers explain</h3><ul class="findings">'+f.map(x=>'<li>'+x+'</li>').join('')+'</ul>';
const cur=s.current?renderCurrent(s.current):{v:'',d:'',notes:''};
$('stats').innerHTML=cur.v;$('datasetTables').innerHTML=cur.d;
$('notesBody').innerHTML=cur.notes+'<details><summary>Earlier Round 1 baseline: four classes, single capture run</summary><div class="muted">These tables describe the 25-epoch baseline that the demo used before, not the loaded model.</div>'+h+'</details>';}
$('previous').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('predict').onclick=predict;$('overlay').onchange=render;$('frames').onchange=()=>{index=Number($('frames').value);render();};
$('upload').onclick=()=>$('file').click();$('file').onchange=upload;
document.onkeydown=event=>{if(event.target.tagName==='SELECT')return;if(event.key==='ArrowLeft')move(-1);if(event.key==='ArrowRight')move(1);if(event.code==='Space'){event.preventDefault();predict();}if(event.key.toLowerCase()==='t'&&results.has(index)){$('overlay').checked=!$('overlay').checked;render();}};
fetch('/stats').then(r=>r.json()).then(s=>{if(s.error)throw Error(s.error);classNames=(s.current&&s.current.names)||s.class_names;renderStats(s);render();}).catch(e=>{$('stats').textContent='Statistics unavailable: '+e.message;});
fetch('/examples').then(r=>r.json()).then(data=>{examples=data;fillOptions();render();}).catch(e=>{$('status').textContent=e.message;});
</script></html>
'''


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
