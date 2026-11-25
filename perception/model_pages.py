"""Road model and speed-sign pages for the perception demo (served by demo.py at /road and /signs).

Optional by design: load() returns (None, reason) when a model or package is missing, and demo.py
keeps serving the detector page unchanged. Modules are imported under their package names
(joint_road_model.road_perception, dreamerv3_attention.speed_sign) so no file name can clash.
Nothing here changes any model; the road model and sign reader are imported read-only.
"""
import base64
import importlib
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EXAMPLES = HERE / "model_pages_examples"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class Extras:
    def __init__(self, device, detector, options, lock):
        import cv2
        import numpy as np
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        road_perception = importlib.import_module("joint_road_model.road_perception")
        self.speed_sign = importlib.import_module("dreamerv3_attention.speed_sign")
        from dedupe import merge_duplicates
        self.cv2, self.np, self.merge = cv2, np, merge_duplicates
        self.device, self.detector, self.options, self.lock = device, detector, options, lock
        package = REPO / "joint_road_model"
        self.road = road_perception.RoadPerception(package / "joint_road_model.pt",
                                                   package / "lane_center_calibration.json", device)
        self.ocr = self.speed_sign.make_sign_reader()
        if self.ocr is None:
            raise RuntimeError("rapidocr_onnxruntime==1.2.3 is not installed")
        self.examples = {k: sorted(p for p in (EXAMPLES / k).glob("*") if p.suffix in (".png", ".jpg"))
                         for k in ("road", "signs")}
        if not all(self.examples.values()):
            raise RuntimeError("model_pages_examples/road or /signs is empty")
        self.uploads = {"road": 0, "signs": 0}
        frame = cv2.imread(str(self.examples["road"][0]))
        self.road(frame)  # warm up
        cv2.imread(str(self.examples["signs"][0]))

    def jpg(self, img, quality=88):
        return base64.b64encode(self.cv2.imencode(".jpg", img, [self.cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()).decode()

    def run_road(self, frame):
        started = time.perf_counter()
        result = self.road(frame)
        elapsed = 1000 * (time.perf_counter() - started)
        lane = result["lane_center"] or {}
        return {"overlay": self.jpg(self.road.draw(frame, result)), "ms": round(elapsed), "device": self.device,
                "crosswalks": result["crosswalks"], "lane_pixels": int(result["lane_mask"].sum()),
                "lane_center": {k: lane.get(k) for k in ("mode", "offset_norm", "heading_deg", "low_confidence")}}

    def run_signs(self, frame):
        cv2, sign = self.cv2, self.speed_sign
        colours = {0: (255, 140, 40), 1: (60, 60, 255), 2: (40, 220, 255), 3: (80, 220, 80),
                   4: (255, 255, 255), 5: (0, 165, 255), 6: (255, 80, 200)}
        started = time.perf_counter()
        boxes = self.detector.predict(frame, **self.options)[0].boxes.data.cpu().numpy()
        rows, _ = self.merge(boxes)
        vis, signs, others = frame.copy(), [], []
        for x1, y1, x2, y2, conf, cls in rows:
            cls = int(cls)
            x1, y1, x2, y2 = (int(v) for v in (x1, y1, x2, y2))
            cv2.rectangle(vis, (x1, y1), (x2, y2), colours.get(cls, (200, 200, 200)), 3 if cls == 6 else 1)
            if cls != 6:
                others.append(f"{self.detector.names[cls]} {conf:.2f}")
                continue
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            item = {"confidence": float(conf), "height": y2 - y1, "read": False, "value": None, "score": 0.0,
                    "cores": [], "crop": None}
            if crop.size:
                width = max(1, int(crop.shape[1] * 72 / crop.shape[0]))
                item["crop"] = self.jpg(cv2.resize(crop, (width, 72), interpolation=cv2.INTER_CUBIC))
            if crop.size and y2 - y1 >= sign.SIGN_MIN_HEIGHT_PX:
                item["read"] = True
                item["cores"] = [self.jpg(c) for c in sign.sign_cores(crop)]
                item["value"], item["score"] = sign.read_speed_limit(self.ocr, crop)
            label = f"{item['value']} km/h" if item["value"] else ("no value" if item["read"] else "too small")
            cv2.putText(vis, label, (x1, max(22, y1 - 8)), 0, .8, (255, 80, 200), 2)
            signs.append(item)
        return {"overlay": self.jpg(vis), "ms": round(1000 * (time.perf_counter() - started)), "device": self.device,
                "signs": signs, "others": others}

    def get(self, kind, rest):
        if rest == "examples":
            return 200, "application/json", json.dumps(
                [{"name": p.stem, "split": None if p.parent.name == "uploads" else "test"} for p in self.examples[kind]]).encode()
        index = int(rest.split("/")[1])
        if not 0 <= index < len(self.examples[kind]):
            raise ValueError("Unknown example image")
        path = self.examples[kind][index]
        return 200, "image/png" if path.suffix == ".png" else "image/jpeg", path.read_bytes()

    def post(self, kind, rest, body):
        cv2, np = self.cv2, self.np
        if rest == "upload":
            image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR) if body else None
            if image is None:
                raise ValueError("That file is not a readable image")
            folder = HERE / "local_runs" / "model_page_uploads" / kind
            folder.mkdir(parents=True, exist_ok=True)
            with self.lock:
                self.uploads[kind] += 1
                path = folder / f"upload{self.uploads[kind]:02d}.png"
                cv2.imwrite(str(path), image)
                self.examples[kind].append(path)
                index = len(self.examples[kind]) - 1
            return {"index": index, "example": {"name": path.stem, "split": None}}
        index = int(rest.split("/")[1])
        if not 0 <= index < len(self.examples[kind]):
            raise ValueError("Unknown example image")
        frame = cv2.imread(str(self.examples[kind][index]))
        with self.lock:
            return self.run_road(frame) if kind == "road" else self.run_signs(frame)


def load(device, detector, options, lock):
    """(Extras, None) when both pages can run, else (None, reason). Never raises."""
    try:
        return Extras(device, detector, options, lock), None
    except Exception as error:  # any missing package, file or model keeps the detector page running
        return None, f"{type(error).__name__}: {error}"


def handle(handler, extras, reason, page_html, method):
    """Serve /road, /signs and /api/road|signs/... Returns False for routes this module does not own."""
    from urllib.parse import urlsplit
    route = urlsplit(handler.path).path
    page = PAGES.get(route)
    match = re.fullmatch(r"/api/(road|signs)/(examples|image/\d+|predict/\d+|upload)", route)
    if page is None and match is None:
        return False

    def reply(status, mime, content):
        handler.send_response(status)
        handler.send_header("Content-Type", mime)
        handler.send_header("Content-Length", str(len(content)))
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        handler.wfile.write(content)

    if page is not None and method == "GET":
        reply(200, "text/html; charset=utf-8", render(page, page_html, None if extras else reason).encode())
        return True
    if match is None or extras is None:
        reply(503 if extras is None else 404, "application/json",
              json.dumps({"error": "This page is unavailable on this machine: " + (reason or "not found")}).encode())
        return True
    kind, rest = match.groups()
    try:
        if method == "GET" and rest.split("/")[0] in ("examples", "image"):
            reply(*extras.get(kind, rest))
        elif method == "POST" and rest.split("/")[0] in ("predict", "upload"):
            length = int(handler.headers.get("Content-Length", "0") or 0)
            if length > MAX_UPLOAD_BYTES:
                raise ValueError("Images must be 25 MB or smaller")
            body = handler.rfile.read(length) if length else b""
            reply(200, "application/json", json.dumps(extras.post(kind, rest, body)).encode())
        else:
            reply(405, "application/json", b'{"error": "Method not allowed"}')
    except ValueError as error:
        reply(400, "application/json", json.dumps({"error": str(error)}).encode())
    except Exception as error:
        print(f"{kind} page failed: {error}", flush=True)
        reply(500, "application/json", b'{"error": "Inference failed. Check the terminal for details."}')
    return True


# ---------------------------------------------------------------- page content
EXTRA_CSS = """
nav a.here{color:var(--dim);pointer-events:none}.kv{display:grid;grid-template-columns:auto 1fr;gap:6px 18px;font-size:14px;margin:0}.kv dt{color:var(--dim)}.kv dd{margin:0;font-variant-numeric:tabular-nums}
.signs{display:flex;flex-wrap:wrap;gap:14px}.sign{background:var(--card2);border-radius:10px;padding:12px;font-size:13px;min-width:200px}.sign img{height:72px;border-radius:4px;background:#fff;margin-right:6px;vertical-align:middle}
.sign .val{font-size:30px;letter-spacing:-.03em;line-height:1.1;margin:6px 0 2px}.steps{counter-reset:s;list-style:none;padding:0;margin:10px 0;max-width:860px}.steps li{counter-increment:s;padding:10px 0 10px 44px;position:relative;border-bottom:1px solid #b5b5b5;line-height:1.55}
.steps li:before{content:counter(s);position:absolute;left:0;top:9px;width:28px;height:28px;border-radius:50%;background:#111;color:#fff;font-size:13px;display:flex;align-items:center;justify-content:center}
.unavailable{background:var(--card);border-radius:14px;padding:36px;margin:0 0 8px;color:#ddd}
"""

VIEWER = """<section class="viewer" id="demo"><h2>{h}</h2><p class="sub">{sub}</p>
<div class="toolbar"><button class="pill" id="previous">Previous</button><select id="frames" aria-label="Choose image"></select><button class="pill" id="next">Next</button><button class="pill accent" id="predict">Run inference</button><button class="pill" id="upload">Upload image</button><input id="file" type="file" accept="image/png,image/jpeg,image/webp,image/bmp" hidden><label class="toggle"><input id="overlay" type="checkbox">Show overlay</label></div>
<div id="status" role="status">Loading examples...</div><div class="scene"><img id="image" alt="CARLA driving scene"></div><div class="det" id="result"></div></section>"""

COMMON_JS = r"""
const $=id=>document.getElementById(id);
function drawField(){const c=$('field'),r=c.getBoundingClientRect(),d=window.devicePixelRatio||1;c.width=r.width*d;c.height=r.height*d;const g=c.getContext('2d');g.scale(d,d);
const vx=r.width*.62,vy=r.height*.30;let s=7;const rnd=()=>(s=(s*16807)%2147483647)/2147483647;
for(let k=-14;k<=14;k++){const ex=vx+k*r.width*.11;for(let i=0;i<70;i++){const t=Math.pow(rnd(),1.6);const x=vx+(ex-vx)*t+(rnd()-.5)*10*t,y=vy+(r.height*1.15-vy)*t+(rnd()-.5)*8*t;
const fade=Math.min(1,Math.max(.08,(x-r.width*.12)/(r.width*.38)));const al=Math.min(.85,.12+t*.9)*(Math.abs(k)<2?1:.75)*fade;g.fillStyle='rgba(255,255,255,'+al+')';g.beginPath();g.arc(x,y,.5+t*1.7*rnd()+.3,0,6.283);g.fill();}}}
drawField();window.addEventListener('resize',drawField);
if($('frames')){
let examples=[],index=0,busy=false;const results=new Map();
function fill(){$('frames').replaceChildren();examples.forEach((e,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i+1)+' / '+examples.length+' - '+e.name+(e.split?' ('+e.split+')':'');$('frames').append(o);});}
function render(){const r=results.get(index),e=examples[index]||{};$('frames').value=String(index);$('overlay').disabled=!r;if(!r)$('overlay').checked=false;
$('image').src=r&&$('overlay').checked?'data:image/jpeg;base64,'+r.overlay:API+'/image/'+index;$('status').className='';
$('status').textContent=r?'Frame '+e.name+': '+r.ms+' ms on '+r.device+'.':'Frame '+e.name+': original image.';showResult(r);}
function move(d){index=(index+d+examples.length)%examples.length;render();}
function setBusy(b){busy=b;$('predict').disabled=b;$('upload').disabled=b;}
async function predict(){if(busy||!examples.length)return;setBusy(true);const q=index;$('status').textContent='Running model inference...';
try{const res=await fetch(API+'/predict/'+q,{method:'POST'});const d=await res.json();if(!res.ok)throw Error(d.error);results.set(q,d);if(index===q){$('overlay').checked=true;render();}}
catch(err){$('status').className='error';$('status').textContent=err.message;}finally{setBusy(false);}}
async function upload(){const file=$('file').files[0];$('file').value='';if(!file||busy)return;setBusy(true);
try{const res=await fetch(API+'/upload',{method:'POST',headers:{'Content-Type':file.type||'application/octet-stream'},body:file});const d=await res.json();if(!res.ok)throw Error(d.error);examples.push(d.example);fill();index=d.index;render();}
catch(err){$('status').className='error';$('status').textContent=err.message;}finally{setBusy(false);}}
$('previous').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('predict').onclick=predict;$('overlay').onchange=render;$('frames').onchange=()=>{index=Number($('frames').value);render();};
$('upload').onclick=()=>$('file').click();$('file').onchange=upload;
document.onkeydown=ev=>{if(ev.target.tagName==='SELECT')return;if(ev.key==='ArrowLeft')move(-1);if(ev.key==='ArrowRight')move(1);if(ev.code==='Space'){ev.preventDefault();predict();}if(ev.key.toLowerCase()==='t'&&results.has(index)){$('overlay').checked=!$('overlay').checked;render();}};
fetch(API+'/examples').then(r=>r.json()).then(d=>{if(d.error)throw Error(d.error);examples=d;fill();render();}).catch(err=>{$('status').className='error';$('status').textContent=err.message;});}
"""


def card(h2, svg, text, links):
    pills = "".join(f'<a class="pill" href="{u}">{t}</a>' for t, u in links)
    return (f'<div class="card"><h2>{h2}</h2><div class="icon"><svg viewBox="0 0 190 190" fill="none" stroke="#fff" '
            f'stroke-width="1.6">{svg}</svg></div><p>{text}</p><div>{pills}</div></div>')


def table(head, rows):
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return '<div class="wrap"><table><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>" + body + "</table></div>"


def low(v):
    return f'<span class="low">{v}</span>'


ICON_TEST = ('<path d="M95 18 V172" stroke="#666"/><ellipse cx="95" cy="62" rx="62" ry="22"/><ellipse cx="95" cy="98" rx="52" '
             'ry="18" fill="#fff" fill-opacity=".08" stroke="#888"/><ellipse cx="95" cy="128" rx="34" ry="12" fill="#fff" '
             'fill-opacity=".08" stroke="#666"/><circle cx="95" cy="150" r="8" fill="#fff" stroke="none"/>')
ICON_USE = ('<ellipse cx="95" cy="95" rx="34" ry="78" transform="rotate(-40 95 95)" fill="#fff" fill-opacity=".1" stroke="none"/>'
            '<ellipse cx="95" cy="95" rx="30" ry="74" transform="rotate(40 95 95)"/><path d="M18 95 H172" stroke="#777"/>'
            '<circle cx="95" cy="95" r="8" fill="#fff" stroke="none"/>')

ROAD = dict(
    title="CARLA road model", here="Road model", api="/api/road", h1="Reading the lane<br>and the crossings",
    lede="One network does three jobs on each 1280 x 720 frame: a mask of the painted lane lines, boxes on pedestrian crossings, and where the centre of our lane is. The driving agent uses the lane offset and heading on open road.",
    viewer=VIEWER.format(h="Run the road model", sub="All 40 frames come from the road model's 10 reserved test episodes, never used for its training or model selection: 4 per episode, picked by a fixed seeded rule. Green is lane paint, magenta boxes are crossings, cyan lines are our lane edges and the red line is the lane centre."),
    body='<section class="cards" aria-label="Summary">' +
    card("One network, three jobs",
         '<path d="M30 160 L80 40"/><path d="M160 160 L110 40"/><path d="M95 160 L95 40" stroke="#777" stroke-dasharray="6 6"/><rect x="58" y="118" width="74" height="22" fill="#fff" fill-opacity=".1"/><circle cx="95" cy="100" r="8" fill="#fff" stroke="none"/>',
         "Lane-paint mask, crossing boxes and lane-centre geometry from one 9.56 M parameter model (38 MB), about 50 ms per frame on an Apple GPU.", [("Try it", "#demo")]) +
    card("Scored once on 10 unseen episodes", ICON_TEST,
         "Lane paint F1 0.895 (2 px tolerance) and crossing AP50 0.784 on 200 reserved test images. Lane-centre offset within 0.10 lane widths on 93% of frames.", [("See the test", "#test")]) +
    card("What the agent takes from it", ICON_USE,
         "Lane offset and heading on open road; crossing distance from the boxes. At junctions, where paint disappears, the planned route takes over.", [("How it is used", "#use")]) +
    "</section>"
    '<section class="light" id="test"><h2>Reserved test</h2><p class="intro">200 images from 10 episodes kept out of training and model selection, evaluated once. The checkpoint was chosen on validation by a rule fixed in advance. 95% ranges come from resampling whole episodes.</p>'
    '<div class="bignums"><div><b>0.895</b>lane paint F1, 2 px tolerance</div><div><b>0.784</b>crossing AP50</div><div><b>93%</b>frames with lane offset within 0.10 lane widths</div></div>' +
    table(["Task", "Metric", "Test value", "95% range", "Target"], [
        ["Lane paint", "F1 with 2 px tolerance", "0.895", "0.872 to 0.926", "0.80, met"],
        ["Lane paint", "Strict pixel IoU", low("0.525"), "0.470 to 0.598", "reported only"],
        ["Crossings", "AP50", "0.784", "0.662 to 0.899", "0.75, met"],
        ["Crossings", "Precision at confidence 0.25", "0.778", "0.611 to 0.886", "0.85, missed"],
        ["Crossings", "Recall at confidence 0.25", "0.787", "0.694 to 0.882", "0.80, within range"],
        ["Lane centre", "Offset within 0.10 lane widths", "93% (98% high confidence)", "-", "-"],
        ["Lane centre", "Median offset / heading error", "0.003 lane widths / 0.6 deg", "-", "-"]]) +
    "<h3>Validation (300 images, 15 episodes, also chose the checkpoint)</h3>" +
    table(["Task", "Metric", "Value", "95% range"], [
        ["Lane paint", "F1 with 2 px tolerance", "0.920", "0.906 to 0.936"], ["Lane paint", "Strict pixel IoU", "0.579", "0.540 to 0.624"],
        ["Crossings", "AP50", "0.775", "0.677 to 0.863"], ["Crossings", "Precision / recall at 0.25", "0.836 / 0.672", "-"],
        ["Lane centre", "Offset within 0.10 lane widths", "91% (97% high confidence)", "-"]]) + "</section>"
    '<section class="light" id="use"><h2>How the agent uses it</h2><p class="intro">The policy never sees pixels. It reads two numbers from this model, the lane offset and the heading, plus the distance to the next crossing.</p><ol class="steps">'
    "<li>Lane-centre pixels are placed in metres using the depth camera where the pixel shows road, and a flat-road projection otherwise, then a straight line is fitted.</li>"
    "<li>From 15 m before a junction to its exit, lane paint is missing or misleading, so the planned route and the car's pose give the lane instead.</li>"
    "<li>If vision loses the lane, the last good estimate is held for up to 2 s. A reading more than 2 m away from the route's estimate is rejected: on Town02 the model sometimes locks onto the next lane.</li>"
    "<li>In the 22 November evaluation run, vision gave the lane on 53.5% of driving steps and the route at junctions on 33.4%. The remaining 13% came from held estimates, route fallbacks and a route override that has since been removed.</li></ol></section>"
    '<section class="light" id="notes"><h2>What the numbers explain</h2><ul class="findings">'
    "<li>Strict IoU looks low because lane lines are about 6 px wide: a label compared with itself shifted by 1 px already scores only 0.62. The tolerance-based F1 is the fair lane metric.</li>"
    "<li>Distant crossings are the main miss: missed crossings have a median height of about 20 px at the model input. The test set has only 89 crossing boxes, so the crossing ranges are wide.</li>"
    "<li>The lane-centre figures measure agreement with the same estimator run on the hand-drawn lane masks, not true lane-keeping accuracy: the episodes have no recorded car pose.</li>"
    "<li>Lane centre assumes the camera sits on the car's centreline, uses a lane-width calibration specific to this camera, and fits straight lines, so sharp bends are approximate.</li>"
    "<li>Test episodes come from the same capture as training, and route overlap with training is unknown. This is a held-out test, not a blind or safety evaluation.</li></ul></section>",
    script=r"""function showResult(r){if(!r){$('result').innerHTML='<div class="muted">Run inference to see the lane, crossings and lane centre.</div>';return;}
const c=r.lane_center||{};const mode={both:'both lane edges found',left_only:'left edge only, right inferred',right_only:'right edge only, left inferred',none:'no lane found'}[c.mode]||c.mode||'-';
$('result').innerHTML='<div><h3>Lane centre</h3><dl class="kv"><dt>Edges</dt><dd>'+mode+'</dd><dt>Offset</dt><dd>'+(c.offset_norm==null?'-':(c.offset_norm>=0?'+':'')+c.offset_norm.toFixed(2)+' lane widths ('+(c.offset_norm>0?'right of centre':c.offset_norm<0?'left of centre':'centred')+')')+'</dd><dt>Heading</dt><dd>'+(c.heading_deg==null?'-':(c.heading_deg>=0?'+':'')+c.heading_deg.toFixed(1)+' deg')+'</dd><dt>Confidence</dt><dd>'+(c.low_confidence?'<span class="low">low (edge extrapolated)</span>':'normal')+'</dd><dt>Lane pixels</dt><dd>'+r.lane_pixels.toLocaleString()+'</dd></dl></div>'+
'<div><h3>Crossings</h3>'+(r.crosswalks.length?r.crosswalks.map(b=>'<span class="tag">crossing '+b.confidence.toFixed(2)+'</span>').join(''):'<div class="muted">No crossing above confidence 0.25.</div>')+'<p class="muted" style="margin-top:14px">Offset is measured at the bottom of the image: 0 is centred, +0.5 means the car is on the right lane edge. Heading is the lane line angle in the image, not a calibrated yaw.</p></div>';}""")

SIGNS = dict(
    title="CARLA speed signs", here="Speed signs", api="/api/signs", h1="Reading the<br>speed limit",
    lede="The detector finds the sign, a colour filter finds its red ring, and OCR reads the number. The car only acts on a value after several matching reads, and applies it once it passes the sign.",
    viewer=VIEWER.format(h="Read the signs", sub="All 30 frames come from the detector's held-out test episodes and contain a hand-labelled speed-limit sign, picked by a fixed seeded rule (one per episode first). Each detected sign shows the crop, the two cleaned cut-outs sent to OCR, and the value read. Signs under 12 px tall are not read, as in the driving stack."),
    body='<section class="cards" aria-label="Summary">' +
    card("Find the plate",
         '<circle cx="95" cy="80" r="46"/><circle cx="95" cy="80" r="34" stroke="#777"/><path d="M95 126 V176" stroke="#777"/><rect x="40" y="26" width="110" height="154" stroke="#555" stroke-dasharray="5 5"/><circle cx="95" cy="80" r="8" fill="#fff" stroke="none"/>',
         "The detector's speed_limit box often includes the pole. A red-hue filter finds the round red ring inside it, so OCR reads only the plate.", [("Try it", "#demo")]) +
    card("Clean and read",
         '<rect x="40" y="50" width="110" height="80" rx="6" fill="#fff" fill-opacity=".1"/><path d="M62 112 V68 L76 82 M108 68 a14 22 0 1 0 0.1 0"/><circle cx="150" cy="130" r="8" fill="#fff" stroke="none"/>',
         "Enlarge, whiten the ring, stretch the contrast, then RapidOCR reads the digits. Only values CARLA's signs use are kept: 10, and 30 to 130 in tens.", [("The steps", "#how")]) +
    card("Vote before acting",
         '<path d="M30 150 H160" stroke="#777"/><rect x="45" y="100" width="20" height="50" fill="#fff" fill-opacity=".1"/><rect x="85" y="60" width="20" height="90"/><rect x="125" y="120" width="20" height="30" fill="#fff" fill-opacity=".1"/><circle cx="95" cy="44" r="8" fill="#fff" stroke="none"/>',
         "A value needs 3 matching reads to lower the limit and 6 to raise it, and twice the reads of any other value. A false raise means speeding; a missed one only means driving slower.", [("Results", "#results")]) +
    "</section>"
    '<section class="light" id="how"><h2>How a sign is read</h2><ol class="steps">'
    "<li>The seven-class detector finds a speed_limit box (held-out test AP50 0.515 for this class).</li>"
    "<li>Inside the box, pixels with red hue and enough saturation are grouped; the largest ring-shaped group, not solid red, is the plate. If none is found, the top square of the box is used.</li>"
    "<li>Two cut-outs are made: the plate with everything outside the inner circle blanked, and an inset rectangle. Both are enlarged to 80 px, the ring is turned white and the contrast stretched.</li>"
    "<li>RapidOCR 1.2.3 reads both cut-outs; its recogniser is called directly because the library's own 0.5 score cut-off drops most small signs. The best valid value wins.</li>"
    "<li>While driving, the nearest sign is read at most every 0.1 s. Its distance comes from the depth camera; a confirmed sign last seen within 15 m that then leaves the view counts as passed, and the limit is applied.</li></ol></section>"
    '<section class="light" id="results"><h2>Results while driving</h2><p class="intro">Held-out Town02 runs, 22 November 2026. This measures the speed limit the car held, not a per-sign accuracy test: no per-sign value labels exist.</p>'
    '<div class="bignums"><div><b>99.0%</b>driving steps where the held limit matched CARLA (11,695 steps after the first sign)</div><div><b>834</b>successful reads</div><div><b>550</b>failed reads</div></div>' +
    table(["Item", "Value"], [["Steps with a limit read and held", "11,695"], ["Held limit matched CARLA's true limit", "99.0%"],
                              ["Steps before the first sign (default speed applies)", "4,806"], ["OCR reads that gave a valid value", "834"],
                              ["OCR reads with no valid value", "550"], ["Steps with no sign visible", "13,591"]]) + "</section>"
    '<section class="light" id="notes"><h2>What the numbers explain</h2><ul class="findings">'
    "<li>Many single reads fail (550 against 834), mostly on small or distant signs, but voting over frames means one bad read never changes the limit. That is why the held limit is right 99.0% of the time.</li>"
    "<li>The 99.0% is time-weighted agreement while driving. It is not the share of signs read correctly, and a sign that is never read leaves the previous limit in place.</li>"
    "<li>Raising needs more votes than lowering on purpose: in an earlier run, 60 was read inside 30 zones. A missed raise only costs time.</li>"
    "<li>Distance to the sign comes from CARLA's simulated depth camera, not from vision. A camera-only car would need a distance estimate from the image.</li>"
    "<li>The reader is tuned to CARLA's round red-ringed signs and their fixed set of values. Real-world signs in other styles are untested.</li></ul></section>",
    script=r"""function showResult(r){if(!r){$('result').innerHTML='<div class="muted">Run inference to detect and read the signs.</div>';return;}
const s=r.signs;$('result').innerHTML='<div><h3>Speed-limit signs</h3>'+(s.length?'<div class="signs">'+s.map(x=>'<div class="sign"><div>'+(x.crop?'<img src="data:image/jpeg;base64,'+x.crop+'" alt="sign crop">':'')+x.cores.map(c=>'<img src="data:image/jpeg;base64,'+c+'" alt="cleaned cut-out">').join('')+'</div><div class="val">'+(x.value?x.value+' km/h':'<span class="low">no value</span>')+'</div><div class="muted">detector '+x.confidence.toFixed(2)+(x.read?', OCR score '+x.score.toFixed(2):'')+', '+x.height+' px tall'+(x.read?'':' (too small, not read)')+'</div></div>').join('')+'</div>':'<div class="muted">No speed-limit sign detected above confidence 0.25.</div>')+
'</div><div><h3>Other detections</h3>'+(r.others.length?r.others.map(d=>'<span class="tag">'+d+'</span>').join(''):'<div class="muted">None.</div>')+'<p class="muted" style="margin-top:14px">A single frame shows single reads. While driving, a value only counts after several matching reads.</p></div>';}""")

PAGES = {"/road": ROAD, "/signs": SIGNS}


def render(page, page_html, unavailable):
    style = re.search(r"<style>(.*?)</style>", page_html, re.S).group(1) + EXTRA_CSS
    nav = "".join(f'<a href="{u}"{" class=here" if n == page["here"] else ""}>{n}</a>'
                  for n, u in (("Detector", "/"), ("Road model", "/road"), ("Speed signs", "/signs")))
    if unavailable:
        main = (f'<section class="unavailable"><h2>Unavailable on this machine</h2><p>{unavailable}</p>'
                '<p class="muted">The detector page still works. Install requirements-perception.txt and restart demo.py.</p></section>')
        script = ""
    else:
        main = page["viewer"] + page["body"]
        script = f"const API='{page['api']}';\n{page['script']}"
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{page['title']}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>{style}</style>
<div class="frame"><div class="topbar"><span>AAI3001 Computer Vision and Deep Learning, team P1-21</span><span>CARLA 0.9.16 &middot; camera perception</span></div>
<header class="hero"><canvas id="field" aria-hidden="true"></canvas><div><div class="brand"><svg viewBox="0 0 26 26" fill="none" stroke="#fff" stroke-width="2"><rect x="3" y="6" width="20" height="14" rx="2"/><circle cx="13" cy="13" r="3.2" fill="#fff" stroke="none"/></svg>CARLA perception</div>
<h1>{page['h1']}</h1></div><nav aria-label="Models">{nav}</nav><p class="lede">{page['lede']}</p></header>
{main}
<footer>Uploaded images are kept for this session only and are not part of any evaluation split. Keyboard: <kbd>Left</kbd> / <kbd>Right</kbd> change image, <kbd>Space</kbd> runs inference, <kbd>T</kbd> toggles the overlay.</footer></div>
<script>{script}
{COMMON_JS}</script></html>"""
