"""Browser camera viewer. The training client retains exclusive simulation ticks."""

import argparse
import json
import struct
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .config import Config
from .telemetry import read_telemetry, telemetry_path


PAGE = b'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>CARLA live view</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#111820;color:#e4edf4;font:16px system-ui,sans-serif}
main{max-width:1280px;margin:28px auto;padding:0 24px}h1{font-size:26px;margin-bottom:8px}
h2{font-size:18px;margin:0 0 12px}p,.muted{color:#aab9c7;line-height:1.6}
.screen{background:#080c10;min-height:240px;aspect-ratio:16/9;display:grid;place-items:center;
border:1px solid #334454;border-radius:12px;overflow:hidden;position:relative}
#camera{width:100%;display:none}#waiting{position:absolute;padding:24px;text-align:center}
#status{display:block;margin:16px 0;color:#8edab1}.cards,.details{display:grid;gap:16px;margin:16px 0}
.cards{grid-template-columns:repeat(4,minmax(0,1fr))}.details{grid-template-columns:1fr 1fr}
.panel{background:#1a2530;border:1px solid #334454;border-radius:12px;padding:18px;min-width:0}
.value{font-size:32px;font-weight:650;display:block;margin:10px 0}.caption{font-size:13px;color:#aab9c7}
.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:10px 7px;border-bottom:1px solid #334454}th{color:#aab9c7;font-weight:500}
select{background:#111820;color:#e4edf4;border:1px solid #526575;border-radius:6px;padding:8px;margin:8px 0 14px}
.bar{display:grid;grid-template-columns:minmax(120px,1fr) 2fr 58px;align-items:center;gap:8px;margin:9px 0;font-size:13px}
.track{height:9px;background:#080c10;border-radius:5px;overflow:hidden}.fill{height:100%;background:#79c8e8}
.positive{color:#8edab1}.negative{color:#f5a7a7}.comparison{margin-top:8px}
progress,meter{width:100%;height:16px;accent-color:#79c8e8}.control{margin:14px 0}
.control label{display:flex;justify-content:space-between;margin-bottom:6px;font-size:14px}
#telemetry-status{padding:10px 0;color:#f3d28a}#source{overflow-wrap:anywhere}
@media(max-width:800px){.details{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.panel{padding:12px}.value{font-size:24px}main{padding:0 12px}}
</style><main><h1>CARLA live view</h1>
<p>Follow the current driving episode. State and attention use the agent's structured observations.</p>
<div class="screen"><img id="camera" alt="Live chase camera"><span id="waiting">Connecting to CARLA...</span></div>
<span id="status" role="status">Connecting...</span>
<div id="telemetry-status" role="status">Waiting for driving telemetry...</div>
<div class="cards">
<section class="panel"><h2>Current speed</h2><span class="value" id="speed">--</span><span class="caption" id="speed-si">Waiting for state</span>
<div class="caption comparison" id="target-speed">Target: --</div><div class="caption" id="speed-difference">Actual vs target: --</div></section>
<section class="panel"><h2>Objects recognized</h2><span class="value" id="object-count">--</span><span class="caption">Present in the agent's object slots</span></section>
<section class="panel"><h2>Episode reward</h2><span class="value" id="reward">--</span><span class="caption">Accumulated since this episode started</span></section>
<section class="panel"><h2>Route completion</h2><span class="value" id="route-completion">--</span>
<progress id="route-progress" max="1" value="0" aria-label="Route completion" hidden></progress><span class="caption" id="route-remaining">Distance remaining: --</span></section>
</div>
<div class="details">
<section class="panel"><h2>Reward breakdown</h2><p class="caption">Signed contributions: positive rewards and negative costs.</p>
<div class="table-wrap"><table><thead><tr><th>Component</th><th>Last decision</th><th>Episode total</th></tr></thead><tbody id="reward-components"></tbody>
<tfoot><tr><th>Total</th><td id="reward-last">--</td><td id="reward-breakdown-total">--</td></tr></tfoot></table></div>
<p class="caption" id="reward-note">Waiting for reward components.</p></section>
<section class="panel"><h2>Applied controls</h2>
<div class="control"><label for="steering-meter">Steering <span id="steering">--</span></label>
<meter id="steering-meter" min="-1" max="1" low="-1" high="1" optimum="0" value="0" hidden></meter><span class="caption">-1 left / 0 straight / +1 right</span></div>
<div class="control"><label for="throttle-meter">Throttle <span id="throttle">--</span></label><progress id="throttle-meter" max="1" value="0" hidden></progress></div>
<div class="control"><label for="brake-meter">Brake <span id="brake">--</span></label><progress id="brake-meter" max="1" value="0" hidden></progress></div>
<p class="caption">Controls applied during the latest decision interval, after action limits.</p></section>
<section class="panel"><h2>Observed objects</h2><p class="caption" id="source">Observation source unavailable</p>
<div class="table-wrap"><table><thead><tr><th>Class / ID</th><th>Forward (m)</th><th>Left (m)</th><th>Confidence</th></tr></thead><tbody id="objects"></tbody></table></div>
<p class="caption" id="objects-note">Waiting for observations.</p></section>
<section class="panel"><h2>Attention over inputs</h2>
<label for="attention-head" class="caption">Attention head</label><br>
<select id="attention-head"><option value="mean">Mean of all heads</option></select>
<div id="attention-bars"></div><p class="caption" id="attention-note">Waiting for model attention.</p>
<p class="caption">The ego query attends to state, route, weather, controls, waypoints and objects.
Weights sum to 100% per head; empty object slots are masked. These are input weights, not a camera heatmap or a causal explanation of the action.</p>
</section></div><p class="caption">Close the viewer process with Ctrl+C when finished.</p></main>
<script>
const $=id=>document.getElementById(id),img=$('camera'),waiting=$('waiting');
let shown=null,latest=null;
function text(id,value){$(id).textContent=value}
function signed(value){return (value>0 ? '+' : '')+value.toFixed(2)}
function drivingDetails(data){
  const target=data?.target_speed_mps,completion=data?.route_completion,remaining=data?.route_remaining_m;
  text('target-speed',target==null ? 'Target: --' : 'Target: '+(target*3.6).toFixed(1)+' km/h ('+target.toFixed(2)+' m/s)');
  const difference=target==null ? null : (data.speed_mps-target)*3.6;
  text('speed-difference',difference==null ? 'Actual vs target: --' : 'Actual vs target: '+signed(difference)+' km/h');
  text('route-completion',completion==null ? '--' : (100*completion).toFixed(1)+'%');
  $('route-progress').hidden=completion==null;$('route-progress').value=completion ?? 0;
  text('route-remaining',remaining==null ? 'Distance remaining: --' : 'Distance remaining: '+remaining.toFixed(1)+' m');
  const action=data?.applied_action;
  ['steering','throttle','brake'].forEach((name,i)=>{
    const value=action?.[i];text(name,value==null ? '--' : i===0 ? signed(value) : (100*value).toFixed(1)+'%');
    $(name+'-meter').hidden=value==null;$(name+'-meter').value=value ?? 0;
  });
  const body=$('reward-components');body.replaceChildren();
  const totals=data?.reward_component_totals ?? {},components=data?.reward_components ?? {};
  Object.keys(totals).forEach(name=>{
    const row=document.createElement('tr'),label=document.createElement('td');label.textContent=name.replaceAll('_',' ');row.append(label);
    [components[name] ?? 0,totals[name]].forEach(value=>{
      const cell=document.createElement('td');cell.textContent=signed(value);cell.className=value>0 ? 'positive' : value<0 ? 'negative' : '';row.append(cell);
    });body.append(row);
  });
  text('reward-last',data?.reward_last==null ? '--' : signed(data.reward_last));
  text('reward-breakdown-total',data ? signed(data.episode_return) : '--');
  text('reward-note',Object.keys(totals).length ? 'Episode totals reset at the start of each driving episode.' : data ? 'No reward components yet; waiting for a driving decision.' : 'Waiting for reward components.');
}
function attentionView(){
  const attention=latest && latest.attention,container=$('attention-bars');container.replaceChildren();
  const count=attention ? attention.heads.length : 0,select=$('attention-head');
  if(select.options.length!==count+1){
    select.replaceChildren(new Option('Mean of all heads','mean'));
    for(let i=0;i<count;i++)select.add(new Option('Head '+(i+1),String(i)));
  }
  select.disabled=!count;
  if(!count){text('attention-note',latest ? 'No model attention: bootstrap / expert-only driving does not run the encoder.' : 'Waiting for model attention.');return}
  const weights=select.value==='mean' ? attention.labels.map((_,i)=>attention.heads.reduce((sum,head)=>sum+head[i],0)/count) : attention.heads[Number(select.value)];
  attention.labels.forEach((label,i)=>{
    if(i>=9 && !latest.objects.some(object=>object.slot===i-9))return;
    const row=document.createElement('div');row.className='bar';
    const name=document.createElement('span');name.textContent=label;
    const track=document.createElement('div');track.className='track';
    const fill=document.createElement('div');fill.className='fill';fill.style.width=(100*weights[i])+'%';track.append(fill);
    const value=document.createElement('span');value.textContent=(100*weights[i]).toFixed(1)+'%';
    row.append(name,track,value);container.append(row);
  });
  text('attention-note','Actual encoder weights for the displayed state at simulation frame '+latest.frame+'.');
}
$('attention-head').onchange=attentionView;
function stateView(data){
  latest=data;drivingDetails(data);const body=$('objects');body.replaceChildren();
  if(!data){
    text('telemetry-status','State unavailable. Start or restart collection/training with telemetry enabled by this version.');
    ['speed','object-count','reward'].forEach(id=>text(id,'--'));text('speed-si','Waiting for state');
    text('source','Observation source unavailable');text('objects-note','Waiting for observations.');attentionView();return;
  }
  const episode='Episode '+data.episode_id+' | '+data.controller+' | State frame '+data.frame;
  const upgrade=data.missing_fields?.length ? ' | Some panels unavailable: the driving process is using older telemetry. Save progress and restart collection/training to enable target speed, route, controls and reward breakdown.' : '';
  text('telemetry-status',episode+(data.stale ? ' | Stale: last update '+data.age_seconds.toFixed(1)+' s ago' : data.active ? ' | Live' : ' | Finished')+upgrade);
  text('speed',(data.speed_mps*3.6).toFixed(1)+' km/h');text('speed-si',data.speed_mps.toFixed(2)+' m/s');
  text('object-count',data.objects.length);text('reward',data.episode_return.toFixed(2));
  text('source',data.source==='carla_ground_truth_with_geometric_visibility' ? 'Source: CARLA ground truth with geometric visibility filtering.' : 'Observation source: '+data.source);
  data.objects.forEach(object=>{
    const row=document.createElement('tr');
    [object.class+' #'+object.id,object.forward_m.toFixed(1),object.left_m.toFixed(1),(100*object.confidence).toFixed(0)+'%'].forEach(value=>{
      const cell=document.createElement('td');cell.textContent=value;row.append(cell);
    });body.append(row);
  });
  text('objects-note',data.objects.length ? 'Positions are relative to the ego vehicle. Confidence comes from the observation provider.' : 'No objects present in the agent observation.');
  attentionView();
}
async function refresh(){
  try{
    const status=await (await fetch('/status',{cache:'no-store'})).json();
    text('status',status.message);waiting.textContent=status.message;stateView(status.telemetry);
    if(status.has_frame){
      const response=await fetch('/frame.png',{cache:'no-store'});
      if(response.ok && response.status!==204){
        const next=URL.createObjectURL(await response.blob());
        await new Promise(resolve=>{img.onload=resolve;img.onerror=resolve;img.src=next});
        if(shown)URL.revokeObjectURL(shown);shown=next;
        img.style.display='block';waiting.style.display='none';
      }
    }else{img.style.display='none';waiting.style.display='block'}
  }catch(error){
    text('status','Viewer disconnected; check the terminal.');
    text('telemetry-status','Disconnected: displayed values are from the last successful update.');
  }
  setTimeout(refresh,150);
}refresh();
</script></html>'''

def image_png(image):
    """Encode CARLA BGRA pixels using only NumPy and the Python standard library."""
    pixels = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)
    rows = np.zeros((image.height, 1 + image.width * 3), np.uint8)
    rows[:, 1:] = pixels[:, :, :3][:, :, ::-1].reshape(image.height, -1)
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
    return (b'\x89PNG\r\n\x1a\n' +
            chunk(b'IHDR', struct.pack('!2I5B', image.width, image.height, 8, 2, 0, 0, 0)) +
            chunk(b'IDAT', zlib.compress(rows.tobytes(), 1)) + chunk(b'IEND', b''))


class CameraFeed:
    def __init__(self, cfg, width=960, height=540):
        self.cfg, self.width, self.height = cfg, width, height
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.camera = None
        self.actor_id = None
        self.frame = None
        self.message = 'Connecting to CARLA...'

    def snapshot(self):
        with self.lock:
            return self.frame, self.message

    def telemetry(self):
        with self.lock:
            actor_id = self.actor_id
        return read_telemetry(telemetry_path(self.cfg), actor_id) if actor_id is not None else None

    def release_camera(self):
        camera, self.camera = self.camera, None
        with self.lock:
            self.actor_id, self.frame = None, None
        if camera is not None:
            try:
                if camera.is_alive:
                    camera.stop()
                    camera.destroy()
            except RuntimeError:
                pass  # The training client may already have removed the parent.

    def receive(self, image, actor_id):
        png = image_png(image)
        with self.lock:
            if self.actor_id == actor_id:
                self.frame = png
                self.message = f'Live camera | Vehicle {actor_id} | Frame {image.frame}'

    def follow(self):
        import carla

        client = carla.Client(self.cfg.environment.host, self.cfg.environment.port)
        client.set_timeout(3.0)
        try:
            while not self.stopped.is_set():
                try:
                    world = client.get_world()
                    ego = next((actor for actor in world.get_actors().filter('vehicle.*')
                                if actor.attributes.get('role_name') == 'world_model_rl'), None)
                    if ego is None or ego.id != self.actor_id or self.camera is None or not self.camera.is_alive:
                        self.release_camera()
                        with self.lock:
                            self.message = 'Waiting for the next driving episode...'
                        if ego is not None:
                            blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
                            blueprint.set_attribute('image_size_x', str(self.width))
                            blueprint.set_attribute('image_size_y', str(self.height))
                            blueprint.set_attribute('fov', '100')
                            blueprint.set_attribute('sensor_tick', '0.1')
                            camera = world.spawn_actor(blueprint,
                                carla.Transform(carla.Location(x=-6, z=3), carla.Rotation(pitch=-15)),
                                attach_to=ego, attachment_type=carla.AttachmentType.Rigid)
                            self.camera = camera
                            with self.lock:
                                self.actor_id = ego.id
                                self.message = 'Camera attached; waiting for simulation frames...'
                            camera.listen(lambda image, identity=ego.id: self.receive(image, identity))
                    self.stopped.wait(0.5)
                except RuntimeError as exc:
                    self.release_camera()
                    with self.lock:
                        self.message = f'Waiting for CARLA: {exc}'
                    self.stopped.wait(1.0)
        finally:
            self.release_camera()


def handler_for(feed):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            frame, message = feed.snapshot()
            if self.path == '/':
                code, mime, body = 200, 'text/html; charset=utf-8', PAGE
            elif self.path == '/status':
                code, mime, body = 200, 'application/json', json.dumps(
                    {'message': message, 'has_frame': frame is not None,
                     'telemetry': feed.telemetry()}).encode()
            elif self.path == '/frame.png':
                code, mime, body = (200 if frame is not None else 204), 'image/png', frame or b''
            else:
                code, mime, body = 404, 'text/plain', b'Not found'
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--port', type=int, default=8080, help='Local browser port (default: 8080)')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    cfg = Config.load(args.config)
    feed = CameraFeed(cfg)
    with ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(feed)) as server:
        worker = threading.Thread(target=feed.follow, daemon=True)
        worker.start()
        print(f'Open http://127.0.0.1:{args.port} to watch driving episodes. Ctrl+C stops the viewer.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            feed.stopped.set()
            worker.join(timeout=8)


if __name__ == '__main__':
    main()
