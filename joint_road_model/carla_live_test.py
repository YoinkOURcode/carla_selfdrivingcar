"""Live CARLA test of the joint road model. The car drives on autopilot; the model sees only the camera image.
CARLA's map supplies an independent reference: true lane-centre offset (waypoints) and crossing locations (map.get_crosswalks()).

    python carla_live_test.py --check                                  # only checks the connection and versions
    python carla_live_test.py --town Town03 --seconds 120 --out runs\\town03

Start CARLA first. Press q in the window to stop early. The script uses synchronous mode, so model speed cannot distort the simulation,
and restores CARLA's settings and removes its actors when it ends.

Outputs in --out: overlay.mp4, frames.csv, summary.json, snapshots/ (every 40th processed frame).
Overlay colours: green lane paint, magenta model crossing, cyan lane edges, red lane centre, white car position, yellow CARLA map crossing.

Reference limits (read before quoting numbers):
  * Map crossings are projected geometrically, with no occlusion check, so a crossing hidden behind a building still counts as visible.
    Crossing counts here are a sanity check and are not comparable with the AP/precision/recall measured on labelled frames.
  * The offset reference is at the camera's position; the model estimates at the bottom image row, a few metres ahead. Curves add some error.
"""
import argparse,csv,json,math,sys,time
from pathlib import Path
import numpy as np,cv2

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=2000);p.add_argument('--tm-port',type=int,default=8000)
    p.add_argument('--check',action='store_true',help='check the connection and versions, then exit')
    p.add_argument('--town',help='map to load, e.g. Town03 (default: keep the current map)')
    p.add_argument('--weather',default='ClearNoon',help='carla.WeatherParameters preset, e.g. ClearNoon, WetCloudySunset, HardRainNoon')
    p.add_argument('--seconds',type=float,default=120,help='simulated seconds to record')
    p.add_argument('--every',type=int,default=5,help='run the model on every Nth simulation step (step = 0.05 s)')
    p.add_argument('--spawn',type=int,default=0,help='spawn point index')
    p.add_argument('--device',default='auto',help='auto, cuda or cpu')
    p.add_argument('--conf',type=float,default=.25);p.add_argument('--no-window',action='store_true')
    # Camera. Training frames: 1280x720, centre-mounted, about level (calibrated horizon y=358 of 720), about 2 m high if lanes are ~3.5 m wide.
    # The real mount and FOV are not recorded in the dataset, so x and fov here are assumptions; replace them if you know the capture settings.
    p.add_argument('--fov',type=float,default=90);p.add_argument('--cam-x',type=float,default=1.5);p.add_argument('--cam-y',type=float,default=0.,help='sideways camera shift in metres (+ = right); used for the off-centre test')
    p.add_argument('--cam-z',type=float,default=2.0);p.add_argument('--cam-pitch',type=float,default=0.)
    p.add_argument('--out',type=Path,default=Path('carla_runs/run'));return p.parse_args(argv)

def intrinsics(w,h,fov):return w/(2*math.tan(math.radians(fov)/2)),w/2,h/2

def project(points,world_to_cam,w,h,fov,near=.5):
    """World points (Nx3) to pixels. Returns (pixels of points in front of the camera, their forward distances)."""
    f,cx,cy=intrinsics(w,h,fov);P=np.c_[points,np.ones(len(points))]@np.array(world_to_cam).T
    keep=P[:,0]>near;P=P[keep]
    if not len(P):return np.zeros((0,2)),np.zeros(0)
    return np.c_[cx+f*P[:,1]/P[:,0],cy-f*P[:,2]/P[:,0]],P[:,0]  # CARLA camera axes: x forward, y right, z up

def split_polygons(locations):
    """map.get_crosswalks() returns a flat list of closed polygons: each polygon repeats its first point at its end."""
    polys=[];cur=[]
    for l in locations:
        q=(l.x,l.y,l.z)
        if cur and np.allclose(q,cur[0],atol=1e-3):
            if len(cur)>=3:polys.append(np.array(cur))
            cur=[]
        else:cur.append(q)
    return polys

def visible_crossings(polys,world_to_cam,w,h,fov,max_dist=50.,min_height=6.):
    out=[]
    for poly in polys:
        px,d=project(poly,world_to_cam,w,h,fov)
        if len(px)<3 or d.min()>max_dist:continue
        x1,y1=np.clip(px.min(0),0,[w,h]);x2,y2=np.clip(px.max(0),0,[w,h])
        if x2-x1<10 or y2-y1<min_height:continue
        out.append({'xyxy':[float(x1),float(y1),float(x2),float(y2)],'dist':float(d.min())})
    return out

def box_iou(a,b):
    i=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i
    return i/u if u>0 else 0.

def match(dets,refs,thr=.3):
    """Greedy one-to-one matching, highest confidence first. Returns (matched reference indices, number of unmatched detections)."""
    used=set();fp=0
    for d in sorted(dets,key=lambda b:-b['confidence']):
        s=[(box_iou(d['xyxy'],r['xyxy']),i) for i,r in enumerate(refs) if i not in used]
        best=max(s,default=(0,None))
        if best[0]>=thr:used.add(best[1])
        else:fp+=1
    return used,fp

def true_offset(world_map,loc,driving):
    """Signed offset of loc from its lane centre, in metres and lane widths. + = right of centre, matching the model's offset_norm."""
    wp=world_map.get_waypoint(loc,project_to_road=True,lane_type=driving);c=wp.transform.location;r=wp.transform.get_right_vector()
    m=(loc.x-c.x)*r.x+(loc.y-c.y)*r.y+(loc.z-c.z)*r.z
    return m,m/wp.lane_width,wp.lane_width,wp.is_junction

def stats(v):
    v=np.abs(np.array(v,dtype=float))
    return None if not len(v) else {'n':int(len(v)),'median':float(np.median(v)),'p90':float(np.percentile(v,90)),'within_0.05':float((v<=.05).mean()),'within_0.10':float((v<=.10).mean())}

def summarise(rows,a,meta):
    have=[r for r in rows if r['offset_norm'] is not None];err=lambda rs:[r['offset_norm']-r['true_offset_norm'] for r in rs]
    good=[r for r in have if not r['low_confidence']];road=[r for r in good if not r['is_junction']]
    bands=[(0,15),(15,30),(30,50)];cross={}
    for lo,hi in bands:
        t=sum(r['refs_band'][f'{lo}-{hi}m'][0] for r in rows);m=sum(r['refs_band'][f'{lo}-{hi}m'][1] for r in rows)
        cross[f'{lo}-{hi}m']={'map_crossings_in_view':t,'matched_by_model':m,'share_matched':m/t if t else None}
    dets=sum(r['n_crossings'] for r in rows);fp=sum(r['unmatched_detections'] for r in rows)
    widths=[r['lane_width_m'] for r in rows];k=float(np.median(widths))/a.cam_z if widths else None
    return {**meta,'frames':len(rows),'mean_ms_per_frame':float(np.mean([r['ms'] for r in rows])) if rows else None,
        'lane_centre':{'frames_with_estimate':len(have),'share_with_estimate':len(have)/len(rows) if rows else None,'share_low_confidence':float(np.mean([bool(r['low_confidence']) for r in have])) if have else None,
            'mean_true_offset_norm':float(np.mean([r['true_offset_norm'] for r in rows])) if rows else None,'mean_model_offset_norm':float(np.mean([r['offset_norm'] for r in have])) if have else None,
            'abs_error_lane_widths_all':stats(err(have)),'abs_error_excl_low_confidence':stats(err(good)),'abs_error_excl_low_confidence_and_junctions':stats(err(road))},
        'crossings':{'by_distance':cross,'model_detections':dets,'detections_not_matching_any_map_crossing':fp,
            'note':'Sanity check only: no occlusion handling, match = box IoU >= 0.3. Not comparable with labelled-frame AP/precision/recall.'},
        'camera_check':{'k_from_this_setup':k,'k_in_calibration':meta['calibration_k'],'note':'k = lane width / camera height. A big difference (over ~25%) means this camera height differs from the training camera, so one-side lane-centre estimates will be off.'}}

def run(a,carla):
    client=carla.Client(a.host,a.port);client.set_timeout(60.)
    server,local=client.get_server_version(),client.get_client_version()
    print(f'CARLA server {server}, python client {local}')
    if server.split('-')[0]!=local.split('-')[0]:print('WARNING: server and client versions differ. Install the matching version: pip install carla==<server version>')
    world=client.get_world();print('current map:',world.get_map().name)
    if a.check:return None
    import torch
    from road_perception import RoadPerception
    device=a.device if a.device!='auto' else 'cuda' if torch.cuda.is_available() else 'cpu'
    rp=RoadPerception(HERE/'joint_road_model.pt',HERE/'lane_center_calibration.json',device,conf=a.conf);print('model device:',device)
    if a.town and a.town not in world.get_map().name:world=client.load_world(a.town)
    world.set_weather(getattr(carla.WeatherParameters,a.weather));world_map=world.get_map()
    polys=split_polygons(world_map.get_crosswalks());print(len(polys),'crossings in this map')
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'snapshots').mkdir(exist_ok=True)
    original=world.get_settings();tm=client.get_trafficmanager(a.tm_port);actors=[];vw=None;rows=[];W,H=1280,720
    try:
        s=world.get_settings();s.synchronous_mode=True;s.fixed_delta_seconds=.05;world.apply_settings(s);tm.set_synchronous_mode(True)
        lib=world.get_blueprint_library();bps=lib.filter('vehicle.lincoln.mkz*') or lib.filter('vehicle.tesla.model3') or lib.filter('vehicle.*')
        spawns=world_map.get_spawn_points();veh=world.spawn_actor(bps[0],spawns[a.spawn%len(spawns)]);actors.append(veh)
        bp=lib.find('sensor.camera.rgb')
        for k,v in(('image_size_x',W),('image_size_y',H),('fov',a.fov)):bp.set_attribute(k,str(v))
        cam=world.spawn_actor(bp,carla.Transform(carla.Location(x=a.cam_x,y=a.cam_y,z=a.cam_z),carla.Rotation(pitch=a.cam_pitch)),attach_to=veh);actors.append(cam)
        images=[];cam.listen(images.append)
        veh.set_autopilot(True,tm.get_port());tm.ignore_lights_percentage(veh,100)  # keep moving: red lights would add long identical stretches
        for _ in range(40):world.tick()  # let the car settle on the road
        ticks=int(a.seconds/.05);driving=carla.LaneType.Driving;bands=[(0,15),(15,30),(30,50)];shown=not a.no_window
        for t in range(ticks):
            world.tick()
            for _ in range(200):
                if images:break
                time.sleep(.01)
            if not images:raise RuntimeError('no image received from the CARLA camera')
            im=images[-1];images.clear()
            if t%a.every:continue
            frame=np.frombuffer(im.raw_data,np.uint8).reshape(im.height,im.width,4)[:,:,:3].copy()
            t0=time.perf_counter();r=rp(frame);ms=1000*(time.perf_counter()-t0);lc=r['lane_center']
            cam_tf=im.transform;off_m,off_n,lw,junction=true_offset(world_map,cam_tf.location,driving)
            refs=visible_crossings(polys,cam_tf.get_inverse_matrix(),W,H,a.fov);used,fp=match(r['crosswalks'],refs)
            band={f'{lo}-{hi}m':[sum(lo<=x['dist']<hi for x in refs),sum(lo<=refs[i]['dist']<hi for i in used)] for lo,hi in bands}
            rows.append({'frame':len(rows),'sim_time':round(t*.05,2),'mode':lc['mode'],'offset_norm':lc['offset_norm'],'true_offset_norm':off_n,'true_offset_m':off_m,'lane_width_m':lw,'is_junction':junction,
                         'heading_deg':lc['heading_deg'],'low_confidence':lc.get('low_confidence'),'n_crossings':len(r['crosswalks']),'map_crossings_in_view':len(refs),'unmatched_detections':fp,'refs_band':band,'ms':ms})
            vis=rp.draw(frame,r)
            for x in refs:
                x1,y1,x2,y2=map(int,x['xyxy']);cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,255),1)
            true_txt=f"CARLA true offset {off_n:+.2f} lane widths{'  junction' if junction else ''}   map crossings in view: {len(refs)}"
            cv2.putText(vis,true_txt,(15,70),0,.8,(0,255,255),2)
            if vw is None:vw=cv2.VideoWriter(str(a.out/'overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),1/(a.every*.05),(W,H))
            vw.write(vis)
            if len(rows)%40==1:cv2.imwrite(str(a.out/'snapshots'/f'{len(rows)-1:05d}.jpg'),vis)
            if len(rows)%20==1:print(f"t={t*.05:6.1f}s  model offset {lc['offset_norm'] if lc['offset_norm'] is None else round(lc['offset_norm'],3)}  true {off_n:+.3f}  mode {lc['mode']}  crossings {len(r['crosswalks'])} (map {len(refs)})  {ms:.0f} ms")
            if shown:
                cv2.imshow('Joint road model - live CARLA (q to stop)',vis)
                if cv2.waitKey(1)&0xFF==ord('q'):break
    finally:
        for act in reversed(actors):
            if hasattr(act,'stop') and act.is_alive:
                try:act.stop()
                except Exception:pass
            act.destroy()
        tm.set_synchronous_mode(False);world.apply_settings(original)
        if vw:vw.release()
        cv2.destroyAllWindows()
    calib=json.loads((HERE/'lane_center_calibration.json').read_text())['k']
    meta={'town':world_map.name,'weather':a.weather,'seconds':a.seconds,'every':a.every,'spawn':a.spawn,'device':device,'camera':{'fov':a.fov,'x':a.cam_x,'y':a.cam_y,'z':a.cam_z,'pitch':a.cam_pitch},
          'carla_server':server,'calibration_k':calib,'ignore_traffic_lights':True,'npc_traffic':False}
    res=summarise(rows,a,meta)
    fields=[k for k in rows[0] if k!='refs_band'] if rows else []
    with open(a.out/'frames.csv','w',newline='') as fh:
        w=csv.DictWriter(fh,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    (a.out/'summary.json').write_text(json.dumps(res,indent=1));print(json.dumps(res,indent=1));print('wrote',a.out)
    return res

if __name__=='__main__':
    args=parse_args()
    try:import carla
    except ImportError:sys.exit('The carla package is not installed. Run setup_windows.bat with your CARLA version, e.g. setup_windows.bat 0.9.16')
    run(args,carla)
