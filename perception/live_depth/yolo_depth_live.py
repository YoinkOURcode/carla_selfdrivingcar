#!/usr/bin/env python
"""
yolo_depth_live.py

Drives a car on autopilot in CARLA with an RGB camera and a depth camera mounted
at the same spot. Every frame, YOLO detects objects in the RGB image and the depth
camera gives the distance to each detected object, shown live in a window.
Detections are labelled with the team's classes from classes.txt.

Camera settings match the team's generate_dataset.py / config.json:
    1280x720, FOV 90, mounted at x=1.5 m, z=2.0 m, synchronous mode at 0.05 s per tick.

Usage (CarlaUE4 must already be running, classes.txt in the same folder):
    py -3.12 yolo_depth_live.py                                  (pretrained COCO model)
    py -3.12 yolo_depth_live.py --model team_model\\best.pt      (the team's trained model)

With the team model, put its model_metadata.json next to best.pt: the script then uses
the team's settings (image size 1280, confidence 0.25) automatically, and merges the
duplicate boxes the model tends to draw, using the rules from the team's perception/dedupe.py.

Controls in the window:
    d       switch between camera view and depth view
    q / Esc quit (cleans up all spawned actors and restores CARLA's settings)

With the pretrained COCO model (yolo26n.pt), COCO detections are translated into
the team's classes:
    car, truck, bus, motorcycle, bicycle -> other_vehicle      (edit COCO_TO_VEHICLE)
    traffic light -> red/yellow/green_light by checking which colour is glowing
    person -> adult/child_pedestrian by estimated real height (from depth)
    speed_limit cannot be detected: COCO has no speed-limit class.
With the team's own model, its class names are used directly.

For every detected object the window shows:
    distance  straight-line distance from the camera (not from the car's bumper)
    angle     direction from straight ahead, e.g. "15 deg L" (left) or "8 deg R" (right)
    speed     vehicles and pedestrians only: the object's own speed over the ground in km/h.
              Each object is followed from frame to frame and its position turned into world
              coordinates with the camera's pose, so our own car's movement cancels out.
              Appears after about 0.25 s of tracking.
Speed limit signs (team model only) are read from the image: the round plate is found,
the digits are read with OCR, and a value is accepted after 3 matching readings that
clearly outvote any other reading. When the car passes a sign whose value was accepted,
that limit applies: the car's top speed is set to it (CARLA's autopilot still slows for
turns on its own). Before any sign is passed, the car follows CARLA's map limit as usual.
Use --no-obey to only display the limits. The top left shows the limit in force, the
next sign ahead, and our own speed.
Needs:  py -3.12 -m pip install rapidocr_onnxruntime==1.2.3
(version 1.2.3 reads small signs best; newer versions also work but miss more signs)

--log detections.csv saves every detection per frame: track id, class, distance, angle,
position (forward/right/up from the camera), speed, closing speed (negative = getting
closer), direction of motion relative to our car, and the speed limit read.
"""

import argparse
import json
import math
import queue
import random
import time
from pathlib import Path

import carla
import cv2
import numpy as np
from ultralytics import YOLO

# --- Camera settings copied from the team's config.json ---
IMAGE_W = 1280
IMAGE_H = 720
FOV = 90
CAMERA_X = 1.5
CAMERA_Z = 2.0
FIXED_DELTA_SECONDS = 0.05
# -----------------------------------------------------------

# Focal length in pixels, from resolution and FOV (needed to turn pixels into meters)
FOCAL = (IMAGE_W / 2.0) / math.tan(math.radians(FOV / 2.0))
CX, CY = IMAGE_W / 2.0, IMAGE_H / 2.0

# --- Pretrained COCO model -> team classes ------------------------------------------
# Which COCO classes count as "other_vehicle". Check with your team whether
# motorcycles and bicycles should be included, and remove them here if not.
COCO_TO_VEHICLE = {"car", "truck", "bus", "motorcycle", "bicycle"}
# A detected person shorter than this (meters, estimated from depth) is labelled a child.
CHILD_MAX_HEIGHT_M = 1.4
# --------------------------------------------------------------------------------------

UNSURE_COLOR = (170, 170, 170)
CLASS_COLORS = {  # BGR
    "other_vehicle": (0, 200, 255),
    "red_light": (40, 40, 255),
    "yellow_light": (0, 230, 255),
    "green_light": (60, 220, 60),
    "adult_pedestrian": (255, 160, 60),
    "child_pedestrian": (255, 80, 220),
    "speed_limit": (255, 255, 255),
}


def load_classes(path):
    """Read classes.txt in the team's 'id:name' format."""
    classes = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            idx, name = line.split(":", 1)
            classes[int(idx.strip())] = name.strip()
        else:
            classes[len(classes)] = line
    return classes


def to_bgr(image):
    """CARLA camera image (BGRA bytes) -> OpenCV BGR array."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    return arr[:, :, :3].copy()


def to_depth_meters(image):
    """CARLA depth image -> per-pixel depth in meters (CARLA's documented decoding)."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    b = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    r = arr[:, :, 2].astype(np.float32)
    normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
    return normalized * 1000.0


def box_depth(depth_m, x1, y1, x2, y2):
    """Median depth of the central half of a box, so background or road at the
    edges of the box doesn't skew the result."""
    h, w = depth_m.shape
    bw, bh = x2 - x1, y2 - y1
    cx1 = int(max(0, x1 + bw * 0.25))
    cx2 = int(min(w, x2 - bw * 0.25))
    cy1 = int(max(0, y1 + bh * 0.25))
    cy2 = int(min(h, y2 - bh * 0.25))
    if cx2 <= cx1:
        cx2 = min(w, cx1 + 1)
    if cy2 <= cy1:
        cy2 = min(h, cy1 + 1)
    region = depth_m[cy1:cy2, cx1:cx2]
    if region.size == 0:
        return None
    return float(np.median(region))


def straight_line_distance(z, u, v):
    """CARLA depth is measured along the camera's forward axis. Convert it to the
    straight-line distance to the point at pixel (u, v)."""
    x = (u - CX) * z / FOCAL
    y = (v - CY) * z / FOCAL
    return math.sqrt(x * x + y * y + z * z)


def light_color(bgr_crop, min_contrast=70):
    """Which lamp is glowing in a traffic light crop: 'red', 'yellow', 'green' or None.
    A lit lamp is a bright spot on a darker face. CARLA's housings are painted yellow,
    so an evenly bright crop (sunlit paint, seen from the side or back) gives None."""
    if bgr_crop.size == 0 or min(bgr_crop.shape[:2]) < 4:
        return None
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(int)
    s = hsv[..., 1].astype(int)
    v = hsv[..., 2].astype(int)
    v99, vmed = np.percentile(v, 99), np.median(v)
    if v99 < 150 or v99 - vmed < min_contrast:
        return None
    mask = (v >= max(150, np.percentile(v, 95))) & (s >= 60)
    if mask.sum() < 3:
        return None
    hh = h[mask]
    red = int(((hh <= 8) | (hh >= 165)).sum())
    yellow = int(((hh > 14) & (hh <= 35)).sum())
    green = int(((hh > 40) & (hh <= 95)).sum())
    total = red + yellow + green
    if total == 0:
        return None
    count, name = max((red, "red"), (yellow, "yellow"), (green, "green"))
    return name if count >= 0.5 * total else None


def pedestrian_class(y1, y2, z):
    """Estimate a person's real height from box height and depth, then decide adult/child.
    Returns (class name or None, height in m or None). None if the person is cut off."""
    if y1 <= 2 or y2 >= IMAGE_H - 2 or z is None:
        return None, None
    height_m = (y2 - y1) * z / FOCAL
    return ("child_pedestrian" if height_m < CHILD_MAX_HEIGHT_M else "adult_pedestrian"), height_m


# --- Duplicate-box merging, copied unchanged from the team's perception/dedupe.py ---
# The team model often draws two boxes on one object; the team README says to run this after detection.
LIGHTS = {1, 2, 3}
PEDESTRIANS = {4, 5}
SPEED_LIMIT = 6


def is_duplicate(big_cls, small_cls, iou, contained):
    # A light or sign cannot sit inside another, but a far car or a child can sit inside a larger box
    if big_cls in LIGHTS and small_cls in LIGHTS:
        return contained >= 0.7
    if big_cls == SPEED_LIMIT and small_cls == SPEED_LIMIT:
        return contained >= 0.7
    if big_cls in PEDESTRIANS and small_cls in PEDESTRIANS:
        return iou >= 0.5
    return big_cls == small_cls and contained >= 0.9 and iou >= 0.3


def merge_duplicates(rows):
    # rows: x1, y1, x2, y2, confidence, class. Keeps the larger box and the more confident class and score
    rows = np.asarray(rows, dtype=float).reshape(-1, 6)
    areas = (rows[:, 2] - rows[:, 0]) * (rows[:, 3] - rows[:, 1])
    kept = []
    for row, area in zip(rows[np.argsort(-areas)], areas[np.argsort(-areas)]):
        for other in kept:
            box = other[0]
            width = min(box[2], row[2]) - max(box[0], row[0])
            height = min(box[3], row[3]) - max(box[1], row[1])
            overlap = max(width, 0) * max(height, 0)
            if area > 0 and is_duplicate(int(other[1]), int(row[5]), overlap / (other[2] + area - overlap), overlap / area):
                if row[4] > box[4]:
                    box[4], box[5] = row[4], row[5]
                other[3] += 1
                break
        else:
            kept.append([row.copy(), int(row[5]), area, 0])
    return np.array([k[0] for k in kept]).reshape(-1, 6), sum(k[3] for k in kept)
# -------------------------------------------------------------------------------------


# --- Reading the number on a speed limit sign ----------------------------------------
# CARLA's speed limit signs: white round plate, red ring, dark digits.
VALID_LIMITS = {10, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130}  # values used by CARLA's sign assets
SIGN_MIN_HEIGHT_PX = 12   # smaller signs are not read at all
SIGN_READ_INTERVAL_S = 0.1  # read each sign at most this often (simulation time)
SIGN_MIN_VOTES = 3        # a value must be read this many times, and at least twice as often as any other
SIGN_PASS_DISTANCE_M = 15.0  # a confirmed sign last seen this close, then out of view, counts as passed


def find_sign_disc(crop):
    """(x, y, w, h) of the round red-ringed plate inside the crop, or None."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = (((hsv[..., 0] <= 10) | (hsv[..., 0] >= 165)) & (hsv[..., 1] >= 80) & (hsv[..., 2] >= 60)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(red, 8)
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < 6 or h < 6 or not 0.6 < w / h < 1.6:
            continue
        if area / (w * h) > 0.8:  # solid red, not a ring
            continue
        if best is None or w * h > best[2] * best[3]:
            best = (x, y, w, h)
    return best


def _clean_sign(core, circle):
    """Enlarge, whiten the red ring, optionally blank outside the inner circle, stretch contrast."""
    scale = 80.0 / core.shape[0]
    core = cv2.resize(core, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)
    # Only bright, saturated red counts as ring: dark digit pixels have unreliable saturation
    red = ((hsv[..., 0] <= 12) | (hsv[..., 0] >= 160)) & (hsv[..., 1] > 70) & (hsv[..., 2] > 80)
    gray[red] = 255
    if circle:  # blank the ring area, background and pole outside the white inner circle
        ch, cw = gray.shape
        yy, xx = np.ogrid[:ch, :cw]
        gray[((xx - cw / 2) / (cw * 0.40)) ** 2 + ((yy - ch / 2) / (ch * 0.40)) ** 2 > 1] = 255
    lo, hi = np.percentile(gray, 3), np.percentile(gray, 97)  # small signs have faint grey digits
    if hi - lo > 10:
        gray = np.clip((gray.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.copyMakeBorder(out, 12, 12, 16, 16, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def sign_cores(crop):
    """Two cut-outs of the sign's centre for reading: circle-masked and inset rectangle."""
    disc = find_sign_disc(crop)
    if disc is None:  # no ring found: assume the plate is the top square of the box (box may include the pole)
        h, w = crop.shape[:2]
        disc = (0, 0, w, min(w, h))
    x, y, w, h = disc
    full = crop[y:y + h, x:x + w]
    mx, my = int(round(w * 0.14)), int(round(h * 0.14))
    inset = crop[y + my:y + h - my, x + mx:x + w - mx]
    return [_clean_sign(c, circle) for c, circle in ((full, True), (inset, False))
            if c.size and min(c.shape[:2]) >= 4]


def parse_limit(text):
    digits = "".join(ch for ch in text.upper().replace("O", "0").replace("D", "0") if ch.isdigit())
    if not digits:
        return None
    value = int(digits)
    return value if value in VALID_LIMITS else None


def read_speed_limit(ocr, crop):
    """Read one sign crop. Returns (value or None, score, raw text)."""
    cores = sign_cores(crop)
    if not cores:
        return None, 0.0, ""
    # Call the recognizer directly: the library's own 0.5 score cut-off drops most small signs.
    # Trust comes from the valid-limits check and from voting over several frames instead.
    # The recognizer's name differs between versions of the OCR package (1.2: text_recognizer, 1.3+: text_rec)
    recognize = getattr(ocr, "text_rec", None) or getattr(ocr, "text_recognizer")
    results = recognize(cores)[0]
    best = (None, 0.0, "")
    for item in results:
        text, score = item[0], item[1]
        value = parse_limit(text)
        if value is not None and float(score) > best[1]:
            best = (value, float(score), text)
    return best


def make_sign_reader():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        print("Speed limit numbers will NOT be read: the OCR package is missing. Install it with:")
        print("    py -3.12 -m pip install rapidocr_onnxruntime==1.2.3")
        return None
    ocr = RapidOCR()
    if getattr(ocr, "text_rec", None) is None and getattr(ocr, "text_recognizer", None) is None:
        print("Speed limit numbers will NOT be read: this version of rapidocr_onnxruntime is not supported.")
        return None
    return ocr
# -------------------------------------------------------------------------------------


# --- Position, angle, tracking and velocity ------------------------------------------
GATE_M = {"vehicle": 3.0, "pedestrian": 1.5, "light": 2.0, "sign": 2.0}  # max jump between frames
TRACK_MAX_AGE_S = 0.5      # forget an object not seen for this long
VELOCITY_WINDOW_S = 0.6    # velocity is fitted over the last this-many seconds
VELOCITY_MIN_SAMPLES = 5
MOVING_GROUPS = {"vehicle", "pedestrian"}  # lights and signs don't move, so no speed is shown


def object_group(cls_name, raw_name):
    name = cls_name or raw_name or ""
    if name in ("other_vehicle",) or raw_name in COCO_TO_VEHICLE:
        return "vehicle"
    if name.endswith("_light") or raw_name == "traffic light":
        return "light"
    if name.endswith("pedestrian") or raw_name == "person":
        return "pedestrian"
    if name == "speed_limit":
        return "sign"
    return None


def camera_point(u, v, z):
    """Pixel (u, v) at depth z -> (forward, right, up) in meters relative to the camera.
    The camera faces straight ahead, so these are also the car's directions."""
    return z, (u - CX) * z / FOCAL, -(v - CY) * z / FOCAL


def angle_text(deg):
    if abs(deg) < 0.5:
        return "0 deg"
    return f"{abs(deg):.0f} deg {'R' if deg > 0 else 'L'}"


class Track:
    def __init__(self, tid, group, cls_name):
        self.id, self.group, self.cls_name = tid, group, cls_name
        self.history = []          # (t, world xyz, distance)
        self.votes = {}            # speed limit readings: value -> count
        self.last_read_t = -1e9

    def add(self, t, world, dist):
        self.history.append((t, np.asarray(world, dtype=float), dist))
        self.history = [h for h in self.history if t - h[0] <= max(VELOCITY_WINDOW_S, 1.0)]

    @property
    def last_t(self):
        return self.history[-1][0]

    def velocity(self):
        """World velocity (m/s, xyz) and closing speed (m/s, negative = getting closer), or (None, None)."""
        t_now = self.history[-1][0]
        recent = [h for h in self.history if t_now - h[0] <= VELOCITY_WINDOW_S]
        if len(recent) < VELOCITY_MIN_SAMPLES or recent[-1][0] - recent[0][0] < 0.2:
            return None, None
        ts = np.array([h[0] for h in recent])
        ts = ts - ts.mean()
        denom = float((ts ** 2).sum())
        pos = np.array([h[1] for h in recent])
        vel = (ts[:, None] * (pos - pos.mean(axis=0))).sum(axis=0) / denom
        dists = np.array([h[2] for h in recent])
        closing = float((ts * (dists - dists.mean())).sum() / denom)
        return vel, closing

    def predicted(self, t):
        vel, _ = self.velocity()
        last = self.history[-1][1]
        return last if vel is None else last + vel * (t - self.last_t)

    def confirmed_limit(self):
        if not self.votes:
            return None
        ranked = sorted(self.votes.items(), key=lambda kv: -kv[1])
        top_value, top_count = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0
        return top_value if top_count >= SIGN_MIN_VOTES and top_count >= 2 * second else None


class Tracker:
    """Follows each object across frames by matching world positions within the same kind of object."""

    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def update(self, dets, t):
        self.tracks = [tr for tr in self.tracks if t - tr.last_t <= TRACK_MAX_AGE_S]
        pairs = []
        for di, d in enumerate(dets):
            if d["world"] is None:
                continue
            for ti, tr in enumerate(self.tracks):
                if tr.group != d["group"]:
                    continue
                cost = float(np.linalg.norm(tr.predicted(t)[:2] - d["world"][:2]))
                if cost <= GATE_M[d["group"]]:
                    pairs.append((cost, di, ti))
        used_d, used_t = set(), set()
        for cost, di, ti in sorted(pairs):
            if di in used_d or ti in used_t:
                continue
            used_d.add(di)
            used_t.add(ti)
            dets[di]["track"] = self.tracks[ti]
        for di, d in enumerate(dets):
            if d["world"] is None:
                d["track"] = None
                continue
            if di not in used_d:
                tr = Track(self.next_id, d["group"], d["cls_name"])
                self.next_id += 1
                self.tracks.append(tr)
                d["track"] = tr
            d["track"].cls_name = d["cls_name"]
            d["track"].add(t, d["world"], d["dist"])
# -------------------------------------------------------------------------------------


def wait_for_frame(q, frame, timeout=10.0):
    """Return the sensor image that belongs to this exact simulation frame."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"Timed out waiting for sensor frame {frame}")
        image = q.get(timeout=remaining)
        if image.frame == frame:
            return image
        if image.frame > frame:
            raise RuntimeError("Sensor is ahead of the world tick. Is another script ticking CARLA?")


def overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def draw_box(canvas, x1, y1, x2, y2, label, color, placed):
    """Draw a box and its label; move the label below the box if it would cover another label."""
    p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
    cv2.rectangle(canvas, p1, p2, color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(th + 4, p1[1])
    rect = (p1[0], top - th - 4, p1[0] + tw + 4, top)
    if any(overlaps(rect, r) for r in placed):
        top = min(IMAGE_H - 1, p2[1] + th + 4)
        rect = (p1[0], top - th - 4, p1[0] + tw + 4, top)
    placed.append(rect)
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), color, -1)
    cv2.putText(canvas, label, (rect[0] + 2, rect[3] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def spawn_walkers(world, library, count):
    """Spawn pedestrians that walk around, so adult/child detection can be tested."""
    walker_bps = list(library.filter("walker.pedestrian.*"))
    walkers, controllers = [], []
    controller_bp = library.find("controller.ai.walker")
    for _ in range(count * 3):
        if len(walkers) >= count:
            break
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        walker = world.try_spawn_actor(random.choice(walker_bps), carla.Transform(loc))
        if walker is None:
            continue
        controller = world.try_spawn_actor(controller_bp, carla.Transform(), walker)
        if controller is None:
            walker.destroy()
            continue
        walkers.append(walker)
        controllers.append(controller)
    return walkers, controllers


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="YOLO weights: the team's best.pt, or yolo26n.pt (pretrained). Default: the team model at "
                             "../carla_yolo26_demo_results/mac_demo/best.pt if this script is in the team repository, "
                             "otherwise yolo26n.pt")
    parser.add_argument("--classes", default=str(script_dir / "classes.txt"), help="Team classes file (id:name per line)")
    parser.add_argument("--conf", type=float, default=None,
                        help="Minimum detection confidence to show. Default: the model's model_metadata.json "
                             "if present (0.25 for the team model), otherwise 0.4")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Size YOLO resizes each frame to. Default: the model's model_metadata.json if present "
                             "(1280 for the team model), otherwise the size the model was trained at, otherwise 640")
    parser.add_argument("--no-merge", action="store_true",
                        help="Team model only: don't merge duplicate boxes (the team README recommends merging)")
    parser.add_argument("--vehicles", type=int, default=30, help="Number of traffic vehicles to spawn")
    parser.add_argument("--walkers", type=int, default=50, help="Number of pedestrians to spawn")
    parser.add_argument("--ego", default="vehicle.tesla.model3", help="Blueprint of the car carrying the cameras")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--log", help="Save every detection (distance, angle, speed, speed limit) to this CSV file")
    parser.add_argument("--no-obey", action="store_true",
                        help="Only display the speed limits read from signs; don't make the car follow them")
    args = parser.parse_args()
    if args.model is None:
        repo_model = script_dir.parent / "carla_yolo26_demo_results" / "mac_demo" / "best.pt"
        args.model = str(repo_model) if repo_model.exists() else "yolo26n.pt"

    team_classes = load_classes(args.classes)
    print("Team classes:", team_classes)

    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)
    names = model.names
    coco_mode = "traffic light" in names.values() and "car" in names.values()
    if coco_mode:
        print("Pretrained COCO model: translating detections into team classes. "
              "speed_limit cannot be detected until the team's own model is used.")
    else:
        model_names = {int(k): v for k, v in names.items()}
        if model_names != team_classes:
            print("WARNING: the model's classes do not match classes.txt!")
            print("  model:      ", model_names)
            print("  classes.txt:", team_classes)
            print("  Labels shown come from the model. Check the class order with your team.")
    # Settings: command line first, then the model's model_metadata.json, then what it was trained at
    metadata = {}
    meta_path = Path(args.model).resolve().parent / "model_metadata.json"
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            print(f"Using settings from {meta_path}")
        except (OSError, ValueError):
            print(f"Could not read {meta_path}, ignoring it")
    trained_imgsz = None
    try:
        trained_imgsz = int((getattr(model, "ckpt", None) or {}).get("train_args", {}).get("imgsz"))
    except (TypeError, ValueError):
        pass
    imgsz = args.imgsz or metadata.get("imgsz") or trained_imgsz or 640
    conf = args.conf if args.conf is not None else metadata.get("display_confidence", 0.4)
    # The duplicate-merging rules use the team's class ids, so only apply them to a model with the team's classes
    merge = (not coco_mode and not args.no_merge
             and {int(k): v for k, v in names.items()} == team_classes)

    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = 0
        print("Running YOLO on:", f"GPU ({torch.cuda.get_device_name(0)})" if device == 0 else "CPU")
    except Exception:
        pass
    print(f"Image size {imgsz}, confidence {conf}, duplicate merging {'on' if merge else 'off'}")

    sign_reader = None
    if "speed_limit" in names.values():
        sign_reader = make_sign_reader()
        if sign_reader is not None:
            print("Speed limit numbers: read from the camera image")
    tracker = Tracker()
    obey = sign_reader is not None and not args.no_obey
    if obey:
        print("The car will follow the speed limits read from signs (after passing each sign). "
              "Use --no-obey to only display them.")
    applied_limit = None  # speed limit of the last sign we passed (km/h)
    signs_ahead = {}      # track id -> [confirmed value, last distance in m] for signs not yet passed

    log_file = log_writer = None
    if args.log:
        import csv
        log_file = open(args.log, "w", newline="", encoding="utf-8")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["frame", "time_s", "track_id", "class", "confidence", "distance_m", "angle_deg",
                             "forward_m", "right_m", "up_m", "speed_kmh", "closing_speed_mps",
                             "motion_dir_deg", "speed_limit_kmh", "applied_limit_kmh"])
        print(f"Logging detections to {Path(args.log).resolve()}")

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    original_settings = world.get_settings()
    tm = client.get_trafficmanager(args.tm_port)

    actors, sensors, walkers, controllers = [], [], [], []
    try:
        # Synchronous mode: the script advances the simulation one step at a time,
        # so the RGB and depth images always come from the same moment.
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        world.apply_settings(settings)
        tm.set_synchronous_mode(True)

        library = world.get_blueprint_library()
        spawn_points = list(world.get_map().get_spawn_points())
        random.shuffle(spawn_points)

        ego_bp = library.find(args.ego)
        ego_bp.set_attribute("role_name", "hero")
        ego = None
        used = 0
        for i, transform in enumerate(spawn_points):
            ego = world.try_spawn_actor(ego_bp, transform)
            if ego is not None:
                used = i + 1
                break
        if ego is None:
            raise RuntimeError("Could not spawn the camera car")
        actors.append(ego)
        ego.set_autopilot(True, tm.get_port())

        vehicle_bps = list(library.filter("vehicle.*"))
        spawned = 0
        for transform in spawn_points[used:]:
            if spawned >= args.vehicles:
                break
            vehicle = world.try_spawn_actor(random.choice(vehicle_bps), transform)
            if vehicle is not None:
                vehicle.set_autopilot(True, tm.get_port())
                actors.append(vehicle)
                spawned += 1

        walkers, controllers = spawn_walkers(world, library, args.walkers)
        world.tick()
        for controller in controllers:
            controller.start()
            target = world.get_random_location_from_navigation()
            if target is not None:
                controller.go_to_location(target)
            controller.set_max_speed(random.uniform(0.8, 1.8))
        print(f"Spawned camera car ({ego.type_id}), {spawned} traffic vehicles and {len(walkers)} pedestrians.")

        # RGB and depth cameras with identical settings at the identical spot
        cam_transform = carla.Transform(carla.Location(x=CAMERA_X, z=CAMERA_Z))
        queues = {}
        for name, bp_id in (("rgb", "sensor.camera.rgb"), ("depth", "sensor.camera.depth")):
            bp = library.find(bp_id)
            bp.set_attribute("image_size_x", str(IMAGE_W))
            bp.set_attribute("image_size_y", str(IMAGE_H))
            bp.set_attribute("fov", str(FOV))
            bp.set_attribute("sensor_tick", "0")
            sensor = world.spawn_actor(bp, cam_transform, attach_to=ego)
            q = queue.Queue()
            sensor.listen(q.put)
            queues[name] = q
            sensors.append(sensor)

        window = "YOLO + depth (d: toggle depth view, q: quit)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, IMAGE_W, IMAGE_H)
        show_depth = False
        last = time.monotonic()
        spectator = world.get_spectator()

        while True:
            frame = world.tick()
            rgb_img = wait_for_frame(queues["rgb"], frame)
            depth_img = wait_for_frame(queues["depth"], frame)

            bgr = to_bgr(rgb_img)
            depth_m = to_depth_meters(depth_img)

            # Keep the CarlaUE4 window looking at the camera car from behind
            t = ego.get_transform()
            back = t.get_forward_vector() * -8.0
            spectator.set_transform(carla.Transform(
                t.location + carla.Location(x=back.x, y=back.y, z=4.0),
                carla.Rotation(pitch=-15, yaw=t.rotation.yaw)))

            result = model.predict(bgr, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
            canvas = depth_view(depth_m) if show_depth else bgr.copy()
            placed, shown = [], 0

            rows = result.boxes.data.cpu().numpy()  # x1, y1, x2, y2, confidence, class id
            if merge:
                rows, _ = merge_duplicates(rows)

            # Camera pose at this exact frame: turns camera-relative positions into world positions,
            # so our own car's movement cancels out of the objects' velocities
            t_sim = rgb_img.timestamp
            cam_to_world = np.array(rgb_img.transform.get_matrix())
            cam_rot = cam_to_world[:3, :3]

            dets = []
            for x1, y1, x2, y2, score, cls_id in rows.tolist():
                raw_name = names[int(cls_id)]
                z = box_depth(depth_m, x1, y1, x2, y2)
                extra = ""

                if coco_mode:
                    if raw_name in COCO_TO_VEHICLE:
                        cls_name = "other_vehicle"
                    elif raw_name == "traffic light":
                        color = light_color(bgr[int(y1):int(y2), int(x1):int(x2)])
                        cls_name = f"{color}_light" if color else None
                        if cls_name is None:
                            extra = "traffic light, colour unclear"
                    elif raw_name == "person":
                        cls_name, height = pedestrian_class(y1, y2, z)
                        if cls_name is None:
                            extra = "pedestrian, cut off"
                        else:
                            extra = f"~{height:.1f} m tall"
                    else:
                        continue  # not one of the team's classes
                else:
                    cls_name = raw_name

                group = object_group(cls_name, raw_name)
                det = {"box": (x1, y1, x2, y2), "score": score, "cls_name": cls_name, "extra": extra,
                       "group": group, "dist": None, "angle": None, "rel": None, "world": None}
                if z is not None and z < 900:  # 1000 m means sky / nothing measured
                    u, v = (x1 + x2) / 2, (y1 + y2) / 2
                    fwd, right, up = camera_point(u, v, z)
                    det["rel"] = (fwd, right, up)
                    det["dist"] = math.sqrt(fwd * fwd + right * right + up * up)
                    det["angle"] = math.degrees(math.atan2(right, fwd))
                    if group is not None:
                        det["world"] = (cam_to_world @ np.array([fwd, right, up, 1.0]))[:3]
                dets.append(det)

            tracker.update(dets, t_sim)

            for det in dets:
                x1, y1, x2, y2 = det["box"]
                tr = det.get("track")
                speed_kmh = closing = motion_dir = None
                if tr is not None and det["group"] in MOVING_GROUPS:
                    vel, closing = tr.velocity()
                    if vel is not None:
                        speed_kmh = float(np.linalg.norm(vel[:2])) * 3.6
                        v_cam = cam_rot.T @ vel  # velocity in the car's directions (forward, right, up)
                        if speed_kmh > 1.0:
                            motion_dir = math.degrees(math.atan2(v_cam[1], v_cam[0]))

                limit = None
                if det["group"] == "sign" and tr is not None:
                    if (sign_reader is not None and y2 - y1 >= SIGN_MIN_HEIGHT_PX
                            and t_sim - tr.last_read_t >= SIGN_READ_INTERVAL_S):
                        tr.last_read_t = t_sim
                        crop = bgr[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
                        if crop.size:
                            value, _, _ = read_speed_limit(sign_reader, crop)
                            if value is not None:
                                tr.votes[value] = tr.votes.get(value, 0) + 1
                    limit = tr.confirmed_limit()
                    if limit is not None:
                        entry = signs_ahead.setdefault(tr.id, [limit, det["dist"]])
                        entry[0] = limit
                        if det["dist"] is not None:
                            entry[1] = det["dist"]

                cls_name = det["cls_name"]
                label = cls_name if cls_name else det["extra"]
                if cls_name == "speed_limit":
                    label += f" {limit}" if limit is not None else (" ?" if sign_reader is not None else "")
                if cls_name and det["extra"]:
                    label += f" ({det['extra']})"
                label += f" {det['score']:.2f}"
                if det["dist"] is not None:
                    label += f" | {det['dist']:.1f} m | {angle_text(det['angle'])}"
                if speed_kmh is not None:
                    label += f" | {speed_kmh:.0f} km/h"
                draw_box(canvas, x1, y1, x2, y2, label, CLASS_COLORS.get(cls_name, UNSURE_COLOR), placed)
                shown += 1

                if log_writer is not None:
                    rel = det["rel"] or (None, None, None)
                    fmt = lambda x, n=2: "" if x is None else round(float(x), n)
                    log_writer.writerow([frame, round(t_sim, 3), tr.id if tr else "", cls_name or det["extra"],
                                         fmt(det["score"], 3), fmt(det["dist"]), fmt(det["angle"], 1),
                                         fmt(rel[0]), fmt(rel[1]), fmt(rel[2]), fmt(speed_kmh, 1),
                                         fmt(closing), fmt(motion_dir, 1), limit if limit is not None else "",
                                         applied_limit if applied_limit is not None else ""])

            # A confirmed sign that has left the view while close to us has been passed: its limit now applies
            alive = {t.id for t in tracker.tracks}
            for tid in list(signs_ahead):
                if tid in alive:
                    continue
                value, last_dist = signs_ahead.pop(tid)
                if last_dist is not None and last_dist <= SIGN_PASS_DISTANCE_M:
                    applied_limit = value
                    if obey:
                        tm.set_desired_speed(ego, float(value))
                    print(f"Passed a {value} km/h sign" + (f": car now limited to {value} km/h" if obey else ""))

            now = time.monotonic()
            fps = 1.0 / max(now - last, 1e-6)
            last = now
            vel = ego.get_velocity()
            our_speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2) * 3.6
            hud = [f"{fps:.1f} FPS  |  {shown} detections  |  distance from camera"]
            if sign_reader is not None:
                if applied_limit is not None:
                    hud.append(f"Speed limit: {applied_limit} km/h (from last sign passed)"
                               + (" - car obeys" if obey else ""))
                else:
                    hud.append("Speed limit: no sign passed yet - car follows CARLA's map limit")
                if signs_ahead:
                    value, dist = min(signs_ahead.values(), key=lambda e: e[1] if e[1] is not None else 1e9)
                    hud.append(f"Sign ahead: {value} km/h" + (f" in {dist:.0f} m" if dist is not None else ""))
            hud.append(f"Our speed: {our_speed:.0f} km/h"
                       + (f" (max {applied_limit})" if obey and applied_limit is not None else ""))
            for i, line in enumerate(hud):
                cv2.putText(canvas, line, (10, 25 + 27 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                show_depth = not show_depth
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("Cleaning up...")
        for sensor in sensors:
            try:
                sensor.stop()
            except RuntimeError:
                pass
        for controller in controllers:
            try:
                controller.stop()
            except RuntimeError:
                pass
        try:
            to_destroy = sensors + controllers + walkers + actors
            client.apply_batch_sync([carla.command.DestroyActor(a.id) for a in to_destroy], False)
        except RuntimeError as exc:
            print(f"Cleanup warning: {exc}")
        try:
            tm.set_synchronous_mode(False)
        except RuntimeError:
            pass
        try:
            world.apply_settings(original_settings)
        except RuntimeError as exc:
            print(f"Could not restore CARLA settings ({exc}). Restart CarlaUE4 if the world looks frozen.")
        cv2.destroyAllWindows()
        if log_file is not None:
            log_file.close()
            print(f"Detections saved to {Path(args.log).resolve()}")
        print("Done.")


def depth_view(depth_m, max_m=100.0):
    """Grayscale picture of the depth: near = bright, far = dark."""
    vis = 1.0 - np.clip(depth_m / max_m, 0.0, 1.0)
    vis = (vis * 255).astype(np.uint8)
    return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    main()
