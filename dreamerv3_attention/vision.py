"""Frozen RGB models, the aligned depth camera and OCR for external observations.

No simulator actors, signal states or weather are consulted. The camera decides
light colour; the stop position comes from navigation: signalised stop lines on
the planned route (map geometry, as a satnav knows junctions) plus static vehicle
geometry. A red/yellow with no route stop on the planned path is not for this
route. Without route data the range is unknown. Near and inside route junctions,
where ego-lane paint is absent, lane error and heading come from the planned
path. The path also supports low-speed recovery after a lane estimate expires.
Speed numbers use OCR.
"""

import math
import os
from pathlib import Path

import numpy as np

from .object_tracker import ObjectTracker
from .sensors import PerceptionPrediction, associate_depth
from .stop_lines import project_to_path
from .traffic_signals import SignalTracker
from .speed_sign import SIGN_MIN_HEIGHT_PX, SpeedLimitTracker, make_sign_reader, read_speed_limit

BUMPER_ALLOWANCE_M = 2.5  # same fixed ego-origin-to-bumper allowance as crossings
# Wide enough to recover a car that drifted while vision was lost (experiment_28
# eval2 deadlocks: 1.3-1.8 m, 31 deg), narrow enough to reject a mis-projected route.
ROUTE_LANE_FALLBACK_MAX_OFFSET_M = 2.0
ROUTE_LANE_FALLBACK_MAX_HEADING_RAD = math.radians(40.)
# On Town02's two-lane roads vision can lock onto the neighbouring lane's paint and
# report an offset 3-6 m off (experiment_28 eval4/7/10, same spots every run). Correct
# readings stay within 0.75 m (p95), and the route offset within 0.14 m (p99).
VISION_LANE_MAX_DISAGREEMENT_M = 2.0
# At bend entry the straight-line lane model locks onto the wrong paint and a hold carries
# it on (Town02 88->61, 41->93: lane 0.6-1.0 m and heading sign wrong; held estimates in
# sharp bends 98% wrong). Where the aligned route (planned path + odometry) disagrees by
# more than this, it is the closer estimate on 96-100% of frames; fires on ~1% of straights.
# A depth point this far above or below the road plane is not road (e.g. a car over the
# extrapolated centre line), so that lane pixel falls back to the flat-road projection.
ROAD_HEIGHT_TOLERANCE_M = 0.5


def path_lane(path):
    """(lane_error_m, heading_error_rad) of the ego relative to a planned ego-frame path.

    Same signs as the vision fit: lane error is the ego's lateral offset from
    the centre line, heading error is minus the line's angle. None if unusable.
    """
    projection = project_to_path((0., 0.), path) if path is not None else None
    if projection is None:
        return None
    station, lateral = projection
    path = np.asarray(path, dtype=float)
    delta = np.diff(path, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    starts = np.r_[0., np.cumsum(lengths)[:-1]]
    usable = np.flatnonzero(lengths > 1e-8)
    index = usable[max(0, int(np.searchsorted(starts[usable], station, "right")) - 1)]
    return float(lateral), float(-math.atan2(delta[index, 1], delta[index, 0]))


LIGHT_CLASSES = (1, 2, 3)
RED_LIGHT_CLASS = 1
# Share of the smaller light box inside the larger one that marks both as the same head.
# Annotators boxed either the whole housing or only the lit lamp, so the detector often
# returns both on one head: 558 nested pairs on 500 held-out Round 2 test frames, all but
# one the same colour. Same rule as perception/dedupe.py, which merges them for display.
LIGHT_NESTED_CONTAINMENT = 0.7


def merge_nested_lights(boxes, confidences, classes, containment=LIGHT_NESTED_CONTAINMENT):
    """Collapse traffic-light boxes nested on one head into the larger box.

    The larger box is kept with the higher confidence. A conflicting colour pair
    resolves as the signal tracker does for competing heads: red if either is red,
    otherwise the more confident colour. Vehicles, pedestrians and signs pass
    through unchanged, so a child box inside an adult box is never dropped here.
    Returns filtered (boxes, confidences, classes) in the original order.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    confidences = np.asarray(confidences, dtype=float).reshape(-1).copy()
    classes = np.asarray(classes).astype(int).reshape(-1).copy()
    if not (len(boxes) == len(confidences) == len(classes)):
        raise ValueError("boxes, confidences and classes must have the same length")
    areas = np.clip(boxes[:, 2]-boxes[:, 0], 0., None) * np.clip(boxes[:, 3]-boxes[:, 1], 0., None)
    lights = [i for i in np.argsort(-areas, kind="stable")
              if classes[i] in LIGHT_CLASSES and np.isfinite(boxes[i]).all() and areas[i] > 0.]
    keep = np.ones(len(boxes), bool)
    kept = []
    for i in lights:
        for k in kept:
            width = min(boxes[k, 2], boxes[i, 2]) - max(boxes[k, 0], boxes[i, 0])
            height = min(boxes[k, 3], boxes[i, 3]) - max(boxes[k, 1], boxes[i, 1])
            if max(width, 0.) * max(height, 0.) / areas[i] >= containment:
                if RED_LIGHT_CLASS in (classes[k], classes[i]):
                    classes[k] = RED_LIGHT_CLASS
                elif confidences[i] > confidences[k]:
                    classes[k] = classes[i]
                confidences[k] = max(confidences[k], confidences[i])
                keep[i] = False
                break
        else:
            kept.append(i)
    return boxes[keep], confidences[keep], classes[keep]


def _env_float(name, default, low, high):
    value = float(os.environ.get(name, default))
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


class VisionPerceptionProvider:
    camera_only = True
    uses_depth_camera = True
    sign_reader = speed_limits = sign_debug = None

    def __init__(self):
        import torch
        from ultralytics import YOLO
        from joint_road_model.road_perception import RoadPerception

        root = Path(__file__).resolve().parents[1]
        self.device = os.environ.get("VISION_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.road = RoadPerception(root / "joint_road_model/joint_road_model.pt",
                                   root / "joint_road_model/lane_center_calibration.json", self.device)
        self.detector = YOLO(root / "perception/carla_yolo26_demo_results/mac_demo/best.pt")
        self.road.model.requires_grad_(False)
        self.detector.model.requires_grad_(False)
        self.detector.model.eval()
        expected = {0: "other_vehicle", 1: "red_light", 2: "yellow_light", 3: "green_light",
                    4: "adult_pedestrian", 5: "child_pedestrian", 6: "speed_limit"}
        if self.detector.names != expected:
            raise ValueError("Unexpected perception checkpoint class mapping")
        # Lane acceptance: maximum boundary extrapolation share and hold time for the last valid estimate.
        self.max_extrapolation = float(os.environ.get("VISION_MAX_EXTRAPOLATION", "0.7"))
        self.lane_hold_s = float(os.environ.get("VISION_LANE_HOLD_S", "2.0"))
        if not (0. < self.max_extrapolation <= 1. and 0. <= self.lane_hold_s <= 10.):
            raise ValueError("VISION_MAX_EXTRAPOLATION must be in (0, 1] and VISION_LANE_HOLD_S in [0, 10]")
        self._last_lane = None
        self.signal_tracker = SignalTracker(_env_float("VISION_SIGNAL_HOLD_S", "2.0", 0., 10.))
        self.object_tracker = ObjectTracker()
        self.sign_reader = make_sign_reader()
        self.speed_limits = SpeedLimitTracker()
        if self.sign_reader is None:
            print("VisionPerceptionProvider: rapidocr_onnxruntime missing, speed-limit numbers will not be read")

    def reset(self):
        self._last_lane = None
        self.lane_debug = self.sign_debug = None
        if getattr(self, "signal_tracker", None) is not None:
            self.signal_tracker.reset()
        if getattr(self, "speed_limits", None) is not None:
            self.speed_limits.reset()
        if getattr(self, "object_tracker", None) is not None:
            self.object_tracker.reset()

    @staticmethod
    def ground_point(u, v, calibration):
        """Intersect a pixel ray with a flat road, for a level front camera."""
        intrinsics = calibration["intrinsics_px"]
        extrinsics = calibration["extrinsics_ego"]
        if not np.allclose(extrinsics["rotation_pitch_yaw_roll_deg"], (0., 0., 0.)):
            raise ValueError("Vision ground projection currently requires a level front camera")
        location = np.asarray(extrinsics["location_m"], dtype=float)
        ray = np.array([1., (u-intrinsics["cx"])/intrinsics["fx"],
                        -(v-intrinsics["cy"])/intrinsics["fy"]])
        if not np.isfinite([*ray, *location]).all() or location[2] <= 0 or ray[2] >= -1e-4:
            return None
        point = location - location[2] / ray[2] * ray
        return point if 0. < point[0] <= 80. else None

    @staticmethod
    def depth_point(u, v, depth_m, calibration):
        """Ego-frame point of pixel (u, v) at CARLA depth (distance along the camera axis)."""
        intrinsics = calibration["intrinsics_px"]
        extrinsics = calibration["extrinsics_ego"]
        if not np.allclose(extrinsics["rotation_pitch_yaw_roll_deg"], (0., 0., 0.)):
            raise ValueError("Vision depth projection currently requires a level front camera")
        if not (math.isfinite(depth_m) and depth_m > 0.):
            return None
        location = np.asarray(extrinsics["location_m"], dtype=float)
        return location + depth_m * np.array([1., (u-intrinsics["cx"])/intrinsics["fx"],
                                              -(v-intrinsics["cy"])/intrinsics["fy"]])

    def lane_point(self, u, v, depth_image, calibration):
        """Ego-frame point of a lane-centre pixel: measured depth where the pixel shows road,
        otherwise the flat-road projection."""
        if depth_image is not None:
            col, row = int(round(u)), int(round(v))
            if 0 <= col < depth_image.shape[1] and 0 <= row < depth_image.shape[0]:
                patch = depth_image[max(0, row-2):row+3, max(0, col-2):col+3]
                values = patch[np.isfinite(patch) & (patch < 999.)]
                point = self.depth_point(u, v, float(np.median(values)), calibration) if values.size else None
                if point is not None and abs(point[2]) <= ROAD_HEIGHT_TOLERANCE_M:
                    return point
        return self.ground_point(u, v, calibration)

    def _signals(self, bundle, bgr, road, points, lights, calibration, depth_image):
        path = bundle.ego_path
        center = road.get("lane_center") or {}
        if path is None and len(points) >= 2:
            points = points[np.argsort(points[:, 0])]
            slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
            path = np.vstack(([0., intercept], points[:, :2]))
        lane_width = 3.5  # same nominal width used when only one lane edge is visible
        if depth_image is not None and center.get('left') and center.get('right'):
            v = int(.75*bgr.shape[0])
            edge_points = []
            for edge in (center['left'], center['right']):
                u = int(round(edge['a']*v+edge['b']))
                patch = depth_image[max(0,v-2):v+3, max(0,u-2):u+3] if 0 <= u < bgr.shape[1] else []
                values = np.asarray(patch)
                values = values[np.isfinite(values) & (values > 0.) & (values < 999.)]
                if len(values):
                    edge_points.append(self.depth_point(u, v, float(np.median(values)), calibration))
            if len(edge_points) == 2:
                width = float(np.linalg.norm(edge_points[1][:2]-edge_points[0][:2]))
                if 1.5 <= width <= 8.:
                    lane_width = width
        return self._track(bundle, lights, path, lane_width)

    def _track(self, bundle, lights, path, lane_width=3.5):
        if getattr(self, 'signal_tracker', None) is None:
            self.signal_tracker = SignalTracker()
        route = bundle.route_stop_lines
        lines = [dict(points_ego=np.asarray(line, float), confidence=1.) for line in route or ()]
        fields, signals = self.signal_tracker.update(float(bundle.timestamp), lines, lights, path,
                                                     bundle.front_bumper_m, bundle.ego_pose, lane_width)
        if route is not None:
            # Navigation knows every signalised stop on the route; without one on
            # the planned path ahead the lamp controls another road.
            if fields['red_light_distance_m'] is None:
                fields.update(red_light=False)
            if fields['yellow_light_distance_m'] is None:
                fields.update(yellow_light=False, yellow_light_confidence=0.)
        return fields, signals

    def predict(self, bundle):
        fields = dict(speed_limit_mps=None, speed_limit_confidence=0., lane_error_m=0.,
                      heading_error_rad=0., crossing_id=None, crossing_distance_m=None,
                      crossing_occupied=False, red_light=False, red_light_distance_m=None,
                      yellow_light=False, yellow_light_distance_m=None, yellow_light_confidence=0.,
                      visibility_m=30., observations_valid=False,
                      lane_source="unavailable", observation_invalid_reason=None)
        rows = []
        calibration = bundle.calibration.get("sensors", {}).get("rgb")
        if bundle.rgb is None or not bundle.valid.get("rgb", False) or calibration is None:
            fields["observation_invalid_reason"] = (
                "rgb_unavailable" if bundle.rgb is None or not bundle.valid.get("rgb", False)
                else "rgb_calibration_missing")
            signals = None
            if getattr(self, 'signal_tracker', None) is not None:
                held, signals = self._track(bundle, [], bundle.ego_path)
                fields.update(held)
            self.lane_debug = None
            return self._prediction(bundle, rows, fields, signals)
        depth_image = bundle.depth if bundle.valid.get("depth", bundle.depth is not None) else None
        if depth_image is not None and np.shape(depth_image) != bundle.rgb.shape[:2]:
            depth_image = None  # not pixel-aligned with the RGB frame
        # RoadPerception and YOLO's ndarray interface both accept BGR.
        bgr = np.ascontiguousarray(bundle.rgb[:, :, ::-1])
        road = self.road(bgr)
        detections = self.detector.predict(bgr, imgsz=1280, conf=0.25,
                                           device=self.device, verbose=False)[0].boxes
        center = road.get("lane_center") or {}
        points = [self.lane_point(u, v, depth_image, calibration) for u, v in center.get("center_line", [])]
        points = np.asarray([p for p in points if p is not None and p[0] <= 30.])
        # Accept by extrapolation share when reported, otherwise by the estimator's own confidence flag.
        extrapolated = center.get("extrapolated_fraction")
        max_extrapolation = getattr(self, "max_extrapolation", 0.5)
        lane_hold_s = getattr(self, "lane_hold_s", 0.)
        last_lane = getattr(self, "_last_lane", None)
        acceptable = (extrapolated <= max_extrapolation if extrapolated is not None
                      else not center.get("low_confidence", True))
        lane_geometry_valid = len(points) >= 2 and np.ptp(points[:, 0]) > 1.
        lane_age = float(bundle.timestamp) - last_lane[0] if last_lane is not None else None
        current_route = path_lane(bundle.ego_path) if bundle.ego_path is not None else None
        route_lane = current_route if bundle.near_junction else None
        # Navigation starts this path at the ego's projected route position.
        # Check distance to that point as well as alignment with the path.
        aligned_route = current_route if (
            current_route is not None and
            math.hypot(*bundle.ego_path[0]) <= ROUTE_LANE_FALLBACK_MAX_OFFSET_M and
            abs(current_route[1]) <= ROUTE_LANE_FALLBACK_MAX_HEADING_RAD) else None
        held = None
        if lane_hold_s > 0. and lane_age is not None and 0. <= lane_age <= lane_hold_s:
            # Bridge dash gaps and short unpainted stretches with the last valid estimate.
            # Move it with the ego's motion relative to the route: a frozen value kept the
            # policy correcting for 2 s and drove it up to 2.3 m across the lane.
            # Not odometry yaw for heading: holds cluster in bends, and a yaw-based heading
            # read following the bend as an error (experiment_28 eval6: 0/6, 5 lost in holds).
            _, error, heading, seed_route = last_lane
            if seed_route is not None and current_route is not None:
                error += current_route[0] - seed_route[0]
                heading += current_route[1] - seed_route[1]
            held = (error, heading)
        vision_lane = fitted_lane = None
        if lane_geometry_valid and acceptable:
            # Fit y=a*x+b in metres; image-space heading_deg is not physical yaw.
            slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
            vision_lane = (float(-intercept / math.hypot(1., slope)), float(-math.atan(slope)))
            fitted_lane = vision_lane
            # A neighbouring-lane reading is far from the route offset, or from the held
            # estimate when there is no aligned route. The route is the reference when it
            # exists: a hold seeded by a wrong reading would accept the next wrong one.
            reference = aligned_route if aligned_route is not None else held
            if reference is not None and abs(vision_lane[0] - reference[0]) > VISION_LANE_MAX_DISAGREEMENT_M:
                vision_lane = None
        estimate = vision_lane if vision_lane is not None else held
        if route_lane is not None:
            # Junctions have no ego-lane paint; vision estimates there are missing or
            # wrong, so navigation (planned path + odometry) is the lane reference.
            fields.update(lane_error_m=route_lane[0], heading_error_rad=route_lane[1],
                          observations_valid=True, lane_source="route_junction")
            self._last_lane = (float(bundle.timestamp), *route_lane, route_lane)
        elif vision_lane is not None:
            fields.update(lane_error_m=vision_lane[0], heading_error_rad=vision_lane[1],
                          observations_valid=True, lane_source="vision")
            self._last_lane = (float(bundle.timestamp), *vision_lane, current_route)
        elif held is not None:
            fields.update(lane_error_m=held[0], heading_error_rad=held[1],
                          observations_valid=True, lane_source="held")
        else:
            # No estimate yet (e.g. at spawn) has nothing to hold, so the route may stand in at once.
            route_lane = aligned_route if lane_age is None or lane_age > lane_hold_s else None
            if route_lane is not None:
                fields.update(lane_error_m=route_lane[0], heading_error_rad=route_lane[1],
                              observations_valid=True, lane_source="route_fallback")
            elif not lane_geometry_valid:
                fields["observation_invalid_reason"] = "lane_geometry_unavailable"
            else:
                fields["observation_invalid_reason"] = "lane_low_confidence"
        # Evidence behind the lane estimate, written to evaluation traces for diagnosis only.
        self.lane_debug = dict(
            source=fields["lane_source"], points_ego_m=np.round(points[:, :2], 3).tolist() if len(points) else [],
            extrapolated_fraction=None if extrapolated is None else float(extrapolated), accepted=bool(acceptable), lane_age_s=lane_age,
            fit=fitted_lane, held=held, route=current_route, aligned_route=aligned_route,
            near_junction=bool(bundle.near_junction))
        boxes = detections.xyxy.cpu().numpy().reshape(-1, 4)
        confidences = detections.conf.cpu().numpy()
        classes = detections.cls.cpu().numpy().astype(int)
        boxes, confidences, classes = merge_nested_lights(boxes, confidences, classes)
        depths, uncertainties, depth_valid = associate_depth(depth_image, boxes)
        depth_valid &= depths < 999.
        lamp_depths, _, lamp_valid = associate_depth(
            np.where(depth_image < 999., depth_image, np.nan) if depth_image is not None else None,
            boxes, min_valid_fraction=.1)
        lights, signs = [], []
        for box, confidence, class_id, depth, uncertainty, has_depth, lamp_depth, lamp_has_depth in zip(
                boxes, confidences, classes, depths, uncertainties, depth_valid, lamp_depths, lamp_valid):
            if class_id in (1, 2, 3):
                u, v = (box[0]+box[2])/2., (box[1]+box[3])/2.
                # Ignore sky saturation while measuring the small lamp foreground.
                point = self.depth_point(u, v, float(lamp_depth), calibration) if lamp_has_depth else None
                intrinsics = calibration['intrinsics_px']
                lights.append(dict(point_ego=point, class_id=class_id, confidence=float(confidence),
                                   bearing=((u-intrinsics['cx'])/intrinsics['fx'],
                                            (v-intrinsics['cy'])/intrinsics['fy']),
                                   camera_location=np.asarray(calibration['extrinsics_ego']['location_m']),
                                   bearing_radius=np.array([(box[2]-box[0])/2./intrinsics['fx'],
                                                            (box[3]-box[1])/2./intrinsics['fy']])))
            elif class_id == 6:
                signs.append((box, depth, has_depth))
            elif class_id in (0, 4, 5):
                u, v = (box[0]+box[2])/2., box[3]
                if has_depth:
                    point = self.depth_point(u, v, float(depth), calibration)
                elif box[3] >= bundle.rgb.shape[0]-2:
                    point = None
                else:
                    point = self.ground_point(u, v, calibration)
                if point is None:
                    # No depth and truncated/above-horizon feet: cannot be located reliably.
                    fields["observations_valid"] = False
                    if fields["observation_invalid_reason"] is None:
                        fields["observation_invalid_reason"] = "object_position_unavailable"
                    continue
                row = np.zeros(16, np.float32)
                row[:3] = point / 100.
                row[6:11] = [float(class_id in (4, 5)), float(class_id == 5),
                             1., float(np.linalg.norm(point)/100.), float(confidence)]
                # Measured depth and spread ride along until the rows are sorted, then move to
                # PerceptionPrediction.depth_m/uncertainty, which the environment writes back.
                row[12:15] = [depth/100., 1., uncertainty/100.] if has_depth else [0., 0., 0.]
                rows.append(row)
        tracker = getattr(self, "object_tracker", None)
        if tracker is not None and bundle.ego_pose is not None:
            # Velocity channels: relative ego-frame velocity /30, as in the ground-truth rows.
            velocities = tracker.update(bundle.timestamp, bundle.ego_pose,
                                        [row[:3]*100. for row in rows], [bool(row[6] > .5) for row in rows])
            for row, velocity in zip(rows, velocities):
                row[3:6] = velocity/30.
        signal_fields, signals = self._signals(bundle, bgr, road, points, lights, calibration, depth_image)
        fields.update(signal_fields)
        self._read_speed_limit(bundle, bgr, signs, calibration, fields)
        crossing_points = []
        for crossing in road.get("crosswalks", []):
            x1, _, x2, bottom = crossing["xyxy"]
            point = self.ground_point((x1+x2)/2., bottom, calibration)
            if point is not None and abs(point[1]) < 3.5:
                crossing_points.append(point)
        if crossing_points:
            nearest = min(crossing_points, key=lambda point: point[0])
            fields.update(crossing_id="vision_crosswalk", crossing_distance_m=max(0., float(nearest[0])-2.5),
                          crossing_occupied=any(row[6] > .5 and
                                                abs(row[0]*100.-nearest[0]) < 5. and
                                                abs(row[1]*100.) < 5. for row in rows))
        return self._prediction(bundle, rows, fields, signals)

    def _read_speed_limit(self, bundle, bgr, signs, calibration, fields):
        """Vote on the nearest sign on our side; report the limit of the last sign passed."""
        tracker = getattr(self, "speed_limits", None)
        if tracker is None:
            return
        timestamp = float(bundle.timestamp)
        nearest = None
        for box, depth, has_depth in signs:
            if not has_depth:
                continue
            point = self.depth_point((box[0]+box[2])/2., (box[1]+box[3])/2., float(depth), calibration)
            if point is not None and point[1] >= -1. and (nearest is None or point[0] < nearest[1][0]):
                nearest = (box, point)
        value, score, read, reason = None, 0.0, False, "no_sign" if not signs else "no_depth_or_side"
        reader = getattr(self, "sign_reader", None)
        if nearest is not None:
            x1, y1, x2, y2 = nearest[0]
            if reader is None:
                reason = "no_reader"
            elif y2 - y1 < SIGN_MIN_HEIGHT_PX:
                reason = "too_small"
            elif not tracker.should_read(timestamp):
                reason = "interval"
            else:
                crop = bgr[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
                if crop.size:
                    value, score = read_speed_limit(reader, crop)
                    read, reason = True, "read" if value is not None else "ocr_failed"
                else:
                    reason = "empty_crop"
        distance = None if nearest is None else float(np.linalg.norm(nearest[1][:2]))
        limit_kmh, confidence = tracker.update(timestamp, distance, value, read)
        # Per-frame record for the evaluation trace: why a sign was or was not read.
        self.sign_debug = dict(
            detected=len(signs), distance_m=distance,
            height_px=None if nearest is None else float(nearest[0][3] - nearest[0][1]),
            reason=reason, ocr_value=value, ocr_score=score, votes=dict(tracker.votes),
            limit_kmh=limit_kmh)
        if limit_kmh is not None:
            fields.update(speed_limit_mps=limit_kmh / 3.6, speed_limit_confidence=float(confidence))

    @staticmethod
    def _prediction(bundle, rows, fields, signals=None):
        objects = np.asarray(sorted(rows, key=lambda row: row[9]), dtype=np.float32).reshape(-1, 16)
        depth_m = uncertainty = None
        if len(objects) and (objects[:, 13] > .5).any():
            measured = objects[:, 13] > .5
            depth_m = np.where(measured, objects[:, 12]*100., np.nan).astype(np.float32)
            uncertainty = np.where(measured, objects[:, 14]*100., np.nan).astype(np.float32)
        objects[:, 12:15] = 0.
        return PerceptionPrediction(objects, depth_m=depth_m, uncertainty=uncertainty, driving=fields,
                                    frame=bundle.frame, timestamp=bundle.timestamp,
                                    child_labels_authored=True, signals=signals)
