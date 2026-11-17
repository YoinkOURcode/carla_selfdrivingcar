"""Frozen RGB models, the aligned depth camera and OCR for external observations.

No simulator actors, maps or weather are consulted. Object and traffic-light
distances come from the CARLA depth camera mounted with the RGB camera (a
stand-in for a real depth sensor). When depth is missing, objects fall back to
flat-road projection (ego z=0, supplied camera calibration) and lights keep an
unknown distance. Speed-limit numbers are read from sign crops with OCR.
"""

import math
import os
from pathlib import Path

import numpy as np

from .sensors import PerceptionPrediction, associate_depth
from .speed_sign import SIGN_MIN_HEIGHT_PX, SpeedLimitTracker, make_sign_reader, read_speed_limit

BUMPER_ALLOWANCE_M = 2.5  # same fixed ego-origin-to-bumper allowance as crossings


def _env_float(name, default, low, high):
    value = float(os.environ.get(name, default))
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


class VisionPerceptionProvider:
    camera_only = True
    uses_depth_camera = True
    # Defaults, overridable through the environment variables read in __init__.
    light_stop_offset_m, light_min_lateral_m, light_max_lateral_m = 7.1, -1.0, 6.0
    sign_reader = speed_limits = None

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
        # Lamp head to stop line along the lane: 7.1 m median over all 24 Town02 stop points
        # (range 6.9 to 7.4), heads 2.3 to 2.7 m right of the lane centre.
        self.light_stop_offset_m = _env_float("VISION_LIGHT_STOP_OFFSET_M", "7.1", 0., 30.)
        self.light_min_lateral_m = _env_float("VISION_LIGHT_MIN_LATERAL_M", "-1.0", -20., 20.)
        self.light_max_lateral_m = _env_float("VISION_LIGHT_MAX_LATERAL_M", "6.0", -20., 20.)
        self.sign_reader = make_sign_reader()
        self.speed_limits = SpeedLimitTracker()
        if self.sign_reader is None:
            print("VisionPerceptionProvider: rapidocr_onnxruntime missing, speed-limit numbers will not be read")

    def reset(self):
        self._last_lane = None
        if getattr(self, "speed_limits", None) is not None:
            self.speed_limits.reset()

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

    def _signal(self, found, calibration):
        """(applies, stop-line distance or None) for the detected lights of one colour.

        A light with measured depth outside our side of the road belongs to another
        lane and is ignored. A light without depth still applies, at unknown range.
        """
        applies, best = False, None
        for (x1, y1, x2, y2), depth, has_depth in found:
            point = (self.depth_point((x1+x2)/2., (y1+y2)/2., float(depth), calibration)
                     if has_depth else None)
            if point is None:
                applies = True
                continue
            if not self.light_min_lateral_m <= point[1] <= self.light_max_lateral_m:
                continue
            applies = True
            distance = max(0., float(point[0]) - self.light_stop_offset_m - BUMPER_ALLOWANCE_M)
            best = distance if best is None else min(best, distance)
        # Any applicable light without range keeps the conservative unknown distance.
        if any(not has_depth for _, _, has_depth in found):
            best = None
        return applies, best

    def predict(self, bundle):
        fields = dict(speed_limit_mps=None, speed_limit_confidence=0., lane_error_m=0.,
                      heading_error_rad=0., crossing_id=None, crossing_distance_m=None,
                      crossing_occupied=False, red_light=False, red_light_distance_m=None,
                      yellow_light=False, yellow_light_distance_m=None, yellow_light_confidence=0.,
                      visibility_m=30., observations_valid=False)
        rows = []
        if bundle.rgb is None or not bundle.valid.get("rgb", False):
            return self._prediction(bundle, rows, fields)
        calibration = bundle.calibration.get("sensors", {}).get("rgb")
        if calibration is None:
            return self._prediction(bundle, rows, fields)
        depth_image = bundle.depth if bundle.valid.get("depth", bundle.depth is not None) else None
        if depth_image is not None and np.shape(depth_image) != bundle.rgb.shape[:2]:
            depth_image = None  # not pixel-aligned with the RGB frame
        # RoadPerception and YOLO's ndarray interface both accept BGR.
        bgr = np.ascontiguousarray(bundle.rgb[:, :, ::-1])
        road = self.road(bgr)
        detections = self.detector.predict(bgr, imgsz=1280, conf=0.25,
                                           device=self.device, verbose=False)[0].boxes
        center = road.get("lane_center") or {}
        points = [self.ground_point(u, v, calibration) for u, v in center.get("center_line", [])]
        points = np.asarray([p for p in points if p is not None and p[0] <= 30.])
        # Accept by extrapolation share when reported, otherwise by the estimator's own confidence flag.
        extrapolated = center.get("extrapolated_fraction")
        max_extrapolation = getattr(self, "max_extrapolation", 0.5)
        lane_hold_s = getattr(self, "lane_hold_s", 0.)
        last_lane = getattr(self, "_last_lane", None)
        acceptable = (extrapolated <= max_extrapolation if extrapolated is not None
                      else not center.get("low_confidence", True))
        if len(points) >= 2 and np.ptp(points[:, 0]) > 1. and acceptable:
            # Fit y=a*x+b in metres; image-space heading_deg is not physical yaw.
            slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
            fields.update(lane_error_m=float(-intercept / math.hypot(1., slope)),
                          heading_error_rad=float(-math.atan(slope)), observations_valid=True)
            self._last_lane = (float(bundle.timestamp), fields["lane_error_m"], fields["heading_error_rad"])
        elif (lane_hold_s > 0. and last_lane is not None and
              0. <= float(bundle.timestamp) - last_lane[0] <= lane_hold_s):
            # Bridge dash gaps and short unpainted stretches with the last valid estimate.
            fields.update(lane_error_m=last_lane[1], heading_error_rad=last_lane[2],
                          observations_valid=True)
        boxes = detections.xyxy.cpu().numpy().reshape(-1, 4)
        confidences = detections.conf.cpu().numpy()
        classes = detections.cls.cpu().numpy().astype(int)
        depths, uncertainties, depth_valid = associate_depth(depth_image, boxes)
        lights, signs = {1: [], 2: []}, []
        for box, confidence, class_id, depth, uncertainty, has_depth in zip(
                boxes, confidences, classes, depths, uncertainties, depth_valid):
            if class_id in (1, 2):
                lights[class_id].append((box, depth, has_depth, float(confidence)))
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
                    continue
                row = np.zeros(16, np.float32)
                row[:3] = point / 100.
                row[6:11] = [float(class_id in (4, 5)), float(class_id == 5),
                             1., float(np.linalg.norm(point)/100.), float(confidence)]
                # Measured depth and spread ride along until the rows are sorted, then move to
                # PerceptionPrediction.depth_m/uncertainty, which the environment writes back.
                row[12:15] = [depth/100., 1., uncertainty/100.] if has_depth else [0., 0., 0.]
                rows.append(row)
        fields["red_light"], fields["red_light_distance_m"] = self._signal(
            [light[:3] for light in lights[1]], calibration)
        fields["yellow_light"], fields["yellow_light_distance_m"] = self._signal(
            [light[:3] for light in lights[2]], calibration)
        if fields["yellow_light"]:
            fields["yellow_light_confidence"] = max(light[3] for light in lights[2])
        if fields["red_light"]:
            # Conflicting visible colours: red takes priority.
            fields["yellow_light"] = False
            fields["yellow_light_confidence"] = 0.
            fields["yellow_light_distance_m"] = None
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
        return self._prediction(bundle, rows, fields)

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
        value, read = None, False
        reader = getattr(self, "sign_reader", None)
        if nearest is not None and reader is not None and tracker.should_read(timestamp):
            x1, y1, x2, y2 = nearest[0]
            if y2 - y1 >= SIGN_MIN_HEIGHT_PX:
                crop = bgr[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
                if crop.size:
                    value, _ = read_speed_limit(reader, crop)
                    read = True
        distance = None if nearest is None else float(np.linalg.norm(nearest[1][:2]))
        limit_kmh, confidence = tracker.update(timestamp, distance, value, read)
        if limit_kmh is not None:
            fields.update(speed_limit_mps=limit_kmh / 3.6, speed_limit_confidence=float(confidence))

    @staticmethod
    def _prediction(bundle, rows, fields):
        objects = np.asarray(sorted(rows, key=lambda row: row[9]), dtype=np.float32).reshape(-1, 16)
        depth_m = uncertainty = None
        if len(objects) and (objects[:, 13] > .5).any():
            measured = objects[:, 13] > .5
            depth_m = np.where(measured, objects[:, 12]*100., np.nan).astype(np.float32)
            uncertainty = np.where(measured, objects[:, 14]*100., np.nan).astype(np.float32)
        objects[:, 12:15] = 0.
        return PerceptionPrediction(objects, depth_m=depth_m, uncertainty=uncertainty, driving=fields,
                                    frame=bundle.frame, timestamp=bundle.timestamp,
                                    child_labels_authored=True)
