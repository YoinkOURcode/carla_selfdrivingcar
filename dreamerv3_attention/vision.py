"""Frozen RGB models and approximate flat-road geometry for external observations.

No simulator depth, actors, maps or weather are consulted. Distances assume a
flat road at ego z=0 and the supplied camera calibration; they are estimates.
"""

import math
import os
from pathlib import Path

import numpy as np

from .sensors import PerceptionPrediction


class VisionPerceptionProvider:
    camera_only = True

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
        # RoadPerception and YOLO's ndarray interface both accept BGR.
        bgr = np.ascontiguousarray(bundle.rgb[:, :, ::-1])
        road = self.road(bgr)
        detections = self.detector.predict(bgr, imgsz=1280, conf=0.25,
                                           device=self.device, verbose=False)[0].boxes
        center = road.get("lane_center") or {}
        points = [self.ground_point(u, v, calibration) for u, v in center.get("center_line", [])]
        points = np.asarray([p for p in points if p is not None and p[0] <= 30.])
        if (len(points) >= 2 and np.ptp(points[:, 0]) > 1. and
                not center.get("low_confidence", True)):
            # Fit y=a*x+b in metres; image-space heading_deg is not physical yaw.
            slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
            fields.update(lane_error_m=float(-intercept / math.hypot(1., slope)),
                          heading_error_rad=float(-math.atan(slope)), observations_valid=True)
        for box, confidence, class_id in zip(detections.xyxy.cpu().numpy(),
                                             detections.conf.cpu().numpy(),
                                             detections.cls.cpu().numpy().astype(int)):
            if class_id == 1:
                fields["red_light"] = True
            elif class_id == 2:
                fields["yellow_light"] = True
                fields["yellow_light_confidence"] = max(fields["yellow_light_confidence"], float(confidence))
            elif class_id in (0, 4, 5):
                point = self.ground_point((box[0]+box[2])/2., box[3], calibration)
                if point is None or box[3] >= bundle.rgb.shape[0]-2:
                    # Truncated/above-horizon feet cannot be located reliably.
                    fields["observations_valid"] = False
                    continue
                row = np.zeros(16, np.float32)
                row[:3] = point / 100.
                row[6:11] = [float(class_id in (4, 5)), float(class_id == 5),
                             1., float(np.linalg.norm(point)/100.), float(confidence)]
                rows.append(row)
        if fields["red_light"]:
            # Conflicting visible colours: red takes priority. Range remains unknown.
            fields["yellow_light"] = False
            fields["yellow_light_confidence"] = 0.
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

    @staticmethod
    def _prediction(bundle, rows, fields):
        objects = np.asarray(sorted(rows, key=lambda row: row[9]), dtype=np.float32).reshape(-1, 16)
        return PerceptionPrediction(objects, driving=fields, frame=bundle.frame,
                                    timestamp=bundle.timestamp, child_labels_authored=True)
