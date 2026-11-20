"""Associate stop lines and lamp heads with a path, then track in odometry.

Association is geometric, not learned. Individual candidates remain available
to attention. Stop lines come from the caller (the vision provider passes the
route's map stop lines); without one the range stays unknown, and no
lamp-to-line offset is inferred. No light actor or traffic phase is used.
"""

import numpy as np

from .observations import STOP_LINE_TOKEN, TRAFFIC_LIGHT_TOKEN
from .stop_lines import path_intersection, project_to_path

# Association-score advantage a new head needs to replace a tracked red head.
LOCK_SWITCH_MARGIN = 1.


def transform_points(points, pose, inverse=False):
    points = np.asarray(points, dtype=float)
    pose = np.asarray(pose, dtype=float)
    c, s = np.cos(pose[3]), np.sin(pose[3])
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return (points-pose[:3]) @ rotation if inverse else points @ rotation.T + pose[:3]


def stop_clearance(line, path, front_bumper):
    """Signed path distance from the projected front bumper to a measured line."""
    if front_bumper is None or not np.isfinite(front_bumper) or front_bumper < 0.:
        return None
    front = project_to_path((front_bumper, 0.), path)
    crossing = path_intersection(line, path)
    if front is None:
        return None
    if crossing is not None:
        path = np.asarray(path, dtype=float)
        delta = np.diff(path, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        valid = np.flatnonzero(lengths > 1e-8)
        front_station = front[0]
        if len(valid) and front_station <= 1e-8:
            along = float((np.array([front_bumper, 0.])-path[0]) @ (delta[valid[0]]/lengths[valid[0]]))
            front_station = min(0., along)
        return crossing[0]-front_station
    # A line behind the finite path must be retired, rather than clamped to 0.
    path = np.asarray(path, dtype=float)
    delta = np.diff(path, axis=0)
    valid = np.flatnonzero(np.linalg.norm(delta, axis=1) > 1e-8)
    if len(valid):
        tangent = delta[valid[0]] / np.linalg.norm(delta[valid[0]])
        behind = float((np.mean(line, axis=0)[:2]-[front_bumper, 0.]) @ tangent)
        if behind < 0.:
            return behind
    return None


class SignalTracker:
    def __init__(self, hold_s=2.):
        self.hold_s = hold_s
        self.reset()

    def reset(self):
        self.line = self.light = self.colour = self.last_seen = self.last_time = None
        self.confidence = 0.
        self.line_confidence = 0.
        self.passed_lights = []

    def update(self, timestamp, lines, lights, path, front_bumper, ego_pose, lane_width=3.5):
        fields = dict(red_light=False, red_light_distance_m=None, yellow_light=False,
                      yellow_light_distance_m=None, yellow_light_confidence=0.)
        if self.last_time is not None and timestamp < self.last_time:
            self.reset()
        self.last_time = timestamp
        pose = np.asarray(ego_pose, float) if ego_pose is not None else None
        if pose is not None and (pose.shape != (4,) or not np.isfinite(pose).all()):
            pose = None
        if pose is None:
            self.line = self.light = None
        lane_width = float(lane_width) if np.isfinite(lane_width) and lane_width > 0 else 3.5
        usable_path = project_to_path((0., 0.), path) is not None
        # Without odometry there is no valid way to carry geometry between frames.
        old_line = transform_points(self.line, pose, True) if self.line is not None and pose is not None else None
        old_distance = stop_clearance(old_line, path, front_bumper) if old_line is not None and usable_path else None
        if old_distance is not None and old_distance < 0.:
            if self.light is not None:
                self.passed_lights.append(self.light.copy())
            self.line = self.light = self.colour = self.last_seen = None
            old_line = None
        self.passed_lights = [p for p in self.passed_lights
                              if pose is not None and np.linalg.norm(p[:2]-pose[:2]) < 100.]
        candidates = []
        if usable_path:
            for item in lines:
                clearance = stop_clearance(item['points_ego'], path, front_bumper)
                hit = path_intersection(item['points_ego'], path)
                if hit is not None and (clearance is None or clearance >= 0.):
                    candidates.append((hit[0], item, clearance))
        candidate = min(candidates, key=lambda x:x[0], default=None)
        line = candidate[1]['points_ego'] if candidate is not None else old_line
        if candidate is not None:
            if old_line is not None and np.linalg.norm(np.mean(line, axis=0)-np.mean(old_line, axis=0)) > lane_width:
                self.light = self.colour = None
            self.line = transform_points(line, pose) if pose is not None else None
            self.last_seen = timestamp
            self.line_confidence = float(candidate[1]['confidence'])
        hit = path_intersection(line, path) if line is not None and usable_path else None
        distance = stop_clearance(line, path, front_bumper) if line is not None and usable_path else None
        fresh = candidate is not None or (self.last_seen is not None and timestamp-self.last_seen <= self.hold_s)
        if not fresh:
            distance = None

        def association(point):
            """How well a head (ego xyz) fits our stop: lower is better; None if it cannot control it."""
            projection = project_to_path(point[:2], path) if usable_path else None
            if point[0] <= 0. or projection is None or abs(projection[1]) > lane_width:
                return None
            station, lateral = projection
            score = abs(lateral)/lane_width
            if hit is not None:
                # Relative geometry replaces the fixed lateral/longitudinal offsets.
                score += abs(station-hit[0])/max(lane_width, hit[0])
            return score

        eligible, rows, unknown = [], [], []
        for light in lights:
            point = light['point_ego']
            row = np.zeros(16, np.float32)
            row[3:5] = light.get('bearing', (0., 0.))
            row[10], row[11], row[15] = light['confidence'], light['class_id']/3., TRAFFIC_LIGHT_TOKEN
            if point is not None:
                row[:3], row[8], row[9] = point/100., 1., np.linalg.norm(point)/100.
                location = transform_points(point, pose) if pose is not None else None
                passed = location is not None and any(np.linalg.norm(location-p) < lane_width/2. for p in self.passed_lights)
                score = association(point)
                if not passed and score is not None:
                    matched = (location is not None and self.light is not None and
                               np.linalg.norm(location-self.light) <= lane_width/2.)
                    eligible.append((not matched, score, light, location, row))
            else:
                passed = False
                if pose is not None and 'camera_location' in light and 'bearing_radius' in light:
                    for position in self.passed_lights:
                        ray = transform_points(position, pose, True)-light['camera_location']
                        if ray[0] > 0.:
                            bearing = np.array([ray[1]/ray[0], -ray[2]/ray[0]])
                            if np.all(np.abs(bearing-light['bearing']) <= light['bearing_radius']):
                                passed = True
                                break
                if not passed:
                    unknown.append(light)
            rows.append(row)
        eligible.sort(key=lambda item:(item[0], item[1]))
        selected = eligible[0] if eligible else None
        # Other lamps cannot release a tracked red merely by becoming green. The lock
        # yields only to a head that fits our stop clearly better: one taken at long
        # range on another approach's or a far-side head drifts off the path, or far
        # past the stop line, as the ego nears its own head.
        if selected is not None and self.light is not None and selected[0] and self.colour == 1:
            tracked = association(transform_points(self.light, pose, True)) if pose is not None else 0.
            if tracked is not None and selected[1] + LOCK_SWITCH_MARGIN >= tracked:
                selected = None
        if selected is not None:
            _, score, light, location, row = selected
            # Conflicting equally plausible heads: any red stays red (conservative); otherwise
            # the more confident head wins. Class order alone let a 0.25 yellow box beat a
            # 0.5 green on the same lamp and hold a stopped car at the line (town_transfer E07).
            competing = [e for e in eligible if e[0] == selected[0] and e[1] <= score+.15]
            selected = min(competing, key=lambda e:(e[2]['class_id'] != 1, -e[2]['confidence'],
                                                    e[2]['class_id']))
            _, score, light, location, row = selected
            row[5] = 1./(1.+score)
            self.colour, self.confidence = light['class_id'], light['confidence']
            self.light = location
        if self.light is not None and pose is not None:
            head = transform_points(self.light, pose, True)
            if head[0] <= 0. and old_line is None:
                self.light = self.colour = None
        # Lost geometry is an unknown range, not evidence that a red became green.
        colour = self.colour
        if unknown:
            # An unassociated red cannot be cleared by another head's known green.
            unknown_colour = min(l['class_id'] for l in unknown)
            if colour is None or unknown_colour < colour:
                colour = self.colour = unknown_colour
                self.confidence = max(l['confidence'] for l in unknown if l['class_id'] == colour)
        if colour == 1:
            fields.update(red_light=True, red_light_distance_m=distance)
        elif colour == 2:
            fields.update(yellow_light=True, yellow_light_distance_m=distance,
                          yellow_light_confidence=float(self.confidence))
        if line is not None and fresh and hit is not None:
            row = np.zeros(16, np.float32)
            row[:3] = np.r_[hit[1], np.mean(line[:, 2])]/100.
            row[3:6] = (line[1]-line[0])/100.
            row[8], row[9], row[10], row[15] = 1., (distance or 0.)/100., self.line_confidence, STOP_LINE_TOKEN
            rows.insert(0, row)
        # Geometry-relevant heads precede other candidates when token capacity is tight.
        rows.sort(key=lambda row:(row[15] != STOP_LINE_TOKEN, -row[5], row[9]))
        return fields, np.asarray(rows, np.float32).reshape(-1, 16)
