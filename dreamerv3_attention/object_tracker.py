"""Object velocity from depth-camera positions tracked across frames.

Ported from perception/live_depth/yolo_depth_live.py (Track/Tracker): detections
are placed in world coordinates with the ego pose, matched greedily to existing
tracks of the same kind, and each velocity is a least-squares fit over a short
window. Output matches the ground-truth object token: velocity relative to the
ego, in the ego frame.
"""

import math

import numpy as np

GATE_M = {False: 3.0, True: 1.5}  # vehicle, pedestrian: largest plausible jump between frames
MAX_AGE_S = 0.5     # forget a track not seen for this long
WINDOW_S = 0.6      # velocity is fitted over this window
MIN_SAMPLES = 5
MIN_SPAN_S = 0.2


def fit_velocity(history):
    """Least-squares xy velocity (m/s) of [(t, xy)] over the last WINDOW_S, or None."""
    t_now = history[-1][0]
    recent = [h for h in history if t_now - h[0] <= WINDOW_S]
    if len(recent) < MIN_SAMPLES or recent[-1][0] - recent[0][0] < MIN_SPAN_S:
        return None
    ts = np.array([h[0] for h in recent])
    ts -= ts.mean()
    xy = np.array([h[1] for h in recent])
    return (ts[:, None] * (xy - xy.mean(0))).sum(0) / float((ts**2).sum())


class ObjectTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tracks = []  # dicts: pedestrian flag, history [(t, world xy)]
        self.ego = []     # [(t, world xy)]

    def update(self, timestamp, ego_pose, points_ego, pedestrian):
        """Relative velocity (m/s, ego frame, xyz) for each ego-frame point.

        Zero until an object (and the ego) has MIN_SAMPLES positions in the window.
        """
        t = float(timestamp)
        x, y, _, yaw = ego_pose
        c, s = math.cos(yaw), math.sin(yaw)
        world = [np.array([x + c*p[0] - s*p[1], y + s*p[0] + c*p[1]]) for p in points_ego]
        self.ego = [h for h in self.ego if t - h[0] <= WINDOW_S] + [(t, np.array([x, y]))]
        self.tracks = [tr for tr in self.tracks if t - tr["history"][-1][0] <= MAX_AGE_S]

        pairs = []
        for di, (point, walker) in enumerate(zip(world, pedestrian)):
            for ti, tr in enumerate(self.tracks):
                if tr["pedestrian"] != walker:
                    continue
                last_t, last = tr["history"][-1]
                velocity = fit_velocity(tr["history"])
                predicted = last if velocity is None else last + velocity*(t - last_t)
                cost = float(np.linalg.norm(predicted - point))
                if cost <= GATE_M[walker]:
                    pairs.append((cost, di, ti))
        matched, used = {}, set()
        for _, di, ti in sorted(pairs):
            if di not in matched and ti not in used:
                matched[di] = self.tracks[ti]
                used.add(ti)

        ego_velocity = fit_velocity(self.ego)
        out = []
        for di, (point, walker) in enumerate(zip(world, pedestrian)):
            track = matched.get(di)
            if track is None:
                track = {"pedestrian": walker, "history": []}
                self.tracks.append(track)
            track["history"] = [h for h in track["history"] if t - h[0] <= WINDOW_S] + [(t, point)]
            velocity = fit_velocity(track["history"])
            if velocity is None or ego_velocity is None:
                out.append(np.zeros(3))
                continue
            rx, ry = velocity - ego_velocity
            out.append(np.array([c*rx + s*ry, -s*rx + c*ry, 0.]))
        return out
