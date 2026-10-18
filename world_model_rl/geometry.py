"""Geometry independent of CARLA, also used by offline checks."""

import math

import numpy as np


def ego_relative(vector, yaw_radians):
    x, y = vector[:2]
    c, s = math.cos(yaw_radians), math.sin(yaw_radians)
    return np.array([c * x + s * y, s * x - c * y], np.float32)


def crosses_line(before, after, centre, forward, half_width):
    before, after, centre, forward = map(np.asarray, (before, after, centre, forward))
    a, b = np.dot(before - centre, forward), np.dot(after - centre, forward)
    if not a < 0 <= b or b - a < 1e-8:
        return False
    crossing = before + (-a / (b - a)) * (after - before)
    lateral = np.array([-forward[1], forward[0]])
    return bool(abs(np.dot(crossing - centre, lateral)) <= half_width)


class Route:
    def __init__(self, points):
        points = np.asarray(points, np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("A route needs at least two 2D points")
        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-4]
        self.points = points[keep]
        self.vectors = np.diff(self.points, axis=0)
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        if not len(self.lengths):
            raise ValueError("Route has zero length")
        self.arc = np.r_[0.0, np.cumsum(self.lengths)]
        self.length = float(self.arc[-1])
        self.progress = 0.0
        self.high_water = 0.0

    def project(self, position, local=False):
        delta = np.asarray(position) - self.points[:-1]
        t = np.clip(np.sum(delta * self.vectors, axis=1) / self.lengths ** 2, 0, 1)
        nearest = self.points[:-1] + self.vectors * t[:, None]
        distance = np.linalg.norm(nearest - position, axis=1)
        along = self.arc[:-1] + t * self.lengths
        if local:
            # Prevent jumping to a later pass at an intersection or a nearby parallel road.
            allowed = (along >= self.progress - 8) & (along <= self.progress + 15)
            distance = np.where(allowed, distance, np.inf)
            if not np.isfinite(distance).any():
                raise RuntimeError("Lost route projection")
        index = int(np.argmin(distance))
        return float(along[index]), float(distance[index]), index

    def update(self, position):
        value, _, _ = self.project(position, local=True)
        # Repeatedly reversing over a segment cannot earn progress again.
        reward_distance = max(0, value - self.high_water) - 0.5 * max(0, self.progress - value)
        self.progress = value
        self.high_water = max(self.high_water, value)
        return reward_distance

    def at(self, distances):
        distances = np.clip(np.asarray(distances), 0, self.length)
        return np.stack([np.interp(distances, self.arc, self.points[:, i]) for i in range(2)], -1)


class TrackedSlots:
    """Retain numeric actor identities while visible; slots are never semantic features."""

    def __init__(self, size):
        self.ids = [-1] * size

    def assign(self, candidates):
        # Candidate tuples: (numeric identity, risk/distance priority, data).
        candidates = sorted(candidates, key=lambda item: item[1])[:len(self.ids)]
        available_ids = [item[0] for item in candidates]
        self.ids = [value if value in available_ids else -1 for value in self.ids]
        for identity, _, _ in candidates:
            if identity not in self.ids and -1 in self.ids:
                self.ids[self.ids.index(-1)] = identity
        return [next((item[2] for item in candidates if item[0] == identity), None)
                if identity != -1 else None for identity in self.ids]
