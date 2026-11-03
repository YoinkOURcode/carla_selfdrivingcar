"""Route projection and actual route/crosswalk intersections in world XY."""

import numpy as np


class Route:
    def __init__(self, points, *, lane_widths=None, junctions=None):
        self.points = np.asarray(points, dtype=np.float64)[:, :2]
        keep = np.r_[True, np.linalg.norm(np.diff(self.points, axis=0), axis=1) > 0.05]
        if lane_widths is not None and len(lane_widths) != len(self.points):
            raise ValueError("Route lane widths must match route points")
        if junctions is not None and len(junctions) != len(self.points):
            raise ValueError("Route junction flags must match route points")
        self.lane_widths = (np.asarray(lane_widths, dtype=np.float64)[keep]
                            if lane_widths is not None else None)
        self.junctions = (np.asarray(junctions, dtype=bool)[keep]
                          if junctions is not None else None)
        self.points = self.points[keep]
        if len(self.points) < 2:
            raise ValueError("Route needs at least two distinct points")
        self.delta = np.diff(self.points, axis=0)
        self.lengths = np.linalg.norm(self.delta, axis=1)
        self.cumulative = np.r_[0, self.lengths.cumsum()]
        self.index = 0

    def project(self, point, *, local=True):
        point = np.asarray(point)[:2]
        start = max(0, self.index - 3) if local else 0
        stop = min(len(self.delta), self.index + 30) if local else len(self.delta)
        delta = self.delta[start:stop]
        t = np.clip(((point - self.points[start:stop]) * delta).sum(-1) / self.lengths[start:stop] ** 2, 0, 1)
        projected = self.points[start:stop] + t[:, None] * delta
        distances = np.linalg.norm(projected - point, axis=1)
        j = int(np.argmin(distances))
        if local:
            # At overlapping or crossing segments, retain the nearby route branch.
            ties = np.flatnonzero(distances <= distances[j] + 0.05)
            j = int(ties[np.argmin(np.abs(start + ties - self.index))])
        index = start + j
        if local:
            self.index = index
        unit = self.delta[index] / self.lengths[index]
        difference = point - projected[j]
        lateral = float(unit[0] * difference[1] - unit[1] * difference[0])
        return float(self.cumulative[index] + t[j] * self.lengths[index]), lateral, float(np.arctan2(unit[1], unit[0]))

    def lane_geometry(self, progress_m, fallback_width):
        """Return planned lane width and whether this route segment enters a junction."""
        if self.lane_widths is None:
            return fallback_width, False
        index = min(max(int(np.searchsorted(self.cumulative, progress_m, side="right") - 1), 0),
                    len(self.delta) - 1)
        fraction = np.clip((progress_m - self.cumulative[index]) / self.lengths[index], 0., 1.)
        a, b = self.lane_widths[index:index+2]
        width = float(a + fraction * (b - a))
        if not np.isfinite(width) or width <= 0.:
            width = fallback_width
        junction = bool(self.junctions is not None and (self.junctions[index] or self.junctions[index+1]))
        return width, junction

    @property
    def length(self):
        return float(self.cumulative[-1])

    def crossings(self, polygons):
        """Intersect centreline with polygon edges, rejecting unrelated roads."""
        found = []
        for number, polygon in enumerate(polygons):
            polygon = np.asarray(polygon)[:, :2]
            intersections = []
            for i, (p, d) in enumerate(zip(self.points[:-1], self.delta)):
                for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                    edge = b - a
                    denominator = d[0] * edge[1] - d[1] * edge[0]
                    if abs(denominator) < 1e-9:
                        continue
                    offset = a - p
                    t = (offset[0] * edge[1] - offset[1] * edge[0]) / denominator
                    u = (offset[0] * d[1] - offset[1] * d[0]) / denominator
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersections.append(self.cumulative[i] + t * self.lengths[i])
            if len(intersections) >= 2 and max(intersections)-min(intersections) > 0.1:
                found.append((str(number), float(min(intersections)), float(max(intersections))))
        return sorted(found, key=lambda crossing: crossing[1])


def crosswalk_polygons(locations):
    polygons, current = [], []
    for location in locations:
        point = (location.x, location.y)
        if len(current) >= 3 and np.linalg.norm(np.subtract(point, current[0])) < 0.01:
            polygons.append(current)
            current = []
        else:
            current.append(point)
    if current:
        raise ValueError("Map crosswalk annotations contain an unclosed polygon")
    return polygons
