"""Measured white road stripes crossing a camera-derived planned path.

Only aligned RGB/depth, camera calibration and the supplied path are used.
No stripe means unknown geometry. Crosswalk detections exclude their paint.
"""

import cv2
import numpy as np


def _path_segments(path_ego):
    path = np.asarray(path_ego, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2 or not np.isfinite(path).all():
        return None
    delta = np.diff(path, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    starts = np.r_[0., np.cumsum(lengths)[:-1]]
    keep = lengths > 1e-8
    if not keep.any():
        return None
    return path[:-1][keep], delta[keep], lengths[keep], starts[keep]


def project_to_path(point_xy, path_ego):
    """Return nearest (arc station, signed lateral metres), or None.

    Station starts at path[0], which need not coincide with the ego origin.
    Lateral is tangent cross displacement: positive ego y for a +x path.
    Projections stop at the path's finite endpoints.
    """
    segments = _path_segments(path_ego)
    point = np.asarray(point_xy, dtype=float)
    if segments is None or point.shape != (2,) or not np.isfinite(point).all():
        return None
    origins, delta, lengths, starts = segments
    fraction = np.clip(np.sum((point - origins) * delta, axis=1) / lengths**2, 0., 1.)
    offset = point - (origins + fraction[:, None] * delta)
    index = int(np.argmin(np.sum(offset**2, axis=1)))
    lateral = (delta[index, 0] * offset[index, 1] - delta[index, 1] * offset[index, 0]) / lengths[index]
    return float(starts[index] + fraction[index] * lengths[index]), float(lateral)


def path_intersection(points_ego, path_ego):
    """Return first (arc station, xy) where a measured segment crosses path.

    Parallel/collinear segments provide no unique transverse intersection.
    Endpoints retain their signed ego-frame coordinates.
    """
    points = np.asarray(points_ego, dtype=float)
    segments = _path_segments(path_ego)
    if (segments is None or points.ndim != 2 or points.shape[0] != 2
            or points.shape[1] not in (2, 3) or not np.isfinite(points).all()):
        return None
    origins, delta, lengths, starts = segments
    stripe = points[1, :2] - points[0, :2]
    denominator = delta[:, 0] * stripe[1] - delta[:, 1] * stripe[0]
    relative = points[0, :2] - origins
    nonparallel = np.abs(denominator) > 1e-8
    safe = np.where(nonparallel, denominator, 1.)
    along = (relative[:, 0] * stripe[1] - relative[:, 1] * stripe[0]) / safe
    across = (relative[:, 0] * delta[:, 1] - relative[:, 1] * delta[:, 0]) / safe
    valid = nonparallel & (along >= -1e-8) & (along <= 1. + 1e-8) & (across >= -1e-8) & (across <= 1. + 1e-8)
    if not valid.any():
        return None
    indices = np.flatnonzero(valid)
    index = indices[np.argmin(starts[indices] + along[indices] * lengths[indices])]
    return float(starts[index] + along[index] * lengths[index]), origins[index] + along[index] * delta[index]


def _road_plane(depth, white, intrinsics, location, path_ego, lane_width_m):
    """Fit the dominant road surface near the path from non-paint depth.

    The camera must be above the road, but ego z=0 need not lie on it.
    Broad support and a majority consensus prevent a small raised object
    from supplying its own ground reference.
    """
    fx, fy, cx, cy = intrinsics
    step = max(4, min(depth.shape) // 60)
    v, u = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
    axial = depth[v, u].astype(float)
    valid = ((v > cy) & np.isfinite(axial) & (axial > 0.) & (axial < 80.) & (white[v, u] == 0))
    u, v, axial = u[valid], v[valid], axial[valid]
    points = location + np.column_stack((axial, axial * (u-cx)/fx, -axial * (v-cy)/fy))
    origins, deltas, lengths, _ = _path_segments(path_ego)
    distance_sq = np.full(len(points), np.inf)
    for origin, delta, length in zip(origins, deltas, lengths):
        fraction = np.clip((points[:, :2] - origin) @ delta / length**2, 0., 1.)
        offsets = points[:, :2] - origin - fraction[:, None] * delta
        distance_sq = np.minimum(distance_sq, np.sum(offsets**2, axis=1))
    points = points[(distance_sq <= (.65*lane_width_m)**2) & (points[:, 2] < location[2]-.5)]
    if len(points) < 30:
        return None
    design = np.column_stack((points[:, :2], np.ones(len(points))))
    rng = np.random.default_rng(0)
    best = np.zeros(len(points), dtype=bool)
    for _ in range(64):
        indices = rng.choice(len(points), 3, replace=False)
        try:
            plane = np.linalg.solve(design[indices], points[indices, 2])
        except np.linalg.LinAlgError:
            continue
        if np.linalg.norm(plane[:2]) > .35:
            continue
        residual = np.abs(np.sum(design * plane, axis=1) - points[:, 2])
        inliers = residual < .12
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 30 or best.mean() < .55:
        return None
    support = points[best, :2]
    if np.ptp(support[:, 0]) < 5. or np.ptp(support[:, 1]) < .4*lane_width_m:
        return None
    plane, _, rank, _ = np.linalg.lstsq(design[best], points[best, 2], rcond=None)
    return plane if rank == 3 and np.linalg.norm(plane[:2]) <= .35 else None


def extract_stop_lines(bgr, depth_m, calibration, path_ego, lane_width_m=3.5, crosswalk_boxes=()):
    """Return measured stripe endpoints, pixels and confidence, nearest first.

    Calibration follows vision.py's level front-camera contract. Axial depth
    must be aligned, finite, positive and below the 1000 m saturation value.
    Surrounding non-paint depth must establish a local road plane. White
    elongated paint on that plane must span half the supplied lane width
    and cross the path transversely. This cannot classify arbitrary
    unlabelled transverse paint semantically; caller crosswalk boxes exclude
    zebra paint. Occluded, low-contrast or very distant paint may be missed.
    """
    if bgr is None or depth_m is None or calibration is None or _path_segments(path_ego) is None:
        return []
    image, depth = np.asarray(bgr), np.asarray(depth_m)
    if image.ndim != 3 or image.shape[2] != 3 or depth.shape != image.shape[:2]:
        return []
    if not np.isfinite(lane_width_m) or lane_width_m <= 0.:
        return []
    intrinsics, extrinsics = calibration["intrinsics_px"], calibration["extrinsics_ego"]
    if not np.allclose(extrinsics["rotation_pitch_yaw_roll_deg"], (0., 0., 0.)):
        raise ValueError("Stop-line depth projection requires a level front camera")
    fx, fy, cx, cy = [float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy")]
    location = np.asarray(extrinsics["location_m"], dtype=float)
    if location.shape != (3,) or not np.isfinite([fx, fy, cx, cy, *location]).all() or min(fx, fy) <= 0.:
        return []
    high, low = image.max(axis=2).astype(float), image.min(axis=2).astype(float)
    white = ((low >= 170.) & (high - low <= 65.)).astype(np.uint8)
    plane = _road_plane(depth, white, (fx, fy, cx, cy), location, path_ego, lane_width_m)
    if plane is None:
        return []
    # Exclude whole paint components touching a detected crossing, preventing
    # an artificial stop stripe at a clipped crosswalk-box boundary.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    excluded = set()
    height, width = white.shape
    for box in crosswalk_boxes:
        box = np.asarray(box, dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            continue
        x1, y1 = np.maximum(np.floor(box[:2]).astype(int), 0)
        x2, y2 = np.minimum(np.ceil(box[2:]).astype(int) + 1, (width, height))
        if x2 > x1 and y2 > y1:
            excluded.update(np.unique(labels[y1:y2, x1:x2]).tolist())
    if excluded:
        white[np.isin(labels, list(excluded))] = 0
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((1, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    results = []
    path_segments = _path_segments(path_ego)
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] < 12:
            continue
        left, top, component_width, component_height = stats[component, :4]
        v, u = np.nonzero(labels[top:top+component_height, left:left+component_width] == component)
        u, v = u + left, v + top
        axial = depth[v, u].astype(float)
        valid = np.isfinite(axial) & (axial > 0.) & (axial < 999.)
        if valid.sum() < 12 or valid.mean() < .75:
            continue
        u, v, axial = u[valid], v[valid], axial[valid]
        points = location + np.column_stack((axial, axial * (u-cx)/fx, -axial * (v-cy)/fy))
        plane_height = points[:, :2] @ plane[:2] + plane[2]
        ground = np.abs(points[:, 2] - plane_height) <= .18
        if ground.mean() < .85 or ground.sum() < 12:
            continue
        points, u, v = points[ground], u[ground], v[ground]
        center = np.median(points, axis=0)
        _, _, axes = np.linalg.svd(points[:, :2] - center[:2], full_matrices=False)
        direction = axes[0]
        coordinate = (points[:, :2] - center[:2]) @ direction
        ends = np.percentile(coordinate, (1., 99.))
        span = ends[1] - ends[0]
        normal = np.array([-direction[1], direction[0]])
        thickness = np.percentile(np.abs((points[:, :2] - center[:2]) @ normal), 90.) * 2.
        if not .5 * lane_width_m <= span <= 2.5 * lane_width_m or thickness > .8 or thickness > .25 * span:
            continue
        endpoints = np.repeat(center[None], 2, axis=0)
        endpoints[:, :2] += ends[:, None] * direction
        endpoints[:, 2] = np.polyval(np.polyfit(coordinate, points[:, 2], 1), ends)
        intersection = path_intersection(endpoints, path_ego)
        if intersection is None:
            continue
        station, _ = intersection
        _, path_delta, path_lengths, path_starts = path_segments
        index = int(np.argmin(np.abs(np.clip(station-path_starts, 0., path_lengths) + path_starts - station)))
        alignment = abs(float(direction @ (path_delta[index] / path_lengths[index])))
        if alignment > .45:
            continue
        # Endpoint pixels interpolate the same depth-supported stripe fit.
        pixels = np.column_stack([np.polyval(np.polyfit(coordinate, values, 1), ends) for values in (u, v)])
        confidence = float(np.clip(valid.mean() * ground.mean() * (1.-alignment) * (1.-thickness/span), 0., 1.))
        results.append((station, dict(points_ego=endpoints, pixels=pixels, confidence=confidence)))
    return [candidate for _, candidate in sorted(results, key=lambda item: item[0])]
