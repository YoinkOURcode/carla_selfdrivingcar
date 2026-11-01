"""Remove duplicate boxes after detection. The largest box in a duplicate pair stays."""
import numpy as np

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
