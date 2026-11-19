"""Speed-limit number reading for the vision provider.

The sign-reading steps (find the red-ringed plate, clean it, OCR it, keep only
values CARLA's sign assets use, vote over several frames) are taken from
perception/live_depth/yolo_depth_live.py. Only the reading and voting parts are
reused; that script's own autopilot loop, tracker and display are not.

OCR uses rapidocr_onnxruntime (1.2.3 reads small signs best). When it is not
installed, make_sign_reader() returns None and no limit is ever reported.
"""

import cv2
import numpy as np

VALID_LIMITS = {10, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130}  # CARLA sign asset values
SIGN_MIN_HEIGHT_PX = 12      # smaller signs are not read at all
SIGN_READ_INTERVAL_S = 0.1   # read the visible sign at most this often (simulation time)
SIGN_MIN_VOTES = 3           # a value must be read this often, and at least twice as often as any other
SIGN_PASS_DISTANCE_M = 15.0  # a confirmed sign last seen this close, then out of view, counts as passed
SIGN_PASS_GAP_S = 0.3        # out of view this long counts as left the view
SIGN_FORGET_S = 1.5          # an unconfirmed or far sign out of view this long is forgotten


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
    red = ((hsv[..., 0] <= 12) | (hsv[..., 0] >= 160)) & (hsv[..., 1] > 70) & (hsv[..., 2] > 80)
    gray[red] = 255
    if circle:
        ch, cw = gray.shape
        yy, xx = np.ogrid[:ch, :cw]
        gray[((xx - cw / 2) / (cw * 0.40)) ** 2 + ((yy - ch / 2) / (ch * 0.40)) ** 2 > 1] = 255
    lo, hi = np.percentile(gray, 3), np.percentile(gray, 97)
    if hi - lo > 10:
        gray = np.clip((gray.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.copyMakeBorder(out, 12, 12, 16, 16, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def sign_cores(crop):
    """Two cut-outs of the sign's centre for reading: circle-masked and inset rectangle."""
    disc = find_sign_disc(crop)
    if disc is None:  # no ring found: assume the plate is the top square of the box
        h, w = crop.shape[:2]
        disc = (0, 0, w, min(w, h))
    x, y, w, h = disc
    full = crop[y:y + h, x:x + w]
    mx, my = int(round(w * 0.14)), int(round(h * 0.14))
    inset = crop[y + my:y + h - my, x + mx:x + w - mx]
    return [_clean_sign(c, circle) for c, circle in ((full, True), (inset, False))
            if c.size and min(c.shape[:2]) >= 4]


def parse_limit(text):
    # ASCII only: OCR can emit glyphs like "③" that isdigit() accepts but int() rejects.
    digits = "".join(ch for ch in text.upper().replace("O", "0").replace("D", "0") if ch in "0123456789")
    if not digits:
        return None
    value = int(digits)
    return value if value in VALID_LIMITS else None


def read_speed_limit(ocr, crop):
    """Read one BGR sign crop. Returns (value in km/h or None, score)."""
    cores = sign_cores(crop)
    if not cores:
        return None, 0.0
    # The recognizer is called directly: the library's own 0.5 score cut-off drops most small signs.
    recognize = getattr(ocr, "text_rec", None) or getattr(ocr, "text_recognizer")
    best = (None, 0.0)
    for item in recognize(cores)[0]:
        value = parse_limit(item[0])
        if value is not None and float(item[1]) > best[1]:
            best = (value, float(item[1]))
    return best


def make_sign_reader():
    """RapidOCR instance, or None when the package is missing or unsupported."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return None
    ocr = RapidOCR()
    if getattr(ocr, "text_rec", None) is None and getattr(ocr, "text_recognizer", None) is None:
        return None
    return ocr


class SpeedLimitTracker:
    """Vote on the nearest visible sign; apply its value once the car passes it.

    Before any sign is passed the limit is unknown (None), so the driving
    stack's default speed applies, as before.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.limit_kmh, self.confidence = None, 0.0
        self.votes, self.last_seen_t, self.last_distance_m, self.last_read_t = {}, None, None, None

    def confirmed(self):
        if not self.votes:
            return None, 0.0
        ranked = sorted(self.votes.values(), reverse=True)
        top, second = ranked[0], (ranked[1] if len(ranked) > 1 else 0)
        if top < SIGN_MIN_VOTES or top < 2 * second:
            return None, 0.0
        value = max(self.votes, key=self.votes.get)
        return value, top / sum(self.votes.values())

    def should_read(self, timestamp):
        return self.last_read_t is None or timestamp - self.last_read_t >= SIGN_READ_INTERVAL_S

    def update(self, timestamp, sign_distance_m=None, value=None, read=False):
        """One frame: distance of the nearest visible sign (None if none) and its reading, if read."""
        if sign_distance_m is not None:
            self.last_seen_t, self.last_distance_m = timestamp, sign_distance_m
            if read:
                self.last_read_t = timestamp
                if value is not None:
                    self.votes[value] = self.votes.get(value, 0) + 1
        elif self.last_seen_t is not None:
            gap = timestamp - self.last_seen_t
            value, share = self.confirmed()
            if (value is not None and gap >= SIGN_PASS_GAP_S and
                    self.last_distance_m is not None and self.last_distance_m <= SIGN_PASS_DISTANCE_M):
                self.limit_kmh, self.confidence = value, share
                self.votes, self.last_seen_t, self.last_distance_m, self.last_read_t = {}, None, None, None
            elif gap >= SIGN_FORGET_S or gap < 0:
                self.votes, self.last_seen_t, self.last_distance_m, self.last_read_t = {}, None, None, None
        return self.limit_kmh, self.confidence
