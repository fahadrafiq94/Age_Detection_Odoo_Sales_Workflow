"""
ERP Lab Kiosk  -  smile + age based drink ordering (improved version)
=====================================================================

What it does
------------
A person stands inside the red guide box and smiles. The kiosk estimates the
age group, and sends an order to the ERP (Node-RED) endpoint:
    adult -> "Hugo Cocktail"          child -> "Lemonade"

What is new compared with the previous version
----------------------------------------------
1. Smile detection : MediaPipe facial-expression score (mouthSmileLeft/Right)
                      instead of the Haar smile cascade (which missed clear smiles).
2. Face detection  : MediaPipe Face Landmarker instead of the Haar face cascade.
                      Works with glasses, face masks, low light and small faces
                      where the Haar cascade returned "No face detected".
3. Multiple faces  : only faces whose centre is INSIDE the red box are considered.
                      Faces outside the box are ignored. If two similar-sized
                      people stand inside the box, the kiosk pauses.
4. Age estimation  : several crops + mirrored crops per frame, probabilities
                      averaged over several frames, decision by "adult
                      probability" instead of "4 identical buckets in a row".
                      If the model stays unsure, the safe choice (child ->
                      Lemonade) is used after a timeout. Very dark faces are
                      not classified at all ("too dark" message).

If MediaPipe (or its model file) is not available, the kiosk automatically
falls back to the old Haar cascades and prints a clear warning.

Keys: q = quit, r = reset current customer
"""

import math
import os
import time
import urllib.request
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import requests

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_IMPORT_ERROR = None
except Exception as _exc:                      # not installed / unsupported Python
    mp = mp_python = mp_vision = None
    MEDIAPIPE_IMPORT_ERROR = _exc

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AGE_PROTO = os.path.join(SCRIPT_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(SCRIPT_DIR, "age_net.caffemodel")

# MediaPipe face model (~3.7 MB). Downloaded automatically on first run.
LANDMARK_MODEL_PATH = os.path.join(SCRIPT_DIR, "face_landmarker.task")
LANDMARK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

CAMERA_INDEX = 0
WINDOW_NAME = "ERP Lab Kiosk"

# --- Kiosk flow --------------------------------------------------------------
SMILING_FRAMES_REQUIRED = 5      # consecutive smiling frames needed to order
TRIGGER_COOLDOWN_SEC = 30        # wait time between two orders
FACE_LOST_GRACE_SEC = 0.7        # a short detection drop-out does not reset the customer
NEW_PERSON_JUMP = 1.0            # face moved more than this many face-widths between
                                 # frames -> treated as a different person
HINT_AFTER_SEC = 6               # show "remove mask / sunglasses" hint after this long

# --- Face rules --------------------------------------------------------------
MIN_FACE_AREA_FRAC = 0.090       # face box must cover this fraction of the frame
MIN_ROI_PIX = 40                 # minimum face width/height in pixels
MAX_FACES = 4                    # faces tracked per frame
BLOCK_MULTIPLE_PEOPLE = True     # pause if 2 similar-sized faces are inside the box
SECOND_FACE_RATIO = 0.5          # 2nd face must be >= 50 % of the largest to block
# The face box is derived from the face mesh so that it matches the box the old
# Haar detector produced (same size/centre the age model was tuned with).
FACE_BOX_SCALE = 1.11
FACE_BOX_SHIFT_Y = -0.08

# --- Smile -------------------------------------------------------------------
SMILE_THRESHOLD = 0.40           # 0..1  higher = stricter, lower = more sensitive
SMILE_SMOOTHING_FRAMES = 5       # average of the last N frames (stops flicker)

# --- Age ---------------------------------------------------------------------
AGE_CROP_SCALES = (1.1, 1.3)     # face crops (relative to the face box) fed to the model
AGE_USE_MIRROR = True            # also feed mirrored crops (more stable)
AGE_INFER_EVERY_N_FRAMES = 3     # run the age model every N frames while analysing
                                 # (increase on a slow PC, decrease for faster lock)
AGE_WINDOW = 8                   # number of recent age predictions that are averaged
AGE_MIN_SAMPLES = 5              # predictions needed before an age can be locked
ADULT_PROB_MIN = 0.60            # P(adult) >= this  -> adult
CHILD_PROB_MAX = 0.40            # P(adult) <= this  -> child
UNCERTAIN_TIMEOUT_SEC = 8        # still in between after this long -> safe choice = child
                                 # (set to 0 to keep analysing forever instead)
MIN_FACE_BRIGHTNESS = 30         # 0..255 average face brightness; darker = "too dark"

AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']
MINOR_BUCKET_COUNT = 4           # first 4 buckets (up to 20) count as child
AGE_MEAN = (78.4263377603, 87.7689143744, 114.895847746)

DRINK_ADULT = "Hugo Cocktail"
DRINK_CHILD = "Lemonade"

# --- ERP / Node-RED ----------------------------------------------------------
ERP_API_POST_URL = "http://localhost:1880/wserpbar"
ERP_API_HEADERS = {"Content-Type": "application/json"}


# =============================================================================
# Face data + detectors
# =============================================================================
@dataclass
class Face:
    """A detected face as a square box (centre + side) and a smile score 0..1."""
    cx: float
    cy: float
    side: float
    smile: float = 0.0

    def box(self, frame_w, frame_h):
        """Clamped integer box (x, y, w, h) inside the frame."""
        x1 = max(0, int(round(self.cx - self.side / 2)))
        y1 = max(0, int(round(self.cy - self.side / 2)))
        x2 = min(frame_w, int(round(self.cx + self.side / 2)))
        y2 = min(frame_h, int(round(self.cy + self.side / 2)))
        return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def ensure_landmark_model():
    """Return the path of the MediaPipe model, downloading it once if needed."""
    if os.path.exists(LANDMARK_MODEL_PATH):
        return LANDMARK_MODEL_PATH
    try:
        print("Downloading MediaPipe face model (one time, ~3.7 MB)...")
        tmp_path = LANDMARK_MODEL_PATH + ".part"
        with urllib.request.urlopen(LANDMARK_MODEL_URL, timeout=30) as resp, \
                open(tmp_path, "wb") as out:
            out.write(resp.read())
        os.replace(tmp_path, LANDMARK_MODEL_PATH)
        print("Model downloaded.")
        return LANDMARK_MODEL_PATH
    except Exception as exc:
        print("Could not download the MediaPipe model:", exc)
        print("Download it manually and save it next to this script as "
              "'face_landmarker.task':")
        print("   ", LANDMARK_MODEL_URL)
        return None


class MediaPipeDetector:
    """Faces + smile score from the MediaPipe Face Landmarker (robust to glasses/masks)."""
    name = "MediaPipe"

    def __init__(self, model_path):
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=MAX_FACES,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._last_ts = 0

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(self._last_ts + 1, int(time.monotonic() * 1000))   # must increase
        self._last_ts = ts
        result = self._landmarker.detect_for_video(mp_image, ts)

        faces = []
        for i, landmarks in enumerate(result.face_landmarks):
            pts = np.array([(p.x * w, p.y * h) for p in landmarks])
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            mesh_h = max(y2 - y1, 1.0)

            smile = 0.0
            if i < len(result.face_blendshapes):
                scores = {c.category_name: c.score for c in result.face_blendshapes[i]}
                smile = (scores.get("mouthSmileLeft", 0.0) +
                         scores.get("mouthSmileRight", 0.0)) / 2.0

            faces.append(Face(cx=float((x1 + x2) / 2),
                              cy=float((y1 + y2) / 2 + FACE_BOX_SHIFT_Y * mesh_h),
                              side=float(FACE_BOX_SCALE * mesh_h),
                              smile=float(smile)))
        return faces

    def close(self):
        self._landmarker.close()


class HaarDetector:
    """Fallback (old method): Haar face + Haar smile cascade. Less reliable."""
    name = "Haar (fallback)"

    def __init__(self):
        self._face = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._smile = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml")
        if self._face.empty() or self._smile.empty():
            raise SystemExit("Error loading Haar cascades.")

    def detect(self, frame_bgr):
        gray = cv2.equalizeHist(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
        rects = self._face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                            minSize=(MIN_ROI_PIX, MIN_ROI_PIX))
        faces = []
        for (x, y, w, h) in rects:
            lower_half = gray[y + h // 2:y + h, x:x + w]
            smiles = self._smile.detectMultiScale(lower_half, scaleFactor=1.8,
                                                  minNeighbors=20)
            faces.append(Face(cx=x + w / 2, cy=y + h / 2, side=float(max(w, h)),
                              smile=1.0 if len(smiles) > 0 else 0.0))
        return faces

    def close(self):
        pass


def create_detector():
    """MediaPipe if possible, otherwise the Haar fallback (with a loud warning)."""
    if MEDIAPIPE_IMPORT_ERROR is not None:
        print("WARNING: MediaPipe is not available (%s)." % MEDIAPIPE_IMPORT_ERROR)
    else:
        model_path = ensure_landmark_model()
        if model_path:
            try:
                return MediaPipeDetector(model_path)
            except Exception as exc:
                print("WARNING: could not start MediaPipe:", exc)
    print("*" * 70)
    print("WARNING: using the OLD Haar detectors (less accurate smile/face detection).")
    print("Fix: pip install mediapipe   (and allow the one-time model download)")
    print("*" * 70)
    return HaarDetector()


# =============================================================================
# Age estimation
# =============================================================================
def _square_crop(frame, cx, cy, side):
    """Square crop centred on (cx, cy); parts outside the frame are edge-padded."""
    side = max(int(round(side)), 8)
    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x2, y2 = x1 + side, y1 + side
    h, w = frame.shape[:2]
    sx1, sy1, sx2, sy2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    if sx2 <= sx1 or sy2 <= sy1:
        return None
    crop = frame[sy1:sy2, sx1:sx2]
    pad = (sy1 - y1, y2 - sy2, sx1 - x1, x2 - sx2)          # top, bottom, left, right
    if any(pad):
        crop = cv2.copyMakeBorder(crop, *pad, cv2.BORDER_REPLICATE)
    return crop


class AgeEstimator:
    """Caffe age network. Returns the 8 bucket probabilities for one face."""

    def __init__(self, proto, model):
        self.net = cv2.dnn.readNetFromCaffe(proto, model)

    def predict(self, frame_bgr, face):
        crops = []
        for scale in AGE_CROP_SCALES:
            crop = _square_crop(frame_bgr, face.cx, face.cy, face.side * scale)
            if crop is None:
                continue
            crop = cv2.resize(crop, (227, 227))
            crops.append(crop)
            if AGE_USE_MIRROR:
                crops.append(cv2.flip(crop, 1))
        if not crops:
            return None
        blob = cv2.dnn.blobFromImages(crops, 1.0, (227, 227), AGE_MEAN, swapRB=False)
        self.net.setInput(blob)
        return self.net.forward().mean(axis=0)               # average over all crops


@dataclass
class AgeDecision:
    locked: bool = False
    age_class: str = "unknown"      # 'adult' | 'child' | 'unknown'
    bucket: str = "Unknown"
    p_adult: float = 0.0
    uncertain: bool = False         # True if 'child' was chosen only as the safe default
    samples: int = 0


def decide_age(samples, uncertain_seconds=0.0):
    """Turn a window of probability vectors into a (possibly locked) decision."""
    n = len(samples)
    if n == 0:
        return AgeDecision()
    avg = np.mean(np.asarray(samples), axis=0)
    p_adult = float(avg[MINOR_BUCKET_COUNT:].sum())
    decision = AgeDecision(samples=n, p_adult=p_adult,
                           bucket=AGE_BUCKETS[int(avg.argmax())])
    if n < AGE_MIN_SAMPLES:
        return decision

    if p_adult >= ADULT_PROB_MIN:
        age_class = "adult"
    elif p_adult <= CHILD_PROB_MAX:
        age_class = "child"
    elif UNCERTAIN_TIMEOUT_SEC and uncertain_seconds >= UNCERTAIN_TIMEOUT_SEC:
        age_class = "child"                     # model unsure -> safe choice
        decision.uncertain = True
    else:
        return decision                         # still analysing

    lo, hi = ((MINOR_BUCKET_COUNT, len(AGE_BUCKETS)) if age_class == "adult"
              else (0, MINOR_BUCKET_COUNT))
    decision.bucket = AGE_BUCKETS[lo + int(avg[lo:hi].argmax())]
    decision.age_class = age_class
    decision.locked = True
    return decision


def face_brightness(gray, box):
    """Average brightness (0..255) of the central part of the face box."""
    x, y, w, h = box
    region = gray[y + h // 4: y + 3 * h // 4, x + w // 4: x + 3 * w // 4]
    return float(region.mean()) if region.size else 0.0


# =============================================================================
# ERP
# =============================================================================
def send_order_to_erp(drink, age_class):
    payload = {"name": f"{age_class} | {drink}"}
    try:
        print("Order initiated for ERP submission...")
        r = requests.post(ERP_API_POST_URL, json=payload,
                          headers=ERP_API_HEADERS, timeout=5)
        if r.status_code == 200:
            print("ERP API response:", r.text)
            return True
        print("ERP POST failed. Status:", r.status_code, "Body:", r.text)
        return False
    except requests.RequestException as e:
        print("ERP POST exception:", e)
        return False


# =============================================================================
# Drawing helpers
# =============================================================================
def put_text(img, text, org, scale=0.6, color=(255, 255, 255), thickness=2,
             font=cv2.FONT_HERSHEY_SIMPLEX):
    """Text with a dark outline so it stays readable on any background."""
    cv2.putText(img, text, org, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


def put_text_fitted(img, text, x, y, max_width, scale=0.65, color=(255, 255, 255)):
    """Like put_text, but shrinks the text so it fits into max_width."""
    while scale > 0.35:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        if tw <= max_width:
            break
        scale -= 0.05
    put_text(img, text, (max(5, x), y), scale, color)


def show_result_screen(cap, success, drink):
    """Full-screen feedback after an order. Returns False if the user pressed 'q'."""
    start = time.time()
    duration = 5 if success else 3
    while time.time() - start < duration:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        if success:
            overlay = frame.copy()
            cv2.rectangle(overlay, (int(fw * 0.05), int(fh * 0.33)),
                          (int(fw * 0.95), int(fh * 0.71)), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            cv2.putText(frame, "ORDER SUCCESSFUL!", (int(fw * 0.1), int(fh * 0.45)),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 4)
            cv2.putText(frame, "Suggested:", (int(fw * 0.1), int(fh * 0.56)),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (140, 200, 255), 2)
            cv2.putText(frame, f"Drink: {drink}", (int(fw * 0.13), int(fh * 0.62)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, "Next customer please...", (int(fw * 0.1), int(fh * 0.80)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (180, 255, 255), 2)
        else:
            cv2.putText(frame, "ORDER FAILED", (int(fw * 0.08), int(fh * 0.45)),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 4)
            cv2.putText(frame, "Please try again", (int(fw * 0.12), int(fh * 0.55)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            return False
    return True


# =============================================================================
# Kiosk logic (one call of process() = one camera frame)
# =============================================================================
@dataclass
class Seen:
    """A face seen in the current frame plus how it relates to the guide box."""
    face: Face
    box: tuple
    area: int
    frac: float
    in_box: bool
    large_enough: bool


class Kiosk:
    def __init__(self, detector, age_estimator):
        self.detector = detector
        self.age = age_estimator
        self.last_trigger_time = 0.0
        self.frame_idx = 0
        self.last_status = ""           # message currently shown to the customer
        self.reset_customer()

    # ---- state ------------------------------------------------------------
    def reset_customer(self):
        self.age_samples = deque(maxlen=AGE_WINDOW)
        self.smile_hist = deque(maxlen=SMILE_SMOOTHING_FRAMES)
        self.smile_counter = 0
        self.decision = AgeDecision()
        self.last_target = None          # (cx, cy, side) of the previous frame
        self.last_target_time = None
        self.uncertain_since = None
        self.locked_since = None

    # ---- main per-frame entry point -----------------------------------------
    def process(self, frame_raw):
        """Returns (annotated_frame, order_or_None)."""
        frame = cv2.flip(frame_raw, 1)
        fh, fw = frame.shape[:2]
        now = time.time()
        self.frame_idx += 1

        # red guide box (centered, same size as before)
        box_w, box_h = int(fw * 0.5), int(fh * 0.6)
        bx1, by1 = (fw - box_w) // 2, (fh - box_h) // 2
        bx2, by2 = bx1 + box_w, by1 + box_h

        # 1) detect all faces and describe them relative to the guide box
        try:
            faces = self.detector.detect(frame)
        except Exception as exc:
            print("Face detection error:", exc)
            faces = []
        seen = []
        for f in faces:
            x, y, w, h = f.box(fw, fh)
            area = w * h
            frac = area / float(fw * fh)
            in_box = bx1 <= f.cx <= bx2 and by1 <= f.cy <= by2
            large = frac >= MIN_FACE_AREA_FRAC and w >= MIN_ROI_PIX and h >= MIN_ROI_PIX
            seen.append(Seen(f, (x, y, w, h), area, frac, in_box, large))

        # 2) choose the target: only faces INSIDE the box, largest first
        candidates = sorted([s for s in seen if s.in_box], key=lambda s: -s.area)
        target = candidates[0] if candidates else None
        multiple = (BLOCK_MULTIPLE_PEOPLE and len(candidates) >= 2 and
                    candidates[1].area >= SECOND_FACE_RATIO * candidates[0].area)

        order = None
        status, status_color = "", (255, 255, 255)

        if target is None:
            self._face_lost(now)
            if seen:
                status, status_color = "Please step into the RED box", (0, 165, 255)
            else:
                status = "Stand inside the RED box and smile"
        elif multiple:
            self.reset_customer()
            status, status_color = "One person at a time, please", (0, 165, 255)
        else:
            order, status, status_color = self._update_target(frame, target, now)

        self.last_status = status
        self.last_target_info = target

        # 3) draw everything
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
        for s in seen:                                   # faces that are ignored
            if s is not target:
                x, y, w, h = s.box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 150, 150), 1)
        if target is not None:
            self._draw_target(frame, target, multiple)
        if not seen:
            put_text(frame, "No face detected", (50, fh - 30), 0.7, (0, 0, 255))
        put_text_fitted(frame, status, bx1 - 20, by2 + 30, fw - 20, 0.65, status_color)

        # debug info (top left, so it never collides with the status line)
        in_box_txt = target.in_box if target else False
        size_txt = target.frac if target else 0.0
        smooth = (sum(self.smile_hist) / len(self.smile_hist)) if self.smile_hist else 0.0
        put_text(frame, f"In box: {in_box_txt}", (10, 18), 0.5, thickness=1)
        put_text(frame, f"Face size: {size_txt:.3f}", (10, 36), 0.5, thickness=1)
        put_text(frame, f"Age locked: {self.decision.locked}  (P adult {self.decision.p_adult:.2f})",
                 (10, 54), 0.5, thickness=1)
        put_text(frame, f"Smile: {smooth:.2f}   [{self.detector.name}]",
                 (10, 72), 0.5, thickness=1)
        return frame, order

    # ---- helpers ------------------------------------------------------------
    def _face_lost(self, now):
        self.smile_counter = 0
        if (self.last_target_time is not None and
                now - self.last_target_time > FACE_LOST_GRACE_SEC):
            self.reset_customer()

    def _update_target(self, frame, target, now):
        face = target.face

        # a big jump between frames = somebody else stepped in
        if self.last_target is not None:
            pcx, pcy, pside = self.last_target
            if math.hypot(face.cx - pcx, face.cy - pcy) > NEW_PERSON_JUMP * max(face.side, pside):
                self.reset_customer()
        self.last_target = (face.cx, face.cy, face.side)
        self.last_target_time = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        too_dark = face_brightness(gray, target.box) < MIN_FACE_BRIGHTNESS

        # ---- age: sample only when the face is big enough and bright enough
        if target.large_enough and not too_dark:
            interval = AGE_INFER_EVERY_N_FRAMES * (4 if self.decision.locked else 1)
            if not self.age_samples or self.frame_idx % interval == 0:
                try:
                    probs = self.age.predict(frame, face)
                except Exception as exc:
                    print("Age estimation error:", exc)
                    probs = None
                if probs is not None:
                    self.age_samples.append(probs)

        unc_secs = (now - self.uncertain_since) if self.uncertain_since else 0.0
        self.decision = decide_age(self.age_samples, unc_secs)
        d = self.decision
        if not d.locked and d.samples >= AGE_MIN_SAMPLES:
            if self.uncertain_since is None:
                self.uncertain_since = now
        elif d.locked and not d.uncertain:
            self.uncertain_since = None
        if d.locked:
            if self.locked_since is None:
                self.locked_since = now
        else:
            self.locked_since = None

        # ---- smile
        self.smile_hist.append(face.smile)
        smiling = (sum(self.smile_hist) / len(self.smile_hist)) >= SMILE_THRESHOLD
        if smiling and target.in_box and target.large_enough and d.locked:
            self.smile_counter += 1
        else:
            self.smile_counter = 0

        # ---- trigger
        order = None
        trigger_allowed = (now - self.last_trigger_time) > TRIGGER_COOLDOWN_SEC
        if self.smile_counter >= SMILING_FRAMES_REQUIRED and trigger_allowed:
            order = {
                "drink": DRINK_ADULT if d.age_class == "adult" else DRINK_CHILD,
                "age_bucket": d.bucket,
                "age_class": d.age_class,
                "uncertain": d.uncertain,
                "timestamp": now,
            }
            self.last_trigger_time = now

        # ---- status message for the customer
        if not target.large_enough:
            status, color = "Come a little closer", (0, 165, 255)
        elif too_dark:
            status, color = "Too dark - please step into the light", (0, 165, 255)
        elif not d.locked:
            status, color = "Analysing... please hold still", (255, 255, 0)
        elif not trigger_allowed:
            wait = int(TRIGGER_COOLDOWN_SEC - (now - self.last_trigger_time)) + 1
            status, color = f"Next order possible in {wait}s", (200, 200, 200)
        elif smiling:
            status, color = "Great! Hold your smile...", (0, 255, 0)
        elif self.locked_since and now - self.locked_since > HINT_AFTER_SEC:
            status, color = "Smile to order (remove mask/sunglasses)", (0, 165, 255)
        else:
            status, color = "Smile to confirm your order", (200, 200, 200)
        return order, status, color

    def _draw_target(self, frame, target, multiple):
        x, y, w, h = target.box
        ok = target.in_box and target.large_enough and not multiple
        box_color = (0, 255, 0) if ok else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

        d = self.decision
        if d.locked:
            label = f"Age: {d.bucket} ({d.age_class}{' ?' if d.uncertain else ''})"
        else:
            label = "Age: analysing..."
        put_text(frame, label, (x, max(15, y - 10)), 0.55, (255, 255, 0))

        smooth = (sum(self.smile_hist) / len(self.smile_hist)) if self.smile_hist else 0.0
        smiling = smooth >= SMILE_THRESHOLD
        put_text(frame, "Smiling" if smiling else "Not Smiling", (x, y + h + 20), 0.55,
                 (120, 255, 120) if smiling else (200, 200, 255))


# =============================================================================
# Main
# =============================================================================
def run_kiosk():
    try:
        age_estimator = AgeEstimator(AGE_PROTO, AGE_MODEL)
        print("Age model loaded.")
    except Exception as e:
        print("Error loading age model:", e)
        raise SystemExit("Missing age model files.")

    detector = create_detector()
    print("Face/smile detector:", detector.name)
    kiosk = Kiosk(detector, age_estimator)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Cannot open camera.")
        detector.close()
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print("Starting ERP Lab Kiosk (press 'q' to quit, 'r' to reset).")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame not available.")
                break

            display, order = kiosk.process(frame)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                kiosk.reset_customer()
                print("Manual reset.")

            if order:
                print("Suggestion locked:", order)
                success = send_order_to_erp(order["drink"], order["age_class"])
                print("ERP send status:", success)
                if not success:
                    kiosk.last_trigger_time = 0.0      # allow an immediate retry
                if not show_result_screen(cap, success, order["drink"]):
                    break
                kiosk.reset_customer()
                print("Order completed. Ready for next customer." if success
                      else "Order failed. Ready for retry.")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("Kiosk closed.")


if __name__ == "__main__":
    run_kiosk()
