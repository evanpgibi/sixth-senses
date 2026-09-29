"""Minimal colour-card reader.
Upload a photo of the printed card with a sample in the box.
Returns the sample colour (RGB + hex) normalised against the 9 reference patches.

Run:  pip install flask opencv-python-headless numpy
      python app.py   ->  http://localhost:5000
"""
import base64
import cv2
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Canonical card size = size of the original card image
W, H = 877, 536

# Patch x-ranges (y is 38..234) in canonical coordinates
PATCH_X = [(32, 116), (124, 207), (215, 298), (305, 388), (396, 479),
           (487, 570), (577, 661), (668, 751), (759, 842)]
PATCH_Y = (38, 234)

# True RGB of the 9 patches, measured from the card artwork
REF = np.array([
    [243, 243, 241], [122, 122, 120], [52, 52, 52],
    [161, 63, 64], [90, 146, 81], [56, 61, 145],
    [231, 225, 203], [113, 74, 127], [37, 23, 49],
], dtype=np.float64)

BOX = (250, 300, 626, 491)        # x0, y0, x1, y1 of the "place strip here" box
BOX_MARGIN = 10                   # ignore the printed outline
PAPER_REGION = (40, 320, 230, 480)  # blank paper, used as the "white" reference
DELTA_E_MIN = 18                  # how far from paper colour counts as "not white"
WHITE_CHROMA = 14                 # bright pixels with less chroma than this are treated as white


# ---------- geometry ----------
def order_points(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)  # tl,tr,br,bl


def find_cards(img):
    """Return candidate 4-corner outlines (largest first). The right one is chosen later
    by checking how well the 9 patches fit, so a wrong outline can't win."""
    h, w = img.shape[:2]
    scale = 800 / max(h, w)
    small = cv2.resize(img, None, fx=scale, fy=scale)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 40, 120), np.ones((3, 3), np.uint8), iterations=2)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    total = small.shape[0] * small.shape[1]
    found = []
    for m in (edges, otsu, 255 - otsu):
        cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            area = cv2.contourArea(c)
            if area < 0.08 * total or area >= 0.995 * total:
                continue
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                found.append((area, order_points(approx) / scale))
    found.sort(key=lambda t: -t[0])
    quads = []
    for _, q in found:
        if all(np.abs(q - o).max() > 0.03 * max(h, w) for o in quads):  # drop near-duplicates
            quads.append(q)
    quads = quads[:6]
    quads.append(np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32))  # photo is already the card
    return quads


def warp_quad(img, quad):
    top = np.linalg.norm(quad[1] - quad[0])
    left = np.linalg.norm(quad[3] - quad[0])
    if left > top:  # portrait outline: shift corners so the card ends up landscape
        quad = np.roll(quad, -1, axis=0)
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_AREA)


# ---------- colour ----------
def patch_medians(img):
    out = []
    for x0, x1 in PATCH_X:
        r = img[PATCH_Y[0] + 30:PATCH_Y[1] - 30, x0 + 15:x1 - 15]
        out.append(np.median(r.reshape(-1, 3), axis=0)[::-1])  # BGR -> RGB
    return np.array(out, dtype=np.float64)


def to_linear(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c):
    c = np.clip(c, 0, 1)
    return 255 * np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def fit_correction(measured, ref, lam=1e-3):
    """Affine map (3x4) in linear RGB: measured -> ref. Ridge-regularised least squares."""
    X = np.hstack([to_linear(measured), np.ones((len(measured), 1))])
    Y = to_linear(ref)
    A = np.linalg.solve(X.T @ X + lam * np.eye(4), X.T @ Y)
    return A


def apply_correction(rgb, A):
    flat = rgb.reshape(-1, 3).astype(np.float64)
    X = np.hstack([to_linear(flat), np.ones((len(flat), 1))])
    return to_srgb(X @ A).reshape(rgb.shape)


def lab(rgb_u8):
    return cv2.cvtColor(rgb_u8.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)


def analyse(img):
    quads = find_cards(img)
    best = None
    for qi, quad in enumerate(quads):
        w0 = warp_quad(img, quad)
        for flip in (False, True):  # card may be upside down
            w2 = cv2.rotate(w0, cv2.ROTATE_180) if flip else w0
            m = patch_medians(w2)
            A = fit_correction(m, REF)
            err = np.sqrt(np.mean((apply_correction(m, A) - REF) ** 2))
            if best is None or err < best[0]:
                best = (err, w2, m, A, qi < len(quads) - 1)
    fit_rmse, warped, meas, A, found = best

    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    corrected = apply_correction(rgb, A).clip(0, 255).astype(np.uint8)

    # paper colour (the "white" to compare the sample against)
    px0, py0, px1, py1 = PAPER_REGION
    paper = np.median(corrected[py0:py1, px0:px1].reshape(-1, 3), axis=0)

    x0, y0, x1, y1 = BOX
    m_ = BOX_MARGIN
    box = corrected[y0 + m_:y1 - m_, x0 + m_:x1 - m_]
    lab_box = lab(box)
    lab_paper = lab(paper.reshape(1, 1, 3))[0, 0]
    dE = np.linalg.norm(lab_box - lab_paper, axis=2)
    chroma = np.hypot(lab_box[..., 1] - 128, lab_box[..., 2] - 128)
    whiteish = (lab_box[..., 0] * 100 / 255 > 70) & (chroma < WHITE_CHROMA)  # bright and neutral = white
    mask = ((dE > DELTA_E_MIN) & ~whiteish).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    result = None
    if n > 1:
        k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        comp = (labels == k).astype(np.uint8)
        core = cv2.erode(comp, np.ones((9, 9), np.uint8))
        if core.sum() < 50:
            core = comp
        sel = core.astype(bool)
        col = np.median(box[sel], axis=0).astype(int)
        raw_box = rgb[y0 + m_:y1 - m_, x0 + m_:x1 - m_]
        raw_col = np.median(raw_box[sel], axis=0).astype(int)
        result = dict(rgb=tuple(int(v) for v in col),
                      hex="#%02X%02X%02X" % tuple(int(v) for v in col),
                      raw=tuple(int(v) for v in raw_col),
                      pixels=int(sel.sum()))
        # overlay for the preview
        ov = corrected.copy()
        full = np.zeros(corrected.shape[:2], bool)
        full[y0 + m_:y1 - m_, x0 + m_:x1 - m_] = sel
        ov[full] = (0.5 * ov[full] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        corrected_preview = ov
    else:
        corrected_preview = corrected.copy()
    cv2.rectangle(corrected_preview, (x0, y0), (x1, y1), (255, 0, 0), 2)

    return dict(result=result, found=found, fit_rmse=float(fit_rmse),
                warped=warped, corrected=cv2.cvtColor(corrected_preview, cv2.COLOR_RGB2BGR),
                measured=meas)


def b64(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode()


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Colour card reader</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;color:#222}
img{max-width:100%;border:1px solid #ccc}
.sw{display:inline-block;width:4rem;height:4rem;border:1px solid #888;vertical-align:middle}
code{background:#f0f0f0;padding:.1em .3em}
.warn{color:#a40}
</style>
<h1>Colour card reader</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=image accept="image/*" required>
  <button>Analyse</button>
</form>
{% if error %}<p class=warn>{{ error }}</p>{% endif %}
{% if out %}
  {% if out.result %}
    <h2>Normalised colour</h2>
    <p><span class=sw style="background:{{ out.result.hex }}"></span>
       &nbsp;<b>{{ out.result.hex }}</b> &nbsp; rgb({{ out.result.rgb|join(', ') }})</p>
    <p>Before normalising: rgb({{ out.result.raw|join(', ') }}) &middot; sample pixels: {{ out.result.pixels }}</p>
  {% else %}
    <p class=warn>Could not read a colour. Make sure the whole card with all 9 patches is visible and the sample is inside the box, then retake.</p>
  {% endif %}
  <p>Reference fit error (RMS, 0-255 scale): <b>{{ '%.1f'|format(out.fit_rmse) }}</b>
     {% if out.fit_rmse > 12 %}<span class=warn>&mdash; high, retake the photo (glare, shadow or blur)</span>{% endif %}<br>
     {% if not out.found %}<span class=warn>Card edges not found; used the whole photo as the card.</span>{% endif %}</p>
  <h3>Cropped card</h3><img src="data:image/jpeg;base64,{{ out.warped_b64 }}">
  <h3>Normalised card (green = detected sample, blue = sample box)</h3><img src="data:image/jpeg;base64,{{ out.corrected_b64 }}">
{% endif %}
"""


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(PAGE)
    data = np.frombuffer(request.files["image"].read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return render_template_string(PAGE, error="Could not read that image.")
    # keep processing fast on big phone photos
    if max(img.shape[:2]) > 2400:
        s = 2400 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    r = analyse(img)
    if r["fit_rmse"] > 25:
        r["result"] = None
    r["warped_b64"] = b64(r.pop("warped"))
    r["corrected_b64"] = b64(r.pop("corrected"))
    return render_template_string(PAGE, out=r)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)