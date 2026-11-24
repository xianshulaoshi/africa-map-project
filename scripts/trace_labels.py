#!/usr/bin/env python3
"""Step 5b of the source-map trace: lettering.

Every label found by trace_text.py (OCR + hand corrections) is fitted to the
ink of the scan: the text is rendered with the font of its style and the
size, tracking and position that correlate best with the ink are kept.  The
same fit ranks candidate fonts (`trace_labels.py rank`).

Output: data/trace_labels_text.json
  fonts   [[css family, file], ...]              embedded later (subset)
  styles  {id: {ff, i, w, c, h, hw, ls}}         text style table for the page
  labels  [[x, y, t, style, size, track, anchor, rot], ...]   baseline x / y
  dots    [[x, y], ...]
"""
import glob
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data')
FDIR = os.path.join(ROOT, 'scratch', 'redraw', 'fonts')
SS = 4

LUM = np.asarray(Image.open(os.path.join(ROOT, 'source-map.png')).convert('RGB')).astype(float) @ \
    np.array([0.299, 0.587, 0.114])
MED = ndi.median_filter(LUM, 3)
_rgb = np.asarray(Image.open(os.path.join(ROOT, 'source-map.png')).convert('RGB')).astype(float)
CHROMA = _rgb.max(-1) - _rgb.min(-1)

# font id -> (file, weight axis value or None)
FONTS = {}
for p in sorted(glob.glob(os.path.join(FDIR, '*.ttf'))):
    FONTS[os.path.basename(p)[:-4]] = p


def font_of(fid, wght=None):
    path = FONTS[fid]
    f = ImageFont.truetype(path, 20)
    vals = []
    try:
        for a in f.get_variation_axes():
            nm = a['name'].decode() if isinstance(a['name'], bytes) else str(a['name'])
            if 'Weight' in nm and wght is not None:
                vals.append(min(max(wght, a['minimum']), a['maximum']))
            else:
                vals.append(a['default'])
    except Exception:
        vals = []
    return path, vals


def render(font, text, size, track, W, H, ox, oy, italic=0.0):
    path, var = font
    f = ImageFont.truetype(path, size * SS)
    if var:
        f.set_variation_by_axes(var)
    img = Image.new('L', (max(1, W * SS), max(1, H * SS)), 0)
    d = ImageDraw.Draw(img)
    x, yb = ox * SS, oy * SS
    for ch in text:
        if ch != ' ':
            d.text((x, yb), ch, fill=255, font=f, anchor='ls')
        x += f.getlength(ch) + track * SS
    if italic:
        img = img.transform(img.size, Image.AFFINE, (1, -italic, italic * yb, 0, 1, 0), resample=Image.BICUBIC)
    return np.asarray(img.resize((W, H), Image.BOX)).astype(float) / 255


def cap_em(font, italic):
    a = render(font, 'H', 40, 0, 60, 60, 5, 50, italic)
    rows = np.nonzero((a > 0.5).any(1))[0]
    return (50 - rows[0]) / 40 if len(rows) else 0


def ink_width(font, text, size, track, italic):
    a = render(font, text, size, track, 700, 60, 5, 45, italic)
    cols = np.nonzero((a > 0.5).any(0))[0]
    return (cols[-1] - cols[0] + 1) if len(cols) else 0


def label_ink(box):
    """Darkness map of a window (box grown by 3 px) and the mask of the ink
    that belongs to the label: the ink components whose centre lies inside
    the box (1 px tolerance)."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    X0, Y0, X1, Y1 = max(0, x0 - 3), max(0, y0 - 3), min(LUM.shape[1], x1 + 3), min(LUM.shape[0], y1 + 3)
    lum, med = LUM[Y0:Y1, X0:X1], MED[Y0:Y1, X0:X1]
    bg = ndi.maximum_filter(med, size=7)
    # ink is dark and not saturated: a dark blue fill next to a lighter one
    # is as dark as lettering but far more colourful
    chroma = ndi.median_filter(CHROMA[Y0:Y1, X0:X1], 3)
    dark = ((bg - lum) > 45) & ((lum < 78) | (chroma < 60))
    if dark.sum() < 3:
        return None
    ink_level = np.percentile(med[dark], 15)
    D = np.clip((bg - lum) / np.maximum(bg - ink_level, 1), 0, 1)
    lab, n = ndi.label(dark, structure=np.ones((3, 3), bool))
    keep = []
    for i, sl in enumerate(ndi.find_objects(lab)):
        cy = (sl[0].start + sl[0].stop) / 2 + Y0
        cx = (sl[1].start + sl[1].stop) / 2 + X0
        gw, gh = sl[1].stop - sl[1].start, sl[0].stop - sl[0].start
        area = (lab[sl] == i + 1).sum()
        thin_line = max(gw, gh) >= 7 and area < 0.35 * gw * gh
        dot = 2 <= gw <= 3 and 2 <= gh <= 3 and area >= 0.75 * gw * gh      # city dot
        inside = x0 - 1 <= cx <= x1 + 1 and y0 - 1 <= cy <= y1 + 1
        if inside and not thin_line and not dot:
            keep.append(i + 1)                 # centred in the box: no neighbour line, dot or leader
    if not keep:
        return None
    M = np.isin(lab, keep)
    return D, M, (X0, Y0)


def text_mask(M):
    """Ink of the label = the components overlapping the band of rows that
    carries the lettering (a leader line, a border or an accent passing
    through the window lies outside that band)."""
    for _ in range(2):
        rows = M.sum(1).astype(float)
        if rows.max() == 0:
            return M
        band = np.nonzero(rows >= 0.3 * rows.max())[0]
        lab, n = ndi.label(M, structure=np.ones((3, 3), bool))
        keep = [i + 1 for i, sl in enumerate(ndi.find_objects(lab))
                if sl[0].start < band[-1] - 1 and sl[0].stop > band[0] + 1]
        M = np.isin(lab, keep)
    return M


def measure(M):
    cols = np.nonzero(M.any(0))[0]
    rows = M.sum(1).astype(float)
    body = np.nonzero(rows >= 0.3 * rows.max())[0]
    y1 = int(body[-1]) + 1
    lab, n = ndi.label(M, structure=np.ones((3, 3), bool))
    hs = [sl[0].stop - sl[0].start for sl in ndi.find_objects(lab)
          if abs(sl[0].stop - y1) <= 1 and sl[0].start >= body[0] - 1]
    return dict(h=max(hs) if hs else float(body[-1] - body[0] + 1), w=int(cols[-1] - cols[0] + 1),
                x0=int(cols[0]), y1=y1)


def fit(font, italic, text, box, sizes=None, tracks=None, coarse=False, wide=False):
    li = label_ink(box)
    if li is None:
        return None
    D, M, (X0, Y0) = li
    M = text_mask(M)
    if not M.any():
        return None
    D = D * ndi.binary_dilation(M, structure=np.ones((3, 3), bool))
    H, W = D.shape
    m = measure(M)
    ce = cap_em(font, italic)
    if ce <= 0:
        return None
    s0 = m['h'] / ce
    if sizes is None:
        sizes = s0 * np.array([0.93, 0.97, 1.0, 1.03, 1.07])
    Db = ndi.gaussian_filter(D, 0.6)
    best = None
    n = max(1, len(text) - 1)
    for size in sizes:
        wn = ink_width(font, text, size, 0, italic)
        tr0 = (m['w'] - wn) / n
        for track in (tr0 + np.array([-0.12, 0, 0.12]) if tracks is None else tracks):
            track = min(max(track, -0.45), 7.9)
            step = 0.5 if coarse else 0.25
            rng = 3.0 if wide else 1.0
            for dx in np.arange(-rng, rng + 0.01, step):
                for dy in np.arange(-rng, rng + 0.01, step):
                    a = render(font, text, size, track, W, H, m['x0'] + dx, m['y1'] + dy, italic)
                    ab = ndi.gaussian_filter(a, 0.6)
                    num = ((Db - Db.mean()) * (ab - ab.mean())).sum()
                    den = np.sqrt(((Db - Db.mean()) ** 2).sum() * ((ab - ab.mean()) ** 2).sum())
                    sc = num / den if den else 0
                    if best is None or sc > best[0]:
                        best = (sc, m['x0'] + dx + X0, m['y1'] + dy + Y0, size, track)
    if best is None:
        return None
    # refine the position around the best (step 0.1 px)
    _, bx, by, size, track = best
    bx, by = bx - X0, by - Y0
    for dx in np.arange(-0.45, 0.46, 0.1):
        for dy in np.arange(-0.45, 0.46, 0.1):
            a = render(font, text, size, track, W, H, bx + dx, by + dy, italic)
            ab = ndi.gaussian_filter(a, 0.6)
            num = ((Db - Db.mean()) * (ab - ab.mean())).sum()
            den = np.sqrt(((Db - Db.mean()) ** 2).sum() * ((ab - ab.mean()) ** 2).sum())
            sc = num / den if den else 0
            if sc > best[0]:
                best = (sc, bx + dx + X0, by + dy + Y0, size, track)
    # colour of the lettering: median of its darkest pixels
    yy, xx = np.nonzero(M)
    sub = _rgb[yy + Y0, xx + X0]
    dark = sub.sum(1) <= np.percentile(sub.sum(1), 25)
    colour = '#%02x%02x%02x' % tuple(int(v) for v in np.median(sub[dark], 0))
    return dict(score=best[0], x=best[1], y=best[2], size=best[3], track=best[4], h=m['h'], w=m['w'],
                colour=colour)


def fit_rotated(font, italic, text, box, rot, wide=False):
    """Rotated lettering: the page is turned by the label angle so that the
    line is horizontal, fitted, and the baseline start is mapped back."""
    if abs(rot) < 1:
        return fit(font, italic, text, box, wide=wide)
    global LUM, MED, _rgb, CHROMA
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    long_, short = max(x1 - x0, y1 - y0), min(x1 - x0, y1 - y0)
    half = int(long_ / 2 + short + 6)
    X0, Y0 = int(cx) - half, int(cy) - half
    X1, Y1 = int(cx) + half + 1, int(cy) + half + 1
    page = Image.open(os.path.join(ROOT, 'source-map.png')).convert('RGB')
    crop = page.crop((X0, Y0, X1, Y1)).rotate(float(rot), resample=Image.BICUBIC, expand=True)
    a = math.radians(float(rot))
    ca, sa = math.cos(a), math.sin(a)
    ci = ((X1 - X0) / 2.0, (Y1 - Y0) / 2.0)          # centre of the crop (rotation centre)
    cr = (crop.width / 2.0, crop.height / 2.0)       # centre of the turned crop
    # image centre of the label -> turned crop coordinates
    dx, dy = cx - X0 - ci[0], cy - Y0 - ci[1]
    px, py = ca * dx + sa * dy + cr[0], -sa * dx + ca * dy + cr[1]
    w, h = (short, long_) if abs(rot) > 45 else (long_, short)
    rgb = np.asarray(crop).astype(float)
    keep = (LUM, MED, _rgb, CHROMA)
    LUM = rgb @ np.array([0.299, 0.587, 0.114])
    MED = ndi.median_filter(LUM, 3)
    _rgb = rgb
    CHROMA = rgb.max(-1) - rgb.min(-1)
    try:
        res = fit(font, italic, text, (px - w / 2 - 1, py - h / 2 - 1, px + w / 2 + 1, py + h / 2 + 1),
                  sizes=None, tracks=None, coarse=True, wide=wide)
    finally:
        LUM, MED, _rgb, CHROMA = keep
    if res is None:
        return None
    # baseline start back to the page
    ex, ey = res['x'] - cr[0], res['y'] - cr[1]
    res['x'] = ca * ex - sa * ey + ci[0] + X0
    res['y'] = sa * ex + ca * ey + ci[1] + Y0
    return res


def rank():
    samples = [
        ('T', 'TUNISIE', (280, 79, 314, 89)), ('T', 'NIGERIA', (239, 304, 277, 313)),
        ('T', 'ABYSSINIE', (518, 327, 563, 335)), ('T', 'HAUTE-VOLTA', (117, 283, 175, 293)),
        ('TW', 'ALGÉRIE', (222, 154, 258, 164)), ('TW', 'MAROC', (143, 98, 178, 107)),
        ('C', 'Casablanca', (112, 78, 154, 88)), ('C', 'Alexandrie', (406, 107, 445, 117)),
        ('C', 'Moçamedes', (250, 522, 302, 534)), ('C', 'Brazzaville', (278, 424, 317, 434)),
        ('R', 'OCÉAN', (82, 375, 125, 385)), ('R', 'ATLANTIQUE', (64, 394, 141, 405)),
    ]
    res = {}
    for fid in sorted(FONTS):
        weights = (400, 500, 600, 700) if '[' in fid else (None,)
        for wg in weights:
            try:
                font = font_of(fid, wg)
                ImageFont.truetype(font[0], 20).set_variation_by_axes(font[1]) if font[1] else None
            except Exception:
                break
            for st, text, box in samples:
                try:
                    r = fit(font, 0.0, text, box, coarse=True)
                except Exception:
                    r = None
                if r:
                    res.setdefault(st, {}).setdefault('%s@%s' % (fid, wg), []).append(r['score'])
    for st, d in res.items():
        rows = sorted(((np.mean(v), k) for k, v in d.items()), reverse=True)
        print(st, '  '.join('%s:%.3f' % (k, s) for s, k in rows[:14]))


# ---- styles ---------------------------------------------------------------------------
FCOND, FCONDB, FQ, FASAP = 'IBMPlexSansCondensed-Regular', 'IBMPlexSansCondensed-SemiBold', 'Questrial-Regular', 'AsapCondensed-Bold'
STYLES = {                       # id: (italic shear, halo width in px or 0)
    'T':  dict(f=FASAP,  i=0.0,  halo=0),     # territory names: condensed bold caps
    'TW': dict(f=FCOND,  i=0.0,  halo=0),     # MAROC / ALGÉRIE: condensed regular caps, tracked
    'P':  dict(f=FCOND,  i=0.0,  halo=0),     # names in brackets
    'C':  dict(f=FQ,     i=0.0,  halo=1.3),   # cities
    'CI': dict(f=FQ,     i=0.25, halo=1.3),   # cities, italic
    'R':  dict(f=FQ,     i=0.0,  halo=1.2),   # letter-spaced capitals (region titles, oceans)
    'W':  dict(f=FQ,     i=0.25, halo=1.2),   # water names, italic
    'WI': dict(f=FQ,     i=0.0,  halo=1.2),   # water names, upright
    'D':  dict(f=FQ,     i=0.0,  halo=1.0),   # 20°O, 10°S ...
}
T_TEXTS = set((
    'TUNISIE NIGER TCHAD SOUDAN FRANÇAIS HAUTE-VOLTA SÉNÉGAL GAMBIE GUINÉE PORT. SIERRA LEONE CÔTE D\'IVOIRE GOLD COAST '
    'NIGERIA ÉGYPTE LIBYE ÉRYTHRÉE ANGLO- EGYPTIEN SOMALIE BRIT. ITALIENNE OUGANDA AFRIQUE- ORIENTALE BRITANNIQUE '
    'ALLEMANDE ALLEMAND CAMEROUN GABON MOYEN-CONGO CONGO BELGE ANGOLA RHODÉSIE DU NORD DU SUD SUD-OUEST AFRICAIN '
    'BECHUANALAND SWAZILAND BASUTOLAND MADAGASCAR UNION D\'AFRIQUE DU SUD (TRANSVAAL) (ZULULAND) (NATAL) (KAFFRARIA) '
    '(COLONIE DU CAP) (SULTANAT DE ZANZIBAR) (EMPIRE OTTOMAN) (ÉTHIOPIE) SAHARA ESP. RÍO DEL ORO MAROC ESP. '
    'AFRIQUE ÉQUATORIALE FRANÇAISE NYASSALAND MOZAMBIQUE DAHOMEY TOGO ABYSSINIE SOMALIE').split())
TW_TEXTS = {'MAROC', 'ALGÉRIE'}
R_TEXTS = {'AFRIQUE-OCCIDENTALE FRANÇAISE', 'OCÉAN', 'ATLANTIQUE', 'INDIEN'}
W_TEXTS = {'Mer Méditerranée', 'Mer Rouge', 'Golfe Persique', 'Nil', 'Niger', 'Congo', 'Zambèze',
           'Tropique du Cancer', 'Tropique du Capricorne', 'Équateur'}


# legend lettering, measured on the scan: text -> (x, baseline y, style, size, tracking)
LEGEND = {
    'Territoires sous domination': (13.5, 438.6, 'H', 9.0, 0.0),
    '1870': (19.0, 457.6, 'Y', 7.6, 0.2), '1900': (50.0, 457.6, 'Y', 7.6, 0.2), '1914': (81.5, 457.6, 'Y', 7.6, 0.2),
    'française': (119.5, 472.6, 'L', 9.4, 0.1), 'britannique': (119.5, 489.0, 'L', 9.4, 0.1),
    'allemande': (119.5, 505.6, 'L', 9.4, 0.1), 'belge': (119.5, 521.6, 'L', 9.4, 0.1),
    'portugaise': (119.5, 538.4, 'L', 9.4, 0.1), 'espagnole': (119.5, 555.0, 'L', 9.4, 0.1),
    'italienne': (119.5, 571.0, 'L', 9.4, 0.1), 'turque': (119.5, 587.0, 'L', 9.4, 0.1),
    'des Boers /': (119.5, 598.0, 'L', 9.4, 0.1), 'Afrikaners': (119.5, 607.6, 'L', 9.4, 0.1),
    'État indépendant': (119.5, 620.6, 'L', 9.4, 0.1), 'Frontière en 1914': (119.5, 642.6, 'L', 9.4, 0.1),
    'Frontière en 1870': (119.5, 664.6, 'L', 9.4, 0.1), '(NATAL)': (78.5, 685.6, 'LI', 8.6, 0.1),
    'Nom de territoire': (119.5, 685.6, 'L', 9.4, 0.1), 'en 1870': (119.5, 696.0, 'L', 9.4, 0.1),
}
STYLES['L'] = dict(f=FQ, i=0.0, halo=0)       # legend entries
STYLES['LI'] = dict(f=FQ, i=0.25, halo=0)     # (NATAL) sample

FORCE_STYLE = {'Territoires sous domination': 'H', '1870': 'Y', '1900': 'Y', '1914': 'Y', '(NATAL)': 'P',
               "L'EXPANSION COLONIALE EN AFRIQUE (1870-1914)": 'TT', '(NATAL)': 'LI'}
LEGEND_TEXTS = set(LEGEND)
# lettering whose fit is unreliable (touching a border or a dot): x, baseline y, style, size, tracking, rotation
EXPLICIT = {
    'DAHOMEY': (213.6, 336.5, 'T', 9.4, 0.3, -90),       # vertical, reading upward, east of Togo
    'GOLD': (164.6, 317.8, 'T', 7.8, 0.5, 0), 'COAST': (163.6, 327.0, 'T', 7.8, 0.5, 0),
    'Mafeking': (377.6, 628.8, 'C', 7.3, 0.3, 0), 'TOGO': (203.6, 333.0, 'T', 8.4, 0.5, -90),
}
DEFAULT_SIZE = {'H': (8.6, 0.0), 'Y': (7.2, 0.4), 'TT': (20.5, 0.0), 'T': (8.6, 0.0), 'TW': (8.0, 0.5), 'P': (7.4, 0.2),
                'C': (7.3, 0.3), 'CI': (7.3, 0.3), 'R': (8.6, 2.0), 'W': (6.6, 0.8), 'WI': (6.6, 0.8), 'D': (6.4, 0.3)}
STYLES['H'] = dict(f='Lato-Bold', i=0.0, halo=0)      # legend heading
STYLES['Y'] = dict(f='Lato-Bold', i=0.0, halo=0, c='#ffffff')   # years printed on the first swatch
STYLES['TT'] = dict(f=FASAP, i=0.0, halo=0)            # title (ultra condensed: scaled horizontally)


def style_of(text):
    if text in LEGEND_TEXTS and text != '(NATAL)':
        return LEGEND[text][2]
    if text in FORCE_STYLE:
        return FORCE_STYLE[text]
    if text in T_TEXTS:
        return 'T'
    if text in TW_TEXTS:
        return 'TW'
    if text in R_TEXTS:
        return 'R'
    if text in W_TEXTS:
        return 'W'
    if text.startswith('('):
        return 'P'
    if '°' in text:
        return 'D'
    return 'C'


def clean(text):
    t = text.replace('•', '').replace('30*E', '30°E').strip()
    if t.endswith(' o'):
        t = t[:-2]
    return t.strip()


def medians(labels):
    out = {}
    for st in STYLES:
        good = [l for l in labels if l['s'] in (st, 'CI') and l['score'] >= 0.7 and l['s'] == st]
        if good:
            out[st] = (float(np.median([l['size'] for l in good])), float(np.median([l['tr'] for l in good])))
    return out


def wide_rotated(font, italic, text, box, rot):
    """Rotated label whose angle is only known roughly: search the angle and
    a wider position range."""
    best = None
    for a_ in (rot - 8, rot - 4, rot, rot + 4, rot + 8):
        x0, y0, x1, y1 = box
        h = y1 - y0
        w = x1 - x0
        try:
            r = fit_rotated(font, italic, text, (x0 - 2, y0 - 2, x1 + 2, y1 + 2), a_, wide=True)
        except Exception:
            r = None
        if r and (best is None or r['score'] > best[0]['score']):
            best = (r, a_)
    return best


def band_box(box, rot):
    """Position of a line of lettering from the ink inside its box, without
    sorting the components: the band of rows carrying most of the ink is the
    line; its bottom row is the baseline, its first / last ink column the
    ends of the text."""
    if abs(rot) > 45:
        return None
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    li = label_ink((x0 - 1, y0 - 1, x1 + 1, y1 + 1))
    if li is None:
        return None
    D, M, (X0, Y0) = li
    # keep only components that are not line-like and not dots
    lab, n = ndi.label(M, structure=np.ones((3, 3), bool))
    keep = []
    for i, sl in enumerate(ndi.find_objects(lab)):
        gw, gh = sl[1].stop - sl[1].start, sl[0].stop - sl[0].start
        area = (lab[sl] == i + 1).sum()
        if (max(gw, gh) >= 7 and area < 0.35 * gw * gh) or (2 <= gw <= 3 and 2 <= gh <= 3 and area >= 0.75 * gw * gh) or gh > 13:
            continue
        keep.append(i + 1)
    M = np.isin(lab, keep)
    if not M.any():
        return None
    rows = M.sum(1).astype(float)
    band = np.nonzero(rows >= 0.35 * rows.max())[0]
    # contiguous part of the band around its busiest row
    peak = int(rows.argmax())
    lo = peak
    while lo - 1 >= 0 and rows[lo - 1] >= 0.35 * rows.max():
        lo -= 1
    hi = peak
    while hi + 1 < len(rows) and rows[hi + 1] >= 0.35 * rows.max():
        hi += 1
    sub = M[lo:hi + 1]
    cols = np.nonzero(sub.any(0))[0]
    if not len(cols):
        return None
    return dict(x=int(cols[0]) + X0, x1=int(cols[-1]) + X0 + 1, top=int(lo) + Y0, bottom=int(hi) + Y0 + 1)


def fallback(text, st, box, rot, med):
    """No reliable fit: place the label on its box with the median size of
    its style, tracking adjusted to the width of the box."""
    font = font_of(STYLES[st]['f'])
    italic = STYLES[st]['i']
    size, tr = med.get(st, DEFAULT_SIZE.get(st, (7.0, 0.3)))
    x0, y0, x1, y1 = box
    if st in ('R', 'TW', 'T'):
        n = max(1, len(text) - 1)
        tr = min(max((x1 - x0 - ink_width(font, text, size, 0, italic)) / n, -0.3), 6)
    caps = st in ('R', 'TW', 'T', 'P', 'D', 'H', 'Y', 'TT') or text.isupper()
    desc = 0 if caps or not any(c in text for c in 'gjpqy,;()') else 1.5
    bb = band_box(box, rot)
    if bb and 3 <= bb['bottom'] - bb['top'] <= 11:
        # the band is the label only if it is about as long as the label
        nat = ink_width(font, text, size, 0, italic)
        ratio = (bb['x1'] - bb['x']) / max(nat, 1)
        if (st == 'R' and 0.9 <= ratio <= 2.6) or (st != 'R' and 0.7 <= ratio <= 1.4):
            if st != 'R':
                size = float(np.clip(size * ratio, size * 0.8, size * 1.25))
            x0, x1, y1 = bb['x'], bb['x1'], bb['bottom']
            if st in ('R', 'TW', 'T'):
                n = max(1, len(text) - 1)
                tr = min(max((x1 - x0 - ink_width(font, text, size, 0, italic)) / n, -0.3), 6)
    if abs(rot) > 45:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        L = ink_width(font, text, size, tr, italic)
        a_ = math.radians(rot)
        x, y = cx - math.cos(a_) * L / 2 - math.sin(a_) * 0, cy + math.sin(a_) * L / 2
        return dict(x=round(x, 2), y=round(y, 2), score=0.0, colour='#1d1d1f', size=size, tr=tr)
    return dict(x=round(x0 + 0.5, 2), y=round(y1 - 1 - desc, 2), score=0.0, colour='#1d1d1f', size=size, tr=tr)


def main(refit=False):
    data = json.load(open(os.path.join(OUT, 'trace_text.json')))
    fkeys = {FCOND: font_of(FCOND), FCONDB: font_of(FCONDB), FQ: font_of(FQ), FASAP: font_of(FASAP),
             'Lato-Bold': font_of('Lato-Bold')}
    tpath = os.path.join(OUT, 'trace_labels_fit.json')
    prev = json.load(open(tpath))['labels'] if (refit and os.path.exists(tpath)) else []
    by_text = {l['t']: l for l in prev}
    med = medians(prev) if prev else {}
    out, problems = [], []
    for r in data['labels']:
        text = clean(r['t'])
        st = style_of(text)
        box = (r['x0'], r['y0'], r['x1'], r['y1'])
        font = fkeys[STYLES[st]['f']]
        italic = STYLES[st]['i']
        old = by_text.get(text)
        if text in EXPLICIT:
            ex, ey, est, esize, etr, erot = EXPLICIT[text]
            out.append(dict(x=ex, y=ey, t=text, s=est, size=esize, tr=etr, rot=erot, score=1.0, colour='#1d1d1f', sx=1.0))
            continue
        if text in LEGEND and (r['rot'] == 0 or text == '(NATAL)'):
            lx, ly, lst, lsize, ltr = LEGEND[text]
            out.append(dict(x=lx, y=ly, t=text, s=lst, size=lsize, tr=ltr, rot=0, score=1.0, colour='#1d1d1f', sx=1.0))
            continue
        if refit and old and old['score'] >= 0.5:
            out.append(old)
            continue

        def both(it, bb=box, rr=r['rot'], wide=False):
            if wide:
                return wide_rotated(font, it, text, bb, rr)
            return fit_rotated(font, it, text, bb, rr)

        res, used_rot = None, r['rot']
        if r['rot'] and not refit:
            res = both(italic)
            if isinstance(res, tuple):
                res, used_rot = res
        elif r['rot'] and refit:
            got = both(italic, wide=True)
            if got:
                res, used_rot = got
        elif st in ('C', 'P', 'W'):
            r0, r1 = both(0.0), both(0.25)
            if r0 is None and r1 is None:
                pass
            elif r0 is None or (r1 and r1['score'] > r0['score'] + 0.012):
                res, italic = r1, 0.25
                st = 'CI' if st == 'C' else st
            else:
                res = r0
                st = 'WI' if st == 'W' else st
        else:
            res = both(italic)
        score = res['score'] if res else -1
        ref = med.get(st, DEFAULT_SIZE.get(st, (7.0, 0.3)))[0]
        if res is not None and 'tr' not in res:
            res['tr'] = res.get('track', 0.0)
        sane = res is not None and score >= 0.5 and 0.65 * ref <= res['size'] <= 1.5 * ref
        if not sane:
            if res is not None and score >= 0.5:
                # a good fit but an implausible size: fall back to the default size in place
                res['size'] = ref
                problems.append((text, 'size', round(float(res['size']), 1)))
            else:
                res = fallback(text, st, box, used_rot, med)
                problems.append((text, round(float(score), 2)))
        sx = 1.0
        if st == 'TT' and res.get('w'):
            sx = round(res['w'] / max(ink_width(font, text, res['size'], 0, 0.0), 1), 3)
        elif st == 'TT':
            sx = 0.77
        out.append(dict(x=round(float(res['x']), 2), y=round(float(res['y']), 2), t=text, s=st,
                        size=round(float(res.get('size', 7.0)), 2), tr=round(float(res.get('tr', 0.0)), 2),
                        rot=used_rot, score=round(float(res['score']), 3), colour=res.get('colour', '#1d1d1f'),
                        sx=sx))
    json.dump({'labels': out, 'dots': data['dots']}, open(tpath, 'w'), ensure_ascii=False)
    print('%d labels, %d placed on their box (fit unreliable)' % (len(out), len(problems)))
    print(' '.join('%s:%s' % (p[0][:10], p[-1]) for p in problems))


def restyle(styles):
    """Refit only the labels of the given styles (after a font change)."""
    data = json.load(open(os.path.join(OUT, 'trace_text.json')))
    tpath = os.path.join(OUT, 'trace_labels_fit.json')
    cur = json.load(open(tpath))
    done = {l['t']: l for l in cur['labels']}
    boxes = {clean(r['t']): r for r in data['labels']}
    n = 0
    for l in cur['labels']:
        if l['s'] not in styles or l['t'] not in boxes:
            continue
        r = boxes[l['t']]
        st = l['s']
        font = font_of(STYLES[st]['f'])
        res = fit_rotated(font, STYLES[st]['i'], l['t'], (r['x0'], r['y0'], r['x1'], r['y1']), r['rot'])
        if res is None or res['score'] < 0.5:
            continue
        l.update(x=round(float(res['x']), 2), y=round(float(res['y']), 2), size=round(float(res['size']), 2),
                 tr=round(float(res['track']), 2), score=round(float(res['score']), 3), colour=res['colour'],
                 sx=round(res['w'] / max(ink_width(font, l['t'], res['size'], 0, 0.0), 1), 3) if st == 'TT' else 1.0)
        n += 1
    json.dump(cur, open(tpath, 'w'), ensure_ascii=False)
    print('restyled %d labels' % n)


def retexts(texts):
    """Refit the labels carrying the given texts (several labels may share a
    text: they are matched to their boxes in order of appearance)."""
    data = json.load(open(os.path.join(OUT, 'trace_text.json')))
    tpath = os.path.join(OUT, 'trace_labels_fit.json')
    cur = json.load(open(tpath))
    queue = {}
    for r in data['labels']:
        if clean(r['t']) in texts:
            queue.setdefault(clean(r['t']), []).append(r)
    n = 0
    for l in cur['labels']:
        if l['t'] not in texts or l['t'] in EXPLICIT or not queue.get(l['t']):
            continue
        r = queue[l['t']].pop(0)
        st = style_of(l['t'])
        if l['t'] in LEGEND:
            continue
        font = font_of(STYLES[st]['f'])
        res = fit_rotated(font, STYLES[st]['i'], l['t'], (r['x0'], r['y0'], r['x1'], r['y1']), r['rot'])
        if res is None:
            continue
        res['tr'] = res.get('track', 0.0)
        l.update(s=st, x=round(float(res['x']), 2), y=round(float(res['y']), 2), size=round(float(res['size']), 2),
                 tr=round(float(res['tr']), 2), rot=r['rot'], score=round(float(res['score']), 3), colour=res['colour'])
        n += 1
    json.dump(cur, open(tpath, 'w'), ensure_ascii=False)
    print('refitted %d labels' % n)


if __name__ == '__main__':
    if sys.argv[1:] == ['rank']:
        rank()
    elif sys.argv[1:] == ['refit']:
        main(refit=True)
    elif sys.argv[1:3] == ['styles']:
        restyle(sys.argv[3])
    elif sys.argv[1:2] == ['retexts']:
        retexts(set(sys.argv[2:]))
    else:
        main()
