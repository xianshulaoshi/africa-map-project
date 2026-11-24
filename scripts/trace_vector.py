#!/usr/bin/env python3
"""Step 2 of the source-map trace: vectorise the class map.

  * every flat-land class becomes polygons (with holes), traced at the 0.5
    level of a lightly smoothed mask, so outlines are sub-pixel exact but
    free of staircase steps;
  * the land polygons are everything that is not open sea (lakes are holes,
    the water-shade layer shows through them);
  * pixels are assigned to territories by a watershed on the "barrier" image
    (height of white border lines, faint lines and light lines), flooded from
    one marker per territory.  A marker sits on the most interior pixel of
    the largest patch of the territory's colours inside its TERR_BOX_SEEDS
    box, or on the explicit TERR_SEEDS points.  Shade boundaries inside a
    territory have no barrier, any border line has, whatever the size of the
    territory.  Where the scan draws no internal border at all (French West
    Africa) the split between the labelled territories is the geodesic
    middle between their markers: invisible, it only decides what a click
    selects.

Output: data/trace_geo.json (polygons as SVG path data, image pixels).
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from shapely.geometry import Polygon
from skimage import measure
from skimage.segmentation import watershed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data')
SEA, WHITE = -1, -2
SMOOTH = 0.7          # gaussian pre-smoothing of the mask (px)
TOL = 0.3             # Douglas-Peucker tolerance (px)
LINE_RESP = 22.0      # ridge response above which a pixel is on a light line
LINE_MIN_PX = 10      # shorter light structures are noise or frontier dots
GAP = 6               # half length (px) of the straight segments used to bridge gaps
RIDGE_BARRIER = 0.35  # barrier height of a light line
MIN_SEED_BARRIER = 0.3

CLASSES = ['ctx', 'fr-l', 'fr-m', 'fr-d', 'uk-l', 'uk-m', 'uk-d', 'de-l', 'de-m', 'be',
           'pt-l', 'pt-m', 'pt-d', 'es-l', 'es-m', 'es-d', 'it-l', 'it-m', 'ot', 'indep', 'boer']

# territory id -> (x0, y0, x1, y1) box of source pixels around the territory
TERR_BOX_SEEDS = {
    'maroc': (50, 90, 205, 200), 'algerie': (150, 120, 300, 232), 'tunisie': (283, 63, 320, 105),
    'maroc-esp': (160, 70, 196, 82), 'sahara-esp': (70, 128, 140, 152), 'rio-oro': (58, 152, 138, 192),
    'mauritanie': (70, 150, 200, 215), 'senegal': (50, 205, 118, 262), 'gambie': (55, 262, 82, 270),
    'guinee-pt': (58, 268, 76, 286), 'soudan-fr': (120, 255, 225, 295), 'haute-volta': (118, 283, 172, 297),
    'niger': (230, 232, 300, 275), 'tchad': (330, 235, 420, 290), 'guinee': (86, 299, 99, 313),
    'sierra': (74, 304, 90, 324), 'liberia': (121, 322, 143, 344), 'cote-ivoire': (145, 298, 186, 345),
    'gold-coast': (187, 292, 203, 335), 'togo': (203, 293, 210, 340), 'dahomey': (214, 295, 226, 340),
    'nigeria': (226, 268, 320, 352), 'cameroun': (263, 335, 345, 378), 'gabon': (270, 385, 330, 418),
    'moyen-congo': (270, 412, 350, 432), 'oubangui': (345, 310, 415, 360), 'congo-belge': (345, 350, 465, 520),
    'libye': (284, 102, 399, 195), 'egypte': (400, 108, 515, 195), 'soudan-eg': (355, 195, 520, 375),
    'erythree': (505, 235, 558, 285), 'abyssinie': (485, 280, 597, 364), 'somalie-fr': (556, 286, 570, 299),
    'somalie-br': (568, 293, 618, 326), 'somalie-it': (565, 288, 634, 389), 'ouganda': (455, 372, 502, 410),
    'bea': (502, 365, 560, 440), 'gea': (445, 405, 540, 500), 'zanzibar': (524, 443, 538, 453),
    'angola': (285, 440, 400, 556), 'gswa': (292, 547, 376, 652), 'walvis-bay': (327, 595, 339, 605),
    'bechuanaland': (372, 560, 452, 640), 'afriquesud': (320, 650, 452, 702), 'rhodesie-n': (385, 495, 478, 548),
    'rhodesie-s': (430, 545, 480, 592), 'mozambique': (470, 495, 545, 625), 'nyassaland': (474, 470, 502, 520),
    'swaziland': (456, 622, 476, 638), 'zululand': (448, 636, 472, 650), 'natal': (444, 649, 470, 663),
    'basutoland': (427, 653, 446, 667), 'kaffraria': (415, 650, 431, 663),
    'madagascar': (550, 505, 625, 632), 'transvaal': (428, 600, 460, 632), 'orange': (405, 632, 445, 662),
}
# explicit seed points (source pixels) where a box would reach into a
# neighbouring territory; they override TERR_BOX_SEEDS
TERR_SEEDS = {
    'maroc': [(100, 165), (140, 112)], 'algerie': [(240, 100), (262, 186), (300, 168), (238, 150)],
    'mauritanie': [(100, 186), (170, 198), (130, 214)], 'senegal': [(84, 236)],
    'soudan-fr': [(150, 268), (196, 280), (206, 258)], 'haute-volta': [(146, 292), (128, 296)],
    'niger': [(262, 250), (290, 246), (272, 262)], 'tchad': [(368, 262), (352, 238), (396, 244)],
    'gabon': [(300, 400)], 'moyen-congo': [(300, 422)], 'oubangui': [(355, 340), (390, 345)],
}
# territory id -> power prefix(es) of the colour classes it is drawn with
TERR_POWER = {
    'maroc': ['fr'], 'algerie': ['fr'], 'tunisie': ['fr'], 'maroc-esp': ['es'], 'sahara-esp': ['es'],
    'rio-oro': ['es'], 'mauritanie': ['fr'], 'senegal': ['fr'], 'guinee-pt': ['pt'], 'gambie': ['uk'],
    'soudan-fr': ['fr'], 'haute-volta': ['fr'], 'niger': ['fr'], 'tchad': ['fr'], 'guinee': ['uk'],
    'sierra': ['uk'], 'liberia': ['indep'], 'cote-ivoire': ['fr'], 'gold-coast': ['uk'], 'togo': ['de'],
    'dahomey': ['uk'], 'nigeria': ['uk'], 'cameroun': ['de'], 'gabon': ['fr'], 'moyen-congo': ['fr'],
    'oubangui': ['fr'], 'congo-belge': ['be'], 'libye': ['ot', 'it'], 'egypte': ['uk'], 'soudan-eg': ['uk'],
    'erythree': ['it'], 'abyssinie': ['indep'], 'somalie-fr': ['fr'], 'somalie-br': ['uk'], 'somalie-it': ['it'],
    'ouganda': ['uk'], 'bea': ['uk'], 'gea': ['de'], 'zanzibar': ['uk'], 'angola': ['pt'], 'gswa': ['de'],
    'walvis-bay': ['uk'], 'bechuanaland': ['uk'], 'afriquesud': ['uk'], 'rhodesie-n': ['uk'], 'rhodesie-s': ['uk'],
    'mozambique': ['pt'], 'nyassaland': ['uk'], 'madagascar': ['fr'], 'transvaal': ['boer'], 'orange': ['boer'],
    'swaziland': ['uk'], 'zululand': ['uk'], 'natal': ['uk'], 'basutoland': ['uk'], 'kaffraria': ['uk'],
    'canaries': ['es'], 'madeira': ['pt'], 'cabo-verde': ['pt'], 'sao-tome': ['pt'], 'reunion': ['fr'],
    'maurice': ['uk'], 'seychelles': ['uk'], 'comores': ['fr'],
}
# small islands: every land pixel of their colour inside the box belongs to them
TERR_BOXES = {
    'canaries': (78, 156, 106, 180), 'madeira': (66, 126, 86, 138), 'cabo-verde': (44, 186, 60, 202),
    'sao-tome': (252, 376, 270, 390), 'reunion': (548, 548, 558, 558), 'maurice': (584, 556, 596, 566),
    'seychelles': (630, 474, 646, 488), 'comores': (560, 484, 582, 500),
}


def signed_area(r):
    x, y = r[:, 0], r[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def mask_to_polygons(mask, min_area, smooth=SMOOTH, offset=(0, 0)):
    """Polygons of a binary mask: [[outer_ring, hole, ...], ...], image coords."""
    res = []
    lab, n = ndi.label(mask, structure=np.ones((3, 3), bool))
    for i, sl in enumerate(ndi.find_objects(lab)):
        if sl is None:
            continue
        m = lab[sl] == i + 1
        if m.sum() < min_area:
            continue
        pad = 3
        mp = np.pad(m, pad).astype(float)
        if smooth and m.sum() >= 15:
            mp = ndi.gaussian_filter(mp, smooth)
        cs = measure.find_contours(mp, 0.5)
        oy, ox = sl[0].start - pad + 0.5 + offset[1], sl[1].start - pad + 0.5 + offset[0]
        rings = [np.column_stack([c[:, 1] + ox, c[:, 0] + oy]) for c in cs]
        rings = [r for r in rings if len(r) >= 4 and abs(signed_area(r)) > 0.5]
        outers, holes = [], []
        for r in rings:
            top = r[np.lexsort((r[:, 0], r[:, 1]))[0]]          # topmost vertex
            inside = ndi.map_coordinates(mp, np.array([[top[1] - oy + 0.8], [top[0] - ox]]), order=1)[0] > 0.5
            (outers if inside else holes).append(r)
        polys = [[o] for o in outers]
        for h in holes:
            pt = Polygon(h).representative_point()
            cand = [p for p in polys if Polygon(p[0]).contains(pt)]
            if cand:
                min(cand, key=lambda p: Polygon(p[0]).area).append(h)
        res.extend(polys)
    return res


def num(v):
    s = ('%.1f' % v).rstrip('0').rstrip('.')
    return '0' if s in ('-0', '') else s


def join_nums(nums):
    """SVG path numbers with the minimal separators (a '-' or a leading '.'
    starts a new number by itself)."""
    out = ''
    for v in nums:
        out += (v if not out or v[0] in '-.' else ' ' + v)
    return out


def path_d(poly_rings, tol=TOL):
    """Compact SVG path: absolute M, relative l, one decimal, minimal separators."""
    parts = []
    for r in poly_rings:
        p = Polygon(r)
        if p.is_valid and tol:
            p = p.simplify(tol, preserve_topology=True)
        pts = list(p.exterior.coords)[:-1]
        if len(pts) < 3:
            continue
        nums = [num(pts[0][0]), num(pts[0][1])]
        px, py = pts[0]
        for x, y in pts[1:]:
            if x == px and y == py:
                continue
            nums += [num(x - px), num(y - py)]
            px, py = x, y
        parts.append('M' + join_nums(nums[:2]) + 'l' + join_nums(nums[2:]) + 'z')
    return ''.join(parts)


def light_lines(halozone):
    """Thin bright lines of the scan (white borders, even faint ones).

    Response = 2*L - L(left) - L(right) across the line, max over 24
    orientations, L = slightly smoothed luminance.  A half-white line scores
    ~50, a shade boundary (a plain blend) ~0, JPEG ringing < 15.  Lettering
    and its halo are excluded, specks are dropped, and the gaps left where a
    border runs under a label are bridged.
    """
    lum = ndi.gaussian_filter(
        np.asarray(Image.open(os.path.join(ROOT, 'source-map.png')).convert('L')).astype(float), 0.8)
    H, W = lum.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    best = np.full((H, W), -1e9)
    for k in range(24):
        th = np.pi * k / 24
        dx, dy = np.cos(th) * 1.1, np.sin(th) * 1.1
        a = ndi.map_coordinates(lum, [yy + dy, xx + dx], order=1, mode='nearest')
        b = ndi.map_coordinates(lum, [yy - dy, xx - dx], order=1, mode='nearest')
        best = np.maximum(best, 2 * lum - a - b)
    lines = (best > LINE_RESP) & ~halozone
    lab, n = ndi.label(lines, structure=np.ones((3, 3), bool))
    sizes = np.bincount(lab.ravel())
    keep = np.zeros(n + 1, bool); keep[1:] = sizes[1:] >= LINE_MIN_PX
    lines &= keep[lab]
    # bridge the gaps where a border runs under a label: close along straight
    # segments (so only a line continuing its own direction is bridged), and
    # only inside the halo of lettering
    if lines.any():
        gy, gx = np.mgrid[-GAP:GAP + 1, -GAP:GAP + 1].astype(float)
        for k in range(16):
            th = np.pi * k / 16
            along = gx * np.cos(th) + gy * np.sin(th)
            across = np.abs(-gx * np.sin(th) + gy * np.cos(th))
            se = (across <= 0.9) & (np.abs(along) <= GAP)
            if se.sum() < 3:
                continue
            lines |= ndi.binary_closing(lines, structure=se, border_value=0) & halozone
    return lines


def power(name):
    return name.split('-')[0]


# watershed domains: colours that can belong to one and the same territory
GROUPS = {'fr': ['fr'], 'uk': ['uk'], 'de': ['de'], 'es': ['es'], 'pt': ['pt'], 'be': ['be'],
          'it': ['it', 'ot'], 'indep': ['indep'], 'boer': ['boer']}


def allowed_classes(terr):
    pws = TERR_POWER[terr]
    return [i for i, c in enumerate(CLASSES) if any(c == pw or c.startswith(pw + '-') for pw in pws)]


def seed_from_box(box, ok):
    """Most interior pixel of the largest ok-patch inside the box."""
    x0, y0, x1, y1 = box
    sub = ok[y0:y1, x0:x1]
    lab, n = ndi.label(sub)
    if not n:
        return None
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    comp = lab == sizes.argmax()
    dt = ndi.distance_transform_edt(comp)
    sy, sx = np.unravel_index(dt.argmax(), dt.shape)
    return int(sx + x0), int(sy + y0), int(sizes.max())


# straight cuts (x0, y0, x1, y1) in source pixels closing gaps of faint
# borders where lettering hides the line; found by comparing the territory
# boundaries with the white lines of the scan
CUTS = [(278, 104, 278, 63)]          # Alger | Tunisie, hidden by "10°E" and "TUNISIE"


def draw_cuts(mask, cuts):
    from skimage.draw import line as sk_line
    for (x0, y0, x1, y1) in cuts:
        rr, cc = sk_line(y0, x0, y1, x1)
        rr = rr[(rr >= 0) & (rr < mask.shape[0])]
        cc = cc[(cc >= 0) & (cc < mask.shape[1])]
        mask[rr, cc] = True
    return mask


def label_territories(cls, loose, barrier, halozone):
    """Pixel map of territory ids (index into the returned name list, 0 = none)."""
    H, W = cls.shape
    loose = loose.copy()
    loose[draw_cuts(np.zeros((H, W), bool), CUTS)] = WHITE
    land = loose >= 0
    # quantised barrier: flooding happens level by level, and inside one level
    # the markers grow geodesically (first in, first out).  With the raw
    # values the noise of 1e-4 would decide who wins a flat area.
    B = np.round(barrier * 10) / 10
    light = light_lines(halozone)
    B[land & light & (B < RIDGE_BARRIER)] = RIDGE_BARRIER
    cls_power = np.array([power(c) for c in CLASSES])
    px_power = cls_power[np.clip(loose, 0, None)]

    names = [None] + sorted(TERR_POWER)
    idx = {t: i for i, t in enumerate(names) if t}
    markers = np.zeros((H, W), np.int32)
    for terr, seeds in TERR_SEEDS.items():
        for (x, y) in seeds:
            markers[y, x] = idx[terr]
    for terr, box in TERR_BOX_SEEDS.items():
        if terr in TERR_SEEDS:
            continue
        ok = np.isin(loose, allowed_classes(terr)) & (barrier < MIN_SEED_BARRIER)
        seed = seed_from_box(box, ok)
        if not seed:
            print('no patch found for %s in %s' % (terr, box))
            continue
        sx, sy, area = seed
        markers[sy, sx] = idx[terr]
    L = np.zeros((H, W), np.int32)
    for pws in GROUPS.values():
        own = [k for k, n in enumerate(names) if n and TERR_POWER[n][0] in pws]
        mk = np.where(np.isin(markers, own), markers, 0)
        if not mk.any():
            continue
        dom = land & np.isin(px_power, pws)
        lab = watershed(B, markers=mk, mask=dom, connectivity=1)
        L[lab > 0] = lab[lab > 0]
    # small islands by bounding box
    for terr, (bx0, by0, bx1, by1) in TERR_BOXES.items():
        m = np.zeros((H, W), bool); m[by0:by1, bx0:bx1] = True
        L[m & land & (L == 0) & np.isin(loose, allowed_classes(terr))] = idx[terr]
    # leftovers (islets, river slivers of another colour) take the territory
    # around them
    for _ in range(40):
        rest = land & (L == 0) & (loose != 0)
        if not rest.any():
            break
        ring = ndi.binary_dilation(L > 0, structure=np.ones((7, 7), bool)) & rest
        if not ring.any():
            break
        idxn = ndi.distance_transform_edt(L == 0, return_distances=False, return_indices=True)
        L[ring] = L[idxn[0][ring], idxn[1][ring]]
    left = land & (L == 0) & (loose != 0)
    if left.any():
        lab, n = ndi.label(left)
        for i, sl in enumerate(ndi.find_objects(lab)):
            a = int((lab[sl] == i + 1).sum())
            if a >= 150:
                print('unassigned land: %d px at x %d-%d y %d-%d' % (
                    a, sl[1].start, sl[1].stop, sl[0].start, sl[0].stop))
    return L, names, markers


def main():
    cls = np.load(os.path.join(OUT, 'trace_classes.npy'))
    loose = np.load(os.path.join(OUT, 'trace_loose.npy'))
    barrier = np.load(os.path.join(OUT, 'trace_barrier.npy'))
    H, W = cls.shape

    # open sea = water connected to the paper frame; small cut-off water specks
    # are handed to the nearest land class, bigger ones stay as lakes
    lab, n = ndi.label(cls == SEA)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    border = set(border[border > 0].tolist())
    sizes = np.bincount(lab.ravel())
    ocean = np.isin(lab, list(border))
    specks = (cls == SEA) & ~ocean & (sizes[lab] < 10)
    if specks.any():
        idx = ndi.distance_transform_edt(cls == SEA, return_distances=False, return_indices=True)
        cls = cls.copy()
        cls[specks] = cls[idx[0][specks], idx[1][specks]]
        ocean |= specks
    print('ocean %d px, lake px %d, water specks filled %d' % (
        ocean.sum(), ((cls == SEA) & ~ocean).sum(), specks.sum()))

    geo = {'W': W, 'H': H, 'land': [], 'patches': []}
    for rings in mask_to_polygons(~ocean, min_area=2):
        geo['land'].append(path_d(rings))

    L, names, markers = label_territories(cls, loose, barrier, np.load(os.path.join(OUT, 'trace_halo.npy')))
    np.save(os.path.join(OUT, 'trace_labels.npy'), L)
    per = {}
    for k, terr in enumerate(names):
        if not terr:
            continue
        per[terr] = int((L == k).sum())
        for ci, cname in enumerate(CLASSES):
            m = (L == k) & (cls == ci)
            if m.sum() < 2:
                continue
            for rings in mask_to_polygons(m, min_area=2):
                pg = Polygon(rings[0], rings[1:])
                x0, y0, x1, y1 = pg.bounds
                geo['patches'].append({'cls': cname, 'd': path_d(rings), 'terr': terr,
                                       'cx': round((x0 + x1) / 2, 1), 'cy': round((y0 + y1) / 2, 1)})
    # land outside Africa is one class of its own (not selectable)
    m = loose == 0
    if m.any():
        for rings in mask_to_polygons(m, min_area=2):
            geo['patches'].append({'cls': 'ctx', 'd': path_d(rings), 'terr': 'ctx'})
    print('land polys %d, patches %d' % (len(geo['land']), len(geo['patches'])))
    print('px per territory:', dict(sorted(per.items(), key=lambda t: -t[1])))

    # debug image: one colour per territory over the source, markers as dots
    rng = np.random.RandomState(5)
    cols = rng.randint(40, 255, (len(names), 3))
    vis = np.asarray(Image.open(os.path.join(ROOT, 'source-map.png')).convert('RGB')).copy()
    m = L > 0
    vis[m] = (0.6 * vis[m] + 0.4 * cols[L[m]]).astype(np.uint8)
    img = Image.fromarray(vis)
    dr = ImageDraw.Draw(img)
    cents = ndi.center_of_mass(np.ones_like(L), L, range(1, len(names)))
    for k, (cy, cx) in enumerate(cents, 1):
        if not np.isfinite(cy):
            continue
        dr.text((cx - 8, cy - 4), names[k], fill=(255, 255, 0))
    ys, xs = np.nonzero(markers)
    for y, x in zip(ys, xs):
        dr.ellipse([x - 1, y - 1, x + 1, y + 1], outline=(255, 0, 0))
    img.resize((W * 2, H * 2), Image.LANCZOS).save(os.path.join(ROOT, 'scratch', 'redraw', 'territories.png'))
    # review image: boundaries between territories in black over the scan
    rev = np.asarray(Image.open(os.path.join(ROOT, 'source-map.png')).convert('RGB')).copy()
    bnd = np.zeros((H, W), bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = L[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)]
        b = L[max(0, -dy):H + min(0, -dy), max(0, -dx):W + min(0, -dx)]
        m = (a != b) & ((a > 0) | (b > 0)) & ~(((a == 0) | (b == 0)) & (cls[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)] == SEA))
        bnd[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)][m] = True
    rev[bnd] = (0, 0, 0)
    Image.fromarray(rev).resize((W * 2, H * 2), Image.NEAREST).save(os.path.join(ROOT, 'scratch', 'redraw', 'review.png'))

    with open(os.path.join(OUT, 'trace_geo.json'), 'w') as f:
        json.dump(geo, f, ensure_ascii=False, separators=(',', ':'))
    print('trace_geo.json bytes', os.path.getsize(os.path.join(OUT, 'trace_geo.json')))


if __name__ == '__main__':
    main()
