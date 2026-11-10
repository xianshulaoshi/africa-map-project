#!/usr/bin/env python3
"""Mirror the original map's fill colors pixel-by-pixel.

For every territory:
1. Rasterize it into the original image's pixel space (global calibrated Mercator).
2. Find a local translation that best aligns its own power-family colors (regional
   scan-warp compensation), searching +-16px.
3. Classify every interior pixel into one of the territory's allowed shades
   (deep/mid/light per its power, + special shades). Unknown pixels (text, rivers,
   borders, hatching) get filled from nearest classified neighbors, then median-filtered.
4. Vectorize each shade mask into contours (marching squares) -> zone polygons.
5. Save measured zones (lon/lat) to data/measured_zones.json.

postprocess.py then uses these zones instead of hand-drawn boxes, and
verify_fill.py re-checks every pixel.
"""
import numpy as np, json, math
from PIL import Image, ImageDraw
from shapely.geometry import shape as shp_shape, Polygon as ShPoly
from shapely.ops import unary_union
from skimage import measure

BASE = '/Users/weiaiwen1/Desktop/my-space/map-project'
img = np.asarray(Image.open(BASE + '/source-map.png').convert('RGB')).astype(int)
H, W = img.shape[:2]

calib = json.load(open(BASE + '/data/calib.json'))
osx, osy, otx, oty = calib['t']
RAD = math.pi / 180
def mercY(lat): return math.log(math.tan(math.pi/4 + lat*RAD/2)) / RAD
def invMercY(y): return (2*math.atan(math.exp(y*RAD)) - math.pi/2) / RAD
def to_px(lon, lat): return (lon*osx + otx, oty - mercY(lat)*osy)
def to_ll(x, y): return ((x - otx)/osx, invMercY((oty - y)/osy))

# shade palettes per power: name -> hex + rgb
PAL = {
 'fr': [('deep', '#0c78b9', (0x0c,0x78,0xb9)), ('mid', '#4a97cd', (0x4a,0x97,0xcd)), ('light', '#8abbe3', (0x8a,0xbb,0xe3))],
 'uk': [('deep', '#a664a2', (0xa6,0x64,0xa2)), ('mid', '#c288b8', (0xc2,0x88,0xb8)), ('light', '#e2afce', (0xe2,0xaf,0xce))],
 'de': [('mid', '#689a7f', (0x68,0x9a,0x7f)), ('light', '#90bca5', (0x90,0xbc,0xa5))],
 'be': [('mid', '#d7af3d', (0xd7,0xaf,0x3d))],
 'pt': [('deep', '#7e8d3a', (0x7e,0x8d,0x3a)), ('mid', '#a1af5a', (0xa1,0xaf,0x5a)), ('light', '#c4d181', (0xc4,0xd1,0x81))],
 'es': [('deep', '#bc6b1a', (0xbc,0x6b,0x1a)), ('mid', '#dc8f37', (0xdc,0x8f,0x37)), ('light', '#f8b55a', (0xf8,0xb5,0x5a))],
 'it': [('mid', '#b7a168', (0xb7,0xa1,0x68)), ('light', '#d6c68e', (0xd6,0xc6,0x8e))],
 'ot': [('mid', '#8fcabb', (0x8f,0xca,0xbb)), ('light', '#d6c68e', (0xd6,0xc6,0x8e))],
}
SPECIAL = {'gray': ('#9090a0', (0x90,0x90,0xa0))}

def allowed_shades(t):
    out = list(PAL.get(t['power'], []))
    if t['id'] in ('afriquesud',):
        out.append(('boer', *SPECIAL['gray']))
    return out

TOL = 50  # per-channel sum tolerance (own-family shades only; nearest + ambiguity gap keeps safety)

def classify(rgb, shades):
    """Nearest shade within TOL; -2 if two shades nearly equidistant (ambiguous)."""
    dists = sorted(((abs(rgb[0]-c[0]) + abs(rgb[1]-c[1]) + abs(rgb[2]-c[2]), i) for i, (_, _, c) in enumerate(shades)))
    d1, i1 = dists[0]
    if d1 >= TOL: return -1
    if len(dists) > 1 and dists[1][0] < TOL and dists[1][0] - d1 < 18: return -2
    return i1

txt = open(BASE + '/data.js').read()
data = json.loads(txt[txt.index('{'):txt.rindex('}')+1])
terr = {t['id']: t for t in data['territories']}

from scipy import ndimage

def territory_mask(t):
    """Rasterize territory polygons into orig px space, erode 1.5px."""
    m = Image.new('1', (W, H), 0)
    dr = ImageDraw.Draw(m)
    for poly in t['geometry']:
        for ring in poly:
            pts = [to_px(p[0], p[1]) for p in ring]
            if len(pts) >= 3:
                dr.polygon(pts, fill=1)
        for ring in poly[1:]:
            pts = [to_px(p[0], p[1]) for p in ring]
            if len(pts) >= 3:
                dr.polygon(pts, fill=0)
    mask = np.asarray(m, bool)
    mask = ndimage.binary_erosion(mask, iterations=1)
    return mask

def _aff_apply(aff, x, y):
    c00, c01, t0, c10, c11, t1 = aff
    return (c00*x + c01*y + t0, c10*x + c11*y + t1)

def _aff_from(sx, sy, rot, dx, dy):
    import math as _m
    r = _m.radians(rot)
    return (_m.cos(r)*sx, -_m.sin(r)*sx, dx, _m.sin(r)*sy, _m.cos(r)*sy, dy)

def fit_affine(mask, shades, max_shift=10):
    """Per-territory affine (translation + anisotropic scale + small rotation)
    maximising family agreement of samples that stay inside the mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0: return (1, 0, 0, 0, 1, 0), 0.0
    idx = np.linspace(0, len(xs)-1, min(350, len(xs))).astype(int)
    sx_, sy_ = xs[idx].astype(float), ys[idx].astype(float)
    def score(sx, sy, rot, dx, dy):
        aff = _aff_from(sx, sy, rot, dx, dy)
        ok = 0
        for x, y in zip(sx_, sy_):
            u, v = _aff_apply(aff, x, y)
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < W and 0 <= vi < H and mask[vi, ui]:
                if classify(img[vi, ui], shades) >= 0: ok += 1
        return ok / len(sx_)
    best = {'s': score(1, 1, 0, 0, 0), 'sx': 1, 'sy': 1, 'r': 0, 'dx': 0, 'dy': 0}
    for r_ in [4, 2, 1]:
        cand = dict(best)
        for dy in range(-max_shift, max_shift+1):
            for dx in range(-max_shift, max_shift+1):
                if abs(dx-best['dx']) > r_ or abs(dy-best['dy']) > r_: continue
                s = score(best['sx'], best['sy'], best['r'], dx, dy)
                if s > cand['s']: cand = {'s': s, 'sx': best['sx'], 'sy': best['sy'], 'r': best['r'], 'dx': dx, 'dy': dy}
        best = cand
    if mask.sum() >= 400:
        for r_ in [0.02, 0.01, 0.005]:
            cand = dict(best)
            for dsa in (-1, 0, 1):
                for dsb in (-1, 0, 1):
                    for dr in (-1, 0, 1):
                        sx2 = best['sx']*(1+dsa*r_); sy2 = best['sy']*(1+dsb*r_); r2 = best['r'] + dr*r_*30
                        if abs(sx2-1) > 0.05 or abs(sy2-1) > 0.05 or abs(r2) > 1.5: continue
                        s = score(sx2, sy2, r2, best['dx'], best['dy'])
                        if s > cand['s']: cand = {'s': s, 'sx': sx2, 'sy': sy2, 'r': r2, 'dx': best['dx'], 'dy': best['dy']}
            best = cand
    aff = _aff_from(best['sx'], best['sy'], best['r'], best['dx'], best['dy'])
    return aff, best['s']

def shade_index(mask, shades, aff):
    """Classify mask pixels by sampling the original through the local affine."""
    res = np.full(mask.shape, -1, np.int8)
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs):
        u, v = _aff_apply(aff, x, y)
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < W and 0 <= vi < H and mask[vi, ui]:
            res[y, x] = classify(img[vi, ui], shades)
    return res

def clean_index(res):
    """Fill unknown pixels by nearest classified neighbor ONLY (no smoothing):
    zone contours then hug the raw classification pixel boundaries exactly."""
    unknown = res < 0  # includes ambiguous (-2)
    if unknown.all(): return res
    idx = ndimage.distance_transform_edt(unknown, return_distances=False, return_indices=True)
    out = res.copy()
    out[unknown] = res[tuple(idx)][unknown]
    return out

def px_ring_to_ll(ring):
    """Ring is already in territory pixel space -> lon/lat."""
    out = []
    for x, y in ring:
        lon, lat = to_ll(x, y)
        out.append([round(lon, 4), round(lat, 4)])
    if len(out) > 1 and out[0] == out[-1]: out.pop()
    return out

def mask_to_polygons(smask, min_area=1):
    """Exact polygonization: 4-connected components as unions of unit pixel boxes.
    No marching-squares diagonal ambiguity; holes arise naturally."""
    from shapely.geometry import box as SBoxB
    labs, n = ndimage.label(smask)
    polys = []
    for i in range(1, n + 1):
        comp = labs == i
        if comp.sum() < min_area: continue
        ys, xs = np.nonzero(comp)
        if len(xs) > 20000:  # safety: subsample via fill for huge regions
            comp2 = ndimage.binary_fill_holes(comp)
            contours = measure.find_contours(comp2.astype(float), 0.5)
            for c in contours:
                if len(c) >= 4:
                    ring = [(float(x), float(y)) for y, x in c]
                    p = ShPoly(ring)
                    if p.is_valid and p.area >= min_area:
                        polys.append(p)
            continue
        boxes = [SBoxB(int(x), int(y), int(x)+1, int(y)+1) for x, y in zip(xs, ys)]
        u = unary_union(boxes)
        if u.is_empty: continue
        for g in (u.geoms if u.geom_type == 'MultiPolygon' else [u]):
            if g.area >= min_area:
                polys.append(g)
    return polys

def simplify_ring(ring, tol_px=1.0):
    p = ShPoly(ring)
    if not p.is_valid: p = p.buffer(0)
    if p.is_empty: return None
    s = p.simplify(tol_px, preserve_topology=True)
    if s.is_empty or len(s.exterior.coords) < 4: return None
    return [(x, y) for x, y in s.exterior.coords]

results = {}
report = []
for t in data['territories']:
    tid = t['id']
    power = t['power']
    if power == 'indep' or tid in ('congo-belge',):
        continue  # flat fill (indep pattern / single gold) — nothing to measure
    shades = allowed_shades(t)
    mask = territory_mask(t)
    if mask.sum() < 25:
        report.append((tid, 'tiny', 0, 0))
        continue
    aff, f = fit_affine(mask, shades)
    if f < 0.45 and mask.sum() >= 2000:
        report.append((tid, 'low-conf %.2f' % f, '-', '-'))
        # still proceed; fallback handled by single-shade rule
    res = shade_index(mask, shades, aff)
    # text/rivers/borders contaminate nearby pixels (anti-aliased halos classify into
    # wrong shades, leaving letter-shaped blobs). Discard classified pixels within
    # 2px of any unknown pixel; they are inpainted from solid fill by nearest fill.
    unknown = res < 0
    suspect = (~unknown) & ndimage.binary_dilation(unknown, iterations=2)
    res[suspect] = -1
    known = res >= 0
    thr = 0.02 if mask.sum() < 2000 else (0.10 if mask.sum() < 1500 else 0.15)
    if known.sum() < thr * mask.sum():
        report.append((tid, 'too few classified', '-', '-'))
        continue
    res = clean_index(res)
    res[~mask] = -1
    counts = np.bincount(res[mask].astype(int), minlength=len(shades))
    order = np.argsort(counts)[::-1]
    base_i = int(order[0])
    share = counts[base_i] / counts.sum()
    zones = []
    if True:  # emit zones whenever any other shade exists
        for si in range(len(shades)):
            if si == base_i: continue
            if counts[si] < 1: continue
            smask = (res == si)
            polys = mask_to_polygons(smask, min_area=1)
            pass  # keep exact geometry
            kept = []
            for g in polys:
                rings_ll = []
                for ring in [g.exterior.coords] + [h.coords for h in g.interiors]:
                    ll = px_ring_to_ll([(x, y) for x, y in ring])
                    if len(ll) >= 4:
                        rings_ll.append(ll)
                if rings_ll:
                    kept.append(rings_ll)
            if kept:
                zones.append({'fill': shades[si][1], 'name': shades[si][0], 'geometry': kept,
                              '_area': int(counts[si])})
        zones.sort(key=lambda z: -z['_area'])  # painter order: big first, islands on top
        for z in zones: z.pop('_area', None)
    results[tid] = {'base': shades[base_i][1], 'baseName': shades[base_i][0], 'zones': zones,
                    'aff': [round(v, 5) for v in aff], 'conf': round(float(f), 2)}
    report.append((tid, f'base={shades[base_i][0]} {share:.0%}', f'zones={len(zones)}', f'aff=({aff[2]:.1f},{aff[5]:.1f})'))

json.dump(results, open(BASE + '/data/measured_zones.json', 'w'))
print('%-13s %-22s %-12s %s' % ('id', 'base', 'zones', 'shift'))
for r in report:
    print('%-13s %-22s %-12s %s' % r)
