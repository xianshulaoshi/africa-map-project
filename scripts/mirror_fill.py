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
 'fr': [('deep', '#0878b8', (0x08,0x78,0xb8)), ('mid', '#4890c8', (0x48,0x90,0xc8)), ('light', '#88b8e0', (0x88,0xb8,0xe0))],
 'uk': [('deep', '#a060a0', (0xa0,0x60,0xa0)), ('mid', '#c088b8', (0xc0,0x88,0xb8)), ('light', '#e0a8c8', (0xe0,0xa8,0xc8))],
 'de': [('mid', '#689878', (0x68,0x98,0x78)), ('light', '#90b8a0', (0x90,0xb8,0xa0))],
 'be': [('mid', '#d0a838', (0xd0,0xa8,0x38))],
 'pt': [('deep', '#788838', (0x78,0x88,0x38)), ('mid', '#a0a858', (0xa0,0xa8,0x58)), ('light', '#c0d080', (0xc0,0xd0,0x80))],
 'es': [('deep', '#b86818', (0xb8,0x68,0x18)), ('mid', '#d88830', (0xd8,0x88,0x30)), ('light', '#f8b058', (0xf8,0xb0,0x58))],
 'it': [('mid', '#b0a068', (0xb0,0xa0,0x68)), ('light', '#d0c088', (0xd0,0xc0,0x88))],
 'ot': [('mid', '#88c8b8', (0x88,0xc8,0xb8)), ('light', '#d0c088', (0xd0,0xc0,0x88))],
}
SPECIAL = {'gray': ('#9090a0', (0x90,0x90,0xa0))}

def allowed_shades(t):
    out = list(PAL.get(t['power'], []))
    if t['id'] in ('afriquesud',):
        out.append(('boer', *SPECIAL['gray']))
    return out

TOL = 44  # per-channel sum tolerance

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

def best_shift(mask, shades, max_shift=10):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0: return (0, 0, 0.0)
    idx = np.linspace(0, len(xs)-1, min(400, len(xs))).astype(int)
    sx_, sy_ = xs[idx], ys[idx]
    def score(dx, dy):
        """# of sample points that classify into our family AND stay inside the mask."""
        ok = 0
        for x, y in zip(sx_+dx, sy_+dy):
            if 0 <= x < W and 0 <= y < H and mask[y, x]:
                if classify(img[y, x], shades) >= 0: ok += 1
        return ok / len(sx_)
    best = (score(0, 0), 0, 0)
    for r in [5, 2, 1]:
        cand = best
        for dy in range(max(-max_shift, cand[2]-r), min(max_shift, cand[2]+r)+1, 1):
            for dx in range(max(-max_shift, cand[1]-r), min(max_shift, cand[1]+r)+1, 1):
                s = score(dx, dy)
                if s > cand[0]: cand = (s, dx, dy)
        best = cand
    return best[1], best[2], best[0]

def shade_index(mask, shades, dx, dy):
    """Classify all mask pixels (sampled with shift) -> int array, -1 unknown.
    Samples landing outside the territory mask are treated as unknown."""
    res = np.full(mask.shape, -1, np.int8)
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs):
        yy, xx = y+dy, x+dx
        if 0 <= xx < W and 0 <= yy < H and mask[y, x] and mask[yy, xx]:
            res[y, x] = classify(img[yy, xx], shades)
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
    dx, dy, f = best_shift(mask, shades)
    if f < 0.45:
        report.append((tid, 'low-conf %.2f' % f, dx, dy))
        # still proceed with best shift; fallback handled by single-shade rule
    res = shade_index(mask, shades, dx, dy)
    known = res >= 0
    thr = 0.05 if mask.sum() < 700 else (0.10 if mask.sum() < 1500 else 0.15)
    if known.sum() < thr * mask.sum():
        report.append((tid, 'too few classified', dx, dy))
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
                    'shift': [dx, dy], 'conf': round(float(f), 2)}
    report.append((tid, f'base={shades[base_i][0]} {share:.0%}', f'zones={len(zones)}', f'shift=({dx},{dy})'))

json.dump(results, open(BASE + '/data/measured_zones.json', 'w'))
print('%-13s %-22s %-12s %s' % ('id', 'base', 'zones', 'shift'))
for r in report:
    print('%-13s %-22s %-12s %s' % r)
