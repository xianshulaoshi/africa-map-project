#!/usr/bin/env python3
"""Pixel-by-pixel verification: for every territory, compare the ORIGINAL's raw
classified fill (uncleaned) against the composed structure (base fill + zone rings)
at every classifiable pixel inside the territory. This is the honest metric:
text/river/border pixels (unclassifiable) are excluded from both sides."""
import numpy as np, json, math, sys
from PIL import Image, ImageDraw
from shapely.geometry import shape as shp_shape
from scipy import ndimage

BASE = '/Users/weiaiwen1/Desktop/my-space/map-project'
img = np.asarray(Image.open(BASE + '/source-map.png').convert('RGB')).astype(int)
H, W = img.shape[:2]
calib = json.load(open(BASE + '/data/calib.json'))
osx, osy, otx, oty = calib['t']
RAD = math.pi / 180
def mercY(lat): return math.log(math.tan(math.pi/4 + lat*RAD/2)) / RAD
def to_px(lon, lat): return (lon*osx + otx, oty - mercY(lat)*sy if False else oty - mercY(lat)*osy)

PAL = {
 'fr': [('deep', (0x0c,0x78,0xb9)), ('mid', (0x4a,0x97,0xcd)), ('light', (0x8a,0xbb,0xe3))],
 'uk': [('deep', (0xa6,0x64,0xa2)), ('mid', (0xc2,0x88,0xb8)), ('light', (0xe2,0xaf,0xce))],
 'de': [('mid', (0x68,0x9a,0x7f)), ('light', (0x90,0xbc,0xa5))],
 'be': [('mid', (0xd7,0xaf,0x3d))],
 'pt': [('deep', (0x7e,0x8d,0x3a)), ('mid', (0xa1,0xaf,0x5a)), ('light', (0xc4,0xd1,0x81))],
 'es': [('deep', (0xbc,0x6b,0x1a)), ('mid', (0xdc,0x8f,0x37)), ('light', (0xf8,0xb5,0x5a))],
 'it': [('mid', (0xb7,0xa1,0x68)), ('light', (0xd6,0xc6,0x8e))],
 'ot': [('mid', (0x8f,0xca,0xbb)), ('light', (0xd6,0xc6,0x8e))],
}
TOL = 50

txt = open(BASE + '/data.js').read()
data = json.loads(txt[txt.index('{'):txt.rindex('}')+1])
MEAS = json.load(open(BASE + '/data/measured_zones.json'))

def allowed(t):
    out = list(PAL.get(t['power'], []))
    if t['id'] == 'afriquesud':
        out.append(('boer', (0x90,0x90,0xa0)))
    return out

def hex_to_name(shades, hx):
    for n, c in shades:
        h = '#%02x%02x%02x' % c
        if h == hx: return n
    if hx == '#9090a0': return 'boer'
    return None

def classify(rgb, shades):
    """Best shade, or -2 if ambiguous (two shades nearly equidistant -> no ground truth)."""
    dists = sorted(((abs(rgb[0]-c[0]) + abs(rgb[1]-c[1]) + abs(rgb[2]-c[2]), i) for i, (_, c) in enumerate(shades)))
    d1, i1 = dists[0]
    if d1 >= TOL: return -1
    if len(dists) > 1 and dists[1][0] < TOL and dists[1][0] - d1 < 18:
        return -2  # ambiguous antialias pixel — excluded from metric
    return i1

def territory_mask(t):
    m = Image.new('1', (W, H), 0)
    dr = ImageDraw.Draw(m)
    for poly in t['geometry']:
        for ring in poly:
            pts = [to_px(p[0], p[1]) for p in ring]
            if len(pts) >= 3: dr.polygon(pts, fill=1)
        for ring in poly[1:]:
            pts = [to_px(p[0], p[1]) for p in ring]
            if len(pts) >= 3: dr.polygon(pts, fill=0)
    mask = np.asarray(m, bool)
    return ndimage.binary_erosion(mask, iterations=1)

from matplotlib.path import Path as MplPath

_centers = None
def paint_polys(polys_px):
    """Exact even-odd containment test at pixel centers (matches SVG fill semantics)."""
    global _centers
    if _centers is None:
        yy, xx = np.mgrid[0:H, 0:W]
        _centers = np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5])
    out = np.zeros((H, W), bool)
    for rings in polys_px:
        if not rings or len(rings[0]) < 3: continue
        try:
            outer = MplPath(np.asarray(rings[0] + [rings[0][0]], float))
            m = outer.contains_points(_centers).reshape(H, W)
            for hole in rings[1:]:
                if len(hole) < 3: continue
                hp = MplPath(np.asarray(hole + [hole[0]], float))
                m &= ~hp.contains_points(_centers).reshape(H, W)
            out |= m
        except Exception:
            pass
    return out

def my_shade_raster(t, shades):
    """Compose base + zones -> shade-index raster over the whole image (int8, -1 outside)."""
    names = [n for n, _ in shades]
    res = np.full((H, W), -1, np.int8)
    mask = territory_mask(t)
    bi = names.index(hex_to_name(shades, t['fill'])) if hex_to_name(shades, t['fill']) in names else -1
    if bi < 0:
        # base not in this family (fallback single shade like OT teal for libye)
        # find nearest palette entry by hex
        best, bd = 0, 1e9
        for i, (_, c) in enumerate(shades):
            hx = t['fill'].lstrip('#')
            d = abs(int(hx[0:2],16)-c[0]) + abs(int(hx[2:4],16)-c[1]) + abs(int(hx[4:6],16)-c[2])
            if d < bd: best, bd = i, d
        bi = best
    res[mask] = bi
    for z in t.get('zones', []):
        zi = hex_to_name(shades, z['fill'])
        if zi is None or zi not in names: continue
        k = names.index(zi)
        polys_px = []
        for poly in z['geometry']:
            rings_px = []
            for ring in poly:
                pts = [to_px(p[0], p[1]) for p in ring]
                if len(pts) >= 3: rings_px.append(pts)
            if rings_px: polys_px.append(rings_px)
        zm = paint_polys(polys_px) & mask
        res[zm] = k
    return res, mask

rows = []
tot_ok = tot_n = 0
worst = []
for t in data['territories']:
    if t['power'] == 'indep': continue
    shades = allowed(t)
    if not shades: continue
    mine, mask = my_shade_raster(t, shades)
    meas = MEAS.get(t['id'])
    aff = (meas.get('aff') if meas else None) or (
        [1, 0, meas['shift'][0], 0, 1, meas['shift'][1]] if meas and 'shift' in meas else [1, 0, 0, 0, 1, 0])
    # raw original classification sampled through the same affine
    orig_idx = np.full((H, W), -1, np.int8)
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs):
        u, v = aff[0]*x + aff[1]*y + aff[2], aff[3]*x + aff[4]*y + aff[5]
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < W and 0 <= vi < H and mask[vi, ui]:
            orig_idx[y, x] = classify(img[vi, ui], shades)
    # exclude text-halo contamination: classified pixels within 2px of unknown
    unk = orig_idx < 0
    orig_idx[(orig_idx >= 0) & ndimage.binary_dilation(unk, iterations=2)] = -1
    known = orig_idx >= 0
    ambig = int((orig_idx == -2).sum())
    n = int(known.sum())
    if n < 5: continue
    agree = int((mine[known] == orig_idx[known]).sum())
    tot_ok += agree; tot_n += n
    pct = agree * 100.0 / n
    rows.append((t['id'], n, pct))
    if pct < 97:
        from collections import Counter
        cc = Counter()
        names = [nm for nm, _ in shades]
        m_, o_ = mine[known], orig_idx[known]
        for a, b in zip(m_[m_ != o_], o_[m_ != o_]):
            cc[f'{names[a]}<-{names[b]}'] += 1
        worst.append((t['id'], n, pct, ' '.join(f'{k}:{v}' for k, v in cc.most_common(3))))

rows.sort(key=lambda r: r[2])
print('%-13s %7s %8s' % ('territory', 'pixels', 'agree%'))
for tid, n, pct in rows:
    flag = '' if pct >= 99 else ('  ~' if pct >= 97 else '  << FIX')
    print('%-13s %7d %7.2f%%%s' % (tid, n, pct, flag))
print('ambiguous excluded pixels total: (see per-territory)') if False else None
print('\nOVERALL: %d/%d = %.2f%%' % (tot_ok, tot_n, tot_ok*100.0/tot_n))
print('\nworst offenders (mine<-orig confusion):')
for w in sorted(worst, key=lambda x: x[2]):
    print('  %-13s %5d px %6.2f%%  %s' % w)
