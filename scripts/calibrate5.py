#!/usr/bin/env python3
"""Robust calibration v5: grid-search affine (scale x/y + translation) per projection
using fast PIL resampling of a pre-warped dataset raster; score ocean/land IoU
excluding the legend box."""
import numpy as np, json
from PIL import Image, ImageDraw
from shapely.geometry import shape, box
from shapely.ops import unary_union

BASE = '/Users/weiaiwen1/Desktop/my-space/map-project'
img = np.asarray(Image.open(BASE + '/source-map.png').convert('RGB')).astype(int)
H, W = img.shape[:2]
R_, B_ = img[:,:,0], img[:,:,2]

# image ocean mask
ocean = (B_ - R_ >= 5) & (img.mean(axis=2) > 150)
img_land = ~ocean
# kill small land specks (text) in ocean
from scipy import ndimage
lab, n = ndimage.label(img_land, structure=np.ones((3,3)))
sizes = ndimage.sum(img_land, lab, range(1, n+1))
keep = np.zeros(n+1, bool); keep[1:] = sizes > 120
img_land = keep[lab]
# legend exclusion box (measured from pixel inspection, generous)
excl = np.zeros((H, W), bool)
excl[470:675, 0:235] = True
score_valid = ~excl

# dataset land raster (equirect grid) at 4px/deg — clipped to modern coastlines
feats = json.load(open(BASE + '/data/world_1914.geojson'))['features']
frame = box(-32, -40, 62, 44)
nefeats = json.load(open(BASE + '/data/ne_50m_admin_0_countries.geojson'))['features']
land_ref = unary_union([shape(f['geometry']) for f in nefeats]).simplify(0.05)
geoms = []
for f in feats:
    p = f.get('properties') or {}
    if p.get('NAME') is None: continue
    try:
        g = shape(f['geometry']).intersection(frame).intersection(land_ref)
        if not g.is_empty: geoms.append(g)
    except Exception:
        pass

SCALE = 4
LON0, LON1, LAT0, LAT1 = -32.0, 62.0, 44.0, -40.0   # lat top->bottom
BW, BH = int((LON1-LON0)*SCALE), int((LAT0-LAT1)*SCALE)
base = Image.new('1', (BW, BH), 0)
dr = ImageDraw.Draw(base)
for g in geoms:
    gs = g.simplify(0.1)
    polys = gs.geoms if gs.geom_type == 'MultiPolygon' else [gs]
    for poly in polys:
        try:
            rings = [[((x-LON0)*SCALE, (LAT0-y)*SCALE) for x, y in poly.exterior.coords]]
            rings += [[((x-LON0)*SCALE, (LAT0-y)*SCALE) for x, y in r.coords] for r in poly.interiors]
            dr.polygon(rings[0], fill=1)
            for hole in rings[1:]:
                dr.polygon(hole, fill=0)
        except Exception:
            pass
base_arr = np.asarray(base, bool)
print('base raster', base_arr.shape, 'land frac', base_arr.mean().round(3))

import math as m
def proj_equirect(lon, lat): return lon, lat
def proj_mercator(lon, lat):
    return lon, m.degrees(m.log(m.tan(m.pi/4 + m.radians(lat)/2)))
def proj_miller(lon, lat):
    return lon, m.degrees(1.25*m.log(m.tan(m.pi/4 + 0.4*m.radians(lat))))

# pre-warp base to projection coords at 4px/unit, PY from -50 (bottom) to +90 (top)
PY0, PY1 = -50.0, 90.0
PW, PH = int((LON1-LON0)*SCALE), int((PY1-PY0)*SCALE)

def base_for(pf):
    out = np.zeros((PH, PW), bool)
    # for each output row, compute lat via inverse of projection (numeric)
    rows = np.arange(PH)
    py = PY1 - rows / SCALE  # top row = PY1
    # invert: find lat such that pf(0,lat)[1] == py  (monotonic) — bisection vectorized
    lo = np.full_like(py, -89.9); hi = np.full_like(py, 89.9)
    for _ in range(60):
        mid = (lo + hi) / 2
        vals = np.array([pf(0, l)[1] for l in mid])
        # projection y increases with lat
        lo = np.where(vals < py, mid, lo)
        hi = np.where(vals >= py, mid, hi)
    lat = (lo + hi) / 2
    src_y = (LAT0 - lat) * SCALE
    src_y = np.clip(np.round(src_y).astype(int), 0, BH-1)
    out = base_arr[src_y, :]
    return out

def render(t, baseP):
    sx, sy, tx, ty = t
    im = Image.fromarray((baseP*255).astype('uint8'))
    # want out[y,x] = baseP[(y-ty)/(-sy)+0? ] : y = ty - sy*PY_row_from_top... define PY_row r: y = ty - (PH - r)/SCALE * sy?? simpler:
    # x = sx*LON + tx -> LON = (x-tx)/sx ; raster col c = (LON-LON0)*SCALE = (x-tx)*SCALE/sx - LON0*SCALE
    # y = ty - sy*PY  (PY measured upward) ; row r (top=PY1): PY = PY1 - r/SCALE -> y = ty - sy*(PY1 - r/SCALE)
    #   -> r = SCALE*(PY1 - (ty-y)/sy) = (y - ty)/sy*SCALE + SCALE*PY1
    a = SCALE/sx; c = -SCALE*tx/sx - LON0*SCALE
    e = SCALE/sy; f = SCALE*PY1 - SCALE*ty/sy
    return im.transform((W, H), Image.AFFINE, (a, 0, c, 0, e, f), resample=Image.BILINEAR)

def score(t, baseP):
    arr = np.asarray(render(t, baseP)) > 127
    inter = (arr & img_land & score_valid).sum()
    union = (arr | img_land).sum() - ((arr | img_land) & ~score_valid).sum()
    return inter / union if union else 0

results = {}
for pname, pf in [('equirect', proj_equirect), ('mercator', proj_mercator), ('miller', proj_miller)]:
    baseP = base_for(pf)
    best = (-1, None)
    # anchor-parameterized grid: lon at x=0 (Lx), PY(lat Ly) at y=0
    for sx in np.arange(7.2, 9.81, 0.3):
        for sy in np.arange(7.2, 9.81, 0.3):
            for Lx in np.arange(-36.0, -19.9, 1.5):
                for Ly in np.arange(46.0, 33.9, -1.0):
                    tx = -sx * Lx
                    ty = sy * pf(0, Ly)[1]
                    v = score((sx, sy, tx, ty), baseP)
                    if v > best[0]:
                        best = (v, (sx, sy, tx, ty))
    print(f'{pname}: coarse best {best[0]:.4f} at sx,sy,Lx,Ly-equivalent {np.round(best[1],2)}')
    # refine
    cur = best
    for fac in [6, 3, 1.5, 0.8, 0.4, 0.2]:
        improved = True
        while improved:
            improved = False
            sx, sy, tx, ty = cur[1]
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    for ex in (-1,0,1):
                        for wy in (-1,0,1):
                            if dx==dy==ex==wy==0: continue
                            t = (sx+dx*0.1*fac, sy+dy*0.1*fac, tx+ex*4*fac, ty+wy*4*fac)
                            v = score(t, baseP)
                            if v > cur[0] + 1e-6:
                                cur = (v, t); improved = True
        print(f'   refine fac={fac}: {cur[0]:.4f}')
    results[pname] = cur

pname, (v, t) = max(results.items(), key=lambda kv: kv[1][0])
print('\nBEST:', pname, 'IoU=%.4f' % v, 't=', np.round(t, 4))
json.dump({'projection': pname, 't': list(map(float, t)), 'iou': v}, open(BASE + '/data/calib.json', 'w'), indent=1)

# diagnostics: mismatch clusters
baseP = base_for({'equirect': proj_equirect, 'mercator': proj_mercator, 'miller': proj_miller}[pname])
pred = np.asarray(render(t, baseP)) > 127
mis = (pred != img_land)
labm, nm = ndimage.label(mis & score_valid, structure=np.ones((3,3)))
sizes = ndimage.sum(mis & score_valid, labm, range(1, nm+1))
order = np.argsort(sizes)[::-1][:12] + 1
print('\nlargest mismatch clusters (area, bbox):')
for i in order:
    sl = ndimage.find_objects((labm == i).astype(int))[0]
    print(f'  area={int(sizes[i-1])} x[{sl[1].start}..{sl[1].stop}] y[{sl[0].start}..{sl[0].stop}]')
