#!/usr/bin/env python3
"""Clean land mask -> fit projection+affine -> sample territory fill colors."""
import numpy as np, json
from PIL import Image, ImageDraw
from shapely.geometry import shape, box
from shapely.ops import unary_union
from collections import Counter

BASE = '/Users/weiaiwen1/Desktop/my-space/map-project'
img_pil = Image.open(BASE + '/source-map.png').convert('RGB')
img = np.asarray(img_pil).astype(int)
H, W = img.shape[:2]
R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]

# ---------- 1. content bbox: non-white pixels ----------
nonwhite = (img.min(axis=2) < 240) | ((B - R) >= 6)
colfrac = nonwhite.mean(axis=0)
rowfrac = nonwhite.mean(axis=1)
cols = np.nonzero(colfrac > 0.04)[0]
rows = np.nonzero(rowfrac > 0.04)[0]
x0, x1, y0, y1 = cols[0], cols[-1], rows[0], rows[-1]
print('content bbox: x', x0, x1, ' y', y0, y1)

# ---------- 2. land mask within bbox ----------
sub = img[y0:y1+1, x0:x1+1]
ocean = (sub[:,:,2] - sub[:,:,0] >= 6) & (sub.mean(axis=2) > 150)
not_ocean = ~ocean
from scipy import ndimage
lab, n = ndimage.label(not_ocean, structure=np.ones((3,3)))
sizes = ndimage.sum(not_ocean, lab, range(1, n+1))
keep = np.zeros(n+1, bool)
keep[1:] = sizes > 60
land = keep[lab]
# white legend box: largest bright component whose bbox lies fully inside lower-left and doesn't touch borders
lab2, n2 = ndimage.label(land, structure=np.ones((3,3)))
objs = ndimage.find_objects(lab2)
legend_box = None
h_, w_ = land.shape
for i, sl in enumerate(objs, start=1):
    if sl is None: continue
    ys, xs = sl
    hh, ww = ys.stop - ys.start, xs.stop - xs.start
    area = (lab2[sl] == i).sum()
    if area > 2000 and hh > 100 and ww > 80 and area / (hh*ww) > 0.85 \
       and ys.start > h_*0.5 and xs.start < w_*0.35 \
       and ys.stop < h_-5 and xs.stop < w_*0.6:
        mean_rgb = sub[sl][lab2[sl] == i].mean(axis=0)
        if mean_rgb.min() > 200 and (mean_rgb.max() - mean_rgb.min()) < 30:
            legend_box = (xs.start, xs.stop, ys.start, ys.stop)
            print('legend box detected (subframe coords):', legend_box, 'area', area)
            land[lab2 == i] = False
np.save(BASE + '/data/land_mask.npy', land)
np.save(BASE + '/data/content_bbox.npy', np.array([x0, x1, y0, y1]))
Image.fromarray((land*255).astype('uint8')).save(BASE + '/land-mask.png')
print('land px:', int(land.sum()))

# ---------- 3. dataset land rasterization + IoU fit ----------
feats = json.load(open(BASE + '/data/world_1914.geojson'))['features']
frame = box(-32, -40, 62, 44)
geoms = []
for f in feats:
    p = f.get('properties') or {}
    if p.get('NAME') is None: continue
    try:
        g = shape(f['geometry']).intersection(frame)
        if not g.is_empty: geoms.append(g)
    except Exception:
        pass
geoms_s = [g.simplify(0.15) for g in geoms]

def proj_equirect(lon, lat): return lon, lat
def proj_mercator(lon, lat):
    import math as m
    return lon, m.degrees(m.log(m.tan(m.pi/4 + m.radians(lat)/2)))
def proj_miller(lon, lat):
    import math as m
    return lon, m.degrees(1.25*m.log(m.tan(m.pi/4 + 0.4*m.radians(lat))))

def rasterize(t, pf):
    a, b, c, d, e, f = t
    im = Image.new('1', (W, H), 0)
    dr = ImageDraw.Draw(im)
    for g in geoms_s:
        polys = g.geoms if g.geom_type == 'MultiPolygon' else [g]
        for poly in polys:
            try:
                pj = []
                for ring in [poly.exterior.coords] + [r.coords for r in poly.interiors]:
                    pts = [(a*lon + b*pf(lon, lat)[1] + e, c*lon + d*pf(lon, lat)[1] + f) for lon, lat in ring]
                    pj.append(pts)
                dr.polygon(pj[0], fill=1)
                for hole in pj[1:]:
                    dr.polygon(hole, fill=0)
            except Exception:
                pass
    return np.asarray(im, bool)

ys, xs = np.nonzero(land)
full_land = np.zeros((H, W), bool)
full_land[y0:y1+1, x0:x1+1] = land

def iou(t, pf):
    arr = rasterize(t, pf)
    inter = (arr & full_land).sum()
    union = (arr | full_land).sum()
    return inter / union if union else 0

results = {}
for pname, pf in [('equirect', proj_equirect), ('mercator', proj_mercator), ('miller', proj_miller)]:
    # init: from land area & centroid
    land_deg = unary_union(geoms)
    cxd, cyd = land_deg.centroid.x, land_deg.centroid.y
    pyd = pf(cxd, cyd)[1]
    s = np.sqrt(len(xs) / land_deg.area)
    # rough spans for independent scales
    minx, miny, maxx, maxy = 1e9, 1e9, -1e9, -1e9
    for g in geoms_s:
        b_ = g.bounds
        x1_, y1_ = pf(b_[0], 0)[0], min(pf(0, b_[1])[1], 1e9)
        x2_, y2_ = pf(b_[2], 0)[0], pf(0, b_[3])[1]
        minx = min(minx, b_[0]); maxx = max(maxx, b_[2])
        miny = min(miny, pf(0, b_[1])[1]); maxy = max(maxy, pf(0, b_[3])[1])
    sx = (xs.max()-xs.min()) / (maxx-minx)
    sy = (ys.max()-ys.min()) / (maxy-miny)
    tx = xs.mean() - sx*minx - sx*(cxd-minx)*0  # simplify: anchor centroid
    tx = xs.mean() - sx*cxd
    ty = ys.mean() + sy*pyd   # y flipped
    sy = -abs(sy)
    t0 = (sx, 0, 0, sy, tx, ty)
    v0 = iou(t0, pf)
    # bounded refine
    best = (v0, t0)
    steps = [4, 2, 1]
    for st in steps:
        improved = True
        it = 0
        while improved and it < 30:
            improved = False; it += 1
            for dsx in (-1,0,1):
                for dsy in (-1,0,1):
                    for dtx in (-1,0,1):
                        for dty in (-1,0,1):
                            if dsx==dsy==dtx==dty==0: continue
                            s0_, s1_, _, s3_, s4_, s5_ = best[1]
                            t = (s0_*(1+dsx*0.01*st/2), 0, 0, s3_*(1+dsy*0.01*st/2), s4_+dtx*st*2, s5_+dty*st*2)
                            v = iou(t, pf)
                            if v > best[0] + 1e-5:
                                best = (v, t); improved = True
    results[pname] = best
    print(f'{pname}: init {v0:.3f} -> best {best[0]:.4f}  t={np.round(best[1],3)}')

pname, (v, t) = max(results.items(), key=lambda kv: kv[1][0])
print('\nBEST:', pname, 'IoU=%.4f' % v, np.round(t, 4))
json.dump({'projection': pname, 'transform': list(t), 'iou': v, 'bbox': [int(x0),int(x1),int(y0),int(y1)]},
          open(BASE + '/data/calib.json', 'w'), indent=1)
