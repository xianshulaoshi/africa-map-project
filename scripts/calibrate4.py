#!/usr/bin/env python3
"""Fit projection on the OCEAN mask (robust), excluding white legend box.
ASCII-diff diagnostic included."""
import numpy as np, json
from PIL import Image, ImageDraw
from shapely.geometry import shape, box
from shapely.ops import unary_union

BASE = '/Users/weiaiwen1/Desktop/my-space/map-project'
img = np.asarray(Image.open(BASE + '/source-map.png').convert('RGB')).astype(int)
H, W = img.shape[:2]
R_, B_ = img[:,:,0], img[:,:,2]

ocean = (B_ - R_ >= 5) & (img.mean(axis=2) > 150)
# legend box: white rectangle over ocean, x~10..215, y~485..650; detect biggest bright blob
from scipy import ndimage
bright = (img.min(axis=2) > 195)
lab, n = ndimage.label(bright, structure=np.ones((3,3)))
objs = ndimage.find_objects(lab)
excl = np.zeros((H, W), bool)
for i, sl in enumerate(objs, 1):
    if sl is None: continue
    a = (lab[sl] == i).sum()
    if a > 8000:
        ys, xs = sl
        print('big bright blob: x', xs.start, xs.stop, 'y', ys.start, ys.stop, 'area', a)
        if xs.stop - xs.start < W*0.6 and ys.stop - ys.start < H*0.5:
            # pad by 4
            excl[max(0,ys.start-4):min(H,ys.stop+4), max(0,xs.start-4):min(W,xs.stop+4)] = True
np.save(BASE + '/data/ocean_mask.npy', ocean)
np.save(BASE + '/data/excl_mask.npy', excl)
valid = ocean | excl  # pixels usable: ocean or outside legend
print('ocean px:', int(ocean.sum()), 'excluded px:', int(excl.sum()))

# dataset land
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

def dataset_ocean(t, pf):
    a, b, c, d, e, f = t
    im = Image.new('1', (W, H), 1)
    dr = ImageDraw.Draw(im)
    for g in geoms_s:
        polys = g.geoms if g.geom_type == 'MultiPolygon' else [g]
        for poly in polys:
            try:
                pj = []
                for ring in [poly.exterior.coords] + [r.coords for r in poly.interiors]:
                    pts = [(a*lon + b*pf(lon, lat)[1] + e, c*lon + d*pf(lon, lat)[1] + f) for lon, lat in ring]
                    pj.append(pts)
                dr.polygon(pj[0], fill=0)
                for hole in pj[1:]:
                    dr.polygon(hole, fill=1)
            except Exception:
                pass
    return np.asarray(im, bool)

def score(t, pf):
    ds_ocean = dataset_ocean(t, pf)
    # score on pixels that are ocean in image OR ocean in dataset, excluding legend
    inter = (ds_ocean & ocean & ~excl).sum()
    union = ((ds_ocean | ocean) & ~excl).sum()
    return inter / union if union else 0

# init from ocean bbox: ocean frame corners are the plot frame corners
ys, xs = np.nonzero(ocean)
results = {}
for pname, pf in [('equirect', proj_equirect), ('mercator', proj_mercator), ('miller', proj_miller)]:
    # dataset frame corners projected
    corners = [(-26, 40), (56, 40), (-26, -38), (56, -38)]
    pc = [pf(lon, lat) for lon, lat in corners]
    dminx = min(p[0] for p in pc); dmaxx = max(p[0] for p in pc)
    dminy = min(p[1] for p in pc); dmaxy = max(p[1] for p in pc)
    # image plot frame: use ocean bbox
    sx = (xs.max()-xs.min()) / (dmaxx-dminx)
    sy = (ys.max()-ys.min()) / (dmaxy-dminy)
    sy = -abs(sy)
    tx = xs.min() - sx*dminx
    ty = ys.max() - sy*dminy
    t0 = (sx, 0, 0, sy, tx, ty)
    v0 = score(t0, pf)
    best = (v0, t0)
    for st in [6, 3, 1.5, 0.8, 0.4]:
        improved = True
        it = 0
        while improved and it < 40:
            improved = False; it += 1
            for dsx in (-1,0,1):
                for dsy in (-1,0,1):
                    for dtx in (-1,0,1):
                        for dty in (-1,0,1):
                            if dsx==dsy==dtx==dty==0: continue
                            b0,b1,b2,b3,b4,b5 = best[1]
                            t = (b0*(1+dsx*0.008*st/3), 0, 0, b3*(1+dsy*0.008*st/3), b4+dtx*st*2, b5+dty*st*2)
                            v = score(t, pf)
                            if v > best[0] + 1e-5:
                                best = (v, t); improved = True
    results[pname] = best
    print(f'{pname}: init {v0:.3f} -> best {best[0]:.4f}  t={np.round(best[1],3)}')

pname, (v, t) = max(results.items(), key=lambda kv: kv[1][0])
print('\nBEST:', pname, 'IoU=%.4f' % v, np.round(t, 4))
json.dump({'projection': pname, 'transform': list(t), 'iou': v}, open(BASE + '/data/calib.json', 'w'), indent=1)

# ASCII diff diagnostic: mismatch clusters
ds_ocean = dataset_ocean(t, globals()['proj_' + pname])
mis = (ds_ocean != ocean) & ~excl
mis = mis & ~ndimage.binary_erosion(np.zeros_like(mis))  # no-op keep
labm, nm = ndimage.label(mis, structure=np.ones((3,3)))
sizes = ndimage.sum(mis, labm, range(1, nm+1))
order = np.argsort(sizes)[::-1] + 1
print('\nlargest mismatch clusters:')
for i in order[:15]:
    sl = ndimage.find_objects(labm == i)[0]
    print(f'  area={int(sizes[i-1])} x[{sl[1].start}..{sl[1].stop}] y[{sl[0].start}..{sl[0].stop}]')
