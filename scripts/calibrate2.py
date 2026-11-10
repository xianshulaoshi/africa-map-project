#!/usr/bin/env python3
"""Objective calibration:
1. Build land mask from the original image (not-ocean, denoised).
2. Rasterize dataset land for each projection; fit translation+scale maximizing IoU.
3. Sample territory fill colors at verified interior points -> palette per power.
"""
import numpy as np, json
from PIL import Image
from shapely.geometry import shape, box
from shapely.ops import unary_union
from collections import Counter

BASE = '/Users/weiaiwen1/Desktop/my-space/map-project'
img = np.asarray(Image.open(BASE + '/source-map.png').convert('RGB')).astype(int)
H, W = img.shape[:2]

# ---------- 1. land mask ----------
R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
# ocean: bluish pale (B > R by >= 8) and bright
ocean = (B - R >= 6) & (img.mean(axis=2) > 150)
not_ocean = ~ocean
# remove small specks (text, thin lines): label components
try:
    from scipy import ndimage
    lab, n = ndimage.label(not_ocean, structure=np.ones((3,3)))
    sizes = ndimage.sum(not_ocean, lab, range(1, n+1))
    keep = np.zeros(n+1, bool)
    keep[1:] = sizes > 60
    land = keep[lab]
    print('scipy labeling ok, components:', n, 'land px:', int(land.sum()))
except ImportError:
    land = not_ocean
    print('no scipy; raw mask')

np.save(BASE + '/data/land_mask.npy', land)

# ---------- 2. rasterize dataset land ----------
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
land_deg = unary_union(geoms)
print('dataset land area deg2:', round(land_deg.area, 1))

# sample dataset land boundary points as proxies: rasterize via matplotlib-free method:
# use shapely prepared + point-in-polygon on a grid — too slow in python for full search.
# Instead: extract land polygon vertices as edge points; and centroid+area only.
cx_deg, cy_deg = land_deg.centroid.x, land_deg.centroid.y
area_deg = land_deg.area

ys, xs = np.nonzero(land)
px_area = len(xs)
pcx, pcy = xs.mean(), ys.mean()
# rough isotropic scale: px per degree  (assume aspect ~ equal area scaling)
# area in px2 / area in deg2 -> scale^2. BUT projections distort; use as init only.
s0 = math_sqrt = float(np.sqrt(px_area / area_deg))
print(f'image land: centroid=({pcx:.1f},{pcy:.1f}) area={px_area}px  s0={s0:.2f}px/deg')
print(f'dataset land centroid: ({cx_deg:.2f},{cy_deg:.2f})')

# projections return (x, y) in degrees-like units
def proj_equirect(lon, lat): return lon, lat
def proj_mercator(lon, lat):
    import math as m
    return lon, m.degrees(m.log(m.tan(m.pi/4 + m.radians(lat)/2)))
def proj_miller(lon, lat):
    import math as m
    return lon, m.degrees(1.25*m.log(m.tan(m.pi/4 + 0.4*m.radians(lat))))

# get land edge points (polygon vertices) for fast rasterized IoU
edges = []
for g in geoms:
    gs = g.simplify(0.15)
    for poly in (gs.geoms if gs.geom_type == 'MultiPolygon' else [gs]):
        xs_ = np.array(poly.exterior.coords)
        edges.append(xs_)
edge_pts = np.vstack(edges)
print('edge points:', edge_pts.shape)

def score_iou(transform, projf, n_edge=None, sample_step=2):
    """IoU between image land and dataset land raster: use filled rasterization via
    edge points is not enough; instead rasterize polygons with PIL ImageDraw."""
    from PIL import Image as PImage, ImageDraw as PDraw
    a, b, c, d, e, f = transform  # x = a*px + b*py + e ; y = c*px + d*py + f  (px=lon, py=proj y)
    im = PImage.new('1', (W, H), 0)
    dr = PDraw.Draw(im)
    for g in geoms:
        gs = g.simplify(0.15)
        polys = gs.geoms if gs.geom_type == 'MultiPolygon' else [gs]
        for poly in polys:
            rings = [poly.exterior.coords] + [r.coords for r in poly.interiors]
            pj = []
            for ring in rings:
                pts = []
                for lon, lat in ring:
                    px_, py_ = projf(lon, lat)
                    pts.append((a*lon + b*py_ + e, c*lon + d*py_ + f))
                pj.append(pts)
            dr.polygon(pj[0], fill=1)
            for hole in pj[1:]:
                dr.polygon(hole, fill=0)
    arr = np.asarray(im, bool)
    inter = (arr & land).sum()
    union = (arr | land).sum()
    return inter / union if union else 0

best = None
for pname, pf in [('equirect', proj_equirect), ('mercator', proj_mercator), ('miller', proj_miller)]:
    # init transform: identity scale mapping centroid
    py_deg = pf(cx_deg, cy_deg)[1]
    # estimate y-extent of projected land to set scale: use edge points
    ex = edge_pts
    pxy = np.array([pf(lon, lat) for lon, lat in ex[:, :2]])
    span_x = pxy[:,0].max() - pxy[:,0].min()
    span_y = pxy[:,1].max() - pxy[:,1].min()
    lx = xs.max() - xs.min(); ly = ys.max() - ys.min()
    sx = lx / span_x; sy = ly / span_y
    # centroids of edge proxy (not true centroid but ok for init)
    tx = pcx - (sx * pxy[:,0].mean())
    ty = pcy - (sy * pxy[:,1].mean())  # note: image y down, projection y up -> use negative scale
    # y axis: image y grows downward; projection y grows upward -> sy should be negative
    sy = -abs(sy)
    ty = pcy - sy * pxy[:,1].mean()
    # refine: local search
    base_t = (sx, 0, 0, sy, tx, ty)
    best_local = (score_iou(base_t, pf), base_t)
    print(f'{pname}: init IoU={best_local[0]:.3f} t={np.round(base_t,3)}')
    for it, step in enumerate([6, 3, 1.5, 0.7]):
        improved = True
        while improved:
            improved = False
            s0v, y0v = best_local[1][0], best_local[1][3]
            txc, tyc = best_local[1][4], best_local[1][5]
            for dsx in (-1, 0, 1):
                for dsy in (-1, 0, 1):
                    for dtx in (-1, 0, 1):
                        for dty in (-1, 0, 1):
                            if dsx == dsy == dtx == dty == 0: continue
                            t = (s0v + dsx*step*0.004*s0v, 0, 0, y0v + dsy*step*0.004*abs(y0v), txc + dtx*step, tyc + dty*step)
                            v = score_iou(t, pf)
                            if v > best_local[0] + 1e-5:
                                best_local = (v, t); improved = True
        print(f'  refine step {step}: IoU={best_local[0]:.4f}')
    if best is None or best_local[0] > best[0]:
        best = (best_local[0], best_local[1], pname)

print('\nBEST:', best[2], 'IoU=%.4f' % best[0], 'transform=', np.round(best[1], 4))
json.dump({'projection': best[2], 'transform': list(best[1]), 'iou': best[0]},
          open(BASE + '/data/calib.json', 'w'))
