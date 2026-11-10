#!/usr/bin/env python3
"""Calibrate projection + affine transform between the 1914 dataset and the original
map image by matching color masses (French blue, Belgian gold, German olive, etc.)."""
import numpy as np, json
from PIL import Image
from shapely.geometry import shape, MultiPolygon
from shapely.ops import unary_union
from collections import deque

IMG = '/Users/weiaiwen1/Desktop/my-space/map-project/source-map.png'
DATA = '/Users/weiaiwen1/Desktop/my-space/map-project/data/world_1914.geojson'
img = np.asarray(Image.open(IMG).convert('RGB')).astype(int)
H, W = img.shape[:2]

# ---------- color masks ----------
def mask(rgb, tol=30):
    return (np.abs(img - np.array(rgb)).sum(axis=2) < tol)

masks = {
  'blue':  mask((0x88,0xb8,0xe0)),
  'gold':  mask((0xd0,0xa8,0x38)),
  'olive': mask((0xc0,0xd0,0x80)),
  'sage':  mask((0xe0,0xf0,0xd0)),
  'dgreen':mask((0x68,0x98,0x78)),
  'mauve': mask((0xc0,0x88,0xb8)),
}
for k, m in masks.items():
    print(k, 'pixels:', int(m.sum()))

# ---------- connected components (4-conn BFS on downsampled mask) ----------
def components(m, min_area=300):
    mm = m.copy()
    comps = []
    ys, xs = np.nonzero(mm)
    seen = np.zeros_like(mm, bool)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]: continue
        q = deque([(y0, x0)]); seen[y0, x0] = True
        pts = []
        while q:
            y, x = q.popleft(); pts.append((y, x))
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)):
                ny, nx = y+dy, x+dx
                if 0<=ny<H and 0<=nx<W and mm[ny,nx] and not seen[ny,nx]:
                    seen[ny,nx]=True; q.append((ny,nx))
        if len(pts) >= min_area:
            pts = np.array(pts)
            comps.append((len(pts), pts[:,1].mean(), pts[:,0].mean(), pts[:,1].min(), pts[:,1].max(), pts[:,0].min(), pts[:,0].max()))
    comps.sort(reverse=True)
    return comps

comp_info = {}
for k, m in masks.items():
    comps = components(m)
    comp_info[k] = comps
    print(f'--- {k}: {len(comps)} components (area,cx,cy,x0,x1,y0,y1):')
    for c in comps[:6]:
        print('   area=%d c=(%.0f,%.0f) bbox x[%d..%d] y[%d..%d]' % c)

# ---------- dataset polygons ----------
feats = json.load(open(DATA))['features']
byname = {}
for f in feats:
    n = (f.get('properties') or {}).get('NAME')
    if n is None: continue
    byname.setdefault(n, []).append(shape(f['geometry']))

def union(names):
    return unary_union([g for n in names for g in byname.get(n, [])])

datasets = {
  'blue':  union(['French West Africa','Algeria','Tunisia','Madagascar (France)','Djibouti','Morocco']),
  'gold':  union(['Belgian Congo']),
  'olive': union(['German E. Africa (Tanganyika)']),
  'olive2':union(['German South-West Africa']),
  'sage':  union(['Libya']),
  'dgreen':union(['Angola']),
  'mauve': union(['Egypt','Anglo-Egyptian Sudan']),
}

# ---------- projections ----------
def proj_equirect(lon, lat): return np.array([lon, lat])
def proj_mercator(lon, lat):
    r = np.radians
    return np.array([r(lon), np.degrees(np.log(np.tan(np.pi/4 + r(lat)/2)))])
def proj_miller(lon, lat):
    r = np.radians
    return np.array([r(lon), np.degrees(1.25*np.log(np.tan(np.pi/4 + 0.4*r(lat))))])

projections = {'equirect': proj_equirect, 'mercator': proj_mercator, 'miller': proj_miller}

def centroid_of(geom):
    c = geom.centroid
    return (c.x, c.y)

gcp_pairs = []
for key in ['blue','gold','olive','olive2','sage','dgreen']:
    comps = comp_info.get(key, [])
    if not comps: print('no comp for', key); continue
    area, cx, cy, *_ = comps[0]
    if key == 'olive2' and len(comps) > 1:  # GSW is smaller than GEA; GEA is comps[0]? check by y
        pass
    lon, lat = centroid_of(datasets[key])
    gcp_pairs.append((key, cx, cy, lon, lat))
# German components: larger area = GEA (usually), second = GSW or Kamerun.
olive_c = comp_info['olive']
# GEA is southern (larger y); Kamerun/GSW smaller. Pick component whose centroid y is biggest as GEA:
olive_sorted_by_y = sorted(olive_c, key=lambda c: -c[2])
gcp_map = {k: (cx, cy, lon, lat) for k, cx, cy, lon, lat in gcp_pairs}
if len(olive_sorted_by_y) >= 2:
    # GEA = southernmost big component
    area, cx, cy, *_ = olive_sorted_by_y[0]
    lon, lat = centroid_of(datasets['olive'])
    gcp_map['olive'] = (cx, cy, lon, lat)
    # GSW = the one with max y among remaining
    area2, cx2, cy2, *_ = olive_sorted_by_y[1]
    lon2, lat2 = centroid_of(datasets['olive2'])
    gcp_map['olive2'] = (cx2, cy2, lon2, lat2)

# mauve: rough stage-1 fit first using existing GCPs
def fit_affine(pairs, projf):
    src, dst = [], []
    for k, (cx, cy, lon, lat) in pairs.items():
        p = projf(lon, lat)
        src.append([p[0], p[1], 0, 0, 1, 0])
        src.append([0, 0, p[0], p[1], 0, 1])
        dst += [cx, cy]
    A = np.array(src); b = np.array(dst)
    coef, res, *_ = np.linalg.lstsq(A, b, rcond=None)
    pred = A @ coef
    rms = float(np.sqrt(((pred - b)**2).mean()))
    return coef, rms

def apply_affine(coef, lon, lat):
    a, b, c, d, e, f = coef
    x = a*lon + b*lat + e
    y = c*lon + d*lat + f
    return x, y

print('\n=== stage-1 fits (GCPs: %s) ===' % sorted(gcp_map))
best = None
for name, pf in projections.items():
    coef, rms = fit_affine(gcp_map, pf)
    print(f'{name}: rms={rms:.1f}px coef={np.round(coef,3)}')
    if best is None or rms < best[1]: best = (name, rms, coef)

pname, rms, coef = best
print('best projection:', pname)

# add mauve GCP: mauve pixels near predicted Egypt/Sudan centroid
lon0, lat0 = centroid_of(datasets['mauve'])
px, py = apply_affine(coef, *proj_dict_dummy if False else (lambda: ( (pname=='equirect' and (lon0,lat0)) or None ))() )
# simpler: re-apply projection then affine
def project_point(pf, lon, lat):
    p = pf(lon, lat)
    return apply_affine(coef, p[0], p[1])
px, py = project_point(projections[pname], lon0, lat0)
m = masks['mauve'] & False
yy, xx = np.mgrid[0:H, 0:W]
near = masks['mauve'] & (np.abs(xx-px) < 120) & (np.abs(yy-py) < 120)
if near.sum() > 500:
    cy2, cx2 = yy[near].mean(), xx[near].mean()
    gcp_map['mauve'] = (cx2, cy2, lon0, lat0)
    coef2, rms2 = fit_affine(gcp_map, projections[pname])
    print('stage-2 with mauve: rms=%.1fpx' % rms2)
    coef = coef2

print('final coef:', np.round(coef, 4))

# save mapping + report residuals
print('\nGCP residuals:')
for k, (cx, cy, lon, lat) in gcp_map.items():
    p = projections[pname](lon, lat)
    qx, qy = apply_affine(coef, p[0], p[1])
    print(f'  {k:8s} img=({cx:6.1f},{cy:6.1f}) pred=({qx:6.1f},{qy:6.1f}) err={np.hypot(qx-cx,qy-cy):.1f}px')

json.dump({'projection': pname, 'coef': list(coef)}, open('/Users/weiaiwen1/Desktop/my-space/map-project/data/calib.json','w'))
print('saved data/calib.json')
