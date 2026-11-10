#!/usr/bin/env python3
"""Final calibration check with per-row ocean model + per-territory fill sampling."""
import numpy as np, json, math
from PIL import Image, ImageDraw
from shapely.geometry import shape, box, Point
from shapely.ops import unary_union
from collections import Counter

BASE = '/Users/weiaiwen1/Desktop/my-space/map-project'
img = np.asarray(Image.open(BASE + '/source-map.png').convert('RGB')).astype(int)
H, W = img.shape[:2]

# ---------- per-row ocean color from edge strips ----------
rowOcean = np.zeros((H, 3))
edgeL = img[:, 2:18].reshape(H, -1, 3)
edgeR = img[:, W-18:W-2].reshape(H, -1, 3)
for y in range(H):
    samples = np.vstack([edgeL[y], edgeR[y]])
    # oceanish: B > R + 3, bright
    ok = (samples[:, 2] > samples[:, 0] + 3) & (samples.mean(axis=1) > 130)
    s = samples[ok] if ok.sum() > 10 else samples
    rowOcean[y] = np.median(s, axis=0)
# smooth
for _ in range(3):
    rowOcean[1:-1] = (rowOcean[:-2] + rowOcean[1:-1] + rowOcean[2:]) / 3

diff = np.abs(img - rowOcean[:, None, :]).max(axis=2)
ocean = diff < 14
img_land = ~ocean
from scipy import ndimage
lab, n = ndimage.label(img_land, structure=np.ones((3,3)))
sizes = ndimage.sum(img_land, lab, range(1, n+1))
keep = np.zeros(n+1, bool); keep[1:] = sizes > 150
img_land = keep[lab]
print('ocean px:', int(ocean.sum()), 'land px:', int(img_land.sum()))

# ---------- dataset raster (reuse calibrate5 base machinery) ----------
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
geoms_s = [g.simplify(0.1) for g in geoms]

def proj_equirect(lon, lat): return lon, lat
def proj_mercator(lon, lat):
    return lon, math.degrees(math.log(math.tan(math.pi/4 + math.radians(lat)/2)))
def proj_miller(lon, lat):
    return lon, math.degrees(1.25*math.log(math.tan(math.pi/4 + 0.4*math.radians(lat))))

SCALE = 4
LON0, LON1, LAT0, LAT1 = -32.0, 62.0, 44.0, -40.0
BW, BH = int((LON1-LON0)*SCALE), int((LAT0-LAT1)*SCALE)
base = Image.new('1', (BW, BH), 0)
dr = ImageDraw.Draw(base)
for g in geoms_s:
    polys = g.geoms if g.geom_type == 'MultiPolygon' else [g]
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

PY0, PY1 = -50.0, 90.0
PW, PH = int((LON1-LON0)*SCALE), int((PY1-PY0)*SCALE)

def base_for(pf):
    rows = np.arange(PH)
    py = PY1 - rows / SCALE
    lo = np.full_like(py, -89.9); hi = np.full_like(py, 89.9)
    for _ in range(60):
        mid = (lo + hi) / 2
        vals = np.array([pf(0, l)[1] for l in mid])
        lo = np.where(vals < py, mid, lo)
        hi = np.where(vals >= py, mid, hi)
    lat = (lo + hi) / 2
    src_y = np.clip(np.round((LAT0 - lat) * SCALE).astype(int), 0, BH-1)
    return base_arr[src_y, :]

def render(t, baseP):
    sx, sy, tx, ty = t
    im = Image.fromarray((baseP*255).astype('uint8'))
    a = SCALE/sx; c = -SCALE*tx/sx - LON0*SCALE
    e = SCALE/sy; f = SCALE*PY1 - SCALE*ty/sy
    return np.asarray(im.transform((W, H), Image.AFFINE, (a, 0, c, 0, e, f), resample=Image.BILINEAR)) > 127

excl = np.zeros((H, W), bool)
excl[470:675, 0:235] = True

def score(t, baseP):
    arr = render(t, baseP)
    inter = (arr & img_land & ~excl).sum()
    union = ((arr | img_land) & ~excl).sum()
    return inter / union if union else 0

results = {}
for pname, pf in [('equirect', proj_equirect), ('mercator', proj_mercator), ('miller', proj_miller)]:
    baseP = base_for(pf)
    best = (-1, None)
    for sx in np.arange(7.2, 9.81, 0.3):
        for sy in np.arange(7.2, 9.81, 0.3):
            for Lx in np.arange(-36.0, -19.9, 1.5):
                for Ly in np.arange(46.0, 33.9, -1.0):
                    t = (sx, sy, -sx*Lx, sy*pf(0, Ly)[1])
                    v = score(t, baseP)
                    if v > best[0]:
                        best = (v, t)
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
    results[pname] = cur
    print(f'{pname}: IoU={cur[0]:.4f} t={np.round(cur[1],3)}')

pname, (v, t) = max(results.items(), key=lambda kv: kv[1][0])
sx, sy, tx, ty = t
print('\nBEST:', pname, 'IoU=%.4f' % v)
print('frame: lon %.2f..%.2f lat %.2f..%.2f' % ((0-tx)/sx, (W-tx)/sx, ty/sy, (ty-H)/sy))
json.dump({'projection': pname, 't': list(map(float, t)), 'iou': float(v)}, open(BASE + '/data/calib.json', 'w'), indent=1)

# ---------- per-territory fill sampling ----------
def to_px(lon, lat):
    px_, py_ = {'equirect': proj_equirect, 'mercator': proj_mercator, 'miller': proj_miller}[pname](lon, lat)
    return (px_*sx + tx, py_*sy - ty if False else -py_*sy + ty)
# careful: y = ty - sy*PY  =>  y = ty - sy*PY  (PY up); we defined render with y=(ty-y)/sy=PY => y = ty - sy*PY ✓
def to_px2(lon, lat):
    px_, py_ = {'equirect': proj_equirect, 'mercator': proj_mercator, 'miller': proj_miller}[pname](lon, lat)
    return (round(px_*sx + tx), round(ty - py_*sy))

# load final territory geometry
txt = open(BASE + '/data.js').read()
data = json.loads(txt[txt.index('{'):txt.rindex('}')+1])

def interior_points(geom_coords, n=200):
    """sample points inside polygon rings via bbox rejection on outer rings"""
    pts = []
    for poly in geom_coords:
        ring = poly[0]
        xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
        from shapely.geometry import Polygon as ShPoly
        try:
            sp = ShPoly(ring, poly[1:])
        except Exception:
            continue
        if sp.area < 0.02: continue
        minx, miny, maxx, maxy = sp.bounds
        rng = np.random.default_rng(42)
        tries = 0
        got = 0
        while got < n and tries < n*30:
            tries += 1
            p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
            if sp.contains(p):
                pts.append((p.x, p.y)); got += 1
    return pts

power_names = {'fr':'FR','uk':'UK','de':'DE','be':'BE','pt':'PT','es':'ES','it':'IT','indep':'INDEP','ot':'OT'}
print('\n%-22s %-6s %-28s %s' % ('territory','power','modal fill','2nd/3rd'))
fills = {}
for terr in data['territories']:
    pts = interior_points(terr['geometry'], n=180)
    colors = []
    for lon, lat in pts:
        x, y = to_px2(lon, lat)
        if 0 <= x < W and 0 <= y < H:
            colors.append(tuple((img[y, x] // 8 * 8).tolist()))
    cnt = Counter(colors)
    top = cnt.most_common(3)
    tot = max(1, len(colors))
    desc = ' '.join('#%02x%02x%02x:%d%%' % (c[0], c[1], c[2], n*100//tot) for c, n in top)
    fills[terr['id']] = [(list(c), n / tot) for c, n in top]
    print('%-22s %-6s %s' % (terr['id'], power_names[terr['power']], desc))

json.dump(fills, open(BASE + '/data/territory_fills.json', 'w'), indent=1)
