#!/usr/bin/env python3
"""calibrate7: test Lambert Conformal Conic (common for French atlas Africa maps)
against the three cylindric candidates; then per-territory fill sampling with the
winner + area-weighted power palette."""
import numpy as np, json, math
from PIL import Image, ImageDraw
from shapely.geometry import shape, box
from shapely.ops import unary_union
from collections import Counter

BASE = '/Users/weiaiwen1/Desktop/my-space/map-project'
img = np.asarray(Image.open(BASE + '/source-map.png').convert('RGB')).astype(int)
H, W = img.shape[:2]

# ocean/land masks (as calibrate6)
rowOcean = np.zeros((H, 3))
edgeL = img[:, 2:18].reshape(H, -1, 3); edgeR = img[:, W-18:W-2].reshape(H, -1, 3)
for y in range(H):
    samples = np.vstack([edgeL[y], edgeR[y]])
    ok = (samples[:, 2] > samples[:, 0] + 3) & (samples.mean(axis=1) > 130)
    s = samples[ok] if ok.sum() > 10 else samples
    rowOcean[y] = np.median(s, axis=0)
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
excl = np.zeros((H, W), bool); excl[470:675, 0:235] = True

# dataset equirect base raster
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
LON0, LON1, LAT0, LAT1 = -32.0, 62.0, 44.0, -40.0
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

def proj_equirect(lon, lat): return lon, lat
def proj_mercator(lon, lat):
    return lon, math.degrees(math.log(math.tan(math.pi/4 + math.radians(lat)/2)))
def proj_miller(lon, lat):
    return lon, math.degrees(1.25*math.log(math.tan(math.pi/4 + 0.4*math.radians(lat))))

def make_lcc(p1, p2, lon0):
    p1, p2, lon0 = math.radians(p1), math.radians(p2), math.radians(lon0)
    n = math.log(math.cos(p1)/math.cos(p2)) / math.log(math.tan(math.pi/4+p2/2)/math.tan(math.pi/4+p1/2))
    F = math.cos(p1) * math.tan(math.pi/4+p1/2)**n / n
    rho0 = F / math.tan(math.pi/4 + 0/2)**n
    def f(lon, lat):
        l, ph = math.radians(lon), math.radians(lat)
        rho = F / math.tan(math.pi/4 + ph/2)**n
        th = n * (l - lon0)
        return math.degrees(rho*math.sin(th)), math.degrees(rho0 - rho*math.cos(th))
    return f

def base_for(fn, xrange=None, yrange=None):
    """Build raster in projected coords at SCALE px/unit by inverse-sampling.
    fn must be invertible numerically: use forward projection of a dense grid instead:
    project every equirect base pixel to projected coords and splat (nearest, keeping max)."""
    if xrange is None or yrange is None:
        # determine extent: project frame corners+grid
        lons = np.arange(LON0, LON1+1e-9, 1.0); lats = np.arange(LAT0, LAT1-1e-9, -1.0)
        LO, LA = np.meshgrid(lons, lats)
        pts = [fn(lo, la) for lo, la in zip(LO.ravel(), LA.ravel())]
        xs_ = [p[0] for p in pts]; ys_ = [p[1] for p in pts]
        xrange = (min(xs_)-2, max(xs_)+2); yrange = (min(ys_)-2, max(ys_)+2)
    PW = int((xrange[1]-xrange[0])*SCALE); PH = int((yrange[1]-yrange[0])*SCALE)
    out = np.zeros((PH, PW), bool)
    yy, xx = np.mgrid[0:BH, 0:BW]
    lon = LON0 + xx/SCALE; lat = LAT0 - yy/SCALE
    PX = np.zeros_like(lon); PY = np.zeros_like(lat)
    for i in range(lon.shape[0]):
        for j in range(lon.shape[1]):
            px_, py_ = fn(lon[i, j], lat[i, j])
            PX[i, j] = (px_ - xrange[0])*SCALE
            PY[i, j] = (yrange[1] - py_)*SCALE
    PXi = np.round(PX).astype(int); PYi = np.round(PY).astype(int)
    ok = (PXi >= 0) & (PXi < PW) & (PYi >= 0) & (PYi < PH)
    out[PYi[ok], PXi[ok]] = base_arr[ok]
    return out, xrange, yrange

def render(t, baseP, xrange, yrange):
    sx, sy, tx, ty = t
    im = Image.fromarray((baseP*255).astype('uint8'))
    a = SCALE/sx; c = -SCALE*tx/sx - xrange[0]*SCALE
    e = SCALE/sy; f = SCALE*yrange[1] - SCALE*ty/sy
    return np.asarray(im.transform((W, H), Image.AFFINE, (a, 0, c, 0, e, f), resample=Image.BILINEAR)) > 127

def score(t, baseP, xr, yr):
    arr = render(t, baseP, xr, yr)
    inter = (arr & img_land & ~excl).sum()
    union = ((arr | img_land) & ~excl).sum()
    return inter / union if union else 0

def fit(fn, coarse_only=False, tag=''):
    baseP, xr, yr = base_for(fn)
    best = (-1, None)
    for sx in np.arange(7.0, 10.01, 0.3):
        for sy in np.arange(7.0, 10.01, 0.3):
            for Lx in np.arange(-38.0, -17.9, 2.0):
                for Ly in np.arange(47.0, 32.9, -1.5):
                    t = (sx, sy, -sx*Lx, sy*fn(0, Ly)[1])
                    v = score(t, baseP, xr, yr)
                    if v > best[0]:
                        best = (v, t)
    if not coarse_only:
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
                                v = score(t, baseP, xr, yr)
                                if v > cur[0] + 1e-6:
                                    cur = (v, t); improved = True
        best = cur
    print(f'{tag}: IoU={best[0]:.4f} t={np.round(best[1],3)}')
    return best

results = {}
# cylindric references
results['miller'] = fit(proj_miller, tag='miller')

# conic sweep (coarse first, refine top-3)
conic_cfgs = []
for p1, p2 in [(0,20),(5,25),(10,30),(15,35),(-10,10),(20,40),(-5,15),(0,25),(10,25),(5,20)]:
    for lon0 in [10, 15, 20, 25]:
        conic_cfgs.append((p1, p2, lon0))
conic_scores = []
for p1, p2, lon0 in conic_cfgs:
    fn = make_lcc(p1, p2, lon0)
    b = fit(fn, coarse_only=True, tag=f'lcc({p1},{p2},{lon0})')
    conic_scores.append((b[0], (p1, p2, lon0), b[1]))
conic_scores.sort(reverse=True)
print('\ntop conic configs:', [(round(s,4), cfg) for s, cfg, _ in conic_scores[:5]])

for v0, cfg, t0 in conic_scores[:3]:
    p1, p2, lon0 = cfg
    fn = make_lcc(p1, p2, lon0)
    b = fit(fn, tag=f'lcc-refine({p1},{p2},{lon0})')
    results[f'lcc-{p1}-{p2}-{lon0}'] = b

bestname, (v, t) = max(results.items(), key=lambda kv: kv[1][0])
print('\nOVERALL BEST:', bestname, 'IoU=%.4f' % v, np.round(t, 3))
json.dump({'projection': bestname, 't': list(map(float, t)), 'iou': float(v)}, open(BASE + '/data/calib.json', 'w'), indent=1)
