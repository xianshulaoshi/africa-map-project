#!/usr/bin/env python3
"""Objective fidelity check: rasterize the page's territory geometry (Mercator,
same frame as app.js) and compute land-mask IoU vs the original map image."""
import numpy as np, json, math
from PIL import Image, ImageDraw
from shapely.geometry import Polygon as SPoly

BASE = '/Users/weiaiwen1/Desktop/my-space/map-project'
W, H = 673, 710
LON0, LAT0, LON1, LAT1 = -26, 43, 58, -37
RAD = math.pi / 180

def mercY(lat): return math.log(math.tan(math.pi / 4 + lat * RAD / 2)) / RAD
S = min(W / (LON1 - LON0), H / (mercY(LAT0) - mercY(LAT1))) * 0.985
def to_px(lon, lat):
    return ((lon - LON0) * S + (W - (LON1 - LON0) * S) / 2,
            (mercY(LAT0) - mercY(lat)) * S + (H - (mercY(LAT0) - mercY(LAT1)) * S) / 2)

txt = open(BASE + '/data.js').read()
data = json.loads(txt[txt.index('{'):txt.rindex('}') + 1])

im = Image.new('1', (W, H), 0)
dr = ImageDraw.Draw(im)
n = 0
for t in data['territories']:
    for poly in t['geometry']:
        try:
            sp = SPoly(poly[0], poly[1:])
            if sp.area < 0.001: continue
            sp = sp.simplify(0.02)
            rings = [sp.exterior.coords] + [r.coords for r in sp.interiors]
            pj = []
            for ring in rings:
                pj.append([to_px(x, y) for x, y in ring])
            dr.polygon(pj[0], fill=1)
            for hole in pj[1:]:
                dr.polygon(hole, fill=0)
            n += 1
        except Exception as e:
            pass
mine = np.asarray(im, bool)
print('rendered polys:', n, 'land px:', int(mine.sum()))
Image.fromarray((mine * 255).astype('uint8')).save(BASE + '/my-land-mask.png')

# original land mask (row-ocean model, as calibrated before)
img = np.asarray(Image.open(BASE + '/source-map.png').convert('RGB')).astype(int)
oH, oW = img.shape[:2]
rowOcean = np.zeros((oH, 3))
edgeL = img[:, 2:18].reshape(oH, -1, 3); edgeR = img[:, oW - 18:oW - 2].reshape(oH, -1, 3)
for y in range(oH):
    s = np.vstack([edgeL[y], edgeR[y]])
    ok = (s[:, 2] > s[:, 0] + 3) & (s.mean(axis=1) > 130)
    s2 = s[ok] if ok.sum() > 10 else s
    rowOcean[y] = np.median(s2, axis=0)
for _ in range(3):
    rowOcean[1:-1] = (rowOcean[:-2] + rowOcean[1:-1] + rowOcean[2:]) / 3
diff = np.abs(img - rowOcean[:, None, :]).max(axis=2)
orig_ocean = diff < 14
orig_land = ~orig_ocean
from scipy import ndimage
lab, nlab = ndimage.label(orig_land, structure=np.ones((3, 3)))
sizes = ndimage.sum(orig_land, lab, range(1, nlab + 1))
keep = np.zeros(nlab + 1, bool); keep[1:] = sizes > 150
orig_land = keep[lab]
# exclude legend box
excl = np.zeros((oH, oW), bool); excl[470:675, 0:235] = True

# optimize alignment (translation only, small search around identity)
best = (0, (0, 0))
for dy in range(-14, 15, 2):
    for dx in range(-14, 15, 2):
        shifted = np.roll(np.roll(mine, dy, axis=0), dx, axis=1)
        inter = (shifted & orig_land & ~excl).sum()
        union = ((shifted | orig_land) & ~excl).sum()
        v = inter / union if union else 0
        if v > best[0]: best = (v, (dx, dy))
print('IoU at best shift (%d,%d): %.3f' % (best[1][0], best[1][1], best[0]))
# also unshifted
inter = (mine & orig_land & ~excl).sum(); union = ((mine | orig_land) & ~excl).sum()
print('IoU no shift: %.3f' % (inter / union))
