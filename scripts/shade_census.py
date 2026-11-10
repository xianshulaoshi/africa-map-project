#!/usr/bin/env python3
"""Definitive shade census: for each territory, scan a dense interior grid through the
calibrated transform, classify every pixel against the decoded legend palette, and
report shade fractions overall + per latitude band. Robust: inset from borders,
reject non-palette pixels (text/labels/ocean)."""
import numpy as np, json, math
from PIL import Image
from shapely.geometry import shape as shp_shape, Point, box
from shapely.ops import unary_union
from collections import Counter, defaultdict

BASE = '/Users/weiaiwen1/Desktop/my-space/map-project'
img = np.asarray(Image.open(BASE + '/source-map.png').convert('RGB')).astype(int)
H, W = img.shape[:2]
calib = json.load(open(BASE + '/data/calib.json'))
sx, sy, tx, ty = calib['t']
RAD = math.pi / 180
def mercY(lat): return math.log(math.tan(math.pi / 4 + lat * RAD / 2)) / RAD
def to_px(lon, lat): return (lon * sx + tx, ty - mercY(lat) * sy)

# palette (quantized bins ±14 tolerance)
FAMILY = {
  'FR-deep': (0x08, 0x78, 0xb8), 'FR-mid': (0x48, 0x90, 0xc8), 'FR-light': (0x88, 0xb8, 0xe0),
  'UK-deep': (0xa0, 0x60, 0xa0), 'UK-mid': (0xc0, 0x88, 0xb8), 'UK-light': (0xe0, 0xa8, 0xc8),
  'DE-mid': (0x68, 0x98, 0x78), 'DE-light': (0x90, 0xb8, 0xa0),
  'BE-gold': (0xd0, 0xa8, 0x38),
  'PT-deep': (0x78, 0x88, 0x38), 'PT-mid': (0xa0, 0xa8, 0x58), 'PT-light': (0xc0, 0xd0, 0x80),
  'ES-deep': (0xb8, 0x68, 0x18), 'ES-mid': (0xd8, 0x88, 0x30), 'ES-light': (0xf8, 0xb0, 0x58),
  'IT-mid': (0xb0, 0xa0, 0x68), 'IT-light': (0xd0, 0xc0, 0x88),
  'OT-teal': (0x88, 0xc8, 0xb8),
  'BOER-gray': (0x90, 0x90, 0xa0),
  'INDEP': (0xe8, 0xe6, 0xe0),
}
FAM_LIST = list(FAMILY.items())
def classify(rgb):
    for name, c in FAM_LIST:
        if abs(rgb[0]-c[0]) + abs(rgb[1]-c[1]) + abs(rgb[2]-c[2]) < 42:
            return name
    return None

txt = open(BASE + '/data.js').read()
data = json.loads(txt[txt.index('{'):txt.rindex('}')+1])

POWER_FAM = {'fr': 'FR', 'uk': 'UK', 'de': 'DE', 'be': 'BE', 'pt': 'PT', 'es': 'ES', 'it': 'IT', 'ot': 'OT', 'indep': 'INDEP'}

print('%-14s %-8s %-44s %s' % ('territory', 'power', 'shade fractions (family)', 'note'))
results = {}
for t in data['territories']:
    fam = POWER_FAM[t['power']]
    # build shapely union, inset ~0.12deg to avoid borders/neighbors
    try:
        geom = unary_union([shp_shape({'type':'Polygon','coordinates': p}) if len(p)==1 else shp_shape({'type':'Polygon','coordinates':p}) for p in t['geometry']])
    except Exception:
        continue
    try:
        geom = geom.buffer(-0.1, join_style=2)
    except Exception:
        pass
    if geom.is_empty: continue
    minx, miny, maxx, maxy = geom.bounds
    counter = Counter(); bands = defaultdict(Counter)
    ny = max(4, int((maxy-miny)/0.15))
    for i in range(ny+1):
        lat = miny + (maxy-miny)*i/ny
        xs = np.arange(minx, maxx+0.001, 0.08)
        for lon in xs:
            if not geom.contains(Point(float(lon), float(lat))): continue
            x, y = to_px(float(lon), float(lat))
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < W and 0 <= yi < H): continue
            cl = classify(img[yi, xi])
            if cl and (cl.startswith(fam) or cl in ('BOER-gray','INDEP')):
                counter[cl] += 1
                bands[round(lat*2)/2][cl] += 1  # 0.5° bands
    tot = sum(counter.values())
    if tot < 20:
        results[t['id']] = (counter, bands, tot)
        print('%-14s %-8s %-44s %s' % (t['id'], fam, '(too few px: %d)' % tot, ''))
        continue
    frac = ', '.join(f'{k}:{v*100//tot}%' for k, v in counter.most_common(4))
    results[t['id']] = (counter, bands, tot)
    print('%-14s %-8s %-44s' % (t['id'], fam, frac))

# per-band detail for the multi-shade candidates
json.dump({k: { 'tot': v[2], 'bands': {str(b): dict(c) for b, c in sorted(v[1].items())} }
           for k, v in results.items()}, open(BASE + '/data/shade_census.json', 'w'))
print('\nsaved data/shade_census.json')
