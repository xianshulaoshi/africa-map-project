#!/usr/bin/env python3
"""Objective pixel-level comparison: classify shades at identical geographic grid
points in (a) the original map and (b) a screenshot of the rendered page, using
each image's own calibrated projection. Reports per-territory agreement."""
import numpy as np, json, math
from PIL import Image
from shapely.geometry import shape as shp_shape, Point
from shapely.ops import unary_union
from collections import Counter

BASE = '/Users/weiaiwen1/Desktop/my-space/map-project'
orig = np.asarray(Image.open(BASE + '/source-map.png').convert('RGB')).astype(int)
shot = np.asarray(Image.open(BASE + '/shot-census.png').convert('RGB')).astype(int)

# --- original mapping (calib.json, mercator) ---
calib = json.load(open(BASE + '/data/calib.json'))
osx, osy, otx, oty = calib['t']
RAD = math.pi / 180
def mercY(lat): return math.log(math.tan(math.pi/4 + lat*RAD/2))/RAD
def orig_px(lon, lat): return (lon*osx + otx, oty - mercY(lat)*osy)

# --- my page mapping (view state captured from the live page) ---
V = {'k': 1, 'tx': 0, 'ty': 0, 'S': 8.950923608111188, 'W': 1110, 'H': 796,
     'svgX': 330, 'svgY': 66, 'dpr': 2}
LON0, LAT0, LON1 = -26, 43, 58
U_W = LON1 - LON0
U_H = mercY(LAT0) - mercY(-37)
def my_px(lon, lat):
    ux, uy = lon - LON0, mercY(LAT0) - mercY(lat)
    sc = V['S'] * V['k']
    x = V['W']/2 + (ux + V['tx'] - U_W/2) * sc
    y = V['H']/2 + (uy + V['ty'] - U_H/2) * sc
    return (x * V['dpr'] + V['svgX'] * V['dpr'], y * V['dpr'] + V['svgY'] * V['dpr'])

FAMILY = {
  'FR-deep': (0x08,0x78,0xb8),'FR-mid': (0x48,0x90,0xc8),'FR-light': (0x88,0xb8,0xe0),
  'UK-deep': (0xa0,0x60,0xa0),'UK-mid': (0xc0,0x88,0xb8),'UK-light': (0xe0,0xa8,0xc8),
  'DE-mid': (0x68,0x98,0x78),'DE-light': (0x90,0xb8,0xa0),
  'BE': (0xd0,0xa8,0x38),
  'PT-deep': (0x78,0x88,0x38),'PT-mid': (0xa0,0xa8,0x58),'PT-light': (0xc0,0xd0,0x80),
  'ES-deep': (0xb8,0x68,0x18),'ES-mid': (0xd8,0x88,0x30),'ES-light': (0xf8,0xb0,0x58),
  'IT-mid': (0xb0,0xa0,0x68),'IT-light': (0xd0,0xc0,0x88),
  'OT': (0x88,0xc8,0xb8), 'GRAY': (0x90,0x90,0xa0),
}
FAM = list(FAMILY.items())
def classify(rgb):
    for name, c in FAM:
        if abs(rgb[0]-c[0]) + abs(rgb[1]-c[1]) + abs(rgb[2]-c[2]) < 36:
            return name
    return None

def fam_norm(name):
    return name.split('-')[0] if name else None

txt = open(BASE + '/data.js').read()
data = json.loads(txt[txt.index('{'):txt.rindex('}')+1])
terr = {t['id']: t for t in data['territories']}
OH, OW = orig.shape[:2]
SH, SW = shot.shape[:2]

print('%-13s %-6s %-7s %-28s %s' % ('territory','n','agree','mine vs orig (top diffs)','status'))
bad = []
for t in data['territories']:
    try:
        geom = unary_union([shp_shape({'type':'Polygon','coordinates': p}) for p in t['geometry']]).buffer(-0.1, join_style=2)
    except Exception:
        continue
    if geom.is_empty: continue
    minx, miny, maxx, maxy = geom.bounds
    agree = 0; tot = 0; diffs = Counter()
    for lat in np.arange(miny, maxy, 0.15):
        for lon in np.arange(minx, maxx, 0.1):
            if not geom.contains(Point(float(lon), float(lat))): continue
            ox, oy = orig_px(float(lon), float(lat))
            mx, my = my_px(float(lon), float(lat))
            oi = (int(round(oy)), int(round(ox)))
            mi = (int(round(my)), int(round(mx)))
            if not (0 <= oi[0] < OH and 0 <= oi[1] < OW and 0 <= mi[0] < SH and 0 <= mi[1] < SW):
                continue
            oc = classify(orig[oi]); mc = classify(shot[mi])
            if oc is None or mc is None: continue
            # compare deep/mid/light level within family (and gray/ot as own)
            def lvl(c):
                f = c.split('-')[0]
                return f + '-' + (c.split('-')[1] if '-' in c else 'x')
            tot += 1
            if lvl(oc) == lvl(mc): agree += 1
            else: diffs[f'{oc}->{mc}'] += 1
    if tot < 10: continue
    pct = agree * 100 // tot
    d = ' '.join(f'{k}:{v}' for k, v in diffs.most_common(2))
    status = 'OK' if pct >= 75 else ('~' if pct >= 55 else 'BAD')
    if pct < 75: bad.append(t['id'])
    print('%-13s %-6d %-7s %-28s %s' % (t['id'], tot, f'{pct}%', d, status))
print('\nbelow 75%:', bad)
