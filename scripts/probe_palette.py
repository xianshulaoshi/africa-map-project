#!/usr/bin/env python3
"""Detect graticule grid lines in the original map to identify the projection,
and sample territory fill colors (palette) via rough geographic anchors."""
import numpy as np
from PIL import Image
import sys

IMG = '/Users/weiaiwen1/Desktop/my-space/map-project/source-map.png'
img = np.asarray(Image.open(IMG).convert('RGB')).astype(int)
H, W = img.shape[:2]
print(f'image {W}x{H}')

# --- 1. Grid line detection: look for near-horizontal/vertical lines of darker gray ---
gray = img.mean(axis=2)
dark = gray < 200  # grid lines & borders & text

# Sum over a central column band (avoid land text) for horizontal lines
colband = dark[:, int(W*0.30):int(W*0.42)]
row_scores = colband.mean(axis=1)
# Sum over central row band for vertical lines
rowband = dark[int(H*0.35):int(H*0.55), :]
col_scores = rowband.mean(axis=0)

def peaks(scores, min_dist=6, topk=30):
    idx = np.argsort(scores)[::-1]
    picked = []
    for i in idx:
        if scores[i] < 0.35:  # must be dark across >35% of band
            break
        if all(abs(i - p) >= min_dist for p in picked):
            picked.append(i)
        if len(picked) >= topk:
            break
    return sorted(picked)

hp = peaks(row_scores)
vp = peaks(col_scores)
print('candidate horizontal grid/border lines (y px):', hp)
print('candidate vertical grid/border lines (x px):', vp)

# --- 2. Palette sampling via rough anchors (equirect assumption with manual frame guess) ---
# Frame guess: lon -25..51 (76 deg) maps to x 55..W-30; lat 39..-36 (75 deg) maps y 30..H-60
def lonlat_to_px(lon, lat):
    x = 55 + (lon + 25) / 76 * (W - 85)
    y = 30 + (39 - lat) / 75 * (H - 90)
    return int(round(x)), int(round(y))

anchors = [
    ('FR_Sudan',   -10.0, 15.0, 'France'),
    ('Algerie',      3.5, 27.5, 'France'),
    ('Gabon',       11.5, -1.0, 'France'),
    ('Nigeria',      8.0,  9.5, 'UK'),
    ('Rhodesia',    28.5, -17.5, 'UK'),
    ('Egypte',      30.0, 26.0, 'UK/BE'),
    ('CongoBelge',  23.0, -2.5, 'Belgium'),
    ('Angola',      17.0, -12.0, 'Portugal'),
    ('Mozambique',  35.0, -17.0, 'Portugal'),
    ('GermanEA',    33.0, -6.5, 'Germany'),
    ('GermanSW',    17.5, -22.0, 'Germany'),
    ('Cameroun',    13.0,   5.0, 'Germany'),
    ('Libye',       20.0, 26.5, 'Italy'),
    ('Liberia',     -9.4,  6.3, 'indep'),
    ('Abyssinie',   39.5,  9.0, 'indep'),
    ('Maroc',       -6.5, 32.0, 'Franceprot'),
    ('Senegal',    -15.5, 14.5, 'France'),
    ('TchadAEF',    19.0, 15.0, 'France'),
]
from collections import Counter
for name, lon, lat, expect in anchors:
    x, y = lonlat_to_px(lon, lat)
    if not (0 <= x-15 and x+15 < W and 0 <= y-15 and y+15 < H):
        print(f'{name:12s} anchor out of image'); continue
    box = img[y-12:y+12, x-12:x+12].reshape(-1, 3)
    # modal color quantized to 8
    q = (box // 8 * 8)
    cnt = Counter(map(tuple, q))
    top = cnt.most_common(3)
    tops = ' '.join('#%02x%02x%02x:%d' % (c[0], c[1], c[2], n) for c, n in top)
    print(f'{name:12s} expect={expect:10s} px=({x},{y}) {tops}')
