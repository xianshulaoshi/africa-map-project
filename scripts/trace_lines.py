#!/usr/bin/env python3
"""Step 6 of the source-map trace: line work.

  * rivers: thin blue lines of the scan (blue on a non-blue fill, lighter
    blue on the French blues), thinned to their skeleton and chained into
    polylines;
  * graticule (dotted), Equator (solid) and the two tropics (dashed): their
    positions are read from the scan.

Output: data/trace_lines.json
  lines  [{d, c, w, op, dash?, cap?}, ...]          SVG path data, image pixels
"""
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from shapely.geometry import LineString
from skimage.morphology import skeletonize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data')
CLASSES = ['ctx', 'fr-l', 'fr-m', 'fr-d', 'uk-l', 'uk-m', 'uk-d', 'de-l', 'de-m', 'be',
           'pt-l', 'pt-m', 'pt-d', 'es-l', 'es-m', 'es-d', 'it-l', 'it-m', 'ot', 'indep', 'boer']
SEA, WHITE = -1, -2
RIVER_COLOURS = [(140, 160, 212), (138, 187, 227), (185, 215, 242)]


def num(v):
    s = ('%.1f' % v).rstrip('0').rstrip('.')
    return '0' if s in ('-0', '') else s


def path_of(pts):
    """M x y L ... (absolute, one decimal)."""
    return 'M' + 'L'.join('%s %s' % (num(x), num(y)) for x, y in pts)


def chain(skel):
    """Polylines of a skeleton: walk from every end / junction through the
    chain pixels to the next end / junction."""
    H, W = skel.shape
    pts = {(int(x), int(y)) for y, x in zip(*np.nonzero(skel))}
    nbr = {}
    for (x, y) in pts:
        ns = [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
              if (dx or dy) and (x + dx, y + dy) in pts]
        nbr[(x, y)] = ns
    nodes = [p for p in pts if len(nbr[p]) != 2]
    used = set()
    lines = []

    def walk(a, b):
        line = [a, b]
        used.add((a, b)); used.add((b, a))
        prev, cur = a, b
        while len(nbr[cur]) == 2:
            nxt = nbr[cur][0] if nbr[cur][0] != prev else nbr[cur][1]
            if (cur, nxt) in used:
                break
            used.add((cur, nxt)); used.add((nxt, cur))
            line.append(nxt)
            prev, cur = cur, nxt
        return line

    for n in nodes:
        for b in nbr[n]:
            if (n, b) not in used:
                lines.append(walk(n, b))
    # closed loops without nodes
    for p in pts:
        if any((p, b) not in used for b in nbr[p]):
            a = p
            b = [b for b in nbr[a] if (a, b) not in used][0]
            lines.append(walk(a, b))
    return lines


def rivers():
    rgb = np.asarray(Image.open(os.path.join(ROOT, 'source-map.png')).convert('RGB')).astype(float)
    med = ndi.median_filter(rgb, size=(3, 3, 1))
    cls = np.load(os.path.join(OUT, 'trace_classes.npy'))
    loose = np.load(os.path.join(OUT, 'trace_loose.npy'))
    ink = np.load(os.path.join(OUT, 'trace_ink.npy'))
    colors = json.load(open(os.path.join(OUT, 'trace_colors.json')))
    pal = np.array([[int(colors[c][i:i + 2], 16) for i in (1, 3, 5)] for c in CLASSES], float)
    # every land pixel has the colour of its class as background (non-core
    # pixels inherited the class of the nearest flat pixel).  A river pixel
    # is a blend of that background with a light blue: on a pink, green or
    # yellow fill the line is a muted blue, on the French blues it is the
    # next lighter blue.
    land = cls >= 0
    bgc = pal[np.clip(cls, 0, None)]
    d = rgb - bgc
    best_f = np.zeros(rgb.shape[:2])
    best_r = np.full(rgb.shape[:2], 1e9)
    for rv in RIVER_COLOURS:
        v = np.array(rv) - bgc
        vv = (v * v).sum(-1)
        f = (d * v).sum(-1) / np.maximum(vv, 1)
        r = np.sqrt(((d - f[..., None] * v) ** 2).sum(-1))
        ok = (vv >= 55 ** 2) & (r < best_r)
        best_r = np.where(ok, r, best_r)
        best_f = np.where(ok, f, best_f)
    m = ~ndi.binary_dilation(ink, structure=np.ones((3, 3), bool)) & ~ndi.binary_dilation(
        (cls < 0) | (loose == WHITE), structure=np.ones((5, 5), bool)) & land
    riv = m & (best_f > 0.33) & (best_f < 1.4) & (best_r < 20)
    riv &= ~ndi.binary_opening(riv, structure=np.ones((3, 3), bool))      # rivers are thin
    rgb = med
    # colour of the rivers
    col = np.median(rgb[riv], axis=0) if riv.any() else (120, 180, 232)
    lab, n = ndi.label(riv, structure=np.ones((3, 3), bool))
    sizes = np.bincount(lab.ravel())
    keep = np.zeros(n + 1, bool); keep[1:] = sizes[1:] >= 7
    riv &= keep[lab]
    skel = skeletonize(riv)
    lines, on_french = [], []
    for line in chain(skel):
        if len(line) < 5:
            continue
        pts = [(x + 0.5, y + 0.5) for x, y in line]
        sm = LineString(pts).simplify(0.45, preserve_topology=False)
        if sm.length >= 4:
            on_fr = bool(np.mean(np.isin(cls[tuple(zip(*line))], [1, 2, 3])) > 0.5)
            lines.append([list(p) for p in sm.coords])
            on_french.append(on_fr)
    print('rivers: %d px, %d polylines, colour #%02x%02x%02x' % (int(riv.sum()), len(lines), *[int(v) for v in col]))
    # review image
    vis = np.asarray(Image.open(os.path.join(ROOT, 'source-map.png')).convert('RGB')).copy()
    vis[skel] = (255, 0, 0)
    Image.fromarray(vis).resize((1346, 1420), Image.NEAREST).save(os.path.join(ROOT, 'scratch', 'redraw', 'rivers.png'))
    return lines, '#%02x%02x%02x' % tuple(int(v) for v in col), on_french


def straight_lines():
    """Rows / columns of the graticule, Equator and tropics: the median
    luminance of a row is compared with the rows 2 px above and below."""
    lum = np.asarray(Image.open(os.path.join(ROOT, 'source-map.png')).convert('L')).astype(float)
    med = ndi.median_filter(lum, 3)
    cls = np.load(os.path.join(OUT, 'trace_classes.npy'))
    ink = np.load(os.path.join(OUT, 'trace_ink.npy'))
    clean = (cls == SEA) & ~ndi.binary_dilation(ink, structure=np.ones((7, 7), bool))

    def row_energy(y):
        d = med[y] - (med[y - 2] + med[y + 2]) / 2
        return float(np.mean(np.clip(np.abs(d), 0, 12)[clean[y - 2] & clean[y + 2]])) if (clean[y - 2] & clean[y + 2]).sum() > 200 else 0

    def col_energy(x):
        d = med[:, x] - (med[:, x - 2] + med[:, x + 2]) / 2
        return float(np.mean(np.clip(np.abs(d), 0, 12)[clean[:, x - 2] & clean[:, x + 2]])) if (clean[:, x - 2] & clean[:, x + 2]).sum() > 200 else 0

    rows = [(round(row_energy(y), 2), y) for y in range(6, 704)]
    cols = [(round(col_energy(x), 2), x) for x in range(6, 667)]
    print('rows with line energy (top 24):', sorted(rows, reverse=True)[:24])
    print('cols with line energy (top 24):', sorted(cols, reverse=True)[:24])
    return rows, cols


if __name__ == '__main__':
    lines, col, on_fr = rivers()
    json.dump({'lines': lines, 'colour': col, 'fr': on_fr},
              open(os.path.join(OUT, 'trace_riverlines.json'), 'w'))
    # Equator (solid, y read on the scan) and the two tropics (faint dashes at
    # the height of their labels); the dotted graticule of the scan is too
    # faint to be measured and is not drawn
    straight = [
        dict(d='M2 388.6H671', c='#8ea6c2', w=0.7, op=0.55, cap='butt'),
        dict(d='M2 174.2H671', c='#9aa0a0', w=0.8, op=0.55, dash='3 3', cap='butt'),
        dict(d='M2 617.2H671', c='#9aa0a0', w=0.8, op=0.5, dash='3 3', cap='butt'),
    ]
    json.dump(straight, open(os.path.join(OUT, 'trace_lines.json'), 'w'))
