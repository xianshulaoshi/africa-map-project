#!/usr/bin/env python3
"""Locate legend swatch rectangles precisely, group into rows, annotate, cross-check."""
import numpy as np
from PIL import Image, ImageDraw

IMG = '/Users/weiaiwen1/Desktop/my-space/map-project/source-map.png'
img = Image.open(IMG).convert('RGB')
a = np.asarray(img).astype(int)
H, W = a.shape[:2]

# Legend region scan: bottom-left quadrant. Swatches = small flat saturated rectangles.
# 1) find saturated pixels (channel spread > 25) in x 20..300, y 400..690
reg = a[400:690, 20:300]
spread = reg.max(axis=2) - reg.min(axis=2)
sat = spread > 25
# exclude text (dark):
nonsatdark = reg.mean(axis=2) < 120
sat &= ~nonsatdark
ys, xs = np.nonzero(sat)
print('saturated px in legend region:', len(xs))

# 2) quantize colors
cols = reg[ys, xs]
q = (cols // 12 * 12)
from collections import Counter
cnt = Counter(map(tuple, q))
main = [(c, n) for c, n in cnt.most_common(30) if n > 250]
print('main legend colors:')
for c, n in main:
    print('  #%02x%02x%02x' % tuple(c), n)

# 3) for each main color, bounding boxes of its pixels (cluster by position)
def bboxes(color, tol=14):
    m = (np.abs(reg - np.array(color)).sum(axis=2) < tol)
    m2 = np.zeros_like(m)
    m2[:] = m
    boxes = []
    from collections import deque
    seen = np.zeros(m2.shape, bool)
    mm, nn = m2.shape
    myy, mxx = np.nonzero(m2)
    for y0, x0 in zip(myy, mxx):
        if seen[y0, x0]: continue
        q = deque([(y0, x0)]); seen[y0, x0] = True
        miny=maxy=y0; minx=maxx=x0; area=0
        while q:
            y, x = q.popleft(); area += 1
            miny=min(miny,y);maxy=max(maxy,y);minx=min(minx,x);maxx=max(maxx,x)
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    ny, nx = y+dy, x+dx
                    if 0<=ny<mm and 0<=nx<nn and m2[ny,nx] and not seen[ny,nx]:
                        seen[ny,nx]=True; q.append((ny,nx))
        if area > 150 and (maxy-miny) < 60 and (maxx-minx) < 90:  # swatch-sized blobs
            boxes.append((miny+400, minx+20, maxy+400, maxx+20, area))
    return boxes

all_boxes = []
for c, n in main:
    for b in bboxes(c):
        all_boxes.append((b, c))
# sort by y then x
all_boxes.sort(key=lambda t: (round(t[0][0]/6), t[0][1]))
print('\nswatch-like rectangles (y0,x0,y1,x1,area) + color:')
rows = {}
for (y0, x0, y1, x1, area), c in all_boxes:
    print(f'  y[{y0}-{y1}] x[{x0}-{x1}] area={area} #%02x%02x%02x' % tuple(c))
    rows.setdefault(round(y0/6)*6, []).append((x0, c))

# 4) annotated image
im2 = img.crop((20, 400, 300, 690)).resize((280*4, 290*4), Image.LANCZOS)
d = ImageDraw.Draw(im2)
for i, (ry, items) in enumerate(sorted(rows.items())):
    d.text((5, (ry-400)*4 + 2), f'ROW{i}', fill=(255, 0, 0))
im2.save('/Users/weiaiwen1/Desktop/my-space/map-project/legend-annotated.png')
print('saved legend-annotated.png with', len(rows), 'rows')
