#!/usr/bin/env python3
"""Crop the legend area and extract swatch colors as ordered rows."""
import numpy as np
from PIL import Image

IMG = '/Users/weiaiwen1/Desktop/my-space/map-project/source-map.png'
img = Image.open(IMG).convert('RGB')
W, H = img.size
# Legend is bottom-left, roughly x 30..250, y 430..640
crop = img.crop((25, 420, 280, 660))
crop = crop.resize((crop.width*3, crop.height*3), Image.LANCZOS)
crop.save('/Users/weiaiwen1/Desktop/my-space/map-project/legend-crop.png')
print('saved legend-crop.png', crop.size)

a = np.asarray(crop).astype(int)
# find distinct flat colors (saturated, non-text)
from collections import Counter
q = (a // 12 * 12).reshape(-1, 3)
cnt = Counter(map(tuple, q))
# filter grayish/text and rare
def sat(c):
    mx, mn = max(c), min(c)
    return mx - mn
for c, n in cnt.most_common(40):
    if n < 60: break
    print('#%02x%02x%02x' % tuple(c), n, 'sat', sat(c))
