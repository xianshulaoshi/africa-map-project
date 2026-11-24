#!/usr/bin/env python3
"""Step 7 of the source-map trace: the frame of the sheet.

Title box, legend (swatches, boundary symbols, shadow) and scale bar, with
the geometry and colours measured on the scan.  The lettering of these
elements is fitted by trace_labels.py like any other label.

Output: data/trace_frame.json  (list of SVG primitives for the page)
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data')
COLORS = json.load(open(os.path.join(OUT, 'trace_colors.json')))

# legend swatches: row label -> (y0, [(x0, colour class), ...]) ; cells are 33 x 16
CELL_X = {1870: 12.5, 1900: 45.5, 1914: 78.5}
ROWS = [
    ('française', 444, [(1870, 'fr-d'), (1900, 'fr-m'), (1914, 'fr-l')]),
    ('britannique', 461, [(1870, 'uk-d'), (1900, 'uk-m'), (1914, 'uk-l')]),
    ('allemande', 477, [(1900, 'de-m'), (1914, 'de-l')]),
    ('belge', 494, [(1900, 'be')]),
    ('portugaise', 510, [(1870, 'pt-d'), (1900, 'pt-m'), (1914, 'pt-l')]),
    ('espagnole', 526, [(1870, 'es-d'), (1900, 'es-m'), (1914, 'es-l')]),
    ('italienne', 543, [(1900, 'it-m'), (1914, 'it-l')]),
    ('turque', 559, [(1900, 'ot')]),
    ('des Boers', 575, [(1870, 'boer')]),
    ('État indépendant', 608, [(1914, 'indep')]),
]


def main():
    F = []
    # title: white panel with a thin gold border and a thick gold bar on the left
    F.append(dict(k='rect', x=14, y=11, w=344, h=34, f='#c9c185'))
    F.append(dict(k='rect', x=20, y=13, w=336, h=30, f='#ffffff'))
    # legend: soft shadow, white panel
    F.append(dict(k='shadow', x=8, y=429, w=201, h=277))
    F.append(dict(k='rect', x=7, y=427, w=200, h=276, f='#ffffff'))
    for name, y, cells in ROWS:
        for year, cls in cells:
            F.append(dict(k='rect', x=CELL_X[year], y=y + 0.5, w=33, h=15.5, f=COLORS[cls]))
    # 1914 border: blue strip with a white line ; 1870 border: pink strip, white dashes
    F.append(dict(k='rect', x=78.5, y=633.5, w=33, h=8.5, f=COLORS['fr-d']))
    F.append(dict(k='path', d='M79.5 637.5H110.5', s='#ffffff', sw=1.1, cap='butt'))
    F.append(dict(k='rect', x=78.5, y=655, w=33, h=9, f=COLORS['uk-m']))
    F.append(dict(k='path', d='M79.5 659.5H110.5', s='#ffffff', sw=2.2, dash='2.6 5.6', cap='butt'))
    # scale bar: 1 000 km
    F.append(dict(k='path', d='M231 700.6H299', s='#120f17', sw=1.7, cap='butt'))
    json.dump(F, open(os.path.join(OUT, 'trace_frame.json'), 'w'), ensure_ascii=False)
    print('frame: %d primitives' % len(F))


if __name__ == '__main__':
    main()
