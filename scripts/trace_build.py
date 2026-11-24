#!/usr/bin/env python3
"""Step 3 of the source-map trace: assemble data.js.

  * polygons of step 2 (trace_geo.json) and the measured class colours
    (trace_colors.json);
  * the water shade as a palette PNG (data URI) of the measured sea colours;
  * the curated territory notes of the previous version of the project
    (data/territories_1914.json);
  * lettering, line work, legend ... produced by the later steps, when their
    files exist (trace_labels_*.json, trace_lines.json, trace_frame.json).

Usage: trace_build.py            write data.js
"""
import base64
import io
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data')

CLASSES = ['ctx', 'fr-l', 'fr-m', 'fr-d', 'uk-l', 'uk-m', 'uk-d', 'de-l', 'de-m', 'be',
           'pt-l', 'pt-m', 'pt-d', 'es-l', 'es-m', 'es-d', 'it-l', 'it-m', 'ot', 'indep', 'boer']
WATER_COLORS = 160

# territories that were not separate objects of the previous version
EXTRA_META = {
    'transvaal': dict(name='(TRANSVAAL)', zh='德兰士瓦（布尔）', power='boer', status='1870 年布尔共和国',
                      modern=['南非'], area=None, note='1914 年前并入南非联邦；图中斜线区为 1870 年名称'),
    'orange': dict(name='ORANGE', zh='奥兰治自由邦（布尔）', power='boer', status='1870 年布尔共和国',
                   modern=['南非'], area=None, note='图中未标注名称，灰色"des Boers / Afrikaners"图例色'),
    'walvis-bay': dict(name='Walvis Bay', zh='沃尔维斯湾', power='uk', status='英国飞地（好望角殖民地）',
                       modern=['纳米比亚'], area=None, note='图中以粉红色小方块标出，被德属西南非洲包围'),
    'swaziland': dict(name='(SWAZILAND)', zh='斯威士兰', power='uk', status='1870 年独立王国',
                      modern=['斯威士兰'], area=None, note='图中虚线范围为 1870 年名称'),
    'zululand': dict(name='(ZULULAND)', zh='祖鲁兰', power='uk', status='1870 年祖鲁王国',
                     modern=['南非'], area=None, note='图中虚线范围为 1870 年名称'),
    'natal': dict(name='(NATAL)', zh='纳塔尔', power='uk', status='1870 年英国殖民地',
                  modern=['南非'], area=None, note='图中虚线范围为 1870 年名称'),
    'basutoland': dict(name='(BASUTOLAND)', zh='巴苏陀兰', power='uk', status='1870 年后为英国保护地',
                       modern=['莱索托'], area=None, note='图中虚线范围为 1870 年名称'),
    'kaffraria': dict(name='(KAFFRARIA)', zh='卡弗拉里亚', power='uk', status='1870 年英国属地',
                      modern=['南非'], area=None, note='图中虚线范围为 1870 年名称'),
}
# names of the previous version that differ from the lettering of the scan
RENAME = {'soudan-fr': 'SOUDAN FRANÇAIS'}


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def water_uri():
    """Sea shade as a palette PNG: the measured colours of the trusted water
    pixels (median filtered), quantised to the WATER_COLORS most frequent
    ones; under the land the field is covered anyway and set to the open
    ocean colour."""
    sea = np.load(os.path.join(OUT, 'trace_sea.npy'))
    med = ndi.median_filter(np.asarray(Image.open(os.path.join(ROOT, 'source-map.png')).convert('RGB')), size=(3, 3, 1))
    q = (med[sea] // 4 * 4 + 2).astype(np.int64)
    cols_q, counts = np.unique(q, axis=0, return_counts=True)
    order = np.argsort(-counts)[:WATER_COLORS]
    pal = cols_q[order].astype(float)
    d = ((q[:, None, :] - pal[None, :, :]) ** 2).sum(-1)
    idx = np.zeros(sea.shape, np.uint8)
    idx[sea] = d.argmin(1).astype(np.uint8)
    flat = pal[0]                                      # most frequent = open ocean
    rgb = np.zeros(sea.shape + (3,), np.uint8)
    rgb[:] = flat.astype(np.uint8)
    rgb[sea] = pal[idx[sea]].astype(np.uint8)
    img = Image.fromarray(idx, 'P')
    img.putpalette([int(v) for c in pal for v in c] + [0] * (768 - len(pal) * 3))
    buf = io.BytesIO()
    img.save(buf, 'PNG', optimize=True)
    err = np.abs(rgb.astype(int) - med.astype(int))[sea].mean()
    print('water layer: %d colours, %d bytes, mean |dRGB| on water %.2f' % (len(pal), len(buf.getvalue()), err))
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


# text styles: id -> (font file, css weight, italic, halo width in px)
STYLE_DEF = {
    'T':  ('AsapCondensed-Bold', 700, 0, 0),               # territory names
    'TW': ('IBMPlexSansCondensed-Regular', 400, 0, 0),      # MAROC / ALGÉRIE
    'TT': ('AsapCondensed-Bold', 700, 0, 0),              # title (horizontally compressed)
    'P':  ('IBMPlexSansCondensed-Regular', 400, 0, 0),      # names in brackets
    'C':  ('Questrial-Regular', 400, 0, 1.3),               # cities
    'CI': ('Questrial-Regular', 400, 1, 1.3),               # cities, italic
    'R':  ('Questrial-Regular', 400, 0, 1.2),               # letter-spaced capitals
    'W':  ('Questrial-Regular', 400, 1, 1.2),               # water names
    'WI': ('Questrial-Regular', 400, 0, 1.2),
    'D':  ('Questrial-Regular', 400, 0, 1.0),               # 20°O, 10°S ...
    'H':  ('Lato-Bold', 700, 0, 0),                         # legend heading
    'L':  ('Questrial-Regular', 400, 0, 0),               # legend entries
    'LI': ('Questrial-Regular', 400, 1, 0),               # (NATAL) sample
    'Y':  ('Lato-Bold', 700, 0, 0),                         # years on the legend swatch
}
FONTS = {  # file -> (css family, weight)
    'IBMPlexSansCondensed-Regular': ('"Scan Sans Condensed"', 400),
    'IBMPlexSansCondensed-SemiBold': ('"Scan Sans Condensed"', 600),
    'Questrial-Regular': ('"Scan Sans"', 400),
    'Lato-Bold': ('"Scan Sans"', 700),
    'AsapCondensed-Bold': ('"Scan Narrow"', 700),
}
STYLE_COLOUR = {'Y': '#ffffff'}                  # measured per style otherwise
HALO = 'rgba(250,251,252,.78)'


def subset_woff(path, text):
    """Subset of the font holding `text`, as WOFF bytes."""
    from fontTools import subset
    from fontTools.ttLib import TTFont
    font = TTFont(path)
    sub = subset.Subsetter()
    sub.populate(text=text)
    sub.subset(font)
    font.flavor = 'woff'
    buf = io.BytesIO()
    font.save(buf)
    return buf.getvalue()


def fonts_and_styles(labels):
    files = {}
    for l in labels:
        files.setdefault(STYLE_DEF[l['s']][0], set()).update(l['t'])
    fonts = []
    for fname, chars in sorted(files.items()):
        path = os.path.join(ROOT, 'scratch', 'redraw', 'fonts', fname + '.ttf')
        b64 = base64.b64encode(subset_woff(path, ''.join(sorted(chars)))).decode('ascii')
        fam, wght = FONTS[fname]
        fonts.append([fam, wght, b64])
    index = {f: i for i, f in enumerate(sorted(files))}
    # colour of every style: median of the well fitted labels
    def hexmed(cs):
        v = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in cs])
        return '#%02x%02x%02x' % tuple(int(x) for x in np.median(v, 0))
    colours = {}
    for st in STYLE_DEF:
        cs = [l['colour'] for l in labels if l['s'] == st and l.get('score', 0) >= 0.6 and l.get('colour')]
        colours[st] = hexmed(cs) if cs else '#1d1d1f'
    colours.update(STYLE_COLOUR)
    styles = {}
    for st, (fname, wght, ital, halo) in STYLE_DEF.items():
        styles[st] = dict(ff=index[fname], w=wght, i=ital, c=colours[st], h=HALO if halo else 0, hw=halo)
    return fonts, styles


def river_lines(riv):
    """River polylines in two stroke colours: light on the French blues,
    muted blue on every other fill."""
    out = {}
    for pts, on_fr in zip(riv['lines'], riv.get('fr', [False] * len(riv['lines']))):
        key = 'fr' if on_fr else 'other'
        out.setdefault(key, []).append('M' + 'L'.join('%.1f %.1f' % (x, y) for x, y in pts))
    lines = []
    if out.get('other'):
        lines.append(dict(d=''.join(out['other']), c='#7a97d6', w=0.85, op=0.85, cap='round'))
    if out.get('fr'):
        lines.append(dict(d=''.join(out['fr']), c='#b4d6ef', w=0.85, op=0.9, cap='round'))
    return lines


def main():
    geo = load('trace_geo.json')
    colors = load('trace_colors.json')
    old = load('territories_1914.json')

    meta = {}
    for t in old['territories']:
        m = {k: t[k] for k in ('id', 'name', 'zh', 'power', 'status', 'modern', 'note', 'area') if k in t}
        m['id'] = 'cameroun' if t['id'] == 'kamerun' else t['id']
        meta[m['id']] = m
    for k, v in EXTRA_META.items():
        meta[k] = dict(id=k, **v)
    for k, v in RENAME.items():
        meta[k]['name'] = v

    terr = sorted(set(p['terr'] for p in geo['patches']) - {'ctx'})
    tix = {t: i for i, t in enumerate(terr)}
    patches = [[CLASSES.index(p['cls']), tix.get(p['terr'], -1), p['d']] for p in geo['patches']]

    data = {
        'W': geo['W'], 'H': geo['H'],
        'classes': CLASSES,
        'colors': [colors[c] for c in CLASSES],
        'water': water_uri(),
        'land': geo['land'],
        'terr': terr,
        'patches': patches,
        'meta': {t: meta.get(t, {'id': t, 'name': t.upper(), 'zh': '', 'power': '', 'status': '',
                                 'modern': [], 'note': '', 'area': None}) for t in terr},
    }
    fit = load('trace_labels_fit.json')
    if fit:
        data['fonts'], data['styles'] = fonts_and_styles(fit['labels'])
        keep = [l for l in fit['labels'] if not (l['t'].startswith('Source') and not l['rot'])]
        data['labels'] = [[l['x'], l['y'], l['t'], l['s'], l['size'], l['tr'], l['rot'], l.get('sx', 1)] for l in keep]
        data['dots'] = fit['dots']
    for key, fname in (('frame', 'trace_frame.json'), ('cities', 'trace_cities.json')):
        d = load(fname)
        if d is not None:
            data[key] = d
    riv, straight = load('trace_riverlines.json'), load('trace_lines.json')
    data['lines'] = (straight or []) + (river_lines(riv) if riv else [])

    js = '// Generated by scripts/trace_*.py from source-map.png — do not edit by hand\n' + \
         'window.MAP_DATA = ' + json.dumps(data, ensure_ascii=False, separators=(',', ':')) + ';\n'
    with open(os.path.join(ROOT, 'data.js'), 'w') as f:
        f.write(js)
    print('data.js: %d bytes, %d patches, %d territories' % (len(js.encode('utf-8')), len(patches), len(terr)))


if __name__ == '__main__':
    main()
