#!/usr/bin/env python3
"""Step 1 of the source-map trace: per-pixel classification of source-map.png.

Every pixel of the scan is assigned to exactly one class:
  * a flat land class (context land, one of the colonial shade colours, …)
  * SEA   (ocean and lakes share one water palette)
  * WHITE (1914 border lines, 1870 frontier dots, the title box)
Ink (lettering, city dots), its halo, rivers and the graticule are not kept
as classes: they take the class of the nearest flat pixel.  The class map is written to data/trace_classes.npy
together with the measured water-shade field (data/trace_water.npy) for the
vectorisation step.  The white dots of the 1870 frontiers are part of WHITE.
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'source-map.png')
OUT = os.path.join(ROOT, 'data')

# flat land palette, measured as the median of each class' flat pixels
LAND = {
    'ctx':    (237, 243, 217),   # land outside Africa
    'fr-l':   (138, 187, 227),
    'fr-m':   (74, 151, 205),
    'fr-d':   (12, 120, 185),
    'uk-l':   (226, 175, 206),
    'uk-m':   (194, 136, 184),
    'uk-d':   (166, 100, 162),
    'de-l':   (144, 188, 165),
    'de-m':   (104, 154, 127),
    'be':     (215, 175, 61),
    'pt-l':   (196, 209, 129),
    'pt-m':   (161, 175, 90),
    'pt-d':   (126, 141, 58),
    'es-l':   (247, 181, 90),
    'es-m':   (220, 143, 55),
    'es-d':   (188, 107, 26),
    'it-l':   (215, 198, 142),
    'it-m':   (183, 161, 104),
    'ot':     (141, 203, 188),
    'indep':  (212, 203, 198),
    'boer':   (147, 150, 163),
}
SEA_BASE = np.array([234., 238., 247.])   # open ocean
                                  
SEA_DARK = np.array([136., 187., 222.])
# tolerance overrides: 'fr-l' (138,187,227) sits only ~12 units from the darkest
# coastal water, so both are classified with a tighter margin
TOL = {'fr-l': 6.5}
THIN_WATER_PX = 25  # a body of water may have this many narrow pixels (channels)
FRAME = 2                        # white paper frame around the scanned map (px)
LEGEND_BOX = (6, 437, 208, 704)   # x0, y0, x1, y1 of the legend box (redrawn by hand)
TITLE_BOX = (19, 11, 352, 43)     # title box (redrawn by hand)
# boxes that are lettering on open water in the scan (redrawn as text)
FORCE_SEA = [(644, 318, 673, 704)]           # rotated source credit along the right edge
SEA, WHITE = -1, -2

# low-chroma classes that anti-aliased lettering can mimic are only valid
# inside these boxes (x0, y0, x1, y1)
ONLY_IN = {
    'boer': [(388, 596, 452, 656)],                       # Transvaal / Orange Free State
    'indep': [(488, 288, 592, 372), (92, 332, 128, 378)],  # Abyssinia, Liberia
}

VIVID = {                         # debug colours
    'SEA': (120, 180, 255), 'WHITE': (255, 255, 255),
    'ctx': (190, 210, 120), 'fr-l': (120, 180, 240), 'fr-m': (40, 120, 220), 'fr-d': (0, 40, 160),
    'uk-l': (240, 170, 220), 'uk-m': (200, 100, 180), 'uk-d': (150, 40, 110),
    'de-l': (130, 210, 170), 'de-m': (40, 130, 80), 'be': (240, 200, 0),
    'pt-l': (200, 220, 90), 'pt-m': (130, 150, 20), 'pt-d': (70, 90, 0),
    'es-l': (255, 190, 90), 'es-m': (230, 120, 30), 'es-d': (160, 60, 0),
    'it-l': (220, 200, 120), 'it-m': (150, 120, 40), 'ot': (0, 200, 200),
    'indep': (120, 100, 90), 'boer': (60, 60, 80),
}


def nearest_fill(values, known):
    """values where known, nearest known value elsewhere."""
    idx = ndi.distance_transform_edt(~known, return_distances=False, return_indices=True)
    return values[idx[0], idx[1]]


np_save_sea = [None]


def classify():
    im = np.asarray(Image.open(SRC).convert('RGB')).astype(float)
    med = ndi.median_filter(im, size=(3, 3, 1))
    H, W, _ = med.shape
    FR = [names_ for names_ in (0,)]  # placeholder, replaced below

    names = list(LAND)
    pal = np.array([LAND[n] for n in names], float)                   # (K,3)
    d2 = ((med[:, :, None, :] - pal[None, None, :, :]) ** 2).sum(-1)  # (H,W,K)
    order = np.argsort(d2, axis=2)
    k1 = order[:, :, 0]
    dist = np.sqrt(np.take_along_axis(d2, order[:, :, :2], 2))
    d1, dd = dist[:, :, 0], dist[:, :, 1]
    tol = np.full(len(names), 13.0)
    for nme, tv in TOL.items():
        tol[names.index(nme)] = tv
    core = d1 < np.minimum(tol[k1], 0.42 * dd)   # confident flat-land pixel
    for nme, boxes in ONLY_IN.items():
        inside = np.zeros((H, W), bool)
        for x0, y0, x1, y1 in boxes:
            inside[y0:y1, x0:x1] = True
        core &= ~((k1 == names.index(nme)) & ~inside)

    # Water.  The colours of the sea lie in a narrow band around the segment
    # base -> darkest coastal water (they do not sit exactly on a line: the
    # Red Sea is a little greyer than the line).  The darkest water is almost
    # identical to the light French blue 'fr-l': that class is only accepted
    # where it is flat over 3 x 3 px (the sea is a gradient near the coast).
    axis = SEA_DARK - SEA_BASE
    t = ((med - SEA_BASE) * axis).sum(-1) / (axis * axis).sum()
    tc = np.clip(t, 0, 1.0)
    res = np.sqrt(((med - (SEA_BASE + tc[:, :, None] * axis)) ** 2).sum(-1))
    ifl = names.index('fr-l')
    strict_fl = np.sqrt(d2[:, :, ifl]) < 3.5
    flat_fl = ndi.binary_opening(strict_fl, structure=np.ones((3, 3), bool))
    core &= ~((k1 == ifl) & ~flat_fl)
    water = (res < 14.0) & (t > -0.12) & (t < 1.05) & (med[..., 2] >= 215) & ~core
    # thin water (< 3 px) is only trusted when short (a channel): a white
    # border line blended with a blue fill, or a graticule dotted line on the
    # light French blue, matches the sea colour and would otherwise cut the
    # land as a thin "sea" line
    opened = ndi.binary_opening(water, structure=np.ones((3, 3), bool))
    wlab, wn = ndi.label(water)
    thin_px = np.bincount(wlab.ravel(), minlength=wn + 1) - np.bincount(wlab[opened].ravel(), minlength=wn + 1)
    keep = np.zeros(wn + 1, bool)
    keep[thin_px <= THIN_WATER_PX] = True                 # short narrow channels
    water = opened | np.isin(wlab, np.nonzero(keep)[0])
    np_save_sea[0] = water

    # ink = dark, low-chroma pixels that are not a flat colour (lettering,
    # city dots, the scale bar)
    chroma = im.max(-1) - im.min(-1)
    lum = im @ np.array([0.299, 0.587, 0.114])
    ink = (lum < 125) & (chroma < 55) & ~core
    inkzone = ndi.binary_dilation(ink, structure=np.ones((5, 5), bool))

    # context class of every pixel = class of the nearest confident flat pixel
    # outside the ink zone (symmetric for land and water, so lettering near a
    # coast is not filled from one side only).  White and halos are judged
    # against this colour, not against the nearest palette colour.
    K = np.where(water, SEA, k1)
    # near-white pixels (a border line blended with a blue fill is bluish
    # white) must not pose as water here, or they could never become WHITE
    base_known = (core | (water & (med.min(-1) < 238))) & ~inkzone
    K[~base_known] = nearest_fill(K, base_known)[~base_known]
    near = np.where((K == SEA)[:, :, None], SEA_BASE + tc[:, :, None] * axis,
                    pal[np.clip(K, 0, None)])
    v = near - 255.0
    wfrac = np.clip(((near - im) * v).sum(-1) / np.maximum((v * v).sum(-1), 1e-6), 0, 1)
    blend_err = np.sqrt(((im - (near - wfrac[:, :, None] * v)) ** 2).sum(-1))

    # White, in two thresholds.  The strict one (>= 50 % white) draws the
    # borders with their true width; the loose one (>= 28 %) only keeps
    # patches apart when territories are clustered, because many border lines
    # of the scan are faint 1 px lines.
    valid = (K >= 0) & (blend_err < 24)
    white = valid & (wfrac >= 0.5) & ~water
    loose = valid & (wfrac >= 0.28) & ~water

    # the halo of lettering: pale pixels (>= 25 % white) connected to the ink
    # zone.  They are not borders; they are filled with the flat class below.
    dist_ink = ndi.distance_transform_edt(~inkzone)
    halozone = inkzone | ndi.binary_propagation(inkzone, mask=(K >= 0) & (wfrac >= 0.25) & (dist_ink <= 6))
    white &= ~halozone
    loose &= ~halozone
    white &= ~ink
    # drop isolated white specks (paper noise) that touch neither land nor ink
    wl = ndi.label(white)[0]
    ws = np.bincount(wl.ravel())
    near_something = ndi.binary_dilation(core | ink, structure=np.ones((3, 3), bool))
    white &= ~((ws[wl] < 4) & ~near_something)

    # a halo lying over open water is water: its colour must be bluish and sit
    # on a segment running from white to the water palette
    dh = np.full((H, W), 1e9)
    for tt in np.linspace(-0.1, 1.1, 13):
        u = SEA_BASE + tt * axis - 255.0
        w = np.clip(((im - 255.0) * u).sum(-1) / (u * u).sum(), 0, 1)
        dh = np.minimum(dh, np.sqrt(((im - (255.0 + w[:, :, None] * u)) ** 2).sum(-1)))
    near_water = ndi.binary_dilation(water, structure=np.ones((7, 7), bool))
    water_halo = halozone & (dh < 11) & ((im[..., 2] - im[..., 0]) >= 6) & ~white & near_water

    # rivers are thin blue lines: a thin (<= 2 px) structure of French blue,
    # or clearly blue on a non-blue land colour.  They are traced later as
    # lines instead of being painted as slivers of fill.
    blues = [names.index('fr-l'), names.index('fr-m')]
    frmask = np.isin(k1, blues) & ~water
    thin = frmask & ~ndi.binary_opening(frmask, structure=np.ones((3, 3), bool)) & (d1 < 26)
    blueness = ((im[..., 2] - im[..., 0]) >= 70) & (im[..., 2] >= 185) & ~water
    on_other = (K >= 0) & ~np.isin(K, blues)
    river = ~inkzone & ~white & (thin | (blueness & on_other))
    river = ndi.binary_opening(river, structure=np.ones((2, 2), bool)) | (thin & ~inkzone & ~white)

    # fill sources: confident pixels away from the ink (2 px) and not part of
    # a halo (a shade blended with white can mimic the next lighter shade)
    known = (core | water | white) & ~inkzone & ~river & (white | (wfrac < 0.2))
    cls = np.where(water, SEA, k1)
    cls[white] = WHITE
    cls[~known] = nearest_fill(cls, known)[~known]
    # a pixel whose own colour is land-like (nowhere near the white <-> water
    # range) is filled from the land side only, even inside the halo of a
    # label: lettering on a narrow coastal strip must not turn it into sea,
    # and islets without a single pure pixel keep their colour
    # ... but only for saturated colours: the grey anti-aliasing of lettering
    # must not pose as one of the two grey classes
    landlike = ~ink & (dh > 28) & (d1 < 45) & ~np.isin(k1, [names.index('boer'), names.index('indep')])
    known_land = ((core & ~water) | white) & ~inkzone & ~river & (white | (wfrac < 0.2))
    sel = ~known & landlike
    idx = ndi.distance_transform_edt(~known_land, return_distances=False, return_indices=True)
    cls[sel] = np.where(d1[sel] < 22, k1[sel], cls[idx[0][sel], idx[1][sel]])

    # the legend box is redrawn by hand: treat its footprint as open sea
    x0, y0, x1, y1 = LEGEND_BOX
    boxmask = np.zeros(cls.shape, bool); boxmask[y0:y1, x0:x1] = True
    cls[boxmask] = SEA
    tx0, ty0, tx1, ty1 = TITLE_BOX
    cls[ty0:ty1, tx0:tx1] = WHITE
    for x0, y0, x1, y1 in FORCE_SEA:
        cls[y0:y1, x0:x1] = SEA
    cls[:FRAME, :] = SEA; cls[-FRAME:, :] = SEA; cls[:, :FRAME] = SEA; cls[:, -FRAME:] = SEA

    # barrier height used by the watershed that names the territories:
    # 0 on flat colour and on shade boundaries, rising with the whiteness of
    # (faint) border lines, highest on pure white.  Lettering and its halo are
    # not borders.
    barrier = np.clip((wfrac - 0.24) / 0.2, 0, 1) * 0.5 + white.astype(float) * 0.5
    barrier[halozone] = 0
    barrier[water | water_halo] = 1
    barrier = np.clip(barrier, 0, 1)

    # colour of every flat class: median of the raw pixels within 6 units of
    # the class colour (pure pixels only)
    cols = {}
    for i, nme in enumerate(names):
        m = (d1 < 6) & (k1 == i)
        cols[nme] = '#%02x%02x%02x' % tuple(int(v) for v in np.median(im[m], axis=0))

    # shade measured on trusted water only: lettering over the sea is SEA
    # too, but its colour is ink, not water
    tf = nearest_fill(np.clip(t, 0, 1.15), water | ((cls == WHITE) & ~boxmask))
    tf[boxmask] = 0
    return im, med, cls, tf, names, ink, white, loose, river, barrier, halozone, cols, np_save_sea[0]


def main():
    im, med, cls, tf, names, ink, white, loose, river, barrier, halozone, cols, sea = classify()
    H, W = cls.shape
    np.save(os.path.join(OUT, 'trace_classes.npy'), cls)
    np.save(os.path.join(OUT, 'trace_water.npy'), tf.astype(np.float32))
    loose_cls = np.where(loose, WHITE, cls)
    loose_cls[white] = WHITE
    np.save(os.path.join(OUT, 'trace_loose.npy'), loose_cls)
    np.save(os.path.join(OUT, 'trace_ink.npy'), ink)
    np.save(os.path.join(OUT, 'trace_river.npy'), river)
    np.save(os.path.join(OUT, 'trace_barrier.npy'), barrier.astype(np.float32))
    np.save(os.path.join(OUT, 'trace_halo.npy'), halozone)
    np.save(os.path.join(OUT, 'trace_sea.npy'), sea)
    # colour of every flat class: median of the raw pixels that sit within 6
    # units of the class colour (pure pixels only)
    json.dump(cols, open(os.path.join(OUT, 'trace_colors.json'), 'w'), indent=1)

    cols = {}
    for i, nme in enumerate(names):
        m = cls == i
        if m.sum():
            cols[nme] = [int(v) for v in np.median(med[m], axis=0)]
    print(json.dumps(cols))
    print('ink px', int(ink.sum()), 'white px', int(white.sum()))

    vis = np.zeros((H, W, 3), np.uint8)
    vis[cls == SEA] = VIVID['SEA']; vis[cls == WHITE] = VIVID['WHITE']
    for i, nme in enumerate(names):
        vis[cls == i] = VIVID[nme]
    Image.fromarray(vis).save(os.path.join(ROOT, 'scratch', 'redraw', 'classes.png'))
    Image.fromarray((np.clip(tf, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(ROOT, 'scratch', 'redraw', 'tfield.png'))
    src8 = np.asarray(Image.open(SRC).convert('RGB'))
    areas = {'nw': (40, 60, 330, 290), 'wa': (60, 240, 330, 440), 'e': (360, 220, 673, 500),
             's': (200, 470, 560, 710), 'ne': (250, 40, 673, 260)}
    for k, (x0, y0, x1, y1) in areas.items():
        a = src8[y0:y1, x0:x1]; b = vis[y0:y1, x0:x1]
        both = np.concatenate([a, np.full((a.shape[0], 4, 3), 255, np.uint8), b], axis=1)
        Image.fromarray(both).resize((both.shape[1] * 3, both.shape[0] * 3), Image.NEAREST).save(
            os.path.join(ROOT, 'scratch', 'redraw', 'cmp_%s.png' % k))
    # gridded class map (for choosing territory seed points)
    g = vis.copy()
    for x in range(0, W, 25):
        g[:, x] = (255, 0, 0) if x % 100 == 0 else (0, 0, 0)
    for y in range(0, H, 25):
        g[y, :] = (255, 0, 0) if y % 100 == 0 else (0, 0, 0)
    gi = Image.fromarray(g)
    d = ImageDraw.Draw(gi)
    for x in range(0, W, 50):
        for y in range(0, H, 50):
            d.text((x + 2, y + 1), '%d,%d' % (x, y), fill=(255, 255, 0))
    gi.resize((W * 2, H * 2), Image.NEAREST).save(os.path.join(ROOT, 'scratch', 'redraw', 'grid.png'))


if __name__ == '__main__':
    main()
