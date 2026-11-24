#!/usr/bin/env python3
"""Step 4 of the source-map trace: render the page and compare it with the scan.

  trace_verify.py [scale]        default scale 3 (rendered at 2019 x 2130)

The page is opened with ?bare=1 (map only, exactly 673 x 710 CSS px) in
headless Chrome, screenshotted, box-filtered back to the size of the scan
and compared with source-map.png:

  * colour error over all pixels (mean |dRGB|, share of pixels within a
    tolerance);
  * the same over the "flat" pixels only (no lettering, no line work, no
    legend / title box): this measures the land, borders and water shade;
  * a diff image (error amplified) and a side-by-side sheet in scratch/.

Nothing is written outside scratch/.
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.path.join(ROOT, 'scratch', 'redraw')
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
FLAT_TOL = 14            # per-channel tolerance for "same colour"


def shoot(scale, name='render.png'):
    out = os.path.join(SCRATCH, name)
    cmd = [CHROME, '--headless=new', '--disable-gpu', '--hide-scrollbars', '--no-first-run',
           '--disable-extensions', '--force-device-scale-factor=%s' % scale,
           '--window-size=673,710', '--virtual-time-budget=4000',
           '--screenshot=%s' % out,
           'file://%s/index.html?bare=1' % ROOT]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def flat_mask():
    """Pixels of the scan that show flat colour: away from lettering, white
    lines, rivers, graticule, and outside the legend and the title box."""
    cls = np.load(os.path.join(ROOT, 'data', 'trace_classes.npy'))
    ink = np.load(os.path.join(ROOT, 'data', 'trace_ink.npy'))
    m = cls >= 0                                   # flat land or sea
    m &= ~ndi.binary_dilation(ink, structure=np.ones((5, 5), bool))
    m[380:440, 0:215] = False                      # OCÉAN ATLANTIQUE lettering area kept, legend top margin
    m[430:706, 0:215] = False                      # legend box (+ shadow)
    m[8:48, 14:356] = False                        # title box
    m[:4, :] = False; m[-4:, :] = False; m[:, :4] = False; m[:, -4:] = False
    return m


def main():
    scale = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    os.makedirs(SCRATCH, exist_ok=True)
    shot = shoot(scale)
    big = Image.open(shot).convert('RGB')
    ren = np.asarray(big.resize((673, 710), Image.BOX)).astype(int)
    src = np.asarray(Image.open(os.path.join(ROOT, 'source-map.png')).convert('RGB')).astype(int)
    err = np.abs(ren - src)
    mx = err.max(-1)
    flat = flat_mask()

    def report(name, m):
        e = err[m]
        print('%-28s pixels %6d   mean |dRGB| %5.2f   within %d: %6.2f%%   within 30: %6.2f%%' % (
            name, int(m.sum()), e.mean(), FLAT_TOL, 100 * (mx[m] <= FLAT_TOL).mean(), 100 * (mx[m] <= 30).mean()))

    report('all pixels', np.ones(mx.shape, bool))
    report('flat pixels (land/borders/sea)', flat)
    # neighbourhood of the lettering (outside the legend and title boxes)
    ink = np.load(os.path.join(ROOT, 'data', 'trace_ink.npy'))
    near_ink = ndi.binary_dilation(ink, structure=np.ones((5, 5), bool))
    near_ink[380:440, 0:215] = False; near_ink[430:706, 0:215] = False; near_ink[8:48, 14:356] = False
    report('around lettering', near_ink)

    # where are the flat errors?
    bad = flat & (mx > FLAT_TOL)
    lab, n = ndi.label(bad, structure=np.ones((3, 3), bool))
    print('flat error blobs > 40 px:')
    blobs = []
    for i, sl in enumerate(ndi.find_objects(lab)):
        a = int((lab[sl] == i + 1).sum())
        if a > 40:
            blobs.append((a, sl[1].start, sl[1].stop, sl[0].start, sl[0].stop))
    for a, x0, x1, y0, y1 in sorted(blobs, reverse=True)[:25]:
        print('   %5d px  x %3d-%3d  y %3d-%3d' % (a, x0, x1, y0, y1))

    vis = np.clip(err * 6, 0, 255).astype(np.uint8)
    Image.fromarray(vis).save(os.path.join(SCRATCH, 'diff.png'))
    sheet = np.concatenate([src.astype(np.uint8), np.full((710, 4, 3), 255, np.uint8), ren.astype(np.uint8)], 1)
    Image.fromarray(sheet).save(os.path.join(SCRATCH, 'sheet.png'))
    Image.fromarray(ren.astype(np.uint8)).save(os.path.join(SCRATCH, 'render_1x.png'))


if __name__ == '__main__':
    main()
