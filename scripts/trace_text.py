#!/usr/bin/env python3
"""Step 5a of the source-map trace: locate every piece of lettering.

The text of the map was read with the OCR of macOS (scratch/redraw/ocr,
on a 4x copy of the scan) and corrected by hand in FIX / EXTRA below.  Every
label is then fitted to the ink of the scan: the bounding box is refined to
the ink pixels that lie inside (a little margin around) the OCR box, city
dots excluded.

The OCR step needs macOS (`swift` + the Vision framework): it is run once
and its result is kept in scratch/.

Output: data/trace_text.json
  dots     [[x, y], ...]           city dots (centre)
  labels   [{t, x0, y0, x1, y1, slant, ink: [[x, y], ...]}]
"""
import json
import os
import subprocess

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data')
OCR = os.path.join(ROOT, 'scratch', 'redraw', 'ocr', 'obs.json')


def ocr():
    """Read the lettering of the scan with the text recognition of macOS
    (scripts/trace_ocr.swift) on a 4x copy, unless already done."""
    if os.path.exists(OCR):
        return
    os.makedirs(os.path.dirname(OCR), exist_ok=True)
    up = os.path.join(os.path.dirname(OCR), 'up4.png')
    im = Image.open(os.path.join(ROOT, 'source-map.png')).convert('RGB')
    im.resize((im.width * 4, im.height * 4), Image.LANCZOS).save(up)
    subprocess.run(['swift', os.path.join(ROOT, 'scripts', 'trace_ocr.swift'), up, OCR], check=True)

# OCR text -> corrected text (same box)
FIX = {
    'Orano': 'Oran', 'Ager': 'Alger', 'Casablanca ,': 'Casablanca', '• Fès': 'Fès', 'Madère •': 'Madère',
    'Benghazi •': 'Benghazi', 'Le Caire o': 'Le Caire', '• Maurzauk': 'Mourzouk', 'RIO': 'RÍO',
    'Tropique du Corcel': 'Tropique du Cancer', 'Assouan o': 'Assouan', 'Arguin?': 'Arguin',
    'lles': 'Îles', 'Souckin •': 'Souakin', 'Dongala o': 'Dongola', 'Loc': 'Lac', 'Tchod': 'Tchad',
    'Khartouno': 'Khartoum', 'Massãoca •': 'Massaoua', 'Sckoto': 'Sokoto', '• FortLamy': 'Fort-Lamy',
    'Fachada o': 'Fachoda', 'AcdisAbeto': 'Addis-Abeba', 'Haror': 'Harar', 'Freetown!': 'Freetown',
    '• ogos': 'Lagos', 'Manrovia •': 'Monrovia', 'ccra •': 'Accra', '•Douala': 'Douala',
    '• Mogodiscio': 'Mogadiscio', 'GUINÉE ESP:': 'GUINÉE ESP.', 'Equctenr': 'Équateur', '•Witu': 'Witu',
    '• Kindu': 'Kindu', 'Momu': 'Mombasa', '• Léopoldville': 'Léopoldville', 'rZanzibar': 'Zanzibar',
    '• Dor es Salaam': 'Dar es-Salaam', 'Tangonyika': 'Tanganyika', 'Luanda o': 'Luanda',
    'Elischeftvile': 'Elisabethville', 'Benguelo•': 'Benguela', 'Nyossa': 'Nyassa', 'MoçamedesO': 'Moçamedes',
    '• Pretonic': 'Pretoria', 'Mafeking®': 'Mafeking', '-SWAZILAND': 'SWAZILAND', 'Jchannesburg': 'Johannesburg',
    '-(ZULULAND)': '(ZULULAND)', '-(NATAL)': '(NATAL)', '•BASUTOLAND': 'BASUTOLAND', 'Le Cap ?': 'Le Cap',
    '• Port Eizabeth': 'Port Elizabeth', '30°5': '30°S', 'Trapique de Capricome': 'Tropique du Capricorne',
    'Y(KAFFRARIA)': '(KAFFRARIA)', 'Saintlouis o': 'Saint-Louis', 'Tombouctou o': 'Tombouctou',
    'Cabindo (': 'Cabinda', '10°S': '10°S',
}
# OCR observations that merge several labels: (text of the OCR line) -> [(text, x0, y0, x1, y1), ...]
SPLIT = {
    'O*E O/Tunis': [('10°E', 271, 65, 283, 72), ('Tunis', 286, 64, 311, 74)],
    'MAROC ESP. Tanger:': [('MAROC ESP.', 88, 66, 141, 75), ('Tanger', 145, 66, 173, 74)],
    'Dokor • SÉNÉGAL': [('Dakar', 20, 246, 42, 256), ('SÉNÉGAL', 50, 248, 94, 259)],
    'LIBÉRIA Bingerville "Grand Bossam': [('LIBÉRIA', 74, 338, 108, 350), ('Bingerville', 111, 337, 152, 347),
                                          ('Grand Bassam', 154, 345, 206, 356)],
    'OUGANDA AFRIQUE-': [('OUGANDA', 458, 364, 500, 376), ('AFRIQUE-', 505, 365, 552, 377)],
    'Toboro AFRIQUE.': [('Tabora', 444, 443, 470, 455), ('AFRIQUE-', 474, 443, 520, 456)],
    'Victonia Tangio': [('Victoria', 468, 424, 497, 435), ('Tanga', 503, 424, 528, 436)],
    'Walvis Bay• Windhoek': [('Walvis Bay', 272, 591, 314, 602), ('Windhoek', 318, 596, 360, 607)],
    'Mer Mediterrante': None,                 # rotated, see EXTRA
    'Golfe Persique': None, 'Rouge': None, 'Congo': None, 'OPIOUEZOUATO': None, '0901': None,
    'DAHOMEY': None, 'FRANÇAISE': None, 'Zambèze': None, 'ON': None, '2007.': None,
    'Paris, Larousse,': None, '•': None, '1000 km': None,
}
# (text, x0, y0, x1, y1, rotation in degrees, optional)  lettering the OCR missed or that is rotated
EXTRA = [
    ('Agadir', 118, 110, 140, 121, 0), ('Ifni', 124, 122, 138, 131, 0), ('Beira', 536, 545, 556, 556, 0),
    ('Nil', 452, 140, 466, 158, -70), ('Niger', 195, 250, 212, 274, -68),
    ('Mer Méditerranée', 327, 70, 440, 100, -11), ('Mer Rouge', 503, 190, 541, 232, 72),
    ('Golfe Persique', 586, 122, 646, 172, -36), ('TOGO', 201, 296, 212, 338, -90),
    ('DAHOMEY', 204, 286, 216, 338, -90), ('AFRIQUE ÉQUATORIALE FRANÇAISE', 345, 270, 372, 440, -78),
    ('Congo', 384, 366, 416, 397, -36), ('Zambèze', 408, 520, 452, 564, -33),
    ('NYASSALAND', 473, 498, 490, 560, -80), ('MOZAMBIQUE', 478, 548, 545, 604, -52),
    ('1 000 km', 250, 688, 292, 702, 0),
    # legend and title lettering
    ('Territoires sous domination', 12, 428, 150, 443, 0),
    ('française', 119, 462, 206, 477, 0), ('britannique', 119, 478, 206, 493, 0),
    ('allemande', 119, 495, 206, 510, 0), ('belge', 119, 511, 206, 526, 0),
    ('portugaise', 119, 527, 206, 542, 0), ('espagnole', 119, 543, 206, 558, 0),
    ('italienne', 119, 559, 206, 574, 0), ('turque', 119, 575, 206, 590, 0),
    ('des Boers /', 119, 591, 206, 607, 0), ('Afrikaners', 119, 608, 206, 624, 0),
    ('État indépendant', 119, 624, 206, 639, 0), ('Frontière en 1914', 119, 644, 206, 656, 0),
    ('Frontière en 1870', 119, 662, 206, 676, 0), ('(NATAL)', 77, 676, 110, 689, 0),
    ('Nom de territoire', 119, 675, 206, 688, 0), ('en 1870', 119, 686, 206, 699, 0),
    ('1870', 19, 447, 46, 458, 0), ('1900', 49, 447, 76, 458, 0), ('1914', 81, 447, 108, 458, 0),
    ("L'EXPANSION COLONIALE EN AFRIQUE (1870-1914)", 26, 17, 350, 38, 0),
    ('Source : C. Liauzu, Dictionnaire de la colonisation française, Paris, Larousse, 2007.', 654, 318, 670, 706, -90),
]
# OCR observations to drop entirely (dots, fragments handled elsewhere)
DROP_TEXT = {'•', 'ON', '2007.', 'Paris, Larousse,', 'O*E O/Tunis', 'MAROC ESP. Tanger:', 'Dokor • SÉNÉGAL',
             'LIBÉRIA Bingerville "Grand Bossam', 'OUGANDA AFRIQUE-', 'Toboro AFRIQUE.', 'Victonia Tangio',
             'Walvis Bay• Windhoek', 'Mer Mediterrante', 'Golfe Persique', 'Rouge', 'Congo', 'OPIOUEZOUATO',
             '0901', 'DAHOMEY', 'FRANÇAISE', 'Zambèze', '1000 km'}


def ink_and_dots():
    """Ink = pixels clearly darker than their surroundings (the local
    background = brightest pixel of the 9 x 9 neighbourhood, which also holds
    for lettering inside the white halo of another label).  City dots are the
    small solid dark blobs that nothing else is attached to."""
    rgb = np.asarray(Image.open(os.path.join(ROOT, 'source-map.png')).convert('RGB')).astype(float)
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    med = ndi.median_filter(lum, 3)
    bg = ndi.maximum_filter(med, size=9)
    ink = (bg - med) > 50
    # city dots: a solid near-black blob of 2 x 2 .. 3 x 3 px with a pale ring
    dark = lum < 85
    dlab, dn = ndi.label(dark, structure=np.ones((3, 3), bool))
    dots = np.zeros(ink.shape, bool)
    centres = []
    for i, sl in enumerate(ndi.find_objects(dlab)):
        comp = dlab[sl] == i + 1
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        area = int(comp.sum())
        if not (4 <= area <= 9 and 2 <= h <= 3 and 2 <= w <= 3):
            continue
        r = (slice(max(0, sl[0].start - 3), sl[0].stop + 3), slice(max(0, sl[1].start - 3), sl[1].stop + 3))
        around = dlab[r]
        around_comp = around == i + 1
        ring = ndi.binary_dilation(around_comp, structure=np.ones((5, 5), bool)) & ~ndi.binary_dilation(around_comp)
        if int((dark[r] & ring).sum()) > 1:
            continue                                   # attached to lettering
        if np.median(lum[r][ndi.binary_dilation(around_comp, structure=np.ones((7, 7), bool)) & ~around_comp]) < 140:
            continue                                   # no pale ring: part of a dark area
        dots[sl] |= comp
        ys, xs = np.nonzero(comp)
        centres.append([round(float(xs.mean()) + sl[1].start, 1), round(float(ys.mean()) + sl[0].start, 1)])
    return ink & ~dots, centres


def refine(ink, box, rot=0):
    """Ink of one label: connected ink components inside the box grown by 2 px
    that reach into the box itself.  Returns the pixel list (x, y)."""
    H, W = ink.shape
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    gx0, gy0, gx1, gy1 = max(0, x0 - 2), max(0, y0 - 2), min(W, x1 + 2), min(H, y1 + 2)
    sub = ink[gy0:gy1, gx0:gx1]
    lab, n = ndi.label(sub, structure=np.ones((3, 3), bool))
    inner = np.zeros_like(sub)
    inner[max(0, y0 - gy0):max(0, y1 - gy0), max(0, x0 - gx0):max(0, x1 - gx0)] = True
    keep = np.unique(lab[inner & (lab > 0)])
    m = np.isin(lab, keep)
    ys, xs = np.nonzero(m)
    return [[int(x + gx0), int(y + gy0)] for x, y in zip(xs, ys)]


def main():
    ocr()
    ink, centres = ink_and_dots()
    obs = json.load(open(OCR))
    labels = []
    for o in obs:
        x0 = o['x'] * 673; y0 = (1 - o['y'] - o['h']) * 710; x1 = x0 + o['w'] * 673; y1 = y0 + o['h'] * 710
        t = o['t'].strip()
        if t in DROP_TEXT or t.startswith('Source : C.'):
            continue
        if t.startswith('Territoires') or t in ('1870', '1900 1914', 'française', 'britannique', 'allemande',
                                                 'belge', 'portugaise', 'espagnole', 'italienne', 'turque',
                                                 'des Boers /', 'Afrikaners', 'État indépendant',
                                                 'Frontière en 1914', 'Frontière en 1870', '(NATAL) Nom de territoire',
                                                 'en 1870') or t.startswith("L'EXPANSION"):
            continue                                   # legend and title: trace_frame.py
        if t in FIX:
            t = FIX[t]
        labels.append({'t': t, 'box': [x0, y0, x1, y1], 'rot': 0})
    for key, parts in SPLIT.items():
        if parts:
            for (t, x0, y0, x1, y1) in parts:
                labels.append({'t': t, 'box': [x0, y0, x1, y1], 'rot': 0})
    for (t, x0, y0, x1, y1, rot) in EXTRA:
        labels.append({'t': t, 'box': [x0, y0, x1, y1], 'rot': rot})

    out = []
    for L in labels:
        px = refine(ink, L['box'], L['rot'])
        xs = [p[0] for p in px]; ys = [p[1] for p in px]
        if not px:
            print('no ink for %r in %s' % (L['t'], [round(v) for v in L['box']]))
            continue
        rec = {'t': L['t'], 'x0': min(xs), 'y0': min(ys), 'x1': max(xs) + 1, 'y1': max(ys) + 1,
               'rot': L['rot'], 'ink': px}
        if not L['rot']:
            # slant of the lettering (italic): x of the ink centre of the top third minus bottom third
            a = np.array(px, float)
            ya = a[:, 1]
            top, bot = a[ya < np.quantile(ya, .33)], a[ya > np.quantile(ya, .67)]
            if len(top) and len(bot):
                rec['slant'] = round(float((top[:, 0].mean() - bot[:, 0].mean()) / (rec['y1'] - rec['y0'])), 2)
        out.append(rec)
    out.sort(key=lambda r: (r['y0'], r['x0']))
    json.dump({'dots': centres, 'labels': out}, open(os.path.join(OUT, 'trace_text.json'), 'w'), ensure_ascii=False)
    print('%d labels, %d dots' % (len(out), len(centres)))

    # review sheet: boxes and indices over the scan, 3x
    src = Image.open(os.path.join(ROOT, 'source-map.png')).convert('RGB')
    big = src.resize((673 * 3, 710 * 3), Image.LANCZOS)
    from PIL import ImageDraw
    d = ImageDraw.Draw(big)
    for i, r in enumerate(out):
        d.rectangle([r['x0'] * 3 - 2, r['y0'] * 3 - 2, r['x1'] * 3 + 1, r['y1'] * 3 + 1], outline=(255, 0, 0))
        d.text((r['x0'] * 3 - 2, r['y0'] * 3 - 11), str(i), fill=(0, 0, 255))
    for (x, y) in centres:
        d.ellipse([x * 3 - 6, y * 3 - 6, x * 3 + 6, y * 3 + 6], outline=(0, 200, 0))
    for k, (x0, y0, x1, y1) in enumerate([(0, 0, 340, 360), (330, 0, 673, 360), (0, 350, 340, 710), (330, 350, 673, 710)]):
        big.crop((x0 * 3, y0 * 3, x1 * 3, y1 * 3)).save(os.path.join(ROOT, 'scratch', 'redraw', 'text_q%d.png' % k))


if __name__ == '__main__':
    main()
