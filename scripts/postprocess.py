#!/usr/bin/env python3
"""Post-process site/data.js: add era-coded fill colors (from the decoded legend),
area in km2, and assign each city to its territory."""
import json, math
from shapely.geometry import Polygon, Point

BASE = '/Users/weiaiwen1/Desktop/my-space/map-project'

# palette from the original map legend (segments 1870 / 1900 / 1914)
PAL = {
    'fr': {'1870': '#0878b8', '1900': '#4890c8', '1914': '#88b8e0'},
    'uk': {'1870': '#a060a0', '1900': '#c088b8', '1914': '#e0a8c8'},
    'de': {'1870': None,      '1900': '#689878', '1914': '#90b8a0'},
    'be': {'1870': None,      '1900': '#d0a838', '1914': '#d0a838'},
    'pt': {'1870': '#788838', '1900': '#a0a858', '1914': '#c0d080'},
    'es': {'1870': '#b86818', '1900': '#d88830', '1914': '#f8b058'},
    'it': {'1870': None,      '1900': '#b0a068', '1914': '#d0c088'},
    'ot': {'1870': None,      '1900': '#88c8b8', '1914': '#88c8b8'},
    'indep': {'1870': '#e8e6e0', '1900': '#e8e6e0', '1914': '#e8e6e0'},
}

# territory -> era of its base fill. The original paints several territories with
# SEVERAL era shades side by side (verified by calibrated pixel sampling); those get
# era='zones' + a ZONES spec below. Everything else is one shade per territory.
ERA = {
    'tunisie': '1870', 'reunion': '1870',
    'maroc': '1914', 'mauritanie': '1914', 'tchad': '1914',
    'guinee': '1900', 'cote-ivoire': '1900', 'dahomey': '1900',
    'haute-volta': '1900', 'gabon': '1900', 'moyen-congo': '1900', 'somalie-fr': '1900',
    'madagascar': '1900', 'comores': '1900', 'oubangui': '1900',
    'sierra': '1870', 'gambie': '1870', 'maurice': '1870', 'seychelles': '1870',
    'gold-coast': '1900', 'bechuanaland': '1900', 'basutoland': '1900', 'swaziland': '1900',
    'egypte': '1900', 'ouganda': '1900', 'bea': '1900', 'nyassaland': '1900',
    'zanzibar': '1900', 'somalie-br': '1900', 'rhodesie-n': '1900', 'rhodesie-s': '1900',
    'soudan-eg': '1900', 'nigeria': '1900',
    'kamerun': '1900', 'gea': '1900', 'togo': '1900', 'gswa': '1914',
    'congo-belge': '1900',
    'guinee-pt': '1870', 'cabo-verde': '1870', 'sao-tome': '1870', 'angola': '1900', 'mozambique': '1900',
    'madeira': '1870',
    'sahara-esp': '1900', 'rio-oro': '1900', 'maroc-esp': '1914', 'guinee-esp': '1870',
    'canaries': '1870', 'fernando-poo': '1870',
    'erythree': '1900', 'somalie-it': '1900',
    'libye': 'ot1900',  # Ottoman shade: the original colors Libya in the turque teal
    'abyssinie': 'indep', 'liberia': 'indep',
    # multi-shade territories (base fill + era zones, as in the original):
    'algerie': 'zones', 'senegal': 'zones', 'soudan-fr': 'zones', 'niger': 'zones',
    'afriquesud': 'zones',
}

# era zones: territory -> list of (fill, lon/lat polygon). Clipped to the territory.
from shapely.geometry import box as SBox
def band(lo, hi):  # latitude band, whole width
    return SBox(-30, lo, 60, hi)
ZONES = {
    # Algeria: deep Tell strip (1830–48), medium steppe/north Sahara (1871–1900), light deep Sahara (1900s)
    'algerie': [
        ('#0878b8', band(33.5, 37.5)),
        ('#4890c8', band(29.0, 33.5)),
    ],
    # Senegal: deep old-colony band along the Senegal river (1677–1870s), medium rest (1890s)
    'senegal': [
        ('#0878b8', SBox(-17.7, 15.2, -13.0, 16.8)),
    ],
    # French Sudan: medium south (1880–1900), light Azawad/Sahara north (1900s)
    'soudan-fr': [
        ('#88b8e0', band(16.0, 28.0)),
    ],
    # Niger: medium south (1899–1906), light Air/Ténéré north
    'niger': [
        ('#88b8e0', band(15.5, 28.0)),
    ],
    # Nigeria: deep south (Lagos 1861 / Oil Rivers 1885), medium north (protectorates 1900–01)
    'nigeria': [
        ('#a060a0', SBox(2.5, 3.0, 14.5, 7.4)),
    ],
    # Anglo-Egyptian Sudan: light Darfur & Bahr el-Ghazal (post-1900), medium core
    'soudan-eg': [
        ('#e0a8c8', SBox(22.0, 9.5, 28.0, 15.6)),
        ('#e0a8c8', SBox(24.5, 4.5, 29.5, 9.2)),
    ],
    # Union of South Africa: Boer republics (annexed 1902) kept in Boer gray
    'afriquesud': [
        ('#9090a0', SBox(28.2, -27.6, 32.1, -24.9)),   # Transvaal
        ('#9090a0', SBox(23.9, -30.8, 29.9, -28.4)),   # Orange
    ],
    # Kamerun: light green for 1911 Neukamerun cession (New Kamerun strip + duck's beak)
    'kamerun': [
        ('#90b8a0', Polygon([(14.9, 12.0), (16.4, 12.5), (16.9, 7.5), (16.3, 3.0), (17.3, 1.5), (16.0, -1.0), (15.1, 1.0), (14.7, 6.0), (14.4, 10.5)])),
    ],
    # Mozambique: deep old south (Lourenço Marques etc.), medium north (1890s companies)
    'mozambique': [
        ('#788838', SBox(30.0, -27.0, 41.0, -20.5)),
    ],
}
# base fill for multi-zone territories (the shade covering most of its area)
ZONE_BASE = {
    'algerie': '#88b8e0',   # light Sahara base, darker bands on top
    'senegal': '#4890c8',
    'soudan-fr': '#4890c8',
    'niger': '#4890c8',
    'nigeria': '#c088b8',
    'soudan-eg': '#c088b8',
    'afriquesud': '#c088b8',
    'kamerun': '#689878',
    'mozambique': '#a0a858',
}

def fill_for(tid, power):
    era = ERA.get(tid, '1900')
    if era == 'ot1900':
        return '#88c8b8', 'ot'
    if era == 'indep':
        return PAL['indep']['1900'], 'indep'
    if era == 'zones':
        return ZONE_BASE[tid], 'zones'
    return PAL[power][era], era

txt = open(BASE + '/data.js').read()
data = json.loads(txt[txt.index('{'):txt.rindex('}')+1])

def rings_area_km2(poly):
    """Spherical polygon area (km^2) for [outer, holes...] rings of [lon,lat]."""
    R = 6371.0
    def ring_area(ring):
        if len(ring) < 4: return 0.0
        total = 0.0
        for i in range(len(ring)-1):
            lon1, lat1 = math.radians(ring[i][0]), math.radians(ring[i][1])
            lon2, lat2 = math.radians(ring[i+1][0]), math.radians(ring[i+1][1])
            total += (lon2 - lon1) * (2 + math.sin(lat1) + math.sin(lat2))
        return abs(total) * R * R / 2.0
    a = ring_area(poly[0])
    for hole in poly[1:]:
        a -= ring_area(hole)
    return a

for t in data['territories']:
    fill, era = fill_for(t['id'], t['power'])
    t['fill'] = fill
    t['era'] = era
    t['power'] = 'ot' if t['id'] == 'libye' else t['power']
    t['area'] = round(sum(rings_area_km2(p) for p in t['geometry']))
    # build shapely for city assignment
    polys = []
    for p in t['geometry']:
        try:
            polys.append(Polygon(p[0], p[1:]))
        except Exception:
            pass
    t['_polys'] = polys

for c in data['cities']:
    pt = Point(c['lon'], c['lat'])
    owner = None
    for t in data['territories']:
        for poly in t['_polys']:
            if poly.contains(pt):
                owner = t['id']; break
        if owner: break
    if not owner:
        # coastal city: nearest territory within ~0.35 deg
        best_d, best_t = 0.35, None
        for t in data['territories']:
            for poly in t['_polys']:
                d = poly.distance(pt)
                if d < best_d:
                    best_d, best_t = d, t['id']
        owner = best_t
    c['t'] = owner

# resolve inter-territory overlaps (border slivers from mixed sources): first-drawn wins
acc = None
for t in data['territories']:
    polys = []
    for p in t['geometry']:
        try:
            g = Polygon(p[0], p[1:])
            if not g.is_valid:
                g = g.buffer(0)
            polys.append(g)
        except Exception:
            pass
    if not polys:
        t['_polys'] = []
        continue
    u = polys[0]
    for p in polys[1:]:
        u = u.union(p)
    if not u.is_valid:
        u = u.buffer(0)
    if acc is not None:
        u2 = u.difference(acc)
        if not u2.is_empty and u2.area > 0.01:
            u = u2
    if acc is None:
        acc = u
    else:
        acc = acc.union(u)
    # re-serialize geometry
    out = []
    polys2 = list(u.geoms) if u.geom_type == 'MultiPolygon' else [u]
    for poly in polys2:
        if poly.is_empty or poly.area < 0.0015:
            continue
        rings = [poly.exterior.coords] + [r.coords for r in poly.interiors]
        out.append([[[round(x, 3), round(y, 3)] for x, y in ring] for ring in rings])
    t['geometry'] = out
    t['_polys'] = [u]

# era zones: clip each zone polygon to its (overlap-resolved) territory geometry
def serialize_geom(g, min_area=0.003):
    out = []
    polys = list(g.geoms) if g.geom_type == 'MultiPolygon' else [g]
    for poly in polys:
        if poly.is_empty or poly.area < min_area:
            continue
        rings = [poly.exterior.coords] + [r.coords for r in poly.interiors]
        out.append([[[round(x, 3), round(y, 3)] for x, y in ring] for ring in rings])
    return out

for t in data['territories']:
    if t['id'] in ZONES:
        geom = t.get('_polys', [None])[0]
        zones = []
        if geom is not None:
            for fill, zpoly in ZONES[t['id']]:
                z = geom.intersection(zpoly)
                if z.is_empty or z.area < 0.01:
                    continue
                if not z.is_valid:
                    z = z.buffer(0)
                sz = serialize_geom(z)
                if sz:
                    zones.append({'fill': fill, 'geometry': sz})
        t['zones'] = zones
    t.pop('_polys', None)

# context: paint neutral pale sage (as original paints non-African land)
data['contextFill'] = '#e2ecd8'
data['palette'] = PAL

with open(BASE + '/data.js', 'w') as fh:
    fh.write('// Generated by scripts/build_data.py + scripts/postprocess.py — Africa 1914\n')
    fh.write('// Territory palette decoded from the original Larousse map legend\n')
    fh.write('window.MAP_DATA = ')
    json.dump(data, fh, ensure_ascii=False, separators=(',', ':'))
    fh.write(';\n')

import os
orphan = [c['n'] for c in data['cities'] if not c['t']]
print('cities without territory:', orphan)
print('territories:', len(data['territories']), 'file:', os.path.getsize(BASE+'/data.js')//1024, 'KB')
for t in sorted(data['territories'], key=lambda x: -x['area'])[:8]:
    print(f"  {t['id']:14s} {t['area']:>10,} km²  fill={t['fill']}")
