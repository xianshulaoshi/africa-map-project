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

# territory -> era of its base fill, from a dense pixel census of the original map
# (scripts/shade_census.py): shade = era of occupation (deep=by 1870, mid=1870-1900,
# light=1900-1914), painted sub-regionally exactly as the original does.
ERA = {
    'tunisie': '1870', 'reunion': '1870', 'somalie-fr': '1870', 'basutoland': '1870',
    'maroc': '1914', 'tchad': '1914',
    'guinee': '1900', 'cote-ivoire': '1900', 'dahomey': '1900',
    'haute-volta': '1900', 'gabon': '1900', 'moyen-congo': '1900',
    'madagascar': '1900', 'comores': '1900',
    'sierra': '1870', 'gambie': '1870', 'maurice': '1870', 'seychelles': '1870',
    'bechuanaland': '1900', 'swaziland': '1900', 'ouganda': '1900', 'nyassaland': '1900',
    'zanzibar': '1900', 'rhodesie-n': '1900', 'rhodesie-s': '1900',
    'gea': '1900', 'togo': '1900', 'gswa': '1914', 'congo-belge': '1900',
    'guinee-pt': '1870', 'cabo-verde': '1870', 'sao-tome': '1870',
    'madeira': '1870',
    'sahara-esp': '1870', 'rio-oro': '1870', 'maroc-esp': '1914', 'guinee-esp': '1870',
    'canaries': '1870', 'fernando-poo': '1870',
    'erythree': '1900', 'somalie-it': '1900',
    'libye': 'zones',  # Ottoman teal coast + beige interior (Fezzan) as in the original
    'abyssinie': 'indep', 'liberia': 'indep',
    # multi-shade territories (base + era zones, as in the original):
    'algerie': 'zones', 'senegal': 'zones', 'soudan-fr': 'zones', 'niger': 'zones',
    'mauritanie': 'zones', 'oubangui': 'zones', 'egypte': 'zones', 'bea': 'zones',
    'soudan-eg': 'zones', 'nigeria': 'zones', 'gold-coast': 'zones', 'afriquesud': 'zones', 'ouganda': 'zones',
    'somalie-br': 'zones', 'kamerun': 'zones', 'angola': 'zones', 'mozambique': 'zones',
}

# era zones: territory -> list of (fill, lon/lat polygon), clipped to the territory.
# Band boundaries measured from the per-latitude/longitude census of the original.
from shapely.geometry import box as SBox
def band(lo, hi):  # latitude band, whole width
    return SBox(-30, lo, 60, hi)
ZONES = {
    # Algeria: deep Tell/Atlas band (1830s), narrow mid transition, light deep Sahara
    'algerie': [
        ('#0878b8', band(31.5, 37.5)),
        ('#4890c8', band(30.4, 31.5)),
    ],
    # Senegal: deep old-colony band along the Senegal river, medium rest
    'senegal': [
        ('#0878b8', SBox(-17.7, 15.2, -13.0, 16.8)),
    ],
    # French Sudan: medium south, light north above ~17.8N (Azawad)
    'soudan-fr': [
        ('#88b8e0', band(17.8, 28.0)),
    ],
    # Niger: medium southern strip, light north above ~14.3N
    'niger': [
        ('#88b8e0', band(14.3, 28.0)),
    ],
    # Mauritania: light Sahara, medium southern band (river/Trarza)
    'mauritanie': [
        ('#4890c8', band(16.3, 17.8)),
    ],
    # Ubangi-Shari: medium south, light north above ~7.7N
    'oubangui': [
        ('#88b8e0', band(7.7, 11.5)),
    ],
    # Egypt: medium, light SW desert (oases region)
    'egypte': [
        ('#e0a8c8', SBox(24.7, 21.7, 29.9, 27.4)),
    ],
    # British East Africa: medium, light NE corner (Lake Rudolf / NFD, post-1907)
    'bea': [
        ('#e0a8c8', SBox(35.8, 1.6, 43.2, 4.6)),
    ],
    # Uganda: medium, light NE corner (Karamoja)
    'ouganda': [
        ('#e0a8c8', SBox(34.5, 2.6, 36.3, 4.5)),
    ],
    # Anglo-Egyptian Sudan: LIGHT overall; medium Nile corridor + Red Sea hills
    'soudan-eg': [
        ('#c088b8', SBox(29.6, 7.4, 39.0, 22.5)),
    ],
    # Nigeria: light north (protectorate 1900-03), mid middle, deep south (Lagos 1861/Oil Rivers)
    'nigeria': [
        ('#e0a8c8', band(11.0, 14.5)),
        ('#a060a0', SBox(2.5, 3.0, 14.5, 6.8)),
    ],
    # Gold Coast: medium, deep south (colony 1874/Asante 1896)
    'gold-coast': [
        ('#a060a0', SBox(-3.2, 3.9, 1.2, 7.2)),
    ],
    # Union of South Africa: DEEP Cape base, MID Orange Free State, GRAY Transvaal
    'afriquesud': [
        ('#9090a0', SBox(25.8, -25.9, 31.8, -23.3)),
        ('#c088b8', SBox(23.5, -30.7, 29.9, -28.2)),
    ],
    # British Somaliland: light, medium coastal north band
    'somalie-br': [
        ('#c088b8', band(10.2, 12.5)),
    ],
    # Kamerun: mid green south (1884), light green north (Adamawa, conquered 1901-06)
    'kamerun': [
        ('#90b8a0', band(7.2, 13.6)),
    ],
    # Angola: deep coastal strip (Luanda/Benguela), mid center, light east+far south
    'angola': [
        ('#788838', SBox(11.0, -16.5, 15.8, -4.0)),
        ('#c0d080', SBox(20.2, -18.5, 24.5, -4.0)),
    ],
    # Mozambique: light north (Zambezi+), mid middle, deep old south
    'mozambique': [
        ('#788838', SBox(30.0, -27.0, 41.0, -19.6)),
        ('#a0a858', SBox(30.0, -19.6, 41.0, -18.6)),
    ],
}
# Libya: teal coast (Tripolitania/Cyrenaica), beige SE interior (Fezzan, turque 1914 shade)
ZONES['libye'] = [
    ('#d0c088', Polygon([(15.0, 24.5), (20.5, 28.2), (25.3, 28.2), (25.3, 19.5), (15.0, 19.5)])),
]
ZONE_BASE = {
    'libye': '#88c8b8',
    'algerie': '#88b8e0', 'senegal': '#4890c8', 'soudan-fr': '#4890c8', 'niger': '#4890c8',
    'mauritanie': '#88b8e0', 'oubangui': '#4890c8', 'egypte': '#c088b8', 'bea': '#c088b8', 'ouganda': '#c088b8',
    'soudan-eg': '#e0a8c8', 'nigeria': '#c088b8', 'gold-coast': '#c088b8',
    'afriquesud': '#a060a0', 'somalie-br': '#e0a8c8', 'kamerun': '#689878',
    'angola': '#a0a858', 'mozambique': '#c0d080',
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
