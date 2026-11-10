#!/usr/bin/env python3
"""Build the final territory dataset for the interactive map page.
- Extract African territories from world_1914.geojson
- Split FWA / AEF / Rhodesia blobs into component colonies using modern country
  borders clipped to the 1914 outer shape
- Add islands from Natural Earth
- Attach metadata (French name as on original map, Chinese, power, status)
- Simplify + compact output to site/data.js
"""
import json, math
from shapely.geometry import shape, mapping, Point
from shapely.ops import unary_union
from shapely.algorithms.polylabel import polylabel  # may not exist in 2.2; fallback

BASE = '/Users/weiaiwen1/Desktop/my-space/map-project'
OUT = BASE + '/data.js'

feats1914 = json.load(open(BASE + '/data/world_1914.geojson'))['features']
by1914 = {}
for f in feats1914:
    n = (f.get('properties') or {}).get('NAME')
    if n is None:
        continue
    g = shape(f['geometry'])
    by1914[n] = unary_union([by1914.get(n, None) or g, g]) if n in by1914 else g

ne = json.load(open(BASE + '/data/ne_50m_admin_0_countries.geojson'))['features']
byNE = {}
for f in ne:
    p = f['properties']
    nm = p.get('ADMIN') or p.get('NAME')
    byNE[nm] = byNE.get(nm, []) + [shape(f['geometry'])]

def modern(*names):
    return unary_union([g for n in names for g in byNE.get(n, [])])

def rep_label_point(geom):
    """Visual center via polylabel on largest polygon, fallback centroid."""
    try:
        polys = list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom]
    except Exception:
        polys = [geom]
    biggest = max(polys, key=lambda p: p.area)
    try:
        c = polylabel(biggest, tolerance=0.2)
        return round(c.x, 2), round(c.y, 2)
    except Exception:
        c = geom.centroid
        return round(c.x, 2), round(c.y, 2)

def simplify(g, tol=0.02):
    return g.simplify(tol, preserve_topology=True)

def geom_to_coords(g):
    """Return list of polygons, each [outer, hole1, ...] as arrays of [lon,lat]."""
    out = []
    polys = list(g.geoms) if g.geom_type == 'MultiPolygon' else [g]
    for p in polys:
        if p.is_empty or p.area < 0.0015:  # drop specks (<~150 km2)
            continue
        rings = [p.exterior.coords] + [r.coords for r in p.interiors]
        out.append([[[round(x, 3), round(y, 3)] for x, y in ring] for ring in rings])
    return out

territories = []

def addTerritory(tid, nameFr, nameZh, power, status, modernNames, geom, note='', cities=None, labelPt=None):
    if not geom.is_valid:
        geom = geom.buffer(0)
    geom = simplify(geom)
    lp = labelPt or rep_label_point(geom)
    territories.append({
        'id': tid, 'name': nameFr, 'zh': nameZh, 'power': power, 'status': status,
        'modern': modernNames, 'note': note,
        'cities': cities or [],
        'label': lp,
        'geometry': geom_to_coords(geom),
    })

FR = 'fr'; UK = 'uk'; DE = 'de'; BE = 'be'; PT = 'pt'; ES = 'es'; IT = 'it'; INDEP = 'indep'; OT = 'ot'

# ---------- split FWA ----------
fwa = by1914['French West Africa']
fwa_parts = [
    ('mauritanie', 'MAURITANIE', '毛里塔尼亚', ['毛里塔尼亚'], 'Mauritania'),
    ('senegal', 'SÉNÉGAL', '塞内加尔', ['塞内加尔'], 'Senegal'),
    ('soudan-fr', 'SOUDAN FRANÇAISE', '法属苏丹', ['马里'], 'Mali'),
    ('haute-volta', 'HAUTE-VOLTA', '上沃尔特', ['布基纳法索'], 'Burkina Faso'),
    ('guinee', 'GUINÉE FRANÇAISE', '法属几内亚', ['几内亚'], 'Guinea'),
    ('cote-ivoire', 'CÔTE D’IVOIRE', '科特迪瓦', ['科特迪瓦'], 'Ivory Coast'),
    ('dahomey', 'DAHOMEY', '达荷美', ['贝宁'], 'Benin'),
    ('niger', 'NIGER', '尼日尔', ['尼日尔'], 'Niger'),
]
fwa_geoms = {}
for tid, fr, zh, mod, NEname in fwa_parts:
    m = modern(NEname)
    inter = fwa.intersection(m)
    if inter.is_empty or inter.area < 0.5:
        print('!! FWA part empty:', tid); continue
    fwa_geoms[tid] = (fr, zh, mod, inter)
# merge FWA leftover pieces: the dataset blob extends over Chad (administered from
# French Sudan until 1906, then AEF) and the Saharan belt. Give Chad to the AEF build
# below; southern Territories du Sud -> Algeria; SE Moroccan Sahara -> Morocco;
# subtract Spanish possessions; remaining slivers -> nearest AOF colony.
spanish_ws = unary_union([by1914['Spanish Sahara'], by1914['Rio De Oro']])
kamerun1914 = by1914['Kamerun']
aef_area = unary_union([modern('Chad'), modern('Central African Republic'),
                        modern('Gabon'), modern('Republic of the Congo')]).difference(kamerun1914)
fwa_left = fwa.difference(unary_union([v[3] for v in fwa_geoms.values()]))
fwa_left = fwa_left.difference(spanish_ws).difference(aef_area)
alg_extra = fwa_left.intersection(modern('Algeria'))
mar_extra = fwa_left.intersection(modern('Morocco'))
fwa_left = fwa_left.difference(alg_extra).difference(mar_extra)
if not fwa_left.is_empty:
    pieces = list(fwa_left.geoms) if fwa_left.geom_type == 'MultiPolygon' else [fwa_left]
    for p in pieces:
        if p.area < 0.3: continue
        best_tid = min(fwa_geoms, key=lambda t: fwa_geoms[t][3].distance(p))
        fr, zh, mod, g = fwa_geoms[best_tid]
        fwa_geoms[best_tid] = (fr, zh, mod, unary_union([g, p]))
        print(f'FWA leftover piece (area {p.area:.1f}) merged into {best_tid}')
if not alg_extra.is_empty:
    by1914['Algeria'] = unary_union([by1914['Algeria'], alg_extra])
    print(f'Algeria extended by Territories du Sud: {alg_extra.area:.1f} deg2')
if not mar_extra.is_empty:
    by1914['Morocco'] = unary_union([by1914['Morocco'], mar_extra])
    print(f'Morocco extended by Saharan belt: {mar_extra.area:.1f} deg2')
for tid, (fr, zh, mod, g) in fwa_geoms.items():
    addTerritory(tid, fr, zh, FR, '法属西非殖民地', mod, g, '法属西非（AOF）成员，1895年成立联邦')

# ---------- AEF: dataset blob lacks Chad & Oubangui-Chari ----------
# Build from modern countries minus 1914 Kamerun (which correctly includes 1911 Neukamerun)
aef_defs = [
    ('gabon', 'GABON', '加蓬', ['加蓬'], 'Gabon'),
    ('moyen-congo', 'MOYEN-CONGO', '中央刚果', ['刚果（布）'], 'Republic of the Congo'),
    ('oubangui', 'OUBANGUI-CHARI', '乌班吉沙立', ['中非共和国'], 'Central African Republic'),
    ('tchad', 'TCHAD', '乍得', ['乍得'], 'Chad'),
]
for tid, fr, zh, mod, NEname in aef_defs:
    g = modern(NEname).difference(kamerun1914)
    if g.is_empty or g.area < 0.5:
        print('!! AEF part empty:', tid); continue
    addTerritory(tid, fr, zh, FR, '法属赤道非洲殖民地', mod, g, '法属赤道非洲（AEF）成员，1910年成立联邦')

# ---------- Somalia: dataset's British Somaliland is malformed; rebuild both ----------
from shapely.geometry import Polygon
somalia_mod = unary_union([modern('Somalia'), modern('Somaliland')])  # NE splits Somaliland
eth_ref = modern('Ethiopia')  # ≈ post-1897 Ogaden border
sl_clip = Polygon([(40.5, 12.9), (49.4, 12.2), (49.4, 11.3), (47.1, 7.9), (43.0, 9.3), (40.5, 10.8)])
british_sl = somalia_mod.intersection(sl_clip).difference(eth_ref).difference(by1914['Djibouti'])
addTerritory('somalie-br', 'SOMALIE BRITANNIQUE', '英属索马里', UK, '保护国（1884）', ['索马里兰'], british_sl, '英国保护国')
italian_sl = somalia_mod.difference(british_sl).difference(eth_ref).difference(by1914['British East Africa'])
addTerritory('somalie-it', 'SOMALIE ITALIENNE', '意属索马里', IT, '殖民地（1889年起）', ['索马里南部'], italian_sl, '1892年后逐步扩展，含保护领地')
# Abyssinia: dataset shape overlaps neighbors' coasts -> subtract rebuilt neighbors
abyssinie_geom = by1914['Abyssinia'].difference(british_sl).difference(italian_sl) \
    .difference(by1914['Eritrea']).difference(by1914['Djibouti']).difference(by1914['Egypt']) \
    .difference(by1914['Anglo-Egyptian Sudan']).difference(by1914['British East Africa']) \
    .difference(by1914['Uganda']).difference(by1914['German E. Africa (Tanganyika)'])
by1914['Abyssinia'] = abyssinie_geom

# ---------- split Rhodesia ----------
rh = by1914['Rhodesia']
north = rh.intersection(modern('Zambia'))
south = rh.intersection(modern('Zimbabwe'))
addTerritory('rhodesie-n', 'RHODÉSIE DU NORD', '北罗得西亚', UK, '英国保护国（1924年起直辖）', ['赞比亚'], north, '英国南非公司管理')
addTerritory('rhodesie-s', 'RHODÉSIE DU SUD', '南罗得西亚', UK, '英国殖民地（1923年起自治）', ['津巴布韦'], south, '英国南非公司管理')

# leftover check (informational)
fwa_left2 = fwa.difference(unary_union([v[3] for v in fwa_geoms.values()]))
rh_left = rh.difference(unary_union([modern('Zambia'), modern('Zimbabwe')]))
for nm, g in [('FWA leftover', fwa_left2), ('Rhodesia leftover', rh_left)]:
    if not g.is_empty:
        print(f'{nm}: area={g.area:.2f} deg2')

# ---------- direct territories from the 1914 dataset ----------
direct = [
    # id, dataset NAME, French, zh, power, status, modern
    ('algerie',   'Algeria', 'ALGÉRIE', '阿尔及利亚', FR, '殖民地（1830年征服）', ['阿尔及利亚']),
    ('tunisie',   'Tunisia', 'TUNISIE', '突尼斯', FR, '保护国（1881）', ['突尼斯']),
    ('maroc',     'Morocco', 'MAROC', '摩洛哥', FR, '保护国（1912）', ['摩洛哥']),
    ('maroc-esp', 'Spanish Morocco', 'MAROC ESP.', '西属摩洛哥', ES, '保护领地（1912）', ['摩洛哥北部']),
    ('sahara-esp','Spanish Sahara', 'SAHARA ESP.', '西属撒哈拉', ES, '殖民地（1884）', ['西撒哈拉北部']),
    ('rio-oro',   'Rio De Oro', 'RÍO DE ORO', '里奥德奥罗', ES, '殖民地（1884）', ['西撒哈拉南部']),
    ('libye',     'Libya', 'LIBYE', '利比亚', IT, '意属殖民地（1911–1912夺取）', ['利比亚'], '意土战争（1911–1912）后由意大利占领，原图按奥斯曼色标示'),
    ('egypte',    'Egypt', 'ÉGYPTE', '埃及', UK, '英国占领（1882），名义属奥斯曼帝国', ['埃及']),
    ('soudan-eg', 'Anglo-Egyptian Sudan', 'SOUDAN ÉGYPTIEN', '英埃苏丹', UK, '英埃共管（1899）', ['苏丹', '南苏丹']),
    ('erythree',  'Eritrea', 'ÉRYTHRÉE', '厄立特里亚', IT, '殖民地（1890）', ['厄立特里亚']),
    ('abyssinie', 'Abyssinia', 'ABYSSINIE', '阿比西尼亚（埃塞俄比亚）', INDEP, '独立国家', ['埃塞俄比亚'], '阿杜瓦战役（1896）后保持独立'),
    ('somalie-fr','Djibouti', 'SOMALIE FRANÇAISE', '法属索马里', FR, '殖民地（1883–1887）', ['吉布提']),
    ('gambie',    'Gambia, The', 'GAMBIE', '冈比亚', UK, '殖民地（1816年起）', ['冈比亚']),
    ('sierra',    'Sierra Leone', 'SIERRA LEONE', '塞拉利昂', UK, '殖民地（1808）', ['塞拉利昂']),
    ('gold-coast','Gold Coast', 'GOLD COAST', '黄金海岸', UK, '殖民地（1874起扩展）', ['加纳']),
    ('togo',      'Togoland', 'TOGO', '多哥兰', DE, '保护领地（1884）', ['多哥及加纳/贝宁边缘']),
    ('nigeria',   'Nigeria', 'NIGERIA', '尼日利亚', UK, '殖民地（1914年南北合并）', ['尼日利亚']),
    ('kamerun',   'Kamerun', 'CAMEROUN', '喀麦隆', DE, '保护领地（1884）', ['喀麦隆及尼日利亚/乍得/中非边缘']),
    ('congo-belge','Belgian Congo', 'CONGO BELGE', '比属刚果', BE, '比利时殖民地（1908，前为国王领地）', ['刚果（金）']),
    ('gea',       'German E. Africa (Tanganyika)', 'AFRIQUE ORIENTALE ALLEMANDE', '德属东非', DE, '保护领地（1885）', ['坦桑尼亚、卢旺达、布隆迪']),
    ('bea',       'British East Africa', 'AFRIQUE ORIENTALE BRITANNIQUE', '英属东非', UK, '保护国（1895）', ['肯尼亚']),
    ('ouganda',   'Uganda', 'OUGANDA', '乌干达', UK, '保护国（1894）', ['乌干达']),
    ('gswa',      'German South-West Africa', 'SUD-OUEST ALLEMAND', '德属西南非洲', DE, '保护领地（1884）', ['纳米比亚']),
    ('nyassaland','Malawi', 'NYASSALAND', '尼亚萨兰', UK, '保护国（1891）', ['马拉维']),
    ('bechuanaland','Botswana', 'BECHUANALAND', '贝专纳兰', UK, '保护国（1885）', ['博茨瓦纳']),
    ('swaziland', 'Swaziland', 'SWAZILAND', '斯威士兰', UK, '保护国（1903）', ['斯威士兰']),
    ('basutoland','Lesotho', 'BASUTOLAND', '巴苏陀兰', UK, '保护国（1884）', ['莱索托']),
    ('afriquesud','South Africa', 'UNION D’AFRIQUE DU SUD', '南非联邦', UK, '英自治领（1910年成立）', ['南非']),
    ('angola',    'Angola', 'ANGOLA', '安哥拉', PT, '殖民地（1575年起）', ['安哥拉']),
    ('mozambique','Mozambique', 'MOZAMBIQUE', '莫桑比克', PT, '殖民地（1498年起）', ['莫桑比克']),
    ('guinee-pt', 'Portuguese Guinea', 'GUINÉE PORTUGAISE', '葡属几内亚', PT, '殖民地（1879）', ['几内亚比绍']),
    ('liberia',   'Liberia', 'LIBÉRIA', '利比里亚', INDEP, '独立国家（1847年建国）', ['利比里亚']),
    ('guinee-esp','Equatorial Guinea', 'GUINÉE ESP.', '西属几内亚', ES, '殖民地（1778年起）', ['赤道几内亚'], '含费尔南多波岛（Fernando Póo）'),
    ('madagascar','Madagascar (France)', 'MADAGASCAR', '马达加斯加', FR, '殖民地（1896）', ['马达加斯加']),
]

for row in direct:
    tid, dsname, fr, zh, pw, st, mod = row[:7]
    extra_note = row[7] if len(row) > 7 else ''
    g = by1914.get(dsname)
    if g is None or g.is_empty:
        print('!! missing in 1914 dataset:', dsname); continue
    note = extra_note
    if tid == 'egypte': note = '名义上属奥斯曼帝国，实际由英国占领控制'
    if tid == 'gea': note = '包括卢旺达（RWANDA）与布隆迪（BURUNDI）'
    if tid == 'congo-belge': note = '1885–1908年为利奥波德二世的“刚果自由邦”'
    if tid == 'abyssinie': note = '1896年阿杜瓦战役击败意大利，保持独立'
    addTerritory(tid, fr, zh, pw, st, mod, g, note)

# ---------- islands ----------
def island_from(geom_union, bbox):
    """Extract pieces of a multipolygon within bbox = (lon0, lat0, lon1, lat1)."""
    from shapely.geometry import box
    return geom_union.intersection(box(*bbox))

canaries = island_from(modern('Spain'), (-19.0, 27.0, -13.0, 29.6))
addTerritory('canaries', 'CANARIES', '加那利群岛', ES, '西班牙殖民地', ['加那利群岛（西）'], canaries)
madeira = island_from(modern('Portugal'), (-17.6, 31.5, -15.8, 33.2))
if not madeira.is_empty and madeira.area > 0.005:
    addTerritory('madeira', 'MADEIRA', '马德拉', PT, '葡萄牙殖民地', ['马德拉（葡）'], madeira)
cv = modern('Cabo Verde')
if not cv.is_empty:
    addTerritory('cabo-verde', 'CABO VERDE', '佛得角', PT, '葡萄牙殖民地', ['佛得角'], cv)
stp = modern('São Tomé and Principe') if 'São Tomé and Principe' in byNE else modern('Sao Tome and Principe')
if not stp.is_empty:
    addTerritory('sao-tome', 'SÃO TOMÉ E PRÍNCIPE', '圣多美和普林西比', PT, '葡萄牙殖民地', ['圣多美和普林西比'], stp)
comores = modern('Comoros')
if not comores.is_empty:
    addTerritory('comores', 'COMORES', '科摩罗', FR, '法国保护领地', ['科摩罗'], comores)
maurice = modern('Mauritius')
if not maurice.is_empty:
    addTerritory('maurice', 'MAURICE', '毛里求斯', UK, '英国殖民地（1810年夺取）', ['毛里求斯'], maurice)
seych = modern('Seychelles')
if not seych.is_empty:
    addTerritory('seychelles', 'SÉYCHELLES', '塞舌尔', UK, '英国殖民地', ['塞舌尔'], seych)
reunion = island_from(modern('France'), (50.0, -25.0, 62.0, -15.0))
if not reunion.is_empty and reunion.area > 0.005:
    addTerritory('reunion', 'RÉUNION', '留尼汪', FR, '法国殖民地', ['留尼汪（法）'], reunion)
fernando = island_from(modern('Equatorial Guinea'), (3.0, -1.0, 12.0, 6.5))  # mainland too; island = Bioko
bioko = island_from(modern('Equatorial Guinea'), (8.2, 3.2, 9.5, 4.0))
if not bioko.is_empty:
    addTerritory('fernando-poo', 'FERNANDO PÓO', '费尔南多波岛', ES, '西班牙殖民地', ['比奥科岛'], bioko)

# Zanzibar: hand-drawn simplified (Unguja + Pemba)
zanzibar_poly = {
    'type': 'Polygon',
    'coordinates': [[
        [39.28, -6.45],[39.38, -6.35],[39.48, -6.25],[39.55, -6.18],[39.35, -5.92],[39.22, -5.98],[39.18, -6.15],[39.28, -6.45]
    ]]
}
pemba_poly = {
    'type': 'Polygon',
    'coordinates': [[
        [39.62, -5.45],[39.75, -5.35],[39.88, -5.18],[39.82, -4.87],[39.68, -4.92],[39.60, -5.15],[39.62, -5.45]
    ]]
}
from shapely.geometry import shape as shp_shape
zg = unary_union([shp_shape(zanzibar_poly), shp_shape(pemba_poly)])
addTerritory('zanzibar', 'ZANZIBAR', '桑给巴尔', UK, '英国保护国（1890）', ['桑给巴尔（坦桑尼亚）'], zg)

# ---------- context (non-selectable): Europe/Arabia corners ----------
context_names = ['Spain', 'Portugal', 'France', 'Kingdom of Italy', 'Ottoman Empire', 'Arabia (Nejd)', 'United Kingdom of Great Britain and Ireland', 'Greece', 'Montenegro', 'Albania', 'Serbia', 'Bulgaria']
context = []
from shapely.geometry import box
frame = box(-32, -40, 62, 44)
for n in context_names:
    g = by1914.get(n)
    if g is None or g.is_empty: continue
    g = g.intersection(frame)
    if g.is_empty: continue
    context.append({'name': n, 'geometry': geom_to_coords(simplify(g, 0.05))})

# ---------- lakes ----------
lakes_raw = json.load(open(BASE + '/data/ne_50m_lakes.geojson'))['features']
lake_map = {
    'Lake Victoria': ('Lac Victoria', None),
    'Lake Tanganyika': ('Lac Tanganyika', None),
    'Lake Malawi': ('Lac Nyassa', None),
    'Lake Turkana': ('Lac Rodolphe', None),
    'Lake Chad': ('Lac Tchad', None),
    'Lake Kivu': (None, None),
    'Lake Albert': (None, None),
    'Lake Edward': (None, None),
    'Lake Mweru': (None, None),
    'Lake Bangweulu': (None, None),
    'Lake Rukwa': (None, None),
}
lakes = []
for f in lakes_raw:
    nm = f['properties'].get('name') or ''
    if nm in lake_map and lake_map[nm][0]:
        g = simplify(shape(f['geometry']), 0.03)
        lakes.append({'n': lake_map[nm][0], 'geometry': geom_to_coords(g)})
# historical Lake Chad outline (much larger than today's remnant)
chad_hist = shp_shape({'type': 'Polygon', 'coordinates': [[
    [13.0, 13.6],[13.6, 13.9],[14.3, 13.7],[15.1, 13.4],[15.4, 13.0],[15.3, 12.5],[15.6, 12.1],[15.0, 11.9],[14.4, 12.2],[13.9, 12.6],[13.4, 13.0],[13.0, 13.6]
]]})
lakes = [l for l in lakes if l['n'] != 'Lac Tchad']
lakes.append({'n': 'Lac Tchad', 'geometry': geom_to_coords(chad_hist)})

# ---------- rivers ----------
rivers_raw = json.load(open(BASE + '/data/ne_50m_rivers_lake_centerlines.geojson'))['features']
river_names = {'Nile': 'Nil', 'Niger': 'Niger', 'Congo': 'Congo', 'Zambezi': 'Zambèze', 'Orange': 'Orange', 'Limpopo': 'Limpopo', 'Senegal': 'Sénégal', 'Ubangi': 'Oubangui', 'Lualaba': 'Lualaba', 'Chari': 'Chari'}
rivers = []
for f in rivers_raw:
    nm = f['properties'].get('name') or ''
    if nm in river_names:
        g = shape(f['geometry']).intersection(frame)
        if g.is_empty: continue
        if g.geom_type == 'MultiLineString':
            g = max(g.geoms, key=lambda l: l.length)
        rivers.append({'n': river_names[nm], 'pts': [[round(x,3), round(y,3)] for x, y in g.simplify(0.08).coords]})

# ---------- cities ----------
cities_raw = json.load(open(BASE + '/data/ne_50m_populated_places_simple.geojson'))['features']
city_coords = {}
for f in cities_raw:
    p = f['properties']
    nm = p.get('name')
    if nm:
        city_coords.setdefault(nm, []).append((p.get('adm0name', ''), p.get('latitude'), p.get('longitude')))

def pick(name, country_hint=None):
    lst = city_coords.get(name)
    if not lst: return None
    if country_hint:
        for c, la, lo in lst:
            if country_hint.lower() in c.lower(): return (la, lo)
    return (lst[0][1], lst[0][2])

# (display name, zh, era capital?, modern NE name, country hint) or explicit coords
city_list = [
    ('Dakar', '达喀尔', 1, 'Dakar', 'Senegal'), ('Saint-Louis', '圣路易', 0, None, None, (16.03, -16.49)),
    ('Bamako', '巴马科', 1, 'Bamako', 'Mali'), ('Conakry', '科纳克里', 1, 'Conakry', 'Guinea'),
    ('Abidjan', '阿比让', 0, 'Abidjan', 'Ivory'), ('Porto-Novo', '波多诺伏', 1, 'Porto-Novo', 'Benin'),
    ('Niamey', '尼亚美', 1, 'Niamey', 'Niger'), ('Zinder', '津德尔', 0, 'Zinder', 'Niger'),
    ('Agadez', '阿加德兹', 0, 'Agadez', 'Niger'), ('Ouagadougou', '瓦加杜古', 1, 'Ouagadougou', 'Burkina'),
    ('Tombouctou', '廷巴克图', 0, 'Timbuktu', 'Mali'), ('Kayes', '卡伊', 0, 'Kayes', 'Mali'),
    ('Libreville', '利伯维尔', 1, 'Libreville', 'Gabon'), ('Brazzaville', '布拉柴维尔', 1, 'Brazzaville', 'Congo'),
    ('Bangui', '班吉', 1, 'Bangui', 'Central African'), ('Fort-Lamy', '拉米堡（今恩贾梅纳）', 1, None, None, (12.11, 15.05)),
    ('Abéché', '阿贝歇', 0, None, None, (13.83, 20.83)),
    ('Djibouti', '吉布提', 1, 'Djibouti', 'Djibouti'),
    ('Tananarive', '塔那那利佛', 1, 'Antananarivo', 'Madagascar'), ('Tamatave', '塔马塔夫', 0, None, None, (-18.15, 49.40)),
    ('Diego-Suarez', '迭戈苏瓦雷斯', 0, 'Antsiranana', 'Madagascar'), ('Majunga', '马任加', 0, 'Mahajanga', 'Madagascar'),
    ('Tuléar', '图莱亚尔', 0, 'Toliara', 'Madagascar'),
    ('Rabat', '拉巴特', 1, 'Rabat', 'Morocco'), ('Fès', '非斯', 0, None, None, (34.03, -5.00)),
    ('Marrakech', '马拉喀什', 0, 'Marrakesh', 'Morocco'), ('Casablanca', '卡萨布兰卡', 0, 'Casablanca', 'Morocco'),
    ('Tanger', '丹吉尔', 0, None, None, (35.77, -5.80)),
    ('Alger', '阿尔及尔', 1, 'Algiers', 'Algeria'), ('Oran', '奥兰', 0, 'Oran', 'Algeria'),
    ('Constantine', '君士坦丁', 0, 'Constantine', 'Algeria'),
    ('Tunis', '突尼斯城', 1, 'Tunis', 'Tunisia'), ('Bizerte', '比塞大', 0, None, None, (37.27, 9.87)),
    ('Le Caire', '开罗', 1, 'Cairo', 'Egypt'), ('Alexandrie', '亚历山大港', 0, 'Alexandria', 'Egypt'),
    ('Port-Saïd', '塞得港', 0, None, None, (31.26, 32.30)), ('Assouan', '阿斯旺', 0, 'Aswan', 'Egypt'),
    ('Khartoum', '喀土穆', 1, 'Khartoum', 'Sudan'), ('Omdurman', '恩图曼', 0, 'Omdurman', 'Sudan'),
    ('Freetown', '弗里敦', 1, 'Freetown', 'Sierra'), ('Monrovia', '蒙罗维亚', 1, 'Monrovia', 'Liberia'),
    ('Accra', '阿克拉', 1, 'Accra', 'Ghana'), ('Kumasi', '库马西', 0, None, None, (6.69, -1.62)),
    ('Lagos', '拉各斯', 0, 'Lagos', 'Nigeria'), ('Kano', '卡诺', 0, 'Kano', 'Nigeria'),
    ('Ibadan', '伊巴丹', 0, 'Ibadan', 'Nigeria'),
    ('Lomé', '洛美', 1, 'Lomé', 'Togo'),
    ('Douala', '杜阿拉', 0, 'Douala', 'Cameroon'), ('Yaoundé', '雅温得', 0, 'Yaoundé', 'Cameroon'),
    ('Léopoldville', '利奥波德维尔（今金沙萨）', 1, None, None, (-4.30, 15.30)),
    ('Stanleyville', '斯坦利维尔（今基桑加尼）', 0, None, None, (0.52, 25.20)),
    ('Élisabethville', '伊丽莎白维尔（今卢本巴希）', 0, None, None, (-11.66, 27.47)),
    ('Luanda', '罗安达', 1, 'Luanda', 'Angola'), ('Benguela', '本格拉', 0, 'Benguela', 'Angola'),
    ('Lobito', '洛比托', 0, None, None, (-12.36, 13.54)),
    ('Addis-Abeba', '亚的斯亚贝巴', 1, 'Addis Ababa', 'Ethiopia'), ('Harar', '哈勒尔', 0, None, None, (9.31, 42.13)),
    ('Mogadiscio', '摩加迪沙', 1, 'Mogadishu', 'Somalia'), ('Berbera', '柏培拉', 0, None, None, (10.44, 45.01)),
    ('Hargeisa', '哈尔格萨', 0, None, None, (9.56, 44.07)),
    ('Asmara', '阿斯马拉', 1, None, None, (15.34, 38.93)), ('Massaoua', '马萨瓦', 0, None, None, (15.61, 39.45)),
    ('Tripoli', '的黎波里', 1, 'Tripoli', 'Libya'), ('Benghazi', '班加西', 0, None, None, (32.12, 20.07)),
    ('Windhoek', '温得和克', 1, 'Windhoek', 'Namibia'), ('Swakopmund', '斯瓦科普蒙德', 0, None, None, (-22.68, 14.51)),
    ('Lüderitz', '吕德里茨', 0, None, None, (-26.65, 15.16)),
    ('Dar-es-Salaam', '达累斯萨拉姆', 1, 'Dar es Salaam', 'Tanzania'), ('Tabora', '塔波拉', 0, None, None, (-5.03, 32.82)),
    ('Mombasa', '蒙巴萨', 0, 'Mombasa', 'Kenya'), ('Nairobi', '内罗毕', 0, 'Nairobi', 'Kenya'),
    ('Kampala', '坎帕拉', 1, 'Kampala', 'Uganda'), ('Entebbe', '恩德培', 0, None, None, (0.05, 32.46)),
    ('Zanzibar', '桑给巴尔城', 1, 'Zanzibar', 'Tanzania'),
    ('Blantyre', '布兰太尔', 1, 'Blantyre', 'Malawi'), ('Zomba', '松巴', 0, None, None, (-15.38, 35.33)),
    ('Salisbury', '索尔兹伯里（今哈拉雷）', 1, None, None, (-17.83, 31.05)),
    ('Bulawayo', '布拉瓦约', 0, 'Bulawayo', 'Zimbabwe'), ('Livingstone', '利文斯敦', 0, 'Livingstone', 'Zambia'),
    ('Lourenço Marques', '洛伦索马克斯（今马普托）', 1, None, None, (-25.97, 32.57)),
    ('Beira', '贝拉', 0, 'Beira', 'Mozambique'), ('Quelimane', '克利马内', 0, 'Quelimane', 'Mozambique'),
    ('Le Cap', '开普敦', 1, 'Cape Town', 'South Africa'), ('Durban', '德班', 0, 'Durban', 'South Africa'),
    ('Johannesburg', '约翰内斯堡', 0, 'Johannesburg', 'South Africa'), ('Pretoria', '比勒陀利亚', 1, 'Pretoria', 'South Africa'),
    ('Bloemfontein', '布隆方丹', 0, 'Bloemfontein', 'South Africa'), ('Kimberley', '金伯利', 0, None, None, (-28.74, 24.77)),
    ('Port Elizabeth', '伊丽莎白港', 0, 'Port Elizabeth', 'South Africa'),
    ('Maseru', '马塞卢', 1, 'Maseru', 'Lesotho'),
]

cities = []
for row in city_list:
    disp, zh, cap = row[0], row[1], row[2]
    if row[3]:
        got = pick(row[3], row[4] if len(row) > 4 else None)
        if got is None:
            print('!! city not found in NE:', disp, row[3]); continue
        la, lo = got
    else:
        la, lo = row[5]
    cities.append({'n': disp, 'zh': zh, 'lat': round(la, 2), 'lon': round(lo, 2), 'cap': cap})

# ---------- write ----------
data = {
    'territories': territories,
    'context': context,
    'lakes': lakes,
    'rivers': rivers,
    'cities': cities,
}
import os
os.makedirs(BASE + '/site', exist_ok=True)
with open(OUT, 'w') as fh:
    fh.write('// Generated by scripts/build_data.py — Africa 1914 colonial territories\n')
    fh.write('window.MAP_DATA = ')
    json.dump(data, fh, ensure_ascii=False, separators=(',', ':'))
    fh.write(';\n')

ncoord = sum(len(r) for t in territories for poly in t['geometry'] for r in poly)
print(f'territories: {len(territories)}  context: {len(context)}  lakes: {len(lakes)}  rivers: {len(rivers)}  cities: {len(cities)}')
print('total coordinate points:', ncoord, ' file size:', os.path.getsize(OUT) // 1024, 'KB')
for t in territories:
    if not t['geometry']:
        print('!! empty geometry:', t['id'])

# ---------- overlap audit ----------
from shapely.geometry import Polygon as SPolygon
terr_polys = {}
for t in territories:
    ps = []
    for p in t['geometry']:
        try:
            ps.append(SPolygon(p[0], p[1:]))
        except Exception:
            pass
    if ps:
        terr_polys[t['id']] = unary_union(ps)
ids = list(terr_polys)
for i in range(len(ids)):
    for j in range(i+1, len(ids)):
        a, b = terr_polys[ids[i]], terr_polys[ids[j]]
        if a.bounds[2] < b.bounds[0] or b.bounds[2] < a.bounds[0] or a.bounds[3] < b.bounds[1] or b.bounds[3] < a.bounds[1]:
            continue
        ov = a.intersection(b).area
        if ov > 0.05:
            print(f'OVERLAP: {ids[i]} ∩ {ids[j]} = {ov:.2f} deg2')
