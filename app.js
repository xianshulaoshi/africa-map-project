/* 非洲殖民扩张 1870–1914 交互地图
 * Data: site/data.js (window.MAP_DATA)
 * Projection: Mercator. Unit space: x = lon+26, y = mercY(43)-mercY(lat).
 * Architecture: geometry lives in the scaled #viewport (strokes non-scaling);
 * text & dots live in the unscaled #overlay with screen-space positioning.
 */
(function () {
'use strict';

const D = window.MAP_DATA;
const NS = 'http://www.w3.org/2000/svg';
const svg = document.getElementById('map');
const viewport = document.getElementById('viewport');
const overlay = document.getElementById('overlay');
const gGrat = document.getElementById('gGraticule');
const gCtx = document.getElementById('gContext');
const gTerr = document.getElementById('gTerr');
const gLakes = document.getElementById('gLakes');
const gRivers = document.getElementById('gRivers');

const FRAME = { lon0: -26, lon1: 58, lat0: 43, lat1: -37 };
const RAD = Math.PI / 180;
function mercY(latDeg) { return Math.log(Math.tan(Math.PI / 4 + latDeg * RAD / 2)) / RAD; }
function ux(lon) { return lon - FRAME.lon0; }
function uy(lat) { return mercY(FRAME.lat0) - mercY(lat); }
const U_W = FRAME.lon1 - FRAME.lon0;
const U_H = uy(FRAME.lat1);

const POWERS = {
  fr:  { fr: 'France',            zh: '法国',       color: '#4890c8', swatches: ['#0878b8', '#4890c8', '#88b8e0'] },
  uk:  { fr: 'Royaume-Uni',       zh: '英国',       color: '#c088b8', swatches: ['#a060a0', '#c088b8', '#e0a8c8'] },
  de:  { fr: 'Allemagne',         zh: '德国',       color: '#689878', swatches: ['#689878', '#689878', '#90b8a0'] },
  be:  { fr: 'Belgique',          zh: '比利时',     color: '#d0a838', swatches: ['#d0a838', '#d0a838', '#d0a838'] },
  pt:  { fr: 'Portugal',          zh: '葡萄牙',     color: '#a0a858', swatches: ['#788838', '#a0a858', '#c0d080'] },
  es:  { fr: 'Espagne',           zh: '西班牙',     color: '#d88830', swatches: ['#b86818', '#d88830', '#f8b058'] },
  it:  { fr: 'Italie',            zh: '意大利',     color: '#b0a068', swatches: ['#b0a068', '#b0a068', '#d0c088'] },
  ot:  { fr: 'Empire ottoman',    zh: '奥斯曼帝国', color: '#88c8b8', swatches: ['#88c8b8', '#88c8b8', '#88c8b8'] },
  indep: { fr: 'État indépendant', zh: '独立国家',  color: '#d8d4ca', swatches: ['#d8d4ca', '#d8d4ca', '#d8d4ca'] },
};
const ERA_LABEL = {
  '1870': '1870 年前获得',
  '1900': '1870–1900 年间获得',
  '1914': '1900–1914 年间获得',
  'zones': '分区内按占领时期分色（深→浅 = 1870→1914，与原图一致）',
};

// leader-label offsets (screen px) for small territories: [dx, dy]
const LEADER = {
  'gambie': [-96, -36], 'togo': [10, 66], 'basutoland': [12, 80], 'swaziland': [82, 48],
  'somalie-fr': [-92, -42], 'guinee-pt': [-82, 28], 'maroc-esp': [70, -36], 'rio-oro': [-60, 20],
  'guinee-esp': [100, 38], 'sao-tome': [-44, 34], 'comores': [58, 12], 'maurice': [60, 10],
  'seychelles': [58, -10], 'reunion': [-60, 10], 'canaries': [6, -28], 'madeira': [8, -28],
  'cabo-verde': [6, -30], 'fernando-poo': [78, -14], 'zanzibar': [-78, 14], 'sierra': [-90, -22],
};
const SUBTITLE = { 'egypte': '(Empire ottoman)', 'abyssinie': '(ÉTHIOPIE)' };

function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function pathD(geometry) {
  let d = '';
  for (const poly of geometry) {
    for (const ring of poly) {
      ring.forEach((p, i) => {
        d += (i === 0 ? 'M' : 'L') + ux(p[0]).toFixed(2) + ' ' + uy(p[1]).toFixed(2) + ' ';
      });
      d += 'Z ';
    }
  }
  return d;
}

/* ---------- overlay registry (screen-space text & dots) ---------- */
const overlayItems = []; // {el, ux, uy, minZoom, terrId, kind, leaderOf?}
function addOverlay(elm, x, y, opts) {
  overlay.appendChild(elm);
  overlayItems.push(Object.assign({ el: elm, ux: x, uy: y, minZoom: 0 }, opts || {}));
  return elm;
}
function text(cls, content, x, y, opts) {
  const o = opts || {};
  const t = el('text', { class: cls, 'text-anchor': o.anchor || 'middle' });
  t.textContent = content;
  return addOverlay(t, x, y, o);
}

/* ---------- geometry layers ---------- */
for (const c of D.context) {
  el('path', { d: pathD(c.geometry), class: 'context-land', 'vector-effect': 'non-scaling-stroke' }, gCtx);
}
const terrEls = {}, terrById = {};
for (const t of D.territories) {
  terrById[t.id] = t;
  const isIndep = t.power === 'indep';
  const p = el('path', {
    d: pathD(t.geometry),
    class: 'terr' + (isIndep ? ' indep-fill' : ''),
    fill: isIndep ? undefined : t.fill,
    'vector-effect': 'non-scaling-stroke',
  }, gTerr);
  p.dataset.id = t.id;
  terrEls[t.id] = p;
  p.addEventListener('pointerenter', () => hover(t));
  p.addEventListener('pointerleave', unhover);
  p.addEventListener('click', (ev) => { ev.stopPropagation(); toggleSel(t.id); });
}
// era zones: sub-region shades painted on top of the base fill (as in the original);
// they never catch pointer events, so hover/click keep working through the base path
const zoneEls = {}, outlineEls = {};
for (const t of D.territories) {
  if (!t.zones) continue;
  for (const z of t.zones) {
    const zp = el('path', {
      d: pathD(z.geometry),
      class: 'terr-zone', fill: z.fill,
      'pointer-events': 'none', 'vector-effect': 'non-scaling-stroke',
    }, gTerr);
    zp.dataset.id = t.id;
    (zoneEls[t.id] = zoneEls[t.id] || []).push(zp);
  }
  // re-draw the outline above the zones so borders / selection stay crisp
  outlineEls[t.id] = el('path', {
    d: pathD(t.geometry), class: 'terr-outline', fill: 'none',
    'pointer-events': 'none', 'vector-effect': 'non-scaling-stroke',
  }, gTerr);
}
for (const l of D.lakes) el('path', { d: pathD(l.geometry), class: 'lake', 'vector-effect': 'non-scaling-stroke' }, gLakes);
for (const r of D.rivers) {
  let d = '';
  r.pts.forEach((p, i) => { d += (i === 0 ? 'M' : 'L') + ux(p[0]).toFixed(2) + ' ' + uy(p[1]).toFixed(2) + ' '; });
  el('path', { d, class: 'river', 'vector-effect': 'non-scaling-stroke' }, gRivers);
}
// graticule lines
for (let lon = -20; lon <= 50; lon += 10) {
  el('line', { x1: ux(lon), y1: uy(42.5), x2: ux(lon), y2: uy(-36.5), class: 'grat', 'vector-effect': 'non-scaling-stroke' }, gGrat);
}
for (let lat = 40; lat >= -30; lat -= 10) {
  el('line', { x1: ux(-25.5), y1: uy(lat), x2: ux(57.5), y2: uy(lat), class: 'grat', 'vector-effect': 'non-scaling-stroke' }, gGrat);
}

/* ---------- overlay content ---------- */
// graticule degree labels
for (let lon = -20; lon <= 50; lon += 10)
  text('grat-label', lon + '°', ux(lon), uy(-37) + 0.5, { minZoom: 0, edge: 'bottom' });
for (let lat = 40; lat >= -30; lat -= 10)
  text('grat-label', lat > 0 ? lat + '°' : lat === 0 ? '0°' : (-lat) + '°S', ux(-26), uy(lat), { minZoom: 0, anchor: 'start', dy: 3 });

// ocean labels
const OCEAN_LABELS = [
  ['OCÉAN ATLANTIQUE', -21.5, 8, 11.5],
  ['OCÉAN INDIEN', 51.5, -28, 11.5],
  ['Mer Méditerranée', 17, 36.4, 9],
  ['Golfe de Guinée', 1.5, -3.8, 8],
];
for (const [txt, lon, lat, size] of OCEAN_LABELS)
  text('ocean-label', txt, ux(lon), uy(lat), { size });
// Mer Rouge rotated: handled with a rotated group
{
  const g = el('g', {});
  const t = el('text', { class: 'ocean-label', 'text-anchor': 'middle' });
  t.textContent = 'Mer Rouge';
  g.appendChild(t);
  addOverlay(g, ux(40.2), uy(20), { size: 8, rotate: 55 });
}
// region titles
text('region-title', 'AFRIQUE OCCIDENTALE FRANÇAISE', ux(-4), uy(17.8), { size: 9, minZoom: 0.85 });
text('region-title', 'AFRIQUE ÉQUATORIALE FRANÇAISE', ux(18.5), uy(1.0), { size: 9, minZoom: 0.85 });

// territory labels (+ sub lines + leaders)
const labelRefs = {}; // terrId -> {main, sub, leader, x, y}
for (const t of D.territories) {
  const area = t.area || 0;
  let size, minZoom, small = false;
  if (area > 1500000) { size = 14.5; minZoom = 0; }
  else if (area > 800000) { size = 13; minZoom = 0; }
  else if (area > 350000) { size = 11.5; minZoom = 0.9; }
  else if (area > 120000) { size = 10.5; minZoom = 1.15; small = true; }
  else { size = 9; minZoom = 1.55; small = true; }
  const lead = LEADER[t.id];
  if (lead) minZoom = 0; // leader labels always visible
  const x = ux(t.label[0]), y = uy(t.label[1]);
  const main = text('terr-label' + (small ? ' s' : '') + (lead ? ' leader' : ''), t.name, x, y,
    { size, minZoom, terrId: t.id });
  let sub = null, leader = null;
  if (SUBTITLE[t.id])
    sub = text('terr-sub', SUBTITLE[t.id], x, y,
      { size: Math.round(size * 0.58), minZoom, terrId: t.id, subOffset: Math.round(size * 0.8) + 3 });
  if (lead) {
    leader = el('line', { class: 'leader-line' });
    addOverlay(leader, x, y, { minZoom: 0, leaderFor: t.id });
  }
  labelRefs[t.id] = { main, sub, leader, x, y, lead, size };
}
// ghost sub-labels in German East Africa
text('terr-label s', 'RWANDA', ux(29.9), uy(-2.1), { size: 8, minZoom: 1.7 });
text('terr-label s', 'BURUNDI', ux(29.6), uy(-3.8), { size: 8, minZoom: 1.7 });

// cities
for (const c of D.cities) {
  const dot = el('circle', { r: c.cap ? 2.2 : 1.6, class: 'city-dot' + (c.cap ? ' cap' : '') });
  addOverlay(dot, ux(c.lon), uy(c.lat), { minZoom: c.cap ? 0.95 : 1.45 });
  const lbl = el('text', {
    class: 'city-label', 'text-anchor': 'start',
    'font-style': c.cap ? 'normal' : 'italic',
  });
  if (c.cap) lbl.setAttribute('font-weight', '600');
  lbl.textContent = c.n;
  addOverlay(lbl, ux(c.lon), uy(c.lat), { size: c.cap ? 8 : 7, minZoom: c.cap ? 1.15 : 2.7, dx: 5, dy: -3, anchor: 'start' });
}

/* ---------- view state ---------- */
const view = { k: 1, tx: 0, ty: 0, S: 8, W: 0, H: 0 };
function resize() {
  const r = svg.getBoundingClientRect();
  view.W = r.width; view.H = r.height;
  view.S = Math.min(view.W / U_W, view.H / U_H) * 0.985;
  const ocean = document.getElementById('ocean');
  ocean.setAttribute('x', 0); ocean.setAttribute('y', 0);
  ocean.setAttribute('width', Math.max(1, r.width)); ocean.setAttribute('height', Math.max(1, r.height));
  applyView();
}
function toScreen(x, y) {
  const sc = view.S * view.k;
  return [view.W / 2 + (x + view.tx - U_W / 2) * sc, view.H / 2 + (y + view.ty - U_H / 2) * sc];
}
function clampView() {
  const sc = view.S * view.k;
  // keep at least 30% of the view showing map area
  const marginU = 0.35 * view.W / sc; // horizontal slack in units
  view.tx = Math.max(-U_W + marginU, Math.min(U_W - marginU, view.tx));
  const marginV = 0.35 * view.H / sc;
  view.ty = Math.max(-U_H + marginV, Math.min(U_H - marginV, view.ty));
}
function applyView() {
  clampView();
  const sc = view.S * view.k;
  // screen = W/2 + (u + tx - U_W/2) * sc  =>  translate = W/2 + (tx - U_W/2) * sc
  viewport.setAttribute('transform',
    `translate(${(view.W / 2 + (view.tx - U_W / 2) * sc).toFixed(2)},${(view.H / 2 + (view.ty - U_H / 2) * sc).toFixed(2)}) scale(${sc.toFixed(4)})`);
  // overlay items -> screen space
  for (const it of overlayItems) {
    if (it.leaderFor) continue; // drawn after labels are positioned
    let [sx, sy] = toScreen(it.ux, it.uy);
    const L = it.terrId ? labelRefs[it.terrId] : null;
    if (L && L.lead) { sx += L.lead[0]; sy += L.lead[1]; }
    if (it.subOffset) sy += it.subOffset;
    if (it.dx) sx += it.dx;
    if (it.dy) sy += it.dy;
    if (it.rotate) {
      it.el.setAttribute('transform', `translate(${sx.toFixed(1)},${sy.toFixed(1)}) rotate(${it.rotate})`);
    } else if (it.el.tagName === 'text') {
      it.el.setAttribute('x', sx.toFixed(1));
      it.el.setAttribute('y', sy.toFixed(1));
    } else {
      it.el.setAttribute('cx', sx.toFixed(1));
      it.el.setAttribute('cy', sy.toFixed(1));
    }
    if (it.size) it.el.setAttribute('font-size', it.size);
    it.el.style.display = (view.k >= it.minZoom || (it.terrId && selected.has(it.terrId))) ? '' : 'none';
  }
  // leader lines follow their labels
  for (const id in labelRefs) {
    const L = labelRefs[id];
    if (!L.leader) continue;
    const vis = L.main.style.display !== 'none' && L.main.style.opacity !== '0.15';
    const [ax, ay] = toScreen(L.x, L.y);
    const bx = parseFloat(L.main.getAttribute('x')) - Math.sign(L.lead[0]) * 8;
    const by = parseFloat(L.main.getAttribute('y'));
    L.leader.setAttribute('x1', ax.toFixed(1)); L.leader.setAttribute('y1', ay.toFixed(1));
    L.leader.setAttribute('x2', bx.toFixed(1)); L.leader.setAttribute('y2', by.toFixed(1));
    L.leader.style.display = vis ? '' : 'none';
  }
  updateScaleBar();
}

/* ---------- pan & zoom ---------- */
let drag = null;
svg.addEventListener('pointerdown', (ev) => {
  if (ev.button !== 0) return;
  ev.preventDefault(); // block native text selection / image drag during pan
  drag = { x: ev.clientX, y: ev.clientY, tx: view.tx, ty: view.ty, moved: false };
  try { svg.setPointerCapture(ev.pointerId); } catch (e) {}
  svg.classList.add('panning');
});
svg.addEventListener('dragstart', (ev) => ev.preventDefault());
svg.addEventListener('pointermove', (ev) => {
  if (!drag) return;
  const sc = view.S * view.k;
  const dx = (ev.clientX - drag.x) / sc, dy = (ev.clientY - drag.y) / sc;
  if (Math.abs(ev.clientX - drag.x) + Math.abs(ev.clientY - drag.y) > 3) drag.moved = true;
  if (drag.moved) {
    view.tx = drag.tx + dx; view.ty = drag.ty + dy;
    applyView();
  }
});
svg.addEventListener('pointerup', () => { svg.classList.remove('panning'); drag = null; });
svg.addEventListener('pointerleave', () => { svg.classList.remove('panning'); if (drag) drag = null; });
svg.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  zoomAt(ev.clientX - svg.getBoundingClientRect().left, ev.clientY - svg.getBoundingClientRect().top,
    ev.deltaY < 0 ? 1.2 : 1 / 1.2);
}, { passive: false });
svg.addEventListener('dblclick', (ev) => {
  if (ev.target.closest('.terr')) return;
  zoomAt(ev.clientX - svg.getBoundingClientRect().left, ev.clientY - svg.getBoundingClientRect().top, 1.8);
});
function zoomAt(px, py, f) {
  const k2 = Math.min(30, Math.max(0.85, view.k * f));
  const sc1 = view.S * view.k, sc2 = view.S * k2;
  const wx = (px - view.W / 2) / sc1 - view.tx + U_W / 2;
  const wy = (py - view.H / 2) / sc1 - view.ty + U_H / 2;
  view.k = k2;
  view.tx = (px - view.W / 2) / sc2 - wx + U_W / 2;
  view.ty = (py - view.H / 2) / sc2 - wy + U_H / 2;
  applyView();
}
document.getElementById('zoomIn').onclick = () => zoomAt(view.W / 2, view.H / 2, 1.45);
document.getElementById('zoomOut').onclick = () => zoomAt(view.W / 2, view.H / 2, 1 / 1.45);
document.getElementById('zoomReset').onclick = () => { view.k = 1; view.tx = 0; view.ty = 0; applyView(); };

function flyTo(lon, lat, k) {
  const target = { k: k || 2.2, tx: U_W / 2 - ux(lon), ty: U_H / 2 - uy(lat) };
  const start = { k: view.k, tx: view.tx, ty: view.ty }, t0 = performance.now();
  (function step(now) {
    const p = Math.min(1, (now - t0) / 520), e = 1 - Math.pow(1 - p, 3);
    view.k = start.k + (target.k - start.k) * e;
    view.tx = start.tx + (target.tx - start.tx) * e;
    view.ty = start.ty + (target.ty - start.ty) * e;
    applyView();
    if (p < 1) requestAnimationFrame(step);
  })(t0);
}

/* ---------- scale bar ---------- */
function updateScaleBar() {
  const sc = view.S * view.k;
  const ucy = U_H / 2 + view.ty;
  const my = mercY(FRAME.lat0) - ucy;
  const latNow = Math.max(-34, Math.min(42, (2 * Math.atan(Math.exp(my * RAD)) - Math.PI / 2) / RAD));
  const kmPerPx = 111.32 * Math.cos(latNow * RAD) / sc;
  const targets = [100, 200, 500, 1000, 2000, 3000];
  let px = 120, km = 0;
  for (const t of targets) { km = t; px = t / kmPerPx; if (px >= 70 && px <= 190) break; }
  document.getElementById('scaleBarLine').style.width = px.toFixed(0) + 'px';
  document.getElementById('scaleBarText').textContent = km.toLocaleString() + ' km';
}

/* ---------- selection & tooltip ---------- */
const selected = new Set();
let filterPower = null;
const tooltip = document.getElementById('tooltip');

function toggleSel(id, fly) {
  if (selected.has(id)) selected.delete(id); else selected.add(id);
  syncSelection();
  if (selected.has(id) && fly) {
    const t = terrById[id];
    flyTo(t.label[0], t.label[1], t.area > 800000 ? 1.35 : 3);
  }
}
function syncSelection() {
  for (const id in terrEls) terrEls[id].classList.toggle('selected', selected.has(id));
  for (const id in outlineEls) outlineEls[id].classList.toggle('selected', selected.has(id));
  document.getElementById('selCount').textContent = selected.size;
  renderDetail();
  renderList();
  applyView();
  // make the detail panel visible in the sidebar (it sits below the list)
  if (selected.size) {
    const sb = document.getElementById('sidebar');
    const dp = document.getElementById('detailPanel');
    sb.scrollTo({ top: dp.offsetTop - sb.offsetTop - 6, behavior: 'smooth' });
  }
}
function hover(t) {
  svg.classList.add('hovering');
  if (outlineEls[t.id]) outlineEls[t.id].classList.add('hovered');
  const P = POWERS[t.power];
  tooltip.hidden = false;
  tooltip.innerHTML =
    `<div class="tt-name">${esc(t.name)}</div>` +
    `<div class="tt-zh">${esc(t.zh)} · 约 ${fmtWan(t.area)} 万 km²</div>` +
    `<span class="tt-power" style="background:${P.color};color:#fff">${esc(P.fr)} ${esc(P.zh)}</span>` +
    (t.status ? `<div class="tt-zh" style="margin-top:3px">${esc(t.status)}</div>` : '');
}
function unhover() {
  svg.classList.remove('hovering'); tooltip.hidden = true;
  document.querySelectorAll('.terr-outline.hovered').forEach(e => e.classList.remove('hovered'));
}
document.addEventListener('pointermove', (ev) => {
  if (tooltip.hidden) return;
  const wrap = document.getElementById('mapWrap').getBoundingClientRect();
  let x = ev.clientX - wrap.left + 14, y = ev.clientY - wrap.top + 14;
  if (x > wrap.width - 280) x -= 300;
  if (y > wrap.height - 130) y -= 140;
  tooltip.style.left = x + 'px'; tooltip.style.top = y + 'px';
});
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
function fmtWan(a) { return a >= 100000 ? Math.round(a / 10000) : (a / 10000).toFixed(1); }

/* ---------- power filter ---------- */
function setFilter(power) {
  filterPower = (filterPower === power) ? null : power;
  for (const t of D.territories) {
    const dim = filterPower && t.power !== filterPower;
    terrEls[t.id].classList.toggle('dimmed', dim);
    if (zoneEls[t.id]) zoneEls[t.id].forEach(z => z.style.opacity = dim ? 0.15 : 1);
    const L = labelRefs[t.id];
    if (L) { L.main.style.opacity = dim ? 0.15 : 1; if (L.sub) L.sub.style.opacity = dim ? 0.15 : 1; if (L.leader) L.leader.style.opacity = dim ? 0.15 : 1; }
  }
  document.querySelectorAll('.leg-row[data-power]').forEach(r =>
    r.classList.toggle('active', r.dataset.power === filterPower));
  renderList();
}

/* ---------- sidebar ---------- */
const legend = document.getElementById('legend');
for (const key of ['fr', 'uk', 'de', 'be', 'pt', 'es', 'it', 'ot']) {
  const P = POWERS[key];
  const row = document.createElement('div');
  row.className = 'leg-row'; row.dataset.power = key;
  const n = D.territories.filter(t => t.power === key).length;
  row.innerHTML = `<span class="leg-swatches">${P.swatches.map(c => `<i style="background:${c}"></i>`).join('')}</span>` +
    `<span class="leg-name"><b>${P.fr}</b> <span>${P.zh}</span></span><span class="leg-count">${n}</span>`;
  row.title = '点击高亮该列强的领地';
  row.onclick = () => setFilter(key);
  legend.appendChild(row);
}

const terrList = document.getElementById('terrList');
function renderList() {
  const q = document.getElementById('search').value.trim().toLowerCase();
  terrList.innerHTML = '';
  const groups = {};
  for (const t of D.territories) {
    if (filterPower && t.power !== filterPower) continue;
    if (q && !(t.name.toLowerCase().includes(q) || t.zh.includes(q) ||
        (POWERS[t.power].fr + POWERS[t.power].zh).toLowerCase().includes(q) || t.id.includes(q))) continue;
    (groups[t.power] = groups[t.power] || []).push(t);
  }
  const order = ['fr', 'uk', 'de', 'be', 'pt', 'es', 'it', 'ot', 'indep'];
  for (const key of order) {
    const list = groups[key]; if (!list) continue;
    const P = POWERS[key];
    const h = document.createElement('div');
    h.className = 'grp-head';
    h.innerHTML = `<span class="dot" style="background:${P.color}"></span>${P.fr} · ${P.zh}（${list.length}）`;
    terrList.appendChild(h);
    for (const t of list) {
      const item = document.createElement('div');
      item.className = 'tr-item' + (selected.has(t.id) ? ' selected' : '');
      item.innerHTML = `<span class="frn">${esc(t.name)}</span><span class="zhn">${esc(t.zh)}</span>` +
        `<span class="area">${fmtArea(t.area)}</span>`;
      item.onclick = () => toggleSel(t.id, !selected.has(t.id));
      terrList.appendChild(item);
    }
  }
  const shown = Object.values(groups).reduce((a, b) => a + b.length, 0);
  document.getElementById('listCount').textContent = `（${shown}/${D.territories.length}）`;
}
function fmtArea(a) {
  if (a >= 1000000) return (a / 1000000).toFixed(2) + 'M';
  if (a >= 1000) return Math.round(a / 1000) + 'k';
  return String(a);
}
function renderDetail() {
  const detail = document.getElementById('detail');
  const multi = document.getElementById('multiSummary');
  const ids = [...selected];
  if (ids.length === 0) {
    detail.className = 'empty';
    detail.innerHTML = '尚未选择区域<br><small>点击地图或列表选择一个区域；再次点击取消。</small>';
    multi.hidden = true; return;
  }
  const t = terrById[ids[ids.length - 1]];
  const P = POWERS[t.power];
  const cities = D.cities.filter(c => c.t === t.id).sort((a, b) => (b.cap - a.cap) || a.n.localeCompare(b.n));
  const cap = cities.find(c => c.cap);
  detail.className = '';
  detail.innerHTML =
    `<div class="d-name">${esc(t.name)}</div>` +
    (SUBTITLE[t.id] ? `<div class="d-sub">${esc(SUBTITLE[t.id])}</div>` : '') +
    `<div class="d-zh">${esc(t.zh)}</div>` +
    `<div class="d-chip"><i style="background:${t.power === 'indep' ? '#d8d4ca' : t.fill}"></i>${esc(P.fr)} ${esc(P.zh)}</div>` +
    `<div class="row"><b>获得时期</b><span>${t.power === 'indep' ? '—' : esc(ERA_LABEL[t.era] || '')}</span></div>` +
    `<div class="row"><b>地位</b><span>${esc(t.status)}</span></div>` +
    `<div class="row"><b>面积</b><span>约 ${t.area.toLocaleString()} km²</span></div>` +
    `<div class="row"><b>今属</b><span>${t.modern.map(esc).join('、')}</span></div>` +
    (cap ? `<div class="row"><b>首府</b><span>${esc(cap.n)}（${esc(cap.zh)}）</span></div>` : '') +
    (cities.length ? `<div class="row"><b>图中城市</b><span class="cities">${cities.slice(0, 14).map(c => `<span class="city-chip${c.cap ? ' cap' : ''}">${esc(c.n)}</span>`).join('')}</span></div>` : '') +
    (t.note ? `<div class="note">${esc(t.note)}</div>` : '');
  if (ids.length > 1) {
    const total = ids.reduce((a, id) => a + terrById[id].area, 0);
    multi.hidden = false;
    multi.innerHTML = `已选 <b>${ids.length}</b> 个区域，合计约 <b>${total.toLocaleString()}</b> km²<Br>` +
      ids.map(id => terrById[id].name).map(esc).join(' · ');
  } else multi.hidden = true;
}

document.getElementById('btnClearSel').onclick = () => { selected.clear(); syncSelection(); };
document.getElementById('search').addEventListener('input', renderList);
document.getElementById('search').addEventListener('keydown', ev => {
  if (ev.key === 'Enter') { const first = terrList.querySelector('.tr-item'); if (first) first.click(); }
});
document.addEventListener('keydown', ev => {
  if (ev.key === 'Escape') {
    if (filterPower) setFilter(filterPower);
    else { selected.clear(); syncSelection(); }
  }
});

/* ---------- init ---------- */
renderList();
renderDetail();
window.addEventListener('resize', resize);
window.addEventListener('load', () => { resize(); setTimeout(resize, 200); });
resize();

// debug/QA hook
window.__map = {
  reset() { view.k = 1; view.tx = 0; view.ty = 0; applyView(); },
  setView(k, tx, ty) { view.k = k; view.tx = tx; view.ty = ty; applyView(); },
  select(ids) { selected.clear(); ids.forEach(i => selected.add(i)); syncSelection(); },
  state() { return { ...view, selected: [...selected] }; },
};
})();
