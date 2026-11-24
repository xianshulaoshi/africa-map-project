/* 非洲殖民扩张 1870–1914 · 交互地图（原图复刻版）
 *
 * Everything drawn here is traced from source-map.png: the coordinate system
 * of the SVG is the pixel grid of the scan (673 x 710).  data.js holds
 *   water     shade of the sea, as an image layer
 *   land      land outlines (lakes are holes)
 *   patches   [colour class, territory, path] of every flat colour area
 *   lines     line work (rivers, 1870 frontiers, graticule ...)
 *   labels    lettering, with a table of text styles
 *   frame     title box, legend, scale bar, source line
 *   meta      curated notes per territory
 */
(function () {
'use strict';

const D = window.MAP_DATA;
const NS = 'http://www.w3.org/2000/svg';
const svg = document.getElementById('map');
const W = D.W, H = D.H;

const POWERS = [
  { id: 'fr', fr: 'française', zh: '法国', cls: ['fr-d', 'fr-m', 'fr-l'] },
  { id: 'uk', fr: 'britannique', zh: '英国', cls: ['uk-d', 'uk-m', 'uk-l'] },
  { id: 'de', fr: 'allemande', zh: '德国', cls: ['de-m', 'de-m', 'de-l'] },
  { id: 'be', fr: 'belge', zh: '比利时', cls: ['be'] },
  { id: 'pt', fr: 'portugaise', zh: '葡萄牙', cls: ['pt-d', 'pt-m', 'pt-l'] },
  { id: 'es', fr: 'espagnole', zh: '西班牙', cls: ['es-d', 'es-m', 'es-l'] },
  { id: 'it', fr: 'italienne', zh: '意大利', cls: ['it-m', 'it-l'] },
  { id: 'ot', fr: 'turque', zh: '奥斯曼', cls: ['ot'] },
  { id: 'boer', fr: 'des Boers / Afrikaners', zh: '布尔人', cls: ['boer'] },
  { id: 'indep', fr: 'État indépendant', zh: '独立国家', cls: ['indep'] },
];
const POWER = Object.fromEntries(POWERS.map(p => [p.id, p]));
const colorOf = c => D.colors[D.classes.indexOf(c)];

function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}

/* ---------- static layers ---------- */
const g = id => document.getElementById(id);

function drawWater() {
  el('image', { href: D.water, x: 0, y: 0, width: W, height: H, preserveAspectRatio: 'none' }, g('gWater'));
}
function drawLand() {
  const layer = g('gLand');
  for (const d of D.land) el('path', { d, fill: '#ffffff', stroke: '#ffffff', 'stroke-width': .3, 'stroke-linejoin': 'round' }, layer);
}
const patchEls = {};                        // territory -> [path]
const patchByTerr = {};
function drawTerr() {
  const layer = g('gTerr');
  for (const [ci, ti, d] of D.patches) {
    const fill = D.colors[ci];
    const p = el('path', {
      d, fill, stroke: fill, 'stroke-width': .3, class: 'pt', 'fill-rule': 'evenodd',
    }, layer);
    p.dataset.t = ti;
    (patchByTerr[ti] = patchByTerr[ti] || []).push(p);
  }
}
function drawLines() {
  if (!D.lines) return;
  const layer = g('gLines');
  for (const L of D.lines) {
    const a = { d: L.d, fill: 'none', stroke: L.c, 'stroke-width': L.w, 'stroke-linecap': L.cap || 'butt', 'stroke-linejoin': 'round' };
    if (L.dash) a['stroke-dasharray'] = L.dash;
    if (L.op != null) a['stroke-opacity'] = L.op;
    el('path', a, layer);
  }
}
/* lettering: every label is a <text> placed with translate / rotate / skewX
 * (the italic is an explicit shear, so that every browser draws it alike) */
function injectFonts() {
  if (!D.fonts) return;
  const css = D.fonts.map(([fam, w, b64]) =>
    `@font-face{font-family:${fam};font-weight:${w};font-style:normal;src:url(data:font/woff;base64,${b64}) format('woff')}`).join('');
  const st = document.createElement('style');
  st.textContent = css;
  document.head.appendChild(st);
}
function drawDots() {
  if (!D.dots) return;
  const layer = g('gLabels');
  for (const [x, y] of D.dots) {
    el('circle', { cx: x, cy: y + 0.3, r: 1.75, fill: '#fbfcfd', 'fill-opacity': .85 }, layer);
    el('circle', { cx: x, cy: y + 0.3, r: 1.15, fill: '#101014' }, layer);
  }
}
function drawLabels() {
  if (!D.labels) return;
  const layer = g('gLabels');
  for (const [x, y, txt, sid, size, track, rot, sx] of D.labels) {
    const s = D.styles[sid];
    const tf = [`translate(${x} ${y})`];
    if (rot) tf.push(`rotate(${rot})`);
    if (sx && sx !== 1) tf.push(`scale(${sx} 1)`);
    if (s.i) tf.push('skewX(-14)');
    const a = {
      transform: tf.join(' '), x: 0, y: 0, fill: s.c || '#1d1d1f',
      'font-family': D.fonts[s.ff][0], 'font-size': size, 'font-weight': s.w || 400,
      'letter-spacing': track || 0,
    };
    if (s.h) { a.stroke = s.h; a['stroke-width'] = s.hw || 1; a['paint-order'] = 'stroke'; a['stroke-linejoin'] = 'round'; }
    const t = el('text', a, layer);
    t.textContent = txt;
  }
}
function drawFrame() {
  if (!D.frame) return;
  const layer = g('gFrame');
  for (const P of D.frame) {
    if (P.k === 'rect') el('rect', { x: P.x, y: P.y, width: P.w, height: P.h, fill: P.f || 'none', stroke: P.s || 'none', 'stroke-width': P.sw || 0 }, layer);
    else if (P.k === 'shadow') el('rect', { x: P.x + 3, y: P.y + 3, width: P.w, height: P.h, fill: '#5c6478', 'fill-opacity': .5, filter: 'url(#softShadow)' }, layer);
    else if (P.k === 'path') {
      const a = { d: P.d, fill: P.f || 'none', stroke: P.s || 'none', 'stroke-width': P.sw || 0, 'stroke-linecap': P.cap || 'butt' };
      if (P.dash) a['stroke-dasharray'] = P.dash;
      el('path', a, layer);
    }
  }
}

injectFonts();
// the panels of the frame (title, legend) come before the lettering: their own text is lettering
drawWater(); drawLand(); drawTerr(); drawLines(); drawFrame(); drawDots(); drawLabels();
// once the embedded fonts are in, lay the map out again for the QA hook
if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { window.__fontsReady = true; });

/* ---------- view (zoom / pan through the viewBox) ---------- */
const view = { x: 0, y: 0, w: W, h: H };
function applyView() {
  svg.setAttribute('viewBox', `${view.x.toFixed(3)} ${view.y.toFixed(3)} ${view.w.toFixed(3)} ${view.h.toFixed(3)}`);
}
function toMap(ev) {
  const r = svg.getBoundingClientRect();
  return [(ev.clientX - r.left) / r.width * view.w + view.x, (ev.clientY - r.top) / r.height * view.h + view.y];
}
function zoomAt(mx, my, f) {
  const k = Math.min(40, Math.max(1, W / view.w * f));
  const w2 = W / k, h2 = H / k;
  view.x = mx - (mx - view.x) * w2 / view.w;
  view.y = my - (my - view.y) * h2 / view.h;
  view.w = w2; view.h = h2;
  clampView(); applyView();
}
function clampView() {
  view.x = Math.max(-view.w * .6, Math.min(W - view.w * .4, view.x));
  view.y = Math.max(-view.h * .6, Math.min(H - view.h * .4, view.y));
}
function resetView() { view.x = 0; view.y = 0; view.w = W; view.h = H; applyView(); }

let drag = null;
svg.addEventListener('pointerdown', ev => {
  if (ev.button !== 0) return;
  drag = { x: ev.clientX, y: ev.clientY, vx: view.x, vy: view.y, moved: false };
  try { svg.setPointerCapture(ev.pointerId); } catch (e) { /* ignore */ }
});
svg.addEventListener('pointermove', ev => {
  if (!drag) return;
  const r = svg.getBoundingClientRect();
  const dx = (ev.clientX - drag.x) / r.width * view.w, dy = (ev.clientY - drag.y) / r.height * view.h;
  if (Math.abs(ev.clientX - drag.x) + Math.abs(ev.clientY - drag.y) > 3) { drag.moved = true; svg.classList.add('panning'); }
  if (drag.moved) { view.x = drag.vx - dx; view.y = drag.vy - dy; clampView(); applyView(); }
});
const endDrag = () => { svg.classList.remove('panning'); drag = null; };
svg.addEventListener('pointerup', endDrag);
svg.addEventListener('pointercancel', endDrag);
svg.addEventListener('wheel', ev => {
  ev.preventDefault();
  const [mx, my] = toMap(ev);
  zoomAt(mx, my, ev.deltaY < 0 ? 1.25 : 1 / 1.25);
}, { passive: false });
svg.addEventListener('dblclick', ev => {
  if (ev.target.closest('.pt')) return;
  const [mx, my] = toMap(ev);
  zoomAt(mx, my, 1.8);
});
document.getElementById('zoomIn').onclick = () => zoomAt(view.x + view.w / 2, view.y + view.h / 2, 1.45);
document.getElementById('zoomOut').onclick = () => zoomAt(view.x + view.w / 2, view.y + view.h / 2, 1 / 1.45);
document.getElementById('zoomReset').onclick = resetView;

/* ---------- selection, filter, tooltip ---------- */
const selected = new Set();
let filterPower = null;
const terrById = Object.fromEntries(D.terr.map((t, i) => [t, i]));
const powerOf = t => (D.meta[t] || {}).power || '';
const tooltip = document.getElementById('tooltip');
const fmtArea = a => a == null ? '—' : a >= 1000000 ? (a / 1000000).toFixed(2) + 'M' : a >= 1000 ? Math.round(a / 1000) + 'k' : String(a);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function syncFx() {
  const active = selected.size > 0 || !!filterPower;
  for (const tiStr in patchByTerr) {
    const ti = +tiStr, t = D.terr[ti];
    const sel = selected.has(t), dim = filterPower ? powerOf(t) !== filterPower : false;
    for (const p of patchByTerr[tiStr]) {
      p.classList.toggle('selected', sel);
      p.classList.toggle('dim', active && !sel && dim);
    }
  }
  document.getElementById('selCount').textContent = selected.size;
}
function toggleSel(t, fromList) {
  if (selected.has(t)) selected.delete(t); else selected.add(t);
  syncFx(); renderList(); renderDetail();
  if (selected.has(t) && !fromList) focusTerr(t);
}
function focusTerr(t) {
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (const p of patchByTerr[terrById[t]] || []) {
    const b = p.getBBox();
    x0 = Math.min(x0, b.x); y0 = Math.min(y0, b.y); x1 = Math.max(x1, b.x + b.width); y1 = Math.max(y1, b.y + b.height);
  }
  if (x0 > x1) return;
  const k = Math.min(12, Math.max(W / view.w, Math.min(900 / (x1 - x0), 700 / (y1 - y0))));
  view.w = W / k; view.h = H / k;
  view.x = (x0 + x1) / 2 - view.w / 2; view.y = (y0 + y1) / 2 - view.h / 2;
  clampView(); applyView();
}
function hover(ev, t) {
  svg.classList.add('hovering');
  const m = D.meta[t] || {}, P = POWER[m.power] || { fr: '', zh: '' };
  tooltip.hidden = false;
  tooltip.innerHTML = `<div class="tt-name">${esc(m.name || t)}</div>` +
    `<div class="tt-zh">${esc(m.zh || '')}${m.area ? ' · 约 ' + esc(fmtArea(m.area)) + ' km²' : ''}</div>` +
    `<div class="tt-zh">${esc(P.fr)} ${esc(P.zh)}</div>`;
  moveTip(ev);
}
function moveTip(ev) {
  if (tooltip.hidden) return;
  const r = document.getElementById('stage').getBoundingClientRect();
  let x = ev.clientX - r.left + 14, y = ev.clientY - r.top + 14;
  if (x > r.width - 240) x -= 270;
  if (y > r.height - 110) y -= 120;
  tooltip.style.left = x + 'px'; tooltip.style.top = y + 'px';
}
svg.addEventListener('pointerover', ev => {
  const p = ev.target.closest('.pt');
  if (p) hover(ev, D.terr[+p.dataset.t]);
});
svg.addEventListener('pointerout', ev => {
  if (ev.target.closest('.pt')) { tooltip.hidden = true; svg.classList.remove('hovering'); }
});
svg.addEventListener('pointermove', moveTip);
svg.addEventListener('click', ev => {
  const p = ev.target.closest('.pt');
  if (!p || (drag && drag.moved)) return;
  toggleSel(D.terr[+p.dataset.t]);
});

/* ---------- side panel ---------- */
const panel = document.getElementById('panel');
let legendDone = false;
document.getElementById('btnPanel').onclick = () => {
  panel.hidden = !panel.hidden;
  document.getElementById('btnPanel').classList.toggle('on', !panel.hidden);
  if (!panel.hidden && !legendDone) buildLegend();
};
const isBare = new URLSearchParams(location.search).has('bare');
if (isBare) document.body.classList.add('bare');
else if (window.innerWidth >= 1280) document.getElementById('btnPanel').click();

function buildLegend() {
  legendDone = true;
  const legend = document.getElementById('legend');
  for (const P of POWERS) {
    const row = document.createElement('div');
    row.className = 'leg-row'; row.dataset.power = P.id;
    const n = D.terr.filter(t => powerOf(t) === P.id).length;
    row.innerHTML = `<span class="leg-sw">${P.cls.map(c => `<i style="background:${colorOf(c)}"></i>`).join('')}</span>` +
      `<span class="leg-name">${esc(P.fr)} <span>${esc(P.zh)}</span></span><span class="leg-count">${n}</span>`;
    row.onclick = () => setFilter(P.id);
    legend.appendChild(row);
  }
}
function setFilter(power) {
  filterPower = filterPower === power ? null : power;
  document.querySelectorAll('.leg-row').forEach(r => r.classList.toggle('active', r.dataset.power === filterPower));
  syncFx(); renderList();
}
function renderList() {
  const q = document.getElementById('search').value.trim().toLowerCase();
  const list = document.getElementById('terrList');
  list.innerHTML = '';
  let shown = 0;
  for (const P of POWERS) {
    const items = D.terr.filter(t => powerOf(t) === P.id)
      .filter(t => !filterPower || filterPower === P.id)
      .filter(t => !q || (D.meta[t].name || t).toLowerCase().includes(q) || (D.meta[t].zh || '').includes(q) || t.includes(q))
      .sort((a, b) => ((D.meta[b].area || 0) - (D.meta[a].area || 0)) || a.localeCompare(b));
    if (!items.length) continue;
    const h = document.createElement('div');
    h.className = 'grp-head';
    h.innerHTML = `<span class="dot" style="background:${colorOf(P.cls[0])}"></span>${esc(P.fr)} · ${esc(P.zh)}（${items.length}）`;
    list.appendChild(h);
    for (const t of items) {
      const m = D.meta[t];
      const item = document.createElement('div');
      item.className = 'tr-item' + (selected.has(t) ? ' selected' : '');
      item.innerHTML = `<span class="frn">${esc(m.name || t)}</span><span class="zhn">${esc(m.zh || '')}</span>` +
        (m.area ? `<span class="area">${esc(fmtArea(m.area))}</span>` : '');
      item.onclick = () => { toggleSel(t, true); if (selected.has(t)) focusTerr(t); };
      list.appendChild(item);
      shown++;
    }
  }
  document.getElementById('listCount').textContent = `（${shown}/${D.terr.length}）`;
}
function renderDetail() {
  const detail = document.getElementById('detail');
  if (!selected.size) {
    detail.className = 'empty';
    detail.innerHTML = '尚未选择领地<br><small>点击地图或列表选择一个领地；再次点击取消。</small>';
    return;
  }
  detail.className = '';
  detail.innerHTML = [...selected].map(t => {
    const m = D.meta[t] || {}, P = POWER[m.power] || { fr: m.power, zh: '' };
    return `<div class="d-name">${esc(m.name || t)}</div><div class="d-zh">${esc(m.zh || '')}</div>` +
      `<div class="d-chip"><i style="background:${colorOf(P.cls[0])}"></i>${esc(P.fr)} ${esc(P.zh)}</div>` +
      (m.status ? `<div class="row"><b>地位</b><span>${esc(m.status)}</span></div>` : '') +
      (m.area ? `<div class="row"><b>面积</b><span>约 ${Number(m.area).toLocaleString()} km²</span></div>` : '') +
      (m.modern && m.modern.length ? `<div class="row"><b>今属</b><span>${m.modern.map(esc).join('、')}</span></div>` : '') +
      (m.note ? `<div class="note">${esc(m.note)}</div>` : '');
  }).join('<hr style="border:none;border-top:1px solid var(--line);margin:10px 0">');
}
document.getElementById('btnClearSel').onclick = () => { selected.clear(); syncFx(); renderList(); renderDetail(); };
document.getElementById('search').addEventListener('input', renderList);
document.addEventListener('keydown', ev => {
  if (ev.key === 'Escape') {
    if (filterPower) setFilter(filterPower);
    else { selected.clear(); syncFx(); renderList(); renderDetail(); }
  }
});

applyView();
renderList(); renderDetail();

// QA self test: index.html?selftest=1 writes the results to <body data-selftest>
async function selfTest() {
  const r = {};
  await (document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve());
  await new Promise(res => setTimeout(res, 300));
  try {
    r.patches = document.querySelectorAll('#gTerr .pt').length;
    r.labels = document.querySelectorAll('#gLabels text').length;
    r.dots = document.querySelectorAll('#gLabels circle').length;
    r.territories = D.terr.length;
    const ids = ['egypte', 'niger', 'congo-belge'].filter(t => D.terr.includes(t));
    window.__map.select(ids);
    r.selectedPaths = document.querySelectorAll('.pt.selected').length;
    r.selCount = document.getElementById('selCount').textContent;
    r.detailHasName = ids.some(t => document.getElementById('detail').textContent.includes((D.meta[t].name || t)));
    const first = document.querySelector('.pt.selected');
    first.dispatchEvent(new PointerEvent('pointerover', { bubbles: true, clientX: 300, clientY: 300 }));
    r.tooltipShown = !document.getElementById('tooltip').hidden &&
      document.getElementById('tooltip').textContent.length > 3;
    first.dispatchEvent(new PointerEvent('pointerout', { bubbles: true }));
    r.tooltipHidden = document.getElementById('tooltip').hidden;
    r.listItems = document.querySelectorAll('#terrList .tr-item').length;
    const search = document.getElementById('search');
    search.value = '刚果'; search.dispatchEvent(new Event('input'));
    r.searchZh = document.querySelectorAll('#terrList .tr-item').length;
    search.value = 'gold'; search.dispatchEvent(new Event('input'));
    r.searchFr = document.querySelectorAll('#terrList .tr-item').length;
    search.value = ''; search.dispatchEvent(new Event('input'));
    document.getElementById('btnPanel').click();
    r.legendRows = document.querySelectorAll('.leg-row').length;
    setFilter('uk');
    r.dimPaths = document.querySelectorAll('.pt.dim').length;
    setFilter('uk');
    r.dimCleared = document.querySelectorAll('.pt.dim').length === 0;
    zoomAt(336, 355, 2);
    r.zoomed = svg.getAttribute('viewBox');
    resetView();
    r.viewReset = svg.getAttribute('viewBox');
    r.fontLoaded = document.fonts.check('12px "Scan Sans"') && document.fonts.check('600 12px "Scan Sans Condensed"') &&
      document.fonts.check('700 12px "Scan Narrow"');
  } catch (e) {
    r.error = String(e && e.stack || e);
  }
  document.body.setAttribute('data-selftest', JSON.stringify(r));
}
if (new URLSearchParams(location.search).has('selftest')) selfTest();

// QA / debug hook
window.__map = {
  reset() { resetView(); selected.clear(); if (filterPower) setFilter(filterPower); syncFx(); renderList(); renderDetail(); },
  view(v) { Object.assign(view, v || { x: 0, y: 0, w: W, h: H }); clampView(); applyView(); },
  select(ids) { selected.clear(); ids.forEach(i => selected.add(i)); syncFx(); renderList(); renderDetail(); },
  state() { return { ...view, selected: [...selected], filterPower };
  },
};
})();
