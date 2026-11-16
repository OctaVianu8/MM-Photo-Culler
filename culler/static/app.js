'use strict';
/* MM Photo Culler - review UI. No framework, no build step. */

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const S = {
  folder: null, version: -1,
  photos: [], byId: new Map(), moments: new Map(),   // moment id -> photos (shooting order)
  visible: [], cur: -1,                              // filtered list + index into it
  mode: 'grid',                                      // grid | loupe | compare
  zoom: false, zoomRaw: false, pan: { x: 0, y: 0 },
  blind: false, overlays: true, filter: 'all', autoAdvance: false,
  undo: [], views: [], viewKey: '', warned: null,
};

// ---------------------------------------------------------------- api
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

// ---------------------------------------------------------------- model
/** Effective flag: your decision wins; in blind mode the machine has no say. */
function eff(p) {
  if (p.user != null) return p.user === 'none' ? null : p.user;
  return S.blind ? null : p.auto;
}
const isAuto = (p) => p.user == null && p.auto != null && !S.blind;
const cur = () => S.visible[S.cur];
const momentOf = (p) => S.moments.get(p.moment) || [p];

function passes(p) {
  switch (S.filter) {
    case 'pick': return eff(p) === 'pick';
    case 'reject': return eff(p) === 'reject';
    case 'none': return eff(p) == null;
    case 'unreviewed': return p.user == null;
    default: return true;
  }
}

async function loadPhotos() {
  const data = await api('/api/photos');
  const same = data.photos.length === S.photos.length && data.photos.every((p, i) => p.id === S.photos[i].id && p.moment === S.photos[i].moment);
  if (same) {                       // analysis progressing: update in place, keep the DOM
    data.photos.forEach((p, i) => {
      const old = S.photos[i];
      const changed = old.auto !== p.auto || old.score !== p.score || old.user !== p.user || old.rating !== p.rating;
      Object.assign(old, p);
      if (changed) refreshPhoto(old, false);
    });
    updateCounts();
    if (S.mode !== 'grid') renderInfo();
    return;
  }
  const keep = cur()?.id;
  S.photos = data.photos;
  S.byId = new Map(S.photos.map((p) => [p.id, p]));
  S.moments = new Map();
  for (const p of S.photos) { if (!S.moments.has(p.moment)) S.moments.set(p.moment, []); S.moments.get(p.moment).push(p); }
  rebuild(keep);
}

function rebuild(keepId) {
  S.visible = S.photos.filter(passes);
  let i = S.visible.findIndex((p) => p.id === keepId);
  if (i < 0) i = S.visible.length ? 0 : -1;
  S.cur = i;
  renderGrid();
  updateCounts();
  S.viewKey = '';
  show();
}

// ---------------------------------------------------------------- formatting
const two = (n) => String(n).padStart(2, '0');
function fmtTime(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000);     // stored as camera wall-clock time, so read back as UTC
  return `${two(d.getUTCHours())}:${two(d.getUTCMinutes())}:${two(d.getUTCSeconds())}`;
}
const fmtShutter = (s) => !s ? '' : s >= 0.4 ? `${+s.toFixed(1)} s` : `1/${Math.round(1 / s)} s`;
const stars = (n) => n ? '★'.repeat(n) : '';

// ---------------------------------------------------------------- cells (grid + filmstrip)
function makeCell(p) {
  const c = el('div', 'cell');
  c.dataset.id = p.id;
  const img = el('img');
  img.loading = 'lazy'; img.decoding = 'async'; img.src = `/api/thumb/${p.id}`; img.alt = '';
  c.append(img, el('span', 'badge'), el('span', 'stars'), el('span', 'score'));
  paintCell(c, p);
  return c;
}
function paintCell(c, p) {
  const f = eff(p);
  c.classList.toggle('pick', f === 'pick');
  c.classList.toggle('reject', f === 'reject');
  c.classList.toggle('auto', isAuto(p));
  c.classList.toggle('cur', cur() === p);
  c.children[1].textContent = f === 'pick' ? 'P' : f === 'reject' ? 'X' : '';
  c.children[1].hidden = !f;
  c.children[2].textContent = stars(p.rating);
  c.children[3].textContent = p.score == null ? '' : p.score.toFixed(2);
  c.title = p.name;
}

function renderGrid() {
  const grid = $('#grid');
  grid.textContent = '';
  let box = null, last = null, n = 0;
  const frag = document.createDocumentFragment();
  for (const p of S.visible) {
    if (p.moment !== last) {
      last = p.moment; n += 1;
      const all = momentOf(p);
      const m = el('div', 'moment');
      m.append(el('h3', '', `#${p.moment ?? '–'} · ${all.length} frame${all.length === 1 ? '' : 's'} · ${fmtTime(all[0].ts)}`));
      box = el('div');
      m.append(box);
      frag.append(m);
    }
    box.append(makeCell(p));
  }
  grid.append(frag);
  $('#empty').hidden = S.photos.length > 0 || !!S.folder;
}

function refreshPhoto(p, info = true) {
  for (const c of $$(`.cell[data-id="${p.id}"]`)) paintCell(c, p);
  const v = S.views.find((v) => v.p === p);
  if (v) paintView(v);
  if (info && cur() === p && S.mode !== 'grid') renderInfo();
}

function updateCounts() {
  let picks = 0, rejects = 0;
  for (const p of S.photos) { const f = eff(p); picks += f === 'pick'; rejects += f === 'reject'; }
  $('#counts').textContent = S.photos.length
    ? `${picks} picks · ${rejects} rejects · ${S.photos.length} photos · ${S.moments.size} moments` : '';
}

// ---------------------------------------------------------------- navigation
function setCur(i, scroll = true) {
  if (i < 0 || i >= S.visible.length) return;
  const prev = cur();
  S.cur = i;
  const p = cur();
  for (const q of [prev, p]) if (q) for (const c of $$(`.cell[data-id="${q.id}"]`)) c.classList.toggle('cur', q === p);
  if (scroll && S.mode === 'grid') $(`#grid .cell[data-id="${p.id}"]`)?.scrollIntoView({ block: 'nearest' });
  if (S.mode !== 'grid') renderStage();
}
function step(d) { setCur(S.cur + d); }
function stepMoment(d) {
  const p = cur(); if (!p) return;
  let i = S.cur;
  if (d > 0) { while (i < S.visible.length && S.visible[i].moment === p.moment) i++; }
  else {
    while (i >= 0 && S.visible[i].moment === p.moment) i--;          // last frame of previous moment
    if (i >= 0) { const m = S.visible[i].moment; while (i > 0 && S.visible[i - 1].moment === m) i--; }
  }
  if (i >= 0 && i < S.visible.length) setCur(i);
}

function show() {
  const has = S.photos.length > 0;
  $('#grid').hidden = !has || S.mode !== 'grid';
  $('#stage').hidden = !has || S.mode === 'grid';
  if (!has) return;
  if (S.mode === 'grid') {
    S.zoom = false; S.views = []; S.viewKey = '';
    const p = cur();
    if (p) $(`#grid .cell[data-id="${p.id}"]`)?.scrollIntoView({ block: 'center' });
  } else renderStage();
}
function setMode(m) { S.mode = m; show(); }

// ---------------------------------------------------------------- loupe / compare
function stagePhotos() {
  const p = cur();
  if (!p) return [];
  if (S.mode === 'loupe') return [p];
  const all = S.visible.filter((q) => q.moment === p.moment);
  if (all.length <= 12) return all;
  const i = all.indexOf(p), a = Math.max(0, Math.min(all.length - 12, i - 6));   // window around current
  return all.slice(a, a + 12);
}

function renderStage() {
  const list = stagePhotos();
  const key = S.mode + ':' + list.map((p) => p.id).join(',');
  if (key !== S.viewKey) {
    S.viewKey = key;
    const box = $('#views');
    box.textContent = '';
    const n = list.length;
    const cols = n <= 1 ? 1 : n <= 2 ? 2 : n <= 6 ? 3 : 4;
    box.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    box.style.gridTemplateRows = `repeat(${Math.ceil(n / cols)}, 1fr)`;
    S.views = list.map((p) => {
      const d = el('div', 'view');
      d.dataset.id = p.id;
      const img = el('img'); img.alt = ''; img.draggable = false;
      const v = { p, el: d, img, overlay: el('div', 'overlay'), tag: el('div', 'tag'), name: el('div', 'tag br'), fullKey: null };
      d.append(img, v.overlay, v.tag, v.name);
      box.append(d);
      return v;
    });
    renderStrip();
  }
  for (const v of S.views) { paintView(v); layoutView(v); }
  for (const c of $$('#strip .cell')) c.classList.toggle('cur', +c.dataset.id === cur()?.id);
  $(`#strip .cell.cur`)?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  renderInfo();
  preload();
}

function renderStrip() {
  const strip = $('#strip');
  strip.textContent = '';
  const p = cur(); if (!p) return;
  for (const q of S.visible.filter((q) => q.moment === p.moment)) strip.append(makeCell(q));
}

function paintView(v) {
  const p = v.p, f = eff(p);
  v.el.classList.toggle('cur', cur() === p);
  v.el.classList.toggle('pick', f === 'pick');
  v.el.classList.toggle('reject', f === 'reject');
  v.el.classList.toggle('zoom', S.zoom);
  const who = f ? (isAuto(p) ? ' (auto)' : '') : '';
  const rank = (!S.blind && p.rank) ? `<span class="autoinfo muted">#${p.rank} · ${p.score?.toFixed(2) ?? ''}</span>` : '';
  v.tag.innerHTML = (f ? `<b class="${f}">${f === 'pick' ? 'PICK' : 'REJECT'}${who}</b>` : '')
    + (p.rating ? `<span class="stars">${stars(p.rating)}</span>` : '') + rank;
  v.tag.hidden = !v.tag.innerHTML;
  v.name.textContent = p.name.split(/[\\/]/).pop() + (S.zoom ? (S.zoomRaw ? ' · RAW 100%' : ' · 100%') : '');
  // boxes
  v.overlay.textContent = '';
  if (S.overlays && !S.blind && p.subject) {
    const add = (b, cls) => { const i = el('i', cls); i.style.cssText = `left:${b[0] * 100}%;top:${b[1] * 100}%;width:${b[2] * 100}%;height:${b[3] * 100}%`; v.overlay.append(i); };
    for (const fc of p.faces || []) add(fc.box, fc.blink != null && fc.blink > 0.6 ? 'closed' : '');
    if (p.subject.src !== 'frame') add(p.subject.box, 'subject');
  }
}

/** Position the image (and its overlay) by hand: one code path for fit and 100 %. */
function layoutView(v) {
  const p = v.p, vw = v.el.clientWidth, vh = v.el.clientHeight;
  if (!vw || !vh) return;
  let w, h, left, top;
  if (!S.zoom) {
    const s = Math.min(vw / p.w, vh / p.h);
    w = p.w * s; h = p.h * s; left = (vw - w) / 2; top = (vh - h) / 2;
    if (v.fullKey !== null || !v.img.src) { v.fullKey = null; v.img.src = `/api/preview/${p.id}`; }
  } else {
    const key = `${p.id}:${S.zoomRaw ? 1 : 0}`;
    if (v.fullKey !== key) {
      // show the cached preview stretched to 100 % at once, swap in the real pixels when loaded
      v.fullKey = key; v.fullW = p.full_w; v.fullH = p.full_h;
      if (!v.img.src) v.img.src = `/api/preview/${p.id}`;
      const im = new Image();
      im.onload = () => { if (v.fullKey === key && S.zoom) { v.img.src = im.src; v.fullW = im.naturalWidth; v.fullH = im.naturalHeight; layoutView(v); } };
      im.onerror = () => toast('could not load the full-resolution image');
      im.src = `/api/full/${p.id}?raw_decode=${S.zoomRaw ? 1 : 0}`;
    }
    const s = 1 / (window.devicePixelRatio || 1);       // 1 image pixel = 1 screen pixel
    w = v.fullW * s; h = v.fullH * s;
    const b = p.subject && p.subject.src !== 'frame' ? p.subject.box : [0.5, 0.5, 0, 0];
    left = vw / 2 - (b[0] + b[2] / 2) * w + S.pan.x;
    top = vh / 2 - (b[1] + b[3] / 2) * h + S.pan.y;
    left = w <= vw ? (vw - w) / 2 : Math.min(0, Math.max(vw - w, left));
    top = h <= vh ? (vh - h) / 2 : Math.min(0, Math.max(vh - h, top));
  }
  const css = `left:${left}px;top:${top}px;width:${w}px;height:${h}px`;
  v.img.style.cssText = css;
  v.overlay.style.cssText = css;
}

function preload() {
  for (const d of [1, 2, -1]) {
    const p = S.visible[S.cur + d];
    if (p) new Image().src = `/api/preview/${p.id}`;
  }
}

function renderInfo() {
  const p = cur(), box = $('#info');
  if (!p) { box.textContent = ''; return; }
  const f = eff(p), all = momentOf(p), c = p.comp || {};
  const bar = (label, v, pen) => `<div class="bar ${pen ? 'pen' : ''}"><span>${label}</span><span><i style="width:${Math.round((v || 0) * 100)}%"></i></span><span>${(v || 0).toFixed(2)}</span></div>`;
  const expo = [fmtShutter(p.shutter), p.aperture ? `f/${+p.aperture.toFixed(1)}` : '', p.iso ? `ISO ${Math.round(p.iso)}` : '', p.focal ? `${Math.round(p.focal)} mm` : ''].filter(Boolean).join(' · ');
  const srcName = { face: 'face', person: 'person (head & shoulders)', borrowed: 'box borrowed from a sibling frame', frame: 'whole frame (no subject found)' };
  let h = `<h4>${esc(p.name)}</h4><div class="muted">${fmtTime(p.ts)}${p.camera ? ' · ' + esc(p.camera) : ''}</div><div>${expo}</div>`
    + `<div class="muted">moment #${p.moment} · frame ${all.indexOf(p) + 1} of ${all.length}</div><hr>`
    + `<div>${f ? `<span class="flag ${f}">${f === 'pick' ? 'Pick' : 'Reject'}</span> <span class="muted">${isAuto(p) ? 'suggested' : 'your decision'}</span>` : '<span class="muted">no flag</span>'}`
    + `${p.rating ? ` <span style="color:var(--accent)">${stars(p.rating)}</span>` : ''}</div>`;
  if (!S.blind) {
    h += '<div class="autoinfo"><hr>';
    if (p.score != null) {
      h += `<div><b>score ${p.score.toFixed(2)}</b> <span class="muted">· #${p.rank} of ${all.length}</span></div>`
        + bar('sharpness', c.sharp) + bar('eyes', c.eyes, 1) + bar('pose', c.pose, 1) + bar('motion', c.motion, 1) + bar('exposure', c.exposure, 1)
        + (p.reasons.length ? `<ul>${p.reasons.map((r) => `<li>${esc(r)}</li>`).join('')}</ul>` : '')
        + `<div class="muted">measured on: ${srcName[c.src] || c.src || '?'}</div>`;
    } else h += `<div class="muted">${p.analysed ? 'not scored' : 'analysis pending…'}</div>`;
    h += '</div>';
  }
  const small = Math.max(p.full_w || 0, p.full_h || 0) < 2400;
  h += `<hr><div class="muted">${{ embedded: 'embedded preview', companion: 'paired JPEG', file: 'JPEG file', raw_decode: 'decoded from RAW (no embedded preview)' }[p.src] || ''} ${p.full_w}×${p.full_h}`
    + `${small && p.kind === 'raw' ? '<br>Small preview: use Shift+Z for real RAW pixels.' : ''}</div>`;
  if (S.autoAdvance) h += '<div class="muted">auto-advance on (T)</div>';
  box.innerHTML = h;
}

// ---------------------------------------------------------------- decisions
async function applyItems(items) {
  for (const it of items) {
    const p = S.byId.get(it.id); if (!p) continue;
    if ('flag' in it) p.user = it.flag;
    if ('rating' in it) p.rating = it.rating;
    refreshPhoto(p);
  }
  updateCounts();
  try { await api('/api/decisions', { items }); } catch (e) { toast('could not save: ' + e.message); }
}
function decide(items) {
  if (!items.length) return;
  S.undo.push(items.map((it) => { const p = S.byId.get(it.id); return { id: p.id, flag: p.user, rating: p.rating }; }));
  if (S.undo.length > 200) S.undo.shift();
  return applyItems(items);
}
function flag(f) {
  const p = cur(); if (!p) return;
  decide([{ id: p.id, flag: f }]);
  if (S.autoAdvance && f !== 'none') step(1);
}
function rate(n) { const p = cur(); if (p) decide([{ id: p.id, rating: n || null }]); }
function pickOnly() {
  const p = cur(); if (!p) return;
  decide(momentOf(p).filter((q) => q === p || q.user !== 'pick').map((q) => ({ id: q.id, flag: q === p ? 'pick' : 'reject' })));
  toast('picked this frame, rejected the rest of the moment');
}
function acceptAuto() {
  const p = cur(); if (!p || S.blind) return;
  decide(momentOf(p).filter((q) => q.user == null).map((q) => ({ id: q.id, flag: q.auto || 'none' })));
  stepMoment(1);
}
function undo() {
  const items = S.undo.pop();
  if (!items) return toast('nothing to undo');
  applyItems(items);
}

// ---------------------------------------------------------------- toggles
function toggleZoom(raw) {
  if (S.mode === 'grid') S.mode = 'loupe';
  if (S.zoom && raw === S.zoomRaw) S.zoom = false;
  else { S.zoom = true; S.zoomRaw = raw; S.pan = { x: 0, y: 0 }; if (raw) toast('decoding RAW…'); }
  show();
}
function setBlind(on) {
  S.blind = on;
  document.body.classList.toggle('blind', on);
  $('#btn-blind').classList.toggle('on', on);
  rebuild(cur()?.id);
  toast(on ? 'Blind mode: automatic suggestions hidden' : 'Blind mode off');
}
function setFilter(f) { S.filter = f; $('#filter').value = f; rebuild(cur()?.id); }

let toastTimer;
function toast(msg) {
  const t = $('#toast');
  t.textContent = msg; t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 2200);
}

// ---------------------------------------------------------------- keyboard & mouse
document.addEventListener('keydown', (e) => {
  if (e.target.closest('input, select, textarea') || $('dialog[open]')) return;
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if ((e.ctrlKey || e.metaKey) && k === 'z') { e.preventDefault(); return undo(); }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const map = {
    ArrowRight: () => step(1), ArrowLeft: () => step(-1),
    ArrowDown: () => stepMoment(1), ArrowUp: () => stepMoment(-1),
    p: () => (e.shiftKey ? pickOnly() : flag('pick')), x: () => flag('reject'), u: () => flag('none'),
    a: acceptAuto,
    c: () => setMode(S.mode === 'compare' ? 'loupe' : 'compare'),
    z: () => toggleZoom(e.shiftKey),
    g: () => setMode('grid'), e: () => setMode('loupe'),
    Enter: () => setMode(S.mode === 'grid' ? 'loupe' : 'grid'),
    ' ': () => setMode(S.mode === 'grid' ? 'loupe' : 'grid'),
    Escape: () => { if (S.zoom) { S.zoom = false; show(); } else setMode('grid'); },
    i: () => { S.overlays = !S.overlays; S.views.forEach(paintView); },
    b: () => setBlind(!S.blind),
    t: () => { S.autoAdvance = !S.autoAdvance; toast('auto-advance ' + (S.autoAdvance ? 'on' : 'off')); renderInfo(); },
    f: () => { const o = ['all', 'pick', 'reject', 'none', 'unreviewed']; setFilter(o[(o.indexOf(S.filter) + 1) % o.length]); toast('showing: ' + $('#filter').selectedOptions[0].text); },
    o: openDialog, '?': () => $('#dlg-help').showModal(),
  };
  if (k >= '0' && k <= '5' && k.length === 1) { e.preventDefault(); return rate(+k); }
  if (map[k]) { e.preventDefault(); map[k](); }
});

function clickCell(e, dbl) {
  const c = e.target.closest('.cell'); if (!c) return;
  const i = S.visible.findIndex((p) => p.id === +c.dataset.id);
  if (i >= 0) setCur(i, false);
  if (dbl) setMode('loupe');
}
$('#grid').addEventListener('click', (e) => clickCell(e, false));
$('#grid').addEventListener('dblclick', (e) => clickCell(e, true));
$('#strip').addEventListener('click', (e) => clickCell(e, false));
$('#views').addEventListener('click', (e) => {
  const d = e.target.closest('.view'); if (!d || S.dragged) return;
  const i = S.visible.findIndex((p) => p.id === +d.dataset.id);
  if (i >= 0 && i !== S.cur) setCur(i);
});
$('#views').addEventListener('dblclick', () => toggleZoom(false));
// drag to pan at 100 % (all compared frames move together)
$('#views').addEventListener('pointerdown', (e) => {
  if (!S.zoom) return;
  S.drag = { x: e.clientX, y: e.clientY }; S.dragged = false;
  e.currentTarget.setPointerCapture(e.pointerId);
});
$('#views').addEventListener('pointermove', (e) => {
  if (!S.drag) return;
  S.pan.x += e.clientX - S.drag.x; S.pan.y += e.clientY - S.drag.y;
  S.drag = { x: e.clientX, y: e.clientY }; S.dragged = true;
  S.views.forEach(layoutView);
});
$('#views').addEventListener('pointerup', () => { S.drag = null; setTimeout(() => { S.dragged = false; }, 0); });
window.addEventListener('resize', () => S.views.forEach(layoutView));

$('#filter').addEventListener('change', (e) => { setFilter(e.target.value); e.target.blur(); });
$('#btn-blind').addEventListener('click', () => setBlind(!S.blind));
$('#btn-help').addEventListener('click', () => $('#dlg-help').showModal());
for (const b of $$('dialog .close')) b.addEventListener('click', () => b.closest('dialog').close());

// ---------------------------------------------------------------- open folder
async function browse(path) {
  try {
    const d = await api('/api/browse?path=' + encodeURIComponent(path || ''));
    $('#open-path').value = d.path;
    $('#open-info').textContent = d.photos ? `${d.photos} photo files directly in this folder` : 'no photo files directly in this folder (sub-folders are scanned too)';
    const ul = $('#open-dirs'); ul.textContent = '';
    if (d.parent) { const li = el('li', '', '⬑ ..'); li.onclick = () => browse(d.parent); ul.append(li); }
    for (const name of d.dirs) { const li = el('li'); li.textContent = name; li.onclick = () => browse(d.path.replace(/[\\/]$/, '') + '/' + name); ul.append(li); }
    const rc = $('#open-recent'); rc.textContent = '';
    if (d.recent.length) { rc.append(el('div', 'muted', 'Recent')); for (const r of d.recent) { const a = el('a'); a.textContent = r; a.onclick = () => openFolder(r); rc.append(a); } }
  } catch (e) { $('#open-info').textContent = e.message; }
}
function openDialog() { $('#dlg-open').showModal(); browse(S.folder || ''); }
async function openFolder(path) {
  try {
    await api('/api/open', { folder: path });
    $('#dlg-open').close();
    S.photos = []; S.byId.clear(); S.moments.clear(); S.version = -1; S.undo = []; S.mode = 'grid'; S.warned = null;
    rebuild();
    poll(true);
  } catch (e) { $('#open-info').textContent = 'Cannot open: ' + e.message; }
}
$('#btn-open').addEventListener('click', openDialog);
$('#btn-open2').addEventListener('click', openDialog);
$('#open-go').addEventListener('click', () => openFolder($('#open-path').value));
$('#open-path').addEventListener('keydown', (e) => { if (e.key === 'Enter') browse(e.target.value); });
$('#btn-cancel').addEventListener('click', () => api('/api/cancel', {}));

// ---------------------------------------------------------------- export
function exportOptions(dry) {
  return { dry_run: dry, pick_label: $('#ex-pick').value.trim(), reject_label: $('#ex-reject').value.trim(),
    min_pick_rating: +$('#ex-minrating').value, include_auto: $('#ex-auto').checked,
    keepers_dir: $('#ex-keepers').value.trim() || null, keepers_mode: $('#ex-mode').value };
}
async function runExport(dry) {
  const o = exportOptions(dry);
  if (!dry && o.keepers_dir && o.keepers_mode === 'move'
    && !confirm('Move the picked originals out of this folder into\n' + o.keepers_dir + ' ?')) return;
  $('#ex-report').textContent = 'working…';
  try { $('#ex-report').textContent = (await api('/api/export', o)).text; }
  catch (e) { $('#ex-report').textContent = 'Error: ' + e.message; }
}
$('#btn-export').addEventListener('click', () => { if (S.folder) { $('#ex-report').textContent = ''; $('#dlg-export').showModal(); } });
$('#ex-preview').addEventListener('click', () => runExport(true));
$('#ex-run').addEventListener('click', () => runExport(false));

// ---------------------------------------------------------------- status polling
let pollTimer;
async function poll(now) {
  clearTimeout(pollTimer);
  let running = false;
  try {
    const st = await api('/api/status');
    const pr = st.progress;
    running = pr.running;
    $('#folder').textContent = st.folder || 'no folder';
    $('#folder').title = st.folder || '';
    $('#progress').hidden = !running;
    if (running) {
      $('#pbar').removeAttribute('value');
      if (pr.total) { $('#pbar').max = pr.total; $('#pbar').value = pr.done; }
      $('#ptext').textContent = `${pr.message || pr.stage}${pr.total ? ` ${pr.done}/${pr.total}` : ''}${pr.eta_s ? ` · ~${Math.ceil(pr.eta_s)} s left` : ''}`;
    }
    if (pr.error) toast('Processing failed: ' + pr.error);
    if (st.folder !== S.folder || st.version !== S.version) {
      S.folder = st.folder; S.version = st.version;
      if (st.folder) await loadPhotos();
    }
    const caps = st.capabilities;
    if (caps && !running && S.warned !== st.folder) {
      S.warned = st.folder;
      const off = [!caps.faces && 'face detection', !caps.eyes && 'eye / pose detection', !caps.persons && 'person detection'].filter(Boolean);
      if (off.length) toast(`Running without ${off.join(', ')} - see README (models / mediapipe)`);
      else if (st.errors?.length) toast(`${st.errors.length} file(s) could not be read`);
    }
  } catch (e) { /* server restarting */ }
  pollTimer = setTimeout(poll, running ? 600 : 2500);
}
poll(true);
