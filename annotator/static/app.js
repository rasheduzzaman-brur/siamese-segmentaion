"use strict";
// Siamese annotator UI. All annotation geometry is kept in original-image
// pixels; `view` maps image -> screen and is shared by both canvases so the
// reference (stretched to the inspected image's size) stays aligned.

const PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
                 "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990"];
const HANDLE = 6;           // handle half-size, screen px
const CLOSE_DIST = 9;       // polygon close distance, screen px

const $ = (id) => document.getElementById(id);
const mainCanvas = $("mainCanvas"), refCanvas = $("refCanvas");
const mctx = mainCanvas.getContext("2d"), rctx = refCanvas.getContext("2d");

const state = {
  project: null,
  filter: "all",
  imageId: null,
  meta: null,                 // image record from the server
  img: null, refImg: null,
  anns: [],                   // UI annotations, see makeAnn()
  status: "unlabeled",
  reference: null,
  selected: null,             // ann object
  tool: "box",
  activeClass: 1,
  brush: 18,
  view: { scale: 1, ox: 0, oy: 0 },
  drawing: null,              // in-progress gesture
  hover: null,                // last pointer position in image coords
  spaceDown: false,
  dirty: false,
  undo: [], redo: [],
  saveTimer: null,
};
let uidCounter = 1;

// ---------------------------------------------------------------- helpers
const colorOf = (cid) => PALETTE[(cid - 1) % PALETTE.length];
const className = (cid) => state.project.classes[cid - 1] ?? `class ${cid}`;
const W = () => state.meta.width, H = () => state.meta.height;
const status = (msg) => { $("statusbar").textContent = msg; };

function toImage(canvas, ev) {
  const r = canvas.getBoundingClientRect();
  const { scale, ox, oy } = state.view;
  return { x: (ev.clientX - r.left - ox) / scale, y: (ev.clientY - r.top - oy) / scale };
}
const screenDist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y) * state.view.scale;

async function api(method, url, body) {
  const res = await fetch(url, {
    method, headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch (_) { /* not json */ }
    throw new Error(detail);
  }
  return res.json();
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error(`could not load ${src}`));
    im.src = src;
  });
}
const fileUrl = (kind, rel) => `/files/${kind}/${rel.split("/").map(encodeURIComponent).join("/")}`;

// ------------------------------------------------------------ annotations
function newMaskCanvas() {
  const c = document.createElement("canvas");
  c.width = W(); c.height = H();
  return c;
}
function maskCtx(ann) { return ann.mask.getContext("2d", { willReadFrequently: true }); }

function makeAnn(fields) {
  return { uid: uidCounter++, category_id: state.activeClass, shape: "box", source: "manual",
           score: null, bbox: null, polys: null, mask: null, visible: true, ...fields };
}

function fillPolysInto(ctx, polys, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  for (const p of polys) {
    ctx.moveTo(p[0], p[1]);
    for (let i = 2; i < p.length; i += 2) ctx.lineTo(p[i], p[i + 1]);
    ctx.closePath();
  }
  ctx.fill();
}

function fromServer(a) {
  if (a.shape === "mask") {
    const ann = makeAnn({ category_id: a.category_id, shape: "mask", source: a.source });
    ann.mask = newMaskCanvas();
    fillPolysInto(maskCtx(ann), a.segmentation, colorOf(a.category_id));
    ann._bounds = a.bbox.slice();
    return ann;
  }
  if (a.shape === "polygon" || (a.segmentation && a.segmentation.length)) {
    return makeAnn({ category_id: a.category_id, shape: "polygon", source: a.source,
                     polys: a.segmentation.map((p) => p.slice()) });
  }
  return makeAnn({ category_id: a.category_id, shape: "box", source: a.source ?? "manual",
                   score: a.score ?? null, bbox: a.bbox.slice() });
}

function toServer(ann) {
  const out = { category_id: ann.category_id, shape: ann.shape, source: ann.source };
  if (ann.score != null) out.score = ann.score;
  if (ann.shape === "box") out.bbox = ann.bbox;
  else if (ann.shape === "polygon") out.segmentation = ann.polys;
  else out.mask_png = ann.mask.toDataURL("image/png");
  return out;
}

function annBounds(ann) {
  if (ann.shape === "box") return ann.bbox;
  if (ann.shape === "polygon") {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const p of ann.polys) for (let i = 0; i < p.length; i += 2) {
      x0 = Math.min(x0, p[i]); x1 = Math.max(x1, p[i]);
      y0 = Math.min(y0, p[i + 1]); y1 = Math.max(y1, p[i + 1]);
    }
    return [x0, y0, x1 - x0, y1 - y0];
  }
  return maskBounds(ann);
}

function maskBounds(ann) {
  const { width: w, height: h } = ann.mask;
  const d = maskCtx(ann).getImageData(0, 0, w, h).data;
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) {
    const row = y * w * 4;
    for (let x = 0; x < w; x++) {
      if (d[row + x * 4 + 3]) {
        if (x < x0) x0 = x; if (x > x1) x1 = x;
        if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
    }
  }
  return x1 < 0 ? null : [x0, y0, x1 - x0 + 1, y1 - y0 + 1];
}

function recolorMask(ann) {
  const ctx = maskCtx(ann);
  ctx.save();
  ctx.globalCompositeOperation = "source-in";
  ctx.fillStyle = colorOf(ann.category_id);
  ctx.fillRect(0, 0, ann.mask.width, ann.mask.height);
  ctx.restore();
}

function pointInPoly(pt, p) {
  let inside = false;
  for (let i = 0, j = p.length - 2; i < p.length; j = i, i += 2) {
    const xi = p[i], yi = p[i + 1], xj = p[j], yj = p[j + 1];
    if ((yi > pt.y) !== (yj > pt.y) && pt.x < ((xj - xi) * (pt.y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function hitTest(pt) {
  for (let i = state.anns.length - 1; i >= 0; i--) {
    const a = state.anns[i];
    if (!a.visible) continue;
    if (a.shape === "box") {
      const [x, y, w, h] = a.bbox;
      if (pt.x >= x && pt.x <= x + w && pt.y >= y && pt.y <= y + h) return a;
    } else if (a.shape === "polygon") {
      if (a.polys.some((p) => pointInPoly(pt, p))) return a;
    } else {
      const x = Math.floor(pt.x), y = Math.floor(pt.y);
      if (x >= 0 && y >= 0 && x < W() && y < H() && maskCtx(a).getImageData(x, y, 1, 1).data[3]) return a;
    }
  }
  return null;
}

// handles of the selected annotation: [{x, y, kind, ...}]
function handlesOf(ann) {
  if (!ann || !ann.visible) return [];
  if (ann.shape === "box") {
    const [x, y, w, h] = ann.bbox;
    return [{ x, y, corner: "tl" }, { x: x + w, y, corner: "tr" },
            { x, y: y + h, corner: "bl" }, { x: x + w, y: y + h, corner: "br" }];
  }
  if (ann.shape === "polygon") {
    const hs = [];
    ann.polys.forEach((p, pi) => { for (let i = 0; i < p.length; i += 2) hs.push({ x: p[i], y: p[i + 1], pi, vi: i }); });
    return hs;
  }
  return [];
}

// ---------------------------------------------------------------- undo
function snapshot() {
  return state.anns.map((a) => ({ ...a, bbox: a.bbox && a.bbox.slice(),
    polys: a.polys && a.polys.map((p) => p.slice()),
    mask: a.mask ? a.mask.toDataURL() : null }));
}
async function restore(snap) {
  const anns = [];
  for (const s of snap) {
    const a = { ...s };
    if (s.mask) {
      a.mask = newMaskCanvas();
      maskCtx(a).drawImage(await loadImage(s.mask), 0, 0);
      a._bounds = maskBounds(a);
    }
    anns.push(a);
  }
  state.anns = anns;
  state.selected = anns.find((a) => a.uid === state.selected?.uid) ?? null;
}
function pushUndo() {
  state.undo.push(snapshot());
  if (state.undo.length > 60) state.undo.shift();
  state.redo = [];
}
async function undo() {
  if (!state.undo.length) return;
  state.redo.push(snapshot());
  await restore(state.undo.pop());
  changed();
}
async function redo() {
  if (!state.redo.length) return;
  state.undo.push(snapshot());
  await restore(state.redo.pop());
  changed();
}

// call after any mutation of the current image
function changed() {
  state.dirty = true;
  if (state.anns.length && state.status !== "labeled") state.status = "labeled";
  if (!state.anns.length && state.status === "labeled") state.status = "unlabeled";
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(() => save().catch((e) => status(`save failed: ${e.message}`)), 1500);
  renderPanels();
  draw();
}

// ---------------------------------------------------------------- server I/O
async function loadProject() {
  state.project = await api("GET", "/api/project");
  $("btnAssist").disabled = !state.project.assist_available;
  if (!state.project.assist_available) $("btnAssist").title = "Start with --weights CKPT to enable";
  const sel = $("refSelect");
  sel.innerHTML = "";
  for (const r of state.project.references) sel.add(new Option(r, r));
  renderClasses();
  renderImageList();
}

async function openImage(id) {
  if (state.imageId != null && state.dirty) await save();
  const { image, annotations } = await api("GET", `/api/image/${id}`);
  state.imageId = id;
  state.meta = image;
  state.status = image.status;
  state.reference = image.reference_image;
  state.undo = []; state.redo = [];
  state.selected = null; state.drawing = null;
  [state.img, state.refImg] = await Promise.all([
    loadImage(fileUrl("images", image.file_name)),
    image.reference_image ? loadImage(fileUrl("references", image.reference_image)).catch(() => null) : null,
  ]);
  state.anns = annotations.map(fromServer);
  state.dirty = false;
  $("refSelect").value = image.reference_image ?? "";
  $("imgName").textContent = image.file_name;
  fit();
  renderImageList();
  renderPanels();
  draw();
  status(`${image.file_name}  ${image.width}×${image.height}`);
}

async function save() {
  clearTimeout(state.saveTimer);
  if (state.imageId == null || !state.dirty) return;
  const id = state.imageId;
  state.dirty = false;
  try {
    const res = await api("PUT", `/api/image/${id}`, {
      status: state.status, reference_image: state.reference, annotations: state.anns.map(toServer),
    });
    const rec = state.project.images.find((i) => i.id === id);
    Object.assign(rec, { status: res.image.status, reference_image: res.image.reference_image,
                         n_annotations: res.annotations.length });
    if (id === state.imageId) state.status = res.image.status;
    renderImageList();
    renderPanels();
    status(`saved ${rec.file_name}`);
  } catch (e) {
    state.dirty = true;
    throw e;
  }
}

function visibleImages() {
  return state.project.images.filter((i) => state.filter === "all" || i.status === state.filter);
}
function step(delta) {
  const list = visibleImages();
  if (!list.length) return;
  const idx = list.findIndex((i) => i.id === state.imageId);
  const next = list[Math.min(list.length - 1, Math.max(0, idx + delta))];
  if (next && next.id !== state.imageId) openImage(next.id).catch((e) => status(e.message));
}

// ---------------------------------------------------------------- rendering
function renderClasses() {
  const ul = $("classList");
  ul.innerHTML = "";
  state.project.classes.forEach((name, i) => {
    const li = document.createElement("li");
    li.className = i + 1 === state.activeClass ? "active" : "";
    li.innerHTML = `<span class="swatch" style="background:${colorOf(i + 1)}"></span><span>${i + 1}. ${name}</span>`;
    li.onclick = () => setClass(i + 1);
    ul.appendChild(li);
  });
}

function renderImageList() {
  const ul = $("imageList");
  ul.innerHTML = "";
  for (const im of visibleImages()) {
    const li = document.createElement("li");
    li.className = im.id === state.imageId ? "current" : "";
    li.title = im.file_name;
    li.innerHTML = `<span class="dot ${im.status}"></span><span class="name"></span><span class="muted">${im.n_annotations || ""}</span>`;
    li.querySelector(".name").textContent = im.file_name.split("/").pop();
    li.onclick = () => openImage(im.id).catch((e) => status(e.message));
    ul.appendChild(li);
  }
  const all = state.project.images;
  const done = all.filter((i) => i.status !== "unlabeled").length;
  $("progress").textContent = `${done} / ${all.length} done`;
  ul.querySelector(".current")?.scrollIntoView({ block: "nearest" });
}

function renderPanels() {
  const badge = $("imgStatus");
  badge.textContent = state.status;
  badge.className = `badge ${state.status}`;
  $("btnGood").classList.toggle("active", state.status === "good");
  $("btnUndo").disabled = !state.undo.length;
  $("btnRedo").disabled = !state.redo.length;
  for (const b of document.querySelectorAll("#tools button")) b.classList.toggle("active", b.dataset.tool === state.tool);

  const ul = $("annList");
  ul.innerHTML = "";
  $("annCount").textContent = state.anns.length ? `(${state.anns.length})` : "";
  state.anns.forEach((a, i) => {
    const li = document.createElement("li");
    li.className = [a === state.selected ? "selected" : "", a.source === "model" ? "model" : "",
                    a.visible ? "" : "hidden"].join(" ");
    const kind = { box: "box", polygon: "polygon", mask: "mask" }[a.shape];
    const extra = a.source === "model" ? ` · model ${(a.score ?? 0).toFixed(2)}` : "";
    li.innerHTML = `<span class="swatch" style="background:${colorOf(a.category_id)}"></span>
      <span class="meta">${i + 1}. ${className(a.category_id)} · ${kind}${extra}</span>
      ${a.source === "model" ? '<button data-act="accept" title="Accept (K)">✓</button>' : ""}
      <button data-act="vis" title="Hide/show (H)">${a.visible ? "◉" : "○"}</button>
      <button data-act="del" title="Delete (Del)">✕</button>`;
    li.onclick = (ev) => {
      const act = ev.target.dataset?.act;
      if (act === "del") { pushUndo(); removeAnn(a); changed(); return; }
      if (act === "vis") { a.visible = !a.visible; renderPanels(); draw(); return; }
      if (act === "accept") { pushUndo(); a.source = "manual"; changed(); return; }
      state.selected = a; renderPanels(); draw();
    };
    ul.appendChild(li);
  });
}

function resizeCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  return dpr;
}

function setView(ctx, dpr) {
  const { scale, ox, oy } = state.view;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.setTransform(dpr * scale, 0, 0, dpr * scale, dpr * ox, dpr * oy);
  ctx.imageSmoothingEnabled = scale < 2;
}

function drawAnn(ctx, a, { dimmed = false } = {}) {
  if (!a.visible) return;
  const s = state.view.scale, color = colorOf(a.category_id), sel = a === state.selected;
  const drag = state.drawing?.kind === "move" && state.drawing.ann === a ? state.drawing : null;
  const dx = drag ? drag.dx : 0, dy = drag ? drag.dy : 0;
  ctx.save();
  ctx.translate(dx, dy);
  ctx.lineWidth = (sel ? 2.5 : 1.5) / s;
  ctx.strokeStyle = color;
  if (a.source === "model") ctx.setLineDash([6 / s, 4 / s]);
  if (a.shape === "box") {
    const [x, y, w, h] = a.bbox;
    ctx.globalAlpha = dimmed ? 0.12 : sel ? 0.22 : 0.1;
    ctx.fillStyle = color; ctx.fillRect(x, y, w, h);
    ctx.globalAlpha = dimmed ? 0.6 : 1;
    ctx.strokeRect(x, y, w, h);
  } else if (a.shape === "polygon") {
    ctx.globalAlpha = dimmed ? 0.2 : sel ? 0.4 : 0.28;
    fillPolysInto(ctx, a.polys, color);
    ctx.globalAlpha = dimmed ? 0.6 : 1;
    ctx.beginPath();
    for (const p of a.polys) {
      ctx.moveTo(p[0], p[1]);
      for (let i = 2; i < p.length; i += 2) ctx.lineTo(p[i], p[i + 1]);
      ctx.closePath();
    }
    ctx.stroke();
  } else {
    ctx.globalAlpha = dimmed ? 0.3 : sel ? 0.6 : 0.45;
    ctx.drawImage(a.mask, 0, 0);
  }
  ctx.restore();

  // class tag
  const b = a.shape === "mask" ? (a._bounds ?? null) : annBounds(a);
  if (b && !dimmed) {
    const label = className(a.category_id) + (a.source === "model" ? ` ${(a.score ?? 0).toFixed(2)}` : "");
    ctx.save();
    ctx.font = `${12 / s}px system-ui, sans-serif`;
    const tw = ctx.measureText(label).width + 8 / s, th = 16 / s;
    const tx = b[0] + dx, ty = Math.max(0, b[1] + dy - th);
    ctx.fillStyle = color; ctx.fillRect(tx, ty, tw, th);
    ctx.fillStyle = "#fff"; ctx.fillText(label, tx + 4 / s, ty + 12 / s);
    ctx.restore();
  }
}

function draw() {
  if (!state.img) return;
  const s = state.view.scale;

  // inspected
  const dpr = resizeCanvas(mainCanvas);
  setView(mctx, dpr);
  mctx.drawImage(state.img, 0, 0);
  for (const a of state.anns) drawAnn(mctx, a);

  // selection handles + mask bbox
  const sel = state.selected;
  if (sel && sel.visible) {
    mctx.save();
    mctx.lineWidth = 1 / s;
    if (sel.shape === "mask" && sel._bounds) {
      mctx.setLineDash([4 / s, 3 / s]); mctx.strokeStyle = colorOf(sel.category_id);
      const [x, y, w, h] = sel._bounds; mctx.strokeRect(x, y, w, h);
      mctx.setLineDash([]);
    }
    for (const h of handlesOf(sel)) {
      mctx.fillStyle = "#fff"; mctx.strokeStyle = "#000";
      mctx.fillRect(h.x - HANDLE / s, h.y - HANDLE / s, 2 * HANDLE / s, 2 * HANDLE / s);
      mctx.strokeRect(h.x - HANDLE / s, h.y - HANDLE / s, 2 * HANDLE / s, 2 * HANDLE / s);
    }
    mctx.restore();
  }

  // in-progress gestures
  const d = state.drawing;
  mctx.save();
  mctx.strokeStyle = colorOf(state.activeClass);
  mctx.lineWidth = 1.5 / s;
  if (d?.kind === "box") {
    const x = Math.min(d.x0, d.x1), y = Math.min(d.y0, d.y1);
    mctx.strokeRect(x, y, Math.abs(d.x1 - d.x0), Math.abs(d.y1 - d.y0));
  } else if (d?.kind === "polygon") {
    mctx.beginPath();
    mctx.moveTo(d.pts[0], d.pts[1]);
    for (let i = 2; i < d.pts.length; i += 2) mctx.lineTo(d.pts[i], d.pts[i + 1]);
    if (state.hover) mctx.lineTo(state.hover.x, state.hover.y);
    mctx.stroke();
    mctx.fillStyle = "#fff";
    for (let i = 0; i < d.pts.length; i += 2) mctx.fillRect(d.pts[i] - 3 / s, d.pts[i + 1] - 3 / s, 6 / s, 6 / s);
  }
  if ((state.tool === "brush" || state.tool === "eraser") && state.hover) {
    mctx.beginPath();
    mctx.setLineDash(state.tool === "eraser" ? [3 / s, 3 / s] : []);
    mctx.strokeStyle = state.tool === "eraser" ? "#fff" : colorOf(state.activeClass);
    mctx.arc(state.hover.x, state.hover.y, state.brush / 2 / s, 0, Math.PI * 2);
    mctx.stroke();
  }
  mctx.restore();

  // reference
  const rdpr = resizeCanvas(refCanvas);
  setView(rctx, rdpr);
  if (state.refImg) rctx.drawImage(state.refImg, 0, 0, W(), H());
  else {
    rctx.save(); rctx.setTransform(rdpr, 0, 0, rdpr, 0, 0);
    rctx.fillStyle = "#888"; rctx.font = "14px system-ui"; rctx.fillText("no reference image", 16, 28);
    rctx.restore();
  }
  if ($("showOnRef").checked) for (const a of state.anns) drawAnn(rctx, a, { dimmed: true });
  if (state.hover) {
    // crosshair at the same location, to compare reference vs inspected
    rctx.save(); rctx.strokeStyle = "#ff0"; rctx.lineWidth = 1 / s;
    const { x, y } = state.hover, r = 10 / s;
    rctx.beginPath(); rctx.moveTo(x - r, y); rctx.lineTo(x + r, y); rctx.moveTo(x, y - r); rctx.lineTo(x, y + r);
    rctx.stroke(); rctx.restore();
  }
}

function fit() {
  if (!state.meta) return;
  const r = mainCanvas.getBoundingClientRect();
  const scale = Math.min(r.width / W(), r.height / H()) * 0.96;
  state.view = { scale, ox: (r.width - W() * scale) / 2, oy: (r.height - H() * scale) / 2 };
  draw();
}

// ---------------------------------------------------------------- actions
function setTool(t) {
  if (state.drawing?.kind === "polygon") finishPolygon();
  state.tool = t;
  state.drawing = null;
  mainCanvas.style.cursor = t === "select" ? "default" : "crosshair";
  renderPanels(); draw();
}

function setClass(cid) {
  if (cid < 1 || cid > state.project.classes.length) return;
  state.activeClass = cid;
  if (state.selected && state.selected.category_id !== cid) {
    pushUndo();
    state.selected.category_id = cid;
    if (state.selected.shape === "mask") recolorMask(state.selected);
    changed();
  }
  renderClasses(); draw();
}

function removeAnn(a) {
  state.anns = state.anns.filter((x) => x !== a);
  if (state.selected === a) state.selected = null;
}

function finishPolygon() {
  const d = state.drawing;
  state.drawing = null;
  if (!d || d.pts.length < 6) { draw(); return; }
  pushUndo();
  const ann = makeAnn({ shape: "polygon", polys: [clampPts(d.pts)] });
  state.anns.push(ann);
  state.selected = ann;
  changed();
}

function clampPts(pts) {
  return pts.map((v, i) => Math.min(i % 2 ? H() : W(), Math.max(0, v)));
}

async function toggleGood() {
  if (state.status === "good") { state.status = "unlabeled"; changed(); return; }
  if (state.anns.length && !confirm(`Remove all ${state.anns.length} annotations and mark this image as good (no defects)?`)) return;
  pushUndo();
  state.anns = []; state.selected = null;
  state.status = "good";
  changed();
  await save();
  step(1);
}

async function assist() {
  if (!state.project.assist_available || state.imageId == null) return;
  status("running model…");
  try {
    const { proposals } = await api("POST", `/api/assist/${state.imageId}`, {});
    pushUndo();
    // replace previous unreviewed proposals
    state.anns = state.anns.filter((a) => a.source !== "model");
    for (const p of proposals) state.anns.push(fromServer(p));
    changed();
    status(`${proposals.length} model proposal(s) — accept with K / Shift+K, or delete`);
  } catch (e) { status(`assist failed: ${e.message}`); }
}

function acceptAll() {
  if (!state.anns.some((a) => a.source === "model")) return;
  pushUndo();
  for (const a of state.anns) a.source = "manual";
  changed();
}

// ---------------------------------------------------------------- pointer
function paintStroke(ann, from, to, erase) {
  const ctx = maskCtx(ann);
  ctx.save();
  ctx.globalCompositeOperation = erase ? "destination-out" : "source-over";
  ctx.strokeStyle = ctx.fillStyle = colorOf(ann.category_id);
  ctx.lineWidth = state.brush / state.view.scale;
  ctx.lineCap = ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x + 0.01, to.y);
  ctx.stroke();
  ctx.restore();
}

function handleAt(pt) {
  for (const h of handlesOf(state.selected)) if (screenDist(pt, h) <= HANDLE + 2) return h;
  return null;
}

mainCanvas.addEventListener("pointerdown", (ev) => {
  if (!state.img) return;
  mainCanvas.focus();
  mainCanvas.setPointerCapture(ev.pointerId);
  const pt = toImage(mainCanvas, ev);

  if (ev.button === 1 || (ev.button === 0 && state.spaceDown)) {
    state.drawing = { kind: "pan", sx: ev.clientX, sy: ev.clientY, ox: state.view.ox, oy: state.view.oy };
    return;
  }
  if (ev.button !== 0) return;

  switch (state.tool) {
    case "select": {
      const h = handleAt(pt);
      if (h) {
        pushUndo();
        const a = state.selected, d = { kind: "handle", handle: h, ann: a };
        if (a.shape === "box") {
          const [x, y, w, bh] = a.bbox;
          d.fixed = { x: h.corner[1] === "l" ? x + w : x, y: h.corner[0] === "t" ? y + bh : y };
        }
        state.drawing = d;
        return;
      }
      const hit = hitTest(pt);
      state.selected = hit;
      if (hit) {
        state.activeClass = hit.category_id; renderClasses();
        state.drawing = { kind: "move", ann: hit, start: pt, dx: 0, dy: 0 };
      } else {
        state.drawing = { kind: "pan", sx: ev.clientX, sy: ev.clientY, ox: state.view.ox, oy: state.view.oy };
      }
      renderPanels(); draw();
      return;
    }
    case "box": {
      const x = Math.min(W(), Math.max(0, pt.x)), y = Math.min(H(), Math.max(0, pt.y));
      state.drawing = { kind: "box", x0: x, y0: y, x1: x, y1: y };
      return;
    }
    case "polygon": {
      const d = state.drawing;
      if (!d) { state.drawing = { kind: "polygon", pts: [pt.x, pt.y] }; draw(); return; }
      if (d.pts.length >= 6 && screenDist(pt, { x: d.pts[0], y: d.pts[1] }) <= CLOSE_DIST) { finishPolygon(); return; }
      d.pts.push(pt.x, pt.y); draw();
      return;
    }
    case "brush":
    case "eraser": {
      const erase = state.tool === "eraser";
      let target = state.selected?.shape === "mask" ? state.selected : null;
      if (!target && erase) {
        const hit = hitTest(pt);
        target = hit?.shape === "mask" ? hit : null;
      }
      pushUndo();
      if (!target) {
        if (erase) { state.undo.pop(); status("eraser: select a mask first"); return; }
        target = makeAnn({ shape: "mask" });
        target.mask = newMaskCanvas();
        state.anns.push(target);
      }
      state.selected = target;
      paintStroke(target, pt, pt, erase);
      state.drawing = { kind: "paint", ann: target, last: pt, erase };
      draw();
    }
  }
});

mainCanvas.addEventListener("pointermove", (ev) => {
  if (!state.img) return;
  const pt = toImage(mainCanvas, ev);
  state.hover = pt;
  const d = state.drawing;
  if (d?.kind === "pan") {
    state.view.ox = d.ox + ev.clientX - d.sx;
    state.view.oy = d.oy + ev.clientY - d.sy;
  } else if (d?.kind === "box") {
    d.x1 = Math.min(W(), Math.max(0, pt.x)); d.y1 = Math.min(H(), Math.max(0, pt.y));
  } else if (d?.kind === "paint") {
    paintStroke(d.ann, d.last, pt, d.erase);
    d.last = pt;
  } else if (d?.kind === "move") {
    d.dx = pt.x - d.start.x; d.dy = pt.y - d.start.y;
  } else if (d?.kind === "handle") {
    const a = d.ann, h = d.handle;
    const x = Math.min(W(), Math.max(0, pt.x)), y = Math.min(H(), Math.max(0, pt.y));
    if (a.shape === "box") {
      const f = d.fixed;  // opposite corner stays put
      a.bbox = [Math.min(f.x, x), Math.min(f.y, y), Math.abs(x - f.x), Math.abs(y - f.y)];
    } else {
      a.polys[h.pi][h.vi] = x; a.polys[h.pi][h.vi + 1] = y;
    }
  } else if (state.tool === "select") {
    mainCanvas.style.cursor = handleAt(pt) ? "pointer" : hitTest(pt) ? "move" : "default";
  }
  draw();
});

mainCanvas.addEventListener("pointerup", () => {
  const d = state.drawing;
  if (!d) return;
  if (d.kind === "polygon") return;  // polygon keeps going until closed
  state.drawing = null;
  if (d.kind === "box") {
    const w = Math.abs(d.x1 - d.x0), h = Math.abs(d.y1 - d.y0);
    if (w * state.view.scale >= 3 && h * state.view.scale >= 3) {
      pushUndo();
      const ann = makeAnn({ shape: "box", bbox: [Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), w, h] });
      state.anns.push(ann);
      state.selected = ann;
      changed();
    }
  } else if (d.kind === "paint") {
    d.ann._bounds = maskBounds(d.ann);
    if (!d.ann._bounds) removeAnn(d.ann);   // erased away completely
    changed();
  } else if (d.kind === "move" && (d.dx || d.dy)) {
    pushUndo();
    moveAnn(d.ann, d.dx, d.dy);
    changed();
  } else if (d.kind === "handle") {
    if (d.ann.shape === "box" && (d.ann.bbox[2] < 1 || d.ann.bbox[3] < 1)) removeAnn(d.ann);
    changed();
  }
  draw();
});

mainCanvas.addEventListener("dblclick", () => {
  const d = state.drawing;
  if (d?.kind !== "polygon") return;
  // the two clicks of the double-click each added a point at the same spot
  if (d.pts.length >= 4) d.pts.length -= 2;
  finishPolygon();
});

mainCanvas.addEventListener("pointerleave", () => { state.hover = null; draw(); });

function moveAnn(a, dx, dy) {
  if (a.shape === "box") {
    a.bbox[0] = Math.min(W() - a.bbox[2], Math.max(0, a.bbox[0] + dx));
    a.bbox[1] = Math.min(H() - a.bbox[3], Math.max(0, a.bbox[1] + dy));
  } else if (a.shape === "polygon") {
    a.polys = a.polys.map((p) => clampPts(p.map((v, i) => v + (i % 2 ? dy : dx))));
  } else {
    const copy = newMaskCanvas();
    copy.getContext("2d").drawImage(a.mask, 0, 0);
    const ctx = maskCtx(a);
    ctx.clearRect(0, 0, W(), H());
    ctx.drawImage(copy, Math.round(dx), Math.round(dy));
    a._bounds = maskBounds(a);
  }
}

function zoomAt(canvas, ev, factor) {
  const r = canvas.getBoundingClientRect();
  const sx = ev.clientX - r.left, sy = ev.clientY - r.top;
  const v = state.view;
  const ns = Math.min(40, Math.max(0.05, v.scale * factor));
  v.ox = sx - ((sx - v.ox) * ns) / v.scale;
  v.oy = sy - ((sy - v.oy) * ns) / v.scale;
  v.scale = ns;
  draw();
}
for (const c of [mainCanvas, refCanvas]) {
  c.addEventListener("wheel", (ev) => { ev.preventDefault(); zoomAt(c, ev, Math.exp(-ev.deltaY * 0.0015)); }, { passive: false });
}
// panning on the reference pane too (stays synced)
refCanvas.addEventListener("pointerdown", (ev) => {
  refCanvas.setPointerCapture(ev.pointerId);
  state.drawing = state.drawing?.kind === "polygon" ? state.drawing
    : { kind: "refpan", sx: ev.clientX, sy: ev.clientY, ox: state.view.ox, oy: state.view.oy };
});
refCanvas.addEventListener("pointermove", (ev) => {
  const d = state.drawing;
  if (d?.kind === "refpan") {
    state.view.ox = d.ox + ev.clientX - d.sx; state.view.oy = d.oy + ev.clientY - d.sy;
  }
  state.hover = toImage(refCanvas, ev);
  draw();
});
refCanvas.addEventListener("pointerup", () => { if (state.drawing?.kind === "refpan") state.drawing = null; });

// ---------------------------------------------------------------- keyboard
document.addEventListener("keydown", (ev) => {
  if (ev.target.closest("dialog, input, select")) return;
  const k = ev.key, ctrl = ev.ctrlKey || ev.metaKey;
  if (ctrl && k.toLowerCase() === "z") { ev.preventDefault(); ev.shiftKey ? redo() : undo(); return; }
  if (ctrl && k.toLowerCase() === "y") { ev.preventDefault(); redo(); return; }
  if (ctrl && k.toLowerCase() === "s") { ev.preventDefault(); state.dirty = true; save().catch((e) => status(e.message)); return; }
  if (ctrl) return;

  const d = state.drawing;
  if (d?.kind === "polygon") {
    if (k === "Enter") { finishPolygon(); return; }
    if (k === "Escape") { state.drawing = null; draw(); return; }
    if (k === "Backspace") { d.pts.length -= 2; if (!d.pts.length) state.drawing = null; draw(); ev.preventDefault(); return; }
  }
  if (k === " ") { state.spaceDown = true; ev.preventDefault(); return; }
  if (/^[1-9]$/.test(k)) { setClass(Number(k)); return; }
  switch (k.toLowerCase()) {
    case "v": setTool("select"); break;
    case "b": setTool("box"); break;
    case "p": setTool("polygon"); break;
    case "r": setTool("brush"); break;
    case "e": setTool("eraser"); break;
    case "g": toggleGood(); break;
    case "a": assist(); break;
    case "f": fit(); break;
    case "[": state.brush = Math.max(2, state.brush - 4); $("brushSize").value = state.brush; draw(); break;
    case "]": state.brush = Math.min(120, state.brush + 4); $("brushSize").value = state.brush; draw(); break;
    case "h": if (state.selected) { state.selected.visible = !state.selected.visible; renderPanels(); draw(); } break;
    case "k":
      if (ev.shiftKey) acceptAll();
      else if (state.selected?.source === "model") { pushUndo(); state.selected.source = "manual"; changed(); }
      break;
    case "escape": state.selected = null; state.drawing = null; renderPanels(); draw(); break;
    case "delete":
    case "backspace":
      if (state.selected) { pushUndo(); removeAnn(state.selected); changed(); ev.preventDefault(); }
      break;
    case "arrowright": step(1); ev.preventDefault(); break;
    case "arrowleft": step(-1); ev.preventDefault(); break;
  }
});
document.addEventListener("keyup", (ev) => { if (ev.key === " ") state.spaceDown = false; });

// ---------------------------------------------------------------- controls
for (const b of document.querySelectorAll("#tools button")) b.onclick = () => setTool(b.dataset.tool);
$("brushSize").oninput = (ev) => { state.brush = Number(ev.target.value); draw(); };
$("btnGood").onclick = toggleGood;
$("btnAssist").onclick = assist;
$("btnUndo").onclick = undo;
$("btnRedo").onclick = redo;
$("btnSave").onclick = () => { state.dirty = true; save().catch((e) => status(e.message)); };
$("btnAcceptAll").onclick = acceptAll;
$("btnHelp").onclick = () => $("helpDialog").showModal();
$("showOnRef").onchange = draw;
$("filter").onchange = (ev) => { state.filter = ev.target.value; renderImageList(); };
$("btnRescan").onclick = async () => {
  const { added } = await api("POST", "/api/rescan");
  await loadProject();
  status(`${added} new image(s) found`);
};
$("refSelect").onchange = async (ev) => {
  state.reference = ev.target.value;
  state.refImg = await loadImage(fileUrl("references", state.reference)).catch(() => null);
  state.dirty = true;
  save().catch((e) => status(e.message));
  draw();
};

$("btnExport").onclick = async () => {
  if (state.dirty) await save();
  const form = $("exportDialog").querySelector("form");
  for (const [k, v] of Object.entries(state.project.export_defaults)) {
    const el = form.elements[k];
    if (!el) continue;
    if (el.type === "checkbox") el.checked = !!v; else el.value = v;
  }
  $("exportResult").textContent = "";
  $("exportDialog").showModal();
};
$("btnDoExport").onclick = async (ev) => {
  ev.preventDefault();
  const form = $("exportDialog").querySelector("form");
  const body = {};
  for (const el of form.elements) {
    if (!el.name) continue;
    body[el.name] = el.type === "checkbox" ? el.checked : el.value;
  }
  $("exportResult").textContent = "exporting…";
  try {
    const res = await api("POST", "/api/export", body);
    state.project.export_defaults = { ...state.project.export_defaults, ...body };
    $("exportResult").textContent = JSON.stringify(res, null, 2);
  } catch (e) { $("exportResult").textContent = `failed: ${e.message}`; }
};

new ResizeObserver(() => draw()).observe(mainCanvas);
window.addEventListener("beforeunload", (ev) => { if (state.dirty) { save(); ev.preventDefault(); } });

(async () => {
  try {
    await loadProject();
    setTool("box");
    const first = state.project.images.find((i) => i.status === "unlabeled") ?? state.project.images[0];
    if (first) await openImage(first.id);
    else status("no images found in the images folder");
  } catch (e) { status(`failed to load project: ${e.message}`); }
})();
