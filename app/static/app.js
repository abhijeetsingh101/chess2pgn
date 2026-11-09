// Chess Diagram -> PGN  front end (vanilla JS, no external dependencies)
"use strict";

const PIECES = ["K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p"];
const pieceImg = p => `/static/pieces/${p === p.toUpperCase() ? "w" : "b"}${p.toUpperCase()}.svg`;
const $ = (s, el = document) => el.querySelector(s);
const state = { puzzles: [], pages: [], tool: "move", sel: null, counter: 0 };

// ------------------------------------------------------------------ FEN helpers
function parseFen(fen) {
  const parts = fen.trim().split(/\s+/);
  const board = Array(64).fill("");
  const rows = (parts[0] || "").split("/");
  if (rows.length !== 8) throw new Error("FEN must have 8 ranks");
  rows.forEach((r, ri) => {
    let c = 0;
    for (const ch of r) {
      if (/\d/.test(ch)) c += +ch;
      else if (PIECES.includes(ch)) { if (c < 8) board[ri * 8 + c] = ch; c++; }
      else throw new Error(`Bad character “${ch}” in FEN`);
    }
    if (c !== 8) throw new Error(`Rank ${8 - ri} does not have 8 squares`);
  });
  const stm = parts[1] === "b" ? "b" : "w";
  const cs = parts[2] || "-";
  return { board, stm, castling: { K: cs.includes("K"), Q: cs.includes("Q"), k: cs.includes("k"), q: cs.includes("q") } };
}

function castlingPossible(b) {
  return {
    K: b[60] === "K" && b[63] === "R", Q: b[60] === "K" && b[56] === "R",
    k: b[4] === "k" && b[7] === "r", q: b[4] === "k" && b[0] === "r",
  };
}

function toFen(p) {
  let rows = [];
  for (let r = 0; r < 8; r++) {
    let s = "", e = 0;
    for (let c = 0; c < 8; c++) {
      const x = p.board[r * 8 + c];
      if (!x) e++; else { if (e) { s += e; e = 0; } s += x; }
    }
    if (e) s += e;
    rows.push(s);
  }
  const poss = castlingPossible(p.board);
  let cs = ["K", "Q", "k", "q"].filter(k => p.castling[k] && poss[k]).join("") || "-";
  return `${rows.join("/")} ${p.stm} ${cs} - 0 1`;
}

// ------------------------------------------------------------------ upload
const drop = $("#drop"), fileInput = $("#fileInput");
$("#browse").onclick = e => { e.preventDefault(); fileInput.click(); };
drop.addEventListener("click", e => { if (e.target === drop || e.target.closest(".drop-inner")) if (e.target.id !== "browse") fileInput.click(); });
fileInput.onchange = () => { upload([...fileInput.files]); fileInput.value = ""; };
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => upload([...e.dataTransfer.files]));
window.addEventListener("paste", e => {
  if (e.target.closest("input, textarea")) return;
  const files = [...(e.clipboardData?.files || [])].filter(f => f.type.startsWith("image/"));
  if (files.length) upload(files.map((f, i) => new File([f], `pasted-${Date.now()}-${i}.png`, { type: f.type })));
});

async function upload(files) {
  files = files.filter(f => f.type.startsWith("image/") || /\.pdf$/i.test(f.name));
  if (!files.length) return toast("Please choose PDF or image files");
  const fd = new FormData();
  files.forEach(f => fd.append("files", f, f.name));
  const prog = $("#progress");
  prog.hidden = false;
  setProgress(0, 1, "Uploading…");
  let res;
  try { res = await (await fetch("/api/upload", { method: "POST", body: fd })).json(); }
  catch (e) { setProgress(0, 1, "Upload failed: " + e); return; }
  if (res.error) return setProgress(0, 1, res.error);
  poll(res.job);
}

function setProgress(a, b, text) {
  $("#progress .bar span").style.width = `${Math.round(100 * a / Math.max(b, 1))}%`;
  $("#progress .ptext").textContent = text;
}

async function poll(job) {
  let since = 0, psince = 0;
  for (;;) {
    const r = await (await fetch(`/api/job/${job}?since=${since}&psince=${psince}`)).json();
    r.puzzles.forEach(addPuzzle);
    r.pages.forEach(addPage);
    since = r.n_puzzles; psince = r.n_pages;
    setProgress(r.progress, r.total, `Processed ${r.progress} of ${r.total} page${r.total > 1 ? "s" : ""} · ${state.puzzles.length} diagram${state.puzzles.length === 1 ? "" : "s"} found`);
    if (r.done) {
      r.warnings.forEach(w => { const d = document.createElement("div"); d.className = "warn"; d.textContent = w; $("#warnings").append(d); });
      if (r.error) setProgress(r.progress, r.total, "Error: " + r.error);
      setTimeout(() => { if (!r.error) $("#progress").hidden = true; }, 2500);
      return;
    }
    await new Promise(ok => setTimeout(ok, 600));
  }
}

// ------------------------------------------------------------------ pages / manual selection
function addPage(pg) {
  state.pages.push(pg);
  $("#pagesBar").hidden = false;
  const b = document.createElement("button");
  b.className = "ghost";
  b.textContent = pg.label;
  b.onclick = () => openPage(pg);
  $("#pageList").append(b);
}

let modalPage = null, dragStart = null, selRect = null;
function openPage(pg) {
  modalPage = pg; selRect = null;
  $("#modalTitle").textContent = pg.label;
  $("#sel").style.display = "none";
  $("#modalOk").disabled = true;
  $("#pageImg").src = `/api/page/${pg.id}.png`;
  $("#modal").hidden = false;
}
$("#modalCancel").onclick = () => { $("#modal").hidden = true; };
$("#modal").addEventListener("click", e => { if (e.target.id === "modal") $("#modal").hidden = true; });
const wrap = $("#pageWrap");
wrap.addEventListener("pointerdown", e => {
  e.preventDefault();
  const r = wrap.getBoundingClientRect();
  dragStart = [e.clientX - r.left, e.clientY - r.top];
  wrap.setPointerCapture(e.pointerId);
});
wrap.addEventListener("pointermove", e => {
  if (!dragStart) return;
  const r = wrap.getBoundingClientRect();
  const x = Math.min(Math.max(e.clientX - r.left, 0), r.width), y = Math.min(Math.max(e.clientY - r.top, 0), r.height);
  const x0 = Math.min(dragStart[0], x), y0 = Math.min(dragStart[1], y);
  // keep it square-ish: boards are square, so use the larger side unless clearly not square
  let w = Math.abs(x - dragStart[0]), h = Math.abs(y - dragStart[1]);
  selRect = [x0, y0, w, h];
  Object.assign($("#sel").style, { display: "block", left: x0 + "px", top: y0 + "px", width: w + "px", height: h + "px" });
});
wrap.addEventListener("pointerup", () => {
  dragStart = null;
  $("#modalOk").disabled = !(selRect && selRect[2] > 20 && selRect[3] > 20);
});
$("#modalOk").onclick = async () => {
  const img = $("#pageImg"), sx = modalPage.w / img.clientWidth, sy = modalPage.h / img.clientHeight;
  const [x, y, w, h] = selRect;
  const corners = [[x * sx, y * sy], [(x + w) * sx, y * sy], [(x + w) * sx, (y + h) * sy], [x * sx, (y + h) * sy]];
  $("#modalOk").disabled = true;
  $("#modalOk").textContent = "Recognising…";
  const r = await (await fetch("/api/manual", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ page_id: modalPage.id, corners, source: modalPage.label }),
  })).json();
  $("#modalOk").textContent = "Recognise selection";
  $("#modal").hidden = true;
  r.puzzles.forEach(p => addPuzzle(p, true));
  toast("Board added");
};

// ------------------------------------------------------------------ palette
function buildPalette() {
  const mk = p => {
    const b = document.createElement("button");
    b.className = "tool"; b.dataset.tool = p; b.title = p;
    b.innerHTML = `<img src="${pieceImg(p)}" alt="${p}">`;
    b.draggable = true;
    b.addEventListener("dragstart", e => e.dataTransfer.setData("text/plain", JSON.stringify({ piece: p })));
    return b;
  };
  "KQRBNP".split("").forEach(p => $("#whitePieces").append(mk(p)));
  "kqrbnp".split("").forEach(p => $("#blackPieces").append(mk(p)));
  document.querySelectorAll(".tool").forEach(b => b.onclick = () => setTool(b.dataset.tool));
}
function setTool(t) {
  state.tool = t; state.sel = null;
  document.querySelectorAll(".tool").forEach(b => b.classList.toggle("active", b.dataset.tool === t));
  document.querySelectorAll(".sq.sel").forEach(s => s.classList.remove("sel"));
}

// ------------------------------------------------------------------ puzzle cards
function addPuzzle(raw, scrollTo = false) {
  const parsed = parseFen(raw.fen);
  const n = ++state.counter;
  const capNum = (raw.caption || "").match(/^\s*(?:No\.?|#|Diagram|Puzzle|Exercise|Problem)?\s*(\d{1,4})\b/i);
  const p = {
    ...raw, ...parsed, n,
    title: capNum ? `Puzzle ${capNum[1]}` : `Puzzle ${n}`,
    solution: "", include: true, uncertain: new Set(raw.uncertain || []),
  };
  state.puzzles.push(p);
  renderCard(p);
  updateChrome();
  validate(p);
  if (scrollTo) p.el.scrollIntoView({ behavior: "smooth", block: "center" });
}

function renderCard(p) {
  const el = document.createElement("article");
  el.className = "card";
  el.innerHTML = `
    <div class="card-head">
      <span class="num">#${p.n}</span>
      <input class="title" value="" spellcheck="false" title="Event / title used in the PGN">
      <span class="src"></span>
      <label class="chk"><input type="checkbox" class="inc" checked> Export</label>
      <button class="ghost small del" title="Remove this board">✕</button>
    </div>
    <div class="card-body">
      <div class="orig"><div class="label">Original</div><img alt="original diagram"><div class="cap"></div></div>
      <div>
        <div class="boardwrap">
          <div class="ranks">${[8, 7, 6, 5, 4, 3, 2, 1].map(r => `<span>${r}</span>`).join("")}</div>
          <div class="board"></div>
          <div class="files">${"abcdefgh".split("").map(f => `<span>${f}</span>`).join("")}</div>
        </div>
        <div class="boardtools">
          <button class="small rot" title="The diagram was shown from Black's side: rotate the position 180°">⟲ Rotate 180°</button>
          <button class="small clr" title="Empty the board">Clear</button>
          <a class="small lich" target="_blank" rel="noopener"><button class="small">Open in Lichess ↗</button></a>
        </div>
      </div>
      <div class="side">
        <div class="row"><span class="lbl">To move</span>
          <div class="seg"><button data-s="w">White</button><button data-s="b">Black</button></div>
          <span class="note stmnote"></span>
        </div>
        <div class="row castle"><span class="lbl">Castling</span>
          ${["K", "Q", "k", "q"].map(k => `<label data-k="${k}"><input type="checkbox" data-k="${k}"> ${{ K: "White O-O", Q: "White O-O-O", k: "Black O-O", q: "Black O-O-O" }[k]}</label>`).join("")}
        </div>
        <div class="row"><span class="lbl">FEN</span><input class="fen" spellcheck="false"><button class="small cp">Copy</button></div>
        <div><div class="row" style="margin-bottom:4px"><span class="lbl">Solution</span><span class="note">optional · SAN, e.g. <code>1. Rxh7+ Kxh7 2. Qh5+ Kg8 3. Ng5</code></span></div>
          <textarea class="sol" spellcheck="false" placeholder="Type the moves of the solution (or leave empty)"></textarea></div>
        <div class="status info">Checking…</div>
      </div>
    </div>`;
  p.el = el;
  $(".title", el).value = p.title;
  $(".src", el).textContent = p.source || "";
  $(".orig img", el).src = `/api/crop/${p.id}.png`;
  $(".cap", el).textContent = p.caption || "";
  if (p.flipped) $(".stmnote", el).textContent = "";
  $(".title", el).oninput = e => { p.title = e.target.value; };
  $(".inc", el).onchange = e => { p.include = e.target.checked; el.classList.toggle("excluded", !p.include); };
  $(".del", el).onclick = () => { state.puzzles = state.puzzles.filter(x => x !== p); el.remove(); updateChrome(); };
  $(".rot", el).onclick = () => {
    p.board.reverse();
    p.uncertain = new Set([...p.uncertain].map(i => 63 - i));
    p.flipped = !p.flipped;
    changed(p);
  };
  $(".clr", el).onclick = () => { p.board.fill(""); p.uncertain.clear(); changed(p); };
  el.querySelectorAll(".seg button").forEach(b => b.onclick = () => { p.stm = b.dataset.s; changed(p); });
  el.querySelectorAll(".castle input").forEach(c => c.onchange = () => { p.castling[c.dataset.k] = c.checked; changed(p, false); });
  $(".fen", el).onchange = e => {
    try { Object.assign(p, parseFen(e.target.value)); p.uncertain.clear(); changed(p); }
    catch (err) { setStatus(p, "bad", err.message); }
  };
  $(".cp", el).onclick = () => copy(toFen(p), "FEN copied");
  let t;
  $(".sol", el).oninput = e => { p.solution = e.target.value; clearTimeout(t); t = setTimeout(() => validate(p), 450); };
  $("#cards").append(el);
  drawBoard(p);
}

function drawBoard(p) {
  const el = p.el, bd = $(".board", el);
  bd.innerHTML = "";
  for (let i = 0; i < 64; i++) {
    const r = i >> 3, c = i & 7;
    const sq = document.createElement("div");
    sq.className = `sq ${(r + c) % 2 ? "d" : "l"}` + (p.uncertain.has(i) ? " unsure" : "");
    sq.dataset.i = i;
    if (p.board[i]) {
      const im = document.createElement("img");
      im.src = pieceImg(p.board[i]); im.alt = p.board[i]; im.draggable = true;
      im.addEventListener("dragstart", e => e.dataTransfer.setData("text/plain", JSON.stringify({ from: i, piece: p.board[i] })));
      sq.append(im);
    }
    sq.addEventListener("click", () => squareClick(p, i, sq));
    sq.addEventListener("contextmenu", e => { e.preventDefault(); p.board[i] = ""; p.uncertain.delete(i); changed(p); });
    sq.addEventListener("dragover", e => { e.preventDefault(); sq.classList.add("drop-target"); });
    sq.addEventListener("dragleave", () => sq.classList.remove("drop-target"));
    sq.addEventListener("drop", e => {
      e.preventDefault(); sq.classList.remove("drop-target");
      let d; try { d = JSON.parse(e.dataTransfer.getData("text/plain")); } catch { return; }
      if (d.from !== undefined && d.from !== i) { p.board[d.from] = ""; p.uncertain.delete(d.from); }
      p.board[i] = d.piece; p.uncertain.delete(i);
      changed(p);
    });
    bd.append(sq);
  }
  // side panel
  el.querySelectorAll(".seg button").forEach(b => b.classList.toggle("on", b.dataset.s === p.stm));
  const poss = castlingPossible(p.board);
  el.querySelectorAll(".castle label").forEach(l => {
    const k = l.dataset.k, inp = $("input", l);
    inp.disabled = !poss[k]; inp.checked = poss[k] && p.castling[k];
    l.classList.toggle("disabled", !poss[k]);
  });
  const fen = toFen(p);
  $(".fen", el).value = fen;
  $(".lich", el).href = "https://lichess.org/analysis/standard/" + fen.replace(/ /g, "_");
  $(".stmnote", el).textContent = p.flipped ? "Shown from Black's side in the original, so it was rotated (and Black to move assumed)" : "";
}

function squareClick(p, i, sq) {
  if (state.tool === "move") {
    if (state.sel && state.sel.p === p && state.sel.i !== i) {
      const from = state.sel.i;
      if (p.board[from]) { p.board[i] = p.board[from]; p.board[from] = ""; p.uncertain.delete(i); p.uncertain.delete(from); }
      state.sel = null; changed(p);
    } else if (p.board[i]) {
      document.querySelectorAll(".sq.sel").forEach(s => s.classList.remove("sel"));
      state.sel = { p, i }; sq.classList.add("sel");
    }
    return;
  }
  const v = state.tool === "x" ? "" : state.tool;
  p.board[i] = p.board[i] === v ? "" : v;   // clicking the same piece again removes it
  p.uncertain.delete(i);
  changed(p);
}

function changed(p, redraw = true) {
  if (redraw) drawBoard(p); else $(".fen", p.el).value = toFen(p);
  clearTimeout(p._vt);
  p._vt = setTimeout(() => validate(p), 250);
}

async function validate(p) {
  const fen = toFen(p);
  if (p.el) $(".lich", p.el).href = "https://lichess.org/analysis/standard/" + fen.replace(/ /g, "_");
  let r;
  try {
    r = await (await fetch("/api/validate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ fen, solution: p.solution }) })).json();
  } catch { return; }
  p.problems = r.problems;
  p.solErr = r.solution_error;
  const unsure = p.uncertain.size;
  if (r.problems.length) setStatus(p, "bad", "⚠ " + r.problems.join(" · "));
  else if (r.solution_error) setStatus(p, "bad", "⚠ Solution: " + r.solution_error);
  else {
    let msg = "✓ Legal position";
    if (r.solution_san && r.solution_san.length) msg += " · solution OK: " + r.solution_san;
    if (unsure) msg += ` · ${unsure} square${unsure > 1 ? "s" : ""} worth a second look (orange)`;
    setStatus(p, unsure ? "info" : "ok", msg);
  }
  p.el.classList.toggle("bad", !!(r.problems.length || r.solution_error));
  applyFilter();
}

function setStatus(p, cls, text) {
  const s = $(".status", p.el);
  s.className = "status " + cls;
  s.textContent = text;
}

function updateChrome() {
  const n = state.puzzles.length;
  $("#palette").hidden = !n; $("#listHead").hidden = !n; $("#empty").hidden = !!n;
  $("#count").textContent = `${n} board${n === 1 ? "" : "s"}`;
  state.puzzles.forEach((p, i) => { $(".num", p.el).textContent = "#" + (i + 1); });
}

function applyFilter() {
  const only = $("#onlyProblems").checked;
  state.puzzles.forEach(p => {
    const needs = (p.problems && p.problems.length) || p.solErr || p.uncertain.size;
    p.el.style.display = only && !needs ? "none" : "";
  });
}
$("#onlyProblems").onchange = applyFilter;
$("#btnClear").onclick = () => {
  if (!confirm("Remove all boards?")) return;
  state.puzzles.forEach(p => p.el.remove()); state.puzzles = []; updateChrome();
};

// ------------------------------------------------------------------ export
function exportPayload() {
  const list = state.puzzles.filter(p => p.include);
  return {
    filename: $("#filename").value.trim() || "puzzles",
    event: $("#filename").value.trim() || "Puzzles",
    puzzles: list.map(p => ({ fen: toFen(p), solution: p.solution, title: p.title, source: p.source })),
  };
}
async function fetchPgn() {
  const payload = exportPayload();
  if (!payload.puzzles.length) { toast("Nothing to export yet"); return null; }
  const bad = state.puzzles.filter(p => p.include && p.problems && p.problems.length).length;
  if (bad && !confirm(`${bad} board(s) are not legal positions yet. Export anyway?`)) return null;
  const res = await fetch("/api/export", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  return { blob: await res.blob(), name: payload.filename + ".pgn", n: payload.puzzles.length };
}
$("#btnDownload").onclick = async () => {
  const r = await fetchPgn(); if (!r) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(r.blob); a.download = r.name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  toast(`Saved ${r.n} puzzle${r.n > 1 ? "s" : ""} to ${r.name}`);
};
$("#btnCopy").onclick = async () => { const r = await fetchPgn(); if (r) copy(await r.blob.text(), "PGN copied to clipboard"); };

function copy(text, msg) {
  (navigator.clipboard?.writeText(text) || Promise.reject()).then(() => toast(msg)).catch(() => {
    const ta = document.createElement("textarea"); ta.value = text; document.body.append(ta); ta.select();
    document.execCommand("copy"); ta.remove(); toast(msg);
  });
}
let toastT;
function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toastT); toastT = setTimeout(() => t.hidden = true, 2200);
}
window.addEventListener("beforeunload", e => { if (state.puzzles.length) { e.preventDefault(); e.returnValue = ""; } });

buildPalette();
updateChrome();
