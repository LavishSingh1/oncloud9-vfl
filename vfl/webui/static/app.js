/* VFL control center: one page for data prep, deployment, training, attacks, tests/benchmarks, analytics and
 * reports. Vanilla JS, no external dependencies. Every action calls an API that runs the same CLI / library
 * code as the command line. Figures are rendered server-side by bench/analyze.py (identical to report.md). */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
// v2.3 (RT-04): links built from data another node supplied (party dashboard URLs) must be http(s); a party
// could register ui_url "javascript:..." and run script in the operator's browser. The server refuses such
// URLs too and the CSP blocks javascript: navigation; this is the third layer.
const safeUrl = (u) => { try { const x = new URL(String(u ?? ""), location.href); return (x.protocol === "http:" || x.protocol === "https:") ? x.href : "#"; } catch { return "#"; } };
const J = (v) => JSON.stringify(v === undefined ? null : v);
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "—" : typeof v === "number" ? (Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(d)) : String(v));
const ago = (t) => (t ? new Date(t * 1000).toLocaleTimeString() : "—");
const PAGE = { cur: null, timer: null };
let WHO = {};

/* ---------------- token (per viewer, convenience only) ---------------- */
let TOKEN = new URLSearchParams(location.search).get("token") || "";
try { TOKEN = TOKEN || sessionStorage.getItem("vfl-token") || ""; } catch { /* storage may be blocked */ }
function setToken(t) { TOKEN = t; try { sessionStorage.setItem("vfl-token", t); } catch { /* ignore */ } }
if (new URLSearchParams(location.search).has("token")) {   // keep it out of history / bookmarks (v2.3)
  setToken(TOKEN);
  const u = new URL(location.href); u.searchParams.delete("token"); history.replaceState(null, "", u.pathname + u.search + u.hash);
}

async function api(method, url, body, opts = {}) {
  const h = { "X-VFL-Request": "1" };            // CSRF guard: custom header = same-origin only (v2.3)
  if (TOKEN) h.Authorization = `Bearer ${TOKEN}`;
  let payload;
  if (body instanceof FormData) payload = body;
  else if (body !== undefined) { h["Content-Type"] = "application/json"; payload = JSON.stringify(body); }
  const r = await fetch(url, { method, headers: h, body: payload, cache: "no-store" });
  const txt = await r.text();
  let j;
  try { j = JSON.parse(txt.replace(/\bNaN\b/g, "null")); } catch { j = { raw: txt }; }
  if (!r.ok && !opts.soft) throw new Error(j.error || r.statusText);
  return j;
}
const get = (u) => api("GET", u);
const post = (u, b) => api("POST", u, b || {});
function msg(el, text, kind = "") { if (el) { el.className = `form-msg ${kind}`; el.textContent = text; } }

/* ---------------- generic widgets ---------------- */
function opt(v, label, sel) { return `<option value="${esc(J(v))}"${sel ? " selected" : ""}>${esc(label ?? (v === null ? "—" : v))}</option>`; }
function selectHTML(name, options, value, attrs = "") {
  return `<select name="${esc(name)}" ${attrs}>${options.map((o) => (typeof o === "object" && o !== null && "value" in o
    ? opt(o.value, o.label, J(o.value) === J(value)) : opt(o, o === null ? "—" : String(o), J(o) === J(value)))).join("")}</select>`;
}
const val = (el) => (el ? JSON.parse(el.value) : undefined);
function chipsHTML(name, options, values) {
  const on = new Set((values || []).map((x) => J(x)));
  return `<div class="chips" data-multi="${esc(name)}">${options.map((o) => `<label class="chip${on.has(J(o)) ? " on" : ""}">
    <input type="checkbox" value="${esc(J(o))}"${on.has(J(o)) ? " checked" : ""}>${esc(o)}</label>`).join("")}</div>`;
}
function chipsVal(el) { return $$("input:checked", el).map((i) => JSON.parse(i.value)); }
document.addEventListener("change", (e) => { const l = e.target.closest("label.chip"); if (l) l.classList.toggle("on", e.target.checked); });
function status(s) { return `<span class="status ${esc(s)}">${esc(s)}</span>`; }

/* watch a job: poll its view, render status + log; resolve with the final view */
function watch(url, box, { onDone, interval = 900 } = {}) {
  let stop = false;
  const tick = async () => {
    if (stop) return;
    try {
      const j = await get(url);
      if (box) box.innerHTML = `<div class="row">${status(j.status)} <span class="muted">${esc(j.label || j.name || "")}</span>
        ${j.progress ? `<span class="muted">${esc(J(j.progress))}</span>` : ""}
        ${["running", "queued"].includes(j.status) ? `<button class="btn mini" data-cancel="${esc(url)}">cancel</button>` : ""}</div>
        <pre class="logbox">${esc(j.log || "")}</pre>`;
      if (box) { const pre = $("pre", box); if (pre) pre.scrollTop = pre.scrollHeight; }
      if (["done", "failed", "cancelled"].includes(j.status) && j.ended !== null) { if (onDone) onDone(j); return; }
    } catch (e) { if (box) box.innerHTML = `<p class="form-msg err">${esc(e.message)}</p>`; return; }
    setTimeout(tick, interval);
  };
  tick();
  return () => { stop = true; };
}
document.addEventListener("click", async (e) => {
  const b = e.target.closest("[data-cancel]");
  if (b) { try { await post(b.dataset.cancel.replace(/\?.*$/, "") + "/cancel"); } catch (x) { alert(x.message); } }
});

/* ---------------- router ---------------- */
const PAGES = {};
function go() {
  const page = (location.hash || "").slice(1) || WHO.start_page || "dataprep";
  if (!PAGES[page]) { location.hash = "#dataprep"; return; }
  $$("#nav .tab").forEach((t) => t.classList.toggle("active", t.dataset.page === page));
  if (PAGE.timer) { clearInterval(PAGE.timer); PAGE.timer = null; }
  PAGE.cur = page;
  $("#page").innerHTML = "";
  PAGES[page]($("#page"));
}
function every(ms, fn) { if (PAGE.timer) clearInterval(PAGE.timer); fn(); PAGE.timer = setInterval(fn, ms); }
window.addEventListener("hashchange", go);

async function boot() {
  $("#tok").value = TOKEN;
  $("#tok").addEventListener("change", (e) => { setToken(e.target.value.trim()); refreshWho(); });
  await refreshWho();
  go();
}
async function refreshWho() {
  WHO = await get("/api/whoami");
  $("#ws-line").textContent = `workspace ${WHO.workspace}`;
  $("#lab-pill").hidden = !WHO.lab_mode;
  $("#auth-pill").innerHTML = WHO.token_configured
    ? (WHO.operator ? `<span class="dot ok"></span>operator` : `<span class="dot warn"></span>viewer (enter token)`)
    : `<span class="dot warn"></span>no token set`;
}

/* ======================================================================== DATA PREPARATION (item 3) */
const PREP = { values: {}, form: null, columns: [], manual: {}, preview: null };
PAGES.dataprep = async (root) => {
  root.innerHTML = `<p class="page-intro">Every option of <code>vfl.cli prepare-data</code>, as dropdowns and toggles. Preview runs the
    real command into a scratch folder; Generate runs it into <code>datasets/NAME</code> (byte-identical to the CLI).</p>
  <div class="grid two">
    <section class="card"><div class="card-head"><h2>Prepare a dataset</h2><span class="muted" id="prep-kind"></span></div>
      <div class="fields" id="prep-fields"></div><div id="prep-manual"></div><div id="prep-notes"></div>
      <div class="sticky-actions"><label class="fld"><span class="lbl">save as</span><select id="prep-name"></select></label>
        <label class="chip"><input type="checkbox" id="prep-over"> overwrite</label>
        <button class="btn" id="prep-preview">Preview</button><button class="btn primary" id="prep-gen">Generate</button>
        <span class="form-msg" id="prep-msg"></span></div>
      <div id="prep-job"></div></section>
    <section class="card"><div class="card-head"><h2>Preview</h2><span class="muted">reference AUROC is experiment-only</span></div>
      <div id="prep-out"><p class="muted">Preview shows per-party columns/shape, rows, intersection size, label balance,
      importance share and reference AUROC before anything is saved.</p></div></section>
  </div>
  <section class="card"><div class="card-head"><h2>Saved datasets</h2><span class="muted">datasets/ · pick them in Deployment, Training and Benchmarks</span></div>
    <div class="table-wrap"><table id="ds-table"><thead><tr><th>Name</th><th>Dataset</th><th>Split</th><th class="num">Parties</th>
    <th class="num">Entities</th><th class="num">Intersection</th><th>Classes / balance</th><th>Files</th><th></th></tr></thead><tbody></tbody></table></div></section>
  <div class="grid two">
    <section class="card"><div class="card-head"><h2>Certificates</h2><span class="muted"><code>vfl.cli certs</code></span></div>
      <div class="fields"><label class="fld"><span class="lbl">passive parties</span>${selectHTML("c-pas", [1, 2, 3, 4, 5, 6, 7, 8], 1)}</label>
      <label class="fld"><span class="lbl">mode</span>${selectHTML("c-mode", [{ value: "static", label: "static (one CA)" }, { value: "pki", label: "pki (root + intermediate, short-lived)" }], "static")}</label>
      <label class="fld wide"><span class="lbl">extra host addresses for the certificate SANs</span><div id="c-hosts" class="chips"></div></label></div>
      <div class="sticky-actions"><button class="btn primary" id="c-go">Generate certificates</button><span class="form-msg" id="c-msg"></span></div>
      <div id="c-job"></div>
      <div class="table-wrap"><table id="c-table"><thead><tr><th>Set</th><th>Mode</th><th>Nodes</th></tr></thead><tbody></tbody></table></div></section>
    <section class="card"><div class="card-head"><h2>Uploads &amp; text embeddings</h2><span class="muted"><code>vfl.cli embed-text</code></span></div>
      <div class="row"><input type="file" id="up-file"><button class="btn" id="up-go">Upload</button><span class="form-msg" id="up-msg"></span></div>
      <p class="muted">CSV or .npz for datasets; a .zip (e.g. UCI HAR) is extracted under uploads/.</p>
      <div class="fields" id="emb-fields"></div>
      <div class="sticky-actions"><button class="btn" id="emb-go">Embed text column</button><span class="form-msg" id="emb-msg"></span></div>
      <div id="emb-job"></div></section>
  </div>`;
  $("#prep-preview").onclick = () => prepRun("preview");
  $("#prep-gen").onclick = () => prepRun("generate");
  $("#c-go").onclick = makeCerts;
  $("#up-go").onclick = doUpload;
  $("#emb-go").onclick = doEmbed;
  await prepForm();
  every(4000, refreshPrepLists);
};

async function prepForm() {
  const f = await post("/api/prep/form", { values: PREP.values });
  PREP.form = f; PREP.values = f.values;
  $("#prep-kind").textContent = `${f.kind} · ${f.passives} passive + 1 active`;
  const box = $("#prep-fields");
  box.innerHTML = f.fields.filter((x) => x.widget !== "columns_table").map((x) => {
    let w;
    if (x.widget === "toggle") w = `<label class="chip${x.value ? " on" : ""}"><input type="checkbox" data-dest="${x.dest}"${x.value ? " checked" : ""}> ${x.flag}</label>`;
    else if (x.widget === "upload" || x.widget === "dir" || x.widget === "column")
      w = selectHTML(x.dest, [{ value: null, label: x.options.length ? "— choose —" : (x.widget === "column" ? "choose the CSV first" : "upload a file first") }, ...x.options], x.value, `data-dest="${x.dest}"`);
    else w = selectHTML(x.dest, x.options, x.value, `data-dest="${x.dest}"`);
    return `<label class="fld"><span class="lbl">${esc(x.flag)}</span>${w}<span class="hlp">${esc(x.help)}</span></label>`;
  }).join("");
  $$("[data-dest]", box).forEach((el) => el.addEventListener("change", () => {
    PREP.values[el.dataset.dest] = el.type === "checkbox" ? el.checked : val(el);
    prepForm();
  }));
  $("#prep-notes").innerHTML = (f.notes || []).map((n) => `<div class="note">${esc(n)}</div>`).join("");
  const manual = f.fields.find((x) => x.widget === "columns_table");
  if (manual) await renderManual(); else $("#prep-manual").innerHTML = "";
  const v = f.values, part = f.fields.some((x) => x.dest === "partition") ? v.partition : "fixed";
  const names = [`${v.dataset}-${part}-${v.parties}p-s${v.seed}`, `${v.dataset}-${v.partitioner}-${v.parties}p`, `${v.dataset}-${v.parties}p`];
  $("#prep-name").innerHTML = [...new Set(names)].map((n) => opt(n, n)).join("");
}

async function renderManual() {
  const n = PREP.values.parties;
  let cols = [];
  try { cols = (await post("/api/prep/columns", { values: PREP.values })).columns; } catch (e) { $("#prep-manual").innerHTML = `<p class="form-msg err">${esc(e.message)}</p>`; return; }
  PREP.columns = cols;
  cols.forEach((c, i) => { if (PREP.manual[c] === undefined || PREP.manual[c] >= n) PREP.manual[c] = i % n; });
  const names = ["active", ...Array.from({ length: n - 1 }, (_, i) => `passive-${i + 1}`)];
  const sync = () => { PREP.values.columns_json = names.map((_, p) => cols.filter((c) => PREP.manual[c] === p)); };
  sync();
  $("#prep-manual").innerHTML = `<h3 class="sub">Manual split: column → party (${cols.length} columns)</h3>
    <div class="row">${names.map((nm, p) => `<button class="btn mini" data-all="${p}">all → ${nm}</button>`).join("")}</div>
    <div class="table-wrap" style="max-height:280px;overflow:auto"><table><thead><tr><th>Column</th><th>Party</th></tr></thead><tbody>
    ${cols.map((c) => `<tr><td class="mono">${esc(c)}</td><td>${selectHTML("m-" + c, names.map((nm, p) => ({ value: p, label: nm })), PREP.manual[c], `data-col="${esc(c)}"`)}</td></tr>`).join("")}
    </tbody></table></div><p class="muted" id="m-count"></p>`;
  const count = () => { $("#m-count").textContent = names.map((nm, p) => `${nm}: ${cols.filter((c) => PREP.manual[c] === p).length}`).join(" · "); };
  count();
  $$("[data-col]").forEach((s) => s.addEventListener("change", () => { PREP.manual[s.dataset.col] = val(s); sync(); count(); }));
  $$("[data-all]").forEach((b) => b.addEventListener("click", () => { cols.forEach((c) => (PREP.manual[c] = +b.dataset.all)); renderManual(); }));
}

async function prepRun(kind) {
  const m = $("#prep-msg");
  try {
    const r = kind === "preview" ? await post("/api/prep/preview", { values: PREP.values })
      : await post("/api/prep/generate", { values: PREP.values, name: val($("#prep-name")), overwrite: $("#prep-over").checked });
    msg(m, kind === "preview" ? "previewing…" : `running: vfl.cli ${(r.argv || []).join(" ")}`);
    watch(`/api/ws-jobs/${r.job}`, $("#prep-job"), {
      onDone: (j) => { msg(m, j.status === "done" ? (kind === "preview" ? "preview ready" : "saved") : j.status, j.status === "done" ? "ok" : "err");
        if (j.result) renderSummary($("#prep-out"), j.result, kind === "preview"); refreshPrepLists(); },
    });
  } catch (e) { msg(m, e.message, "err"); }
}

function renderSummary(el, s, preview) {
  const lb = s.label_distribution ? Object.entries(s.label_distribution).map(([k, v]) => `${k}: ${(v * 100).toFixed(1)}%`).join(" · ") : fmt(s.label_balance);
  const ref = s.reference_auroc && typeof s.reference_auroc === "object" ? s.reference_auroc : null;
  el.innerHTML = `<div class="kvs">
    <div class="k">dataset</div><div class="v">${esc(s.dataset)} (${esc(s.partitioner)}, ${esc(s.strategy)})</div>
    <div class="k">entities</div><div class="v">${fmt(s.entities_total, 0)}</div>
    <div class="k">intersection</div><div class="v">${fmt(s.intersection_size, 0)} (${fmt((s.intersection_ratio || 0) * 100, 1)}%)</div>
    <div class="k">classes</div><div class="v">${fmt(s.num_classes, 0)}</div>
    <div class="k">label balance</div><div class="v">${esc(lb)}</div>
    ${s.image_shape ? `<div class="k">image</div><div class="v">${esc(s.image_shape.join("×"))}</div>` : ""}</div>
    <div class="table-wrap"><table><thead><tr><th>Party</th><th>Role</th><th class="num">Rows</th><th class="num">Inputs</th><th>Shape</th>
    <th class="num">Importance</th><th class="num">Ref. AUROC</th></tr></thead><tbody>
    ${s.parties.map((p) => `<tr><td class="mono">${esc(p.name)}</td><td>${esc(p.role)}</td><td class="num">${fmt(p.rows, 0)}</td>
      <td class="num">${fmt(p.inputs, 0)}</td><td class="mono">${p.shape ? esc(p.shape.join("×")) : "—"}</td>
      <td class="num">${p.importance_share !== undefined && p.importance_share !== null ? fmt(p.importance_share * 100, 1) + "%" : p.mutual_info_sum !== undefined ? "MI " + fmt(p.mutual_info_sum) : "—"}</td>
      <td class="num">${ref && ref[p.name] !== undefined ? fmt(ref[p.name], 4) : "—"}</td></tr>`).join("")}
    ${ref && ref.all_features !== undefined ? `<tr><td colspan="6" class="muted">all features together</td><td class="num">${fmt(ref.all_features, 4)}</td></tr>` : ""}
    </tbody></table></div>
    ${ref ? `<div class="note">Reference AUROC is <b>experiment-only</b>: it puts every party's columns and the labels in one place. It is a planning aid, not something the deployment computes.</div>` : ""}
    ${preview ? `<p class="muted">Preview only: nothing was saved. Choose Generate to save it.</p>` : `<div class="note ok">Saved. Give each party only its own file and <code>.manifest.json</code>.</div>`}`;
}

async function refreshPrepLists() {
  if (PAGE.cur !== "dataprep") return;
  const [ds, certs, up] = await Promise.all([get("/api/prep/datasets"), get("/api/prep/certs"), get("/api/uploads")]);
  $("#ds-table tbody").innerHTML = ds.length ? ds.map((d) => `<tr><td class="mono"><b>${esc(d.name)}</b><br><span class="muted">${ago(d.created)}</span></td>
    <td>${esc(d.dataset)}</td><td>${esc(d.partitioner)} · ${esc(d.strategy)}</td><td class="num">${d.parties.length}</td>
    <td class="num">${fmt(d.entities_total, 0)}</td><td class="num">${fmt(d.intersection_size, 0)}</td>
    <td>${fmt(d.num_classes, 0)} · ${d.label_balance !== null && d.label_balance !== undefined ? fmt(d.label_balance) : "—"}</td>
    <td class="mono muted">${esc(d.parties.map((p) => p.file).join(", "))}</td>
    <td><button class="btn mini" data-show="${esc(d.name)}">manifest</button><button class="btn mini danger" data-del="${esc(d.name)}">delete</button></td></tr>`).join("")
    : `<tr><td colspan="9" class="empty">No saved datasets yet.</td></tr>`;
  $$("[data-show]").forEach((b) => (b.onclick = async () => renderSummary($("#prep-out"), await get(`/api/prep/datasets/${b.dataset.show}`), false)));
  $$("[data-del]").forEach((b) => (b.onclick = async () => { if (confirm(`Delete dataset ${b.dataset.del}?`)) { await api("DELETE", `/api/prep/datasets/${b.dataset.del}`); refreshPrepLists(); } }));
  const hb = $("#c-hosts");
  if (!hb.dataset.done) { hb.dataset.done = 1; hb.innerHTML = certs.addresses.map((a) => `<label class="chip"><input type="checkbox" value="${esc(J(a))}">${esc(a)}</label>`).join(""); }
  $("#c-table tbody").innerHTML = certs.sets.length ? certs.sets.map((c) => `<tr><td class="mono">${esc(c.name)}</td><td>${esc(c.mode)}</td><td class="mono muted">${esc(c.nodes.join(", "))}</td></tr>`).join("")
    : `<tr><td colspan="3" class="empty">No certificate sets.</td></tr>`;
  const ef = $("#emb-fields");
  const csvs = up.files.filter((f) => f.endsWith(".csv"));
  const cur = ef.dataset.csv && csvs.includes(ef.dataset.csv) ? ef.dataset.csv : csvs[0] || null;
  if (ef.dataset.sig !== J([csvs, cur])) {
    ef.dataset.sig = J([csvs, cur]); ef.dataset.csv = cur || "";
    let cols = [];
    if (cur) try { cols = (await get(`/api/uploads/columns?file=${encodeURIComponent(cur)}`)).columns; } catch { /* unreadable */ }
    ef.innerHTML = `<label class="fld"><span class="lbl">--csv</span>${selectHTML("e-csv", csvs.length ? csvs : [{ value: null, label: "upload a CSV first" }], cur)}</label>
      <label class="fld"><span class="lbl">--text-col</span>${selectHTML("e-col", cols.length ? cols : [{ value: null, label: "—" }], cols.includes("text") ? "text" : cols[0])}</label>
      <label class="fld"><span class="lbl">--encoder</span>${selectHTML("e-enc", ["hashing", "minilm", "distilbert"], "hashing")}</label>
      <label class="fld"><span class="lbl">--dim</span>${selectHTML("e-dim", [16, 32, 64, 128, 256], 64)}</label>`;
    $("[name=e-csv]").onchange = (e) => { ef.dataset.csv = val(e.target); ef.dataset.sig = ""; refreshPrepLists(); };
  }
}

async function makeCerts() {
  const n = val($("[name=c-pas]")), mode = val($("[name=c-mode]"));
  const hosts = $$("#c-hosts input:checked").map((i) => JSON.parse(i.value)).filter((h) => !["127.0.0.1"].includes(h));
  try {
    const r = await post("/api/prep/certs", { name: `certs-${n}p-${mode}`, passives: n, mode, hosts, overwrite: true });
    msg($("#c-msg"), `vfl.cli ${r.argv.join(" ")}`);
    watch(`/api/ws-jobs/${r.job}`, $("#c-job"), { onDone: () => refreshPrepLists() });
  } catch (e) { msg($("#c-msg"), e.message, "err"); }
}
async function doUpload() {
  const f = $("#up-file").files[0];
  if (!f) return msg($("#up-msg"), "choose a file", "err");
  const fd = new FormData(); fd.append("file", f);
  try { const r = await api("POST", "/api/uploads", fd); msg($("#up-msg"), `uploaded ${r.file}${r.dir ? ` → ${r.dir}/` : ""}`, "ok"); prepForm(); refreshPrepLists(); }
  catch (e) { msg($("#up-msg"), e.message, "err"); }
}
async function doEmbed() {
  try {
    const r = await post("/api/prep/embed-text", { csv: val($("[name=e-csv]")), text_col: val($("[name=e-col]")), encoder: val($("[name=e-enc]")), dim: val($("[name=e-dim]")) });
    msg($("#emb-msg"), `→ uploads/${r.out}`);
    watch(`/api/ws-jobs/${r.job}`, $("#emb-job"), { onDone: () => refreshPrepLists() });
  } catch (e) { msg($("#emb-msg"), e.message, "err"); }
}

/* ======================================================================== DEPLOYMENT (item 4) */
const DEP = { log: null, info: null };
PAGES.deployment = async (root) => {
  root.innerHTML = `<p class="page-intro">Start a whole deployment on this machine with one click, or generate one bundle per machine for a LAN / cloud run.
    Node dashboards stay where they are; the links below open them.</p>
  <div class="grid two">
    <section class="card"><div class="card-head"><h2>Local deployment</h2><span class="muted">reuses bench/harness.LocalDeployment</span></div>
      <div class="fields" id="loc-fields"></div><div class="note" id="act-note"></div>
      <div class="sticky-actions"><button class="btn primary" id="loc-go">Launch</button><span class="form-msg" id="loc-msg"></span></div></section>
    <section class="card"><div class="card-head"><h2>Multi-machine bundles</h2><span class="muted">certs + data + command per machine; never ca.key</span></div>
      <div class="fields" id="mm-fields"></div>
      <div class="sticky-actions"><button class="btn primary" id="mm-go">Generate bundles</button><span class="form-msg" id="mm-msg"></span></div></section>
  </div>
  <div id="dep-list"></div>
  <section class="card hidden" id="log-card"><div class="card-head"><h2 id="log-title">Log</h2><button class="btn mini" id="log-close">close</button></div><pre class="logbox" id="log-box" style="max-height:420px"></pre></section>`;
  DEP.info = await get("/api/deploy");
  renderDepForms();
  $("#loc-go").onclick = launchLocal;
  $("#mm-go").onclick = makeBundles;
  $("#log-close").onclick = () => { DEP.log = null; $("#log-card").classList.add("hidden"); };
  every(2000, refreshDeployments);
};
function dsOptions(withNone) {
  const ds = DEP.info.datasets.map((d) => ({ value: d.name, label: `${d.name} (${d.parties.passive.length} passive)` }));
  return ds.length ? ds : [{ value: null, label: withNone || "prepare a dataset first" }];
}
function renderDepForms() {
  const I = DEP.info;
  const ds0 = I.datasets[0];
  const maxP = (name) => Math.min(8, ((I.datasets.find((d) => d.name === name) || {}).parties || { passive: [] }).passive.length || 1);
  const certOpts = [{ value: null, label: "generate for this deployment" }, ...I.certs.map((c) => ({ value: c.name, label: `${c.name} (${c.mode})` }))];
  $("#loc-fields").innerHTML = `
    <label class="fld"><span class="lbl">dataset</span>${selectHTML("l-ds", dsOptions(), ds0 && ds0.name)}</label>
    <label class="fld"><span class="lbl">passive parties</span>${selectHTML("l-np", Array.from({ length: maxP(ds0 && ds0.name) }, (_, i) => i + 1), maxP(ds0 && ds0.name))}</label>
    <label class="fld"><span class="lbl">active parties</span>${selectHTML("l-na", [{ value: 1, label: "1 (fixed: FR-1)" }], 1, "disabled")}</label>
    <label class="fld"><span class="lbl">transport</span>${selectHTML("l-tls", [{ value: "static", label: "mutual TLS" }, { value: "insecure", label: "insecure (debug only)" }], "static")}</label>
    <label class="fld"><span class="lbl">certificates</span>${selectHTML("l-certs", certOpts, null)}</label>
    <label class="fld"><span class="lbl">slot (port block)</span>${selectHTML("l-slot", I.slots.map((s) => ({ value: s, label: s ? `slot ${s}: coordinator UI ${18080 + 100 * s}` : "slot 0: :50050 / UI :8080" })), 0)}</label>
    <label class="fld"><span class="lbl">RTT relay (vfl.cli netem)</span>${selectHTML("l-rtt", [{ value: null, label: "off" }, 0, 10, 20, 50, 100, 200].map((x) => (typeof x === "object" ? x : { value: x, label: `${x} ms` })), null)}</label>
    <label class="fld"><span class="lbl">lab mode</span><label class="chip"><input type="checkbox" id="l-lab"> --lab-mode (attack simulator)</label></label>`;
  $("#act-note").textContent = I.active_note;
  $("[name=l-ds]").onchange = () => { const m = maxP(val($("[name=l-ds]"))); $("[name=l-np]").innerHTML = Array.from({ length: m }, (_, i) => opt(i + 1, String(i + 1), i + 1 === m)).join(""); };
  const addrList = `<datalist id="addrs">${I.addresses.map((a) => `<option value="${esc(a)}">`).join("")}</datalist>`;
  $("#mm-fields").innerHTML = `${addrList}
    <label class="fld"><span class="lbl">dataset</span>${selectHTML("m-ds", dsOptions(), ds0 && ds0.name)}</label>
    <label class="fld"><span class="lbl">certificates</span>${selectHTML("m-certs", I.certs.length ? I.certs.map((c) => ({ value: c.name, label: c.name })) : [{ value: null, label: "make a set in Data preparation" }], I.certs[0] && I.certs[0].name)}</label>
    <label class="fld"><span class="lbl">passive parties</span>${selectHTML("m-np", [1, 2, 3, 4, 5, 6, 7, 8], 1)}</label>
    <label class="fld"><span class="lbl">coordinator host</span><input name="m-coord" list="addrs" pattern="[A-Za-z0-9.:\\-\\[\\]]+" value="${esc(I.addresses[1] || "127.0.0.1")}"></label>
    <label class="fld"><span class="lbl">active host</span><input name="m-act" list="addrs" value="${esc(I.addresses[1] || "127.0.0.1")}"></label>
    <div class="fld wide"><span class="lbl">passive hosts</span><div class="row" id="m-pas"></div><span class="hlp">Host names / IPs are the one typed field: they are your machines.</span></div>`;
  const pas = () => { const n = val($("[name=m-np]")); $("#m-pas").innerHTML = Array.from({ length: n }, (_, i) => `<input name="m-p${i}" list="addrs" placeholder="passive-${i + 1} host" value="${esc(I.addresses[1] || "127.0.0.1")}" style="width:150px">`).join(""); };
  pas(); $("[name=m-np]").onchange = pas;
}
async function launchLocal() {
  const ds = val($("[name=l-ds]")), slot = val($("[name=l-slot]"));
  try {
    const r = await post("/api/deploy/local", { name: `local-${ds}-s${slot}`, dataset: ds, n_passive: val($("[name=l-np]")),
      tls: val($("[name=l-tls]")), certs: val($("[name=l-certs]")), slot, lab_mode: $("#l-lab").checked, rtt_relay_ms: val($("[name=l-rtt]")) });
    msg($("#loc-msg"), `starting ${r.name}…`, "ok"); refreshDeployments();
  } catch (e) { msg($("#loc-msg"), e.message, "err"); }
}
async function makeBundles() {
  const n = val($("[name=m-np]"));
  try {
    const r = await post("/api/deploy/bundles", { name: `lan-${val($("[name=m-ds]"))}`, dataset: val($("[name=m-ds]")), certs: val($("[name=m-certs]")), n_passive: n,
      coordinator_host: $("[name=m-coord]").value.trim(), active_host: $("[name=m-act]").value.trim(),
      passive_hosts: Array.from({ length: n }, (_, i) => $(`[name=m-p${i}]`).value.trim()) });
    msg($("#mm-msg"), `bundles for ${r.machines.length} machines`, "ok");
    if (r.ui_token) {        // shown once: the views never carry it
      const others = Object.entries(r.ui_tokens || {}).filter(([n]) => n !== "coordinator")
        .map(([n, t]) => `<br>${esc(n)} dashboard: <code>${esc(t)}</code>`).join("");
      $("#mm-msg").insertAdjacentHTML("afterend", `<div class="note" id="mm-token">Coordinator dashboard operator token
        (also inside <code>coordinator.zip</code>; not shown again): <code>${esc(r.ui_token)}</code>${others}</div>`);
    }
    refreshDeployments();
  } catch (e) { msg($("#mm-msg"), e.message, "err"); }
}
async function refreshDeployments() {
  if (PAGE.cur !== "deployment") return;
  const I = await get("/api/deploy");
  DEP.info.deployments = I.deployments;
  $("#dep-list").innerHTML = I.deployments.map((d) => d.mode === "local" ? `
    <section class="card"><div class="card-head"><h2>${esc(d.name)}</h2>
      <span>${status(d.running ? "running" : "stopped")} <span class="muted">${esc(d.dataset)} · ${d.n_passive} passive · ${esc(d.tls)} · slot ${d.slot}${d.rtt_relay_ms !== null && d.rtt_relay_ms !== undefined ? ` · relay ${d.rtt_relay_ms} ms` : ""}${d.lab_mode ? " · LAB" : ""}</span>
      ${d.running ? `<button class="btn mini danger" data-dstop="${esc(d.name)}">Stop all</button>` : ""}</span></div>
      <div class="table-wrap"><table><thead><tr><th>Node</th><th>Role</th><th>State</th><th class="num">PID</th><th class="num">gRPC</th><th>Dashboard</th><th></th></tr></thead><tbody>
      ${(d.nodes || []).map((n) => `<tr><td class="mono"><b>${esc(n.name)}</b></td><td><span class="pill role-${esc(n.role)}">${esc(n.role)}</span></td>
        <td>${status(n.state)}</td><td class="num">${n.pid || "—"}</td><td class="num">${n.grpc_port || "—"}</td>
        <td><a href="${esc(safeUrl(n.dashboard))}" target="_blank" rel="noopener">:${n.ui_port} ↗</a></td>
        <td><button class="btn mini" data-log="${esc(d.name)}|${esc(n.name)}">log</button><button class="btn mini" data-restart="${esc(d.name)}|${esc(n.name)}">restart</button><button class="btn mini danger" data-nstop="${esc(d.name)}|${esc(n.name)}">stop</button></td></tr>`).join("")}
      </tbody></table></div>
      <p class="muted">Registered at the coordinator: ${d.registered ? d.registered.map((p) => `${esc(p.party_id)} (${p.alive ? esc(p.status) : "dead"})`).join(", ") || "none yet" : "coordinator not reachable yet"}</p></section>`
    : `<section class="card"><div class="card-head"><h2>${esc(d.name)} <span class="muted">multi-machine</span></h2><span class="muted">${esc(d.coordinator_url)}</span></div>
      <div class="table-wrap"><table><thead><tr><th>Machine</th><th>Host</th><th>Run this there</th><th>Bundle</th><th>Registered</th></tr></thead><tbody>
      ${d.machines.map((m) => { const reg = (d.registered || []).find((p) => p.party_id === m.name); return `<tr><td class="mono"><b>${esc(m.name)}</b></td><td class="mono">${esc(m.host)}</td>
        <td><code class="cmd">${esc(m.cmd)}</code></td><td><a href="/api/deploy/${esc(d.name)}/bundles/${esc(m.name)}.zip${TOKEN ? `?token=${encodeURIComponent(TOKEN)}` : ""}">zip</a><br><span class="muted">${esc(m.contents.join(", "))}</span></td>
        <td>${m.name === "coordinator" ? (d.registered ? status("live") : status("starting")) : reg ? status(reg.alive ? "live" : "stopped") : `<span class="muted">waiting</span>`}</td></tr>`; }).join("")}
      </tbody></table></div><p class="muted">Unzip each bundle into the vfl-system folder on that machine and run its command. Nodes appear here as they register.</p></section>`).join("");
  const act = async (url) => { try { await post(url); refreshDeployments(); } catch (e) { alert(e.message); } };
  $$("[data-dstop]").forEach((b) => (b.onclick = () => act(`/api/deploy/${b.dataset.dstop}/stop`)));
  $$("[data-restart]").forEach((b) => (b.onclick = () => { const [d, n] = b.dataset.restart.split("|"); act(`/api/deploy/${d}/nodes/${n}/restart`); }));
  $$("[data-nstop]").forEach((b) => (b.onclick = () => { const [d, n] = b.dataset.nstop.split("|"); act(`/api/deploy/${d}/nodes/${n}/stop`); }));
  $$("[data-log]").forEach((b) => (b.onclick = () => { DEP.log = b.dataset.log.split("|"); refreshLog(); }));
  refreshLog();
}
async function refreshLog() {
  if (!DEP.log) return;
  const [d, n] = DEP.log;
  try { const r = await get(`/api/deploy/${d}/logs/${n}?n=300`); $("#log-card").classList.remove("hidden"); $("#log-title").textContent = `${d} / ${n}`; const b = $("#log-box"); const atEnd = b.scrollTop + b.clientHeight >= b.scrollHeight - 10; b.textContent = r.log; if (atEnd) b.scrollTop = b.scrollHeight; }
  catch (e) { $("#log-box").textContent = e.message; }
}

/* ======================================================================== TRAINING / SESSIONS */
const TR = { coord: null, form: null, schema: null, state: null };
PAGES.training = async (root) => {
  root.innerHTML = `<p class="page-intro">The session form from the config schema (every key the code reads, dropdowns only), sent to the chosen coordinator's own API.</p>
  <div class="row" style="margin-bottom:12px"><label class="fld"><span class="lbl">coordinator</span><select id="tr-coord"></select></label><span class="muted" id="tr-links"></span></div>
  <div class="grid two">
    <section class="card"><div class="card-head"><h2>Start a session</h2><span class="muted" id="tr-meta"></span></div>
      <form id="tr-form" class="form" autocomplete="off"><div id="tr-cfg" class="cfg-form"></div>
      <div class="sticky-actions"><button class="btn primary" type="submit" id="tr-start">Start session</button><button class="btn danger" type="button" id="tr-stop">Stop current</button>
        <button class="btn" type="button" id="tr-reset">Reset to defaults</button><button class="btn" type="button" id="tr-prev">Preview</button><span class="form-msg" id="tr-msg"></span></div>
      <pre class="cfg-preview hidden" id="tr-pre"></pre></form></section>
    <div>
      <section class="card"><div class="card-head"><h2>Sessions</h2><span class="muted" id="tr-cur"></span></div>
        <div class="table-wrap"><table id="tr-sess"><thead><tr><th>Session</th><th>State</th><th class="num">Epochs</th><th class="num">Test AUROC</th><th></th></tr></thead><tbody></tbody></table></div>
        <pre class="logbox hidden" id="tr-detail"></pre></section>
      <section class="card"><div class="card-head"><h2>Baselines</h2><span class="muted"><code>vfl.cli baseline</code> with this form's config</span></div>
        <div class="row"><select id="bl-ds"></select>${chipsHTML("bl-which", ["b0", "b1", "local_vfl"], ["b0", "b1", "local_vfl"])}<button class="btn" id="bl-go">Run</button><span class="form-msg" id="bl-msg"></span></div><div id="bl-job"></div></section>
      <section class="card"><div class="card-head"><h2>Join token (pki)</h2><span class="muted"><code>vfl.cli token create</code></span></div>
        <div class="row">${selectHTML("tk-role", ["passive", "active"], "passive")}${selectHTML("tk-ttl", [{ value: 900, label: "15 min" }, { value: 3600, label: "1 h" }, { value: 86400, label: "24 h" }], 3600)}<button class="btn" id="tk-go">Create</button></div><pre class="logbox hidden" id="tk-out"></pre></section>
    </div>
  </div>`;
  const cs = await get("/api/train/coordinators");
  $("#tr-coord").innerHTML = cs.length ? cs.map((c) => opt(c.name, `${c.name} — ${c.url}`, c.name === TR.coord)).join("") : opt(null, "no coordinator: launch a deployment");
  TR.coord = cs.length ? (cs.find((c) => c.name === TR.coord) || cs[0]).name : null;
  $("#tr-coord").onchange = () => { TR.coord = val($("#tr-coord")); PAGES.training(root); };
  const ds = await get("/api/prep/datasets");
  $("#bl-ds").innerHTML = ds.length ? ds.map((d) => opt(d.name, d.name)).join("") : opt(null, "prepare a dataset first");
  TR.form = VFLForm.mount($("#tr-cfg"), { onResume: trResume, onRender: ({ changed }) => { $("#tr-meta").textContent = changed ? `${changed} changed from defaults` : "coordinator defaults"; } });
  const q = TR.coord ? `?coord=${encodeURIComponent(TR.coord)}` : "";
  try { TR.schema = await get(`/api/train/schema${q}`); TR.form.load(TR.schema, TR.schema.current); }
  catch (e) { $("#tr-cfg").innerHTML = `<p class="form-msg err">${esc(e.message)}</p>`; }
  if (TR.schema && TR.schema.offline) msg($("#tr-msg"), "coordinator offline: showing configs/session.yaml defaults");
  $("#tr-form").onsubmit = async (e) => { e.preventDefault(); try { const r = await post("/api/train/start", { coord: TR.coord, ...TR.form.payload() }); msg($("#tr-msg"), `started ${r.session_id}`, "ok"); } catch (x) { msg($("#tr-msg"), x.message, "err"); } };
  $("#tr-stop").onclick = async () => { const cur = TR.state && TR.state.current; if (!cur) return; try { await post("/api/train/stop", { coord: TR.coord, session: cur.session_id }); msg($("#tr-msg"), "stop requested", "ok"); } catch (x) { msg($("#tr-msg"), x.message, "err"); } };
  $("#tr-reset").onclick = () => TR.form.reset();
  $("#tr-prev").onclick = async () => { try { const r = await post("/api/train/validate", { coord: TR.coord, ...TR.form.payload() }); $("#tr-pre").textContent = JSON.stringify(r.config, null, 2); $("#tr-pre").classList.remove("hidden"); msg($("#tr-msg"), "valid", "ok"); } catch (x) { msg($("#tr-msg"), x.message, "err"); } };
  $("#bl-go").onclick = async () => { try { const r = await post("/api/train/baseline", { dataset: val($("#bl-ds")), which: chipsVal($("[data-multi=bl-which]")).join(","), form: TR.form.values() });
    watch(`/api/ws-jobs/${r.job}`, $("#bl-job"), { onDone: (j) => { if (j.result) msg($("#bl-msg"), ["b0_centralized", "b1_active_only", "local_vfl"].filter((k) => j.result[k]).map((k) => `${k} ${fmt(j.result[k].final_test.auroc, 4)}`).join(" · "), "ok"); } }); }
    catch (x) { msg($("#bl-msg"), x.message, "err"); } };
  $("#tk-go").onclick = async () => { try { const r = await post("/api/train/token", { coord: TR.coord, role: val($("[name=tk-role]")), ttl_s: val($("[name=tk-ttl]")) }); $("#tk-out").textContent = r.token || JSON.stringify(r); $("#tk-out").classList.remove("hidden"); } catch (x) { $("#tk-out").textContent = x.message; $("#tk-out").classList.remove("hidden"); } };
  every(2000, refreshTraining);
};
async function trResume(sid) {
  try { const d = await get(`/api/train/sessions/${sid}?coord=${encodeURIComponent(TR.coord)}`); TR.form.setValues(d.config, ["resume_from"]); msg($("#tr-msg"), `loaded ${sid}'s config`, "ok"); }
  catch (e) { msg($("#tr-msg"), e.message, "err"); }
}
async function refreshTraining() {
  if (PAGE.cur !== "training" || !TR.coord) return;
  try { TR.state = await get(`/api/train/state?coord=${encodeURIComponent(TR.coord)}`); } catch (e) { $("#tr-cur").textContent = e.message; return; }
  const S = TR.state;
  const parties = S.parties.filter((p) => p.alive);
  const act = parties.find((p) => p.role === "active");
  if (TR.form.loaded) TR.form.updateContext({ parties, sessions: S.sessions.filter((s) => s.completed_epochs > 0 && s.completed_epochs < s.epochs),
    datasets: [...new Set([...(TR.schema.datasets || []), ...S.parties.map((p) => ((p.summary || {}).partition || {}).dataset).filter(Boolean)])], activeRows: act && act.summary ? act.summary.rows : null });
  const cur = S.current;
  $("#tr-cur").textContent = cur ? `${cur.session_id}: ${cur.state} ${cur.completed_epochs}/${cur.epochs}` : "idle";
  $("#tr-start").disabled = !!(cur && ["STARTING", "RUNNING", "STOPPING"].includes(cur.state));
  const coordUrl = (await get("/api/train/coordinators")).find((c) => c.name === TR.coord);
  $("#tr-links").innerHTML = `${coordUrl ? `<a href="${esc(safeUrl(coordUrl.url))}" target="_blank" rel="noopener">coordinator dashboard ↗</a>` : ""} ${S.parties.filter((p) => p.ui_url).map((p) => `<a href="${esc(safeUrl(p.ui_url))}" target="_blank" rel="noopener">${esc(p.party_id)} ↗</a>`).join(" ")}`;
  $("#tr-sess tbody").innerHTML = S.sessions.length ? S.sessions.slice(0, 25).map((s) => `<tr><td class="mono">${esc(s.session_id)}<br><span class="muted">${esc(s.name || "")}</span></td>
    <td>${status(String(s.state).toLowerCase() === "completed" ? "done" : String(s.state).toLowerCase())}</td><td class="num">${s.completed_epochs}/${s.epochs}</td>
    <td class="num">${fmt(s.last_test_auroc, 4)}</td><td><button class="btn mini" data-sd="${esc(s.session_id)}">detail</button></td></tr>`).join("")
    : `<tr><td colspan="5" class="empty">No sessions yet.</td></tr>`;
  $$("[data-sd]").forEach((b) => (b.onclick = async () => { const d = await get(`/api/train/sessions/${b.dataset.sd}?coord=${encodeURIComponent(TR.coord)}`);
    const em = d.epoch_metrics || []; $("#tr-detail").textContent = JSON.stringify({ state: d.state, config: d.config, last_epoch: em[em.length - 1], errors: d.errors }, null, 2); $("#tr-detail").classList.remove("hidden"); }));
}

/* ======================================================================== SECURITY & ATTACKS (item 1) */
PAGES.attacks = async (root) => {
  const A = await get("/api/attacks");
  root.innerHTML = `<p class="page-intro">The standalone attack simulator against your own lab deployment. It runs exactly
    <code>vfl.cli attack … --lab-mode</code>, the same code the CLI and the coordinator's Security tab use, so all three give the same verdicts.</p>
  ${!A.lab_mode ? `<div class="note">The UI was started without <code>--lab-mode</code>: viewing only. Use <code>vfl.cli attack-ui --token T</code> to launch attacks.</div>` : ""}
  ${!A.token_configured ? `<div class="note">No operator token configured: attack runs are refused. Start the UI with <code>--token</code>.</div>` : ""}
  <div class="grid two">
    <section class="card"><div class="card-head"><h2>Run the conformance suite</h2><span class="badge warn">lab only</span></div>
      <div class="fields">
        <label class="fld"><span class="lbl">target deployment</span>${selectHTML("a-t", A.targets.length ? A.targets.map((t) => ({ value: t.name, label: `${t.name} (${t.coordinator})${t.running ? "" : " – stopped"}` })) : [{ value: null, label: "launch a deployment first" }], A.targets[0] && A.targets[0].name)}</label>
        <label class="fld"><span class="lbl">attack</span>${selectHTML("a-c", [{ value: "all", label: "all cases" }, ...Object.keys(A.families).map((f) => ({ value: f, label: `family: ${f}` })), ...A.cases], "all")}</label>
        <label class="fld"><span class="lbl">session (protocol cases)</span><select name="a-s"><option value="null">— none —</option></select></label>
        <label class="fld"><span class="lbl">data (inference cases)</span>${selectHTML("a-d", [{ value: null, label: "— none (skipped) —" }, ...A.datasets], null)}</label>
        <label class="fld wide"><label class="chip"><input type="checkbox" id="a-ack"> I confirm this targets our own lab deployment (--lab-mode)</label></label></div>
      <div class="sticky-actions"><button class="btn primary" id="a-go"${A.lab_mode ? "" : " disabled"}>Run</button><span class="form-msg" id="a-msg"></span></div>
      <div class="note">Targets are limited to the hosts in the deployment's <code>deployment.json</code>. Inference cases measure what an <b>authorised</b> party can learn from tensors it legitimately receives: mTLS does not stop that. Only <code>label_rr_epsilon</code> is a formal guarantee.</div></section>
    <section class="card"><div class="card-head"><h2>Runs</h2></div><div class="table-wrap"><table id="a-jobs"><thead><tr><th>Attack</th><th>When</th><th>Status</th><th>Verdict</th><th>Report</th></tr></thead><tbody></tbody></table></div></section>
  </div>
  <section class="card"><div class="card-head"><h2 id="a-rep-title">Latest report</h2><span id="a-badge"></span></div><div id="a-rep"><p class="muted">No finished run yet.</p></div></section>`;
  const loadSessions = async () => {
    const t = val($("[name=a-t]"));
    if (!t) return;
    try { const S = await get(`/api/train/state?coord=${encodeURIComponent(t)}`); $("[name=a-s]").innerHTML = opt(null, "— none —") + S.sessions.map((s) => opt(s.session_id, `${s.session_id} (${s.state})`)).join(""); } catch { /* coordinator down */ }
  };
  $("[name=a-t]").onchange = loadSessions; loadSessions();
  $("#a-go").onclick = async () => {
    try { const r = await post("/api/attacks/run", { target: val($("[name=a-t]")), case: val($("[name=a-c]")), session: val($("[name=a-s]")), dataset: val($("[name=a-d]")), lab_mode_ack: $("#a-ack").checked });
      msg($("#a-msg"), `running ${r.name}…`, "ok"); } catch (e) { msg($("#a-msg"), e.message, "err"); }
  };
  every(1500, async () => {
    if (PAGE.cur !== "attacks") return;
    const jobs = (await get("/api/attacks")).jobs;
    const tq = TOKEN ? `?token=${encodeURIComponent(TOKEN)}` : "";
    $("#a-jobs tbody").innerHTML = jobs.length ? jobs.map((j) => `<tr><td>${esc(j.name)}</td><td class="mono">${ago(j.t)}</td><td>${status(j.status)}</td>
      <td>${j.summary ? `${status(j.summary.all_as_expected ? "pass" : "fail")} ${j.summary.as_expected}/${j.summary.cases}` : "—"}</td>
      <td>${j.summary ? `<a href="/api/attacks/jobs/${esc(j.id)}/report.json${tq}">json</a> · <a href="/api/attacks/jobs/${esc(j.id)}/report.csv${tq}">csv</a>` : j.status === "running" ? `<button class="btn mini" data-cancel="/api/attacks/jobs/${esc(j.id)}">cancel</button>` : ""}</td></tr>`).join("")
      : `<tr><td colspan="5" class="empty">No runs yet.</td></tr>`;
    const last = jobs.find((j) => j.rows && j.rows.length);
    if (last) {
      $("#a-rep-title").textContent = `Latest report: ${last.name}`;
      $("#a-badge").innerHTML = status(last.summary.all_as_expected ? "pass" : "fail");
      $("#a-rep").innerHTML = `<div class="table-wrap"><table><thead><tr><th>Attack</th><th>Layer</th><th>Expected</th><th>Observed</th><th>Evidence</th><th>Detail</th></tr></thead><tbody>
        ${last.rows.map((r) => `<tr><td>${esc(r.attack)}</td><td class="muted">${esc(r.layer)}</td><td>${esc(r.expected)}</td>
          <td><span class="dot ${r.as_expected ? "ok" : "bad"}"></span> ${esc(r.observed)}</td><td class="muted">${esc(r.evidence)}</td><td class="muted" style="white-space:normal;max-width:420px">${esc((r.detail || "").slice(0, 220))}</td></tr>`).join("")}
        </tbody></table></div><p class="muted">Files: <code>${esc(last.out)}</code> (report.json, report.csv, summary.txt, security.jsonl)</p>`;
    }
  });
};

/* ======================================================================== TESTS & BENCHMARKS (item 5) */
PAGES.tests = async (root) => {
  const [targets, B, ds, deps] = await Promise.all([get("/api/tests/targets"), get("/api/bench/options"), get("/api/prep/datasets"), get("/api/deploy")]);
  root.innerHTML = `<div class="grid two">
    <section class="card"><div class="card-head"><h2>Tests</h2><span class="muted">pytest from the UI</span></div>
      <div class="row">${selectHTML("t-target", targets, "not-slow")}<button class="btn primary" id="t-go">Run pytest</button><span class="form-msg" id="t-msg"></span></div>
      <div id="t-job"></div><div id="t-table"></div></section>
    <section class="card"><div class="card-head"><h2>Benchmark launcher</h2><span class="muted">bench/run_benchmarks.py · queued, one at a time</span></div>
      <div class="fields" id="b-fields"></div>
      <div class="sticky-actions"><button class="btn primary" id="b-go">Add to queue</button><span class="form-msg" id="b-msg"></span></div></section>
  </div>
  <section class="card"><div class="card-head"><h2>Benchmark queue</h2></div><div class="table-wrap"><table id="b-q"><thead><tr><th>Run</th><th>Status</th><th>Progress</th><th>Queued</th><th></th></tr></thead><tbody></tbody></table></div>
    <pre class="logbox hidden" id="b-log"></pre></section>`;
  const O = B.options;
  const field = (k, o) => {
    let w;
    if (o.widget === "multi") w = chipsHTML(k, o.options, o.default);
    else if (o.widget === "toggle") w = `<label class="chip"><input type="checkbox" data-b="${k}"> ${o.flag}</label>`;
    else if (o.widget === "dataset") w = selectHTML(k, [{ value: null, label: "generate (default)" }, ...ds.map((d) => d.name)], null, `data-b="${k}"`);
    else if (o.widget === "config") w = selectHTML(k, [{ value: null, label: "configs/session.yaml" }, ...B.configs], null, `data-b="${k}"`);
    else if (o.widget === "coordinator") w = selectHTML(k, [{ value: null, label: "local (own deployment)" }, ...deps.deployments.filter((d) => d.coordinator_url).map((d) => ({ value: d.coordinator_url, label: `${d.name}: ${d.coordinator_url}` }))], null, `data-b="${k}"`);
    else w = selectHTML(k, o.options.map((x) => (x === null ? { value: null, label: "default" } : x)), o.default, `data-b="${k}"`);
    return `<label class="fld${o.widget === "multi" ? " wide" : ""}"><span class="lbl">${esc(o.flag)}</span>${w}${o.help ? `<span class="hlp">${esc(o.help)}</span>` : ""}</label>`;
  };
  $("#b-fields").innerHTML = field("suites", O.suites) + `<ul class="suite-help wide">${B.suites.map((s) => `<li><code>${s}</code> ${esc(B.suite_help[s])}</li>`).join("")}</ul>`
    + Object.entries(O).filter(([k]) => k !== "suites").map(([k, o]) => field(k, o)).join("");
  $("#t-go").onclick = async () => {
    try { const r = await post("/api/tests/run", { target: val($("[name=t-target]")) }); msg($("#t-msg"), "running…");
      watch(`/api/tests/jobs/${r.job}`, $("#t-job"), { interval: 700, onDone: (j) => { msg($("#t-msg"), j.status === "done" ? "all passed" : j.status, j.status === "done" ? "ok" : "err"); renderJunit(j.result); } }); }
    catch (e) { msg($("#t-msg"), e.message, "err"); }
  };
  $("#b-go").onclick = async () => {
    const spec = { name: `bench-${new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "")}` };
    for (const [k, o] of Object.entries(O)) {
      if (o.widget === "multi") spec[k] = chipsVal($(`[data-multi=${k}]`));
      else if (o.widget === "toggle") spec[k] = $(`[data-b=${k}]`).checked;
      else { const v = val($(`[data-b=${k}]`)); if (v !== null && J(v) !== J(o.default)) spec[k] = v; }
    }
    try { const r = await post("/api/launch", spec); msg($("#b-msg"), `queued ${r.out}`, "ok"); refreshQueue(); } catch (e) { msg($("#b-msg"), e.message, "err"); }
  };
  every(2000, refreshQueue);
};
function renderJunit(res) {
  if (!res) return;
  const rows = [...res.tests].sort((a, b) => (a.outcome === "passed") - (b.outcome === "passed"));
  $("#t-table").innerHTML = `<p>${Object.entries(res.counts).map(([k, v]) => `${status(k === "passed" ? "pass" : k === "skipped" ? "stopped" : "fail")} ${k} ${v}`).join(" ")}</p>
    <div class="table-wrap" style="max-height:340px;overflow:auto"><table><thead><tr><th>Test</th><th>File</th><th>Outcome</th><th class="num">s</th></tr></thead><tbody>
    ${rows.map((r) => `<tr><td class="mono" title="${esc(r.message)}">${esc(r.test)}</td><td class="muted">${esc(r.file)}</td><td>${status(r.outcome === "passed" ? "pass" : r.outcome === "skipped" ? "stopped" : "fail")} ${esc(r.outcome)}${r.message ? `<br><span class="muted" style="white-space:normal">${esc(r.message.slice(0, 160))}</span>` : ""}</td><td class="num">${fmt(r.time_s, 2)}</td></tr>`).join("")}
    </tbody></table></div>`;
}
async function refreshQueue() {
  if (PAGE.cur !== "tests") return;
  const q = await get("/api/jobs");
  $("#b-q tbody").innerHTML = q.length ? q.map((j) => `<tr><td class="mono">${esc(j.label)}</td><td>${status(j.status)}</td><td>${j.progress ? `${j.progress.sessions_done} sessions done` : "—"}</td>
    <td class="mono">${ago(j.queued)}</td><td><button class="btn mini" data-blog="${esc(j.id)}">log</button>
    ${["running", "queued"].includes(j.status) ? `<button class="btn mini danger" data-bcancel="${esc(j.id)}">cancel</button>` : `<button class="btn mini" data-brerun="${esc(j.id)}">rerun</button>`}</td></tr>`).join("")
    : `<tr><td colspan="5" class="empty">Queue empty.</td></tr>`;
  $$("[data-bcancel]").forEach((b) => (b.onclick = () => post(`/api/jobs/${b.dataset.bcancel}/cancel`).then(refreshQueue)));
  $$("[data-brerun]").forEach((b) => (b.onclick = () => post(`/api/jobs/${b.dataset.brerun}/rerun`).then(refreshQueue)));
  $$("[data-blog]").forEach((b) => (b.onclick = () => { const j = q.find((x) => x.id === b.dataset.blog); $("#b-log").textContent = j.log || "(no output yet)"; $("#b-log").classList.remove("hidden"); }));
}

/* ======================================================================== ANALYTICS (item 5) */
const AN = { bench: null, filters: [], group: "bottom_model" };
PAGES.analytics = async (root) => {
  const [benches, cat, keys, runs] = await Promise.all([get("/api/benches"), get("/api/figures"), get("/api/configkeys"), get("/api/runs")]);
  AN.bench = benches.includes(AN.bench) ? AN.bench : benches[benches.length - 1];
  root.innerHTML = `<p class="page-intro">Every figure comes from <code>bench/analyze.py</code>'s figure registry, the same functions that build <code>report.md</code>.
    Without filters you see exactly the report's files; with a filter or grouping key the same function is re-run on the subset.</p>
  <section class="card"><div class="row">
    <label class="fld"><span class="lbl">benchmark</span>${selectHTML("an-b", benches.length ? benches : [{ value: null, label: "no results yet: run a benchmark" }], AN.bench)}</label>
    <label class="fld"><span class="lbl">group by (comparison figure + runs table)</span>${selectHTML("an-g", Object.keys(keys), AN.group)}</label>
    <div class="fld"><span class="lbl">filters</span><div class="row" id="an-f"></div></div>
    <button class="btn mini" id="an-add">+ filter</button></div></section>
  <section class="card"><div class="card-head"><h2>Runs grouped</h2><span class="muted">mean ± std, 95% CI over seeds; ⚠ = std/mean &gt; 5% (NFR-2)</span></div><div class="table-wrap"><table id="an-runs"></table></div></section>
  <section class="card"><div class="card-head"><h2>Compare two runs</h2></div>
    <div class="row">${selectHTML("cmp-a", runs.map((r) => ({ value: `${r.bench}:${r.run_id}`, label: `${r.bench} · ${r.suite} · ${r.label || r.run_id}` })), runs[0] && `${runs[0].bench}:${runs[0].run_id}`)}
    ${selectHTML("cmp-b", runs.map((r) => ({ value: `${r.bench}:${r.run_id}`, label: `${r.bench} · ${r.suite} · ${r.label || r.run_id}` })), runs[1] && `${runs[1].bench}:${runs[1].run_id}`)}<button class="btn" id="cmp-go">Compare</button></div><div id="cmp-out"></div></section>
  <h3 class="sub">Model performance</h3><div class="gallery" id="g-model"></div>
  <h3 class="sub">Systems performance</h3><div class="gallery" id="g-systems"></div>`;
  const drawFilters = () => {
    $("#an-f").innerHTML = AN.filters.map((f, i) => `<span class="row">${selectHTML(`fk${i}`, Object.keys(keys), f.k, `data-fk="${i}"`)}${selectHTML(`fv${i}`, keys[f.k] || [], f.v, `data-fv="${i}"`)}<button class="btn mini" data-fx="${i}">✕</button></span>`).join("") || `<span class="muted">none</span>`;
    $$("[data-fk]").forEach((s) => (s.onchange = () => { const i = +s.dataset.fk; AN.filters[i] = { k: val(s), v: (keys[val(s)] || [])[0] }; drawFilters(); draw(); }));
    $$("[data-fv]").forEach((s) => (s.onchange = () => { AN.filters[+s.dataset.fv].v = val(s); draw(); }));
    $$("[data-fx]").forEach((b) => (b.onclick = () => { AN.filters.splice(+b.dataset.fx, 1); drawFilters(); draw(); }));
  };
  $("#an-add").onclick = () => { const k = Object.keys(keys)[0]; AN.filters.push({ k, v: keys[k][0] }); drawFilters(); draw(); };
  $("[name=an-b]").onchange = () => { AN.bench = val($("[name=an-b]")); draw(); };
  $("[name=an-g]").onchange = () => { AN.group = val($("[name=an-g]")); draw(); };
  $("#cmp-go").onclick = async () => {
    try { const r = await get(`/api/compare?a=${encodeURIComponent(val($("[name=cmp-a]")))}&b=${encodeURIComponent(val($("[name=cmp-b]")))}`);
      $("#cmp-out").innerHTML = `<div class="grid two"><div><h3 class="sub">Config differences</h3><table><tbody>${r.config_diff.map((d) => `<tr><td class="mono">${esc(d[0])}</td><td>${esc(J(d[1]))}</td><td>${esc(J(d[2]))}</td></tr>`).join("") || `<tr><td class="empty">identical configs</td></tr>`}</tbody></table>
        <h3 class="sub">Metrics</h3><table><tbody>${r.metrics.map((m) => `<tr><td class="mono">${esc(m[0])}</td><td class="num">${fmt(m[1], 4)}</td><td class="num">${fmt(m[2], 4)}</td></tr>`).join("")}</tbody></table></div>
        ${r.figure && r.figure.png ? `<figure class="fig"><img src="/api/results-file/${esc(r.figure.png)}?t=${Date.now()}" alt="comparison"><figcaption><span>AUROC and loss vs epoch</span><span><a href="/api/results-file/${esc(r.figure.svg)}">SVG</a> · <a href="/api/results-file/${esc(r.figure.csv)}">CSV</a></span></figcaption></figure>` : ""}</div>`; }
    catch (e) { $("#cmp-out").innerHTML = `<p class="form-msg err">${esc(e.message)}</p>`; }
  };
  const draw = async () => {
    const qs = new URLSearchParams();
    AN.filters.forEach((f) => qs.append(`f.${f.k}`, f.v));
    const grouped = await get(`/api/grouped?group=${encodeURIComponent(AN.group)}&${qs}`);
    $("#an-runs").innerHTML = `<thead><tr><th>${esc(AN.group)}</th><th class="num">runs</th><th>seeds</th><th class="num">AUROC mean</th><th class="num">± std</th><th class="num">95% CI</th></tr></thead><tbody>
      ${grouped.map((g) => `<tr><td class="mono">${esc(J(g.group[AN.group]))}</td><td class="num">${g.n_runs}</td><td class="mono">${esc(g.seeds.join(","))}</td><td class="num">${fmt(g.auroc_mean, 4)}${g.high_variance ? " ⚠" : ""}</td><td class="num">${fmt(g.auroc_std, 4)}</td><td class="num">${g.auroc_ci95 !== undefined ? "±" + fmt(g.auroc_ci95, 4) : "—"}</td></tr>`).join("") || `<tr><td colspan="6" class="empty">No runs match.</td></tr>`}</tbody>`;
    if (!AN.bench) return;
    const fq = new URLSearchParams(qs);
    if (AN.group) fq.set("group", AN.group);
    for (const grp of ["model", "systems"]) {
      $(`#g-${grp}`).innerHTML = cat.filter((c) => c.group === grp).map((c) => {
        const q = c.ctx_keys.includes("group") || AN.filters.length ? `?${c.ctx_keys.includes("group") ? fq : qs}` : "";
        const base = `/api/figure/${encodeURIComponent(AN.bench)}/${c.name}`;
        return `<figure class="fig" data-fig="${c.name}"><img loading="lazy" src="${base}.png${q}" alt="${esc(c.title)}" onerror="this.outerHTML='<div class=nodata>no data for this figure in this benchmark</div>'">
          <figcaption><span>${esc(c.title)}</span><span><a href="${base}.png${q}${q ? "&" : "?"}download=1">PNG</a> · <a href="${base}.svg${q}">SVG</a> · <a href="${base}.csv${q}${q ? "&" : "?"}download=1">CSV</a></span></figcaption></figure>`;
      }).join("");
    }
  };
  drawFilters(); draw();
};

/* ======================================================================== REPORTS */
PAGES.reports = async (root) => {
  const [benches, cli, deps] = await Promise.all([get("/api/benches"), get("/api/cli-map"), get("/api/deploy")]);
  const b0 = benches[benches.length - 1];
  root.innerHTML = `<div class="grid two">
    <section class="card"><div class="card-head"><h2>NFR scorecard</h2>${selectHTML("r-b", benches.length ? benches : [{ value: null, label: "no results yet" }], b0)}</div>
      <div id="r-score"></div><div class="row" id="r-links"></div></section>
    <section class="card"><div class="card-head"><h2>Node dashboards</h2></div><div id="r-nodes"></div>
      <h3 class="sub">Dashboard users</h3><p class="muted"><code>vfl.cli ui-user add</code>: writes ui_users.json in the workspace (scrypt-hashed).</p>
      <div class="row"><input name="u-name" placeholder="user" style="width:120px"><input name="u-pw" type="password" placeholder="password (8+)" style="width:150px">${selectHTML("u-role", ["viewer", "operator"], "viewer")}<button class="btn" id="u-go">Add user</button><span class="form-msg" id="u-msg"></span></div></section>
  </div>
  <section class="card"><div class="card-head"><h2>Every CLI command in the UI</h2><span class="muted">checked by tests/test_webui.py</span></div>
    <div class="table-wrap"><table><thead><tr><th>CLI</th><th>UI page</th><th>API</th></tr></thead><tbody>
    ${Object.entries(cli).map(([k, v]) => `<tr><td class="mono">vfl.cli ${esc(k)}</td><td><a href="#${esc(v.page)}">${esc(v.page)}</a></td><td class="mono muted">${esc(v.method)} ${esc(v.route)}</td></tr>`).join("")}</tbody></table></div></section>`;
  $("#r-nodes").innerHTML = deps.deployments.length ? deps.deployments.map((d) => `<p><b>${esc(d.name)}</b>: ${(d.nodes || d.machines || []).map((n) => `<a href="${esc(safeUrl(n.dashboard || `http://${n.host}:${n.ui}`))}" target="_blank" rel="noopener">${esc(n.name)} ↗</a>`).join(" · ")}</p>`).join("") : `<p class="muted">No deployments. The coordinator dashboard is on :8080 and party dashboards on :8081+ by default.</p>`;
  $("#u-go").onclick = async () => { try { const r = await post("/api/admin/ui-user", { name: $("[name=u-name]").value.trim(), password: $("[name=u-pw]").value, role: val($("[name=u-role]")) }); msg($("#u-msg"), r.hint, "ok"); $("[name=u-pw]").value = ""; } catch (e) { msg($("#u-msg"), e.message, "err"); } };
  const draw = async () => {
    const b = val($("[name=r-b]"));
    if (!b) { $("#r-score").innerHTML = `<p class="muted">Run a benchmark in Tests &amp; Benchmarks; its NFR scorecard appears here.</p>`; return; }
    const s = await get(`/api/scorecard/${encodeURIComponent(b)}`);
    $("#r-score").innerHTML = `<div class="table-wrap"><table><thead><tr><th>NFR</th><th>Name</th><th>Status</th><th>Measured</th><th>Evidence runs</th></tr></thead><tbody>
      ${s.rows.map((r) => `<tr><td class="mono">${esc(r.id)}</td><td>${esc(r.name)}</td><td>${status(r.status === "pass" ? "pass" : r.status === "fail" ? "fail" : "stopped")} <span class="muted">${esc(r.status)}</span></td>
        <td class="muted" style="white-space:normal;max-width:420px">${esc(r.value)}</td>
        <td class="mono" style="white-space:normal">${r.evidence.slice(0, 6).map((sid) => `<a href="/api/results-file/${encodeURIComponent(b)}/sessions/${encodeURIComponent(sid)}.json" target="_blank">${esc(sid.slice(-11))}</a>`).join(" ")}${r.evidence.length > 6 ? ` +${r.evidence.length - 6}` : ""}</td></tr>`).join("")}</tbody></table></div>`;
    $("#r-links").innerHTML = `<a class="btn" href="/report-asset/${encodeURIComponent(b)}/report.md" target="_blank">report.md</a>
      <a class="btn" href="/api/export/${encodeURIComponent(b)}">Download report bundle (.zip)</a><a class="btn" href="#analytics">Figures →</a>`;
  };
  $("[name=r-b]").onchange = draw; draw();
};

boot();
