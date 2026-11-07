/* VFL node console: one frontend for coordinator, active and passive nodes. No external deps. */
"use strict";

const $ = (s) => document.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const PALETTE = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6"];
const color = (i) => css(PALETTE[i % PALETTE.length]);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
// v2.3 (RT-04): links built from data another node supplied (party dashboard URLs) must be http(s); a party
// could register ui_url "javascript:..." and run script in the operator's browser. The server refuses such
// URLs too and the CSP blocks javascript: navigation; this is the third layer.
const safeUrl = (u) => { try { const x = new URL(String(u ?? ""), location.href); return (x.protocol === "http:" || x.protocol === "https:") ? x.href : "#"; } catch { return "#"; } };
// the operator token: taken from ?token= once, kept in sessionStorage, and removed from the address bar so it
// does not stay in history, bookmarks or screenshots (v2.3)
let TOKEN = new URLSearchParams(location.search).get("token");
try {
  if (TOKEN) sessionStorage.setItem("vfl-dash-token", TOKEN); else TOKEN = sessionStorage.getItem("vfl-dash-token");
} catch { /* storage may be blocked */ }
if (new URLSearchParams(location.search).has("token")) {
  const u = new URL(location.href); u.searchParams.delete("token"); history.replaceState(null, "", u.pathname + u.search + u.hash);
}

function fmt(v, d = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (typeof v !== "number") return String(v);
  if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return v.toFixed(d);
}
function bytes(n) {
  if (n === null || n === undefined) return "—";
  const u = ["B", "KiB", "MiB", "GiB"]; let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 2 : 0)} ${u[i]}`;
}
function dur(s) {
  if (s === null || s === undefined) return "—";
  if (s < 60) return `${s.toFixed(0)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}
function pct(arr, q) {
  if (!arr.length) return null;
  const a = [...arr].sort((x, y) => x - y);
  const i = (a.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
  return a[lo] + (a[hi] - a[lo]) * (i - lo);
}
const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);

/* ---------------- tiny line chart ---------------- */
function lineChart(canvas, series, opts = {}) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const pts = series.flatMap((s) => s.points);
  ctx.font = `11px ${css("--mono") || "monospace"}`;
  if (!pts.length) {
    ctx.fillStyle = css("--muted"); ctx.textAlign = "center";
    ctx.fillText(opts.empty || "no data yet", w / 2, h / 2);
    return;
  }
  const pad = { l: 52, r: 12, t: series.length > 1 || opts.legend ? 22 : 10, b: 24 };
  let xmin = Math.min(...pts.map((p) => p[0])), xmax = Math.max(...pts.map((p) => p[0]));
  let ymin = opts.ymin ?? Math.min(...pts.map((p) => p[1])), ymax = opts.ymax ?? Math.max(...pts.map((p) => p[1]));
  if (xmax === xmin) { xmax += 1; xmin -= opts.xint ? 0 : 1; }
  if (ymax === ymin) { ymax += Math.abs(ymax) * 0.1 || 1; ymin -= Math.abs(ymin) * 0.1 || 0; }
  const yr = ymax - ymin; if (opts.ymin === undefined) ymin -= yr * 0.05; if (opts.ymax === undefined) ymax += yr * 0.05;
  const X = (x) => pad.l + ((x - xmin) / (xmax - xmin)) * (w - pad.l - pad.r);
  const Y = (y) => h - pad.b - ((y - ymin) / (ymax - ymin)) * (h - pad.t - pad.b);
  // grid + y ticks
  ctx.strokeStyle = css("--line"); ctx.fillStyle = css("--muted"); ctx.lineWidth = 1;
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const v = ymin + ((ymax - ymin) * i) / 4, y = Y(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillText(Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(Math.abs(v) >= 10 ? 1 : 3), pad.l - 6, y);
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  const nx = Math.min(6, Math.max(2, Math.floor((w - pad.l) / 90)));
  for (let i = 0; i <= nx; i++) {
    const v = xmin + ((xmax - xmin) * i) / nx;
    ctx.fillText(opts.xint ? Math.round(v) : v.toFixed(0), X(v), h - pad.b + 6);
  }
  if (opts.xlabel) { ctx.textAlign = "right"; ctx.fillText(opts.xlabel, w - pad.r, h - 12); }
  // series
  series.forEach((s, i) => {
    const c = s.color || color(i);
    ctx.strokeStyle = c; ctx.fillStyle = c; ctx.lineWidth = 1.6;
    const p = [...s.points].sort((a, b) => a[0] - b[0]);
    if (s.dots || p.length < 30) {
      p.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(X(x), Y(y), 2.5, 0, 2 * Math.PI); ctx.fill(); });
    }
    ctx.beginPath();
    p.forEach(([x, y], j) => (j ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
    ctx.stroke();
  });
  if (series.length > 1 || opts.legend) {
    let x = pad.l; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    series.forEach((s, i) => {
      ctx.fillStyle = s.color || color(i); ctx.fillRect(x, 6, 10, 10);
      ctx.fillStyle = css("--muted"); ctx.fillText(s.name, x + 14, 11);
      x += ctx.measureText(s.name).width + 30;
    });
  }
}

/* ---------------- state + polling ---------------- */
let S = null;
let formInit = false;

async function poll() {
  try {
    const r = await fetch("/api/state", { cache: "no-store", headers: authHdrs() });
    if (r.status === 401) {        // v2.3: a dashboard reachable from the network needs the token for every view
      $("#updated").textContent = "locked";
      $("#status-pill").innerHTML = `<span class="dot warn"></span>open this page with ?token=… (see the node's log)`;
      return;
    }
    const txt = await r.text();
    S = JSON.parse(txt.replace(/\bNaN\b/g, "null").replace(/-?\bInfinity\b/g, "null"));
    render();
    $("#updated").textContent = new Date().toLocaleTimeString();
  } catch (e) {
    $("#updated").textContent = "offline";
    $("#status-pill").innerHTML = `<span class="dot bad"></span>unreachable`;
  }
}

function render() {
  const role = S.role;
  document.documentElement.style.setProperty("--role", css(`--${role === "coordinator" ? "coord" : role}`));
  document.title = `${S.node} · VFL ${role}`;
  $("#role-pill").className = `pill role-${role}`;
  $("#role-pill").textContent = role.toUpperCase();
  $("#tls-pill").innerHTML = S.tls ? `<span class="dot ok"></span>mTLS` : `<span class="dot bad"></span>INSECURE`;
  if (role === "coordinator") renderCoordinator(); else renderParty();
}

function kpis(items, target) {
  $(target || "#kpis").innerHTML = items.map(([l, v, sub]) =>
    `<div class="kpi"><div class="label">${esc(l)}</div><div class="value" title="${esc(v)}">${esc(v)}</div>${sub ? `<div class="sub" title="${esc(sub)}">${esc(sub)}</div>` : ""}</div>`).join("");
}
function statusDot(st) {
  const ok = ["RUNNING", "COMPLETED", "running", "done", "idle"], warn = ["STARTING", "STOPPING", "aligning", "starting", "STOPPED"];
  return ok.includes(st) ? "ok" : warn.includes(st) ? "warn" : "bad";
}

/* ================= coordinator ================= */
let TAB = "overview";
function setupTabs() {
  const nav = $("#coord-tabs");
  if (nav.dataset.wired) return;
  nav.dataset.wired = "1";
  nav.classList.remove("hidden");
  nav.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
    TAB = b.dataset.tab;
    nav.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
    render();
  }));
}
function showPane() {
  for (const id of ["coord-view", "security-view", "inference-view", "identities-view"]) {
    const el = $("#" + id);
    if (el) el.classList.toggle("hidden", el.dataset.pane !== TAB);
  }
  $("#tab-identities").classList.toggle("hidden", !S.pki);
}

function renderCoordinator() {
  setupTabs();
  showPane();
  // security badge: count of succeeded + detected events
  const sec = S.security || {}; const tot = sec.totals || {};
  const flagged = (tot.succeeded || 0) + (tot.detected || 0);
  const badge = $("#sec-badge");
  badge.classList.toggle("hidden", !flagged);
  badge.textContent = flagged;
  badge.className = `tab-badge ${tot.succeeded ? "bad" : "warn"}${flagged ? "" : " hidden"}`;
  if (TAB === "security") return renderSecurity();
  if (TAB === "inference") return renderInference();
  if (TAB === "identities") return renderIdentities();
  $("#node-line").textContent = `${S.node} · uptime ${dur(S.uptime_s)}`;
  const cur = S.current;
  $("#status-pill").innerHTML = cur ? `<span class="dot ${statusDot(cur.state)}"></span>${cur.state}` : `<span class="dot ok"></span>idle`;
  const alive = S.parties.filter((p) => p.alive);
  const lastEp = cur && cur.epoch_metrics.length ? cur.epoch_metrics[cur.epoch_metrics.length - 1] : null;
  const recent = cur ? cur.recent_batches : [];
  const pas = recent.filter((r) => r.role === "passive" && r.phase === "train");
  const lat = pas.slice(-200).map((r) => r.batch_ms);
  const act = recent.filter((r) => r.role === "active" && r.phase === "train");
  let thr = null;
  if (act.length > 5) { const a = act.slice(-100); const dt = a[a.length - 1].t - a[0].t; thr = dt > 0 ? (a.length - 1) / dt : null; }
  const prog = cur ? Object.values(cur.progress || {})[0] || {} : {};
  kpis([
    ["Parties alive", `${alive.length} / ${S.parties.length}`, `${alive.filter((p) => p.role === "active").length} active · ${alive.filter((p) => p.role === "passive").length} passive`],
    ["Session", cur ? cur.state : "none", cur ? cur.session_id : "start one below"],
    ["Epoch", cur ? `${Math.min(cur.completed_epochs + (cur.state === "RUNNING" ? 1 : 0), cur.epochs)} / ${cur.epochs}` : "—", prog.phase ? `${prog.phase} ${prog.batch ?? ""}${prog.n_batches ? "/" + prog.n_batches : ""}` : ""],
    ["Test AUROC", fmt(lastEp?.test?.auroc, 4), lastEp ? `val ${fmt(lastEp?.val?.auroc, 4)} · acc ${fmt(lastEp?.test?.accuracy, 3)}` : ""],
    ["Throughput", thr ? `${fmt(thr, 1)}/s` : "—", "committed batches (recent)"],
    ["Batch p50 / p95", lat.length ? `${fmt(pct(lat, .5), 1)} / ${fmt(pct(lat, .95), 1)}` : "—", "ms, passive view"],
    ["Coordinator", S.resources.rss_mib ? `${fmt(S.resources.rss_mib, 0)} MiB` : "—", `cpu ${fmt(S.resources.cpu_percent, 0)}%`],
  ]);

  // parties
  $("#party-count").textContent = `${alive.length} alive`;
  $("#party-table tbody").innerHTML = S.parties.length ? S.parties.map((p) => {
    const pr = p.progress || {};
    const frac = pr.n_batches ? (pr.batch || 0) / pr.n_batches : 0;
    const d = p.summary || {};
    return `<tr><td><span class="dot ${p.alive ? "ok" : "bad"}"></span> <b>${esc(p.party_id)}</b></td>
      <td><span class="pill role-${p.role}">${p.role}</span></td>
      <td class="mono">${esc(p.hostname || "")}${p.address ? `<br><span class="muted">${esc(p.address)}</span>` : ""}</td>
      <td class="mono">${fmt(d.rows, 0)} × ${fmt(d.n_features, 0)}${d.has_labels ? " + y" : ""}</td>
      <td>${esc(p.status)}${p.alive ? "" : ` <span class="muted">(${fmt(p.age_s, 0)}s ago)</span>`}</td>
      <td>${pr.phase ? `<span class="progress"><span style="width:${(frac * 100).toFixed(1)}%"></span></span> <span class="mono muted">${esc(pr.phase)} e${pr.epoch ?? "-"}</span>` : ""}</td>
      <td>${p.ui_url ? `<a href="${esc(safeUrl(p.ui_url))}" target="_blank" rel="noopener">open ↗</a>` : ""}</td></tr>`;
  }).join("") : `<tr><td colspan="7" class="empty">No parties registered yet. Start the active and passive nodes.</td></tr>`;

  initForm();
  const running = cur && ["STARTING", "RUNNING", "STOPPING"].includes(cur.state);
  $("#start-btn").disabled = !!running || !(FORM && FORM.loaded);
  $("#stop-btn").disabled = !running;
  if (FORM && FORM.loaded) FORM.updateContext(formContext());

  // charts
  const loss = act.map((r, i) => [i, r.loss]).filter((p) => p[1] !== undefined);
  lineChart($("#c-loss"), [{ name: "loss", points: loss.map((p, i) => [i, p[1]]) }], { xlabel: "batch (recent)", xint: true });
  const em = cur ? cur.epoch_metrics : [];
  lineChart($("#c-auc"), [
    { name: "val", points: em.filter((e) => e.val?.auroc != null).map((e) => [e.epoch, e.val.auroc]), dots: true },
    { name: "test", points: em.filter((e) => e.test?.auroc != null).map((e) => [e.epoch, e.test.auroc]), dots: true },
  ], { xint: true, xlabel: "epoch" });
  const byParty = {};
  pas.forEach((r) => (byParty[r.party] = byParty[r.party] || []).push(r));
  lineChart($("#c-lat"), Object.entries(byParty).map(([k, rs]) => ({ name: k, points: rs.slice(-300).map((r, i) => [i, r.batch_ms]) })),
    { xlabel: "batch (recent)", xint: true, ymin: 0, legend: true });
  lineChart($("#c-leak"), [
    { name: "norm attack", points: em.filter((e) => e.leakage?.norm_leak_auc != null).map((e) => [e.epoch, e.leakage.norm_leak_auc]), dots: true },
    { name: "cosine attack", points: em.filter((e) => e.leakage?.cosine_leak_auc != null).map((e) => [e.epoch, e.leakage.cosine_leak_auc]), dots: true },
  ], { xint: true, ymin: 0.4, ymax: 1.0, xlabel: "epoch" });

  // events
  $("#sess-id").textContent = cur ? cur.session_id : "";
  const ev = cur ? cur.events : [];
  const box = $("#events"); const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 10;
  box.innerHTML = ev.length ? ev.map((e) => `<div class="${e.kind === "error" ? "error" : e.kind === "fault" || e.kind === "timeout" ? "warn" : ""}"><span class="t">${esc((e.time || "").slice(11, 19))}</span>${esc(e.msg)}</div>`).join("") : `<div class="muted">No session yet.</div>`;
  if (atBottom) box.scrollTop = box.scrollHeight;

  // history
  $("#hist-table tbody").innerHTML = S.sessions.length ? S.sessions.map((s) =>
    `<tr><td class="mono">${esc(s.session_id)}<br><span class="muted">${esc(s.name || "")}</span></td>
      <td><span class="dot ${statusDot(s.state)}"></span> ${esc(s.state)}</td><td class="num">${s.batch_size ?? ""}</td>
      <td class="num">${fmt(s.inject_rtt_ms, 0)}</td><td class="num">${s.completed_epochs}/${s.epochs}</td>
      <td class="num">${fmt(s.last_test_auroc, 4)}</td>
      <td><a href="/api/sessions/${encodeURIComponent(s.session_id)}" target="_blank">json</a></td></tr>`).join("")
    : `<tr><td colspan="7" class="empty">No sessions yet.</td></tr>`;
}

/* ---- session form: generated from the config schema (config_form.js) ---- */
let FORM = null;
function formContext() {
  const parties = S.parties.filter((p) => p.alive);
  const act = parties.find((p) => p.role === "active");
  const datasets = [...new Set(S.parties.map((p) => ((p.summary || {}).partition || {}).dataset).filter(Boolean))];
  const sessions = S.sessions.filter((s) => s.completed_epochs > 0 && s.completed_epochs < s.epochs);
  return { parties, sessions, datasets, activeRows: act && act.summary ? act.summary.rows : null };
}
async function initForm() {
  if (formInit) return;
  formInit = true;
  const msg = $("#form-msg");
  if (!FORM) {
    FORM = VFLForm.mount($("#cfg-form"), {
      onResume: loadResumeConfig,
      onRender: ({ changed }) => { $("#form-meta").textContent = changed ? `${changed} changed from defaults` : "defaults from session config"; },
    });
    $("#start-form").addEventListener("submit", startSession);
    $("#stop-btn").addEventListener("click", stopSession);
    $("#reset-btn").addEventListener("click", () => { FORM.reset(); msg.className = "form-msg"; msg.textContent = "reset to defaults"; $("#cfg-preview").classList.add("hidden"); });
    $("#preview-btn").addEventListener("click", previewConfig);
  }
  try {
    const r = await fetch("/api/config/schema", { cache: "no-store", headers: authHdrs() });
    const sc = await r.json();
    if (!r.ok) throw new Error(sc.error || r.statusText);
    FORM.load(sc, sc.current);
    FORM.updateContext(formContext());
  } catch (e) {
    formInit = false;
    $("#cfg-form").innerHTML = `<p class="form-msg err">could not load the config schema: ${esc(e.message)}</p>`;
  }
}
async function loadResumeConfig(sid) {
  const msg = $("#form-msg");
  try {
    const r = await fetch(`/api/sessions/${encodeURIComponent(sid)}`, { cache: "no-store", headers: authHdrs() });
    const d = await r.json();
    if (!r.ok || !d.config) throw new Error(d.error || "no config recorded");
    FORM.setValues(d.config, ["resume_from"]);
    msg.className = "form-msg ok"; msg.textContent = `loaded ${sid}'s config; model sizes must match its checkpoint`;
  } catch (e) { msg.className = "form-msg err"; msg.textContent = `could not load ${sid}: ${e.message}`; }
}
async function previewConfig() {
  const msg = $("#form-msg"), pre = $("#cfg-preview");
  try {
    const j = await post("/api/config/validate", FORM.payload());
    pre.textContent = JSON.stringify(j.config, null, 2);
    pre.classList.remove("hidden");
    msg.className = "form-msg ok"; msg.textContent = "valid";
  } catch (e) { pre.classList.add("hidden"); msg.className = "form-msg err"; msg.textContent = e.message; }
}
async function startSession(ev) {
  ev.preventDefault();
  const msg = $("#form-msg");
  msg.className = "form-msg"; msg.textContent = "starting…";
  try {
    const j = await post("/api/sessions", FORM.payload());
    msg.className = "form-msg ok"; msg.textContent = `started ${j.session_id}`;
    poll();
  } catch (e) { msg.className = "form-msg err"; msg.textContent = e.message; }
}
async function stopSession() {
  if (!S.current) return;
  const msg = $("#form-msg");
  const r = await fetch(`/api/sessions/${encodeURIComponent(S.current.session_id)}/stop`, { method: "POST", headers: hdrs() });
  const j = await r.json();
  msg.className = r.ok ? "form-msg ok" : "form-msg err"; msg.textContent = r.ok ? "stop requested" : j.error;
}
function authHdrs() { return TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {}; }
function hdrs() { const h = { "Content-Type": "application/json", "X-VFL-Request": "1" }; if (TOKEN) h.Authorization = `Bearer ${TOKEN}`; return h; }
async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: hdrs(), body: JSON.stringify(body || {}) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

/* ===== security tab ===== */
const OUT_CLS = { blocked: "ok", detected: "warn", succeeded: "bad" };
function renderSecurity() {
  $("#status-pill").innerHTML = `<span class="dot ok"></span>security`;
  const sec = S.security || {}; const tot = sec.totals || {}; const by = sec.by_type || {};
  kpis([
    ["Events", fmt(sec.events || 0, 0), "since start"],
    ["Blocked", fmt(tot.blocked || 0, 0), "requests the gate refused"],
    ["Detected", fmt(tot.detected || 0, 0), "anomalies / join requests"],
    ["Succeeded", fmt(tot.succeeded || 0, 0), tot.succeeded ? "inference or accepted — review" : "none"],
    ["Gate calls", fmt((sec.gate || {}).calls || 0, 0), `${(sec.gate || {}).rate_limited || 0} rate-limited`],
    ["Lab mode", S.lab_mode ? "on" : "off", S.lab_mode ? "attack simulator available" : "read-only"],
  ], "#sec-kpis");
  const layerOf = {};
  (sec.recent || []).forEach((e) => (layerOf[e.attack_type] = e.layer));
  $("#sec-type-table tbody").innerHTML = Object.keys(by).length ? Object.entries(by).sort().map(([t, o]) =>
    `<tr><td>${esc(t)}</td><td class="muted">${esc(layerOf[t] || "")}</td><td class="num">${o.blocked || 0}</td>
     <td class="num">${o.detected || 0}</td><td class="num ${o.succeeded ? "bad" : ""}">${o.succeeded || 0}</td></tr>`).join("")
    : `<tr><td colspan="5" class="empty">No security events yet.</td></tr>`;
  const feed = (sec.recent || []).slice(-250).reverse();
  $("#sec-feed tbody").innerHTML = feed.length ? feed.map((e) =>
    `<tr><td class="mono">${esc((e.time || "").slice(11, 19))}</td><td>${esc(e.attack_type)}</td><td class="muted">${esc(e.layer)}</td>
     <td><span class="dot ${OUT_CLS[e.outcome] || ""}"></span>${esc(e.outcome)}</td><td class="mono">${esc(e.source_ip || "")}</td>
     <td>${esc(e.claimed_identity || "")}</td><td>${esc(e.cert_identity || "")}</td><td class="muted">${esc(e.defence || "")}</td>
     <td class="muted">${esc(e.evidence || "")}</td><td class="muted">${esc((e.detail || "").slice(0, 80))}</td></tr>`).join("")
    : `<tr><td colspan="10" class="empty">No events.</td></tr>`;
  renderLab();
}
function latestAttackRows(jobs) {
  const j = jobs.find((x) => x.rows && x.rows.length);
  if (!j) return "";
  const sm = j.summary || {};
  return `<div class="card-head" style="margin-top:12px"><h2>Latest run: ${esc(j.name)}</h2>
    <span class="badge ${sm.all_as_expected ? "ok" : "warn"}">${sm.all_as_expected ? "PASS" : "FAIL"} ${sm.as_expected}/${sm.cases}</span></div>
    <div class="table-wrap"><table><thead><tr><th>Attack</th><th>Layer</th><th>Expected</th><th>Observed</th><th>Evidence</th></tr></thead><tbody>
    ${j.rows.map((r) => `<tr><td>${esc(r.attack)}</td><td class="muted">${esc(r.layer)}</td><td>${esc(r.expected)}</td>
      <td><span class="dot ${r.as_expected ? "ok" : "bad"}"></span>${esc(r.observed)}</td><td class="muted">${esc(r.evidence)}</td></tr>`).join("")}
    </tbody></table></div><p class="muted">Same command as <code>vfl.cli attack</code>; report files in <code>${esc(j.out || "")}</code>.</p>`;
}
function renderLab() {
  const lab = S.lab;
  if (!S.lab_mode || !lab) { $("#lab-body").innerHTML = `<p class="muted">Start the coordinator with <code>--lab-mode</code> to run the security conformance suite.</p>`; return; }
  const jobs = lab.jobs || [];
  $("#lab-body").innerHTML = `
    <div class="form-actions">
      <select id="atk-case">${["all", ...Object.keys(lab.families || {}), ...(lab.cases || [])].map((c) => `<option>${esc(c)}</option>`).join("")}</select>
      <button class="btn" id="atk-run">Run</button>
      <span class="muted">against this deployment's gate</span>
    </div>
    <div class="table-wrap"><table><thead><tr><th>Case</th><th>When</th><th>Status</th><th>As expected</th></tr></thead><tbody>
    ${jobs.length ? jobs.map((j) => `<tr><td>${esc(j.name)}</td><td class="mono">${new Date(j.t * 1000).toLocaleTimeString()}</td>
      <td>${esc(j.status)}</td><td class="num">${j.summary ? `${j.summary.as_expected}/${j.summary.cases}` : "—"}</td></tr>`).join("")
      : `<tr><td colspan="4" class="empty">No runs yet.</td></tr>`}</tbody></table></div>
    ${latestAttackRows(jobs)}`;
  const btn = $("#atk-run");
  if (btn) btn.addEventListener("click", async () => {
    btn.disabled = true;
    try { await post("/api/lab/attacks", { name: $("#atk-case").value, args: { session: S.current?.session_id } }); poll(); }
    catch (e) { alert(e.message); } finally { btn.disabled = false; }
  });
}

/* ===== inference tab ===== */
function renderInference() {
  $("#status-pill").innerHTML = `<span class="dot ok"></span>inference`;
  const pts = (S.inference || []).filter((p) => p.test_auroc != null);
  kpis([
    ["Sessions", fmt(pts.length, 0), "with a leakage measurement"],
    ["Worst cosine AUC", fmt(Math.max(0.5, ...pts.map((p) => p.cosine_leak_auc || 0.5)), 3), "1.0 = labels exposed"],
    ["Succeeded", fmt(pts.filter((p) => p.attack_outcome === "succeeded").length, 0), "attack AUC > 0.6"],
    ["Formal label-DP", pts.some((p) => p.label_rr_epsilon) ? "used" : "none", "randomized response"],
  ], "#kpis");
  const series = [
    { name: "defence off", points: pts.filter((p) => p.defense === "none").map((p) => [p.cosine_leak_auc, p.test_auroc]), dots: true, color: css("--s5") },
    { name: "grad_noise", points: pts.filter((p) => p.defense === "grad_noise").map((p) => [p.cosine_leak_auc, p.test_auroc]), dots: true, color: css("--s2") },
  ].filter((s) => s.points.length);
  lineChart($("#inf-chart"), series, { xlabel: "cosine attack AUC", ymin: 0.4, ymax: 1.0, xint: false, legend: true, empty: "run a session to measure leakage" });
  $("#inf-table tbody").innerHTML = pts.length ? pts.slice().reverse().map((p) =>
    `<tr><td class="mono">${esc(p.session_id)}</td><td>${esc(p.defense)}${p.label_rr_epsilon ? " + ε-DP" : ""}</td>
     <td class="num">${fmt(p.sigma, 1)}</td><td class="num">${fmt(p.label_rr_epsilon, 2)}</td><td class="num">${fmt(p.test_auroc, 4)}</td>
     <td class="num">${fmt(p.norm_leak_auc, 3)}</td><td class="num">${fmt(p.cosine_leak_auc, 3)}</td>
     <td><span class="dot ${p.attack_outcome === "succeeded" ? "bad" : "ok"}"></span>${esc(p.attack_outcome)}</td></tr>`).join("")
    : `<tr><td colspan="8" class="empty">No finished sessions with leakage metrics.</td></tr>`;
  const withdp = pts.find((p) => p.privacy && (p.privacy.grad_channel_epsilon != null || p.privacy.label_dp_epsilon != null));
  $("#inf-kv").innerHTML = withdp ? [["formal label ε", fmt(withdp.privacy.label_dp_epsilon, 3)],
    ["gradient-channel ε", fmt(withdp.privacy.grad_channel_epsilon, 3)], ["δ", withdp.privacy.delta]]
    .map(([a, b]) => `<div class="k">${esc(a)}</div><div class="v">${esc(b)}</div>`).join("") : "";
}

/* ===== identities tab ===== */
function renderIdentities() {
  $("#status-pill").innerHTML = `<span class="dot ok"></span>identities`;
  const pki = S.pki;
  $("#pki-off").classList.toggle("hidden", !!pki);
  $("#pki-on").classList.toggle("hidden", !pki);
  kpis([], "#kpis");
  if (!pki) return;
  $("#pki-meta").textContent = `domain ${pki.trust_domain} · root ${(pki.root_sha256 || "").slice(0, 12)} · certs ${pki.cert_hours}h`;
  $("#join-mode-row").innerHTML = ["static", "token", "approval"].map((m) =>
    `<button class="btn ${m === pki.mode ? "primary" : ""}" data-mode="${m}">${m}</button>`).join("") + `<span class="muted">join mode</span>`;
  $("#join-mode-row").querySelectorAll("button").forEach((b) => b.addEventListener("click", async () => {
    try { await post("/api/pki/mode", { mode: b.dataset.mode }); poll(); } catch (e) { alert(e.message); }
  }));
  const tb = $("#tok-btn");
  if (tb && !tb.dataset.w) { tb.dataset.w = 1; tb.addEventListener("click", async () => {
    try { const r = await post("/api/pki/tokens", { role: $("#tok-role").value, name: $("#tok-name").value || null }); $("#tok-out").value = r.token; }
    catch (e) { alert(e.message); } }); }
  const reqs = (pki.requests || []).filter((r) => r.status === "pending");
  $("#req-table tbody").innerHTML = reqs.length ? reqs.map((r) =>
    `<tr><td>${esc(r.name)}</td><td>${esc(r.role)}</td><td class="mono">${esc(r.hostname || "")}</td>
     <td class="mono">${new Date(r.t * 1000).toLocaleTimeString()}</td><td>${esc(r.status)}</td>
     <td><button class="btn" data-ap="${esc(r.id)}">Approve</button> <button class="btn danger" data-dn="${esc(r.id)}">Deny</button></td></tr>`).join("")
    : `<tr><td colspan="6" class="empty">No pending join requests.</td></tr>`;
  $("#req-table").querySelectorAll("[data-ap]").forEach((b) => b.addEventListener("click", () => decide(b.dataset.ap, "approve")));
  $("#req-table").querySelectorAll("[data-dn]").forEach((b) => b.addEventListener("click", () => decide(b.dataset.dn, "deny")));
  const ids = pki.identities || []; const revoked = new Set(pki.revoked || []);
  $("#id-table tbody").innerHTML = ids.length ? ids.map((i) => {
    const rev = revoked.has(i.serial);
    return `<tr><td>${esc(i.name)}${rev ? ' <span class="badge" style="color:var(--danger)">revoked</span>' : ""}</td><td>${esc(i.role || "")}</td>
     <td class="mono">${esc((i.serial || "").slice(0, 12))}</td><td class="mono">${i.not_after ? new Date(i.not_after * 1000).toLocaleString() : "—"}</td>
     <td class="muted">${esc(i.via || "")}</td><td>${rev ? "" : `<button class="btn danger" data-rev="${esc(i.serial)}">Revoke</button>`}</td></tr>`;
  }).join("") : `<tr><td colspan="6" class="empty">No identities issued.</td></tr>`;
  $("#id-table").querySelectorAll("[data-rev]").forEach((b) => b.addEventListener("click", async () => {
    if (!confirm("Revoke this certificate? If the party is in a session it will be stopped.")) return;
    try { await post(`/api/pki/revoke/${encodeURIComponent(b.dataset.rev)}`, {}); poll(); } catch (e) { alert(e.message); }
  }));
}
async function decide(id, action) { try { await post(`/api/pki/requests/${id}/${action}`, {}); poll(); } catch (e) { alert(e.message); } }

/* ================= party ================= */
function renderParty() {
  $("#party-view").classList.remove("hidden");
  const isA = S.role === "active";
  const c = S.coordinator;
  $("#node-line").textContent = `${S.node} · coordinator ${c.address} ${c.connected ? "✓" : "✗"} · uptime ${dur(S.uptime_s)}`;
  $("#status-pill").innerHTML = `<span class="dot ${statusDot(S.status)}"></span>${esc(S.status)}`;
  const pr = S.progress || {}, k = S.counters || {};
  const tr = S.recent_batches.filter((r) => r.phase === "train");
  const lat = tr.slice(-200).map((r) => (isA ? r.slot_ms ?? r.compute_ms : r.batch_ms)).filter((x) => x != null);
  const cfg = S.session.config || {};
  kpis([
    ["Status", S.status, S.session.session_id || (c.registered ? "waiting for a session" : (c.error || "registering…"))],
    ["Progress", pr.epoch !== undefined ? `e${pr.epoch + 1}/${pr.epochs ?? cfg.epochs ?? "?"}` : "—",
      pr.phase ? `${pr.phase} ${pr.batch ?? pr.eval_batch ?? ""}${pr.n_batches && pr.phase === "train" ? "/" + pr.n_batches : ""}` : ""],
    [isA ? "Slot time p50/p95" : "Batch p50 / p95", lat.length ? `${fmt(pct(lat, .5), 1)} / ${fmt(pct(lat, .95), 1)}` : "—", "ms"],
    ["Committed batches", fmt(k.committed_batches || 0, 0), isA ? `${k.duplicates_suppressed || 0} duplicates suppressed` : `${k.retries || 0} retries · ${k.rpc_timeouts || 0} timeouts`],
    [isA ? "Tensors in / out" : "Tensors up / down", `${bytes(isA ? k.tensor_bytes_in : k.tensor_bytes_up)}`, `${bytes(isA ? k.tensor_bytes_out : k.tensor_bytes_down)} ${isA ? "out" : "down"}`],
    ["Process", S.resources.rss_mib ? `${fmt(S.resources.rss_mib, 0)} MiB` : "—", `cpu ${fmt(S.resources.cpu_percent, 0)}%`],
    ...(isA ? [["Last loss", fmt(pr.last_loss, 4), S.epochs.length ? `test AUROC ${fmt(S.epochs[S.epochs.length - 1].test?.auroc, 4)}` : ""]] : []),
  ]);

  // privacy boundary
  const pv = S.privacy || {};
  const li = (a) => (a || []).map((x) => `<li>${esc(x)}</li>`).join("");
  const al = pv.alignment;
  const badge = $("#align-badge");
  if (al) { badge.className = `badge ${al.private ? "ok" : "warn"}`; badge.textContent = al.private ? `${al.mode}: private alignment` : "NON-PRIVATE experimental alignment"; }
  else { badge.className = "badge"; badge.textContent = "not aligned yet"; }
  const def = pv.defense && pv.defense.type !== "none" ? pv.defense : null;
  $("#boundary").innerHTML = `
    <div class="box keep"><h3>Stays on this node</h3><ul>${li(pv.holds)}</ul></div>
    <div class="box never"><h3>${isA ? "Never leaves" : "Never leaves / never received"}</h3><ul>${li(pv.never_leaves)}${li(pv.never_receives)}</ul></div>
    <div class="box out"><h3>Sent</h3><ul>${li(pv.sends)}</ul></div>
    <div class="box in"><h3>Received</h3><ul>${li(pv.receives)}</ul></div>
    <div class="full">${al ? `Alignment: ${esc(al.disclosure)} · ${fmt(al.n_aligned, 0)} of ${fmt(al.n_local, 0)} local rows aligned in ${fmt(al.align_ms, 0)} ms.` : ""}
    ${def ? `<br>Defence: ${esc(def.type)} (C=${def.clip_norm}, σ=${def.noise_multiplier}) — ${esc(def.guarantee)}.` : ""}
    ${S.tls ? "<br>Transport: mutual TLS (protects against network observers, not against an authorised recipient)." : "<br><b>Transport: INSECURE</b> — use --tls-dir for networked runs."}</div>`;

  // breakdown
  const recent = tr.slice(-100);
  const parts = isA
    ? [["wait for parties", "wait_parties_ms"], ["forward", "fwd_ms"], ["backward", "bwd_ms"], ["optimizer", "step_ms"], ["defence", "defense_ms"], ["fault delay", "delay_ms"]]
    : [["forward", "fwd_ms"], ["encode", "encode_ms"], ["injected RTT", "injected_rtt_ms"], ["network + gRPC", "network_ms"],
       ["wait at active", "server_wait_ms"], ["active compute", "server_compute_ms"], ["fault delay", "server_delay_ms"], ["backward", "bwd_ms"], ["optimizer", "step_ms"]];
  const vals = parts.map(([n, key]) => [n, mean(recent.map((r) => r[key] || 0)) || 0]);
  const tot = vals.reduce((a, [, v]) => a + v, 0);
  $("#breakdown").innerHTML = recent.length ? `
    <div class="stack">${vals.map(([n, v], i) => `<span title="${esc(n)}: ${v.toFixed(2)} ms" style="width:${tot ? (100 * v) / tot : 0}%;background:${color(i)}"></span>`).join("")}</div>
    <div class="legend">${vals.map(([n, v], i) => `<span><i style="background:${color(i)}"></i>${esc(n)} ${v.toFixed(2)} ms</span>`).join("")}</div>`
    : `<div class="empty">Timing appears once training starts.</div>`;
  const kv = [["session", S.session.session_id], ["batch size", cfg.batch_size], ["epochs", cfg.epochs], ["codec", cfg.codec],
    ["injected RTT", cfg.inject_rtt_ms !== undefined ? `${cfg.inject_rtt_ms} ms` : null], ["alignment", cfg.alignment],
    ["defence", cfg.defense?.type], ["participants", (S.session.participants || []).map((p) => p.party_id).join(", ")]].filter(([, v]) => v !== undefined && v !== null && v !== "");
  $("#session-kv").innerHTML = kv.map(([a, b]) => `<div class="k">${esc(a)}</div><div class="v">${esc(b)}</div>`).join("");

  // charts
  if (isA) {
    $("#p-chart1-title").textContent = "Loss (per batch) and slot time";
    lineChart($("#p-c1"), [{ name: "loss", points: tr.map((r, i) => [i, r.loss]).filter((p) => p[1] != null) }], { xint: true, xlabel: "batch (recent)" });
    $("#p-chart2-title").textContent = "AUROC per epoch";
    lineChart($("#p-c2"), [
      { name: "val", points: S.epochs.filter((e) => e.val?.auroc != null).map((e) => [e.epoch, e.val.auroc]), dots: true },
      { name: "test", points: S.epochs.filter((e) => e.test?.auroc != null).map((e) => [e.epoch, e.test.auroc]), dots: true },
    ], { xint: true, xlabel: "epoch" });
  } else {
    $("#p-chart1-title").textContent = "Batch latency (ms)";
    lineChart($("#p-c1"), [
      { name: "batch", points: tr.slice(-400).map((r, i) => [i, r.batch_ms]) },
      { name: "rpc", points: tr.slice(-400).map((r, i) => [i, r.rpc_ms]) },
    ], { xint: true, ymin: 0, xlabel: "batch (recent)" });
    $("#p-chart2-title").textContent = "Application traffic per epoch (MiB)";
    lineChart($("#p-c2"), [
      { name: "app bytes", points: S.epochs.map((e) => [e.epoch, e.mib_per_epoch_app]), dots: true },
      { name: "tensor payload", points: S.epochs.map((e) => [e.epoch, e.mib_per_epoch_tensor]), dots: true },
    ], { xint: true, ymin: 0, xlabel: "epoch" });
  }

  // epoch table
  const head = isA ? ["epoch", "time s", "loss", "val AUROC", "test AUROC", "test acc", "test F1", "leak(norm)", "leak(cos)"]
    : ["epoch", "time s", "batches", "p50 ms", "p95 ms", "batches/s", "MiB app", "MiB tensor"];
  $("#epoch-table thead").innerHTML = `<tr>${head.map((h, i) => `<th class="${i ? "num" : ""}">${h}</th>`).join("")}</tr>`;
  $("#epoch-table tbody").innerHTML = S.epochs.length ? S.epochs.map((e) => isA
    ? `<tr><td>${e.epoch}</td><td class="num">${fmt(e.epoch_s)}</td><td class="num">${fmt(e.train_loss, 4)}</td><td class="num">${fmt(e.val?.auroc, 4)}</td><td class="num">${fmt(e.test?.auroc, 4)}</td><td class="num">${fmt(e.test?.accuracy, 3)}</td><td class="num">${fmt(e.test?.f1, 3)}</td><td class="num">${fmt(e.leakage?.norm_leak_auc, 3)}</td><td class="num">${fmt(e.leakage?.cosine_leak_auc, 3)}</td></tr>`
    : `<tr><td>${e.epoch}</td><td class="num">${fmt(e.epoch_s)}</td><td class="num">${e.train_batches}</td><td class="num">${fmt(e.batch_ms_p50, 1)}</td><td class="num">${fmt(e.batch_ms_p95, 1)}</td><td class="num">${fmt(e.throughput_batches_per_s, 1)}</td><td class="num">${fmt(e.mib_per_epoch_app, 3)}</td><td class="num">${fmt(e.mib_per_epoch_tensor, 3)}</td></tr>`).join("")
    : `<tr><td colspan="${head.length}" class="empty">No completed epochs in this session.</td></tr>`;

  // data
  const d = S.data;
  $("#data-kv").innerHTML = [["file", d.file], ["rows", d.rows], ["features", d.n_features], ["labels", d.has_labels ? "yes (label owner)" : "no"],
    ...(d.partition ? [["split", `${d.partition.partitioner} · ${d.partition.strategy} · overlap ${d.partition.overlap}`],
      ["manifest", (d.partition.manifest_sha256 || "").slice(0, 12)]] : [["split", "no manifest"]]),
    ...(d.shape ? [["input shape", d.shape.join(" × ")]] : []),
    ...(d.partition && d.partition.num_classes ? [["classes", d.partition.num_classes + (d.partition.binarize ? ` (${d.partition.binarize})` : "")]] : []),
    ["torch", S.env?.torch], ["grpc", S.env?.grpc], ["host", S.env?.hostname]]
    .map(([a, b]) => `<div class="k">${esc(a)}</div><div class="v">${esc(b)}</div>`).join("");
  $("#features").innerHTML = (d.feature_names || []).map((f) => `<span>${esc(f)}</span>`).join("") + (d.n_features > (d.feature_names || []).length ? `<span>+${d.n_features - d.feature_names.length} more</span>` : "");

  // log
  $("#log-dir").textContent = S.log_dir || "";
  const box = $("#node-log"); const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 10;
  box.innerHTML = S.events.map((e) => `<div class="${e.level}"><span class="t">${new Date(e.t * 1000).toLocaleTimeString()}</span>${esc(e.msg)}</div>`).join("");
  if (atBottom) box.scrollTop = box.scrollHeight;
}

window.addEventListener("resize", () => S && render());
poll();
setInterval(poll, 1500);
