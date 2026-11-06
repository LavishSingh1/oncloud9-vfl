/* VFL benchmark dashboard. No external JS. */
"use strict";
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const TOKEN = new URLSearchParams(location.search).get("token");
const fmt = (v, d = 3) => (v == null || Number.isNaN(v)) ? "—" : (typeof v === "number" ? v.toFixed(d) : String(v));
const hdr = () => { const h = { "Content-Type": "application/json" }; if (TOKEN) h.Authorization = `Bearer ${TOKEN}`; return h; };
const get = (u) => fetch(u).then((r) => r.json());

let TAB = "launcher", JOB = null;
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  TAB = b.dataset.tab;
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".pane").forEach((p) => p.classList.toggle("hidden", p.id !== TAB));
  refresh();
}));

async function refresh() {
  try {
    if (TAB === "launcher") await renderJobs();
    else if (TAB === "runs") await renderRuns();
    else if (TAB === "analytics") await renderAnalytics();
    else if (TAB === "scorecard") await renderScorecard();
    const runs = await get("/api/runs");
    $("#sub").textContent = `${runs.length} runs across ${new Set(runs.map((r) => r.bench)).size} benchmark folders`;
  } catch (e) { $("#sub").textContent = "offline"; }
}

$("#launch-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const spec = {};
  for (const el of ev.target.querySelectorAll("[name]")) if (el.value.trim()) spec[el.name] = el.value.trim();
  const msg = $("#launch-msg"); msg.className = "form-msg"; msg.textContent = "launching…";
  try {
    const r = await fetch("/api/launch", { method: "POST", headers: hdr(), body: JSON.stringify(spec) });
    const j = await r.json(); if (!r.ok) throw new Error(j.error || r.statusText);
    msg.className = "form-msg ok"; msg.textContent = `started ${j.job} → ${j.out}`; renderJobs();
  } catch (e) { msg.className = "form-msg err"; msg.textContent = e.message; }
});

async function renderJobs() {
  const jobs = await get("/api/jobs");
  $("#jobs tbody").innerHTML = jobs.length ? jobs.map((j) =>
    `<tr><td class="mono">${new Date(j.t * 1000).toLocaleTimeString()}</td><td>${esc(j.suite)}</td>
     <td class="mono">${esc((j.out || "").split("/").pop())}</td><td>${esc(j.status)}</td>
     <td><button class="btn" data-log="${esc(j.id)}">log</button>${j.status === "running" ? ` <button class="btn danger" data-cancel="${esc(j.id)}">cancel</button>` : ""}
     ${j.spec ? `<button class="btn" data-rerun='${esc(JSON.stringify(j.spec))}'>rerun</button>` : ""}</td></tr>`).join("")
    : `<tr><td colspan="5" class="empty">No jobs yet.</td></tr>`;
  $("#jobs").querySelectorAll("[data-log]").forEach((b) => b.onclick = () => { JOB = b.dataset.log; showLog(jobs); });
  $("#jobs").querySelectorAll("[data-cancel]").forEach((b) => b.onclick = async () => { await fetch(`/api/jobs/${b.dataset.cancel}/cancel`, { method: "POST", headers: hdr() }); renderJobs(); });
  $("#jobs").querySelectorAll("[data-rerun]").forEach((b) => b.onclick = async () => {
    await fetch("/api/launch", { method: "POST", headers: hdr(), body: b.dataset.rerun }); renderJobs();
  });
  if (JOB) showLog(jobs);
}
function showLog(jobs) { const j = jobs.find((x) => x.id === JOB); $("#job-log").textContent = j ? j.log : ""; }

async function renderRuns() {
  const runs = await get("/api/runs");
  $("#runs-n").textContent = `${runs.length} runs`;
  $("#runlist tbody").innerHTML = runs.length ? runs.slice().reverse().map((r) =>
    `<tr><td class="mono">${esc(r.label || r.run_id)}</td><td>${esc(r.suite)}</td><td class="mono">${esc(r.bench)}</td>
     <td class="num">${esc(r.seed)}</td><td class="num">${fmt(r.test_auroc, 4)}</td><td>${esc(r.state)}</td>
     <td class="mono muted">${esc((r.config_hash || "").slice(0, 10))}</td></tr>`).join("")
    : `<tr><td colspan="7" class="empty">No runs. Launch one.</td></tr>`;
  await renderGrouped();
}
$("#group").addEventListener("change", renderGrouped);
$("#filter").addEventListener("change", renderGrouped);
async function renderGrouped() {
  const group = $("#group").value;
  const filt = $("#filter").value.split(";").filter(Boolean).map((p) => "f." + p.trim()).join("&").replace(/=/g, (m, i, s) => s[i - 1] === "." ? "=" : "=");
  const q = "/api/grouped?group=" + encodeURIComponent(group) + (filt ? "&" + filt : "");
  const g = await get(q);
  $("#grouped tbody").innerHTML = g.length ? g.map((r) => {
    const cv = r.auroc_mean ? r.auroc_std / r.auroc_mean : 0;
    const flag = r.n_runs > 1 && cv > 0.05;
    return `<tr><td class="mono">${esc(Object.entries(r.group).map(([k, v]) => `${k}=${v}`).join(" "))}</td>
     <td class="num">${r.n_runs}</td><td class="mono">${esc((r.seeds || []).join(","))}</td>
     <td class="num">${fmt(r.auroc_mean, 4)}</td><td class="num">${fmt(r.auroc_std, 4)}</td>
     <td class="num">${r.auroc_ci95 ? "±" + fmt(r.auroc_ci95, 4) : "—"}</td>
     <td>${flag ? '<span class="dot bad"></span> &gt;5% var' : '<span class="dot ok"></span> ok'}</td></tr>`;
  }).join("") : `<tr><td colspan="7" class="empty">No matching runs.</td></tr>`;
}

async function benchSelect(sel) {
  const benches = await get("/api/benches");
  const keep = sel.value;
  sel.innerHTML = benches.map((b) => `<option>${esc(b)}</option>`).join("");
  if (benches.includes(keep)) sel.value = keep;
  return benches;
}
async function renderAnalytics() {
  const benches = await benchSelect($("#bench-sel"));
  if (!benches.length) { $("#figs").innerHTML = '<p class="empty">No analyzed benchmarks. Run one, or run bench/analyze.py.</p>'; return; }
  const b = $("#bench-sel").value || benches[0];
  $("#export").href = "/api/export/" + encodeURIComponent(b) + (TOKEN ? "?token=" + TOKEN : "");
  const rep = await get("/api/report/" + encodeURIComponent(b));
  $("#figs").innerHTML = rep.figures.length ? rep.figures.map((f) =>
    `<figure><img loading="lazy" src="/report-asset/${encodeURIComponent(b)}/${encodeURIComponent(f)}"><figcaption>${esc(f)}</figcaption></figure>`).join("")
    : '<p class="empty">No figures in this benchmark\'s report.</p>';
}
$("#bench-sel").addEventListener("change", renderAnalytics);

async function renderScorecard() {
  const benches = await get("/api/benches");
  if (!benches.length) { $("#nfr tbody").innerHTML = `<tr><td colspan="3" class="empty">No benchmarks.</td></tr>`; return; }
  const b = benches[benches.length - 1];
  $("#sc-bench").textContent = b;
  const rep = await get("/api/report/" + encodeURIComponent(b));
  const names = { "NFR-1": "Utility", "NFR-2": "Reproducibility", "NFR-3": "Latency", "NFR-4": "Communication",
    "NFR-5": "Throughput", "NFR-6": "Efficiency", "NFR-7": "Transport & logging", "NFR-8": "Privacy eval",
    "NFR-9": "Cloud cost", "NFR-10": "Fault handling" };
  const sc = { pass: "ok", fail: "bad", reported: "warn", "not attempted": "" };
  $("#nfr tbody").innerHTML = Object.keys(names).map((k) => {
    const v = rep.nfr[k] || {}; const st = v.status || "not attempted";
    return `<tr><td><b>${k}</b> ${esc(names[k])}</td><td><span class="dot ${sc[st] || "warn"}"></span>${esc(st)}</td>
     <td class="muted">${esc((v.value || "").slice(0, 160))}</td></tr>`;
  }).join("");
  const s = rep.security;
  $("#sec-sum-card").style.display = s ? "" : "none";
  if (s) $("#sec-sum").innerHTML = Object.entries(s).map(([k, v]) => `<div class="k">${esc(k)}</div><div class="v">${esc(v)}</div>`).join("");
}

refresh();
setInterval(() => { if (TAB === "launcher") renderJobs(); }, 2000);
