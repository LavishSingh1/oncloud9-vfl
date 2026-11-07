/* Session form generated from the config schema (GET /api/config/schema). No external deps, no free-text
 * inputs: numbers and enums are dropdowns of presets, booleans are toggles, per-party settings get one dropdown
 * per detected party, fault slots are epoch/batch rows, resume_from lists past sessions.
 *
 * The pure helpers at the top (initialValues, optionValues, visible, usedKinds, toPayload) are exported for
 * node so tests/test_config_schema.py can check them against vfl/config_schema.py. The server builds the
 * nested config from the flat {key: value} map (config_schema.config_from_form), so validation lives in one place.
 */
"use strict";

const VFLForm = (() => {
  const J = (v) => JSON.stringify(v === undefined ? null : v);
  const clone = (v) => (v === undefined ? undefined : JSON.parse(JSON.stringify(v)));
  const SHAPED = { cnn: 3, cnn1d: 2, gru: 2 };

  function getPath(cfg, key) {
    let node = cfg;
    for (const part of key.split(".")) {
      if (node === null || typeof node !== "object" || !(part in node)) return undefined;
      node = node[part];
    }
    return node;
  }

  /* flat {key: value} for every form field; mirrors config_schema.form_values */
  function initialValues(schema, cfg) {
    const out = {};
    for (const p of schema.params) {
      const v = getPath(cfg || {}, p.key);
      out[p.key] = clone(v === undefined ? p.default : v);
    }
    return out;
  }

  /* dropdown values: null (if nullable), choices, presets, default, and the current value if it is none of these */
  function optionValues(p, current) {
    const vals = [];
    const has = (v) => vals.some((w) => J(w) === J(v));
    if (p.nullable) vals.push(null);
    for (const v of [...(p.choices || []), ...(p.presets || []), p.default]) {
      if (v === null || v === undefined || has(v)) continue;
      vals.push(v);
    }
    if (current !== undefined && !has(current)) vals.push(current);
    return vals;
  }

  function optionLabel(p, v) {
    if (v === null) return p.null_label || "off";
    for (const [vj, lab] of p.labels || []) if (vj === J(v)) return lab;
    if (Array.isArray(v)) return `[${v.join(", ")}]`;
    if (typeof v === "boolean") return v ? "on" : "off";
    if (typeof v === "number") {
      const s = Math.abs(v) > 0 && Math.abs(v) < 1e-3 ? v.toExponential() : String(v);
      return p.unit ? `${s} ${p.unit}` : s;
    }
    return String(v);
  }

  /* bottom-model kinds in use: every explicit per-party kind, plus the default if some detected party (or, with
   * no parties detected yet, anyone) falls back to it */
  function usedKinds(bm, parties) {
    if (typeof bm === "string") return new Set([bm]);
    const m = bm || {};
    const ids = (parties || []).map((x) => x.party_id);
    const out = new Set(Object.entries(m).filter(([k]) => k !== "default").map(([, v]) => v));
    if (!ids.length || ids.some((id) => !(id in m))) out.add(m.default || "mlp");
    return out;
  }

  function condHolds(c, values) {
    if (c[0] === "!") return !c[2].some((v) => J(v) === J(values[c[1]]));
    return c[1].some((v) => J(v) === J(values[c[0]]));
  }

  function visible(p, values, parties) {
    if (p.applies_to && p.applies_to.length) {
      const k = usedKinds(values.bottom_model, parties);
      if (!p.applies_to.some((x) => k.has(x))) return false;
    }
    return (p.show_if || []).every((c) => condHolds(c, values));
  }

  function toPayload(values) {
    return { form: clone(values), resume_from: values.resume_from || null };
  }

  function changedKeys(values, base) {
    return Object.keys(values).filter((k) => J(values[k]) !== J(base[k]));
  }

  /* ------------------------------------------------------------------ DOM */
  function mount(root, opts) {
    const st = {
      schema: null, values: {}, base: {}, ctx: { parties: [], sessions: [], datasets: [] }, ctxSig: "",
      open: new Set(["training", "model", "bottom"]), openAdv: new Set(),
    };
    const h = (tag, attrs = {}, ...kids) => {
      const el = document.createElement(tag);
      for (const [k, v] of Object.entries(attrs)) {
        if (k === "class") el.className = v;
        else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
        else if (v !== false && v !== null && v !== undefined) el.setAttribute(k, v === true ? "" : v);
      }
      for (const kid of kids.flat()) if (kid !== null && kid !== undefined) el.append(kid);
      return el;
    };
    const set = (key, v) => { st.values[key] = v; render(); if (opts.onChange) opts.onChange(st.values); };

    function selectEl(p, cur, onPick, extra) {
      const vals = optionValues(p, cur);
      if (extra) extra.forEach((x) => { if (!vals.some((w) => J(w) === J(x))) vals.push(x); });
      const sel = h("select", { name: p.key, "aria-label": p.key });
      vals.forEach((v, i) => {
        const o = h("option", { value: String(i) }, optionLabel(p, v));
        if (J(v) === J(cur)) o.selected = true;
        sel.append(o);
      });
      sel.addEventListener("change", () => onPick(vals[+sel.value]));
      return sel;
    }

    const W = {
      select: (p) => selectEl(p, st.values[p.key], (v) => set(p.key, v)),
      toggle: (p) => {
        const cb = h("input", { type: "checkbox", name: p.key });
        cb.checked = !!st.values[p.key];
        cb.addEventListener("change", () => set(p.key, cb.checked));
        return h("label", { class: "switch" }, cb, h("span", { class: "slider" }), h("span", { class: "sw-txt" }, cb.checked ? "on" : "off"));
      },
      party_models: (p) => {
        const bm = st.values[p.key];
        const map = typeof bm === "string" ? { default: bm } : { ...(bm || {}) };
        const kinds = p.choices;
        const commit = (m) => {
          const over = Object.keys(m).filter((k) => k !== "default");
          set(p.key, over.length ? m : (m.default || "mlp"));
        };
        const rows = [];
        const dflt = map.default;
        const dsel = h("select", { name: "bottom_model.default", "aria-label": "default bottom model" },
          ...(dflt === undefined ? [h("option", { value: "" }, "mlp (implicit)")] : []),
          ...kinds.map((k) => h("option", { value: k }, k)));
        dsel.value = dflt === undefined ? "" : dflt;
        dsel.addEventListener("change", () => { const m = { ...map }; if (dsel.value) m.default = dsel.value; else delete m.default; commit(m); });
        rows.push(h("div", { class: "pm-row" }, h("span", { class: "pm-id" }, "default"), dsel));
        const ids = [...st.ctx.parties.map((x) => x.party_id)];
        Object.keys(map).filter((k) => k !== "default" && !ids.includes(k)).forEach((k) => ids.push(k));
        for (const id of ids) {
          const party = st.ctx.parties.find((x) => x.party_id === id);
          const shape = party && party.summary ? party.summary.shape : null;
          const sel = h("select", { name: `bottom_model.${id}`, "aria-label": `bottom model for ${id}` },
            h("option", { value: "" }, `(default: ${map.default || "mlp"})`));
          for (const k of kinds) {
            const need = SHAPED[k];
            const ok = !need || !party || (Array.isArray(shape) && shape.length === need);
            const o = h("option", { value: k }, ok ? k : `${k} (needs ${need}-D shape)`);
            if (!ok) o.disabled = true;
            sel.append(o);
          }
          sel.value = map[id] || "";
          sel.addEventListener("change", () => { const m = { ...map }; if (sel.value) m[id] = sel.value; else delete m[id]; commit(m); });
          const role = party ? party.role : "offline";
          rows.push(h("div", { class: "pm-row" }, h("span", { class: "pm-id" }, id, h("span", { class: `muted pm-role` }, ` ${role}`)), sel));
        }
        if (!st.ctx.parties.length) rows.push(h("div", { class: "muted pm-note" }, "Per-party dropdowns appear when parties register."));
        return h("div", { class: "party-models" }, rows);
      },
      pairs: (p) => {
        const pairs = st.values[p.key] || [];
        const nEp = Math.max(1, +st.values.epochs || 1);
        const v = st.values, nRows = st.ctx.activeRows;       // upper bound: local rows >= aligned rows
        const frac = 1 - (+v["split.val"] || 0) - (+v["split.test"] || 0);
        const nB = nRows ? Math.max(1, Math.ceil((nRows * frac) / Math.max(1, +v.batch_size || 64))) : 50;
        const range = (n, cur) => { const a = [...Array(n).keys()]; if (cur >= n) a.push(cur); return a; };
        const rows = pairs.map(([e, b], i) => {
          const es = h("select", { "aria-label": `${p.key} epoch` }, ...range(nEp, e).map((x) => h("option", { value: x }, `epoch ${x}`)));
          es.value = e;
          const bs = h("select", { "aria-label": `${p.key} batch` }, ...range(nB, b).map((x) => h("option", { value: x }, `batch ${x}`)));
          bs.value = b;
          const upd = () => { const c = clone(pairs); c[i] = [+es.value, +bs.value]; set(p.key, c); };
          es.addEventListener("change", upd); bs.addEventListener("change", upd);
          return h("div", { class: "pair-row" }, es, bs,
            h("button", { type: "button", class: "btn mini", title: "remove", onclick: () => set(p.key, pairs.filter((_, j) => j !== i)) }, "✕"));
        });
        return h("div", { class: "pairs" }, rows.length ? rows : h("span", { class: "muted" }, "none"),
          h("button", { type: "button", class: "btn mini", onclick: () => set(p.key, [...pairs, [0, Math.min(3, nB - 1)]]) }, "+ add slot"));
      },
      session: (p) => selectEl({ ...p, presets: [], choices: [] }, st.values[p.key], (v) => {
        set(p.key, v);
        if (v && opts.onResume) opts.onResume(v);
      }, st.ctx.sessions.map((s) => s.session_id)),
      parties: (p) => {
        const cur = st.values[p.key];
        const ids = st.ctx.parties.map((x) => x.party_id);
        (cur || []).forEach((id) => { if (!ids.includes(id)) ids.push(id); });
        const all = h("input", { type: "checkbox", name: "participants.all" });
        all.checked = cur === null || cur === undefined;
        all.addEventListener("change", () => set(p.key, all.checked ? null : ids.slice()));
        const boxes = ids.map((id) => {
          const cb = h("input", { type: "checkbox", name: `participants.${id}` });
          cb.checked = all.checked || (cur || []).includes(id);
          cb.disabled = all.checked;
          cb.addEventListener("change", () => {
            const next = ids.filter((x) => x === id ? cb.checked : (cur || []).includes(x));
            set(p.key, next.length ? next : null);
          });
          return h("label", { class: "chk" }, cb, id);
        });
        return h("div", { class: "chks" }, h("label", { class: "chk" }, all, p.null_label), boxes);
      },
      salt: (p) => {
        const cur = st.values[p.key];
        const NEW = "__new__";
        const sel = h("select", { name: p.key, "aria-label": p.key },
          h("option", { value: "" }, p.null_label),
          ...(cur ? [h("option", { value: cur }, `${cur.slice(0, 12)}…`)] : []),
          h("option", { value: NEW }, "new random salt"));
        sel.value = cur || "";
        sel.addEventListener("change", () => {
          if (sel.value === NEW) {
            const b = new Uint8Array(12); crypto.getRandomValues(b);
            set(p.key, Array.from(b, (x) => x.toString(16).padStart(2, "0")).join(""));
          } else set(p.key, sel.value || null);
        });
        return sel;
      },
      name: (p) => {
        const v = st.values;
        const bm = typeof v.bottom_model === "string" ? v.bottom_model : "mixed";
        const tag = v.dataset_tag && v.dataset_tag !== "unspecified" ? v.dataset_tag : "vfl";
        const n = st.ctx.parties.length || 2;
        const sugg = [`${tag}-${n}party`, `${tag}-${n}party-${bm}-bs${v.batch_size}`, `${tag}-${bm}-e${v.epochs}-s${v.seed}`, "vfl-run"];
        return selectEl({ ...p, presets: [], choices: [] }, v[p.key], (x) => set(p.key, x), sugg);
      },
      dataset: (p) => selectEl({ ...p, presets: [], choices: [] }, st.values[p.key], (x) => set(p.key, x),
        ["unspecified", ...st.ctx.datasets]),
    };

    function field(p) {
      const w = (W[p.widget] || W.select)(p);
      const changed = J(st.values[p.key]) !== J(st.base[p.key]);
      const wide = ["party_models", "pairs", "parties"].includes(p.widget);
      return h("div", { class: `ff${wide ? " wide" : ""}${changed ? " changed" : ""}`, "data-key": p.key },
        h("label", { class: "ff-label", title: p.help }, p.key, changed ? h("span", { class: "dotc", title: "changed from default" }) : null,
          !p.read ? h("span", { class: "muted" }, " (info)") : null),
        w, h("div", { class: "ff-help" }, p.help));
    }

    function render() {
      if (!st.schema) return;
      const keepScroll = root.scrollTop;
      root.innerHTML = "";
      const nChanged = changedKeys(st.values, st.base).length;
      for (const g of st.schema.groups) {
        const ps = st.schema.params.filter((p) => p.group === g.id);
        const vis = ps.filter((p) => visible(p, st.values, st.ctx.parties));
        const main = vis.filter((p) => !p.advanced), adv = vis.filter((p) => p.advanced);
        const nCh = ps.filter((p) => J(st.values[p.key]) !== J(st.base[p.key])).length;
        const det = h("details", { class: "fgroup", "data-group": g.id });
        det.open = st.open.has(g.id);
        det.addEventListener("toggle", () => { if (det.open) st.open.add(g.id); else st.open.delete(g.id); });
        det.append(h("summary", {}, g.label, h("span", { class: "muted" }, ` ${vis.length}/${ps.length}`),
          nCh ? h("span", { class: "gchanged" }, `${nCh} changed`) : null));
        det.append(h("div", { class: "fgrid" }, main.map(field)));
        if (adv.length) {
          const a = h("details", { class: "fadv" }, h("summary", {}, `advanced (${adv.length})`), h("div", { class: "fgrid" }, adv.map(field)));
          a.open = st.openAdv.has(g.id);
          a.addEventListener("toggle", () => { if (a.open) st.openAdv.add(g.id); else st.openAdv.delete(g.id); });
          det.append(a);
        }
        if (!vis.length) det.append(h("div", { class: "muted ff-none" }, "Nothing applies to the current choices."));
        root.append(det);
      }
      root.scrollTop = keepScroll;
      if (opts.onRender) opts.onRender({ changed: nChanged });
    }

    return {
      load(schema, cfg) { st.schema = schema; st.base = initialValues(schema, cfg); st.values = clone(st.base); render(); },
      setValues(cfg, keep) {
        const v = initialValues(st.schema, cfg);
        for (const k of keep || []) v[k] = st.values[k];
        st.values = v; render();
      },
      reset() { st.values = clone(st.base); render(); },
      values() { return clone(st.values); },
      payload() { return toPayload(st.values); },
      updateContext(ctx) {
        const sig = J([ctx.parties.map((p) => [p.party_id, p.role, (p.summary || {}).shape]), ctx.sessions.map((s) => s.session_id),
          ctx.datasets, ctx.activeRows]);
        if (sig === st.ctxSig) return;
        const active = document.activeElement;
        if (active && root.contains(active) && active.tagName === "SELECT") return;   // don't yank an open dropdown; retry next poll
        st.ctxSig = sig; st.ctx = ctx;
        render();
      },
      get loaded() { return !!st.schema; },
    };
  }

  return { initialValues, optionValues, optionLabel, usedKinds, visible, condHolds, toPayload, changedKeys, getPath, mount };
})();

if (typeof module !== "undefined" && module.exports) module.exports = VFLForm;
else window.VFLForm = VFLForm;
