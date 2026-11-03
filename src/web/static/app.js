/* Kasauti front end: vanilla JS, no build step. All profile data is untrusted
   (anyone can POST /ingest), so every value goes through esc() or textContent. */
(() => {
  "use strict";

  const API = "/api/v1";
  const $ = (sel, root = document) => root.querySelector(sel);

  const EXAMPLES = [
    "data engineer with spark and airflow in bangalore",
    "backend engineer with python, django and aws 4-8 years",
    "ml engineer with pytorch and nlp",
    "frontend developer with react and typescript in pune",
    "devops engineer with kubernetes and terraform",
  ];
  const SLIDERS = [
    ["skill_match", "Skills", 25],
    ["experience_match", "Experience", 20],
    ["behavioral_signals", "Engagement", 20],
    ["assessment_score", "Relevance", 15],
    ["cultural_fit", "Semantic fit", 10],
    ["education_match", "Education", 10],
  ];
  const DIM_LABEL = {
    skill_match: "Required skills", semantic_similarity: "Semantic fit",
    cross_encoder_score: "Relevance rank", behavioral_score: "Engagement",
    keyword_match: "Keyword overlap", experience_match: "Experience",
    career_trajectory_score: "Career trajectory", skill_proficiency_score: "Skill depth",
    location_match: "Location", education_match: "Education",
  };
  const STRENGTH_LABEL = {
    verified: "verified", structured: "listed", demonstrated: "used in role", mentioned: "only claimed",
  };

  const state = {
    query: "", results: [], active: null, slidersTouched: false,
    shortlist: loadShortlist(), newIds: new Set(),
  };

  // ── utils ───────────────────────────────────────────
  function esc(v) {
    return String(v ?? "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);
  }
  function pct(x) { return Math.round((Number(x) || 0) * 100); }
  function dimColor(d) { return `var(--d-${d}, #999)`; }
  function highlight(text, start, end) {
    const t = String(text ?? "");
    const s = Math.max(0, Math.min(t.length, start | 0));
    const e = Math.max(s, Math.min(t.length, end | 0));
    return `${esc(t.slice(0, s))}<mark>${esc(t.slice(s, e))}</mark>${esc(t.slice(e))}`;
  }
  function loadShortlist() {
    try { return JSON.parse(localStorage.getItem("kasauti.shortlist") || "{}") || {}; }
    catch { return {}; }
  }
  function saveShortlist() {
    try { localStorage.setItem("kasauti.shortlist", JSON.stringify(state.shortlist)); } catch { /* private mode */ }
    $("#slCount").textContent = Object.keys(state.shortlist).length;
  }
  async function api(path, opts = {}) {
    const res = await fetch(API + path, opts);
    let body = null;
    try { body = await res.json(); } catch { /* non-JSON */ }
    if (!res.ok) {
      const detail = body && body.detail;
      const msg = Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : (detail || res.statusText);
      throw new Error(`${res.status}: ${msg}`);
    }
    return body;
  }

  // ── engine status ───────────────────────────────────
  async function refreshHealth() {
    const el = $("#engine");
    try {
      const h = await api("/health");
      const e = h.engine || {};
      const mode = e.mode === "sample" ? `sample · ${e.dataset}` : (e.mode || "index");
      const backend = e.embedding_backend === "hashing" ? "lite engine" : "transformer";
      el.textContent = `${h.index_size.toLocaleString()} profiles · ${backend} · ${mode}`;
      el.className = "engine " + (h.status === "healthy" ? "ok" : "bad");
    } catch {
      el.textContent = "engine offline";
      el.className = "engine bad";
    }
  }

  // ── controls ────────────────────────────────────────
  function buildControls() {
    const ex = $("#examples");
    for (const q of EXAMPLES) {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = q;
      b.addEventListener("click", () => { $("#q").value = q; search(); });
      ex.appendChild(b);
    }
    const box = $("#sliders");
    for (const [key, label, def] of SLIDERS) {
      const l = document.createElement("label");
      l.innerHTML = `<span>${esc(label)}</span><input type="range" min="0" max="100" step="5" value="${def}" data-key="${key}"><output>${def}</output>`;
      const input = $("input", l);
      input.addEventListener("input", () => { $("output", l).textContent = input.value; state.slidersTouched = true; });
      box.appendChild(l);
    }
    $("#resetWeights").addEventListener("click", () => {
      for (const input of box.querySelectorAll("input")) {
        const def = SLIDERS.find((s) => s[0] === input.dataset.key)[2];
        input.value = def; input.nextElementSibling.textContent = def;
      }
      state.slidersTouched = false;
    });
    const legend = $("#legend");
    legend.innerHTML = Object.entries(DIM_LABEL).slice(0, 8)
      .map(([d, l]) => `<span><i style="background:${dimColor(d)}"></i>${esc(l)}</span>`).join("");
  }

  function requestBody() {
    const body = {
      query: $("#q").value.trim(),
      use_turbo: true,
      max_results: Number($("#maxResults").value) || 15,
      trajectory_mode: $("#currentTenure").checked ? "include_current" : "completed_roles",
      skill_evidence: $("#workHistory").checked ? "work_history" : "listed",
      filters: {},
    };
    const city = $("#city").value.trim();
    if (city) body.filters.location = city;
    const minY = $("#minYears").value;
    if (minY !== "") body.filters.min_experience_years = Number(minY);
    if (state.slidersTouched) {
      body.slider_weights = {};
      for (const input of $("#sliders").querySelectorAll("input")) body.slider_weights[input.dataset.key] = Number(input.value);
    }
    return body;
  }

  // ── search ──────────────────────────────────────────
  async function search() {
    const body = requestBody();
    if (!body.query) { $("#q").focus(); return; }
    state.query = body.query;
    const go = $("#go");
    go.disabled = true; go.textContent = "Ranking…";
    $("#intro").hidden = true;
    $("#workspace").hidden = false;
    $("#results").innerHTML = `<li class="loading">Searching the pool…</li>`;
    try {
      const t0 = performance.now();
      const res = await api("/search", {
        method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
      });
      state.results = res.results || [];
      const ms = Math.round(performance.now() - t0);
      $("#resultsTitle").textContent = `${state.results.length} ranked for “${body.query}”`;
      $("#resultsMeta").textContent = `${ms} ms · ${(res.search_metadata?.methods_used || []).join(", ")}`;
      renderResults();
      renderFairness();
      try { history.replaceState(null, "", "?q=" + encodeURIComponent(body.query)); } catch { /* file:// */ }
      if (state.results.length) select(state.results[0].profile_id);
      else $("#detail").innerHTML = `<div class="empty">No candidates matched. Try fewer constraints.</div>`;
    } catch (err) {
      $("#results").innerHTML = `<li class="error-box">${esc(err.message)}</li>`;
    } finally {
      go.disabled = false; go.textContent = "Rank candidates";
    }
  }

  // How many of the ranked candidates actually prove each requirement: a quick
  // reality check on the brief ("only 3 of 15 can show Airflow in a real job").
  function renderCoverage() {
    const box = $("#coverage");
    const tally = new Map();
    for (const r of state.results) {
      for (const q of r.explanation?.requirements || []) {
        if (q.kind !== "skill") continue;
        const t = tally.get(q.requirement) || { proven: 0, claimed: 0, missing: 0 };
        if (q.status === "missing") t.missing++;
        else if (q.strength === "mentioned") t.claimed++;
        else t.proven++;
        tally.set(q.requirement, t);
      }
    }
    if (!tally.size) { box.hidden = true; return; }
    const n = state.results.length;
    box.hidden = false;
    box.innerHTML = `<span class="cov-title">Of these ${n}</span>` + [...tally.entries()].map(([name, t]) => `
      <span class="cov" title="${t.proven} proven · ${t.claimed} only claimed · ${t.missing} no evidence">
        <b>${esc(name)}</b>
        <span class="cov-bar"><i class="p" style="width:${(t.proven / n) * 100}%"></i><i class="c" style="width:${(t.claimed / n) * 100}%"></i></span>
        <span class="cov-n">${t.proven} proven${t.claimed ? ` · ${t.claimed} claimed` : ""}</span>
      </span>`).join("");
  }

  // Fairness check on the ranked list (reports only; never re-ranks).
  async function renderFairness() {
    const box = $("#fairness");
    if (!state.results.length) { box.hidden = true; return; }
    try {
      const rep = await api("/fairness", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ items: state.results.map((r) => ({ profile_id: r.profile_id, score: r.scores.overall })) }),
      });
      const word = { ok: "no gap", flag: "gap", insufficient: "too few to compare" };
      box.innerHTML = `<span class="cov-title">Fairness check</span>` + rep.dimensions.map((d) => `
        <span class="fair ${esc(d.status)}" title="${esc(d.comparison)}: ${esc(d.observations.join(" · "))}">
          <i></i>${esc(d.label)} <small>${esc(word[d.status] || d.status)}${d.status === "flag" ? ` +${Math.round(d.gap * 100)} pts` : ""}</small>
        </span>`).join("");
      box.hidden = false;
    } catch { box.hidden = true; }
  }

  function renderResults() {
    renderCoverage();
    const ol = $("#results");
    ol.innerHTML = "";
    const tpl = $("#tplCard");
    for (const r of state.results) {
      const li = tpl.content.firstElementChild.cloneNode(true);
      const ex = r.explanation || {};
      li.dataset.pid = r.profile_id;
      $(".rank", li).textContent = r.rank;
      $(".name", li).textContent = r.name || r.profile_id;
      const badge = $(".badge", li);
      badge.textContent = ex.verdict || "";
      badge.className = "badge " + (ex.verdict || "");
      if (state.newIds.has(r.profile_id)) $(".new", li).hidden = false;
      const yrs = r.experience_years != null ? `${Number(r.experience_years).toFixed(1)} yrs` : "";
      $(".role", li).textContent = [r.current_title, r.current_company && `@ ${r.current_company}`, r.location, yrs]
        .filter(Boolean).join(" · ");
      const chips = $(".chips", li);
      const reqs = (ex.requirements || []).filter((q) => q.kind !== "role");
      for (const q of reqs.slice(0, 6)) {
        const c = document.createElement("span");
        c.className = "chip " + q.status;
        c.textContent = q.requirement;
        c.title = q.detail;
        chips.appendChild(c);
      }
      if ((ex.risks || []).some((x) => x.startsWith("Integrity flag"))) {
        li.classList.add("flagged");
        const c = document.createElement("span");
        c.className = "chip risk"; c.textContent = "integrity flag";
        chips.appendChild(c);
      }
      $(".score", li).innerHTML = `${pct(r.scores.overall)}<small>/100</small>`;
      const bar = $(".bar", li);
      for (const c of ex.contributions || []) {
        if (c.points <= 0) continue;
        const s = document.createElement("span");
        s.style.width = `${c.points * 100}%`;
        s.style.background = dimColor(c.dimension);
        s.title = `${c.label}: +${pct(c.points)}`;
        bar.appendChild(s);
      }
      const star = $(".star", li);
      const on = !!state.shortlist[r.profile_id];
      star.classList.toggle("on", on); star.textContent = on ? "★" : "☆";
      star.addEventListener("click", (e) => { e.stopPropagation(); toggleShortlist(r); });
      li.addEventListener("click", () => {
        select(r.profile_id);
        if (window.matchMedia("(max-width: 1020px)").matches) $("#detail").scrollIntoView({ behavior: "smooth" });
      });
      li.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(r.profile_id); }
        if (e.key === "ArrowDown") li.nextElementSibling?.focus();
        if (e.key === "ArrowUp") li.previousElementSibling?.focus();
      });
      ol.appendChild(li);
    }
  }

  function select(pid) {
    state.active = pid;
    for (const li of document.querySelectorAll(".card")) li.classList.toggle("active", li.dataset.pid === pid);
    const r = state.results.find((x) => x.profile_id === pid);
    if (r) renderDetail(r);
  }

  function renderDetail(r) {
    const ex = r.explanation;
    const d = $("#detail");
    if (!ex) { d.innerHTML = `<div class="empty">No explanation available for this result.</div>`; return; }
    const maxPts = Math.max(0.0001, ...ex.contributions.map((c) => c.points));
    const contrib = ex.contributions.map((c) => `
      <span class="lbl">${esc(c.label)}</span>
      <span class="track" title="score ${pct(c.score)} × weight ${pct(c.weight)}%"><span class="fill" style="width:${(c.points / maxPts) * 100}%;background:${dimColor(c.dimension)}"></span></span>
      <span class="pts ${c.points <= 0 ? "muted" : ""}">+${(c.points * 100).toFixed(1)}</span>`).join("");
    const factor = ex.title_factor < 0.999
      ? `<div class="factor">Role fit multiplier ×${ex.title_factor.toFixed(2)}: the current title is on a different track from the role asked for.</div>` : "";
    const reqs = ex.requirements.map((q) => `
      <div class="req">
        <div class="req-top"><span class="req-name">${esc(q.requirement)}</span><span class="req-kind">${esc(q.kind)}</span>
          <span class="req-status ${esc(q.status)}">${esc(q.status)}</span></div>
        <div class="req-detail">${esc(q.detail)}</div>
        ${q.evidence.map((ev) => `
          <div class="ev ${esc(ev.strength)}">
            <div class="ev-src"><span>${esc(ev.label)}</span><span class="str">${esc(STRENGTH_LABEL[ev.strength] || ev.strength)}</span></div>
            <q>${highlight(ev.text, ev.start, ev.end)}</q>
          </div>`).join("")}
      </div>`).join("") || `<p class="hint">The query had no checkable requirements (try naming skills, a city or a years band).</p>`;
    const list = (items, cls) => `<ul class="list ${cls}">${items.length ? items.map((x) => `<li>${esc(x)}</li>`).join("") : `<li class="none">None found</li>`}</ul>`;
    const t = ex.trajectory || {};
    const yrs = r.experience_years != null ? `${Number(r.experience_years).toFixed(1)} yrs` : "";
    d.scrollTop = 0;
    d.innerHTML = `
      <div class="d-head">
        <div class="d-top">
          <div>
            <h2 class="d-name">${esc(r.name || r.profile_id)}</h2>
            <div class="d-sub">${esc([r.current_title, r.current_company && "@ " + r.current_company, r.location, yrs].filter(Boolean).join(" · "))}</div>
          </div>
          <div class="d-score"><b>${pct(ex.overall)}</b><span>rank #${r.rank} · ${esc(ex.verdict)}</span></div>
        </div>
        <p class="d-headline">${esc(ex.headline)}</p>
      </div>
      <div class="d-sec"><h3>Where the score comes from</h3><div class="contrib">${contrib}</div>${factor}</div>
      <div class="d-sec"><h3>Requirements, with evidence</h3>${reqs}</div>
      <div class="d-sec two">
        <div><h3>Strengths</h3>${list(ex.strengths, "pos")}</div>
        <div><h3>Risks</h3>${list(ex.risks, "neg")}</div>
      </div>
      <div class="d-sec"><h3>Career timeline</h3><div class="timeline" id="timeline"><span class="hint">Loading roles…</span></div></div>
      <div class="d-sec"><h3>Career trajectory</h3>
        <div class="facts">
          <span>Roles <b>${esc(t.roles ?? "–")}</b></span>
          <span>Avg tenure <b>${t.avg_tenure_months != null ? esc(Math.round(t.avg_tenure_months)) + " mo" : "–"}</b></span>
          <span>Current role <b>${t.current_tenure_months != null ? esc(t.current_tenure_months) + " mo" : "–"}</b></span>
          <span>Mode <b>${t.mode === "include_current" ? "incl. current role" : "completed roles"}</b></span>
        </div>
      </div>
      <div class="d-sec"><button class="btn ghost small" id="detailStar" type="button">${state.shortlist[r.profile_id] ? "★ On shortlist" : "☆ Add to shortlist"}</button>
        <span class="meta" style="margin-left:8px">${esc(r.profile_id)}</span></div>`;
    $("#detailStar").addEventListener("click", () => toggleShortlist(r));
    renderTimeline(r);
  }

  // ── career timeline (roles as bars; roles that hold evidence are marked) ──
  const profileCache = new Map();
  async function renderTimeline(r) {
    const pid = r.profile_id;
    let profile = profileCache.get(pid);
    if (!profile) {
      try { profile = await api("/profiles/" + encodeURIComponent(pid)); profileCache.set(pid, profile); }
      catch { profile = null; }
    }
    const box = $("#timeline");
    if (!box || state.active !== pid) return; // user moved on
    const roles = (profile?.experience || []).filter((e) => e.start_date);
    if (!roles.length) { box.innerHTML = `<span class="hint">No dated roles in this profile.</span>`; return; }
    const ym = (d) => { const [y, m] = String(d).split("-").map(Number); return y + ((m || 1) - 1) / 12; };
    const now = new Date();
    const nowY = now.getFullYear() + now.getMonth() / 12;
    const spans = roles.map((e) => ({ e, a: ym(e.start_date), b: e.end_date ? ym(e.end_date) : nowY }))
      .filter((x) => Number.isFinite(x.a) && Number.isFinite(x.b) && x.b >= x.a)
      .sort((x, y) => y.a - x.a);
    if (!spans.length) { box.innerHTML = `<span class="hint">Role dates could not be read.</span>`; return; }
    const lo = Math.floor(Math.min(...spans.map((x) => x.a)));
    const hi = Math.ceil(Math.max(...spans.map((x) => x.b)));
    const width = Math.max(1, hi - lo);
    const evidenceLabels = new Set();
    for (const q of r.explanation?.requirements || []) for (const ev of q.evidence) {
      if (ev.source === "experience") evidenceLabels.add(ev.label.replace(/ \(.*\)$/, ""));
    }
    const rows = spans.map(({ e, a, b }) => {
      const months = Math.round((b - a) * 12);
      const proof = evidenceLabels.has(`${e.title} @ ${e.company}`);
      const current = !e.end_date;
      return `<div class="tl-row">
        <div class="tl-lbl" title="${esc(e.title)} @ ${esc(e.company)}"><b>${esc(e.title)}</b> <span>@ ${esc(e.company)}</span></div>
        <div class="tl-track"><span class="tl-bar ${current ? "cur" : ""}" style="left:${((a - lo) / width) * 100}%;width:${Math.max(1.5, ((b - a) / width) * 100)}%"></span></div>
        <div class="tl-meta">${months} mo${proof ? ` <i class="tl-proof" title="This role proves at least one requirement">proof</i>` : ""}</div>
      </div>`;
    }).join("");
    const ticks = [];
    const step = width > 12 ? 4 : width > 6 ? 2 : 1;
    for (let y = lo; y <= hi; y += step) ticks.push(`<span style="left:${((y - lo) / width) * 100}%">${y}</span>`);
    box.innerHTML = rows + `<div class="tl-row tl-axis"><div></div><div class="tl-ticks">${ticks.join("")}</div><div></div></div>`;
  }

  // ── shortlist + requirement matrix ──────────────────
  function toggleShortlist(r) {
    if (state.shortlist[r.profile_id]) delete state.shortlist[r.profile_id];
    else state.shortlist[r.profile_id] = {
      profile_id: r.profile_id, name: r.name, title: r.current_title, company: r.current_company,
      location: r.location, score: r.scores.overall, query: state.query, explanation: r.explanation,
    };
    saveShortlist();
    renderResults();
    if (state.active) select(state.active);
  }

  function renderMatrix() {
    const items = Object.values(state.shortlist);
    const wrap = $("#matrix");
    if (!items.length) { wrap.innerHTML = `<p class="hint">Star candidates in the results to compare them here.</p>`; return; }
    const reqNames = [];
    for (const it of items) for (const q of it.explanation?.requirements || []) {
      if (!reqNames.includes(q.requirement)) reqNames.push(q.requirement);
    }
    const head = items.map((it) => `<th>${esc(it.name || it.profile_id)}<small>${esc(it.title || "")}</small><small>${pct(it.score)}/100</small></th>`).join("");
    const rows = reqNames.map((name) => {
      const cells = items.map((it) => {
        const q = (it.explanation?.requirements || []).find((x) => x.requirement === name);
        if (!q) return `<td><span class="cell">—</span></td>`;
        const cls = q.status === "missing" ? "missing" : (q.strength || q.status);
        const label = q.status === "missing" ? "missing" : (q.strength ? STRENGTH_LABEL[q.strength] : q.status);
        return `<td title="${esc(q.detail)}"><span class="cell ${esc(cls)}">${esc(label)}</span></td>`;
      }).join("");
      return `<tr><td>${esc(name)}</td>${cells}</tr>`;
    }).join("");
    const risks = `<tr><td>Risks</td>${items.map((it) => `<td style="text-align:left;font-size:12px">${esc((it.explanation?.risks || []).join(" · ") || "—")}</td>`).join("")}</tr>`;
    wrap.innerHTML = `<table class="matrix"><thead><tr><th>Requirement</th>${head}</tr></thead><tbody>${rows}${risks}</tbody></table>`;
  }

  function exportCsv() {
    const items = Object.values(state.shortlist);
    if (!items.length) return;
    const cell = (v) => {
      let s = String(v ?? "");
      if (/^[=+\-@\t\r]/.test(s)) s = "'" + s; // no formula injection in spreadsheets
      return `"${s.replace(/"/g, '""')}"`;
    };
    const lines = [["profile_id", "name", "title", "company", "location", "score", "verdict", "query", "headline", "requirements", "risks"].join(",")];
    for (const it of items) {
      const ex = it.explanation || {};
      const reqs = (ex.requirements || []).map((q) => `${q.requirement}: ${q.status}${q.strength ? " (" + STRENGTH_LABEL[q.strength] + ")" : ""}`).join("; ");
      lines.push([it.profile_id, it.name, it.title, it.company, it.location, pct(it.score), ex.verdict, it.query, ex.headline, reqs, (ex.risks || []).join("; ")].map(cell).join(","));
    }
    const blob = new Blob(["﻿" + lines.join("\r\n")], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "kasauti-shortlist.csv";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  // ── ingest ──────────────────────────────────────────
  async function ingestFile(file) {
    const out = $("#ingestResult");
    out.innerHTML = `<p class="hint">Uploading ${esc(file.name)}…</p>`;
    const fd = new FormData();
    fd.append("file", file, file.name);
    try {
      const r = await api("/ingest", { method: "POST", body: fd });
      for (const id of [...(r.profile_ids || [])]) state.newIds.add(id);
      out.innerHTML = `
        <p class="ok">${r.successful} of ${r.total_profiles} profiles in the pool.</p>
        <p>${r.created} new · ${r.updated} updated · ${r.unchanged} unchanged${r.failed ? ` · <span class="err">${r.failed} rejected</span>` : ""}</p>
        ${r.errors?.length ? `<ul class="err">${r.errors.slice(0, 8).map((e) => `<li>${esc(e)}</li>`).join("")}</ul>` : ""}
        <p class="hint">They are embedded and indexed: the next search can rank them.</p>`;
      refreshHealth();
      if (state.query) search();
    } catch (err) {
      out.innerHTML = `<p class="err">${esc(err.message)}</p>`;
    }
  }

  function wireDrawers() {
    const open = (id) => { const el = $(id); el.hidden = false; $("[data-close]", el).focus(); };
    $("#openShortlist").addEventListener("click", () => { renderMatrix(); open("#shortlist"); });
    $("#openIngest").addEventListener("click", () => open("#ingest"));
    for (const dr of document.querySelectorAll(".drawer")) {
      dr.addEventListener("click", (e) => { if (e.target === dr || e.target.closest("[data-close]")) dr.hidden = true; });
    }
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") for (const dr of document.querySelectorAll(".drawer")) dr.hidden = true;
    });
    $("#exportCsv").addEventListener("click", exportCsv);
    $("#clearSl").addEventListener("click", () => { state.shortlist = {}; saveShortlist(); renderMatrix(); renderResults(); });
    const drop = $("#drop");
    $("#file").addEventListener("change", (e) => { if (e.target.files[0]) ingestFile(e.target.files[0]); });
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("over"));
    drop.addEventListener("drop", (e) => {
      e.preventDefault(); drop.classList.remove("over");
      if (e.dataTransfer.files[0]) ingestFile(e.dataTransfer.files[0]);
    });
    $("#loadExample").addEventListener("click", async () => {
      const res = await fetch("/static/example-profile.json");
      const blob = await res.blob();
      await ingestFile(new File([blob], "example-profile.json", { type: "application/json" }));
    });
  }

  // ── why not? explain a profile that is not (or not high) in the list ──
  async function explainAny(pid) {
    const body = requestBody();
    if (!state.query || !pid) return;
    const d = $("#detail");
    d.innerHTML = `<div class="loading">Scoring ${esc(pid)} against “${esc(state.query)}”…</div>`;
    try {
      const ex = await api("/explain", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ query: state.query, profile_id: pid, slider_weights: body.slider_weights || {},
          trajectory_mode: body.trajectory_mode, skill_evidence: body.skill_evidence }),
      });
      let p = profileCache.get(pid);
      if (!p) { p = await api("/profiles/" + encodeURIComponent(pid)); profileCache.set(pid, p); }
      const inList = state.results.find((x) => x.profile_id === pid);
      const r = {
        profile_id: pid, rank: inList ? inList.rank : "–", name: p.personal?.name,
        current_title: p.professional?.current_title, current_company: p.professional?.current_company,
        location: p.personal?.location?.city, experience_years: p.professional?.total_experience_years,
        scores: { overall: ex.overall }, explanation: ex,
      };
      state.active = pid;
      for (const li of document.querySelectorAll(".card")) li.classList.toggle("active", li.dataset.pid === pid);
      renderDetail(r);
      if (!inList) $(".d-score span").textContent = `not in top ${state.results.length} · ${ex.verdict}`;
    } catch (err) {
      d.innerHTML = `<div class="error-box">${esc(err.message.startsWith("404") ? `No profile with ID ${pid}` : err.message)}</div>`;
    }
  }

  // ── boot ────────────────────────────────────────────
  buildControls();
  wireDrawers();
  saveShortlist();
  refreshHealth();
  $("#searchForm").addEventListener("submit", (e) => { e.preventDefault(); search(); });
  $("#whyNot").addEventListener("submit", (e) => { e.preventDefault(); explainAny($("#whyId").value.trim()); });
  const initial = new URLSearchParams(location.search).get("q");
  if (initial) { $("#q").value = initial; search(); }
})();
