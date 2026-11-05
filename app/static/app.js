const $ = (s, root = document) => root.querySelector(s);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};
const icon = (name, weight = "") => el("i", `ph${weight ? `-${weight}` : ""} ph-${name}`);
const app = $("#app");
const mobile = () => matchMedia("(max-width: 1000px)").matches;

/* ---------- Files ---------- */
const EXT = (n) => (n.match(/\.([a-z0-9]+)$/i) || [, ""])[1].toLowerCase();
const KIND = {
  image: ["jpg", "jpeg", "png", "webp", "bmp", "heic", "avif", "tif", "tiff", "ico", "svg"],
  gif: ["gif"],
  video: ["mp4", "mov", "m4v", "webm", "mkv", "avi", "ts"],
  audio: ["mp3", "wav", "m4a", "m4r", "aac", "flac", "ogg", "opus", "aiff", "aif"],
  pdf: ["pdf"],
  text: ["srt", "vtt", "ass", "txt", "csv", "json", "m3u8"],
};
const kindOf = (n) => Object.keys(KIND).find((k) => KIND[k].includes(EXT(n))) || "file";
const KIND_ICON = { image: "image", gif: "gif", video: "film-strip", audio: "waveform", pdf: "file-pdf", text: "file-text", file: "file" };
const PLAYABLE = { video: ["mp4", "mov", "m4v", "webm"], audio: ["mp3", "wav", "m4a", "aac", "flac", "ogg", "opus"] };
const canPlay = (n) => (PLAYABLE[kindOf(n)] || []).includes(EXT(n));
const canShow = (n) => ({ image: !["heic", "tif", "tiff"].includes(EXT(n)), gif: true, video: canPlay(n), audio: canPlay(n) })[kindOf(n)] || false;
const group = (n) => ({ image: "visual", gif: "visual", video: "visual", audio: "audio" })[kindOf(n)];
const fileUrl = (where, name) => `/api/jobs/${job.id}/file/${where}/${name.split("/").map(encodeURIComponent).join("/")}`;
const fmtSize = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1e3))} KB`);
const fmtDur = (s) => (s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`);
const fmtClock = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${(s % 60).toFixed(1).padStart(4, "0")}`;

// The server describes files as e.g. "video h264 1280x720 30fps, audio aac stereo 48kHz, 16.0s, 5.7 MB".
function parseDesc(desc = "") {
  const dims = desc.match(/(\d+)x(\d+)/);
  const dur = desc.match(/(?:^|, )([\d.]+)s(?=,|$)/);
  const fps = desc.match(/([\d.]+)fps/);
  const size = desc.match(/([\d.]+) (KB|MB|GB)\s*$/);
  return {
    dims: dims ? `${dims[1]}×${dims[2]}` : null,
    dur: dur ? +dur[1] : null,
    fps: fps ? +fps[1] : null,
    size: size ? +size[1] * { KB: 1e3, MB: 1e6, GB: 1e9 }[size[2]] : null,
  };
}

function specRows(desc = "") {
  const tidy = (s) => s.replace(/(\d+)x(\d+)/, "$1×$2").replace(/(\d)fps/, "$1 fps").split(" ").join(" · ");
  return desc.split(", ").map((p) => {
    const m = p.match(/^(video|audio|image|pdf)\s+(.*)/);
    if (m) return [m[1][0].toUpperCase() + m[1].slice(1), tidy(m[2])];
    if (/^[\d.]+s$/.test(p)) return ["Duration", fmtDur(parseFloat(p))];
    if (/^[\d.]+ (KB|MB|GB)$/.test(p)) return ["Size", p];
    return ["Details", p];
  });
}

// The model writes light Markdown. Escape first, then format, so no raw HTML gets through.
function md(text) {
  const esc = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const inline = (s) => s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>");
  const out = [];
  let list = false;
  for (const line of esc.split("\n")) {
    const m = line.match(/^\s*[-*]\s+(.*)/);
    if (m) { if (!list) { out.push("<ul>"); list = true; } out.push(`<li>${inline(m[1])}</li>`); continue; }
    if (list) { out.push("</ul>"); list = false; }
    if (line.trim()) out.push(`<p>${inline(line)}</p>`);
  }
  if (list) out.push("</ul>");
  const n = el("div", "md");
  n.innerHTML = out.join("");
  return n;
}

function thumb(name, src, cls = "thumb") {
  const box = el("div", cls);
  const k = kindOf(name);
  const fallback = () => box.replaceChildren(icon(KIND_ICON[k]));
  if ((k === "image" || k === "gif") && canShow(name)) {
    const img = el("img");
    img.src = src;
    img.alt = "";
    img.loading = "lazy";
    img.onerror = fallback;
    box.append(img);
  } else if (k === "video" && canPlay(name)) {
    const v = el("video");
    v.src = `${src}#t=0.5`;
    v.muted = true;
    v.preload = "metadata";
    v.onerror = fallback;
    box.append(v);
  } else fallback();
  return box;
}

function media(where, name, opts = {}) {
  const url = fileUrl(where, name);
  const k = kindOf(name);
  if ((k === "image" || k === "gif") && canShow(name)) {
    const img = el("img", opts.cls);
    img.src = url;
    img.alt = name;
    return img;
  }
  if (k === "video" && canPlay(name)) {
    const v = el("video", opts.cls);
    v.src = url;
    v.playsInline = true;
    Object.assign(v, opts.video || { controls: true });
    return v;
  }
  if (k === "audio" && canPlay(name)) {
    const card = el("div", "audio-card");
    const a = el("audio");
    a.src = url;
    a.controls = true;
    card.append(icon("waveform"), a);
    return card;
  }
  if (k === "pdf") {
    const fr = el("iframe");
    fr.src = url;
    fr.title = name;
    return fr;
  }
  if (k === "text") {
    const pre = el("pre");
    fetch(url).then((r) => r.text()).then((t) => { pre.textContent = t.slice(0, 200000); });
    return pre;
  }
  const n = el("div", "no-preview");
  n.append(icon(KIND_ICON[k]), el("span", null, "No preview for this format. Download it to open."));
  return n;
}

/* ---------- State ---------- */
let attached = [];  // File objects for a new job, or new uploads for a follow-up
let using = [];     // files the next follow-up works on: { where: "in"|"out", name, desc, size, tag }
let job = null;     // { id, state, inputs, files, lastText, source }
let turn = null;    // DOM refs for the turn being streamed

/* ---------- Attachments ---------- */
function attach(list) {
  for (const f of list) if (!attached.some((a) => a.name === f.name && a.size === f.size)) attached.push(f);
  renderAttachments();
}

function chip(name, src, meta, onRemove, tagCls = "") {
  const li = el("li", `att${tagCls}`);
  const text = el("div", "att-text");
  text.append(el("span", "att-name", name), el("span", "att-meta", meta));
  const rm = el("button");
  rm.type = "button";
  rm.setAttribute("aria-label", `Don't use ${name}`);
  rm.append(icon("x"));
  rm.onclick = onRemove;
  li.append(thumb(name, src), text, rm);
  return li;
}

// The latest result is the default input for a follow-up, like an editor keeps working on the current version.
function useFiles(list) {
  using = list;
  renderAttachments();
}

function renderAttachments() {
  const ul = $("#attachments");
  ul.replaceChildren();
  if (job) {
    using.forEach((f, i) => ul.append(chip(f.name, fileUrl(f.where, f.name), f.tag, () => { using.splice(i, 1); renderAttachments(); }, " using")));
    const orig = job.originals || [];
    const onOrig = using.length === orig.length && using.every((u) => u.where === "in" && orig.some((o) => o.name === u.name));
    if (orig.length && !onOrig && !attached.length) {
      const b = el("button", "use-orig");
      b.type = "button";
      b.append(icon("arrow-counter-clockwise"), el("span", null, "Use original"));
      b.onclick = () => useFiles(orig.map((o) => ({ ...o, where: "in", tag: "Original" })));
      ul.append(b);
    }
  }
  attached.forEach((f, i) => {
    const li = el("li", "att");
    const text = el("div", "att-text");
    text.append(el("span", "att-name", f.name), el("span", "att-meta", `${EXT(f.name).toUpperCase() || "FILE"} · ${fmtSize(f.size)}`));
    const rm = el("button");
    rm.type = "button";
    rm.setAttribute("aria-label", `Remove ${f.name}`);
    rm.append(icon("x"));
    rm.onclick = () => { attached.splice(i, 1); renderAttachments(); };
    li.append(thumb(f.name, URL.createObjectURL(f)), text, rm);
    ul.append(li);
  });
  $("#dropzone").classList.toggle("has-files", attached.length > 0);
  ul.hidden = !ul.children.length;
  sync();
}

/* ---------- Views ---------- */
function closeJob() {
  if (job && job.source) job.source.close();
  if (turn) clearInterval(turn.tick);
  job = null;
  turn = null;
  using = [];
}

function newJob() {
  closeJob();
  closePanel();
  app.classList.remove("has-job");
  $("#title").textContent = "";
  $("#thread").replaceChildren();
  history.replaceState(null, "", "/");
  markRecent();
  sync();
  $("#prompt").focus();
}

function openJob(id) {
  closeJob();
  closePanel();
  attached = [];
  using = [];
  job = { id, state: "running", inputs: [], files: [], lastText: "" };
  renderAttachments();
  app.classList.add("has-job");
  $("#thread").replaceChildren();
  history.replaceState(null, "", `/#${id}`);
  markRecent();
  const src = new EventSource(`/api/jobs/${id}/events`);
  job.source = src;
  src.onmessage = (m) => onEvent(JSON.parse(m.data));
  sync();
}

/* ---------- Streaming ---------- */
function onEvent(e) {
  const live = Date.now() / 1000 - e.t < 20;
  if (e.kind === "request") {
    if (!$("#thread").children.length) $("#title").textContent = e.text.split("\n")[0].slice(0, 90);
    if (e.inputs && e.inputs.length) {
      job.inputs = e.inputs;
      if (!job.originals) job.originals = e.inputs;
      const orig = (n) => job.originals.some((o) => o.name === n);
      useFiles(e.inputs.map((i) => ({ ...i, where: "in", tag: orig(i.name) ? "Original" : "Earlier result" })));
    }
    job.lastText = e.text;
    userMessage(e);
    startTurn(e.t, live);
    setState("running");
    if (live && job.inputs.length && $("#panel").hidden) openSource(job.inputs[0]);
  } else if (e.kind === "thinking") {
    if (!turn) startTurn(e.t, live);
    turn.label.textContent = turn.steps.children.length ? "Deciding what's next" : "Reading the files";
  } else if (e.kind === "step") {
    addStep(e);
  } else if (e.kind === "result") {
    endStep(e);
  } else if (e.kind === "reply" || e.kind === "error") {
    finishTurn(e, live);
  }
  if (live) $("#scroll").scrollTop = $("#scroll").scrollHeight;
}

function userMessage(e) {
  const u = el("div", "user");
  if (e.inputs && e.inputs.length) {
    const files = el("div", "user-files");
    for (const f of e.inputs) {
      const b = el("button", "file-chip");
      b.type = "button";
      const text = el("div", "att-text");
      const p = parseDesc(f.desc);
      const meta = [p.dur != null && fmtDur(p.dur), p.dims, fmtSize(f.size || p.size || 0)].filter(Boolean).join(" · ");
      text.append(el("span", "att-name", f.name), el("span", "att-meta", meta));
      b.append(thumb(f.name, fileUrl("in", f.name)), text);
      b.onclick = () => openSource(f);
      files.append(b);
    }
    u.append(files);
  }
  u.append(el("div", "user-text", e.text));
  $("#thread").append(u);
}

function startTurn(t0, live) {
  const wrap = el("div", "turn");
  const card = el("section", "progress");
  card.setAttribute("aria-label", "Progress");
  const head = el("div", "progress-head");
  const label = el("strong", null, "Reading the files");
  const timer = el("span", "timer", live ? "00:00.0" : "");
  head.append(el("span", "pulse"), label, timer);
  const steps = el("ol", "steps");
  card.append(el("div", "bar"), head, steps);
  wrap.append(card);
  $("#thread").append(wrap);
  turn = { wrap, card, label, steps, t0, cmds: [], fails: 0 };
  if (live) turn.tick = setInterval(() => { timer.textContent = fmtClock(Date.now() / 1000 - t0); }, 100);
}

function stepTitle(think, cmd) {
  const line = (think || "").replace(/[*`]/g, "").split(/\n|(?<=\.)\s/)[0].trim();
  if (line) return line.length > 140 ? `${line.slice(0, 137)}...` : line;
  return cmd ? `Running ${cmd.split(" ")[0]}` : "Running a command";
}

function addStep(e) {
  if (!turn) startTurn(e.t, false);
  const li = el("li", "running");
  const state = icon("circle-notch", "bold");
  state.classList.add("state", "run");
  const body = el("div", "step-body");
  const title = stepTitle(e.think, e.cmd);
  body.append(el("div", "step-title", title));
  const cmd = e.foreach ? `${e.cmd}   # for each ${e.foreach}` : e.cmd || "(no command)";
  const code = el("code", "step-cmd", cmd);
  code.title = "Click to expand";
  code.onclick = () => code.classList.toggle("open");
  body.append(code);
  const meta = el("div", "step-meta");
  const dur = el("span", "dur", "");
  meta.append(el("span", "tool", (e.cmd || "?").split(" ")[0]), dur);
  li.append(state, body, meta);
  li.dataset.t = e.t;
  turn.steps.append(li);
  turn.label.textContent = title;
  turn.cmds.push({ title, cmd });
  turn.cur = { li, state, body, dur };
}

function endStep(e) {
  const s = turn && turn.cur;
  if (!s) return;
  s.li.classList.remove("running");
  s.state.className = `${e.ok ? "ph-bold ph-check" : "ph-bold ph-x"} state ${e.ok ? "ok" : "bad"}`;
  s.dur.textContent = `${Math.max(0.1, e.t - s.li.dataset.t).toFixed(1)}s`;
  if (!e.ok) {
    turn.fails++;
    if (e.error) s.body.append(el("div", "step-error", e.error.trim().split("\n").slice(-3).join("\n")));
  }
  turn.cur = null;
}

const SUGGEST = {
  gif: ["Make it under 1 MB", "Make it loop smoothly", "Also export an MP4"],
  video: ["Make it smaller", "Make a 5 second GIF from it", "Extract the audio as MP3"],
  audio: ["Convert it to MP3", "Normalize the loudness", "Trim the silence at the start"],
  image: ["Convert it to WebP", "Resize it to 1080px wide", "Strip the metadata"],
  pdf: ["Make it grayscale"],
};

const FAIL = {
  loop: ["It got partway, then kept repeating itself", "The model kept retrying a command that wasn't working, so it was stopped. This task may be outside what it was trained on."],
  steps: ["It ran out of steps", "It used all 15 steps without finishing."],
  stopped: ["Stopped", "You stopped this run."],
  offline: ["Can't reach the model", "Start it with app/run.sh, then try again."],
};

function failReason(e) {
  if (e.reason) return e.reason;
  if (/repeated/.test(e.text)) return "loop";
  if (/step limit/.test(e.text)) return "steps";
  if (/reach the model/.test(e.text)) return "offline";
  return /^Stopped/.test(e.text) ? "stopped" : "other";
}

function finishTurn(e, live) {
  if (!turn) startTurn(e.t, false);
  clearInterval(turn.tick);
  const failed = e.kind === "error";
  const n = turn.steps.children.length;
  const secs = Math.max(1, Math.round(e.t - turn.t0));
  turn.card.classList.add("done");
  if (n === 0) turn.card.remove();
  else {
    turn.card.classList.add("collapsed");
    const sum = el("button", `summary${failed ? " failed" : ""}`);
    sum.type = "button";
    sum.setAttribute("aria-expanded", "false");
    const lead = icon(failed ? "warning" : "sparkle", failed ? "bold" : "fill");
    lead.classList.add(failed ? "warn" : "spark");
    const fails = turn.fails ? `, ${turn.fails} failed` : "";
    sum.append(lead, el("span", null, `${failed ? "Stopped after" : "Worked for"} ${secs}s · ran ${n} command${n === 1 ? "" : "s"}${fails}`), icon("caret-right", "bold"));
    sum.lastChild.classList.add("chev");
    const card = turn.card;
    sum.onclick = () => {
      const open = card.classList.toggle("collapsed") === false;
      sum.setAttribute("aria-expanded", open);
    };
    turn.wrap.insertBefore(sum, card);
  }
  const files = e.files || [];
  for (const f of files) { f.cmds = turn.cmds; f.inputs = job.inputs; job.files.push(f); }

  if (!failed && files.length) useFiles(files.map((f) => ({ ...f, where: "out", tag: "Latest result" })));
  if (failed) turn.wrap.append(failureCard(e, files));
  else {
    // The model lists each file it made; the result cards already show that, so drop those lines.
    const shown = new Set(files.map((f) => `out/${f.name}`));
    const text = (e.text || "").split("\n").filter((l) => !shown.has((l.match(/^\s*[-*]\s+(\S+?):/) || [])[1])).join("\n").trim();
    turn.wrap.append(md(text || "Done."));
    if (files.length) {
      const grid = el("div", "outputs");
      for (const f of files) grid.append(outputCard(f));
      turn.wrap.append(grid);
      const ideas = SUGGEST[kindOf(files[0].name)];
      if (ideas) {
        const follow = el("div", "follow");
        for (const t of ideas) {
          const b = el("button", null, t);
          b.type = "button";
          b.onclick = () => send(t);
          follow.append(b);
        }
        turn.wrap.append(follow);
      }
    }
  }
  if (live && files.length) openResult(files[0], failed, turn.wrap.querySelector(".out, .partials button"));
  else if (live && $("#panel").dataset.mode === "source") closePanel();
  turn = null;
  setState(failed ? "failed" : "done");
}

function failureCard(e, files) {
  const reason = failReason(e);
  const [head, body] = FAIL[reason] || ["Something went wrong", e.text];
  const box = el("section", "failure");
  const text = el("div");
  text.append(el("h2", null, head), el("p", null, files.length ? `${body} Here's what it made before stopping.` : body));
  box.append(text);
  if (files.length) {
    const grid = el("div", `partials${job.inputs.some((i) => kindOf(i.name) === "pdf") ? " paper" : ""}`);
    for (const f of files) {
      const b = el("button");
      b.type = "button";
      b.setAttribute("aria-label", f.name);
      b.append(thumb(f.name, fileUrl("out", f.name)), el("span", null, f.name));
      b.onclick = () => openResult(f, true, b);
      grid.append(b);
    }
    box.append(grid);
  }
  const actions = el("div", "actions");
  if (files.length) {
    const dl = el("a", "btn primary small");
    dl.href = `${fileUrl("out", files[0].name)}?download=1`;
    dl.append(icon("download-simple", "bold"), el("span", null, files.length > 1 ? "Download first file" : "Download"));
    actions.append(dl);
    if (files.length > 1) {
      const all = el("a", "btn small", `All ${files.length} files · .zip`);
      all.href = `/api/jobs/${job.id}/zip`;
      actions.append(all);
    }
  }
  if (reason !== "stopped" || job.lastText) {
    const retry = el("button", "btn small");
    retry.type = "button";
    retry.append(icon("arrow-clockwise", "bold"), el("span", null, "Try again"));
    retry.onclick = () => send(job.lastText);
    actions.append(retry);
  }
  box.append(actions);
  return box;
}

function outputCard(f) {
  const b = el("button", "out");
  b.type = "button";
  const text = el("div");
  text.style.minWidth = "0";
  text.append(el("div", "out-name", f.name), el("div", "out-meta", f.desc));
  b.append(thumb(f.name, fileUrl("out", f.name)), text, el("span", "out-open", "Open"));
  b.onclick = () => openResult(f, false, b);
  return b;
}

function setState(s) {
  if (!job) return;
  job.state = s;
  sync();
  loadRecents();
}

/* ---------- Panel ---------- */
let panelFile = null;
let panelSrc = null;

function showPanel(mode) {
  $("#panel").hidden = false;
  $("#panel").dataset.mode = mode;
  app.classList.add("panel-open");
  stopMedia();
}

function closePanel() {
  stopMedia();
  $("#panel").hidden = true;
  $("#panel").dataset.mode = "";
  app.classList.remove("panel-open");
  document.querySelectorAll(".out.active, .partials .active").forEach((c) => c.classList.remove("active"));
  panelFile = null;
}

function stopMedia() {
  document.querySelectorAll("#stage video, #stage audio").forEach((m) => m.pause());
}

function setBadge(text, cls) {
  const b = $("#panel-badge");
  b.className = `pill ${cls}`;
  b.textContent = text;
}

function openSource(f) {
  showPanel("source");
  panelFile = null;
  setBadge("Source", "");
  $("#panel-name").textContent = f.name;
  $("#tabs").hidden = true;
  $("#stage").classList.remove("paper");
  $("#stage").replaceChildren(media("in", f.name));
  $("#stats").replaceChildren();
  const spec = $("#spec");
  spec.replaceChildren();
  for (const [k, v] of specRows(f.desc)) spec.append(el("dt", null, k), el("dd", null, v));
  $("#panel-note").textContent = job.state === "running" ? "The result opens here when it's ready." : "";
  panelFoot("in", f, "Original");
}

function panelFoot(where, f, tag) {
  const result = where === "out";
  $("#panel-foot").hidden = false;
  $("#panel-download").hidden = !result;
  $("#panel-all").hidden = !result || job.files.length < 2;
  const b = $("#panel-use");
  const on = using.length === 1 && using[0].where === where && using[0].name === f.name;
  b.classList.toggle("primary", !result);
  b.disabled = on;
  b.lastChild.textContent = on ? "Next edit uses this" : "Use for next edit";
  b.onclick = () => {
    useFiles([{ ...f, where, tag }]);
    panelFoot(where, f, tag);
    $("#prompt").focus();
  };
}

function sourceFor(f) {
  const g = group(f.name);
  const ins = f.inputs || job.inputs;  // the files the turn that made f worked on
  return ins.find((i) => g && group(i.name) === g && i.name !== f.name) || (ins.length === 1 ? ins[0] : null);
}

function openResult(f, partial, card) {
  showPanel("result");
  panelFile = f;
  panelSrc = sourceFor(f);
  document.querySelectorAll(".out.active, .partials .active").forEach((c) => c.classList.remove("active"));
  if (card) card.classList.add("active");
  setBadge(partial ? "Partial result" : "Result", partial ? "warn" : "ok");
  $("#panel-name").textContent = f.name;
  const comparable = panelSrc && group(panelSrc.name) && group(panelSrc.name) === group(f.name) && canShow(panelSrc.name) && canShow(f.name);
  $('[data-tab="compare"]').hidden = !comparable;
  $('[data-tab="commands"]').hidden = !(f.cmds && f.cmds.length);
  $("#tabs").hidden = !comparable && !(f.cmds && f.cmds.length);
  $("#spec").replaceChildren();
  $("#panel-note").textContent = "";
  renderStats(f, panelSrc);
  $("#stage").classList.toggle("paper", !!panelSrc && kindOf(panelSrc.name) === "pdf");
  panelFoot("out", f, partial ? "Partial result" : "Result");
  const dl = $("#panel-download");
  dl.href = `${fileUrl("out", f.name)}?download=1`;
  $("#dl-label").textContent = `Download ${EXT(f.name).toUpperCase() || "file"}`;
  $("#dl-size").textContent = fmtSize(f.size);
  $("#panel-all").href = `/api/jobs/${job.id}/zip`;
  showTab("result");
}

function renderStats(f, src) {
  const box = $("#stats");
  box.replaceChildren();
  if (!src) return;
  const a = parseDesc(src.desc);
  const b = parseDesc(f.desc);
  const cell = (label, from, to, delta) => {
    const c = el("div", "stat");
    const row = el("div");
    row.append(el("span", "from", from), el("span", "from", "→"), el("span", "to", to));
    if (delta) row.append(delta);
    c.append(el("span", null, label), row);
    box.append(c);
  };
  const before = src.size || a.size;
  if (before && f.size) {
    const pct = Math.round((f.size / before - 1) * 100);
    const d = el("span", `delta${pct > 0 ? " up" : ""}`, `${pct > 0 ? "+" : pct < 0 ? "−" : "±"}${Math.abs(pct)}%`);
    cell("Size", fmtSize(before), fmtSize(f.size), d);
  }
  if (a.dims && b.dims) cell("Frame", a.dims, b.dims);
  if (a.dur != null && b.dur != null) cell("Length", fmtDur(a.dur), fmtDur(b.dur));
  const fmt = (name, p) => `${EXT(name).toUpperCase()}${p.fps ? ` · ${p.fps} fps` : ""}`;
  cell("Format", fmt(src.name, a), fmt(f.name, b));
}

function showTab(tab) {
  stopMedia();
  document.querySelectorAll("#tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.tab === tab));
  const stage = $("#stage");
  const f = panelFile;
  if (tab === "compare") return stage.replaceChildren(compareView(f, panelSrc));
  if (tab === "commands") {
    const list = el("div", "cmd-list");
    for (const c of f.cmds) {
      const d = el("div");
      d.append(el("span", null, c.title), el("code", null, c.cmd));
      list.append(d);
    }
    return stage.replaceChildren(list);
  }
  const m = media("out", f.name, { video: { controls: true, autoplay: true } });
  stage.replaceChildren(m);
}

function compareView(f, src) {
  const wrap = el("div", "compare");
  if (group(f.name) === "audio") return abAudio(wrap, src, f);
  const box = el("div", "compare-box");
  const after = media("out", f.name, { cls: "after", video: { loop: true, muted: false, preload: "auto" } });
  const before = media("in", src.name, { cls: "before", video: { loop: true, muted: true, autoplay: kindOf(f.name) !== "video", preload: "auto" } });
  const range = el("input");
  range.type = "range";
  range.min = 0;
  range.max = 100;
  range.value = 50;
  range.setAttribute("aria-label", "Before and after split");
  range.oninput = () => box.style.setProperty("--split", `${range.value}%`);
  const knob = el("span", "knob");
  knob.append(icon("arrows-horizontal", "bold"));
  box.append(after, before, el("span", "handle"), knob, range,
    el("span", "tag l", `Original · ${EXT(src.name).toUpperCase()}`), el("span", "tag r", `Result · ${EXT(f.name).toUpperCase()}`));
  wrap.append(box);
  if (after.tagName === "VIDEO") {
    if (before.tagName === "VIDEO") follow(after, before);
    wrap.append(abBar(after, before.tagName === "VIDEO" ? before : null));
  }
  return wrap;
}

// One clock: the result plays and the original follows it.
function follow(lead, other) {
  const sync = () => { if (Math.abs(other.currentTime - lead.currentTime) > 0.12) other.currentTime = lead.currentTime; };
  lead.addEventListener("timeupdate", sync);
  lead.addEventListener("seeked", sync);
  lead.addEventListener("play", () => { sync(); other.play(); });
  lead.addEventListener("pause", () => other.pause());
}

function abAudio(wrap, src, f) {
  const card = el("div", "audio-card");
  const a = el("audio");
  a.src = fileUrl("out", f.name);
  a.controls = true;
  const b = el("audio");
  b.src = fileUrl("in", src.name);
  b.muted = true;
  follow(a, b);
  card.append(icon("waveform"), a, b);
  wrap.append(card, abBar(a, b));
  return wrap;
}

function abBar(after, before) {
  const bar = el("div", "ab");
  const play = el("button");
  play.type = "button";
  play.setAttribute("aria-label", "Play or pause");
  const setPlay = () => play.replaceChildren(icon(after.paused ? "play" : "pause", "fill"));
  setPlay();
  after.addEventListener("play", setPlay);
  after.addEventListener("pause", setPlay);
  play.onclick = () => (after.paused ? after.play() : after.pause());
  bar.append(play);
  if (before) {
    [["Hear result", after], ["Hear original", before]].forEach(([label, m], i) => {
      const b = el("button", null, label);
      b.type = "button";
      b.setAttribute("aria-pressed", i === 0);
      b.onclick = () => {
        after.muted = m !== after;
        before.muted = m !== before;
        bar.querySelectorAll("[aria-pressed]").forEach((x, j) => x.setAttribute("aria-pressed", j === i));
      };
      bar.append(b);
    });
  }
  return bar;
}

/* ---------- Recents ---------- */
async function loadRecents() {
  let list = [];
  try { list = await (await fetch("/api/jobs")).json(); } catch { return; }
  const nav = $("#recents");
  nav.replaceChildren();
  if (!list.length) return nav.append(el("div", "recents-empty", "Your jobs will show up here."));
  const day = (t) => new Date(t * 1000).toDateString();
  const today = new Date().toDateString();
  const yesterday = new Date(Date.now() - 864e5).toDateString();
  let last = null;
  for (const j of list) {
    const d = day(j.created);
    const label = d === today ? "Today" : d === yesterday ? "Yesterday" : "Earlier";
    if (label !== last) { nav.append(el("h3", null, label)); last = label; }
    const a = el("a");
    a.href = `/#${j.id}`;
    a.dataset.id = j.id;
    const i = j.state === "running" ? icon("circle-notch", "bold") : j.state === "failed" ? icon("warning", "bold") : icon(KIND_ICON[kindOf(j.file || "")]);
    if (j.state === "running") i.classList.add("run");
    if (j.state === "failed") i.classList.add("bad");
    a.append(i, el("span", null, j.title));
    a.onclick = (ev) => { ev.preventDefault(); app.classList.remove("side-open"); openJob(j.id); };
    nav.append(a);
  }
  markRecent();
}

function markRecent() {
  document.querySelectorAll("#recents a").forEach((a) => {
    if (job && a.dataset.id === job.id) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
}

/* ---------- Composer ---------- */
function sync() {
  const running = !!job && job.state === "running";
  const text = $("#prompt").value.trim();
  $("#working").hidden = !running;
  $("#stop").hidden = !running;
  $("#send").hidden = running;
  $("#send").disabled = !text || (!job ? !attached.length : !using.length && !attached.length);
  $("#prompt").placeholder = !job ? "Describe the edit, e.g. make a 5 second GIF from the start, 480px wide"
    : running ? "Working on it. Press Stop to cancel." : "Ask for a change";
  $("#prompt").rows = job ? 1 : 2;
  $("#hint").textContent = running ? "" : !job && !attached.length ? "Attach a file to start" : "Enter to run · Shift+Enter for a new line";
}

function showError(msg) {
  $("#composer-error").textContent = msg;
}

function grow() {
  const t = $("#prompt");
  t.style.height = "auto";
  t.style.height = `${Math.min(t.scrollHeight, 220)}px`;
}

async function send(text) {
  showError("");
  if (!text) return;
  if (job && job.state === "running") return showError("Wait for this run to finish, or stop it.");
  if (!job && !attached.length) return showError("Attach at least one file first.");
  if (job && !using.length && !attached.length) return showError("Pick a file to work on, or attach one.");
  const fd = new FormData();
  $("#send").disabled = true;
  try {
    if (job) {
      fd.append("text", text);
      fd.append("use", JSON.stringify(using.map(({ where, name }) => ({ where, name }))));
      for (const f of attached) fd.append("files", f, f.name);
      const r = await fetch(`/api/jobs/${job.id}/reply`, { method: "POST", body: fd });
      if (!r.ok) throw new Error((await r.json()).detail || "Couldn't send that.");
      job.state = "running";
      attached = [];
      renderAttachments();
    } else {
      fd.append("prompt", text);
      for (const f of attached) fd.append("files", f, f.name);
      const r = await fetch("/api/jobs", { method: "POST", body: fd });
      if (!r.ok) throw new Error((await r.json()).detail || "Couldn't start the job.");
      const { id } = await r.json();
      openJob(id);
      loadRecents();
    }
    $("#prompt").value = "";
    grow();
  } catch (err) {
    showError(err.message);
  }
  sync();
}

async function stop() {
  if (!job) return;
  $("#hint").textContent = "Stopping after the current step...";
  await fetch(`/api/jobs/${job.id}/stop`, { method: "POST" }).catch(() => {});
}

async function checkModel() {
  let on = false;
  let label = "App offline";
  try {
    on = (await (await fetch("/api/health")).json()).model;
    label = on ? "Running on this Mac" : "Model offline";
  } catch { /* app server down */ }
  $("#model").className = `model ${on ? "on" : "off"}`;
  $("#model").textContent = label;
  $("#chip-dot").className = `dot ${on ? "on" : "off"}`;
}

/* ---------- Wire up ---------- */
let depth = 0;
const veil = $("#drop-veil");
document.addEventListener("dragenter", (e) => { if (e.dataTransfer.types.includes("Files")) { depth++; veil.hidden = false; } });
document.addEventListener("dragleave", () => { if (--depth <= 0) { depth = 0; veil.hidden = true; } });
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => {
  e.preventDefault();
  depth = 0;
  veil.hidden = true;
  if (e.dataTransfer.files.length) attach(e.dataTransfer.files);
});
$("#attach").onclick = () => $("#file-input").click();
$("#file-input").onchange = (e) => { attach(e.target.files); e.target.value = ""; };
$("#new-job").onclick = () => { app.classList.remove("side-open"); newJob(); };
$("#hide-side").onclick = () => (mobile() ? app.classList.remove("side-open") : app.classList.add("side-hidden"));
$("#show-side").onclick = () => (mobile() ? app.classList.toggle("side-open") : app.classList.remove("side-hidden"));
$("#panel-close").onclick = closePanel;
$("#stop").onclick = stop;
document.querySelectorAll("#tabs button").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("#panel").hidden) closePanel(); });
$("#composer").onsubmit = (e) => { e.preventDefault(); send($("#prompt").value.trim()); };
$("#prompt").addEventListener("input", () => { grow(); sync(); showError(""); });
$("#prompt").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("#composer").requestSubmit(); } });
document.querySelectorAll(".recipe-grid button").forEach((b) => (b.onclick = () => { $("#prompt").value = b.dataset.prompt; grow(); sync(); $("#prompt").focus(); }));

window.addEventListener("hashchange", () => {
  const id = location.hash.slice(1);
  if (id && (!job || job.id !== id)) openJob(id);
  else if (!id && job) newJob();
});

loadRecents();
checkModel();
setInterval(checkModel, 15000);
if (location.hash.length > 1) openJob(location.hash.slice(1));
else newJob();
