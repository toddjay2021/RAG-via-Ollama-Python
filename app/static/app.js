/* ============================================================
   RAG Knowledge Base Demo — frontend logic
   ============================================================ */

"use strict";

/* ------------------------------------------------- constants & state */

const EXT_META = {
  pdf: { label: "PDF", cls: "pdf" },
  docx: { label: "DOC", cls: "docx" },
  txt: { label: "TXT", cls: "txt" },
  md: { label: "MD", cls: "md" },
  csv: { label: "CSV", cls: "csv" },
};

const state = {
  documents: [],
  history: [],
  streaming: false,
};

/* ------------------------------------------------- DOM shortcuts */

const $ = (id) => document.getElementById(id);

const messagesEl = $("messages");
const heroEl = $("hero");
const suggestionsEl = $("suggestions");
const composerEl = $("composer");
const inputEl = $("questionInput");
const sendBtnEl = $("sendBtn");
const docListEl = $("docList");
const emptyKbEl = $("emptyKb");
const docCountEl = $("docCount");
const kbStatsEl = $("kbStats");
const uploadZoneEl = $("uploadZone");
const fileInputEl = $("fileInput");
const uploadProgressEl = $("uploadProgress");
const uploadStatusEl = $("uploadStatus");
const healthBannerEl = $("healthBanner");
const healthMessageEl = $("healthMessage");
const toastsEl = $("toasts");
const clearChatEl = $("clearChat");

/* ------------------------------------------------- utilities */

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Minimal safe markdown renderer: bold, italic, code, lists, [n] citations. */
function renderMarkdown(text) {
  const codeBlocks = [];
  let processed = text.replace(
    /```(\w*)\n?([\s\S]*?)(```|$)/g,
    (_m, _lang, code) => {
      codeBlocks.push(`<pre><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`);
      return `\u0000CODE${codeBlocks.length - 1}\u0000`;
    }
  );

  processed = escapeHtml(processed);
  processed = processed.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  processed = processed.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  processed = processed.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
  processed = processed.replace(
    /\[(\d{1,2})\]/g,
    '<span class="citation-ref" title="See source $1">$1</span>'
  );

  const lines = processed.split("\n");
  let html = "";
  let list = null;
  for (const line of lines) {
    const trimmed = line.trim();
    const ul = /^[-*]\s+(.*)/.exec(trimmed);
    const ol = /^\d+[.)]\s+(.*)/.exec(trimmed);
    if (ul) {
      if (list !== "ul") {
        if (list) html += `</${list}>`;
        html += "<ul>";
        list = "ul";
      }
      html += `<li>${ul[1]}</li>`;
    } else if (ol) {
      if (list !== "ol") {
        if (list) html += `</${list}>`;
        html += "<ol>";
        list = "ol";
      }
      html += `<li>${ol[1]}</li>`;
    } else {
      if (list) {
        html += `</${list}>`;
        list = null;
      }
      if (trimmed) html += `<p>${trimmed}</p>`;
    }
  }
  if (list) html += `</${list}>`;

  return html.replace(/\u0000CODE(\d+)\u0000/g, (_m, i) => codeBlocks[+i]);
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatTime(iso) {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
}

function extOf(filename) {
  const ext = (filename.split(".").pop() || "").toLowerCase();
  return EXT_META[ext] ? ext : null;
}

function toast(message, type = "info") {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.innerHTML = `<span class="dot"></span><span>${escapeHtml(message)}</span>`;
  toastsEl.appendChild(el);
  setTimeout(() => {
    el.classList.add("hide");
    el.addEventListener("animationend", () => el.remove(), { once: true });
  }, 4500);
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/* ------------------------------------------------- SSE over fetch POST */

async function* sseIterator(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const event = parseSseBlock(block);
      if (event) yield event;
    }
  }
}

function parseSseBlock(block) {
  let event = "message";
  const dataLines = [];
  for (const line of block.split("\n")) {
    if (line.startsWith(":")) continue; // keep-alive comment
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join("\n") };
}

/* ------------------------------------------------- documents */

async function refreshDocuments() {
  try {
    const res = await fetch("/api/documents");
    const data = await res.json();
    state.documents = data.documents || [];
    renderDocuments(data.total_chunks ?? 0);
  } catch {
    toast("Failed to load documents", "error");
  }
}

function renderDocuments(totalChunks) {
  docListEl.innerHTML = "";
  const docs = state.documents;
  docCountEl.textContent = docs.length;
  emptyKbEl.style.display = docs.length ? "none" : "flex";
  kbStatsEl.textContent = `${docs.length} document${docs.length === 1 ? "" : "s"} · ${totalChunks} chunks`;

  for (const doc of docs) {
    const ext = extOf(doc.filename) || "txt";
    const meta = EXT_META[ext];
    const card = document.createElement("div");
    card.className = "doc-card";
    card.innerHTML = `
      <div class="doc-icon ${meta.cls}">${meta.label}</div>
      <div class="doc-info">
        <div class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
        <div class="doc-meta">${doc.chunks} chunks · ${formatSize(doc.size)} · ${formatTime(doc.uploaded_at)}</div>
      </div>
      <button class="doc-delete" title="Remove from knowledge base" aria-label="Delete ${escapeHtml(doc.filename)}">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/>
          <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><path d="M10 11v6"/><path d="M14 11v6"/>
        </svg>
      </button>`;
    card.querySelector(".doc-delete").addEventListener("click", () => deleteDocument(doc));
    docListEl.appendChild(card);
  }
}

async function deleteDocument(doc) {
  if (!confirm(`Remove "${doc.filename}" and its ${doc.chunks} chunks from the knowledge base?`)) return;
  try {
    await fetch(`/api/documents/${doc.doc_id}`, { method: "DELETE" });
    toast(`Removed "${doc.filename}"`, "success");
    await refreshDocuments();
    if (state.documents.length === 0) clearChat(true);
    else if (state.history.length === 0) loadSuggestions();
  } catch {
    toast("Failed to delete document", "error");
  }
}

/* ------------------------------------------------- upload */

function setUploadProgress(visible, text) {
  uploadProgressEl.hidden = !visible;
  if (text) uploadStatusEl.textContent = text;
}

async function uploadFiles(fileList) {
  const files = [...fileList];
  const invalid = files.filter((f) => !extOf(f.name));
  const valid = files.filter((f) => extOf(f.name));

  for (const f of invalid) {
    toast(`Unsupported file type: ${f.name}`, "error");
  }
  if (!valid.length) return;

  setUploadProgress(true, `Processing ${valid.length} file${valid.length > 1 ? "s" : ""}…`);
  uploadZoneEl.classList.add("busy");

  const fd = new FormData();
  valid.forEach((f) => fd.append("files", f));

  try {
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) {
      toast(data.detail || "Upload failed", "error");
    } else {
      for (const r of data.results || []) {
        if (r.status === "ok") {
          toast(`${r.filename}: ${r.chunks} chunks indexed`, "success");
        } else {
          toast(`${r.filename}: ${r.detail}`, "error");
        }
      }
      await refreshDocuments();
      if (state.history.length === 0) {
        updateHero();
        await loadSuggestions();
      }
    }
  } catch {
    toast("Upload failed — is the server running?", "error");
  } finally {
    setUploadProgress(false);
    uploadZoneEl.classList.remove("busy");
    fileInputEl.value = "";
  }
}

function bindUpload() {
  uploadZoneEl.addEventListener("click", () => fileInputEl.click());
  uploadZoneEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") fileInputEl.click();
  });
  fileInputEl.addEventListener("change", () => uploadFiles(fileInputEl.files));

  ["dragenter", "dragover"].forEach((evt) =>
    uploadZoneEl.addEventListener(evt, (e) => {
      e.preventDefault();
      uploadZoneEl.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    uploadZoneEl.addEventListener(evt, (e) => {
      e.preventDefault();
      uploadZoneEl.classList.remove("dragover");
    })
  );
  uploadZoneEl.addEventListener("drop", (e) => {
    if (e.dataTransfer?.files?.length) uploadFiles(e.dataTransfer.files);
  });
}

/* ------------------------------------------------- suggestions */

async function loadSuggestions() {
  if (!state.documents.length) {
    suggestionsEl.innerHTML = "";
    return;
  }
  suggestionsEl.innerHTML = `<span class="suggestions-loading">Generating example questions…</span>`;
  try {
    const res = await fetch("/api/suggestions");
    const data = await res.json();
    const questions = (data.questions || []).filter(Boolean).slice(0, 3);
    suggestionsEl.innerHTML = "";
    questions.forEach((q, i) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "suggestion-chip";
      chip.style.animationDelay = `${i * 0.08}s`;
      chip.textContent = q;
      chip.addEventListener("click", () => {
        if (state.streaming) return;
        inputEl.value = q;
        composerEl.requestSubmit();
      });
      suggestionsEl.appendChild(chip);
    });
  } catch {
    suggestionsEl.innerHTML = "";
  }
}

function updateHero() {
  const hasDocs = state.documents.length > 0;
  $("heroTitle").textContent = hasDocs
    ? "Ask anything about your documents"
    : "Build your knowledge base to get started";
  $("heroSubtitle").textContent = hasDocs
    ? "I answer strictly from your uploaded content and show exactly which passages I used."
    : "Upload files using the panel on the left — PDF, DOCX, TXT, MD, or CSV — then ask questions and watch me cite my sources.";
}

/* ------------------------------------------------- chat messages */

function hideHero() {
  heroEl.style.display = "none";
}

function showHero() {
  heroEl.style.display = "";
}

function appendMessage(role) {
  const row = document.createElement("div");
  row.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role}`;
  if (role === "assistant") {
    avatar.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none">
        <path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z" fill="white"/>
      </svg>`;
  } else {
    avatar.textContent = "You";
  }

  const col = document.createElement("div");
  col.className = "bubble-col";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const sources = document.createElement("div");
  sources.className = "sources";

  col.appendChild(bubble);
  col.appendChild(sources);
  row.appendChild(avatar);
  row.appendChild(col);
  messagesEl.appendChild(row);
  scrollToBottom();

  return { row, bubble, sources };
}

function showTyping(bubble) {
  bubble.innerHTML = `<span class="typing"><span></span><span></span><span></span></span>`;
}

function renderSources(sourcesEl, sources, open = false) {
  if (!sources?.length) {
    sourcesEl.innerHTML = "";
    return;
  }
  sourcesEl.innerHTML = `
    <button type="button" class="sources-toggle ${open ? "open" : ""}">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="m6 9 6 6 6-6"/>
      </svg>
      Sources (${sources.length})
    </button>
    <div class="sources-list" ${open ? "" : "hidden"}></div>`;

  const list = sourcesEl.querySelector(".sources-list");
  sources.forEach((s, i) => {
    const card = document.createElement("div");
    card.className = "source-card";
    const snippet = (s.content || "").slice(0, 400);
    card.innerHTML = `
      <div class="source-head">
        <span class="source-index">${i + 1}</span>
        <span class="source-name" title="${escapeHtml(s.filename)}">${escapeHtml(s.filename)}</span>
        ${s.page != null ? `<span class="source-page">p. ${s.page}</span>` : ""}
        <span class="source-score">
          <span class="score-bar"><span class="score-fill" style="width:${s.score}%"></span></span>
          ${s.score}%
        </span>
      </div>
      <div class="source-snippet">${escapeHtml(snippet)}…</div>`;
    list.appendChild(card);
  });

  sourcesEl.querySelector(".sources-toggle").addEventListener("click", (e) => {
    e.currentTarget.classList.toggle("open");
    list.hidden = !list.hidden;
  });
}

function requestBody(question, stream) {
  return JSON.stringify({
    question,
    history: state.history.slice(0, -1).slice(-4),
    stream,
  });
}

function finalizeAnswer(bubble, sourcesEl, answer, sources) {
  bubble.classList.remove("streaming-cursor");
  bubble.innerHTML = renderMarkdown(answer);
  renderSources(sourcesEl, sources);
  if (answer) state.history.push({ role: "assistant", content: answer });
  scrollToBottom();
}

/** Try SSE streaming first; if the stream fails or hangs before the first
 *  event, fall back to a single non-streaming JSON request. */
async function sendQuestion(question) {
  if (state.streaming || !question.trim()) return;
  state.streaming = true;
  sendBtnEl.disabled = true;

  hideHero();
  appendMessage("user").bubble.textContent = question;
  state.history.push({ role: "user", content: question });

  const { bubble, sources: sourcesEl } = appendMessage("assistant");
  showTyping(bubble);

  let answer = "";
  let sources = [];
  let gotFirstEvent = false;
  let settled = false;

  const controller = new AbortController();
  // Watchdog: if nothing arrives within 20s, abort and use the JSON fallback.
  const watchdog = setTimeout(() => {
    if (!gotFirstEvent) controller.abort();
  }, 20000);

  const renderError = (message) => {
    bubble.classList.add("error");
    bubble.classList.remove("streaming-cursor");
    bubble.innerHTML = `<p>${escapeHtml(message)}</p>`;
    toast("Chat error", "error");
  };

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: requestBody(question, true),
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(`server returned ${res.status}`);

    for await (const evt of sseIterator(res)) {
      if (!gotFirstEvent) {
        gotFirstEvent = true;
        clearTimeout(watchdog);
      }
      if (evt.event === "sources") {
        sources = JSON.parse(evt.data).sources || [];
        renderSources(sourcesEl, sources);
        scrollToBottom();
      } else if (evt.event === "token") {
        answer += JSON.parse(evt.data).content;
        bubble.classList.add("streaming-cursor");
        bubble.innerHTML = renderMarkdown(answer);
        scrollToBottom();
      } else if (evt.event === "done") {
        settled = true;
        finalizeAnswer(bubble, sourcesEl, answer, sources);
      } else if (evt.event === "error") {
        settled = true;
        clearTimeout(watchdog);
        const detail = JSON.parse(evt.data).detail || "unknown error";
        renderError(`Something went wrong while generating the answer: ${detail}`);
      }
    }
    if (!gotFirstEvent) throw new Error("no events received");
  } catch (err) {
    if (settled) {
      // Stream broke after the answer was already delivered — nothing to do.
    } else {
      // Fallback: single JSON request (works where SSE is blocked).
      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: requestBody(question, false),
        });
        if (!res.ok) throw new Error(`server returned ${res.status}`);
        const data = await res.json();
        finalizeAnswer(bubble, sourcesEl, data.answer || "", data.sources || []);
      } catch {
        renderError("Connection to the server was lost. Please try again.");
      }
    }
  } finally {
    clearTimeout(watchdog);
    state.streaming = false;
    updateSendState();
    scrollToBottom();
  }
}

/* ------------------------------------------------- composer */

function updateSendState() {
  sendBtnEl.disabled = state.streaming || !inputEl.value.trim();
}

function autoResize() {
  inputEl.style.height = "auto";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 160)}px`;
}

function bindComposer() {
  inputEl.addEventListener("input", () => {
    autoResize();
    updateSendState();
  });
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      composerEl.requestSubmit();
    }
  });
  composerEl.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = inputEl.value.trim();
    if (!q) return;
    inputEl.value = "";
    autoResize();
    updateSendState();
    sendQuestion(q);
  });
  clearChatEl.addEventListener("click", () => clearChat());
}

function clearChat(silent = false) {
  if (!silent && state.history.length && !confirm("Clear the conversation?")) return;
  state.history = [];
  messagesEl.innerHTML = "";
  messagesEl.appendChild(heroEl);
  showHero();
  updateHero();
  loadSuggestions();
}

/* ------------------------------------------------- health check */

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const h = await res.json();
    if (h.status === "error") {
      healthMessageEl.textContent =
        "Cannot reach Ollama — start it with 'ollama serve' and reload this page.";
      healthBannerEl.hidden = false;
    } else if (!h.llm_ready || !h.embedding_ready) {
      const missing = [];
      if (!h.llm_ready) missing.push(h.llm_model);
      if (!h.embedding_ready) missing.push(h.embedding_model);
      healthMessageEl.textContent =
        `Missing model${missing.length > 1 ? "s" : ""} in Ollama: ${missing.join(", ")} — run 'ollama pull <model>' and reload.`;
      healthBannerEl.hidden = false;
    }
  } catch {
    healthMessageEl.textContent = "Backend is not responding — is the FastAPI server running?";
    healthBannerEl.hidden = false;
  }
}

/* ------------------------------------------------- init */

async function init() {
  bindUpload();
  bindComposer();
  await checkHealth();
  await refreshDocuments();
  updateHero();
  if (state.documents.length) loadSuggestions();
}

init();
