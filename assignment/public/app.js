const elMessages = document.getElementById("messages");
const elInput = document.getElementById("input");
const elSend = document.getElementById("send");
const elClear = document.getElementById("clear");
const elStatus = document.getElementById("status");
const elMode = document.getElementById("mode");

const elResumeFile = document.getElementById("resumeFile");
const elResumeStatus = document.getElementById("resumeStatus");
const elResumePreview = document.getElementById("resumePreview");
const elResumeClear = document.getElementById("resumeClear");

function loadState() {
  try {
    const raw = localStorage.getItem("ps_messages");
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveState(messages) {
  localStorage.setItem("ps_messages", JSON.stringify(messages));
}

function loadResume() {
  try { return localStorage.getItem("ps_resume_text") || ""; }
  catch { return ""; }
}

function saveResume(text) {
  localStorage.setItem("ps_resume_text", text);
}

function clearResume() {
  localStorage.removeItem("ps_resume_text");
}

let messages = loadState();
let resumeText = loadResume();

function bubble(role, html) {
  const wrap = document.createElement("div");
  wrap.className = role === "user" ? "flex justify-end" : "flex justify-start";

  const b = document.createElement("div");
  b.className =
    (role === "user"
      ? "bg-white text-zinc-900"
      : "bg-zinc-950/40 text-zinc-100 border border-zinc-800") +
    " rounded-2xl px-4 py-3 max-w-[85%] text-sm leading-relaxed";

  b.innerHTML = html;
  wrap.appendChild(b);
  return wrap;
}

function escapeHtml(str) {
  return str.replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function render() {
  elMessages.innerHTML = "";
  for (const m of messages) {
    const html = m.role === "assistant"
      ? marked.parse(m.content)
      : `<div class="whitespace-pre-wrap">${escapeHtml(m.content)}</div>`;
    elMessages.appendChild(bubble(m.role, html));
  }
  elMessages.scrollTop = elMessages.scrollHeight;
}

function renderResumeUI() {
  if (!elResumeStatus || !elResumePreview) return;

  if (resumeText && resumeText.trim().length > 0) {
    elResumeStatus.textContent = `Loaded (${resumeText.length} chars)`;
    elResumePreview.textContent = resumeText.slice(0, 6000);
  } else {
    elResumeStatus.textContent = "No resume loaded";
    elResumePreview.textContent = "";
  }
}

async function uploadResume(file) {
  if (!elResumeStatus) return;

  elResumeStatus.textContent = "Uploading...";
  elSend.disabled = true;

  try {
    const fd = new FormData();
    fd.append("file", file);

    const res = await fetch("/api/upload_resume", {
      method: "POST",
      body: fd,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.detail || `Upload failed (${res.status})`);
    }

    const data = await res.json();
    resumeText = data.text || "";
    saveResume(resumeText);
    renderResumeUI();
  } catch (e) {
    elResumeStatus.textContent = `Error: ${String(e.message || e)}`;
  } finally {
    elSend.disabled = false;
  }
}

async function send() {
  const text = elInput.value.trim();
  if (!text) return;

  elSend.disabled = true;
  elStatus.textContent = "Thinking...";

  const resumePrefix = (resumeText && resumeText.trim().length > 0)
    ? `RESUME_CONTEXT:\n${resumeText}\n\n`
    : "";

  // 1) Add the user message
  messages.push({ role: "user", content: resumePrefix + text });

  // 2) Build payload from messages that END with user
  const payload = {
    mode: elMode.value,
    messages: messages
      .filter(m => m.content && m.content.trim().length > 0)
      .slice(-30),
  };

  // 3) Now add assistant placeholder for UI only (NOT sent)
  messages.push({ role: "assistant", content: "_…_" });
  render();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();

    const parts = [];
    parts.push(data.reply_markdown || "");

    if (Array.isArray(data.action_items) && data.action_items.length) {
      parts.push("\n\n---\n\n### Action items\n");
      for (const a of data.action_items) {
        parts.push(`- **${a.title}** _(ETA ${a.eta_minutes}m, ${a.priority})_: ${a.why}`);
      }
    }

    if (Array.isArray(data.follow_up_questions) && data.follow_up_questions.length) {
      parts.push("\n\n---\n\n### Quick questions\n");
      for (const q of data.follow_up_questions) parts.push(`- ${q}`);
    }

    if (Array.isArray(data.warnings) && data.warnings.length) {
      parts.push("\n\n---\n\n### Notes\n");
      for (const w of data.warnings) parts.push(`- ${w}`);
    }

    messages[messages.length - 1] = { role: "assistant", content: parts.join("\n") };
    saveState(messages);
    elStatus.textContent = "Done.";
  } catch (e) {
    messages[messages.length - 1] = {
      role: "assistant",
      content: `**Error:** ${String(e.message || e)}\n\nTry again (or switch to another mode).`,
    };
    elStatus.textContent = "Error.";
  } finally {
    elSend.disabled = false;
    elInput.value = "";
    render();
  }
}

/* Events */
elSend.addEventListener("click", send);

elInput.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") send();
});

elClear.addEventListener("click", () => {
  messages = [];
  saveState(messages);
  render();
  elStatus.textContent = "";
});

document.querySelectorAll(".example").forEach(btn => {
  btn.addEventListener("click", () => {
    elInput.value = btn.dataset.text || "";
    elInput.focus();
  });
});

elResumeFile?.addEventListener("change", async (e) => {
  const f = e.target.files?.[0];
  if (!f) return;

  if (f.size > 2 * 1024 * 1024) {
    if (elResumeStatus) elResumeStatus.textContent = "File too large (max 2MB).";
    elResumeFile.value = "";
    return;
  }

  await uploadResume(f);
  elResumeFile.value = "";
});

elResumeClear?.addEventListener("click", () => {
  resumeText = "";
  clearResume();
  renderResumeUI();
});

/* Initial render */
render();
renderResumeUI();
