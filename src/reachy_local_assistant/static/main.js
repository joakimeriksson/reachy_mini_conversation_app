const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function fetchWithTimeout(url, options = {}, timeoutMs = 2000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(id);
  }
}

async function waitForPersonalityData(timeoutMs = 15000) {
  const loadingText = document.querySelector("#loading p");
  let attempts = 0;
  const deadline = Date.now() + timeoutMs;
  while (true) {
    attempts += 1;
    try {
      const url = new URL("/personalities", window.location.origin);
      url.searchParams.set("_", Date.now().toString());
      const resp = await fetchWithTimeout(url, {}, 2000);
      if (resp.ok) return await resp.json();
    } catch (e) {}

    if (loadingText) {
      loadingText.textContent = attempts > 8 ? "Starting backend…" : "Loading…";
    }
    if (Date.now() >= deadline) return null;
    await sleep(500);
  }
}

// ---------- MCP API ----------
async function getMcpStatus() {
  try {
    const url = new URL("/mcp/status", window.location.origin);
    url.searchParams.set("_", Date.now().toString());
    const resp = await fetchWithTimeout(url, {}, 3000);
    if (!resp.ok) return { servers: "", connected: false };
    return await resp.json();
  } catch (e) {
    return { servers: "", connected: false };
  }
}

async function connectMcp(serversText) {
  const resp = await fetchWithTimeout(
    new URL("/mcp/connect", window.location.origin),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ servers: serversText }),
    },
    15000,
  );
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.error || "connect_failed");
  }
  return await resp.json();
}

// ---------- Conversation transcript API ----------

async function getHistory(since) {
  try {
    const url = new URL("/history", window.location.origin);
    url.searchParams.set("since", String(since || 0));
    const resp = await fetchWithTimeout(url, {}, 3000);
    if (!resp.ok) return null;
    return await resp.json();
  } catch (e) {
    return null;
  }
}

async function clearHistory() {
  const resp = await fetchWithTimeout(
    new URL("/history/clear", window.location.origin),
    { method: "POST" },
    5000,
  );
  if (!resp.ok) throw new Error("clear_failed");
  return await resp.json();
}

// ---------- Local backend (Ollama + TTS) API ----------

async function getBackendsStatus() {
  try {
    const url = new URL("/backends/status", window.location.origin);
    url.searchParams.set("_", Date.now().toString());
    const resp = await fetchWithTimeout(url, {}, 3000);
    if (!resp.ok) return null;
    return await resp.json();
  } catch (e) {
    return null;
  }
}

async function checkBackends() {
  const resp = await fetchWithTimeout(
    new URL("/backends/check", window.location.origin),
    { method: "POST" },
    25000,
  );
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.error || "check_failed");
  return data;
}

async function saveBackends(payload) {
  const resp = await fetchWithTimeout(
    new URL("/backends/save", window.location.origin),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
    25000,
  );
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.status || "save_failed");
  return data;
}

// ---------- Personalities API ----------
async function loadPersonality(name) {
  const url = new URL("/personalities/load", window.location.origin);
  url.searchParams.set("name", name);
  url.searchParams.set("_", Date.now().toString());
  const resp = await fetchWithTimeout(url, {}, 3000);
  if (!resp.ok) throw new Error("load_failed");
  return await resp.json();
}

async function savePersonality(payload) {
  const saveUrl = new URL("/personalities/save", window.location.origin);
  saveUrl.searchParams.set("_", Date.now().toString());
  let resp = await fetchWithTimeout(saveUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }, 5000);
  if (resp.ok) return await resp.json();

  // Fallback to form-encoded POST
  try {
    const form = new URLSearchParams();
    form.set("name", payload.name || "");
    form.set("instructions", payload.instructions || "");
    form.set("tools_text", payload.tools_text || "");
    form.set("voice", payload.voice || "");
    const url = new URL("/personalities/save_raw", window.location.origin);
    url.searchParams.set("_", Date.now().toString());
    resp = await fetchWithTimeout(url, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: form.toString(),
    }, 5000);
    if (resp.ok) return await resp.json();
  } catch {}

  const data = await resp.json().catch(() => ({}));
  throw new Error(data.error || "save_failed");
}

async function applyPersonality(name, { persist = false } = {}) {
  const url = new URL("/personalities/apply", window.location.origin);
  url.searchParams.set("name", name || "");
  if (persist) {
    url.searchParams.set("persist", "1");
  }
  url.searchParams.set("_", Date.now().toString());
  const resp = await fetchWithTimeout(url, { method: "POST" }, 5000);
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.error || "apply_failed");
  }
  return await resp.json();
}

// The configured Piper voice (used as the default for new profiles).
async function getDefaultVoice() {
  try {
    const url = new URL("/voices", window.location.origin);
    url.searchParams.set("_", Date.now().toString());
    const resp = await fetchWithTimeout(url, {}, 3000);
    if (!resp.ok) throw new Error("voices_failed");
    const voices = await resp.json();
    return Array.isArray(voices) && voices.length ? voices[0] : "";
  } catch (e) {
    return "";
  }
}

function show(el, flag) {
  el.classList.toggle("hidden", !flag);
}

async function init() {
  const loading = document.getElementById("loading");
  show(loading, true);
  const statusEl = document.getElementById("status");
  const personalityPanel = document.getElementById("personality-panel");

  // Personality elements
  const pSelect = document.getElementById("personality-select");
  const pApply = document.getElementById("apply-personality");
  const pPersist = document.getElementById("persist-personality");
  const pNew = document.getElementById("new-personality");
  const pSave = document.getElementById("save-personality");
  const pStartupLabel = document.getElementById("startup-label");
  const pName = document.getElementById("personality-name");
  const pInstr = document.getElementById("instructions-ta");
  const pTools = document.getElementById("tools-ta");
  const pStatus = document.getElementById("personality-status");
  const pVoice = document.getElementById("voice-select");
  const pAvail = document.getElementById("tools-available");

  const AUTO_WITH = {
    dance: ["stop_dance"],
    play_emotion: ["stop_emotion"],
  };

  show(personalityPanel, false);
  const defaultVoice = await getDefaultVoice();

  // ---------- MCP panel ----------
  const mcpPanel = document.getElementById("mcp-panel");
  const mcpServers = document.getElementById("mcp-servers-ta");
  const mcpConnectBtn = document.getElementById("mcp-connect-btn");
  const mcpStatus = document.getElementById("mcp-status");
  const mcpChip = document.getElementById("mcp-chip");

  show(mcpPanel, true);
  try {
    const mcpState = await getMcpStatus();
    if (mcpState.servers) {
      mcpServers.value = mcpState.servers.split(",").map((s) => s.trim()).filter(Boolean).join("\n");
    }
    if (mcpState.tool_count > 0) {
      mcpStatus.textContent = `${mcpState.tool_count} tool(s) connected.`;
      mcpStatus.className = "status ok";
      mcpChip.textContent = "Connected";
      mcpChip.className = "chip chip-ok";
    }
  } catch (e) {}

  mcpConnectBtn.addEventListener("click", async () => {
    mcpStatus.textContent = "Connecting...";
    mcpStatus.className = "status";
    mcpChip.textContent = "Connecting";
    mcpChip.className = "chip";
    try {
      const res = await connectMcp(mcpServers.value);
      mcpStatus.textContent = res.status || "Connected.";
      mcpStatus.className = "status ok";
      mcpChip.textContent = res.tool_count > 0 ? "Connected" : "Optional";
      mcpChip.className = res.tool_count > 0 ? "chip chip-ok" : "chip";
    } catch (e) {
      mcpStatus.textContent = `Failed: ${e.message}`;
      mcpStatus.className = "status error";
      mcpChip.textContent = "Error";
      mcpChip.className = "chip";
    }
  });

  // ---------- Conversation transcript ----------
  const transcriptPanel = document.getElementById("transcript-panel");
  const transcriptLog = document.getElementById("transcript-log");
  const transcriptChip = document.getElementById("transcript-chip");
  const transcriptClearBtn = document.getElementById("transcript-clear-btn");
  const transcriptFollow = document.getElementById("transcript-follow");

  // Track the highest seq rendered so each poll asks only for what is new; the
  // server keeps seq monotonic across clears so this never has to reset.
  let transcriptSeq = 0;
  let transcriptGeneration = 0;
  const rendered = new Map();

  function resetTranscriptView(message) {
    transcriptSeq = 0;
    rendered.clear();
    transcriptLog.innerHTML = message ? `<p class="muted small">${message}</p>` : "";
    transcriptChip.textContent = "Idle";
  }

  function renderMessage(msg) {
    // A pending user turn is upgraded in place once Whisper transcribes it, so
    // re-render the existing node rather than appending a duplicate.
    let el = rendered.get(msg.seq);
    if (!el) {
      el = document.createElement("div");
      el.className = `turn turn-${msg.role === "assistant" ? "assistant" : "user"}`;
      const who = document.createElement("span");
      who.className = "turn-role";
      who.textContent = msg.role === "assistant" ? "Reachy" : "You";
      const body = document.createElement("p");
      body.className = "turn-text";
      el.appendChild(who);
      el.appendChild(body);
      transcriptLog.appendChild(el);
      rendered.set(msg.seq, el);
    }
    el.querySelector(".turn-text").textContent = msg.content;
  }

  async function pollTranscript() {
    const data = await getHistory(transcriptSeq);
    if (!data) return;

    // seq stays monotonic across clears, so a bumped generation is the only
    // signal that the conversation was reset (here, another tab, or a
    // personality switch). Drop everything and refetch from the start.
    if (transcriptGeneration && data.generation !== transcriptGeneration) {
      transcriptGeneration = data.generation;
      resetTranscriptView("Conversation reset.");
      return;
    }
    transcriptGeneration = data.generation;

    if (Array.isArray(data.messages) && data.messages.length) {
      const placeholder = transcriptLog.querySelector(".muted");
      if (placeholder) placeholder.remove();
      for (const msg of data.messages) renderMessage(msg);
      transcriptSeq = data.latest_seq;
      if (transcriptFollow.checked) transcriptLog.scrollTop = transcriptLog.scrollHeight;
    }
    // Count what is on screen, not latest_seq: that keeps counting across a
    // clear (it must stay monotonic for polling) and would report stale turns.
    const shown = rendered.size;
    transcriptChip.textContent = shown ? `${shown} turn${shown === 1 ? "" : "s"}` : "Idle";
  }

  show(transcriptPanel, true);
  await pollTranscript();
  setInterval(pollTranscript, 1500);

  transcriptClearBtn.addEventListener("click", async () => {
    try {
      await clearHistory();
      // Adopt the new generation here so our own clear doesn't trip the
      // "reset elsewhere" branch on the next poll.
      const data = await getHistory(0);
      if (data) transcriptGeneration = data.generation;
      resetTranscriptView("Cleared.");
    } catch (e) {}
  });

  // ---------- Local backend panel (Ollama + TTS URLs) ----------
  const backendsPanel = document.getElementById("backends-panel");
  const ollamaUrl = document.getElementById("ollama-url");
  const ttsUrl = document.getElementById("tts-url");
  const ttsVoice = document.getElementById("tts-voice");
  const backendsSaveBtn = document.getElementById("backends-save-btn");
  const backendsCheckBtn = document.getElementById("backends-check-btn");
  const backendsStatus = document.getElementById("backends-status");
  const backendsHealth = document.getElementById("backends-health");

  // A down backend is otherwise invisible: the robot listens, thinks, and says
  // nothing. Render each probe so the missing service names itself.
  function renderHealth(health) {
    backendsHealth.innerHTML = "";
    if (!health || !Array.isArray(health.probes)) return;
    for (const probe of health.probes) {
      const li = document.createElement("li");
      li.className = probe.ok ? "health-item ok" : "health-item error";

      const name = document.createElement("span");
      name.className = "health-name";
      name.textContent = `${probe.ok ? "✓" : "✗"} ${probe.name}`;
      li.appendChild(name);

      const detail = document.createElement("span");
      detail.className = "health-detail";
      detail.textContent = probe.detail || "";
      li.appendChild(detail);

      if (!probe.ok && probe.hint) {
        const hint = document.createElement("span");
        hint.className = "health-hint";
        hint.textContent = `→ ${probe.hint}`;
        li.appendChild(hint);
      }
      backendsHealth.appendChild(li);
    }
  }

  show(backendsPanel, true);
  try {
    const st = await getBackendsStatus();
    if (st) {
      ollamaUrl.value = st.ollama_url || "";
      ttsUrl.value = st.tts_url || "";
      ttsVoice.value = st.tts_voice || "";
      renderHealth(st.health);
    }
  } catch (e) {}

  backendsSaveBtn.addEventListener("click", async () => {
    backendsStatus.textContent = "Saving...";
    backendsStatus.className = "status";
    try {
      const res = await saveBackends({
        ollama_url: ollamaUrl.value.trim(),
        tts_url: ttsUrl.value.trim(),
        tts_voice: ttsVoice.value.trim(),
      });
      backendsStatus.textContent = res.status || "Saved.";
      backendsStatus.className = "status ok";
      const st = await getBackendsStatus();
      if (st) renderHealth(st.health);
    } catch (e) {
      backendsStatus.textContent = `Failed: ${e.message}`;
      backendsStatus.className = "status error";
    }
  });

  backendsCheckBtn.addEventListener("click", async () => {
    backendsStatus.textContent = "Checking...";
    backendsStatus.className = "status";
    try {
      const health = await checkBackends();
      renderHealth(health);
      backendsStatus.textContent = health.ok
        ? "All backends reachable."
        : "Some backends are unreachable — see below.";
      backendsStatus.className = health.ok ? "status ok" : "status error";
    } catch (e) {
      backendsStatus.textContent = `Check failed: ${e.message}`;
      backendsStatus.className = "status error";
    }
  });

  // ---------- Personalities ----------
  const list = (await waitForPersonalityData()) || { choices: [] };
  if (!list.choices.length) {
    show(statusEl, true);
    statusEl.textContent = "Personality endpoints not ready yet. Retry shortly.";
    statusEl.className = "status warn";
    show(loading, false);
    return;
  }

  try {
    const choices = Array.isArray(list.choices) ? list.choices : [];
    const DEFAULT_OPTION = choices[0] || "(built-in default)";
    const startupChoice = choices.includes(list.startup) ? list.startup : DEFAULT_OPTION;
    const currentChoice = choices.includes(list.current) ? list.current : startupChoice;

    function setStartupLabel(name) {
      const display = name && name !== DEFAULT_OPTION ? name : "Built-in default";
      pStartupLabel.textContent = `Launch on start: ${display}`;
    }

    pSelect.innerHTML = "";
    for (const n of choices) {
      const opt = document.createElement("option");
      opt.value = n;
      opt.textContent = n;
      pSelect.appendChild(opt);
    }
    if (choices.length) {
      const preferred = choices.includes(startupChoice) ? startupChoice : currentChoice;
      pSelect.value = preferred;
    }
    setStartupLabel(startupChoice);

    function renderToolCheckboxes(available, enabled) {
      pAvail.innerHTML = "";
      const enabledSet = new Set(enabled);
      for (const t of available) {
        const wrap = document.createElement("div");
        wrap.className = "chk";
        const id = `tool-${t}`;
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.id = id;
        cb.value = t;
        cb.checked = enabledSet.has(t);
        const lab = document.createElement("label");
        lab.htmlFor = id;
        lab.textContent = t;
        wrap.appendChild(cb);
        wrap.appendChild(lab);
        pAvail.appendChild(wrap);
      }
    }

    function getSelectedTools() {
      const selected = new Set();
      pAvail.querySelectorAll('input[type="checkbox"]').forEach((el) => {
        if (el.checked) selected.add(el.value);
      });
      for (const [main, deps] of Object.entries(AUTO_WITH)) {
        if (selected.has(main)) {
          for (const d of deps) selected.add(d);
        }
      }
      return Array.from(selected);
    }

    function syncToolsTextarea() {
      const selected = getSelectedTools();
      const comments = pTools.value
        .split("\n")
        .filter((ln) => ln.trim().startsWith("#"));
      const body = selected.join("\n");
      pTools.value = (comments.join("\n") + (comments.length ? "\n" : "") + body).trim() + "\n";
    }

    function attachToolHandlers() {
      pAvail.addEventListener("change", (ev) => {
        const target = ev.target;
        if (!(target instanceof HTMLInputElement) || target.type !== "checkbox") return;
        const name = target.value;
        if (AUTO_WITH[name]) {
          for (const dep of AUTO_WITH[name]) {
            const depEl = pAvail.querySelector(`input[value="${dep}"]`);
            if (depEl) depEl.checked = target.checked || depEl.checked;
          }
        }
        syncToolsTextarea();
      });
    }

    async function loadSelected() {
      const selected = pSelect.value;
      const data = await loadPersonality(selected);
      pInstr.value = data.instructions || "";
      pTools.value = data.tools_text || "";
      pVoice.value = data.voice || defaultVoice;
      renderToolCheckboxes(data.available_tools, data.enabled_tools);
      attachToolHandlers();
      const idx = selected.lastIndexOf("/");
      pName.value = idx >= 0 ? selected.slice(idx + 1) : "";
      pStatus.textContent = `Loaded ${selected}`;
      pStatus.className = "status";
    }

    pSelect.addEventListener("change", loadSelected);
    await loadSelected();
    show(personalityPanel, true);

    pApply.addEventListener("click", async () => {
      pStatus.textContent = "Applying...";
      pStatus.className = "status";
      try {
        const res = await applyPersonality(pSelect.value);
        if (res.startup) setStartupLabel(res.startup);
        pStatus.textContent = res.status || "Applied.";
        pStatus.className = "status ok";
      } catch (e) {
        pStatus.textContent = `Failed to apply${e.message ? ": " + e.message : ""}`;
        pStatus.className = "status error";
      }
    });

    pPersist.addEventListener("click", async () => {
      pStatus.textContent = "Saving for startup...";
      pStatus.className = "status";
      try {
        const res = await applyPersonality(pSelect.value, { persist: true });
        if (res.startup) setStartupLabel(res.startup);
        pStatus.textContent = res.status || "Saved for startup.";
        pStatus.className = "status ok";
      } catch (e) {
        pStatus.textContent = `Failed to persist${e.message ? ": " + e.message : ""}`;
        pStatus.className = "status error";
      }
    });

    pNew.addEventListener("click", () => {
      pName.value = "";
      pInstr.value = "# Write your instructions here\n# e.g., Keep responses concise and friendly.";
      pTools.value = "# tools enabled for this profile\n";
      pAvail.querySelectorAll('input[type="checkbox"]').forEach((el) => {
        el.checked = false;
      });
      pVoice.value = defaultVoice;
      pStatus.textContent = "Fill fields and click Save.";
      pStatus.className = "status";
    });

    pSave.addEventListener("click", async () => {
      const name = (pName.value || "").trim();
      if (!name) {
        pStatus.textContent = "Enter a valid name.";
        pStatus.className = "status warn";
        return;
      }
      pStatus.textContent = "Saving...";
      pStatus.className = "status";
      try {
        syncToolsTextarea();
        const res = await savePersonality({
          name,
          instructions: pInstr.value || "",
          tools_text: pTools.value || "",
          voice: pVoice.value || defaultVoice,
        });
        pSelect.innerHTML = "";
        for (const n of res.choices) {
          const opt = document.createElement("option");
          opt.value = n;
          opt.textContent = n;
          if (n === res.value) opt.selected = true;
          pSelect.appendChild(opt);
        }
        pStatus.textContent = "Saved.";
        pStatus.className = "status ok";
        try { await applyPersonality(pSelect.value); } catch {}
      } catch (e) {
        pStatus.textContent = "Failed to save.";
        pStatus.className = "status error";
      }
    });
  } catch (e) {
    show(statusEl, true);
    statusEl.textContent = "UI failed to load. Please refresh.";
    statusEl.className = "status warn";
  } finally {
    show(loading, false);
  }
}

window.addEventListener("DOMContentLoaded", init);
