const form = document.getElementById("chat-form");
const input = document.getElementById("input");
const messages = document.getElementById("messages");
const moodEl = document.getElementById("mood");
const hormoneEl = document.getElementById("hormone-status");
const stimulusEl = document.getElementById("stimulus");
const engineEl = document.getElementById("engine-status");
const debugPanel = document.getElementById("debug-panel");
const debugToggle = document.getElementById("debug-toggle");
const debugContent = document.getElementById("debug-content");
const debugTagsEl = document.getElementById("debug-tags");
const debugTraitsEl = document.getElementById("debug-traits");
const debugSamplingEl = document.getElementById("debug-sampling");
const debugPolicyEl = document.getElementById("debug-policy");
const debugIntentEl = document.getElementById("debug-intent");
const debugUpdatedEl = document.getElementById("debug-updated");
const debugMetricsEl = document.getElementById("debug-metrics");
const debugAveragesEl = document.getElementById("debug-averages");
const debugInnerEl = document.getElementById("debug-inner");
let debugExpanded = false;

function formatTime(date) {
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function _shortenText(text, limit) {
  if (!text) return "";
  if (text.length <= limit) return text;
  return `${text.slice(0, Math.max(0, limit - 3))}...`;
}

function addMessage(text, role = "system", meta) {
  const line = document.createElement("div");
  line.className = `line ${role}`;

  const timestamp = document.createElement("span");
  timestamp.className = "timestamp";
  timestamp.textContent = formatTime(new Date());

  const prompt = document.createElement("span");
  prompt.className = "prompt";
  prompt.textContent =
    role === "user" ? "you>" : role === "ai" ? "ai>" : "sys>";

  const body = document.createElement("span");
  body.className = "body";
  body.textContent = text ? text.trim() : "";

  line.append(timestamp, prompt, body);

  if (meta) {
    const metaEl = document.createElement("span");
    metaEl.className = "meta";
    metaEl.textContent = meta;
    line.append(metaEl);
  }

  messages.append(line);
  messages.scrollTo({
    top: messages.scrollHeight,
    behavior: "smooth",
  });
  return line;
}

function updateMeta(line, meta) {
  if (!line) return;
  let metaEl = line.querySelector(".meta");
  if (!meta) {
    if (metaEl) metaEl.remove();
    return;
  }
  if (!metaEl) {
    metaEl = document.createElement("span");
    metaEl.className = "meta";
    line.append(metaEl);
  }
  metaEl.textContent = meta;
}

function renderPersona(persona) {
  if (!hormoneEl) return;
  if (!persona) {
    hormoneEl.textContent = "tone: --";
    return;
  }
  const tone = persona.tone || "steady";
  const signals = Array.isArray(persona.signals)
    ? persona.signals.filter(Boolean).map((s) => s.replace(/\.$/, ""))
    : [];
  const guidance = Array.isArray(persona.guidance)
    ? persona.guidance.filter(Boolean).map((g) => g.replace(/\.$/, ""))
    : [];
  const pieces = [`tone: ${tone}`];
  if (signals.length) {
    pieces.push(signals[0]);
  }
  if (guidance.length) {
    pieces.push(guidance[0]);
  }
  hormoneEl.textContent = pieces.join(" | ");
}

function renderTelemetry(state) {
  if (!state) return;
  if (state.mood && moodEl) {
    moodEl.textContent = state.mood;
  }
  renderPersona(state.persona);
  renderDebug(state);
}

function renderEngineStatus(result) {
  if (!engineEl) return;
  if (!result) {
    engineEl.textContent = "engine: offline";
    return;
  }
  const source = result.source ?? "unknown";
  const parts = [`engine: ${source}`];
  const timings = result.llm?.timings;
  const usage = result.llm?.usage;
  if (usage?.total_tokens) {
    parts.push(`${usage.total_tokens} tok`);
  }
  if (timings?.predicted_ms) {
    parts.push(`${Math.round(timings.predicted_ms)} ms`);
  }
  if (timings?.predicted_per_second) {
    parts.push(`${timings.predicted_per_second.toFixed(1)} tok/s`);
  }
  engineEl.textContent = parts.join(" | ");
}

async function fetchState() {
  try {
    const res = await fetch("/state");
    if (!res.ok) return;
    const state = await res.json();
    renderTelemetry(state);
  } catch (error) {
    console.error("Failed to fetch state", error);
  }
}

function computeMeta({ source, llm }) {
  const parts = [];
  if (source) parts.push(`engine:${source}`);
  const tokens = llm?.usage?.total_tokens;
  if (tokens) parts.push(`${tokens} tok`);
  const latency = llm?.timings?.predicted_ms;
  if (latency) parts.push(`${Math.round(latency)} ms`);
  const tps = llm?.timings?.predicted_per_second;
  if (tps) parts.push(`${tps.toFixed(1)} tok/s`);
  return parts.join(" | ");
}

async function sendMessage(event) {
  event.preventDefault();
  const message = input.value.trim();
  const stimulus = stimulusEl.value || "";
  if (!message) return;

  const meta = stimulus ? `stimulus:${stimulus}` : undefined;
  addMessage(message, "user", meta);

  input.value = "";
  input.focus();

  const payload = { message, stimulus: stimulus || null };
  const aiLine = addMessage("", "ai");
  const aiBody = aiLine.querySelector(".body");
  if (aiBody) {
    aiBody.textContent = "...";
  }

  let finalData = null;
  try {
    renderEngineStatus({ source: "stream" });
    finalData = await streamChat(payload, aiLine);
  } catch (error) {
    console.warn("streaming fallback", error);
  }

  if (!finalData) {
    try {
      finalData = await postChat(payload, aiLine);
    } catch (error) {
      console.error("fallback request failed", error);
      if (aiBody) {
        aiBody.textContent = "(response unavailable)";
      }
      updateMeta(aiLine, null);
      renderEngineStatus(null);
      addMessage("connection error. check console for details.", "system");
      return;
    }
  }

  if (!finalData && aiBody) {
    aiBody.textContent = "(response unavailable)";
    renderEngineStatus(null);
  }
}

async function streamChat(payload, aiLine) {
  const response = await fetch("/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok || !response.body) {
    throw new Error(`stream init failed: ${response.status}`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let accumulated = "";
  let finalData = null;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      while (true) {
        const boundary = buffer.indexOf("\n\n");
        if (boundary === -1) break;
        const rawEvent = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        if (!rawEvent.trim()) continue;
        const lines = rawEvent.split("\n");
        let eventType = "message";
        let dataPayload = "";
        for (const line of lines) {
          if (line.startsWith("event:")) {
            eventType = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            dataPayload += line.slice(5).trim();
          }
        }
        if (!dataPayload) continue;
        let parsed;
        try {
          parsed = JSON.parse(dataPayload);
        } catch (err) {
          console.warn("invalid SSE payload", dataPayload);
          continue;
        }
        if (eventType === "token") {
          const token = parsed.text || "";
          if (token) {
            accumulated += token;
            if (aiLine) {
              const body = aiLine.querySelector(".body");
              if (body) {
                body.textContent = accumulated.replace(/^\s+/, "");
              }
            }
          }
        } else if (eventType === "complete") {
          finalData = parsed;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  if (finalData) {
    const text = finalData.reply || accumulated || "(no response)";
    if (aiLine) {
      const body = aiLine.querySelector(".body");
      if (body) {
        body.textContent = text.trim();
      }
      updateMeta(aiLine, computeMeta(finalData) || null);
    } else {
      addMessage(text, "ai", computeMeta(finalData) || undefined);
    }
    if (finalData.state) {
      renderTelemetry(finalData.state);
    }
    if (finalData.sampling_snapshot) {
      renderDebugSnapshot(finalData.sampling_snapshot, finalData.state?.affect, finalData.reinforcement_metrics, finalData.metric_averages, finalData.inner_reflections);
    }
    renderEngineStatus(finalData);
  }
  return finalData;
}

async function postChat(payload, aiLine) {
  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail);
  }
  const data = await res.json();
  const text = data.reply || "(no response)";
  if (aiLine) {
    const body = aiLine.querySelector(".body");
    if (body) {
      body.textContent = text.trim();
    }
    updateMeta(aiLine, computeMeta(data) || null);
  } else {
    addMessage(text, "ai", computeMeta(data) || undefined);
  }
  if (data.state) {
    renderTelemetry(data.state);
  }
  if (data.sampling_snapshot) {
    renderDebugSnapshot(data.sampling_snapshot, data.state?.affect, data.reinforcement_metrics, data.metric_averages, data.inner_reflections);
  }
  renderEngineStatus(data);
  return data;
}

form.addEventListener("submit", sendMessage);

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (typeof form.requestSubmit === "function") {
      form.requestSubmit();
    } else {
      form.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
    }
  }
});

addMessage("session online. say hello to begin.", "system");
fetchState();
renderEngineStatus(null);
setInterval(fetchState, 8000);

function renderDebug(state) {
  if (!debugPanel) return;
  const debugPayload = state?.debug;
  if (!debugPayload) {
    debugPanel.hidden = true;
    return;
  }
  debugPanel.hidden = false;
  const affect = debugPayload.affect;
  const snapshot = debugPayload.sampling_snapshot;
  const reinforcement = debugPayload.reinforcement_metrics;
  const averages = debugPayload.metric_averages;
  const innerReflections = debugPayload.inner_reflections;
  renderDebugSnapshot(snapshot, affect, reinforcement, averages, innerReflections);
}

function renderDebugSnapshot(snapshot, affect, reinforcement, averages, innerReflections) {
  if (!debugPanel || debugPanel.hidden) return;
  const tags = affect?.tags || snapshot?.policy_preview?.tags || [];
  const traits = affect?.traits || {};
  if (debugTagsEl) {
    debugTagsEl.textContent = Array.isArray(tags) && tags.length ? tags.join(", ") : "--";
  }
  if (debugTraitsEl) {
    const entries = Object.entries(traits)
      .map(([key, value]) => `${key}: ${typeof value === "number" ? value.toFixed(3) : value}`)
      .join(" | ");
    debugTraitsEl.textContent = entries || "--";
  }
  if (debugSamplingEl) {
    const sampling = snapshot?.sampling || {};
    const entries = Object.entries(sampling)
      .map(([key, value]) => `${key}=${value}`)
      .join(" | ");
    debugSamplingEl.textContent = entries || "--";
  }
  if (debugPolicyEl) {
    const policy = snapshot?.policy_preview || {};
    const entries = Object.entries(policy)
      .map(([key, value]) => `${key}=${value}`)
      .join(" | ");
    debugPolicyEl.textContent = entries || "--";
  }
  if (debugIntentEl) {
    const items = [];
    if (snapshot?.intent) items.push(snapshot.intent);
    if (snapshot?.length_label) items.push(`len=${snapshot.length_label}`);
    debugIntentEl.textContent = items.join(" | ") || "--";
  }
  if (debugMetricsEl) {
    const metrics = reinforcement?.metrics || reinforcement || {};
    const entries = Object.entries(metrics)
      .map(([key, value]) => `${key}=${typeof value === "number" ? value.toFixed(3) : value}`)
      .join(" | ");
    debugMetricsEl.textContent = entries || "--";
  }
  if (debugAveragesEl) {
    const averaged = Object.entries(averages || {})
      .map(([key, value]) => `${key}=${value}`)
      .join(" | ");
    debugAveragesEl.textContent = averaged || "--";
  }
  if (debugInnerEl) {
    debugInnerEl.textContent = Array.isArray(innerReflections) && innerReflections.length
      ? innerReflections.map((line) => _shortenText(line, 80)).join(" | ")
      : "--";
  }
  if (debugUpdatedEl) {
    debugUpdatedEl.textContent = snapshot?.timestamp || "--";
  }
  if (debugContent) {
    debugContent.hidden = !debugExpanded;
  }
  if (debugToggle) {
    debugToggle.setAttribute("aria-expanded", String(debugExpanded));
    debugToggle.textContent = debugExpanded ? "collapse" : "expand";
  }
}

if (debugToggle) {
  debugToggle.addEventListener("click", () => {
    debugExpanded = !debugExpanded;
    if (debugContent) {
      debugContent.hidden = !debugExpanded;
    }
    debugToggle.setAttribute("aria-expanded", String(debugExpanded));
    debugToggle.textContent = debugExpanded ? "collapse" : "expand";
  });
}
