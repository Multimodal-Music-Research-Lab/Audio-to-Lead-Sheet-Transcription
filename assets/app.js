/* Demo page: render Kern lead sheets with Verovio and toggle error rates / Kern source. */

import { VerovioToolkit } from "./verovio/verovio.mjs";
import createVerovioModule from "./verovio/verovio-module.mjs";

"use strict";

const MODEL_GROUPS = [
  {
    label: "End-to-End",
    models: [
      { id: "character", label: "Character Tokenization", prediction: true },
      { id: "syllable", label: "Word Tokenization", prediction: true },
      { id: "bpe", label: "BPE Tokenization", prediction: true },
    ],
  },
  {
    label: "Modular",
    models: [
      { id: "eoin-soulx-lyrics", label: "Cummins + SoulXSinger", prediction: true },
      { id: "no-lyrics-qwen-lyrics", label: "Cummins retrained + Qwen3-ASR", prediction: true },
      { id: "eoin-with-lyrics", label: "Cummins + Qwen3-ASR", prediction: true },
      { id: "gt-with-qwen-lyrics", label: "GT + Qwen3-ASR lyrics", prediction: false },
      { id: "gt-with-soulx-lyrics", label: "GT + SoulXSinger lyrics", prediction: false },
    ],
  },
];

const MODELS = MODEL_GROUPS.flatMap((group) => group.models);

const REFERENCE_MODEL = { id: "gt", prediction: false };

const METRIC_COLUMNS = [
  ["sym-er", "SymER"],
  ["hard_sym-er", "Hard SymER"],
  ["char-er", "CharER"],
  ["hard_char-er", "Hard CharER"],
  ["melody_sym_er", "Melody SymER"],
  ["melody_hard_sym_er", "Melody Hard SymER"],
  ["chords_sym_er", "Chords SymER"],
  ["chords_hard_sym_er", "Chords Hard SymER"],
  ["lyrics_sym_er", "Lyrics SymER"],
  ["lyrics_hard_sym_er", "Lyrics Hard SymER"],
  ["melody_char_er", "Melody CharER"],
  ["melody_hard_char_er", "Melody Hard CharER"],
  ["chords_char_er", "Chords CharER"],
  ["chords_hard_char_er", "Chords Hard CharER"],
  ["lyrics_char_er", "Lyrics CharER"],
  ["lyrics_hard_char_er", "Lyrics Hard CharER"],
];

const SAMPLES_URL = "data/samples.json";
// Verovio renders the pre-converted MEI files (see code/convert_kern_to_mei.py);
// the Kern files in krn/ remain the source of truth and are shown as text.
const VEROVIO_OPTIONS = {
  adjustPageHeight: true,
  pageHeight: 2500,
  pageWidth: 2200,
  scale: 40,
  footer: "none",
};

const state = {
  samples: [],
  sampleId: null,
  modelId: MODELS[0].id,
  showErrors: false,
  showKern: false,
  toolkit: null,
  metricsCache: new Map(),
};

const referenceScore = document.getElementById("reference-score");
const modelScore = document.getElementById("model-score");

function fileNameFor(sampleId, model) {
  return model.prediction ? `_${sampleId}__prediction.krn` : `${sampleId}.krn`;
}

function meiUrl(modelId, sampleId) {
  const model = modelId === REFERENCE_MODEL.id ? REFERENCE_MODEL : MODELS.find((entry) => entry.id === modelId);
  return `mei/${model.id}/${fileNameFor(sampleId, model).replace(/\.krn$/, ".mei")}`;
}

function kernUrl(modelId, sampleId) {
  const model = modelId === REFERENCE_MODEL.id ? REFERENCE_MODEL : MODELS.find((entry) => entry.id === modelId);
  return `krn/${model.id}/${fileNameFor(sampleId, model)}`;
}

function setStatus(element, message) {
  element.innerHTML = `<p class="status">${message}</p>`;
}

function shuffle(list) {
  for (let index = list.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(Math.random() * (index + 1));
    [list[index], list[swap]] = [list[swap], list[index]];
  }
  return list;
}

function renderSampleList() {
  const list = document.getElementById("sample-list");
  list.innerHTML = "";
  for (const sample of state.samples) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = sample.id;
    button.classList.toggle("active", sample.id === state.sampleId);
    button.addEventListener("click", () => selectSample(sample.id));
    item.appendChild(button);
    list.appendChild(item);
  }
}

function selectSample(sampleId) {
  state.sampleId = sampleId;
  renderSampleList();
  renderAll();
  const url = new URL(window.location);
  url.searchParams.set("sample", sampleId);
  window.history.replaceState(null, "", url);
}

function pickRandomSample() {
  if (state.samples.length < 2) return;
  const others = state.samples.map((sample) => sample.id).filter((id) => id !== state.sampleId);
  selectSample(others[Math.floor(Math.random() * others.length)]);
}

async function renderScore(element, modelId) {
  if (!state.toolkit) return;
  setStatus(element, "Loading score…");
  try {
    const response = await fetch(meiUrl(modelId, state.sampleId));
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.text();
    if (!state.toolkit.loadData(data)) throw new Error("Verovio rejected this score");
    const svg = state.toolkit.renderToSVG(1);
    if (!svg) throw new Error("Verovio produced no SVG");
    element.innerHTML = svg;
  } catch (error) {
    setStatus(element, `Could not render this score (${error.message}).`);
  }
}

async function renderKernSource(element, modelId) {
  if (!state.showKern) {
    element.hidden = true;
    return;
  }
  try {
    const response = await fetch(kernUrl(modelId, state.sampleId));
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    element.textContent = await response.text();
    element.hidden = false;
  } catch (error) {
    element.textContent = `Could not load the Kern source (${error.message}).`;
    element.hidden = false;
  }
}

async function loadMetrics(modelId) {
  if (state.metricsCache.has(modelId)) return state.metricsCache.get(modelId);
  const response = await fetch(`metrics/${modelId}.json`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const payload = await response.json();
  state.metricsCache.set(modelId, payload);
  return payload;
}

async function renderMetrics() {
  const panel = document.getElementById("metrics");
  if (!state.showErrors) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  const model = MODELS.find((entry) => entry.id === state.modelId);
  document.getElementById("metrics-title").textContent = `Error rates — ${model.label}`;
  const body = document.querySelector("#metrics-table tbody");
  const note = document.getElementById("metrics-note");
  body.innerHTML = "";
  note.textContent = "";
  try {
    const payload = await loadMetrics(state.modelId);
    const sample = payload.samples[state.sampleId];
    if (!sample) {
      note.textContent = "No metrics available for this sample.";
      return;
    }
    const row = document.createElement("tr");
    for (const [key] of METRIC_COLUMNS) {
      const cell = document.createElement("td");
      cell.textContent = sample[key] != null ? `${sample[key].toFixed(2)} %` : "–";
      row.appendChild(cell);
    }
    body.appendChild(row);
    const average = payload.summary;
    note.textContent =
      "Dataset average (100 samples): " +
      METRIC_COLUMNS.map(([key, label]) => `${label} ${average[key] != null ? average[key].toFixed(2) : "–"}`).join(" · ") +
      " %. Error rates are computed against the ground-truth lead sheets with code/metrics.py.";
  } catch (error) {
    note.textContent = `Could not load metrics (${error.message}).`;
  }
}

function renderAll() {
  document.getElementById("model-heading").textContent =
    MODELS.find((entry) => entry.id === state.modelId).label;
  renderScore(referenceScore, REFERENCE_MODEL.id);
  renderScore(modelScore, state.modelId);
  renderKernSource(document.getElementById("reference-kern"), REFERENCE_MODEL.id);
  renderKernSource(document.getElementById("model-kern"), state.modelId);
  renderMetrics();
}

function populateModelSelect() {
  const select = document.getElementById("model-select");
  for (const group of MODEL_GROUPS) {
    const optgroup = document.createElement("optgroup");
    optgroup.label = group.label;
    for (const model of group.models) {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = model.label;
      optgroup.appendChild(option);
    }
    select.appendChild(optgroup);
  }
  select.value = state.modelId;
  select.addEventListener("change", () => {
    state.modelId = select.value;
    renderAll();
  });
}

async function start() {
  populateModelSelect();

  const params = new URLSearchParams(window.location.search);
  const requestedModel = params.get("model");
  if (MODELS.some((model) => model.id === requestedModel)) {
    state.modelId = requestedModel;
  }
  state.showErrors = params.get("errors") === "1";
  state.showKern = params.get("kern") === "1";
  document.getElementById("model-select").value = state.modelId;
  document.getElementById("show-errors").checked = state.showErrors;
  document.getElementById("show-kern").checked = state.showKern;

  document.getElementById("show-errors").addEventListener("change", (event) => {
    state.showErrors = event.target.checked;
    renderMetrics();
  });
  document.getElementById("show-kern").addEventListener("change", (event) => {
    state.showKern = event.target.checked;
    renderKernSource(document.getElementById("reference-kern"), REFERENCE_MODEL.id);
    renderKernSource(document.getElementById("model-kern"), state.modelId);
  });
  document.getElementById("random-sample").addEventListener("click", pickRandomSample);

  const response = await fetch(SAMPLES_URL);
  state.samples = shuffle(await response.json());
  const requested = params.get("sample");
  state.sampleId = state.samples.some((sample) => sample.id === requested)
    ? requested
    : state.samples[0].id;

  renderSampleList();

  try {
    const module = await createVerovioModule();
    state.toolkit = new VerovioToolkit(module);
    state.toolkit.setOptions(VEROVIO_OPTIONS);
  } catch (error) {
    setStatus(referenceScore, `Could not initialise Verovio (${error.message}).`);
    setStatus(modelScore, `Could not initialise Verovio (${error.message}).`);
    return;
  }
  renderAll();
}

window.addEventListener("load", () => {
  start().catch((error) => {
    setStatus(referenceScore, `Page failed to start (${error.message}).`);
  });
});
