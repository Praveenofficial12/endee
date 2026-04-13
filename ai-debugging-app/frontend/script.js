/**
 * script.js — Advanced AI Studio Logic
 */

// Basic Setup
const BACKEND_PORT = 8000;
const API_BASE = (function() {
    const loc = window.location;
    if (loc.port === String(BACKEND_PORT)) return loc.origin;
    return `http://${loc.hostname || 'localhost'}:${BACKEND_PORT}`;
})();

const ANALYZE_ENDPOINT = `${API_BASE}/analyze`;
const SAMPLES_ENDPOINT = `${API_BASE}/samples`;

// Elements
const inputEl = document.getElementById('error-input');
const charCountEl = document.getElementById('char-count');
const analyzeBtn = document.getElementById('analyze-btn');
const samplesGrid = document.getElementById('samples-grid');

const emptyState = document.getElementById('empty-state');
const loadingState = document.getElementById('loading-state');
const resultsState = document.getElementById('results-state');
const errorState = document.getElementById('error-state');

const dropzone = document.getElementById('dropzone');
const uploadBtn = document.getElementById('upload-btn');
const fileInput = document.getElementById('file-input');
const historyList = document.getElementById('history-list');
const newBtn = document.getElementById('new-analysis-btn');

// Local Storage structure
const STORAGE_KEY = 'endee_dbg_history';
let analysisHistory = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');

// Init
document.addEventListener('DOMContentLoaded', () => {
    loadSamples();
    renderHistory();
});

// UI Logic
function setRightPaneState(stateId) {
    [emptyState, loadingState, resultsState, errorState].forEach(el => el.style.display = 'none');
    document.getElementById(stateId).style.display = 'flex';
    if(stateId === 'results-state') document.getElementById(stateId).style.display = 'flex'; // block/flex diff
}

inputEl.addEventListener('input', () => {
    const len = inputEl.value.length;
    charCountEl.textContent = `${len} / 5000`;
});

// File Upload Logic
uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileUpload);
dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        handleFileUpload();
    }
});

function handleFileUpload() {
    const file = fileInput.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        inputEl.value = e.target.result.substring(0, 5000); // truncate if massive
        inputEl.dispatchEvent(new Event('input'));
    };
    reader.readAsText(file);
}

// History Logic
function saveToHistory(query, resultData) {
    const item = {
        id: Date.now(),
        preview: query.substring(0, 30).replace(/\n/g, ' ') + '...',
        query: query,
        result: resultData
    };
    analysisHistory.unshift(item);
    if(analysisHistory.length > 10) analysisHistory.pop();
    localStorage.setItem(STORAGE_KEY, JSON.stringify(analysisHistory));
    renderHistory();
}

function renderHistory() {
    historyList.innerHTML = '';
    analysisHistory.forEach(item => {
        const li = document.createElement('li');
        li.className = 'history-item';
        li.textContent = item.preview;
        li.title = item.query;
        li.onclick = () => loadHistoryItem(item);
        historyList.appendChild(li);
    });
}

function loadHistoryItem(item) {
    inputEl.value = item.query;
    inputEl.dispatchEvent(new Event('input'));
    displayResults(item.result);
}

newBtn.addEventListener('click', () => {
    inputEl.value = '';
    inputEl.dispatchEvent(new Event('input'));
    setRightPaneState('empty-state');
});

// Analysis Flow
analyzeBtn.addEventListener('click', executeAnalysis);
document.getElementById('retry-btn').addEventListener('click', () => setRightPaneState('empty-state'));

async function executeAnalysis() {
    const text = inputEl.value.trim();
    if (!text || text.length < 3) return;

    analyzeBtn.disabled = true;
    setRightPaneState('loading-state');

    // fake progress
    const pBar = document.getElementById('progress-bar');
    const lStep = document.getElementById('loading-step');
    pBar.style.width = '10%'; lStep.textContent = "Vectorizing context...";
    setTimeout(() => { pBar.style.width = '45%'; lStep.textContent = "Querying Endee Space..."; }, 600);
    setTimeout(() => { pBar.style.width = '80%'; lStep.textContent = "Synthesizing output..."; }, 1200);

    try {
        const res = await fetch(ANALYZE_ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input_text: text })
        });

        if (!res.ok) throw new Error("Server communication failed.");

        const data = await res.json();
        pBar.style.width = '100%';
        await new Promise(r => setTimeout(r, 200));

        saveToHistory(text, data);
        displayResults(data);

    } catch (err) {
        console.error("Analysis Error:", err);
        document.getElementById('error-message').textContent = err.message || "Could not connect to the server.";
        setRightPaneState('error-state');
    } finally {
        analyzeBtn.disabled = false;
        setTimeout(()=>{ pBar.style.width='0%'; }, 500);
    }
}

function displayResults(data) {
    setRightPaneState('results-state');
    document.getElementById('result-explanation').textContent = data.explanation || "No data.";
    document.getElementById('result-fix').textContent = data.fix || "No data.";
    
    const codeBlock = document.getElementById('result-code');
    codeBlock.textContent = data.corrected_code || "// No code generated.";
    codeBlock.removeAttribute("data-highlighted");
    
    if (typeof hljs !== 'undefined') hljs.highlightElement(codeBlock);
}

document.getElementById('copy-btn').addEventListener('click', () => {
    const text = document.getElementById('result-code').textContent;
    navigator.clipboard.writeText(text);
    const i = document.querySelector('#copy-btn i');
    i.className = 'fa-solid fa-check text-success';
    setTimeout(() => i.className = 'fa-regular fa-copy', 2000);
});

// Samples
async function loadSamples() {
    const fallback = [
        { label: 'Parse Error', text: "SyntaxError: Unexpected token < in JSON at position 0\n\nfetch('/api').then(r => r.json())" },
        { label: 'DB Connection', text: "MongoTimeoutError: Server selection timed out after 30000 ms" },
        { label: 'Go Panic', text: "panic: runtime error: index out of range [4] with length 3" }
    ];
    let sources = fallback;
    try {
        const r = await fetch(SAMPLES_ENDPOINT);
        if (r.ok) {
            const d = await r.json();
            if(d.samples) sources = d.samples;
        }
    } catch(e) {}

    samplesGrid.innerHTML = '';
    sources.forEach(s => {
        const btn = document.createElement('button');
        btn.className = 'sample-chip';
        btn.textContent = s.label;
        btn.onclick = () => {
            inputEl.value = s.text;
            inputEl.dispatchEvent(new Event('input'));
        };
        samplesGrid.appendChild(btn);
    });
}
