/* ═══════════════════════════════════════════════════════════════
   Pregunta — Client Logic & RAG Integration
   Handles: Ambient particle field, API communication, voice audio
   recording (WAV re-encoding), two-tier answer rendering, and telemetry.
   ═══════════════════════════════════════════════════════════════ */

const $  = (s) => document.querySelector(s);
const esc = (s) => (s ?? '').replace(/[&<>"]/g, (c) => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;' }[c]));
const ms = (v) => `${(+v).toFixed(1)}ms`;

const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.text()).slice(0, 240) || `HTTP ${r.status}`);
  return r.json();
};

/* ───────────────────────── particle field ─────────────────────────
   Displaced dot matrix field responding to live audio mic input
   and active search queries. Cyan/blue liquid palette.               */
(() => {
  const cv = document.getElementById('field');
  if (!cv || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const ctx = cv.getContext('2d', { alpha: true });

  let w = 0, h = 0, dpr = 1, cols = 0, rows = 0, GAP = 30;

  function size() {
    dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    w = cv.clientWidth; h = cv.clientHeight;
    cv.width = w * dpr; cv.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    GAP = Math.max(28, Math.sqrt((w * h) / 2400));
    cols = Math.ceil(w / GAP) + 1;
    rows = Math.ceil(h / GAP) + 1;
  }
  size();
  addEventListener('resize', size, { passive: true });

  const state = { energy: 0, target: 0, t: 0 };
  window.__field = state;

  let px = -999, py = -999;
  addEventListener('pointermove', (e) => { px = e.clientX; py = e.clientY; }, { passive: true });
  addEventListener('pointerleave', () => { px = py = -999; });

  const BINS = 8;
  const bins = Array.from({ length: BINS }, () => []);
  let last = 0;

  function frame(now) {
    requestAnimationFrame(frame);
    if (now - last < 32) return; // ~30fps
    last = now;

    state.t += 0.02;
    state.energy += (state.target - state.energy) * 0.06;
    state.target *= 0.985;

    ctx.clearRect(0, 0, w, h);
    const E = state.energy;
    const hot = E > 0.1;
    for (let b = 0; b < BINS; b++) bins[b].length = 0;

    for (let i = 0; i < cols; i++) {
      const wi = i * 0.22, si = Math.sin(wi + state.t * 2.1);
      const x = i * GAP;
      for (let j = 0; j < rows; j++) {
        const y = j * GAP;
        const wave = si * Math.cos(j * 0.19 - state.t * 1.5)
                   + Math.sin((i + j) * 0.11 + state.t * 1.2);

        const fall = y / h * 1.5 + 0.18;
        const dx = px - x, dy = py - y;
        const d2 = dx * dx + dy * dy;
        const near = d2 < 36100 ? 1 - Math.sqrt(d2) / 190 : 0;

        const a = (0.04 + wave * 0.04 + E * 0.18) * (fall > 1 ? 1 : fall) + near * 0.3;
        if (a <= 0.015) continue;

        const s = Math.min(2.5, (1.0 + wave * 0.7) * (0.5 + E * 1.3) + near * 2.2);
        if (s <= 0.35) continue;

        const b = Math.min(BINS - 1, (a * BINS / 0.45) | 0);
        bins[b].push(x, y + wave * 8 * E, s);
      }
    }

    for (let b = 0; b < BINS; b++) {
      const arr = bins[b];
      if (!arr.length) continue;
      const a = ((b + 0.5) / BINS) * 0.45;
      ctx.fillStyle = hot
        ? `rgba(56,${(189 + 45 * (1 - Math.min(1, E))) | 0},${(248 + 7 * (1 - Math.min(1, E))) | 0},${a})`
        : `rgba(226,232,240,${a})`;
      for (let k = 0; k < arr.length; k += 3) {
        const s = arr[k + 2];
        ctx.fillRect(arr[k], arr[k + 1], s, s);
      }
    }
  }
  requestAnimationFrame(frame);
})();

const pulse = (v) => { if (window.__field) window.__field.target = Math.max(window.__field.target, v); };

/* ───────────────────────── Health & Readiness ───────────────────────── */
let SERVING = [];
api('/health').then((h) => {
  SERVING = h.serving || [];
  const chip = $('#chipIndex');
  chip.classList.add('ready');
  const modeLabel = (h.mode || 'direct').toUpperCase();
  chip.querySelector('span').textContent =
    `${h.total_chunks.toLocaleString()} Chunks · Direct Retrieval [${modeLabel}]`;
  $('#footHost').textContent = `${h.embedder_variant} · ${h.index_tag}`;
  if (!h.stt_configured) {
    $('#micBtn').disabled = true;
    $('#hint').textContent = 'Voice disabled — no STT key on server. Typing works.';
  }
}).catch(() => {
  $('#chipIndex').querySelector('span').textContent = 'Service Unreachable';
  $('#hint').classList.add('err');
  $('#hint').textContent = "Unable to connect — Pregunta couldn't reach the knowledge service. Please try again.";
});

/* ───────────────────────── Answer Rendering ───────────────────────── */
const shell = $('#answerShell');

function setTier(el, state, value) {
  el.dataset.state = state;
  if (value !== undefined) el.querySelector('em').textContent = value;
}

function renderBudget(msValue) {
  const pctRaw = (msValue / 200) * 100;
  const fill = $('#budgetFill');
  fill.style.width = `${Math.min(100, pctRaw)}%`;
  fill.classList.toggle('over', msValue > 200);
  $('#budgetLabel').innerHTML = msValue > 200
    ? `<span style="color:var(--refuse)">${ms(msValue)} — budget exceeded</span>`
    : `${ms(msValue)} · ${(100 - pctRaw).toFixed(0)}% of 200ms budget unused`;
}

function renderAnswer(d, tier) {
  shell.hidden = false;
  document.body.classList.add('answered');

  const ans = $('#answer');
  ans.textContent = d.answer || '(no answer)';
  ans.classList.toggle('muted', ['abstain', 'refusal', 'greeting'].includes(d.answer_source));

  // Tier track
  const t1 = document.querySelector('.tier.t1');
  const t2 = document.querySelector('.tier.t2');
  setTier(t1, 'active', ms(d.fast_path_ms));
  
  const rewritten = !!d.generated_answer && d.generated_answer !== d.extractive_answer;
  if (tier === 'generated') {
    setTier(t2, 'active', `${ms(d.total_ms)} · ${rewritten ? 'synthesized' : 'verbatim'}`);
  } else if (tier === 'pending') {
    setTier(t2, 'pending', '···');
  } else if (d.reason === 'llm_reported_insufficient') {
    setTier(t2, 'declined', 'declined');
  } else if (d.llm_error) {
    setTier(t2, 'declined', 'failed');
  } else {
    setTier(t2, 'idle', '—');
  }

  renderBudget(d.fast_path_ms);

  // Verdicts & Grounding Info
  const v = [];
  const src = d.answer_source;
  const matchPct = d.support ? Math.round(d.support * 100) : null;

  if (d.grounded) {
    v.push(`<span class="v good">Grounded in your knowledge base${matchPct ? ` · ${matchPct}% match` : ''}</span>`);
  } else if (src === 'refusal') {
    v.push(`<span class="v bad">Refused · ${esc(d.reason || 'unsafe intent')}</span>`);
  } else if (src === 'greeting') {
    v.push(`<span class="v">Greeting · No retrieval spent</span>`);
  } else {
    v.push(`<span class="v warn">No sufficiently relevant information in knowledge base</span>`);
  }
  
  if (d.citations?.length) v.push(`<span class="v good">Cited [${d.citations.join(', ')}]</span>`);
  if (src === 'generated') {
    v.push(rewritten
      ? `<span class="v good">LLM Synthesized</span>`
      : `<span class="v">LLM Verbatim</span>`);
  }
  if (d.stt_ms)   v.push(`<span class="v">STT ${ms(d.stt_ms)}</span>`);
  if (d.llm_error) v.push(`<span class="v bad">LLM Error: ${esc(d.llm_error.slice(0, 48))}</span>`);
  $('#verdicts').innerHTML = v.join('');

  // Unsourced fallback view
  const un = $('#unsourced');
  if (un) {
    un.innerHTML = d.unsourced_answer
      ? `<div class="unsourced">
           <div class="tag-un">⚠ Model Knowledge · Not Corpus Grounded</div>
           <p lang="hi">${esc(d.unsourced_answer)}</p>
           <div class="caveat">Unverified — no direct citation found in knowledge corpus</div>
         </div>`
      : '';
  }

  const before = $('#beforeAfter');
  if (before) {
    before.innerHTML = rewritten
      ? `<details><summary>LLM Synthesis Comparison ↘</summary>
           <div class="src"><b>01 Extractive Pass</b><br>${esc(d.extractive_answer)}</div>
           <div class="src" style="border-color:var(--grounded)"><b>02 LLM Synthesis</b><br>${esc(d.generated_answer)}</div>
         </details>`
      : '';
  }

  const displayResults = d.results || (d.sources?.map((s) => ({
    content: s.text,
    score: s.score,
    source: s.unit_id,
    metadata: {}
  })) || []);

  $('#sources').innerHTML = displayResults.length
    ? `<details><summary>${displayResults.length} Matching Knowledge Records ↘</summary>` +
      displayResults.map((s, i) => `
        <div class="src">[${i + 1}] ${esc((s.content || s.text || '').slice(0, 320))}
          <div class="meta">Source: ${esc(s.source || s.unit_id)} · Score: ${s.score}</div>
        </div>`).join('') + `</details>`
    : '';

  shell.scrollIntoView({ block: 'center', behavior: 'smooth' });
}

/* ───────────────────────── Ask Question Flow ───────────────────────── */
let busy = false;

async function ask(question) {
  question = (question || '').trim();
  if (!question || busy) return;
  busy = true;
  pulse(0.9);
  $('#hint').classList.remove('err');
  $('#hint').textContent = 'Searching knowledge base directly…';

  try {
    const res = await api('/ask', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });

    if (!res.success && res.error) {
      $('#hint').classList.add('err');
      $('#hint').textContent = `Error: ${esc(res.error.message || 'Failed to retrieve')}`;
      return;
    }

    renderAnswer(res, 'idle');
    const pct = res.support ? Math.round(res.support * 100) : 0;
    $('#hint').textContent = res.grounded
      ? `Retrieved directly in ${ms(res.fast_path_ms)} (${pct}% match)`
      : `Searched in ${ms(res.fast_path_ms)} — no sufficient match found`;
    pulse(0.5);

    if (res.mode === 'llm' && !['refusal', 'greeting'].includes(res.answer_source)) {
      try {
        const full = await api('/ask', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, generate: true }),
        });
        renderAnswer(full, full.answer_source === 'generated' ? 'generated' : 'idle');
      } catch {
        setTier(document.querySelector('.tier.t2'), 'idle', '—');
      }
    }
  } catch (e) {
    $('#hint').classList.add('err');
    $('#hint').textContent = `Unable to connect — ${esc(e.message || "Pregunta couldn't reach the knowledge service. Please try again.")}`;
  } finally {
    busy = false;
  }
}

$('#askBtn').onclick = () => ask($('#q').value);
$('#q').addEventListener('keydown', (e) => { if (e.key === 'Enter') ask($('#q').value); });
$('#closeAns').onclick = () => {
  shell.hidden = true;
  document.body.classList.remove('answered');
};
document.querySelectorAll('.sample').forEach((b) => {
  b.onclick = () => { $('#q').value = b.dataset.q; ask(b.dataset.q); };
});

/* ───────────────────────── Microphone & Voice STT ───────────────────────── */
let recorder = null, chunks = [], analyser = null, audioCtx = null, meterRAF = 0;

function encodeWav(samples, rate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); str(8, 'WAVE');
  str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true);
  v.setUint16(34, 16, true); str(36, 'data'); v.setUint32(40, samples.length * 2, true);
  let o = 44;
  for (let i = 0; i < samples.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

async function toWav(blob) {
  const ac = new (window.AudioContext || window.webkitAudioContext)();
  const decoded = await ac.decodeAudioData(await blob.arrayBuffer());
  const rate = 16000;
  const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * rate), rate);
  const src = off.createBufferSource();
  src.buffer = decoded; src.connect(off.destination); src.start();
  const out = await off.startRendering();
  ac.close();
  return encodeWav(out.getChannelData(0), rate);
}

function meter() {
  if (!analyser) return;
  const buf = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteTimeDomainData(buf);
  let peak = 0;
  for (let i = 0; i < buf.length; i++) peak = Math.max(peak, Math.abs(buf[i] - 128) / 128);
  if (window.__field) window.__field.target = Math.min(1.4, peak * 3.2);
  meterRAF = requestAnimationFrame(meter);
}

$('#micBtn').onclick = async () => {
  const btn = $('#micBtn');
  if (recorder && recorder.state === 'recording') { recorder.stop(); return; }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks = [];

    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    audioCtx.createMediaStreamSource(stream).connect(analyser);
    meter();

    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => chunks.push(e.data);

    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      cancelAnimationFrame(meterRAF); analyser = null;
      audioCtx?.close(); audioCtx = null;
      btn.classList.remove('rec');
      $('#hint').textContent = 'Transcribing voice query…';

      try {
        const wav = await toWav(new Blob(chunks, { type: chunks[0]?.type || 'audio/webm' }));
        const fd = new FormData();
        fd.append('audio', wav, 'question.wav');
        fd.append('generate', 'true');
        const d = await api('/voice', { method: 'POST', body: fd });

        if (!d.stt_ok || !d.transcript) {
          $('#hint').classList.add('err');
          $('#hint').textContent = `Unable to transcribe — ${esc(d.stt_error || 'Empty audio transcript')}`;
          return;
        }
        $('#q').value = d.transcript;
        renderAnswer(d, d.answer_source === 'generated' ? 'generated' : 'idle');
        $('#hint').textContent =
          `Heard: “${d.transcript}” · STT ${ms(d.stt_ms)} · Fast path ${ms(d.fast_path_ms)}`;
      } catch (e) {
        $('#hint').classList.add('err');
        $('#hint').textContent = `Unable to connect — ${esc(e.message || "Pregunta couldn't reach the voice service. Please try again.")}`;
      }
    };

    recorder.start();
    btn.classList.add('rec');
    $('#hint').classList.remove('err');
    $('#hint').textContent = 'Listening — click microphone again to finish';
    setTimeout(() => { if (recorder?.state === 'recording') recorder.stop(); }, 30000);
  } catch (e) {
    $('#hint').classList.add('err');
    $('#hint').textContent = `Microphone blocked or unavailable — ${esc(e.message)}`;
  }
};

/* ───────────────────────── Live Telemetry Benchmark ───────────────────────── */
function countTo(el, target, decimals = 0, dur = 1100) {
  const from = parseFloat(el.textContent) || 0;
  const t0 = performance.now();
  const set = (v) => { el.textContent = v.toFixed(decimals); };
  let settled = false;

  const step = (now) => {
    if (settled) return;
    const p = Math.min(1, (now - t0) / dur);
    set(from + (target - from) * (1 - Math.pow(1 - p, 3)));
    if (p < 1) requestAnimationFrame(step); else settled = true;
  };
  requestAnimationFrame(step);

  setTimeout(() => { if (!settled) { settled = true; set(target); } }, dur + 150);
}

$('#benchBtn').onclick = async () => {
  const btn = $('#benchBtn');
  if (btn.classList.contains('busy')) return;
  btn.classList.add('busy');
  btn.querySelector('span').textContent = 'Running 100 queries…';
  pulse(1.0);

  try {
    const d = await api('/benchmark?n=100');
    const p = d.fast_path_ms;
    countTo($('#mP50'), p.p50, 1);
    countTo($('#mP70'), p.p70, 1);
    countTo($('#mP100'), p.p100, 1);
    $('#mHit').textContent = `${d.within_budget}/${d.n_queries}`;
    btn.querySelector('span').textContent = `${d.n_queries} Live Queries`;

    $('#stageRows').innerHTML = Object.entries(d.stages_ms).map(([k, s]) =>
      `<tr><td>${esc(k)}</td><td>${s.p50}</td><td>${s.p70}</td><td>${s.p90}</td><td>${s.p99}</td><td>${s.p100}</td></tr>`
    ).join('') +
      `<tr class="total"><td>Fast Path Total</td><td>${p.p50}</td><td>${p.p70}</td><td>${p.p90}</td><td>${p.p99}</td><td>${p.p100}</td></tr>`;
  } catch (e) {
    btn.querySelector('span').textContent = 'Benchmark failed — retry';
  } finally {
    btn.classList.remove('busy');
  }
};

/* ───────────────────────── Strategy Comparison ───────────────────────── */
$('#cmpBtn').onclick = async () => {
  const question = $('#q').value.trim();
  const btn = $('#cmpBtn');
  if (!question) { btn.textContent = 'Type a question above first ↑'; return; }
  btn.textContent = 'Comparing chunking strategies…';
  pulse(0.8);

  try {
    const d = await api('/compare', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    btn.textContent = `Re-run Comparison · ${esc(d.agreement)}`;
    $('#compare').innerHTML = `
      <table class="grid-table">
        <thead><tr><th>Strategy</th><th>Chunks</th><th>Search</th><th>Extract</th><th>Support</th></tr></thead>
        <tbody>${d.configs.map((c) => `
          <tr class="${c.is_served ? 'served' : ''}">
            <td>${esc(c.config)}${c.is_served ? '<span class="tag">Served</span>' : ''}</td>
            <td>${c.chunks.toLocaleString()}</td>
            <td>${c.search_ms}</td>
            <td>${c.extract_ms}</td>
            <td>${c.support}</td>
          </tr>`).join('')}
        </tbody>
      </table>`;
  } catch (e) {
    btn.textContent = 'Comparison failed — retry';
  }
};
