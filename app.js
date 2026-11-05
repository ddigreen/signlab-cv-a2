'use strict';
(() => {
  const CLASSES = ['give_way', 'main_road', 'parking', 'pedestrian_crossing'];
  const NAMES = ['Give way', 'Main road', 'Parking', 'Pedestrian crossing'];
  const SHORT = ['Give way', 'Main road', 'Parking', 'Crossing'];
  const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
  const PAD = 'rgb(124,116,104)';
  const $ = id => document.getElementById(id);
  const state = {image: null, name: '', truth: null, sessions: {}, loading: {}, timing: {}, busy: false, stream: null, result: null, request: 0};
  const input = $('input-canvas'), cam = $('cam-canvas');
  const conditionNotes = {
    clean: 'Original input. No condition simulation applied.',
    dark: 'Deterministic simulation: multiply source RGB values by 0.35.',
    blur: 'Deterministic simulation: a 2 px browser blur on the 256 px source. This is not generated media.',
    low_resolution: 'Deterministic simulation: resize source to 40 × 40, then restore its shape.',
    occlusion: 'Deterministic simulation: cover the central 20% of the source with a gray square.'
  };
  const modelNames = {baseline: 'A1 MobileNetV3 baseline', enhanced: 'A2 validation-selected checkpoint'};
  const browser = /Edg\//.test(navigator.userAgent) ? 'Edge' : /Firefox\//.test(navigator.userAgent) ? 'Firefox' : /Chrome\//.test(navigator.userAgent) ? 'Chrome' : /Safari\//.test(navigator.userAgent) ? 'Safari' : 'Browser';
  const platform = navigator.userAgentData?.platform || navigator.platform || 'device';
  $('device-name').textContent = `${browser} / ${platform}`;

  function error(message) { $('error-box').textContent = message; $('error-box').hidden = !message; }
  function status(message) { $('model-status').textContent = message; }
  function setBusy(busy) { state.busy = busy; $('run-button').disabled = busy || !state.image; $('run-button').innerHTML = busy ? 'Recognizing… <span>↻</span>' : 'Recognize sign <span>→</span>'; }
  function canvas(w, h = w) { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; }
  function copyTo(from, to) { const ctx = to.getContext('2d'); ctx.clearRect(0, 0, to.width, to.height); ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high'; ctx.drawImage(from, 0, 0, to.width, to.height); }
  function softmax(values) { const max = Math.max(...values); const ex = Array.from(values, v => Math.exp(v - max)); const sum = ex.reduce((a, b) => a + b, 0); return ex.map(v => v / sum); }
  function escapeText(value) { return String(value ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c])); }
  function className(value) { const i = CLASSES.indexOf(value); return i >= 0 ? NAMES[i] : String(value ?? 'Unspecified'); }
  function showProbabilities(probs = null, winner = -1) {
    $('probabilities').innerHTML = NAMES.map((name, i) => `<div class="prob-row ${i === winner ? 'winner' : ''}"><span class="prob-name">${name}</span><div class="prob-track"><div class="prob-fill" style="width:${probs ? probs[i] * 100 : 0}%"></div></div><span class="prob-value">${probs ? (probs[i] * 100).toFixed(1) + '%' : '—'}</span></div>`).join('');
  }
  function invalidate() {
    state.result = null; state.request++;
    $('prediction-name').textContent = state.image ? 'Ready to recognize' : 'Waiting for an image';
    $('prediction-detail').textContent = state.image ? 'Run the selected model on this input.' : 'Select an example or upload a photograph.';
    $('confidence-value').textContent = '—'; $('cam-placeholder').classList.remove('hidden');
    $('last-latency').innerHTML = '—<small>ms</small>'; showProbabilities(); updateMean();
  }
  function updateMean() {
    const runs = state.timing[$('model-select').value] || [];
    $('avg-latency').innerHTML = `${runs.length ? (runs.reduce((a, b) => a + b, 0) / runs.length).toFixed(1) : '—'}<small>ms</small>`;
    $('latency-count').textContent = runs.length ? `${runs.length} timed run${runs.length > 1 ? 's' : ''}; first warm-up excluded` : 'No timed runs yet';
  }

  // Browser imageSmoothingQuality='high' approximates the Python bicubic/bilinear
  // resize; cross-browser interpolation can produce small numerical differences.
  function prepareImage(image, condition) {
    const iw = image.naturalWidth || image.width, ih = image.naturalHeight || image.height;
    const scale = 256 / Math.max(iw, ih), w = Math.max(1, Math.round(iw * scale)), h = Math.max(1, Math.round(ih * scale));
    const source = canvas(w, h), sx = source.getContext('2d', {willReadFrequently: true});
    sx.imageSmoothingEnabled = true; sx.imageSmoothingQuality = 'high';
    if (condition === 'low_resolution') {
      const low = canvas(40); low.getContext('2d').drawImage(image, 0, 0, 40, 40); sx.drawImage(low, 0, 0, w, h);
    } else if (condition === 'blur') {
      const clean = canvas(w, h); clean.getContext('2d').drawImage(image, 0, 0, w, h);
      sx.filter = 'blur(2px)'; sx.drawImage(clean, 0, 0); sx.filter = 'none';
    } else { sx.drawImage(image, 0, 0, w, h); }
    if (condition === 'dark') {
      const data = sx.getImageData(0, 0, w, h);
      for (let i = 0; i < data.data.length; i += 4) { data.data[i] *= 0.35; data.data[i + 1] *= 0.35; data.data[i + 2] *= 0.35; }
      sx.putImageData(data, 0, 0);
    }
    if (condition === 'occlusion') {
      const side = Math.sqrt(w * h * 0.2); sx.fillStyle = 'rgb(128,128,128)'; sx.fillRect((w - side) / 2, (h - side) / 2, side, side);
    }
    const box = canvas(256), bx = box.getContext('2d'); bx.fillStyle = PAD; bx.fillRect(0, 0, 256, 256); bx.drawImage(source, Math.floor((256 - w) / 2), Math.floor((256 - h) / 2));
    const resized = canvas(224), rx = resized.getContext('2d', {willReadFrequently: true}); rx.imageSmoothingEnabled = true; rx.imageSmoothingQuality = 'high'; rx.drawImage(box, 0, 0, 224, 224);
    const pixels = rx.getImageData(0, 0, 224, 224).data, tensor = new Float32Array(3 * 224 * 224), plane = 224 * 224;
    for (let p = 0; p < plane; p++) { for (let c = 0; c < 3; c++) tensor[c * plane + p] = (pixels[p * 4 + c] / 255 - MEAN[c]) / STD[c]; }
    return {canvas: resized, tensor};
  }
  function preview() {
    if (!state.image) return;
    copyTo(prepareImage(state.image, $('condition-select').value).canvas, input);
    $('input-placeholder').classList.add('hidden');
    $('input-label').textContent = $('condition-select').value === 'clean' ? '224 × 224' : 'SIMULATED';
  }
  async function loadImage(url, name, truth = null) {
    const img = new Image(); img.decoding = 'async';
    await new Promise((resolve, reject) => { img.onload = resolve; img.onerror = () => reject(new Error('This image could not be decoded. Try a JPG, PNG, or WEBP photograph.')); img.src = url; });
    state.image = img; state.name = name; state.truth = truth; error(''); invalidate(); preview(); setBusy(false);
    $('source-line').textContent = `${name}${truth ? ` · Synthetic reference label: ${className(truth)}` : ' · User-provided image'} · Whole-frame classification; no object localization.`;
    status('Image ready. Choose a model and recognize the sign.');
  }
  async function fileChosen(file) {
    if (!file) return;
    if (!file.type.startsWith('image/')) { error('Choose an image file (JPG, PNG, or WEBP).'); return; }
    if (file.size > 30 * 1024 * 1024) { error('Choose an image smaller than 30 MB.'); return; }
    stopCamera(); const url = URL.createObjectURL(file);
    try { await loadImage(url, file.name); document.querySelectorAll('.sample').forEach(e => e.classList.remove('selected')); }
    catch (e) { error(e.message); }
    finally { URL.revokeObjectURL(url); }
  }

  async function getSession(model) {
    if (state.sessions[model]) return state.sessions[model];
    if (state.loading[model]) return state.loading[model];
    if (!window.ort) throw new Error('The local ONNX Runtime library is missing. Restore web/vendor/ from the complete assignment package and serve the folder over HTTP.');
    ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false; ort.env.wasm.wasmPaths = new URL('vendor/', location.href).href;
    const promise = (async () => {
      status(`Loading ${modelNames[model]}…`);
      const url = `models/${model}.onnx`, response = await fetch(url);
      if (!response.ok) throw new Error(`The ${modelNames[model]} model file is unavailable (${response.status}). Expected web/${url}. No prediction has been made.`);
      const bytes = await response.arrayBuffer();
      if (bytes.byteLength < 1000) throw new Error('The model download is incomplete. Restore the ONNX model from the assignment package.');
      const session = await ort.InferenceSession.create(bytes, {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
      state.sessions[model] = session; return session;
    })();
    state.loading[model] = promise;
    try { return await promise; } finally { delete state.loading[model]; }
  }
  function heatColor(t) {
    const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 3)));
    const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 2)));
    const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 1)));
    return [r * 255, g * 255, b * 255];
  }
  function drawCAM(base, values, width = 7, height = 7) {
    const map = canvas(width, height), mx = map.getContext('2d'), pixels = mx.createImageData(width, height);
    const positive = Array.from(values, v => Math.max(0, Number(v))), min = Math.min(...positive), max = Math.max(...positive), span = max - min;
    for (let i = 0; i < positive.length; i++) {
      const normalized = span > 1e-12 ? (positive[i] - min) / span : 0;
      const color = heatColor(normalized); const k = i * 4;
      pixels.data[k] = color[0]; pixels.data[k + 1] = color[1]; pixels.data[k + 2] = color[2]; pixels.data[k + 3] = span > 1e-12 ? Math.round(145 * normalized) : 0;
    }
    mx.putImageData(pixels, 0, 0); copyTo(base, cam); const cx = cam.getContext('2d'); cx.imageSmoothingEnabled = true; cx.drawImage(map, 0, 0, cam.width, cam.height);
    $('cam-placeholder').classList.add('hidden'); return span > 1e-12;
  }
  async function recognize() {
    if (!state.image || state.busy) return;
    error(''); setBusy(true); const model = $('model-select').value, condition = $('condition-select').value, request = state.request;
    try {
      const session = await getSession(model), prepared = prepareImage(state.image, condition);
      const feeds = {[session.inputNames[0]]: new ort.Tensor('float32', prepared.tensor, [1, 3, 224, 224])};
      if (!state.timing[model]) { status('Warming up the local inference engine…'); await session.run(feeds); state.timing[model] = []; }
      status('Running inference and computing the class explanation…');
      const start = performance.now(), output = await session.run(feeds), elapsed = performance.now() - start;
      if (request !== state.request) { status('Input changed. Run recognition again for the current photograph.'); return; }
      const logits = output.logits || output[session.outputNames[0]], maps = output.cams || output[session.outputNames[1]];
      if (!logits || logits.data.length !== 4) throw new Error('The model output does not match the four-class assignment model.');
      if (!maps || maps.data.length !== 4 * 7 * 7) throw new Error('The model did not return the required class-specific Grad-CAM maps.');
      const probs = softmax(logits.data), winner = probs.indexOf(Math.max(...probs));
      const selectedMap = maps.data.slice(winner * 49, (winner + 1) * 49), hasCAM = drawCAM(prepared.canvas, selectedMap);
      copyTo(prepared.canvas, input); state.timing[model].push(elapsed); state.result = {model, condition, probabilities: probs, predicted: CLASSES[winner], elapsed_ms: elapsed};
      $('prediction-name').textContent = NAMES[winner]; $('confidence-value').textContent = `${(probs[winner] * 100).toFixed(1)}%`;
      $('prediction-detail').textContent = state.truth ? `Synthetic reference label: ${className(state.truth)} · ${CLASSES[winner] === state.truth ? 'Correct on this example' : 'Misclassified on this example'}` : 'Predicted class of the dominant sign in this image.';
      showProbabilities(probs, winner); $('last-latency').innerHTML = `${elapsed.toFixed(1)}<small>ms</small>`; updateMean();
      status(`${modelNames[model]} · Inference complete${hasCAM ? '' : ' · No positive Grad-CAM activation for this class'}.`);
    } catch (e) { console.error(e); error(e.message || String(e)); status('Recognition failed. No result is being substituted.'); }
    finally { setBusy(false); }
  }

  function stopCamera() {
    if (state.stream) state.stream.getTracks().forEach(track => track.stop());
    state.stream = null; $('camera').srcObject = null; $('camera-zone').classList.add('hidden');
  }
  async function startCamera() {
    error('');
    if (!navigator.mediaDevices?.getUserMedia) { error('Camera access requires HTTPS or localhost and a browser with camera support. You can still upload an image.'); return; }
    try {
      stopCamera(); state.stream = await navigator.mediaDevices.getUserMedia({video: {facingMode: {ideal: 'environment'}}, audio: false}); $('camera').srcObject = state.stream; $('camera-zone').classList.remove('hidden'); await $('camera').play();
    } catch (e) { error(`Camera unavailable: ${e.message}. Use Choose image to continue.`); }
  }
  async function captureFrame() {
    const video = $('camera'); if (!video.videoWidth) return;
    const frame = canvas(video.videoWidth, video.videoHeight); frame.getContext('2d').drawImage(video, 0, 0); stopCamera(); await loadImage(frame.toDataURL('image/jpeg', 0.95), 'Camera snapshot');
  }
  async function loadSamples() {
    try {
      const response = await fetch('samples/manifest.json'); if (!response.ok) return;
      const manifest = await response.json(), rows = manifest.images || manifest;
      $('samples').innerHTML = rows.map((row, i) => `<button class="sample" data-index="${i}" aria-label="Load synthetic ${escapeText(className(row.label))} example"><img src="samples/${escapeText(row.path)}" alt="Synthetic ${escapeText(className(row.label))} sign scene"><span>${escapeText(SHORT[CLASSES.indexOf(row.label)] || row.label)}</span></button>`).join('');
      document.querySelectorAll('.sample').forEach(button => button.addEventListener('click', async () => {
        const row = rows[Number(button.dataset.index)]; try { await loadImage(`samples/${row.path}`, row.title || row.path, row.label); document.querySelectorAll('.sample').forEach(b => b.classList.toggle('selected', b === button)); } catch (e) { error(e.message); }
      }));
    } catch (e) { console.warn('Sample gallery could not be loaded', e); }
  }
  function metric(value, percentage = false) {
    if (value === undefined || value === null || !Number.isFinite(Number(value))) return '—';
    const n = Number(value); return percentage ? `${(n <= 1 ? n * 100 : n).toFixed(2)}%` : n.toFixed(1);
  }
  async function loadResults() {
    try {
      const response = await fetch('results/summary.json'); if (!response.ok) throw new Error('unavailable');
      const data = await response.json(), rows = data.comparison || data.rows || data.models;
      if (!Array.isArray(rows) || !rows.length) throw new Error('empty');
      const modelLabels = {baseline:'A1 · measured rerun',real_control:'A2 · real-data control',flat_control:'A2 · flat-background control',maskgen:'A2 · MaskGen backgrounds'};
      const conditions = [...new Set(rows.map(row => row.condition || 'clean'))];
      const conditionLabels = {clean:'Original / clean',dark:'Darkened',blur:'Blurred',low_resolution:'Low resolution',jpeg:'JPEG compression',occlusion:'Occlusion'};
      let html = '';
      if (data.dataset) { const d = data.dataset; html += `<div class="metrics-summary"><span class="metric-pill"><strong>${escapeText(d.real_images ?? 569)}</strong>original photographs</span><span class="metric-pill"><strong>${escapeText(d.test_images ?? 96)}</strong>held-out images</span>${d.synthetic_images !== undefined ? `<span class="metric-pill"><strong>${escapeText(d.synthetic_images)}</strong>synthetic additions</span>` : ''}</div>`; }
      if (Array.isArray(data.training_selection) && data.training_selection.length && data.training_selection.every(row => Number(row.selected_epoch) === 0)) html += '<p class="selection-result">A2 training did not improve validation; the selected classifier retains A1 weights. All nine training runs retained epoch 0.</p>';
      html += `<div class="results-filter"><label for="results-condition">Evaluation condition</label><select id="results-condition">${conditions.map(condition => `<option value="${escapeText(condition)}">${escapeText(conditionLabels[condition] || condition)}</option>`).join('')}<option value="all">All recorded conditions</option></select></div><div id="results-table"></div>`;
      if (data.latency) { const t = data.latency; html += `<div class="recorded-latency"><p class="eyebrow">RECORDED PYTHON TIMING</p><div class="metrics-summary"><span class="metric-pill"><strong>${metric(t.prediction_mean_ms)} ms</strong>prediction</span><span class="metric-pill"><strong>${metric(t.prediction_plus_all_class_cam_mean_ms)} ms</strong>prediction + all-class CAM</span></div><p class="tiny">${escapeText(t.device || '')} · Batch ${escapeText(t.batch_size ?? 1)} · ${escapeText(t.repetitions ?? 'recorded')} repetitions after ${escapeText(t.warmups ?? 'recorded')} warm-ups. ${escapeText(t.scope || '')}. Browser timings are measured separately in the interactive lab.</p></div>`; }
      const notes = Array.isArray(data.notes) ? data.notes : data.notes ? [data.notes] : [];
      html += notes.map(note => `<p class="result-note">${escapeText(note)}</p>`).join('');
      if (data.environment) html += `<p class="tiny">Evaluation environment: ${escapeText(typeof data.environment === 'string' ? data.environment : JSON.stringify(data.environment))}. Browser latency above is measured separately.</p>`;
      $('results-content').innerHTML = html;
      const renderRows = () => {
        const selected = $('results-condition').value, filtered = rows.filter(row => selected === 'all' || (row.condition || 'clean') === selected);
        $('results-table').innerHTML = '<div class="table-wrap"><table><thead><tr><th>Model / training arm</th><th>Condition</th><th>Images</th><th>Seeds</th><th>Accuracy</th><th>Macro-F1</th></tr></thead><tbody>' + filtered.map(row => `<tr><td>${escapeText(modelLabels[row.model] || row.model || row.name || row.variant)}</td><td>${escapeText(conditionLabels[row.condition] || row.condition || 'Clean')}</td><td>${escapeText(row.n ?? row.samples ?? '—')}</td><td>${escapeText(row.seeds ?? '—')}</td><td>${metric(row.accuracy, true)}</td><td>${row.macro_f1 !== undefined ? Number(row.macro_f1).toFixed(4) : '—'}</td></tr>`).join('') + '</tbody></table></div>';
      };
      $('results-condition').addEventListener('change', renderRows); renderRows();
    } catch { $('results-content').innerHTML = '<p class="empty-message">No recorded A2 experiment file is bundled yet. Add the measured results to <code>web/results/summary.json</code>. The interactive lab always uses the actual model outputs.</p>'; }
  }
  async function loadGenerated() {
    try {
      const response = await fetch('generated/manifest.json'); if (!response.ok) throw new Error('unavailable');
      const manifest = await response.json(), images = manifest.images || manifest;
      if (!Array.isArray(images) || !images.length) throw new Error('empty');
      const intro = manifest.description ? `<p class="result-note">${escapeText(manifest.description)}</p>` : '';
      $('generated-content').innerHTML = intro + '<div class="generation-grid">' + images.map(row => `<article class="generation-card"><img loading="lazy" src="generated/${escapeText(row.path || row.file)}" alt="Synthetic ${escapeText(className(row.label))} example"><div><span class="synthetic-label">SYNTHETIC · ${escapeText(row.generator || 'MASKGEN')}</span><strong>${escapeText(className(row.label))}</strong>${row.kind ? `<p class="generation-kind">${escapeText(row.kind)}</p>` : ''}<p>${escapeText(row.prompt || '')}</p><span class="tiny">Seed ${escapeText(row.seed ?? 'recorded in provenance')}${row.accepted !== undefined ? ` · ${row.accepted ? 'Used in training pipeline' : 'Excluded from training'}` : ''}</span></div></article>`).join('') + '</div>';
    } catch { $('generated-content').innerHTML = '<p class="empty-message">No generated images are bundled yet. This gallery only displays real saved generator outputs with their recorded provenance; ordinary blur and darkening are kept in the interactive lab.</p>'; }
  }

  document.querySelectorAll('.tab').forEach(button => {
    button.setAttribute('role', 'tab'); button.setAttribute('aria-controls', `${button.dataset.tab}-panel`);
    button.addEventListener('click', () => { document.querySelectorAll('.tab').forEach(b => { const active = b === button; b.classList.toggle('active', active); b.setAttribute('aria-selected', String(active)); }); document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', p.id === `${button.dataset.tab}-panel`)); if (button.dataset.tab !== 'lab') stopCamera(); document.dispatchEvent(new CustomEvent('signlab:tabchange', {detail: button.dataset.tab})); });
  });
  document.querySelector('.tabs').setAttribute('role', 'tablist');
  $('upload-button').addEventListener('click', () => $('file-input').click());
  $('file-input').addEventListener('change', e => fileChosen(e.target.files[0]));
  const drop = $('drop-zone');
  ['dragenter', 'dragover'].forEach(type => drop.addEventListener(type, e => { e.preventDefault(); drop.classList.add('dragover'); }));
  ['dragleave', 'drop'].forEach(type => drop.addEventListener(type, e => { e.preventDefault(); drop.classList.remove('dragover'); }));
  drop.addEventListener('drop', e => fileChosen(e.dataTransfer.files[0]));
  $('camera-button').addEventListener('click', startCamera); $('capture-button').addEventListener('click', captureFrame); $('stop-camera').addEventListener('click', stopCamera);
  $('condition-select').addEventListener('change', () => { $('condition-note').textContent = conditionNotes[$('condition-select').value]; invalidate(); preview(); });
  $('model-select').addEventListener('change', () => { invalidate(); status('Model changed. Run recognition to compare the current input.'); });
  $('run-button').addEventListener('click', recognize); window.addEventListener('pagehide', stopCamera);
  showProbabilities(); loadSamples(); loadResults(); loadGenerated();
  window.SignLab = {state, prepareImage, softmax, recognize, loadImage};
})();
