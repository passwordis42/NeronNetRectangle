// node llm/tests/learned-combination.browser.cjs [file-or-http URL]
// Browser plugin not available: test the real bundled page with Playwright.
// GEOMETRY_OFFLINE=1 tests the fully extracted offline package without network access.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { pathToFileURL } = require('node:url');
let playwright;
try { playwright = require('playwright'); }
catch (_) { playwright = require(path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')); }
const url = process.argv[2] || pathToFileURL(path.resolve(__dirname, '../神经网络教学.html')).href;
const screenshots = process.env.GEOMETRY_SCREENSHOTS || os.tmpdir();
const ID = 'square-trained-learned-combination-v1';
const near = (a, b, message) => assert.ok(Number.isFinite(a) && Math.abs(a - b) <= 1e-10 * Math.max(1, Math.abs(b)), `${message}: ${a} ≠ ${b}`);
const settle = page => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
const numeric = page => page.evaluate(() => JSON.stringify({ model: currentDisplayModel(), pixels: state.pixels, forward: latestForward,
  context: backpropReplay.active ? backpropReplay.context : null }));
const ready = async (page, step) => {
  await page.waitForFunction(s => backpropReplay.active && !backpropReplay.loading && !!backpropReplay.model && backpropReplay.context.step === s, step, { timeout: 60000 });
  await settle(page);
};
const seek = async (page, step) => {
  await page.locator('[data-geometry-step]').fill(String(step));
  await page.locator('[data-geometry-action="jump"]').click(); await ready(page, step);
};
const selectH2 = async (page, index = 0) => {
  await page.evaluate(i => selectNeuron(2, i), index);
  await page.locator('#secondLayerSpatial').waitFor(); await settle(page);
};
const noOverflow = async page => assert.equal(await page.evaluate(() => Math.max(0, document.documentElement.scrollWidth - innerWidth)), 0);

async function checkSpatial(page, all = false) {
  if (all && await page.locator('[data-spatial-expand]').getAttribute('aria-expanded') !== 'true') await page.locator('[data-spatial-expand]').click();
  const data = await page.evaluate(() => {
    const model = currentDisplayModel(), input = latestForward.activations[0].slice(), j = state.selectedNeuron.neuronIndex;
    const sources = model.layers[1].weights[j].map((w, i) => ({ index: i, weight: w, activation: latestForward.activations[1][i], contribution: w * latestForward.activations[1][i] }))
      .filter(s => s.weight !== 0).sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution) || a.index - b.index);
    const expected = input.map((_, i) => {
      const off = input.slice(), on = input.slice(); off[i] = 0; on[i] = 1;
      const f0 = forward(off), f1 = forward(on);
      return { index: i, z0: f0.preActivations[1][j], z1: f1.preActivations[1][j], a0: f0.activations[2][j], a1: f1.activations[2][j],
        deltaZ: f1.preActivations[1][j] - f0.preActivations[1][j], deltaActivation: f1.activations[2][j] - f0.activations[2][j] };
    });
    const actual = [...document.querySelectorAll('[data-spatial-pixel]')].map(n => Object.fromEntries([
      ['index', Number(n.dataset.spatialPixel)], ...['z0', 'z1', 'a0', 'a1', 'deltaZ', 'deltaActivation'].map(k => [k, Number(n.dataset[k])])
    ]));
    const actualSources = [...document.querySelectorAll('[data-spatial-source]')].map(n => ({ index: Number(n.dataset.spatialSource),
      activation: Number(n.dataset.sourceActivation), weight: Number(n.dataset.sourceWeight), contribution: Number(n.dataset.sourceContribution),
      weights: [...n.querySelectorAll('[data-spatial-source-weight]')].map(c => Number(c.dataset.value)),
      inputs: [...n.querySelectorAll('[data-spatial-source-input]')].map(c => Number(c.dataset.value)) }));
    return { input, expected, actual, sources, actualSources, first: model.layers[0], selected: state.selectedNeuron };
  });
  assert.equal(data.selected.layerIndex, 2);
  assert.equal(data.actual.length, 25);
  for (let i = 0; i < 25; i++) for (const key of ['index', 'z0', 'z1', 'a0', 'a1', 'deltaZ', 'deltaActivation']) near(data.actual[i][key], data.expected[i][key], `Pixel ${i + 1} ${key}`);
  assert.equal(data.actualSources.length, all ? data.sources.length : Math.min(6, data.sources.length));
  for (let k = 0; k < data.actualSources.length; k++) {
    const actual = data.actualSources[k], expected = data.sources[k];
    for (const key of ['index', 'activation', 'weight', 'contribution']) near(actual[key], expected[key], `H1 source ${k} ${key}`);
    assert.equal(actual.weights.length, 25); assert.equal(actual.inputs.length, 25);
    for (let i = 0; i < 25; i++) {
      near(actual.weights[i], data.first.weights[actual.index][i], 'Real source pixel weight');
      near(actual.inputs[i], data.first.weights[actual.index][i] * data.input[i], 'Real source pixel contribution');
    }
  }
  return data;
}

async function checkRanking(page, width, sorted) {
  if (await page.locator('#toggleContributionsButton').getAttribute('aria-expanded') !== 'true') await page.locator('#toggleContributionsButton').click();
  if ((await page.locator('#sortNetworkContributionButton').getAttribute('aria-pressed') === 'true') !== sorted) await page.locator('#sortNetworkContributionButton').click();
  await settle(page);
  const values = await page.evaluate(() => {
    const m = currentDisplayModel(), expected = latestForward.activations[2].map((a, index) => ({ index, activation: a, contribution: a * m.layers[2].weights[0][index] }))
      .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution) || a.index - b.index);
    const rows = [...document.querySelectorAll('#contributionRanking [data-ranking-index]')].map(n => ({ index: Number(n.dataset.rankingIndex), contribution: Number(n.dataset.rankingContribution) }));
    const nodes = [...document.querySelectorAll('#layers .neuron[data-layer="2"]')].map(n => ({ index: Number(n.dataset.neuron),
      activation: n.querySelector('[data-node-activation]')?.dataset.nodeActivation,
      contribution: n.querySelector('[data-node-contribution]')?.dataset.nodeContribution }));
    return { expected, rows, nodes, sum: expected.reduce((s, t) => s + t.contribution, 0), bias: m.layers[2].biases[0], output: latestForward.outputs[0],
      h1: [...document.querySelectorAll('#layers .neuron[data-layer="1"]')].map(n => Number(n.dataset.neuron)) };
  });
  assert.equal(values.rows.length, width); assert.deepEqual(values.rows.map(r => r.index), values.expected.map(r => r.index));
  for (let i = 0; i < width; i++) near(values.rows[i].contribution, values.expected[i].contribution, 'Ranking actual signed a×w');
  near(values.sum + values.bias, values.output, 'Ranking sum + output bias');
  if (sorted) {
    assert.deepEqual(values.nodes.map(n => n.index), values.expected.map(n => n.index));
    assert.deepEqual(values.h1, Array.from({ length: 119 }, (_, i) => i), 'Sorting H2 does not renumber H1');
    for (let i = 0; i < width; i++) {
      near(Number(values.nodes[i].activation), values.expected[i].activation, 'Sorted node response');
      near(Number(values.nodes[i].contribution), values.expected[i].contribution, 'Sorted node contribution');
    }
  }
  return values;
}

async function exportStep(page, type) {
  const pending = page.waitForEvent('download');
  await page.locator(`[data-geometry-action="${type}"]`).click();
  const download = await pending;
  return fs.readFile(await download.path(), 'utf8');
}

(async () => {
  const browser = await playwright.chromium.launch({ headless: true, ...(process.platform === 'win32' ? { executablePath: process.env.GEOMETRY_BROWSER || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe' } : {}) });
  try {
    const context = await browser.newContext({ viewport: { width: 1680, height: 1050 }, acceptDownloads: true });
    if (process.env.GEOMETRY_OFFLINE === '1') await context.setOffline(true);
    const page = await context.newPage(), errors = [];
    page.on('pageerror', e => errors.push(e.message)); await page.goto(url); await settle(page);
    assert.match(await page.title(), /神经网络/);
    assert.equal(await page.locator('#trainedVariantSelect').inputValue(), ID, 'New visits open the autonomous-combination model');
    assert.equal(await page.locator('#pixelGrid button').count(), 25);
    const artifact = await page.evaluate(() => SquareLearnedCombinationModelData), manifest = await page.evaluate(() => SquareLearnedCombinationTraceManifest);
    const width = artifact.model.topology[2], published = JSON.stringify(artifact.model.layers);
    assert.equal(artifact.learningMode, 'learned-combination'); assert.equal(artifact.verification.errors, 0);
    assert.deepEqual(artifact.model.topology, [25, 119, width, 1]);
    assert.equal(manifest.presetId, ID); assert.equal(manifest.steps > 101, true);

    for (let example = 0; example < 6; example++) {
      await page.locator(`[data-geometry-example="${example}"]`).click();
      for (const variant of ['complete', 'missing', 'extra']) {
        await page.locator(`[data-geometry-draw="${variant}"]`).click();
        const output = await page.evaluate(() => latestForward.outputs[0]);
        assert.equal(output > 1, variant === 'complete', `Example ${example} ${variant} classification`);
        await selectH2(page, example % width); await checkSpatial(page);
      }
    }
    await page.locator('[data-geometry-example="0"]').click();
    assert.doesNotMatch(await page.locator('#teachingView').innerText(), /2 · 五项几何条件|没有完整方框或待补角轮廓神经元/);
    assert.match(await page.locator('#teachingView').innerText(), /自主组合/);
    await selectH2(page, 0); assert.match(await page.locator('#weightInspector').innerText(), /没有预设检测目标/);
    assert.doesNotMatch(await page.locator('#weightInspector').innerText(), /满足此条件|不满足此条件/);
    const full = await checkSpatial(page, true); assert.equal(full.first.weights.length, 119, 'All 119 H1 units are possible free H2 sources; naturally zero weights may be omitted from the nonzero gallery');
    assert.equal(await page.locator('.input-comparison-row').count(), 238, 'The original comparison retains all 119 weights and all 119 input contributions, including zeros');
    const card = page.locator('[data-spatial-source]').first(), sourceIndex = Number(await card.getAttribute('data-spatial-source'));
    await card.click(); assert.deepEqual(await page.evaluate(() => state.selectedNeuron), { layerIndex: 1, neuronIndex: sourceIndex });
    await page.locator('[data-spatial-return]').click();
    const beforePixel = await numeric(page); await page.locator('[data-spatial-pixel="0"]').click();
    assert.equal(await numeric(page), beforePixel, 'Pixel inspection never edits the observed image');
    const savedPalette = await page.evaluate(() => state.featurePalette);
    if (savedPalette !== 'grayscale') await page.locator('#featurePaletteButton').click();
    const gray = await page.locator('[data-spatial-pixel]').evaluateAll(nodes => nodes.map(n => ({ value: Number(n.dataset.deltaZ), color: getComputedStyle(n).backgroundColor })));
    const max = Math.max(...gray.map(n => Math.abs(n.value)));
    for (const cell of gray) { const channel = Math.round(255 * ((max ? cell.value / max : 0) + 1) / 2); assert.equal(cell.color, `rgb(${channel}, ${channel}, ${channel})`); }
    await page.locator('#featurePaletteButton').click();
    await checkRanking(page, width, true); await selectH2(page, 0);
    assert.equal(await page.evaluate(() => JSON.stringify(state.model.layers)), published, 'Presentation controls preserve every trained parameter');
    await page.screenshot({ path: path.join(screenshots, 'learned-combination-desktop.png') });
    const roundTrip = await page.evaluate(() => { const before = buildPresetPayload(); importPresetObject(JSON.parse(JSON.stringify(before))); return JSON.stringify(buildPresetPayload().preset) === JSON.stringify(before.preset); });
    assert.equal(roundTrip, true); await page.reload();
    assert.equal(await page.locator('#trainedVariantSelect').inputValue(), ID);
    assert.equal(await page.evaluate(() => JSON.stringify(state.model.layers)), published);

    // A real legacy save, using the original MODE/TRAINED keys, must not migrate to the new default.
    await page.locator('#trainedVariantSelect').selectOption('square-trained-geometry-double-v1');
    await page.locator('#pixelGrid [data-index="0"]').click();
    const old = await page.evaluate(() => ({ storage: Object.fromEntries(Object.entries(localStorage)), model: JSON.stringify(state.model), pixels: state.pixels,
      mode: localStorage.getItem(MODE_STORAGE_KEY), variant: localStorage.getItem(TRAINED_VARIANT_STORAGE_KEY) }));
    assert.equal(old.mode, 'square-trained-geometry-double-v1'); assert.equal(old.variant, old.mode);
    const legacyContext = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    if (process.env.GEOMETRY_OFFLINE === '1') await legacyContext.setOffline(true);
    await legacyContext.addInitScript(storage => { for (const [key, value] of Object.entries(storage)) localStorage.setItem(key, value); }, old.storage);
    const legacyPage = await legacyContext.newPage(); legacyPage.on('pageerror', e => errors.push(e.message)); await legacyPage.goto(url);
    assert.equal(await legacyPage.locator('#trainedVariantSelect').inputValue(), old.mode);
    assert.equal(await legacyPage.evaluate(() => JSON.stringify(state.model)), old.model);
    assert.deepEqual(await legacyPage.evaluate(() => state.pixels), old.pixels); await legacyContext.close();
    await page.locator('#trainedVariantSelect').selectOption(ID);

    await page.locator('[data-geometry-mode="replay"]').click(); await ready(page, 0); await selectH2(page, 0);
    const initial = await page.evaluate(() => ({ model: currentDisplayModel(), info: backpropReplay.parameterInfo(backpropReplay.parameter) }));
    for (const layer of initial.model.layers) {
      assert.ok(layer.weights.flat().every(w => Math.abs(w) <= .1)); assert.ok(layer.biases.every(b => b === 0));
    }
    assert.equal(initial.info.gradient, null); assert.notEqual(JSON.stringify(initial.model.layers), JSON.stringify(artifact.model.layers));
    await checkSpatial(page, true);
    const sources = await page.locator('[data-replay-view="source"] option').evaluateAll(nodes => nodes.map(n => n.value));
    assert.deepEqual(sources, ['classification', 'local', 'boundary', 'total'], 'Only three actual losses appear, with an explicit total');
    const stages = await page.locator('[data-geometry-seek]').evaluateAll(nodes => [...new Set(nodes.map(n => Number(n.dataset.geometrySeek)))]);
    for (const stage of manifest.stages) assert.ok(stages.includes(stage), `Stage ${stage} comes from this manifest`);

    const visited = new Set([0]);
    for (const step of [...new Set([1, 100, 101, manifest.steps])]) {
      await seek(page, step); visited.add(step); await selectH2(page, 0); await checkSpatial(page, true);
      assert.ok(await page.evaluate(() => Number.isFinite(backpropReplay.parameterInfo(backpropReplay.parameter).totalGradient)));
      await checkRanking(page, width, true);
    }
    await page.locator('[data-replay-view="network"]').selectOption('observation');
    await page.locator('[data-geometry-phase]').selectOption('published');
    assert.equal(await page.evaluate(() => JSON.stringify(currentDisplayModel().layers)), JSON.stringify(artifact.model.layers));

    await seek(page, 1); await selectH2(page, 0);
    await page.locator('[data-replay-view="source"]').selectOption('classification');
    await page.locator('[data-replay-view="scope"]').selectOption('batch');
    const record = JSON.parse(await exportStep(page, 'json'));
    assert.equal(record.step, 1); assert.equal(record.traceId, manifest.id);
    assert.equal(record.learningMode, 'learned-combination'); assert.equal(record.before.length, manifest.parameterMap.length);
    assert.deepEqual(record.topology, artifact.model.topology); assert.deepEqual(record.trainingTargets.geometry || [], []);
    assert.equal(record.parameterMap.length, manifest.parameterCount);
    const classIndex = record.lossDefinitions.findIndex(d => d.id === 'classification');
    const upper = record.parameterMap.filter(p => p.layer >= 2);
    assert.ok(upper.some(p => record.gradient[p.index] !== 0), 'The freely learned upper layers receive actual initial classification gradients');
    for (const p of upper) {
      assert.ok(p.lower < 0 && p.upper > 0, 'No H2/output sign restriction');
      for (let k = 0; k < record.details.parts.length; k++) if (k !== classIndex) assert.equal(record.details.parts[k][p.index], 0, 'No H2/output auxiliary gradient');
      near(record.gradient[p.index], record.details.parts[classIndex][p.index], 'Upper-layer total equals classification gradient');
    }
    for (let layer = 2; layer <= 3; layer++) {
      const rows = record.parameterMap.filter(p => p.layer === layer && p.kind === 'weight');
      assert.equal(rows.length, artifact.model.topology[layer - 1] * artifact.model.topology[layer], 'All upper-layer connections are trainable');
    }
    for (const source of ['classification', 'local', 'boundary', 'total']) {
      await page.locator('[data-replay-view="source"]').selectOption(source);
      for (const scope of ['sample', 'batch']) {
        await page.locator('[data-replay-view="scope"]').selectOption(scope);
        const value = await page.evaluate(() => backpropReplay.parameterInfo(backpropReplay.parameter));
        assert.ok(Number.isFinite(value.gradient));
        if (source === 'local' || source === 'boundary') assert.equal(value.gradient, 0);
      }
    }
    await page.locator('[data-geometry-section="loss"]').click();
    assert.match(await page.locator('.trace-section').innerText(), /损失的完整数值计算/);
    assert.ok((await page.locator('.trace-section .trace-equation').first().innerText()).split('=').length >= 4);
    await page.locator('[data-geometry-action="audit"]').click();
    assert.deepEqual(await page.locator('[data-geometry-source] option').evaluateAll(nodes => nodes.map(n => n.value)), ['classification', 'local', 'boundary']);
    await page.locator('[data-geometry-action="table"]').click();
    assert.equal(await page.locator('[data-geometry-rows] tr').count(), manifest.parameterCount);
    await page.locator('[data-geometry-action="table"]').click();
    const csv = await exportStep(page, 'csv'); assert.equal(csv.trim().split(/\r?\n/).length, manifest.parameterCount + 1);
    assert.doesNotMatch(csv.split(/\r?\n/)[0], /几何条件|参与约束/);
    for (const [layer, neuron, input] of [[1, 0, 0], [2, 0, 0], [3, 0, 0], [1, 0, null], [2, 0, null], [3, 0, null]]) {
      const index = await page.evaluate(([l, j, i]) => backpropReplay.parameterIndex(l, j, i), [layer, neuron, input]);
      await page.locator('[data-geometry-parameter-select]').selectOption(String(index));
      const p = await page.evaluate(() => backpropReplay.parameterInfo(backpropReplay.parameter));
      assert.equal(p.layer, layer); assert.equal(p.kind, input === null ? 'bias' : 'weight');
      assert.ok([p.before, p.after, p.totalGradient, p.delta].every(Number.isFinite));
    }
    await page.evaluate(() => backpropReplay.selectConnection(1, 0, 1));
    assert.equal(await page.evaluate(() => backpropReplay.parameterInfo(-1).gradient), null);
    assert.match(await page.locator('#weightInspector').innerText(), /固定为零/);
    const h2Index = await page.evaluate(() => backpropReplay.parameterIndex(2, 0, 0));
    await page.locator('[data-geometry-parameter-select]').selectOption(String(h2Index));
    await page.locator('[data-geometry-section="backward"]').click();
    await page.locator('[data-replay-view="source"]').selectOption('classification');
    await page.locator('[data-replay-view="scope"]').selectOption('sample');
    assert.match(await page.locator('.trace-section').innerText(), /链式法则/);
    await page.locator('[data-geometry-section="update"]').click(); assert.match(await page.locator('.trace-section').innerText(), /Adam 候选值/);

    await seek(page, 101); await selectH2(page, 0);
    await page.locator('[data-replay-view="network"]').selectOption('observation');
    for (const phase of ['before', 'after']) { await page.locator('[data-geometry-phase]').selectOption(phase); await checkSpatial(page, true); await checkRanking(page, width, true); }
    await page.locator('[data-replay-view="network"]').selectOption('sample');
    await page.locator('[data-geometry-sample]').selectOption('1'); await ready(page, 101);
    assert.equal(await page.evaluate(() => backpropReplay.context.sampleIndex), 1);
    const sample = await checkSpatial(page, true), gradient = await page.evaluate(() => backpropReplay.parameterInfo(backpropReplay.parameter).totalGradient);
    if (!await page.evaluate(() => document.getElementById('replayCanvasFold').open)) await page.locator('[data-geometry-action="canvas"]').click();
    await page.locator('#pixelGrid [data-index="0"]').click();
    assert.deepEqual((await checkSpatial(page, true)).actual, sample.actual, 'Editing the observation canvas does not replace the selected training sample');
    assert.equal(await page.evaluate(() => backpropReplay.parameterInfo(backpropReplay.parameter).totalGradient), gradient);
    await page.locator('[data-replay-view="network"]').selectOption('observation'); await checkSpatial(page, true);
    await page.locator('[data-replay-view="layout"]').selectOption('vertical'); await page.locator('[data-network-zoom="in"]').click();
    assert.equal(await page.locator('#networkCanvas').getAttribute('data-replay-layout'), 'vertical');
    await page.locator('[data-network-zoom="fit"]').click();
    await page.locator('[data-geometry-action="fullscreen"]').click();
    await page.waitForFunction(() => !!document.fullscreenElement);
    assert.equal(await page.evaluate(() => !!document.fullscreenElement), true); await noOverflow(page);
    await page.locator('[data-geometry-action="fullscreen"]').click();
    await page.waitForFunction(() => !document.fullscreenElement);
    await page.screenshot({ path: path.join(screenshots, 'learned-combination-replay-desktop.png') });

    const fresh = manifest.chunks.filter(c => c.firstStep > 101 && c.lastStep < manifest.steps).slice(0, 3);
    assert.ok(fresh.length >= 3, 'The actual record has independent chunks for loading/failure checks');
    const [rapid, bad, missing] = fresh;
    const clearing = await page.evaluate(step => { backpropReplay.seekUpdate(step); return { loading: backpropReplay.loading,
      spatial: document.querySelectorAll('#secondLayerSpatial').length, ranked: document.querySelectorAll('[data-ranking-index]').length,
      nodes: document.querySelectorAll('[data-node-contribution]').length }; }, rapid.firstStep);
    assert.equal(clearing.loading, true); assert.equal(clearing.spatial, 0); assert.equal(clearing.ranked, 0); assert.equal(clearing.nodes, 0);
    await page.evaluate(() => { backpropReplay.seekUpdate(100); backpropReplay.seekUpdate(101); }); await ready(page, 101);
    await page.evaluate(c => { SquareLearnedCombinationTraceManifest.chunks[c.index].sha256 = '0'.repeat(64); backpropReplay.seekUpdate(c.firstStep); }, bad);
    await page.locator('.trace-error').waitFor({ timeout: 60000 });
    assert.match(await page.locator('.trace-error').innerText(), /校验失败/);
    assert.equal(await page.locator('#secondLayerSpatial').count(), 0); assert.equal(await page.locator('[data-ranking-index]').count(), 0);
    await page.evaluate(c => { SquareLearnedCombinationTraceManifest.chunks[c.index].sha256 = c.sha256; }, bad);
    await page.locator('[data-geometry-action="retry"]').click(); await ready(page, bad.firstStep); await selectH2(page, 0); await checkSpatial(page, true);
    await page.evaluate(c => { SquareLearnedCombinationTraceManifest.chunks[c.index].file = 'missing-learned-acceptance.js'; backpropReplay.seekUpdate(c.firstStep); }, missing);
    await page.locator('.trace-error').waitFor({ timeout: 60000 }); assert.match(await page.locator('.trace-error').innerText(), /加载失败/);
    assert.equal(await page.locator('#secondLayerSpatial').count(), 0);
    await page.evaluate(c => { SquareLearnedCombinationTraceManifest.chunks[c.index].file = c.file; }, missing);
    await page.locator('[data-geometry-action="retry"]').click(); await ready(page, missing.firstStep);

    await seek(page, 101); await page.reload();
    assert.equal(await page.locator('#trainedVariantSelect').inputValue(), ID);
    await page.locator('[data-geometry-mode="replay"]').click(); await ready(page, 101); await selectH2(page, 0);
    await page.setViewportSize({ width: 390, height: 844 }); await settle(page); await noOverflow(page);
    await checkSpatial(page, true); await checkRanking(page, width, true);
    await page.screenshot({ path: path.join(screenshots, 'learned-combination-replay-mobile.png') });
    await page.locator('[data-geometry-mode="forward"]').click(); await selectH2(page, 0);
    assert.equal(await page.locator('#detailSidebar').getAttribute('aria-modal'), 'true'); await noOverflow(page);
    await page.locator('#spatialInfluenceGrid').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(screenshots, 'learned-combination-details-mobile.png') });
    assert.deepEqual(errors, []);
    console.log(`PASS learned combination ${artifact.model.topology.join('→')}: fresh default/legacy saves, six examples, actual spatial and ranked contributions, random initialization, dynamic ${manifest.steps} real updates, class-only upper gradients, all parameters/exports, observation isolation, block boundaries/final match, failed-load retry, desktop/fullscreen/mobile.`);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
