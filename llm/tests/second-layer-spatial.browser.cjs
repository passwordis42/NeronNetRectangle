// node llm/tests/second-layer-spatial.browser.cjs [file-or-http URL]
// Browser plugin not available: validate the real bundled page with Playwright.
const assert = require('node:assert/strict');
const path = require('node:path');
const os = require('node:os');
const { pathToFileURL } = require('node:url');
let playwright;
try { playwright = require('playwright'); }
catch (_) { playwright = require(path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')); }
const url = process.argv[2] || pathToFileURL(path.resolve(__dirname, '../神经网络教学.html')).href;
const screenshots = process.env.GEOMETRY_SCREENSHOTS || os.tmpdir();
const panel = page => page.locator('#secondLayerSpatial');
const settle = page => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
const near = (actual, expected, label) => assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= 1e-10 * Math.max(1, Math.abs(expected)), `${label}: ${actual} ≠ ${expected}`);
const numeric = page => page.evaluate(() => JSON.stringify({
  model: cloneModelForPreset(currentDisplayModel()), pixels: state.pixels,
  networkPixels: backpropReplay.active ? backpropReplay.networkPixels : state.pixels,
  forward: latestForward, activationTypes: activationTypesOf(state)
}));

async function ready(page, step) {
  await page.waitForFunction(s => backpropReplay.active && !backpropReplay.loading && !!backpropReplay.model && backpropReplay.context.step === s, step, { timeout: 45000 });
  await settle(page);
}

async function selectH2(page, index) {
  await page.evaluate(i => selectNeuron(2, i), index);
  assert.deepEqual(await page.evaluate(() => state.selectedNeuron), { layerIndex: 2, neuronIndex: index });
  await panel(page).waitFor();
}

async function checkSpatial(page, sourceCount) {
  await settle(page);
  assert.equal(await panel(page).count(), 1);
  const result = await page.evaluate(() => {
    const selected = state.selectedNeuron, model = currentDisplayModel();
    const input = (backpropReplay.active ? backpropReplay.networkPixels : state.pixels).slice();
    const weights = model.layers[1].weights[selected.neuronIndex];
    const expectedSources = weights.map((weight, index) => ({ index, weight,
      activation: latestForward.activations[1][index], contribution: weight * latestForward.activations[1][index]
    })).filter(row => row.weight !== 0).sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution) || a.index - b.index);
    const expectedPixels = Array.from({ length: 25 }, (_, index) => {
      const off = input.slice(), on = input.slice(); off[index] = 0; on[index] = 1;
      // Independent reference: the page's actual unified forward pass, not a spatial renderer helper.
      const f0 = forward(off), f1 = forward(on);
      const z0 = f0.preActivations[1][selected.neuronIndex], z1 = f1.preActivations[1][selected.neuronIndex];
      const a0 = f0.activations[2][selected.neuronIndex], a1 = f1.activations[2][selected.neuronIndex];
      return { index, z0, z1, a0, a1, deltaZ: z1 - z0, deltaActivation: a1 - a0 };
    });
    const read = (node, attribute) => node.hasAttribute(attribute) ? Number(node.getAttribute(attribute)) : null;
    const actualSources = [...document.querySelectorAll('#spatialSourceCards [data-spatial-source]')].map(node => ({
      index: Number(node.dataset.spatialSource), activation: read(node, 'data-source-activation'),
      weight: read(node, 'data-source-weight'), contribution: read(node, 'data-source-contribution'),
      weights: [...node.querySelectorAll('[data-spatial-source-weight]')].map(cell => ({ index: Number(cell.dataset.spatialSourceWeight), value: read(cell, 'data-value') })),
      inputs: [...node.querySelectorAll('[data-spatial-source-input]')].map(cell => ({ index: Number(cell.dataset.spatialSourceInput), value: read(cell, 'data-value') }))
    }));
    const actualPixels = [...document.querySelectorAll('#spatialInfluenceGrid [data-spatial-pixel]')].map(node => ({
      index: Number(node.dataset.spatialPixel), deltaZ: read(node, 'data-delta-z'), deltaActivation: read(node, 'data-delta-activation'),
      z0: read(node, 'data-z0'), z1: read(node, 'data-z1'), a0: read(node, 'data-a0'), a1: read(node, 'data-a1')
    }));
    return { selected, input, firstLayer: model.layers[0], expectedSources, expectedPixels, actualSources, actualPixels,
      context: backpropReplay.active ? backpropReplay.context : null,
      text: document.getElementById('secondLayerSpatial').textContent };
  });
  assert.equal(result.selected.layerIndex, 2);
  if (sourceCount === 'all') assert.equal(result.actualSources.length, result.expectedSources.length, 'Expanded view includes every nonzero source connection');
  else if (sourceCount !== undefined) assert.equal(result.actualSources.length, Math.min(sourceCount, result.expectedSources.length));
  else assert.ok(result.actualSources.length === Math.min(6, result.expectedSources.length) || result.actualSources.length === result.expectedSources.length);
  assert.deepEqual(result.actualSources.map(row => row.index), result.expectedSources.slice(0, result.actualSources.length).map(row => row.index), 'Sources use absolute signed contribution order with real-index ties');
  for (let rowIndex = 0; rowIndex < result.actualSources.length; rowIndex++) {
    const actual = result.actualSources[rowIndex], expected = result.expectedSources[rowIndex];
    for (const key of ['activation', 'weight', 'contribution']) near(actual[key], expected[key], `Source H1-${actual.index + 1} ${key}`);
    assert.equal(actual.weights.length, 25); assert.equal(actual.inputs.length, 25);
    assert.deepEqual(actual.weights.map(cell => cell.index), Array.from({ length: 25 }, (_, i) => i));
    assert.deepEqual(actual.inputs.map(cell => cell.index), Array.from({ length: 25 }, (_, i) => i));
    for (let pixel = 0; pixel < 25; pixel++) {
      near(actual.weights[pixel].value, result.firstLayer.weights[actual.index][pixel], `Source ${actual.index} real W1 at X${pixel + 1}`);
      near(actual.inputs[pixel].value, result.input[pixel] * result.firstLayer.weights[actual.index][pixel], `Source ${actual.index} real x×W1 at X${pixel + 1}`);
    }
  }
  assert.equal(result.actualPixels.length, 25);
  assert.deepEqual(result.actualPixels.map(cell => cell.index), Array.from({ length: 25 }, (_, i) => i));
  for (let pixel = 0; pixel < 25; pixel++) for (const key of ['z0', 'z1', 'a0', 'a1', 'deltaZ', 'deltaActivation']) {
    near(result.actualPixels[pixel][key], result.expectedPixels[pixel][key], `Pixel X${pixel + 1} ${key}`);
  }
  assert.match(result.text, /固定权重|不把第二层权重.*像素权重/, 'The spatial map distinguishes conditional influence from a fixed pixel weight');
  assert.match(result.text, /损失.*偏导|偏导.*损失/, 'The spatial map distinguishes pixel changes from loss gradients');
  return result;
}

async function checkPixelClick(page, index = 12) {
  const before = await numeric(page), selection = await page.evaluate(() => state.selectedNeuron);
  await page.locator(`#spatialInfluenceGrid [data-spatial-pixel="${index}"]`).click();
  assert.equal(await numeric(page), before, 'Inspecting a pixel intervention does not edit the actual canvas or model');
  assert.deepEqual(await page.evaluate(() => state.selectedNeuron), selection);
  assert.equal(await page.locator('#spatialPixelDetail').isVisible(), true);
  const expected = await page.evaluate(i => {
    const model = currentDisplayModel(), input = (backpropReplay.active ? backpropReplay.networkPixels : state.pixels).slice();
    const j = state.selectedNeuron.neuronIndex;
    const off = input.slice(), on = input.slice(); off[i] = 0; on[i] = 1;
    return [forward(off).preActivations[1][j], forward(on).preActivations[1][j]].map(v => OutputContributionRanking.format(v));
  }, index);
  const detail = (await page.locator('#spatialPixelDetail').innerText()).replaceAll('−', '-');
  for (const value of expected) assert.ok(detail.includes(value), `Pixel detail includes actual preactivation ${value}`);
}

async function setPalette(page, grayscale) {
  const button = page.locator('#featurePaletteButton');
  const before = await numeric(page);
  if ((await button.getAttribute('aria-pressed') === 'true') !== grayscale) await button.click();
  assert.equal(await numeric(page), before, 'Palette changes are presentation only');
  if (!grayscale) return;
  const cells = await page.locator('#spatialInfluenceGrid [data-spatial-pixel]').evaluateAll(nodes => nodes.map(node => ({
    delta: Number(node.dataset.deltaZ), background: getComputedStyle(node).backgroundColor
  })));
  const max = Math.max(...cells.map(cell => Math.abs(cell.delta)));
  for (const cell of cells) {
    const channel = Math.round(255 * ((max ? cell.delta / max : 0) + 1) / 2);
    assert.equal(cell.background, `rgb(${channel}, ${channel}, ${channel})`, 'Influence grayscale is zero-centered with its own ±max scale');
  }
  assert.match(await panel(page).innerText(), /色标|最大|归一/, 'Influence scale is explained separately from fixed weight grayscale');
}

async function noOverflow(page) {
  assert.equal(await page.evaluate(() => Math.max(0, document.documentElement.scrollWidth - innerWidth)), 0);
}

async function checkCrossEntryReturn(page) {
  await selectH2(page, 0);
  const before = await numeric(page);
  const card = page.locator('#spatialSourceCards [data-spatial-source]').first();
  const firstSource = Number(await card.getAttribute('data-spatial-source'));
  await card.click();
  assert.deepEqual(await page.evaluate(() => state.selectedNeuron), { layerIndex: 1, neuronIndex: firstSource });
  await page.locator('[data-spatial-return]').click();
  assert.deepEqual(await page.evaluate(() => state.selectedNeuron), { layerIndex: 2, neuronIndex: 0 });

  // The old input-contribution list must update the same return destination as
  // a spatial source card, rather than retaining the previously visited H2.
  await selectH2(page, 1);
  const terms = page.locator('#weightInspector details.inspector-terms');
  if (!await terms.evaluate(node => node.open)) await terms.locator('summary').click();
  const oldSource = terms.locator('[data-inspector-source-layer="1"][data-inspector-source]').first();
  const secondSource = Number(await oldSource.getAttribute('data-inspector-source'));
  await oldSource.click();
  assert.deepEqual(await page.evaluate(() => state.selectedNeuron), { layerIndex: 1, neuronIndex: secondSource });
  const back = page.locator('[data-spatial-return]');
  assert.equal(await back.getAttribute('data-spatial-return'), '1', 'Legacy source navigation returns to the immediately preceding H2');
  await back.click();
  assert.deepEqual(await page.evaluate(() => state.selectedNeuron), { layerIndex: 2, neuronIndex: 1 });
  await checkSpatial(page);
  assert.equal(await numeric(page), before, 'Both source navigation paths preserve model, canvas, and forward values');
}

(async () => {
  const browser = await playwright.chromium.launch({ headless: true, ...(process.platform === 'win32' ? { executablePath: process.env.GEOMETRY_BROWSER || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe' } : {}) });
  try {
    const context = await browser.newContext({ viewport: { width: 1680, height: 1050 } });
    if (process.env.GEOMETRY_OFFLINE === '1') await context.setOffline(true);
    const page = await context.newPage(), errors = [];
    page.on('pageerror', error => errors.push(error.message)); await page.goto(url);
    assert.match(await page.title(), /神经/);
    assert.equal(await panel(page).count(), 0, 'H1 details do not mislabel a pixel map as second-layer evidence');
    await selectH2(page, 0); await checkSpatial(page, 6);

    const beforeExpand = await numeric(page);
    await page.locator('[data-spatial-expand]').click();
    const expanded = await checkSpatial(page, 'all');
    assert.equal(await numeric(page), beforeExpand);
    const source = expanded.actualSources[0].index;
    await page.locator(`#spatialSourceCards [data-spatial-source="${source}"]`).click();
    assert.deepEqual(await page.evaluate(() => state.selectedNeuron), { layerIndex: 1, neuronIndex: source });
    assert.equal(await panel(page).count(), 0);
    await page.locator('[data-spatial-return]').click();
    assert.deepEqual(await page.evaluate(() => state.selectedNeuron), { layerIndex: 2, neuronIndex: 0 });
    await checkSpatial(page);

    // Three actual square examples in both eligible models, including 47 unnamed
    // learned H1 features in the old guided model: their maps must still use real W1.
    for (const [id, examples] of [
      ['square-trained-geometry-double-v1', [[0, 0], [1, 3], [5, 4]]],
      ['square-trained-guided-50x30-v1', [[20, 20], [16, 16], [29, 29]]]
    ]) {
      await page.locator('#trainedVariantSelect').selectOption(id === 'square-trained-guided-50x30-v1' ? 'square-trained-guided-50-v1' : id);
      if (id === 'square-trained-guided-50x30-v1') await page.locator('#teachingDepthSelect').selectOption('2');
      if (!await page.locator('#teachingView').isVisible()) await page.locator('#teachingViewButton').click();
      await checkCrossEntryReturn(page);
      for (const [example, neuron] of examples) {
        if (id.includes('geometry')) await page.locator(`[data-geometry-example="${example}"]`).click();
        else {
          await page.locator(`[data-teaching-example="${example}"]`).click();
          await page.locator('[data-teaching-draw-example]').click();
        }
        await selectH2(page, neuron);
        const baseline = await checkSpatial(page); await checkPixelClick(page);
        const model = await page.evaluate(() => JSON.stringify(cloneModelForPreset(state.model)));
        await page.locator('#pixelGrid [data-index="0"]').click();
        const edited = await checkSpatial(page);
        assert.notDeepEqual(edited.actualPixels, baseline.actualPixels, 'Canvas changes recompute the conditional pixel influence');
        assert.equal(await page.evaluate(() => JSON.stringify(cloneModelForPreset(state.model))), model);
        await page.locator('#pixelGrid [data-index="0"]').click();
        await checkSpatial(page);
      }
    }

    await setPalette(page, true); await checkSpatial(page); await setPalette(page, false);
    const beforeImport = await numeric(page);
    await page.evaluate(() => importPresetObject(buildPresetPayload()));
    assert.equal(await numeric(page), beforeImport, 'Inspecting source composition does not alter exported/imported model parameters');
    await checkSpatial(page);
    await page.reload(); await checkSpatial(page);
    assert.equal(await numeric(page), beforeImport, 'Refreshing the selected H2 restores the same actual model and input');
    await page.locator('#spatialSourceCards [data-spatial-source]').first().scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(screenshots, 'second-layer-spatial-sources-desktop.png') });
    await page.locator('#spatialInfluenceGrid').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(screenshots, 'second-layer-spatial-influence-desktop.png') });

    // Only the two requested H2 types receive this panel; an output called layer 2 does not.
    for (const id of ['square-trained-geometry-single-v1', 'square-trained-partial-50-v5', 'square-trained-guided-50-v1']) {
      await page.locator('#trainedVariantSelect').selectOption(id);
      if (id === 'square-trained-guided-50-v1') await page.locator('#teachingDepthSelect').selectOption('1');
      await page.evaluate(() => selectNeuron(2, 0));
      assert.equal(await panel(page).count(), 0);
    }

    await page.locator('#trainedVariantSelect').selectOption('square-trained-geometry-double-v1');
    await selectH2(page, 0);
    await page.locator('[data-geometry-mode="replay"]').click(); await ready(page, 0);
    await selectH2(page, 0);
    const initialization = await checkSpatial(page);
    assert.equal(initialization.context.step, 0, 'Random initialization uses real initial parameters, not a published model');
    await checkPixelClick(page);
    const loading = await page.evaluate(() => {
      backpropReplay.seekUpdate(100);
      return { loading: backpropReplay.loading, panels: document.querySelectorAll('#secondLayerSpatial').length };
    });
    assert.equal(loading.loading, true); assert.equal(loading.panels, 0, 'Historical spatial evidence is cleared while a different update loads');
    await ready(page, 100); await selectH2(page, 0);
    const trained = await checkSpatial(page);
    assert.notDeepEqual(trained.actualPixels, initialization.actualPixels, 'Later updates use their own learned parameters');
    await page.locator('[data-replay-view="network"]').selectOption('observation');
    for (const phase of ['before', 'after']) {
      await page.locator('[data-geometry-phase]').selectOption(phase); await checkSpatial(page);
    }
    const beforeSource = (await checkSpatial(page)).actualPixels;
    await page.locator('[data-replay-view="source"]').selectOption('total');
    assert.deepEqual((await checkSpatial(page)).actualPixels, beforeSource, 'A loss-gradient source selector does not change pixel intervention results');
    await page.locator('[data-replay-view="network"]').selectOption('sample');
    await page.locator('[data-geometry-sample]').selectOption('1');
    await page.waitForFunction(() => !backpropReplay.loading && backpropReplay.context.sampleIndex === 1, null, { timeout: 45000 });
    const sample = await checkSpatial(page);
    const historyGradient = await page.evaluate(() => backpropReplay.parameterInfo(backpropReplay.parameter).totalGradient);
    if (!await page.evaluate(() => document.getElementById('replayCanvasFold').open)) await page.locator('[data-geometry-action="canvas"]').click();
    await page.locator('#pixelGrid [data-index="0"]').click();
    assert.deepEqual((await checkSpatial(page)).actualPixels, sample.actualPixels, 'Editing the observation canvas does not silently replace the selected training sample');
    assert.equal(await page.evaluate(() => backpropReplay.parameterInfo(backpropReplay.parameter).totalGradient), historyGradient);
    await page.locator('[data-replay-view="network"]').selectOption('observation'); await checkSpatial(page);
    await page.locator('[data-replay-view="network"]').selectOption('sample');
    const sourceCard = sample.actualSources[0].index;
    await page.locator(`#spatialSourceCards [data-spatial-source="${sourceCard}"]`).click();
    assert.equal(await panel(page).count(), 0);
    await page.locator('[data-spatial-return]').click(); await checkSpatial(page);
    await setPalette(page, true); await checkSpatial(page); await setPalette(page, false);

    // A deliberately corrupted checksum is confined to this browser session.
    const chunk = await page.evaluate(() => {
      const index = SquareGeometryDoubleTraceManifest.chunks.findIndex(c => c.firstStep === 701);
      return { index, hash: SquareGeometryDoubleTraceManifest.chunks[index].sha256 };
    });
    await page.evaluate(i => { SquareGeometryDoubleTraceManifest.chunks[i].sha256 = '0'.repeat(64); backpropReplay.seekUpdate(701); }, chunk.index);
    await page.locator('.trace-error').waitFor({ timeout: 45000 });
    assert.equal(await panel(page).count(), 0, 'A failed record never shows stale or published-model spatial evidence');
    await page.evaluate(({ index, hash }) => { SquareGeometryDoubleTraceManifest.chunks[index].sha256 = hash; }, chunk);
    await page.locator('[data-geometry-action="retry"]').click(); await ready(page, 701); await selectH2(page, 0); await checkSpatial(page);

    await page.setViewportSize({ width: 390, height: 844 }); await settle(page);
    await checkSpatial(page); await noOverflow(page); await checkPixelClick(page);
    await page.locator('#spatialInfluenceGrid').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(screenshots, 'second-layer-spatial-replay-mobile.png') });
    await page.locator('[data-geometry-mode="forward"]').click(); await selectH2(page, 0);
    assert.equal(await page.locator('#detailSidebar').getAttribute('aria-modal'), 'true');
    await setPalette(page, true); await checkSpatial(page); await noOverflow(page);
    await page.locator('#spatialInfluenceGrid').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(screenshots, 'second-layer-spatial-drawer-mobile.png') });
    assert.deepEqual(errors, []);
    console.log('PASS second-layer spatial: eligible H2 only, top-six/all nonzero real sources and H1 return across spatial cards/legacy input lists in both models, exact 25-cell W1/xW1 and pixel z/a interventions, three square examples in both models, canvas updates, pixel inspection isolation, grayscale, import/refresh, random/history/source/phase fidelity, loading/checksum retry, desktop and 390px.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
