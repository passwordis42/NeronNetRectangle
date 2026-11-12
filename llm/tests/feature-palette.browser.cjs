// node llm/tests/feature-palette.browser.cjs [file-or-http URL]
const assert = require('node:assert/strict');
const path = require('node:path');
const os = require('node:os');
const { pathToFileURL } = require('node:url');
let playwright;
try { playwright = require('playwright'); }
catch (_) { playwright = require(path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')); }
const url = process.argv[2] || pathToFileURL(path.resolve(__dirname, '../神经网络教学.html')).href;
const gray = value => { const channel = Math.round(255 * (Math.max(-1, Math.min(1, value)) + 1) / 2); return `rgb(${channel}, ${channel}, ${channel})`; };
const snapshot = page => page.evaluate(() => {
  const preset = buildPresetPayload().preset;
  // Ordinary experiment names include the export timestamp; actual preset data do not.
  delete preset.name;
  const model = currentDisplayModel();
  // Imported model records may omit the artifact's redundant `activations` field;
  // effective activation settings are already preserved in the exported preset.
  return JSON.stringify({ model: { topology: model.topology, layers: model.layers }, pixels: state.pixels, forward: latestForward, preset });
});
async function activePaletteContrast(page) {
  const actual = await page.locator('#featurePaletteButton').evaluate(button => {
    const css = getComputedStyle(button);
    return { text: button.textContent.trim(), foreground: css.color, background: css.backgroundColor,
      transition: css.transitionDuration, hovered: button.matches(':hover') };
  });
  const luminance = color => {
    const channels = color.match(/[\d.]+/g).slice(0, 3).map(channel => {
      const v = Number(channel) / 255;
      return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4;
    });
    return channels[0] * .2126 + channels[1] * .7152 + channels[2] * .0722;
  };
  const foreground = luminance(actual.foreground), background = luminance(actual.background);
  actual.contrast = (Math.max(foreground, background) + .05) / (Math.min(foreground, background) + .05);
  assert.equal(actual.text, '恢复彩色');
  assert.ok(actual.contrast >= 4.5, `Active palette button must remain legible immediately after activation: ${JSON.stringify(actual)}`);
  return actual;
}
async function palette(page, grayscale, touch = false) {
  const button = page.locator('#featurePaletteButton');
  if ((await button.getAttribute('aria-pressed') === 'true') !== grayscale) {
    if (touch) await button.tap(); else await button.click();
  }
  // Check before any data-table assertions: a background animation must not
  // briefly make the white active label disappear into its old white surface.
  if (grayscale) await activePaletteContrast(page);
  assert.equal(await button.getAttribute('aria-pressed'), String(grayscale));
  assert.equal(await page.evaluate(() => state.featurePalette), grayscale ? 'grayscale' : 'color');
}
async function comparisons(page, count) {
  const result = await page.evaluate(() => {
    const details = getSelectedWeightDetails();
    return ['完整输入权重', '当前输入贡献'].map((label, side) => [...document.querySelectorAll(`#weightInspector .input-comparison [aria-label="${label}"] [data-inspector-pixel]`)].map(node => {
      const item = details.inputWeights[Number(node.dataset.inspectorPixel)], style = getComputedStyle(node);
      return { index: item.index, value: side ? item.contribution : item.value, color: style.backgroundColor, image: style.backgroundImage,
        text: node.querySelector('.contribution-value,b').textContent.trim() };
    }));
  });
  for (const side of result) {
    assert.equal(side.length, count);
    for (const item of side) {
      assert.equal(item.color, gray(item.value), `Fixed gray scale for input ${item.index + 1}, value ${item.value}`);
      assert.equal(item.image, 'none', 'Finite values are not marked unknown');
      const displayed = Number(item.text.replaceAll('−', '-'));
      assert.ok(Math.abs(displayed - item.value) <= 5.1e-7 * Math.max(1, Math.abs(item.value)), 'Palette changes must not clamp or replace the real displayed number');
    }
  }
  return result;
}
async function ready(page, step) {
  await page.waitForFunction(s => backpropReplay.active && !backpropReplay.loading && !!backpropReplay.model && backpropReplay.context.step === s, step, { timeout: 45000 });
}
async function checkMetric(page, legacy = false) {
  const cells = await page.evaluate(old => {
    const details = getSelectedWeightDetails();
    const selector = old ? '#weightInspector .replay-parameter-grid [data-replay-parameter]' : '#weightInspector .geometry-metric-cell';
    return [...document.querySelectorAll(selector)].map(node => {
      const index = old ? Number(node.dataset.replayParameter) : backpropReplay.parameterIndex(details.layerIndex, details.neuronIndex, Number(node.dataset.geometryInput));
      const info = backpropReplay.parameterInfo(index);
      const value = info?.[backpropReplay.metric === 'gradient' ? 'gradient' : backpropReplay.metric === 'delta' ? 'delta' : 'weight'];
      const css = getComputedStyle(node);
      return { value: Number.isFinite(value) ? value : null, color: css.backgroundColor, image: css.backgroundImage,
        text: node.querySelector('.contribution-value,b').textContent.trim() };
    });
  }, legacy);
  assert.ok(cells.length > 0);
  for (const cell of cells) {
    if (cell.value === null) {
      assert.match(cell.image, /gradient/i, 'Unavailable gradient is visibly hatched');
      assert.match(cell.text, /—|未知/, 'Unavailable gradient is not presented as numeric zero');
    } else {
      assert.equal(cell.color, gray(cell.value));
      assert.equal(cell.image, 'none');
    }
  }
  return cells;
}

(async () => {
  const browser = await playwright.chromium.launch({ headless: true, ...(process.platform === 'win32' ? { executablePath: process.env.GEOMETRY_BROWSER || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe' } : {}) });
  try {
    const context = await browser.newContext({ viewport: { width: 1600, height: 1050 } });
    if (process.env.GEOMETRY_OFFLINE === '1') await context.setOffline(true);
    const page = await context.newPage(), errors = [];
    page.on('pageerror', error => errors.push(error.message)); await page.goto(url);
    assert.equal(await page.locator('#featurePaletteButton').getAttribute('aria-pressed'), 'false', 'First visit defaults to color');

    // Temporary ordinary-experiment fixture only: never alter the published models.
    await page.locator('#normalModeButton').click();
    await page.evaluate(() => {
      const values = [-2, -1, -.5, 0, .5, 1, 2];
      state.hiddenLayers = [{ size: 2, activation: 'tanh' }];
      state.outputs = [{ name: 'Palette fixture', threshold: 1, target: 2 }];
      state.outputActivation = 'linear'; state.weightSettings = { min: -2, max: 2, precision: 6 };
      state.pixels = Array(25).fill(1);
      state.model = { topology: [25, 2, 1], layers: [
        { weights: [Array.from({ length: 25 }, (_, i) => values[i] ?? 0), Array.from({ length: 25 }, (_, i) => [-.5, 0, .5][i] ?? 0)], biases: [0, 0] },
        { weights: [[.25, -.5]], biases: [0] }
      ] };
      state.selectedNeuron = { layerIndex: 1, neuronIndex: 0 }; state.inspectorTab = 'features'; renderAll();
    });
    await palette(page, false);
    const before = await snapshot(page);
    const originalColors = await page.locator('.input-comparison [data-inspector-pixel]').evaluateAll(nodes => nodes.map(node => getComputedStyle(node).backgroundColor));
    await palette(page, true);
    assert.equal(await snapshot(page), before, 'Changing palette leaves parameters, forward pass and exported preset intact');
    const fixture = await comparisons(page, 25);
    assert.deepEqual(fixture[0].slice(0, 7).map(item => item.color), [0, 0, 64, 128, 191, 255, 255].map(v => `rgb(${v}, ${v}, ${v})`));
    assert.deepEqual(fixture[0].map(item => item.color), fixture[1].map(item => item.color), 'The same value has the same color in weights and contributions');
    assert.equal(fixture[0][0].text, '-2.000000'); assert.equal(fixture[0][6].text, '2.000000');
    await palette(page, false);
    assert.deepEqual(await page.locator('.input-comparison [data-inspector-pixel]').evaluateAll(nodes => nodes.map(node => getComputedStyle(node).backgroundColor)), originalColors, 'Original signed colors are restored exactly');
    await palette(page, true);
    await page.evaluate(() => selectNeuron(1, 1));
    const second = await comparisons(page, 25);
    assert.equal(second[0][0].color, fixture[0][2].color, 'Changing the largest weight does not normalize equal values to new shades');
    assert.equal(await page.locator('#featurePaletteButton').getAttribute('aria-pressed'), 'true');
    await page.reload();
    assert.equal(await page.locator('#featurePaletteButton').getAttribute('aria-pressed'), 'true', 'Grayscale survives refresh');
    await comparisons(page, 25);

    // Real networks: all 25/119/352/5 incoming items, with truthful dense zero entries.
    await page.locator('#decimalTeachingModeButton').click();
    let doublePreset;
    for (const [kind, selections] of [['double', [[1, 60, 25], [2, 0, 119], [3, 0, 5]]], ['single', [[1, 60, 25], [2, 0, 352]]]]) {
      await page.locator('#trainedVariantSelect').selectOption(`square-trained-geometry-${kind}-v1`);
      await palette(page, true);
      for (const [layer, neuron, count] of selections) {
        await page.evaluate(([l, n]) => selectNeuron(l, n), [layer, neuron]);
        const actual = await snapshot(page); await comparisons(page, count);
        if (kind === 'double' && layer === 1) await page.screenshot({ path: path.join(os.tmpdir(), 'feature-palette-desktop.png') });
        await palette(page, false); await palette(page, true);
        assert.equal(await snapshot(page), actual, `${kind} layer ${layer} is numerically unchanged`);
      }
      if (kind === 'double') doublePreset = await page.evaluate(() => buildPresetPayload());
    }

    // Importing another verified model should keep this display preference, while
    // replacing its actual parameters. Reload checks a real teaching save slot.
    await page.evaluate(payload => importPresetObject(payload), doublePreset);
    assert.equal(await page.locator('#trainedVariantSelect').inputValue(), 'square-trained-geometry-double-v1');
    assert.equal(await page.locator('#featurePaletteButton').getAttribute('aria-pressed'), 'true', 'Import keeps the current palette');
    assert.deepEqual(await page.evaluate(() => state.model), doublePreset.preset.model);
    await page.evaluate(() => selectNeuron(2, 0)); await comparisons(page, 119);
    const teachingSaved = await snapshot(page);
    await page.reload();
    assert.equal(await page.locator('#featurePaletteButton').getAttribute('aria-pressed'), 'true', 'Teaching save slot restores grayscale');
    await comparisons(page, 119);
    assert.equal(await snapshot(page), teachingSaved, 'Reload retains the verified teaching parameters and forward result');

    // New replay: initial gradients are unknown, actual weights/gradients/deltas
    // retain fixed grayscale scaling at every layer and after update changes.
    await page.locator('#trainedVariantSelect').selectOption('square-trained-geometry-double-v1');
    await page.locator('[data-geometry-mode="replay"]').click(); await ready(page, 0); await palette(page, true);
    await page.locator('[data-replay-view="metric"]').selectOption('gradient');
    const initial = await checkMetric(page); assert.ok(initial.every(cell => cell.value === null));
    await page.locator('[data-geometry-step]').fill('100'); await page.locator('[data-geometry-action="jump"]').click(); await ready(page, 100);
    for (const [layer, neuron, input, count] of [[1, 60, 6, 25], [2, 0, 0, 119], [3, 0, 0, 5]]) {
      await page.evaluate(([l, n, i]) => backpropReplay.selectParameter(backpropReplay.parameterIndex(l, n, i)), [layer, neuron, input]);
      await comparisons(page, count);
      for (const metric of ['weight', 'gradient', 'delta']) {
        await page.locator('[data-replay-view="metric"]').selectOption(metric); await checkMetric(page);
      }
      const actual = await snapshot(page); await palette(page, false); await palette(page, true); assert.equal(await snapshot(page), actual);
    }

    // Legacy engine has a different parameter layout but shares the palette.
    await page.locator('#trainedVariantSelect').selectOption('square-trained-partial-50-v5');
    await page.locator('[data-trace-mode="replay"]').click(); await ready(page, 0); await palette(page, true);
    await page.locator('[data-replay-view="metric"]').selectOption('gradient');
    assert.ok((await checkMetric(page, true)).every(cell => cell.value === null));
    await page.locator('#trace-step').fill('6001'); await page.locator('[data-trace-action="jump"]').click(); await ready(page, 6001);
    for (const index of [0, 1300]) {
      await page.evaluate(i => backpropReplay.selectParameter(i), index);
      await comparisons(page, index === 0 ? 25 : 50);
      for (const metric of ['weight', 'gradient', 'delta']) { await page.locator('[data-replay-view="metric"]').selectOption(metric); await checkMetric(page, true); }
    }
    const legacy = await snapshot(page); await palette(page, false); await palette(page, true); assert.equal(await snapshot(page), legacy);

    const phoneContext = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
    if (process.env.GEOMETRY_OFFLINE === '1') await phoneContext.setOffline(true);
    const phone = await phoneContext.newPage(); phone.on('pageerror', error => errors.push(error.message)); await phone.goto(url);
    await phone.locator('#openFeaturesButton').tap();
    assert.equal(await phone.locator('#featurePaletteButton').isVisible(), true);
    const phoneBefore = await snapshot(phone); await palette(phone, true, true); await comparisons(phone, 25);
    const phoneButton = await activePaletteContrast(phone);
    console.log('Mobile palette button after touch:', JSON.stringify(phoneButton));
    assert.equal(await snapshot(phone), phoneBefore);
    assert.equal(await phone.evaluate(() => Math.max(0, document.documentElement.scrollWidth - innerWidth)), 0);
    await phone.screenshot({ path: path.join(os.tmpdir(), 'feature-palette-mobile.png') });
    await phone.locator('#closeInspectorButton').tap();
    await phone.locator('[data-geometry-mode="replay"]').click(); await ready(phone, 0);
    await palette(phone, true); await phone.locator('[data-replay-view="metric"]').selectOption('gradient'); await checkMetric(phone);
    assert.equal(await phone.evaluate(() => Math.max(0, document.documentElement.scrollWidth - innerWidth)), 0);
    assert.deepEqual(errors, []);
    console.log('PASS feature palette: exact fixed RGB mapping and clamping, equal values across maps/neuron scales, color restoration, persistence, 25/119/352/5 and legacy 50 inputs, true replay metrics and unknown hatching, unchanged model/forward/preset and 390px controls.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
