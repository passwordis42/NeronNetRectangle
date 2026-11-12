// node llm/tests/network-contribution-sort.browser.cjs [file-or-http URL]
// Browser plugin not available; exercise the real bundled page with Playwright.
const assert = require('node:assert/strict');
const path = require('node:path');
const os = require('node:os');
const { pathToFileURL } = require('node:url');
let playwright;
try { playwright = require('playwright'); }
catch (_) { playwright = require(path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')); }
const url = process.argv[2] || pathToFileURL(path.resolve(__dirname, '../神经网络教学.html')).href;
const button = page => page.locator('#sortNetworkContributionButton');
const settle = page => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
const numeric = page => page.evaluate(() => JSON.stringify({
  model: cloneModelForPreset(currentDisplayModel()), pixels: state.pixels, forward: latestForward,
  replay: backpropReplay.active ? { context: backpropReplay.context, parameter: backpropReplay.parameter,
    info: backpropReplay.parameterInfo(backpropReplay.parameter) } : null
}));

async function fullNetwork(page) {
  if (!await page.locator('#networkScroll').isVisible()) await page.locator('#teachingViewButton').click();
  await settle(page);
  assert.equal(await page.locator('#networkScroll').isVisible(), true);
}

async function checkOrder(page, sorted) {
  await settle(page);
  const result = await page.evaluate(() => {
    const model = currentDisplayModel(), hidden = model.topology.length - 2;
    const output = Math.min(state.rankingOutput, model.topology.at(-1) - 1);
    const layers = model.topology.map((size, layer) => ({ size, layer,
      nodes: [...document.querySelectorAll(`#layers .neuron[data-layer="${layer}"]`)].map(node => ({
        index: Number(node.dataset.neuron), label: node.getAttribute('aria-label'), text: node.textContent.trim(), identity: node.querySelector('small')?.textContent.trim()
      }))
    }));
    const values = hidden > 0 ? latestForward.activations[hidden].map((value, index) => ({
      index, contribution: value * model.layers.at(-1).weights[output][index]
    })) : [];
    const expected = values.slice().sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution) || a.index - b.index);
    const ranking = [...document.querySelectorAll('#contributionRanking .contribution-ranking-row')].map(row => Number(row.dataset.rankingIndex));
    return { hidden, output, layers, expected, values, ranking, sort: state.networkContributionSort,
      replay: backpropReplay.active, buttonPressed: document.getElementById('sortNetworkContributionButton').getAttribute('aria-pressed') };
  });
  for (const layer of result.layers) {
    const indices = Array.from({ length: layer.size }, (_, i) => i);
    assert.equal(layer.nodes.length, layer.size, `Layer ${layer.layer} retains every real neuron`);
    assert.deepEqual(layer.nodes.map(node => node.index).slice().sort((a, b) => a - b), indices, 'No duplicate or renumbered neurons');
    const expected = sorted && layer.layer === result.hidden && result.hidden > 0 ? result.expected.map(row => row.index) : indices;
    assert.deepEqual(layer.nodes.map(node => node.index), expected, `Layer ${layer.layer} visual order`);
    if (layer.layer > 0) for (const node of layer.nodes) {
      assert.match(node.label, new RegExp(`神经元 ${node.index + 1} 权重详情`), 'Accessible neuron identity stays original');
      if (result.replay) assert.ok(node.text.startsWith(`${layer.layer === result.layers.length - 1 ? 'Y' : `H${layer.layer}-`}${node.index + 1}`));
      else if (sorted && layer.layer === result.hidden) assert.equal(node.identity, `H${layer.layer}-${node.index + 1}`, 'Sorted forward nodes visibly show their original hidden-neuron number');
    }
  }
  if (result.hidden > 0) {
    assert.deepEqual(result.ranking, result.expected.map(row => row.index), 'Network sorting and output contribution ranking use the same current forward pass');
    assert.equal(result.buttonPressed, String(sorted));
  }
  return result;
}

async function setSort(page, enabled) {
  const before = await numeric(page);
  if ((await button(page).getAttribute('aria-pressed') === 'true') !== enabled) await button(page).click();
  await settle(page);
  assert.equal(await page.evaluate(() => state.networkContributionSort), enabled);
  assert.equal(await numeric(page), before, 'Sorting cannot modify parameters, predictions, history position or gradients');
  return checkOrder(page, enabled);
}

async function checkConnections(page) {
  await settle(page);
  const lines = await page.evaluate(() => {
    const model = currentDisplayModel(), horizontal = backpropReplay.active ? backpropReplay.layout !== 'vertical' : state.expandedMode;
    const svg = document.getElementById('connectionSvg'), matrix = svg.getScreenCTM();
    return [...svg.querySelectorAll('line[data-from][data-to]')].map(line => {
      const [fromLayer, fromIndex] = line.dataset.from.split(':').map(Number);
      const [toLayer, toIndex] = line.dataset.to.split(':').map(Number);
      const source = document.querySelector(`#layers .neuron[data-layer="${fromLayer}"][data-neuron="${fromIndex}"]`).getBoundingClientRect();
      const target = document.querySelector(`#layers .neuron[data-layer="${toLayer}"][data-neuron="${toIndex}"]`).getBoundingClientRect();
      const screenPoint = (x, y) => { const point = svg.createSVGPoint(); point.x = x; point.y = y; const screen = point.matrixTransform(matrix); return [screen.x, screen.y]; };
      return { fromLayer, fromIndex, toLayer, toIndex, weight: Number(line.dataset.weight), expectedWeight: model.layers[toLayer - 1].weights[toIndex][fromIndex],
        context: `${model.topology.join('→')} / ${backpropReplay.active ? backpropReplay.layout : 'forward'} / SVG ${svg.getBoundingClientRect().width}×${svg.getBoundingClientRect().height}, viewBox ${svg.getAttribute('viewBox')}`,
        start: screenPoint(line.x1.baseVal.value, line.y1.baseVal.value), end: screenPoint(line.x2.baseVal.value, line.y2.baseVal.value),
        expectedStart: horizontal ? [source.right, source.top + source.height / 2] : [source.left + source.width / 2, source.bottom],
        expectedEnd: horizontal ? [target.left, target.top + target.height / 2] : [target.left + target.width / 2, target.top] };
    });
  });
  assert.ok(lines.length > 0, 'Selected path has actual SVG connections');
  for (const line of lines) {
    assert.equal(line.weight, line.expectedWeight, `Connection ${line.fromLayer}:${line.fromIndex} → ${line.toLayer}:${line.toIndex} keeps its real weight`);
    for (const [actual, expected] of [[line.start, line.expectedStart], [line.end, line.expectedEnd]]) {
      assert.ok(Math.abs(actual[0] - expected[0]) <= 1 && Math.abs(actual[1] - expected[1]) <= 1,
        `Connection endpoint must follow its real neuron after DOM reorder: ${actual} vs ${expected}; ${line.fromLayer}:${line.fromIndex} → ${line.toLayer}:${line.toIndex}; ${line.context}`);
    }
  }
}

async function selectSortedSource(page, snapshot) {
  const index = snapshot.expected[0].index;
  await page.locator(`#layers .neuron[data-layer="${snapshot.hidden}"][data-neuron="${index}"]`).dispatchEvent('click');
  assert.deepEqual(await page.evaluate(() => state.selectedNeuron), { layerIndex: snapshot.hidden, neuronIndex: index });
  assert.deepEqual(await page.evaluate(() => { const d = getSelectedWeightDetails(); return [d.layerIndex, d.neuronIndex]; }), [snapshot.hidden, index]);
  await checkConnections(page);
  if (await page.evaluate(() => backpropReplay.active)) {
    const selector = `#connectionSvg line[data-from="${snapshot.hidden}:${index}"][data-to="${snapshot.hidden + 1}:${snapshot.output}"]`;
    assert.equal(await page.locator(selector).count(), 1);
    await page.locator(selector).dispatchEvent('click');
    const actual = await page.evaluate(() => backpropReplay.parameterInfo(backpropReplay.parameter));
    assert.equal(actual.layer, snapshot.hidden + 1);
    assert.equal(actual.neuron, snapshot.output);
    assert.equal(actual.input, index, 'Clicking a reordered output connection selects its original source parameter');
  }
}

async function ready(page, step) {
  await page.waitForFunction(s => backpropReplay.active && !backpropReplay.loading && !!backpropReplay.model && backpropReplay.context.step === s, step, { timeout: 45000 });
  await settle(page);
}

async function noOverflow(page) {
  assert.equal(await page.evaluate(() => Math.max(0, document.documentElement.scrollWidth - innerWidth)), 0, 'Sorting must not create page horizontal overflow');
}

(async () => {
  const browser = await playwright.chromium.launch({ headless: true, ...(process.platform === 'win32' ? { executablePath: process.env.GEOMETRY_BROWSER || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe' } : {}) });
  try {
    const context = await browser.newContext({ viewport: { width: 1600, height: 1050 } });
    if (process.env.GEOMETRY_OFFLINE === '1') await context.setOffline(true);
    const page = await context.newPage(), errors = [];
    page.on('pageerror', error => errors.push(error.message)); await page.goto(url);
    assert.match(await page.title(), /神经/);
    assert.equal(await page.evaluate(() => state.networkContributionSort), false, 'New visits keep original neuron order');
    assert.equal(await page.locator('#teachingView').isVisible(), true, 'The initial teaching view is the focused feature explanation');
    const initial = await setSort(page, true);
    assert.equal(await page.locator('#networkScroll').isVisible(), true, 'Sorting directly from focused teaching opens the full network');
    const highest = page.locator(`#layers .neuron[data-layer="${initial.hidden}"][data-neuron="${initial.expected[0].index}"]`);
    await highest.scrollIntoViewIfNeeded(); await settle(page);
    assert.equal(await highest.isVisible(), true);
    await page.screenshot({ path: path.join(process.env.GEOMETRY_SCREENSHOTS || os.tmpdir(), 'network-contribution-sort-desktop.png') });
    await setSort(page, false); await checkOrder(page, false);

    // All formal last-hidden-layer sizes; the earlier H1 in the two-hidden-layer model stays numbered.
    for (const [id, count] of [['square-trained-geometry-double-v1', 5], ['square-trained-geometry-single-v1', 352], ['square-trained-partial-50-v5', 50]]) {
      await page.locator('#trainedVariantSelect').selectOption(id); await fullNetwork(page);
      await setSort(page, false);
      const sorted = await setSort(page, true); assert.equal(sorted.expected.length, count);
      await selectSortedSource(page, sorted);
      const weights = await page.evaluate(() => JSON.stringify(state.model)), previous = sorted.values;
      await page.locator('#pixelGrid [data-index="0"]').click();
      const edited = await checkOrder(page, true);
      assert.notDeepEqual(edited.values, previous, 'Canvas editing refreshes contributions for visual ordering');
      assert.equal(await page.evaluate(() => JSON.stringify(state.model)), weights);
      await checkConnections(page); await setSort(page, false);
    }

    // An exact tie fixture proves absolute signed ordering, stable identities and multi-output changes.
    await page.locator('#normalModeButton').click();
    await page.evaluate(() => {
      state.hiddenLayers = [{ size: 4, activation: 'tanh' }];
      state.outputs = [{ name: 'First', threshold: 1, target: 2 }, { name: 'Second', threshold: 1, target: 2 }];
      state.outputActivation = 'linear'; state.weightSettings = { min: -10, max: 10, precision: 6 };
      state.pixels = Array(25).fill(0); state.rankingOutput = 0; state.networkContributionSort = false;
      state.model = { topology: [25, 4, 2], layers: [
        { weights: Array.from({ length: 4 }, (_, n) => Array.from({ length: 25 }, (_, i) => n === 0 && i === 0 ? 2 : 0)), biases: Array(4).fill(Math.atanh(.5)) },
        { weights: [[1, -4, 4, .5], [6, 0, -1, 2]], biases: [.1, -.2] }
      ] };
      state.selectedNeuron = { layerIndex: 1, neuronIndex: 0 }; state.inspectorTab = 'features'; renderAll();
    });
    const firstOutput = await setSort(page, true);
    assert.deepEqual(firstOutput.expected.map(row => row.index), [1, 2, 0, 3]);
    assert.ok(firstOutput.expected[0].contribution < 0 && firstOutput.expected[1].contribution > 0, 'Negative and positive absolute ties retain real index order');
    const beforeOutput = await numeric(page);
    await page.locator('#rankingOutputSelect').selectOption('1');
    assert.equal(await numeric(page), beforeOutput, 'Choosing a ranking output never retrains the model');
    assert.deepEqual((await checkOrder(page, true)).expected.map(row => row.index), [0, 3, 2, 1]);
    await selectSortedSource(page, await checkOrder(page, true));
    await page.reload(); await fullNetwork(page); await checkOrder(page, true);
    assert.equal(await page.locator('#rankingOutputSelect').inputValue(), '1');
    await setSort(page, false); await page.reload(); await fullNetwork(page); await checkOrder(page, false);

    // There are no hidden nodes to reorder in a direct input → output network.
    await page.locator('#settingsTab').click();
    await page.locator('#hiddenCountInput').evaluate(input => { input.value = '0'; input.dispatchEvent(new Event('change', { bubbles: true })); });
    assert.equal(await button(page).isDisabled(), true);
    await checkOrder(page, false);

    await page.locator('#decimalTeachingModeButton').click();
    for (const [kind, count] of [['double', 5], ['single', 352]]) {
      await page.locator('#trainedVariantSelect').selectOption(`square-trained-geometry-${kind}-v1`);
      await page.locator('[data-geometry-mode="replay"]').click(); await ready(page, 0);
      await setSort(page, true); assert.equal((await checkOrder(page, true)).expected.length, count);
      await page.locator('[data-geometry-step]').fill('100'); await page.locator('[data-geometry-action="jump"]').click(); await ready(page, 100);
      for (const layout of ['compact', 'vertical', 'expanded']) {
        await page.locator('[data-replay-view="layout"]').selectOption(layout);
        const sorted = await checkOrder(page, true); await selectSortedSource(page, sorted);
        await noOverflow(page);
      }
      await page.locator('[data-replay-view="layout"]').selectOption('compact');
      await page.locator('[data-replay-view="network"]').selectOption('observation');
      for (const phase of ['before', 'after']) {
        await page.locator('[data-geometry-phase]').selectOption(phase); await checkOrder(page, true);
      }
      await page.locator('[data-replay-view="network"]').selectOption('sample');
      await page.locator('[data-geometry-sample]').selectOption('1');
      await page.waitForFunction(() => !backpropReplay.loading && backpropReplay.context.sampleIndex === 1, null, { timeout: 45000 });
      await checkOrder(page, true); await setSort(page, false);
    }

    await page.locator('#trainedVariantSelect').selectOption('square-trained-partial-50-v5');
    await page.locator('[data-trace-mode="replay"]').click(); await ready(page, 0); await setSort(page, true);
    await page.locator('#trace-step').fill('6001'); await page.locator('[data-trace-action="jump"]').click(); await ready(page, 6001);
    for (const layout of ['compact', 'vertical', 'expanded']) {
      await page.locator('[data-replay-view="layout"]').selectOption(layout);
      const sorted = await checkOrder(page, true); assert.equal(sorted.expected.length, 50); await selectSortedSource(page, sorted);
    }
    await page.locator('[data-replay-view="layout"]').selectOption('compact');
    await page.setViewportSize({ width: 390, height: 844 }); await settle(page);
    await checkOrder(page, true); await checkConnections(page); await noOverflow(page);
    await page.locator('[data-trace-mode="forward"]').click(); await fullNetwork(page);
    const phoneSort = await checkOrder(page, true); await noOverflow(page);
    await page.locator(`#layers .neuron[data-layer="${phoneSort.hidden}"][data-neuron="${phoneSort.expected[0].index}"]`).scrollIntoViewIfNeeded();
    await settle(page);
    await page.screenshot({ path: path.join(process.env.GEOMETRY_SCREENSHOTS || os.tmpdir(), 'network-contribution-sort-mobile.png') });
    assert.deepEqual(errors, []);
    console.log('PASS network contribution sort: opt-in, actual 5/352/50 nodes and stable signed ties, unchanged earlier layers/identities/weights/forward, ranking/output/canvas synchronization, restore and refresh, zero-hidden disabled, new/legacy history, actual SVG endpoints, three replay layouts and 390px.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
