// Real-page acceptance: node llm/tests/contribution-ranking.browser.cjs [file-or-http URL]
// GEOMETRY_OFFLINE=1 also supports the complete extracted offline package.
const assert = require('node:assert/strict');
const path = require('node:path');
const os = require('node:os');
const { pathToFileURL } = require('node:url');
let playwright;
try { playwright = require('playwright'); }
catch (_) { playwright = require(path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')); }
const url = process.argv[2] || pathToFileURL(path.resolve(__dirname, '../神经网络教学.html')).href;
const screenshots = process.env.GEOMETRY_SCREENSHOTS || os.tmpdir();
const near = (actual, expected, label) => assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= 1e-11 * Math.max(1, Math.abs(expected)), `${label}: ${actual} ≠ ${expected}`);
const rankedRows = page => page.locator('#contributionRanking .contribution-ranking-row');

async function openRanking(page) {
  await page.locator('#rankingTab').click();
  assert.equal(await page.locator('#rankingTab').getAttribute('aria-selected'), 'true');
  assert.equal(await page.locator('#contributionRanking').isVisible(), true);
}

async function checkRanking(page, count) {
  const snapshot = await page.evaluate(() => {
    const model = currentDisplayModel(), layer = model.layers.at(-1);
    const select = document.getElementById('rankingOutputSelect');
    const output = select ? Number(select.value) : 0;
    const activations = latestForward.activations.at(-2);
    const expected = activations.map((a, index) => ({ index, contribution: a * layer.weights[output][index] }));
    expected.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution) || a.index - b.index);
    const panel = document.getElementById('contributionRanking');
    const rows = [...panel.querySelectorAll('.contribution-ranking-row')].map(row => ({
      index: Number(row.dataset.rankingIndex), contribution: Number(row.dataset.rankingContribution),
      displayed: row.querySelector('[data-ranking-value]')?.textContent.trim()
    }));
    const metric = name => { const node = panel.querySelector(`[data-ranking-${name}]`); return node ? Number(node.getAttribute(`data-ranking-${name}`)) : NaN; };
    return { expected, rows, sum: expected.reduce((sum, row) => sum + row.contribution, 0), bias: layer.biases[output],
      z: latestForward.preActivations.at(-1)[output], outputValue: latestForward.outputs[output],
      footer: Object.fromEntries(['sum', 'bias', 'z', 'output'].map(name => [name, metric(name)])),
      hiddenLayer: model.topology.length - 2, outputLayer: model.topology.length - 1, output };
  });
  assert.equal(snapshot.rows.length, count ?? snapshot.expected.length);
  assert.deepEqual(snapshot.rows.map(row => row.index), snapshot.expected.map(row => row.index), 'Absolute contribution descending, stable neuron-number ties');
  for (let i = 0; i < snapshot.rows.length; i++) {
    const row = snapshot.rows[i];
    near(row.contribution, snapshot.expected[i].contribution, `Neuron ${row.index + 1} raw signed contribution`);
    assert.ok(row.displayed, `Neuron ${row.index + 1} shows its contribution`);
    const displayed = Number(row.displayed.replaceAll('−', '-').replaceAll('＋', '+').replaceAll(',', ''));
    assert.ok(Number.isFinite(displayed), `Contribution should be numeric: ${row.displayed}`);
    if (displayed !== 0) assert.equal(Math.sign(displayed), Math.sign(row.contribution), 'Visible contribution keeps its sign');
  }
  near(snapshot.footer.sum, snapshot.sum, 'Displayed contribution sum');
  near(snapshot.footer.bias, snapshot.bias, 'Displayed output bias');
  near(snapshot.footer.z, snapshot.z, 'Displayed output weighted sum');
  near(snapshot.footer.output, snapshot.outputValue, 'Displayed actual output');
  near(snapshot.sum + snapshot.bias, snapshot.z, 'Contribution sum + bias = actual forward z');
  return snapshot;
}

async function checkSelectedNeuron(page, layerIndex, neuronIndex) {
  assert.equal(await page.locator('#featuresTab').getAttribute('aria-selected'), 'true');
  assert.deepEqual(await page.evaluate(() => state.selectedNeuron), { layerIndex, neuronIndex });
}

async function checkConnection(page, snapshot, sourceIndex, replay = false) {
  await page.locator(`[data-ranking-connection="${sourceIndex}"]`).click();
  await checkSelectedNeuron(page, snapshot.outputLayer, snapshot.output);
  assert.equal(await page.evaluate(() => state.inspectorPixel), sourceIndex);
  if (replay) {
    assert.equal(await page.evaluate(([layer, neuron, input]) => backpropReplay.parameter === backpropReplay.parameterIndex(layer, neuron, input), [snapshot.outputLayer, snapshot.output, sourceIndex]), true);
  }
}

async function ready(page, step) {
  await page.waitForFunction(s => backpropReplay.active && !backpropReplay.loading && !!backpropReplay.model && backpropReplay.context.step === s, step, { timeout: 45000 });
}

async function checkRankingContext(page) {
  const context = await page.evaluate(() => backpropReplay.context);
  const text = await page.locator('#contributionRanking .ranking-context').innerText();
  if (context.step === 0) {
    assert.match(text, /初始化.*观察画布.*初始参数/);
    assert.doesNotMatch(text, /训练样本/, 'Initialization has no historical training sample');
  } else {
    assert.ok(text.includes(context.step.toLocaleString('en-US')), 'Context names the actual history update');
    if (context.networkSource === 'sample') {
      assert.ok(text.includes(`训练样本 ${context.sampleIndex + 1}`), 'Sample index is displayed as one based');
      assert.match(text, /更新前参数/, 'Training sample always uses pre-update parameters');
      assert.doesNotMatch(text, /观察画布|更新后参数/);
    } else {
      assert.match(text, /观察画布/);
      assert.doesNotMatch(text, /训练样本/);
      assert.ok(text.includes(context.phase === 'before' ? '更新前参数' : context.phase === 'after' ? '更新后参数' : '正式参数'));
    }
  }
}

async function checkNoPageOverflow(page) {
  assert.equal(await page.evaluate(() => Math.max(0, document.documentElement.scrollWidth - innerWidth)), 0);
  const numbers = await page.locator('#contributionRanking [data-ranking-value]').evaluateAll(nodes => nodes.map(node => {
    const range = document.createRange(); range.selectNodeContents(node);
    return { text: node.textContent, lines: range.getClientRects().length, width: range.getBoundingClientRect().width, box: node.clientWidth };
  }));
  for (const number of numbers) {
    assert.equal(number.lines, 1, `Contribution must remain on one line: ${number.text}`);
    assert.ok(number.width <= number.box + 1, `Complete contribution must fit: ${number.text}`);
  }
}

(async () => {
  const browser = await playwright.chromium.launch({ headless: true, ...(process.platform === 'win32' ? { executablePath: process.env.GEOMETRY_BROWSER || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe' } : {}) });
  try {
    const context = await browser.newContext({ viewport: { width: 1600, height: 1050 } });
    if (process.env.GEOMETRY_OFFLINE === '1') await context.setOffline(true);
    const page = await context.newPage(), errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(url);

    // All actual last-hidden-layer sizes, including an older trained network.
    for (const [id, count] of [['square-trained-geometry-single-v1', 352], ['square-trained-geometry-double-v1', 5], ['square-trained-partial-50-v5', 50]]) {
      await page.locator('#trainedVariantSelect').selectOption(id);
      await openRanking(page);
      const snapshot = await checkRanking(page, count), sourceIndex = snapshot.rows[0].index;
      await page.locator(`[data-ranking-neuron="${sourceIndex}"]`).click();
      await checkSelectedNeuron(page, snapshot.hiddenLayer, sourceIndex);
      await openRanking(page); await checkConnection(page, snapshot, sourceIndex);
      await openRanking(page);
      const previous = (await checkRanking(page, count)).rows.map(row => [row.index, row.contribution]);
      await page.locator('#pixelGrid [data-index="0"]').click();
      const next = (await checkRanking(page, count)).rows.map(row => [row.index, row.contribution]);
      assert.notDeepEqual(next, previous, 'Canvas editing recomputes the ranking');
    }

    // Three real tabs must work by keyboard, without trapping focus in a hidden panel.
    const tabs = page.locator('[data-inspector-tab]');
    assert.equal(await tabs.count(), 3);
    const tabIds = await tabs.evaluateAll(nodes => nodes.map(node => node.id));
    await page.locator(`#${tabIds[0]}`).focus();
    for (let n = 1; n <= 3; n++) {
      await page.keyboard.press('ArrowRight');
      assert.equal(await page.evaluate(() => document.activeElement.id), tabIds[n % 3]);
      assert.equal(await page.locator(`#${tabIds[n % 3]}`).getAttribute('aria-selected'), 'true');
    }
    await page.keyboard.press('End'); assert.equal(await page.evaluate(() => document.activeElement.id), tabIds.at(-1));
    await page.keyboard.press('Home'); assert.equal(await page.evaluate(() => document.activeElement.id), tabIds[0]);

    // Ordinary experiment supports more than one output; use its public UI controls.
    await page.locator('#normalModeButton').click();
    await page.locator('#settingsTab').click();
    const outputCount = page.locator('#outputCountInput');
    await outputCount.evaluate(input => { input.value = '2'; input.dispatchEvent(new Event('change', { bubbles: true })); });
    await openRanking(page);
    assert.equal(await page.locator('#rankingOutputSelect option').count(), 2);
    await page.locator('#rankingOutputSelect').selectOption('1');
    const multi = await checkRanking(page);
    assert.equal(multi.output, 1);
    await checkConnection(page, multi, multi.rows[0].index);

    // With no hidden layer the sources are pixels, and their contribution is to z,
    // while the sigmoid output y is reported separately.
    await page.locator('#settingsTab').click();
    await page.locator('#hiddenCountInput').evaluate(input => { input.value = '0'; input.dispatchEvent(new Event('change', { bubbles: true })); });
    await page.locator('#outputActivationSelect').evaluate(select => { select.value = 'sigmoid'; select.dispatchEvent(new Event('change', { bubbles: true })); });
    await openRanking(page); await page.locator('#rankingOutputSelect').selectOption('1');
    const direct = await checkRanking(page, 25);
    assert.equal(direct.hiddenLayer, 0); assert.equal(direct.outputLayer, 1);
    assert.match(await page.locator('#contributionRanking .ranking-head h3').innerText(), /输入贡献排行/);
    assert.match(await page.locator('#contributionRanking .ranking-summary').innerText(), /输出加权和 z/);
    assert.match(await page.locator('#contributionRanking [data-ranking-output]').innerText(), /sigmoid\(/);
    near(direct.outputValue, 1 / (1 + Math.exp(-direct.z)), 'Sigmoid(z) is the distinct actual output y');
    await page.locator(`[data-ranking-neuron="${direct.rows[0].index}"]`).click();
    await checkSelectedNeuron(page, 1, 1);
    assert.equal(await page.evaluate(() => state.inspectorPixel), direct.rows[0].index, 'Pixel source selects its direct output connection');

    await page.locator('#decimalTeachingModeButton').click();
    await page.locator('#trainedVariantSelect').selectOption('square-trained-geometry-single-v1');
    await openRanking(page); await page.locator('#rankingPrecision').check();
    await checkRanking(page, 352); await checkNoPageOverflow(page);
    await page.reload();
    assert.equal(await page.locator('#rankingTab').getAttribute('aria-selected'), 'true');
    assert.equal(await page.locator('#rankingPrecision').isChecked(), true);
    await checkRanking(page, 352); await page.locator('#rankingPrecision').uncheck();

    for (const [kind, count] of [['single', 352], ['double', 5]]) {
      await page.locator('#trainedVariantSelect').selectOption(`square-trained-geometry-${kind}-v1`);
      await page.locator('[data-geometry-mode="replay"]').click();
      await ready(page, 0); await openRanking(page); await checkRanking(page, count); await checkRankingContext(page);
      const steps = await page.evaluate(k => (k === 'single' ? SquareGeometrySingleTraceManifest : SquareGeometryDoubleTraceManifest).steps, kind);
      for (const step of [100, Math.min(6001, steps)]) {
        await page.locator('[data-geometry-step]').fill(String(step));
        await page.locator('[data-geometry-action="jump"]').click();
        await ready(page, step); await checkRanking(page, count); await checkRankingContext(page);
      }
      await page.locator('[data-replay-view="network"]').selectOption('observation');
      for (const phase of ['before', 'after']) {
        await page.locator('[data-geometry-phase]').selectOption(phase); await checkRanking(page, count); await checkRankingContext(page);
      }
      const observation = (await checkRanking(page, count)).rows;
      await page.locator('[data-replay-view="source"]').selectOption('total');
      assert.deepEqual((await checkRanking(page, count)).rows, observation, 'Changing gradient source does not change forward contributions');
      await page.locator('[data-replay-view="network"]').selectOption('sample');
      await page.locator('[data-geometry-sample]').selectOption('1');
      await page.waitForFunction(() => !backpropReplay.loading && backpropReplay.context.sampleIndex === 1, null, { timeout: 45000 });
      const snapshot = await checkRanking(page, count);
      await checkRankingContext(page);
      await checkConnection(page, snapshot, snapshot.rows[0].index, true); await openRanking(page);

      // The synchronous loading render must remove old rows, before new chunk I/O completes.
      const during = await page.evaluate(() => {
        backpropReplay.seekUpdate(701);
        return { loading: backpropReplay.loading, rows: document.querySelectorAll('#contributionRanking .contribution-ranking-row').length,
          message: !!document.querySelector('#contributionRanking [data-ranking-loading]') };
      });
      assert.equal(during.loading, true); assert.equal(during.rows, 0); assert.equal(during.message, true);
      await page.evaluate(() => { backpropReplay.seekUpdate(801); backpropReplay.seekUpdate(101); });
      await ready(page, 101); await checkRanking(page, count);

      if (kind === 'double') {
        // Uncached load failure must never expose the previous step or formal model.
        // HTTP uses request interception; file:// uses a nonexistent local chunk,
        // since browser file loaders need not participate in network interception.
        const manifest = await page.evaluate(() => SquareGeometryDoubleTraceManifest);
        const index = manifest.chunks.findIndex(chunk => chunk.firstStep === 901);
        assert.ok(index >= 0, 'An uncached 901 chunk exists');
        const original = manifest.chunks[index].file;
        const chunkUrl = new URL(manifest.dataPath + original, page.url()).href;
        const intercept = /^https?:/.test(chunkUrl);
        if (intercept) await page.route(chunkUrl, route => route.abort('failed'));
        else await page.evaluate(i => { SquareGeometryDoubleTraceManifest.chunks[i].file = 'missing-ranking-acceptance-test.js'; }, index);
        try {
          await page.locator('[data-geometry-step]').fill('901'); await page.locator('[data-geometry-action="jump"]').click();
          await page.locator('.trace-error').waitFor({ timeout: 45000 });
          assert.match(await page.locator('.trace-error').innerText(), /加载失败/);
          assert.equal(await rankedRows(page).count(), 0, 'Failure clears all previously ranked values');
          assert.equal(await page.locator('#contributionRanking [data-ranking-loading]').isVisible(), true);
        } finally {
          if (intercept) await page.unroute(chunkUrl);
          else await page.evaluate(([i, file]) => { SquareGeometryDoubleTraceManifest.chunks[i].file = file; }, [index, original]);
        }
        await page.locator('[data-geometry-action="retry"]').click(); await ready(page, 901);
        await checkRanking(page, count); await checkRankingContext(page);
      }
    }

    // Ranking stays available in the two-column teaching fullscreen layout.
    await page.locator('[data-geometry-action="fullscreen"]').click();
    await page.waitForFunction(() => !!document.fullscreenElement);
    await openRanking(page); await checkRanking(page, 5);
    assert.equal(await page.locator('.replay-left').isVisible(), true);
    assert.equal(await page.locator('.replay-right').isVisible(), true);
    await page.screenshot({ path: path.join(screenshots, 'contribution-ranking-desktop.png') });
    await page.locator('[data-geometry-action="fullscreen"]').click();
    await page.waitForFunction(() => !document.fullscreenElement);

    // Old replay shares the inspector, but retains its original history engine.
    await page.locator('#trainedVariantSelect').selectOption('square-trained-partial-50-v5');
    await page.locator('[data-trace-mode="replay"]').click();
    await page.locator('#trace-step').fill('6001'); await page.locator('[data-trace-action="jump"]').click();
    await ready(page, 6001); await openRanking(page);
    const legacy = await checkRanking(page, 50), legacySource = legacy.rows[0].index;
    await checkRankingContext(page);
    await checkConnection(page, legacy, legacySource);
    assert.equal(await page.evaluate(() => backpropReplay.parameter), 1300 + legacySource, 'Legacy output connection uses its actual parameter index');

    const mobile = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
    if (process.env.GEOMETRY_OFFLINE === '1') await mobile.setOffline(true);
    const phone = await mobile.newPage(); phone.on('pageerror', error => errors.push(error.message)); await phone.goto(url);
    await phone.locator('#trainedVariantSelect').selectOption('square-trained-geometry-single-v1');
    await phone.locator('#openRankingButton').tap();
    assert.equal(await phone.locator('#detailSidebar').getAttribute('aria-modal'), 'true');
    await checkRanking(phone, 352); await checkNoPageOverflow(phone);
    await phone.locator('#rankingPrecision').check(); await checkRanking(phone, 352); await checkNoPageOverflow(phone);
    await phone.screenshot({ path: path.join(screenshots, 'contribution-ranking-mobile.png') });
    await phone.keyboard.press('Escape');
    assert.equal(await phone.locator('#openRankingButton').getAttribute('aria-expanded'), 'false');
    await phone.locator('#trainedVariantSelect').selectOption('square-trained-geometry-double-v1');
    await phone.locator('[data-geometry-mode="replay"]').click(); await ready(phone, 0);
    await openRanking(phone); await checkRanking(phone, 5); await checkNoPageOverflow(phone);
    assert.deepEqual(errors, []);
    console.log('PASS contribution ranking: 352/5/50 real contributions and signed absolute order, sums and bias, exact new/legacy connections, canvas changes, multiple outputs and zero-hidden sigmoid z/y, precision and refresh, keyboard, history context, loading/failure/retry isolation, fullscreen and 390px layout.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
