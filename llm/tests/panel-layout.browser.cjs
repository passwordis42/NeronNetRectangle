// node llm/tests/panel-layout.browser.cjs [file-or-http URL]
const assert = require('node:assert/strict');
const path = require('node:path');
const os = require('node:os');
const { pathToFileURL } = require('node:url');
let playwright;
try { playwright = require('playwright'); }
catch (_) { playwright = require(path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')); }
const url = process.argv[2] || pathToFileURL(path.resolve(__dirname, '../神经网络教学.html')).href;
const screenshots = process.env.GEOMETRY_SCREENSHOTS || os.tmpdir();
const storageKey = 'nn-teaching-panel-layout-v1';
const axes = {
  'canvas-width': ['x', 1], 'detail-width': ['x', -1], 'canvas-height': ['y', 1],
  'output-height': ['y', 1], 'network-height': ['y', 1], 'detail-height': ['y', 1],
  'contribution-width': ['x', -1], 'stack-network-height': ['y', 1],
  'replay-left-width': ['x', 1], 'replay-top-height': ['y', 1],
  'replay-network-height': ['y', 1], 'replay-network-height-mobile': ['y', 1], 'drawer-width': ['x', -1]
};
const handle = (page, key) => page.locator(`[data-resize-key="${key}"]:visible`);
const close = (a, b, label) => assert.ok(Math.abs(a - b) <= 1, `${label}: ${a} ≠ ${b}`);
const waitLayout = page => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
const value = async (page, key) => Number(await handle(page, key).getAttribute('aria-valuenow'));
const numericState = page => page.evaluate(() => JSON.stringify({ pixels: state.pixels, model: currentDisplayModel(), forward: latestForward }));

async function closeContributionPane(page, replay = false) {
  const before = await workspace(page, replay), numbers = await numericState(page);
  const button = page.locator('#hideContributionsButton');
  const top = await button.evaluate(node => node.getBoundingClientRect().top);
  await page.locator('#contributionRanking').evaluate(node => { node.scrollTop = node.scrollHeight; });
  close(await button.evaluate(node => node.getBoundingClientRect().top), top, 'Hide button stays outside scrolling numbers');
  await button.focus(); await page.keyboard.press('Enter'); await waitLayout(page);
  assert.equal(await page.locator('#networkContributionPane').isVisible(), false);
  assert.equal(await page.locator('#contributionSizeHandle').isVisible(), false);
  assert.equal(await page.locator('#toggleContributionsButton').getAttribute('aria-expanded'), 'false');
  assert.equal(await page.locator('#toggleContributionsButton').innerText(), '显示贡献');
  assert.equal(await page.evaluate(() => document.activeElement.id), 'toggleContributionsButton');
  const after = await workspace(page, replay);
  if (before.stacked) assert.ok(after.primary.height > before.primary.height + 20, 'Hiding contributions releases vertical space');
  else assert.ok(after.primary.width > before.primary.width + 20, 'Hiding contributions releases horizontal space');
  assert.equal(await numericState(page), numbers, 'Visibility cannot change parameters or forward results');
  await page.keyboard.press('Space'); await waitLayout(page);
  assert.equal(await button.isVisible(), true);
  assert.equal(await page.locator('#toggleContributionsButton').getAttribute('aria-expanded'), 'true');
  await workspace(page, replay);
}

async function noOverflow(page) {
  assert.equal(await page.evaluate(() => Math.max(0, document.documentElement.scrollWidth - innerWidth)), 0, 'Resizing must not create page-wide horizontal overflow');
}

async function workspace(page, replay = false) {
  assert.equal(await page.locator('#networkWorkspace').count(), 1);
  assert.equal(await page.locator('#networkPrimary').count(), 1);
  assert.equal(await page.locator('#networkContributionPane #contributionRanking').count(), 1);
  assert.equal(await page.locator('#contributionRanking').count(), 1);
  assert.equal(await page.locator('#contributionRanking').getAttribute('role'), 'region');
  assert.equal(await page.locator('.replay-right #networkWorkspace').count(), replay ? 1 : 0);
  if (replay) assert.equal(await page.locator('#networkPrimary .replay-network-frame #networkScroll').count(), 1);
  const data = await page.evaluate(() => {
    const box = node => { const r = node.getBoundingClientRect(); return { x: r.x, y: r.y, width: r.width, height: r.height, right: r.right, bottom: r.bottom }; };
    const root = document.getElementById('networkWorkspace');
    return { root: box(root), primary: box(document.getElementById('networkPrimary')), pane: box(document.getElementById('networkContributionPane')),
      stacked: root.classList.contains('is-stacked'), shown: state.showContributions };
  });
  if (data.shown) {
    assert.equal(data.stacked, data.root.width < 620, 'Workspace layout follows its actual width');
    if (data.stacked) assert.ok(data.pane.y >= data.primary.bottom - 1, 'Narrow contribution pane follows the network vertically');
    else assert.ok(data.pane.x >= data.primary.right - 1, 'Wide contribution pane is to the right of the network');
  }
  return data;
}

async function drag(page, key, increase = 48) {
  const target = handle(page, key), [axis, direction] = axes[key];
  await target.scrollIntoViewIfNeeded();
  const box = await target.boundingBox(), viewport = page.viewportSize();
  assert.ok(box && box.width > 0 && box.height > 0, `${key} is a visible draggable boundary`);
  const x = Math.max(2, Math.min(viewport.width - 2, box.x + box.width / 2));
  const y = Math.max(2, Math.min(viewport.height - 2, box.y + box.height / 2));
  await page.mouse.move(x, y); await page.mouse.down();
  await page.mouse.move(x + (axis === 'x' ? increase * direction : 0), y + (axis === 'y' ? increase : 0), { steps: 5 });
  await page.mouse.up(); await waitLayout(page);
}

async function exerciseHandle(page, key) {
  const target = handle(page, key), [axis] = axes[key];
  assert.equal(await target.count(), 1, `${key} has exactly one visible handle`);
  assert.equal(await target.getAttribute('role'), 'separator');
  assert.equal(await target.getAttribute('tabindex'), '0');
  assert.equal(await target.getAttribute('aria-orientation'), axis === 'x' ? 'vertical' : 'horizontal');
  const before = await numericState(page);
  // The same key can have a saved size from another responsive layout. Establish
  // this layout's current default rather than treating that saved size as default.
  await target.dblclick(); await waitLayout(page);
  const initial = await value(page, key);
  const min = Number(await target.getAttribute('aria-valuemin')), max = Number(await target.getAttribute('aria-valuemax'));
  assert.ok(Number.isFinite(min) && Number.isFinite(max) && max >= min);
  assert.ok(initial >= min - 1 && initial <= max + 1, `${key} starts within its accessible limits`);
  const plus = axis === 'x' ? 'ArrowRight' : 'ArrowDown', minus = axis === 'x' ? 'ArrowLeft' : 'ArrowUp';
  await target.focus(); await page.keyboard.press('Home'); await waitLayout(page);
  close(await value(page, key), min, `${key} Home reaches minimum`);
  await page.keyboard.press(plus); await waitLayout(page);
  close(await value(page, key), Math.min(max, min + 10), `${key} arrow increment`);
  await page.keyboard.press(`Shift+${plus}`); await waitLayout(page);
  close(await value(page, key), Math.min(max, min + 60), `${key} Shift increment`);
  await page.keyboard.press('Home'); await page.keyboard.press(minus); await waitLayout(page);
  close(await value(page, key), min, `${key} does not cross minimum`);
  if (max - min > 8) {
    const delta = Math.min(40, (max - min) / 2);
    await drag(page, key, delta);
    const dragged = await value(page, key);
    assert.ok(dragged > min + 2 && dragged <= max + 1, `${key} pointer drag actually changes its size`);
  }
  await target.focus(); await page.keyboard.press('End'); await waitLayout(page);
  const currentMax = Number(await target.getAttribute('aria-valuemax'));
  close(await value(page, key), currentMax, `${key} End reaches maximum`);
  await page.keyboard.press(plus); await waitLayout(page);
  assert.ok(await value(page, key) <= Number(await target.getAttribute('aria-valuemax')) + 1, `${key} does not cross maximum`);
  await target.dblclick(); await waitLayout(page);
  close(await value(page, key), initial, `${key} double click restores its default`);
  assert.equal(await numericState(page), before, `${key} only changes presentation, never pixels or parameters`);
  await noOverflow(page);
}

async function fittedNetwork(page) {
  await waitLayout(page); await waitLayout(page);
  const geometry = await page.evaluate(() => {
    const scroll = document.getElementById('networkScroll').getBoundingClientRect();
    const canvas = document.getElementById('networkCanvas').getBoundingClientRect();
    const horizontal = backpropReplay.layout !== 'vertical';
    const errors = [...document.querySelectorAll('#connectionSvg line[data-from][data-to]')].slice(0, 30).flatMap(line => {
      const endpoints = [['from', 'x1', 'y1'], ['to', 'x2', 'y2']];
      return endpoints.map(([side, x, y]) => {
        const [layer, neuron] = line.dataset[side].split(':');
        const rect = document.querySelector(`.neuron[data-layer="${layer}"][data-neuron="${neuron}"]`).getBoundingClientRect();
        const point = new DOMPoint(Number(line.getAttribute(x)), Number(line.getAttribute(y))).matrixTransform(line.getScreenCTM());
        const expectedX = horizontal ? (side === 'from' ? rect.right : rect.left) : rect.left + rect.width / 2;
        const expectedY = horizontal ? rect.top + rect.height / 2 : (side === 'from' ? rect.bottom : rect.top);
        return Math.hypot(point.x - expectedX, point.y - expectedY);
      });
    });
    return { fits: canvas.left >= scroll.left - 1 && canvas.top >= scroll.top - 1 && canvas.right <= scroll.right + 1 && canvas.bottom <= scroll.bottom + 1, errors };
  });
  assert.equal(geometry.fits, true, 'Fit mode keeps the complete network inside the resized viewport');
  assert.ok(geometry.errors.length > 0, 'Visible real connections are available for alignment checks');
  assert.ok(geometry.errors.every(error => error < 1.5), 'Connection endpoints stay aligned with neuron edges after resize');
}

async function cancelledTouchResize(page) {
  const key = 'canvas-height', target = handle(page, key);
  await target.scrollIntoViewIfNeeded();
  const rect = await target.boundingBox(), before = await numericState(page), original = await value(page, key);
  const x = rect.x + rect.width / 2, y = rect.y + rect.height / 2;
  const cdp = await page.context().newCDPSession(page);
  try {
    await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: true, maxTouchPoints: 1 });
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x, y }] });
    assert.equal(await page.evaluate(() => document.body.classList.contains('is-resizing-panels')), true);
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x, y: y + 30 }] });
    await waitLayout(page);
    assert.notEqual(await value(page, key), original, 'Touch movement reaches the resize handler');
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchCancel', touchPoints: [] });
    await waitLayout(page);
    close(await value(page, key), original, 'Pointer cancel rolls back the incomplete resize');
    assert.equal(await numericState(page), before, 'Touch resize/cancel cannot paint pixels or change model values');
    assert.equal(await page.evaluate(() => document.body.classList.contains('is-resizing-panels')), false);
  } finally {
    await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: false }); await cdp.detach();
  }
}

(async () => {
  const browser = await playwright.chromium.launch({ headless: true, ...(process.platform === 'win32' ? { executablePath: process.env.GEOMETRY_BROWSER || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe' } : {}) });
  try {
    const context = await browser.newContext({ viewport: { width: 1920, height: 1080 } });
    if (process.env.GEOMETRY_OFFLINE === '1') await context.setOffline(true);
    const page = await context.newPage(), errors = [];
    page.on('pageerror', error => errors.push(error.message)); await page.goto(url);
    assert.equal(await page.locator('#toggleContributionsButton').getAttribute('aria-expanded'), 'true');
    assert.equal(await page.locator('#rankingTab').count(), 0);
    assert.equal(await page.locator('[data-inspector-tab]').count(), 2);
    await workspace(page); await noOverflow(page);
    for (const key of ['contribution-width', 'canvas-width', 'detail-width', 'canvas-height', 'output-height', 'network-height', 'detail-height']) await exerciseHandle(page, key);

    await closeContributionPane(page);
    await page.locator('#hideContributionsButton').click(); await page.reload(); await waitLayout(page);
    assert.equal(await page.locator('#networkContributionPane').isVisible(), false, 'Header hide preference survives refresh');
    await page.locator('#toggleContributionsButton').click(); await waitLayout(page); await workspace(page);

    // Persist a non-default size across refresh; global reset clears it and shows contributions.
    const defaultWidth = await value(page, 'canvas-width');
    await handle(page, 'canvas-width').focus(); await page.keyboard.press('Shift+ArrowRight'); await waitLayout(page);
    const savedWidth = await value(page, 'canvas-width');
    assert.ok(savedWidth > defaultWidth);
    assert.ok(await page.evaluate(key => !!localStorage.getItem(key), storageKey));
    await page.reload(); await waitLayout(page);
    close(await value(page, 'canvas-width'), savedWidth, 'Panel width survives refresh');
    await page.locator('#toggleContributionsButton').click();
    await page.locator('#resetLayoutButton').click(); await waitLayout(page);
    close(await value(page, 'canvas-width'), defaultWidth, 'Global reset restores width');
    assert.equal(await page.locator('#toggleContributionsButton').getAttribute('aria-expanded'), 'true');

    // Migrate only the old inspector selection; numeric/model state must survive unchanged.
    const beforeMigration = await numericState(page);
    await page.evaluate(() => {
      saveState(); const key = teachingStorageKey(state.teachingPreset), saved = JSON.parse(localStorage.getItem(key));
      saved.inspectorTab = 'ranking'; delete saved.showContributions; localStorage.setItem(key, JSON.stringify(saved));
    });
    await page.reload(); await waitLayout(page);
    assert.equal(await page.locator('#featuresTab').getAttribute('aria-selected'), 'true');
    assert.equal(await page.locator('#toggleContributionsButton').getAttribute('aria-expanded'), 'true');
    assert.equal(await numericState(page), beforeMigration);

    // Medium-width layout uses a drawer; changing viewport never duplicates the ranking.
    await page.setViewportSize({ width: 1000, height: 900 }); await waitLayout(page);
    await workspace(page); await noOverflow(page);
    assert.equal(await handle(page, 'detail-width').count(), 0);
    await page.locator('#openFeaturesButton').click();
    await exerciseHandle(page, 'drawer-width');
    await page.locator('#closeInspectorButton').click();

    // Replay uses the same workspace and preserves the controls in fullscreen.
    await page.setViewportSize({ width: 1920, height: 1080 }); await waitLayout(page);
    await page.locator('[data-geometry-mode="replay"]').click();
    await page.waitForFunction(() => backpropReplay.active && !backpropReplay.loading && !!backpropReplay.model, null, { timeout: 45000 });
    await workspace(page, true);
    for (const key of ['replay-left-width', 'replay-network-height', 'contribution-width']) await exerciseHandle(page, key);
    await page.locator('[data-replay-view="layout"]').selectOption('compact');
    await page.locator('[data-network-zoom="fit"]').click(); await fittedNetwork(page);
    await drag(page, 'replay-left-width', 40); await fittedNetwork(page);
    await handle(page, 'replay-left-width').dblclick(); await waitLayout(page);
    await page.locator('[data-geometry-action="fullscreen"]').click(); await page.waitForFunction(() => !!document.fullscreenElement);
    await workspace(page, true); await noOverflow(page);
    await closeContributionPane(page, true);
    const panes = await page.locator('.replay-left,.replay-right').evaluateAll(nodes => nodes.map(node => { const r = node.getBoundingClientRect(); return { x: r.x, y: r.y, right: r.right, bottom: r.bottom }; }));
    assert.equal(panes.length, 2); assert.ok(panes[1].x > panes[0].x);
    assert.ok(panes.every(pane => pane.x >= 0 && pane.y >= 0 && pane.right <= 1921 && pane.bottom <= 1081));
    await page.screenshot({ path: path.join(screenshots, 'panel-layout-fullscreen.png') });
    await page.locator('[data-geometry-action="fullscreen"]').click(); await page.waitForFunction(() => !document.fullscreenElement);
    for (let cycle = 0; cycle < 3; cycle++) {
      await page.locator('[data-geometry-mode="forward"]').click(); await workspace(page, false);
      assert.equal(await page.locator('.replay-network-surface,.replay-zoom-controls').count(), 0, 'Exiting replay removes its wrappers and controls');
      await page.locator('[data-geometry-mode="replay"]').click();
      await page.waitForFunction(() => backpropReplay.active && !backpropReplay.loading && !!backpropReplay.model, null, { timeout: 45000 });
      await workspace(page, true);
      for (const selector of ['#replayWorkbench', '.replay-network-frame', '.replay-network-surface', '.replay-zoom-controls', '[data-network-zoom="fit"]']) assert.equal(await page.locator(selector).count(), 1, `${selector} is not duplicated after replay cycle ${cycle + 1}`);
    }
    await page.locator('[data-geometry-mode="forward"]').click(); await workspace(page, false);

    // Phone: stacked network/contributions, vertical boundaries, and a bounded detail drawer.
    await page.setViewportSize({ width: 390, height: 844 }); await waitLayout(page);
    assert.equal((await workspace(page)).stacked, true);
    await closeContributionPane(page);
    for (const key of ['stack-network-height', 'canvas-height', 'output-height', 'network-height']) await exerciseHandle(page, key);
    await cancelledTouchResize(page);
    await page.locator('#openFeaturesButton').click(); await exerciseHandle(page, 'drawer-width');
    await page.locator('#closeInspectorButton').click();
    await noOverflow(page);
    await page.locator('[data-geometry-mode="replay"]').click();
    await page.waitForFunction(() => backpropReplay.active && !backpropReplay.loading && !!backpropReplay.model, null, { timeout: 45000 });
    await workspace(page, true);
    for (const key of ['replay-top-height', 'stack-network-height', 'replay-network-height-mobile']) await exerciseHandle(page, key);
    await closeContributionPane(page, true);
    await page.screenshot({ path: path.join(screenshots, 'panel-layout-mobile.png') });
    await page.locator('#resetLayoutButton').click(); await waitLayout(page); await noOverflow(page);
    assert.equal(await page.locator('#toggleContributionsButton').getAttribute('aria-expanded'), 'true');
    assert.deepEqual(errors, []);
    console.log('PASS panel layout: accessible drag/keyboard boundaries, limits, double-click/global reset, persistence and old ranking migration, numeric invariance, contribution visibility, responsive workspace, replay/fullscreen and 390px drawer.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
