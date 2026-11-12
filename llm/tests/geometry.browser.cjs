// UI acceptance against the real bundled page. No generated model fixtures.
// node llm/tests/geometry.browser.cjs [file-or-http URL]
const assert=require('node:assert/strict');
const path=require('node:path');
const os=require('node:os');
const {pathToFileURL}=require('node:url');
let playwright;
try{playwright=require('playwright');}catch(_){playwright=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));}
const url=process.argv[2]||pathToFileURL(path.resolve(__dirname,'../神经网络教学.html')).href;
const screenshots=process.env.GEOMETRY_SCREENSHOTS||os.tmpdir();
const readableComparison=async page=>{
  const boxes=await page.locator('.input-comparison .contribution-value').evaluateAll(nodes=>nodes.map(n=>{
    const r=document.createRange();r.selectNodeContents(n);
    return {text:n.textContent,lines:r.getClientRects().length,width:r.getBoundingClientRect().width,cell:n.parentElement.clientWidth};
  }));
  for(const box of boxes){assert.equal(box.lines,1,`Number must remain on one line: ${box.text}`);assert.ok(box.width<=box.cell,`Complete number must fit its cell: ${box.text}`);}
  const grids=await page.locator('.input-comparison .contribution-grid').evaluateAll(nodes=>nodes.map(n=>({scroll:n.scrollWidth,width:n.clientWidth})));
  for(const grid of grids)assert.ok(grid.scroll<=grid.width+1,'Default six decimals must fit without horizontal scrolling');
};
(async()=>{
  const browser=await playwright.chromium.launch({headless:true,...(process.platform==='win32'?{executablePath:process.env.GEOMETRY_BROWSER||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'}:{})});
  try{
    const context=await browser.newContext({viewport:{width:1600,height:1050}});
    if(process.env.GEOMETRY_OFFLINE==='1')await context.setOffline(true);
    const page=await context.newPage(),errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto(url);await page.waitForTimeout(300);
    assert.match(await page.title(),/神经网络/);
    assert.equal(await page.locator('#trainedVariantSelect').inputValue(),'square-trained-learned-combination-v1');
    await page.locator('#trainedVariantSelect').selectOption('square-trained-geometry-double-v1');
    assert.equal(await page.locator('#pixelGrid button').count(),25);
    assert.equal(await page.getByRole('button',{name:'绘制 · 1',exact:true}).count(),0);
    const pixel=page.locator('#pixelGrid [data-index="0"]');
    assert.equal(await pixel.getAttribute('aria-pressed'),'false');
    await pixel.click();assert.equal(await pixel.getAttribute('aria-pressed'),'true');
    await pixel.click();assert.equal(await pixel.getAttribute('aria-pressed'),'false');
    await pixel.focus();await page.keyboard.press('Space');assert.equal(await pixel.getAttribute('aria-pressed'),'true');
    await page.keyboard.press('Enter');assert.equal(await pixel.getAttribute('aria-pressed'),'false');
    assert.equal(await page.locator('.input-comparison .contribution-cell').count(),50);
    // The details panel is now resizable; wide panels retain the paired layout.
    await page.locator('[data-resize-key="detail-width"]').focus();
    for(let i=0;i<6;i++)await page.keyboard.press('Shift+ArrowRight');
    await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    const columns=await page.locator('.input-comparison-columns > div').evaluateAll(nodes=>nodes.map(n=>({x:n.getBoundingClientRect().x,y:n.getBoundingClientRect().y})));
    assert.equal(columns[0].y,columns[1].y,'Desktop comparisons should be side by side');
    assert.ok(columns[1].x>columns[0].x);
    await readableComparison(page);
    await page.screenshot({path:path.join(screenshots,'geometry-desktop-comparison.png')});
    await page.locator('[data-teaching-layer="2"][data-teaching-neuron="0"]').first().click();
    assert.equal(await page.locator('.input-comparison-row').count(),238);
    for(const kind of ['single','double']){
      await page.locator('#trainedVariantSelect').selectOption(`square-trained-geometry-${kind}-v1`);
      const roundTrip=await page.evaluate(()=>{
        const before=buildPresetPayload();importPresetObject(JSON.parse(JSON.stringify(before)));
        const after=buildPresetPayload();
        return {same:JSON.stringify(before.preset)===JSON.stringify(after.preset),topology:after.preset.model.topology,mode:after.preset.teachingPreset};
      });
      assert.equal(roundTrip.same,true,`${kind} exact preset round trip`);
      assert.deepEqual(roundTrip.topology,kind==='single'?[25,352,1]:[25,119,5,1]);
      await page.reload();assert.equal(await page.locator('#trainedVariantSelect').inputValue(),`square-trained-geometry-${kind}-v1`);
      for(let i=0;i<6;i++){
        await page.locator(`[data-geometry-example="${i}"]`).click();
        await page.locator('[data-geometry-draw="missing"]').click();
        await page.locator('[data-geometry-draw="extra"]').click();
        await page.locator('[data-geometry-draw="complete"]').click();
      }
    }
    await page.locator('#trainedVariantSelect').selectOption('square-trained-geometry-single-v1');
    await page.locator('[data-geometry-group="4"] summary').click();
    await page.locator('[data-geometry-group="4"] [data-teaching-neuron]').first().waitFor();
    assert.equal(await page.locator('[data-geometry-group="4"] [data-teaching-neuron]').count(),233);
    await page.locator('#teachingViewButton').click();
    assert.equal(await page.locator('.neuron[data-layer="1"]').count(),352);
    await page.locator('.neuron[data-layer="2"]').click();assert.equal(await page.locator('.input-comparison-row').count(),704);
    await page.locator('#trainedVariantSelect').selectOption('square-trained-50-v1');
    assert.equal(await page.locator('.input-comparison .contribution-cell').count(),50);
    await page.locator('#trainedVariantSelect').selectOption('square-trained-geometry-double-v1');
    const mobile=await browser.newContext({viewport:{width:390,height:844},isMobile:true,hasTouch:true});
    if(process.env.GEOMETRY_OFFLINE==='1')await mobile.setOffline(true);
    const phone=await mobile.newPage();phone.on('pageerror',e=>errors.push(e.message));await phone.goto(url);
    const touch=phone.locator('#pixelGrid [data-index="0"]');
    await touch.tap();assert.equal(await touch.getAttribute('aria-pressed'),'true');await touch.tap();assert.equal(await touch.getAttribute('aria-pressed'),'false');
    await phone.locator('[data-teaching-layer="1"][data-teaching-neuron="60"]').first().click();
    assert.equal(await phone.locator('#detailSidebar').getAttribute('aria-modal'),'true');
    assert.equal(await phone.evaluate(()=>Math.max(0,document.documentElement.scrollWidth-innerWidth)),0);
    const phoneColumns=await phone.locator('.input-comparison-columns > div').evaluateAll(nodes=>nodes.map(n=>({x:n.getBoundingClientRect().x,y:n.getBoundingClientRect().y})));
    assert.equal(phoneColumns[0].x,phoneColumns[1].x);assert.ok(phoneColumns[1].y>phoneColumns[0].y);
    await readableComparison(phone);
    await phone.screenshot({path:path.join(screenshots,'geometry-mobile-comparison.png')});
    assert.deepEqual(errors,[]);
    console.log('PASS UI: default/new/old presets, mouse/touch/keyboard toggles, exact import/export and restore, six examples, 25/119/352 comparisons, full network, desktop/mobile layout.');
    console.log('Screenshots:',screenshots);
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
