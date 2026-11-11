// Real-record browser acceptance: node llm/tests/geometry-replay.browser.cjs [URL] [single|double]
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const os=require('node:os');
const {pathToFileURL}=require('node:url');
let playwright;try{playwright=require('playwright');}catch(_){playwright=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));}
const url=process.argv[2]||pathToFileURL(path.resolve(__dirname,'../神经网络教学.html')).href;
const kind=process.argv[3]||'double';
(async()=>{
  const browser=await playwright.chromium.launch({headless:true,...(process.platform==='win32'?{executablePath:process.env.GEOMETRY_BROWSER||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'}:{})});
  try{
    const context=await browser.newContext({viewport:{width:1680,height:1050},acceptDownloads:true}),page=await context.newPage(),errors=[];
    if(process.env.GEOMETRY_OFFLINE==='1')await context.setOffline(true);
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto(url);
    await page.locator('#trainedVariantSelect').selectOption(`square-trained-geometry-${kind}-v1`);
    const manifest=await page.evaluate(k=>k==='double'?SquareGeometryDoubleTraceManifest:SquareGeometrySingleTraceManifest,kind);
    const ready=async step=>page.waitForFunction(s=>backpropReplay.active&&!backpropReplay.loading&&!!backpropReplay.model&&backpropReplay.context.step===s,step,{timeout:45000});
    const seek=async step=>{await page.locator('[data-geometry-step]').fill(String(step));await page.locator('[data-geometry-action="jump"]').click();await ready(step);};
    await page.locator('[data-geometry-mode="replay"]').click();await ready(0);
    assert.ok(await page.locator('.neuron-grid').nth(1).evaluate(n=>Number(n.style.getPropertyValue('--cols')))>=12);
    const initial=await page.evaluate(()=>backpropReplay.parameterInfo(backpropReplay.parameter));
    assert.ok(Math.abs(initial.before)<=.1);assert.equal(initial.gradient,null);
    assert.match(await page.locator('.trace-section').innerText(),/尚未反向传播/);
    for(const step of [...new Set([1,100,101,Math.min(4000,manifest.steps),manifest.steps])]){
      await seek(step);
      const info=await page.evaluate(()=>({context:backpropReplay.context,p:backpropReplay.parameterInfo(backpropReplay.parameter),sum:getSelectedWeightDetails().weightedInputSum,bias:getSelectedWeightDetails().bias,z:getSelectedWeightDetails().preActivation}));
      assert.equal(info.context.step,step);assert.ok(Number.isFinite(info.p.totalGradient));assert.ok(Math.abs(info.sum+info.bias-info.z)<1e-10);
    }
    // Tiny final-stage derivatives still occupy one readable line, including full precision.
    await page.locator('[data-replay-view="metric"]').selectOption('gradient');
    await page.locator('[data-replay-view="source"]').selectOption('total');
    await page.locator('[data-replay-view="scope"]').selectOption('batch');
    for(const width of [1680,390]){
      await page.setViewportSize({width,height:width===390?844:1050});
      for(const [layer,neuron,input] of [[1,60,6],...(kind==='double'?[[2,0,0]]:[])]){
        await page.evaluate(([l,j,i])=>backpropReplay.selectParameter(backpropReplay.parameterIndex(l,j,i)),[layer,neuron,input]);
        for(const full of [false,true]){
          await page.locator('[data-geometry-precision]').setChecked(full);
          const values=await page.locator('.geometry-metric-cell b').evaluateAll(nodes=>nodes.map(n=>{const r=document.createRange();r.selectNodeContents(n);return {text:n.textContent,lines:r.getClientRects().length,width:r.getBoundingClientRect().width,box:n.clientWidth};}));
          for(const value of values){assert.equal(value.lines,1,`Metric must not split ${value.text}`);assert.ok(value.width<=value.box,`Complete metric must fit ${value.text}`);}
          assert.equal(await page.evaluate(()=>Math.max(0,document.documentElement.scrollWidth-innerWidth)),0);
        }
      }
    }
    await page.locator('[data-geometry-precision]').setChecked(false);await page.setViewportSize({width:1680,height:1050});
    await page.locator('[data-geometry-phase]').selectOption('published');
    await page.locator('[data-replay-view="network"]').selectOption('observation');
    assert.equal(await page.evaluate(k=>JSON.stringify(currentDisplayModel().layers)===JSON.stringify((k==='double'?SquareGeometryDoubleModelData:SquareGeometrySingleModelData).model.layers),kind),true);
    await seek(101);
    await page.locator('[data-replay-view="network"]').selectOption('sample');
    const selectedLayer=kind==='double'?2:1,selectedNeuron=kind==='double'?0:60,selectedInput=kind==='double'?0:6;
    const index=await page.evaluate(([l,j,i])=>backpropReplay.parameterIndex(l,j,i),[selectedLayer,selectedNeuron,selectedInput]);
    await page.locator('[data-geometry-parameter-select]').selectOption(String(index));
    assert.equal(await page.locator('.input-comparison [data-inspector-pixel]').count(),kind==='double'?238:50);
    for(const metric of ['gradient','delta','weight']){
      await page.locator('[data-replay-view="metric"]').selectOption(metric);
      const display=await page.evaluate(()=>backpropReplay.metricValue(backpropReplay.parameter));assert.ok(Number.isFinite(display));
      const contrast=await page.locator('.geometry-metric-cell').evaluateAll(nodes=>nodes.map(node=>{
        const luminance=color=>{const rgb=color.match(/[\d.]+/g).slice(0,3).map(Number).map(x=>{const c=x/255;return c<=.04045?c/12.92:((c+.055)/1.055)**2.4;});return .2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2];};
        const css=getComputedStyle(node),a=luminance(css.color),b=luminance(css.backgroundColor);
        return (Math.max(a,b)+.05)/(Math.min(a,b)+.05);
      }));
      assert.ok(contrast.every(ratio=>ratio>=4.5),'Metric numbers require readable contrast on light and dark cells');
    }
    for(const source of ['classification','local',...(kind==='double'?['geometry']:[]),'boundary','participation','total']){
      await page.locator('[data-replay-view="source"]').selectOption(source);
      for(const scope of ['sample','batch']){
        await page.locator('[data-replay-view="scope"]').selectOption(scope);
        const value=await page.evaluate(()=>backpropReplay.parameterInfo(backpropReplay.parameter));assert.ok(Number.isFinite(value.gradient));
      }
    }
    await page.locator('[data-replay-view="source"]').selectOption('classification');await page.locator('[data-replay-view="scope"]').selectOption('sample');
    await page.locator('[data-geometry-section="loss"]').click();
    assert.match(await page.locator('.trace-section').innerText(),/损失的完整数值计算/);
    assert.ok((await page.locator('.trace-section .trace-equation').first().innerText()).split('=').length>=4);
    await page.locator('[data-geometry-action="audit"]').click();assert.match(await page.locator('.trace-audit').first().innerText(),/∂L/);
    await page.locator('[data-geometry-section="backward"]').click();assert.match(await page.locator('.trace-section').innerText(),/链式法则/);
    const biasIndex=await page.evaluate(l=>backpropReplay.parameterIndex(l,0,null),kind==='double'?3:2);
    await page.locator('[data-geometry-parameter-select]').selectOption(String(biasIndex));
    assert.match(await page.locator('.trace-section h3').innerText(),/偏置/);
    await page.locator('[data-geometry-section="update"]').click();assert.match(await page.locator('.trace-section').innerText(),/Adam 候选值/);
    // A structural zero has no trainable parameter and must not masquerade as a zero derivative.
    await page.locator('.neuron[data-layer="1"][data-neuron="0"]').click();
    await page.locator('[data-geometry-input="1"]').click();
    assert.match(await page.locator('#weightInspector').innerText(),/固定为零/);
    assert.equal(await page.evaluate(()=>backpropReplay.parameterInfo(-1).gradient),null);
    await page.locator('[data-geometry-parameter-select]').selectOption(String(index));
    const g=await page.evaluate(()=>backpropReplay.parameterInfo(backpropReplay.parameter).totalGradient);
    await page.locator('[data-replay-view="network"]').selectOption('observation');
    await page.locator('[data-geometry-action="canvas"]').click();await page.locator('#pixelGrid [data-index="0"]').click();
    assert.equal(await page.evaluate(()=>backpropReplay.parameterInfo(backpropReplay.parameter).totalGradient),g);
    for(const type of ['json','csv']){
      const pending=page.waitForEvent('download');await page.locator(`[data-geometry-action="${type}"]`).click();const file=await pending;
      const text=await fs.readFile(await file.path(),'utf8');
      if(type==='json'){const data=JSON.parse(text);assert.equal(data.step,101);assert.equal(data.traceId,manifest.id);assert.equal(data.before.length,manifest.parameterMap.length);assert.deepEqual(data.topology,manifest.topology);assert.deepEqual(data.activations,manifest.activations);assert.ok(data.sourceRecord.chunk.sha256);assert.ok(data.trainingTargets.features.length);}
      else {assert.match(text,/Adam候选值/);assert.equal(text.trim().split(/\r?\n/).length,manifest.parameterMap.length+1);}
    }
    await page.locator('[data-replay-view="layout"]').selectOption('vertical');await page.locator('[data-network-zoom="in"]').click();
    assert.equal(await page.locator('#networkCanvas').getAttribute('data-replay-layout'),'vertical');
    await page.locator('[data-network-zoom="fit"]').click();
    await page.screenshot({path:path.join(os.tmpdir(),`geometry-${kind}-replay-desktop.png`)});
    // Rapid requests must resolve only the last chosen update.
    await page.evaluate(()=>{backpropReplay.seekUpdate(301);backpropReplay.seekUpdate(201);backpropReplay.seekUpdate(101);});await ready(101);
    assert.equal(await page.locator('[data-geometry-step]').inputValue(),'101');
    // Corrupt an uncached checksum in this page only, then retry with the real checksum.
    const badIndex=manifest.chunks.findIndex(c=>c.firstStep===701);
    if(badIndex>=0){
      await page.evaluate(([k,i])=>{const m=k==='double'?SquareGeometryDoubleTraceManifest:SquareGeometrySingleTraceManifest;m.chunks[i].sha256='0'.repeat(64);},[kind,badIndex]);
      await page.locator('[data-geometry-step]').fill('701');await page.locator('[data-geometry-action="jump"]').click();
      await page.locator('.trace-error').waitFor({timeout:45000});assert.match(await page.locator('.trace-error').innerText(),/校验失败/);
      assert.equal(await page.locator('#networkScroll').isVisible(),false);
      await page.evaluate(([k,i,hash])=>{const m=k==='double'?SquareGeometryDoubleTraceManifest:SquareGeometrySingleTraceManifest;m.chunks[i].sha256=hash;},[kind,badIndex,manifest.chunks[badIndex].sha256]);
      await page.locator('[data-geometry-action="retry"]').click();await ready(701);
    }
    const missingIndex=manifest.chunks.findIndex(c=>c.firstStep===901);
    if(missingIndex>=0){
      await page.evaluate(([k,i])=>{const m=k==='double'?SquareGeometryDoubleTraceManifest:SquareGeometrySingleTraceManifest;m.chunks[i].file='missing-acceptance-test.js';},[kind,missingIndex]);
      await page.locator('[data-geometry-step]').fill('901');await page.locator('[data-geometry-action="jump"]').click();
      await page.locator('.trace-error').waitFor({timeout:45000});assert.match(await page.locator('.trace-error').innerText(),/加载失败/);
      await page.evaluate(([k,i,file])=>{const m=k==='double'?SquareGeometryDoubleTraceManifest:SquareGeometrySingleTraceManifest;m.chunks[i].file=file;},[kind,missingIndex,manifest.chunks[missingIndex].file]);
      await page.locator('[data-geometry-action="retry"]').click();await ready(901);
    }
    await seek(101);
    await page.locator('#trainedVariantSelect').selectOption('square-trained-partial-50-v5');
    await page.locator('#trainedVariantSelect').selectOption(`square-trained-geometry-${kind}-v1`);
    await page.locator('[data-geometry-mode="replay"]').click();await ready(101);
    await page.reload();await page.locator('[data-geometry-mode="replay"]').click();await ready(101);
    for(const legacy of [{id:'square-trained-partial-50-v5',steps:[0,6001,18000]},{id:'square-trained-partial-50-v4',steps:[12000]}]){
      await page.locator('#trainedVariantSelect').selectOption(legacy.id);
      await page.locator('[data-trace-mode="replay"]').click();
      await page.waitForFunction(()=>backpropReplay.active&&!backpropReplay.loading&&!!backpropReplay.model,null,{timeout:45000});
      for(const step of legacy.steps){
        await page.locator('#trace-step').fill(String(step));await page.locator('[data-trace-action="jump"]').click();
        await ready(step);
        assert.equal(await page.evaluate(()=>currentDisplayModel().topology[1]),50);
        if(step)assert.ok(await page.evaluate(()=>Number.isFinite(backpropReplay.parameterInfo(backpropReplay.parameter).totalGradient)));
      }
      await page.locator('[data-replay-view="network"]').selectOption('observation');
      assert.equal(await page.evaluate(id=>JSON.stringify(currentDisplayModel().layers)===JSON.stringify((id.endsWith('v5')?SquareRandomPartialModelData:SquareShuffledPartialModelData).model.layers),legacy.id),true);
    }
    await page.locator('#trainedVariantSelect').selectOption(`square-trained-geometry-${kind}-v1`);
    await page.locator('[data-geometry-mode="replay"]').click();await ready(101);
    await page.setViewportSize({width:390,height:844});await page.waitForTimeout(100);
    assert.equal(await page.evaluate(()=>Math.max(0,document.documentElement.scrollWidth-innerWidth)),0);
    await page.screenshot({path:path.join(os.tmpdir(),`geometry-${kind}-replay-mobile.png`)});
    assert.deepEqual(errors,[]);
    console.log(`PASS ${kind} real replay: initialization, updates/block boundaries/final parameters, all sources/scopes, H2 and biases, zero-mask semantics, loss/chain/Adam, canvas isolation, JSON/CSV, zoom/layout, rapid requests, checksum/missing-file retries, refresh, old v5/v4 real replay and 390px layout.`);
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
