const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs=require('node:fs');
const {pathToFileURL}=require('node:url');
const path=require('node:path');
const root=path.resolve(__dirname,'../..').replaceAll('\\','/');
const dir=process.env.SQUARE_QA_OUTPUT || require('node:os').tmpdir();
const core=require(root+'/llm/backprop-core.js');
const url=process.argv[2]||pathToFileURL(root+'/llm/神经网络教学.html').href;
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
 const context=await browser.newContext({viewport:{width:1920,height:1080},acceptDownloads:true});
 const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
 async function ready(step){await page.waitForFunction(s=>document.querySelector('#trace-step')?.value===String(s)&&!!document.querySelector('[data-trace-update]'),step,{timeout:45000});}
 async function seek(step){await page.locator('#trace-step').fill(String(step));await page.locator('[data-trace-action="jump"]').click();await ready(step);}
 async function json(){const [d]=await Promise.all([page.waitForEvent('download'),page.locator('[data-trace-action="json"]').click()]);return JSON.parse(fs.readFileSync(await d.path(),'utf8'));}
 async function snap(name){await page.screenshot({path:dir+'/'+name+'.png',fullPage:false});}
 await page.goto(url);await page.locator('#trainedVariantSelect').waitFor();
 assert.equal(await page.locator('#trainedVariantSelect').inputValue(),'square-trained-partial-50-v5');
 await page.locator('[data-trace-mode="replay"]').click();await ready(0);
 const initial=await json();assert.ok(initial.before.every((x,i)=>i<1250||i>=1300&&i<1350?Math.abs(x)<=.1:x===0));
 assert.equal(initial.display.parameter.gradient,null);assert.equal(await page.locator('.neuron').count(),76);
 await page.locator('[data-replay-view="metric"]').selectOption('weight');
 const initialText=await page.locator('[data-replay-parameter="506"] .contribution-value').textContent();
 assert.equal(initialText,initial.before[506].toFixed(6));
 for(const step of [1,100,101,500,1000,2000,4000,6000,6001,8000,10000,12000,12001,18000]){
   await seek(step);const d=await json();
   assert.equal(d.step,step);assert.ok(d.gradientError<1e-8);
   assert.equal(d.display.source,step<=6000?'concept':'classification');
   assert.equal(d.optimizer.learningRate,step<=6000?.01:.003);
   assert.ok(Math.abs(d.after[506]-d.before[506]-d.display.parameter.delta)<1e-14);
   if(step===1)assert.notEqual(d.display.parameter.gradient,0);
 }
 const end=await json();const artifact=JSON.parse(fs.readFileSync(root+'/llm/partial-features-random-model.json','utf8'));
 assert.deepEqual(end.after.map(v=>+v.toFixed(6)),Array.from(core.flatten(artifact.model)));
 assert.ok(Math.abs(end.after[506]-initial.before[506])>2.8);
 await seek(6001);await page.locator('[data-replay-view="metric"]').selectOption('gradient');
 await page.locator('[data-replay-parameter="507"]').click();let d=await json();assert.equal(d.selectedParameter.index,507);
 assert.equal(await page.locator('[data-replay-parameter="507"] .contribution-value').textContent(),d.display.parameter.gradient.toFixed(6));
 await page.locator('[data-replay-view="source"]').selectOption('total');d=await json();assert.equal(d.display.parameter.gradient,d.gradient[507]);
 await page.locator('[data-replay-view="metric"]').selectOption('delta');d=await json();assert.equal(d.display.parameter.delta,d.after[507]-d.before[507]);
 await page.locator('[data-replay-parameter="1270"]').click();d=await json();assert.equal(d.selectedParameter.index,1270);
 await page.locator('#trace-neuron').selectOption('50');assert.equal(await page.locator('.replay-parameter-grid [data-replay-parameter]').count(),50);
 await page.locator('[data-replay-parameter="1320"]').click();d=await json();assert.equal(d.selectedParameter.index,1320);
 await page.locator('#trace-neuron').selectOption('20');await page.locator('[data-replay-view="metric"]').selectOption('gradient');await page.locator('[data-replay-view="source"]').selectOption('auto');
 await page.locator('#trace-sample').selectOption('130');d=await json();assert.equal(d.details.sample.index,130);
 await page.locator('[data-trace-action="find"]').click();await ready(6001);d=await json();assert.ok(d.found>=0);assert.notEqual(d.display.parameter.gradient,0);
 await page.locator('#trace-sample').selectOption('0');await ready(6001);
 // Screen layout and graph interaction; all layers, navigation and detail fit on desktop.
 await page.evaluate(()=>document.querySelector('#backpropView').scrollTop=0);
 let boxes=await page.evaluate(()=>{let out={};for(const [k,s] of Object.entries({left:'.trace-sticky',network:'#networkScroll',right:'#weightInspector'})){let r=document.querySelector(s).getBoundingClientRect();out[k]={x:r.x,y:r.y,width:r.width,height:r.height,bottom:r.bottom};}return out;});
 assert.ok(boxes.network.x>boxes.left.x+boxes.left.width);assert.ok(boxes.right.bottom<=1080);assert.ok(boxes.network.bottom<boxes.right.bottom);
 await snap('random-replay-desktop');
 await page.locator('[data-trace-fullscreen]').click();await page.waitForFunction(()=>Boolean(document.fullscreenElement));await snap('random-replay-fullscreen');await page.evaluate(()=>document.exitFullscreen());
 await page.locator('[data-replay-view="zoom"]').selectOption('1.25');assert.match(await page.locator('#networkCanvas').getAttribute('style'),/1.25/);await page.locator('[data-replay-view="zoom"]').selectOption('1');
 // Synthetic click targets an actual SVG connection, then verifies parameter synchronization.
 await page.locator('line[data-from="0:8"][data-to="1:20"]').dispatchEvent('click');assert.equal((await json()).selectedParameter.index,508);
 await page.locator('[data-trace-action="canvas"]').click();let before=await json();await page.locator('#clearCanvasButton').click();let after=await json();assert.deepEqual(after.gradient,before.gradient);assert.equal(after.observation.pixels.reduce((a,b)=>a+b,0),0);
 await page.locator('[data-replay-view="network"]').selectOption('observation');assert.equal(await page.locator('.neuron[data-layer="0"]').first().textContent(),'X10.00');
 await page.locator('[data-replay-view="network"]').selectOption('sample');
 await page.locator('[data-trace-action="csv"]').click();
 await page.locator('#trace-sample').selectOption('130');await ready(6001);await page.reload();assert.equal(await page.locator('#trainedVariantSelect').inputValue(),'square-trained-partial-50-v5');await page.locator('[data-trace-mode="replay"]').click();await ready(6001);assert.equal((await json()).details.sample.index,130);
 await page.setViewportSize({width:1366,height:768});await page.evaluate(()=>document.querySelector('#backpropView').scrollTop=0);await snap('random-replay-laptop');
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth && document.querySelector('.replay-left').getBoundingClientRect().width<=innerWidth));
 await page.setViewportSize({width:390,height:844});await page.evaluate(()=>window.scrollTo(0,0));await snap('random-replay-mobile');assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth && document.querySelector('.replay-left').getBoundingClientRect().width<=innerWidth));
 await page.locator('[data-trace-mode="forward"]').click();assert.equal(await page.locator('#replayWorkbench').count(),0);assert.ok(await page.locator('#pixelGrid').isVisible());
 await page.setViewportSize({width:1920,height:1080});
 for(let i=0;i<6;i++){
  await page.locator(`[data-composition-example="${i}"]`).click();assert.ok(await page.evaluate(()=>latestForward.outputs[0]>1));
  await page.locator('[data-composition-draw="missing"]').click();assert.ok(await page.evaluate(()=>latestForward.outputs[0]<1));
  await page.locator('[data-composition-draw="extra"]').click();assert.ok(await page.evaluate(()=>latestForward.outputs[0]<1));
 }
 await page.locator('[data-composition-draw="complete"]').click();
 await page.locator('#settingsTab').click();const config=page.locator('[data-panel-key="config"] .collapsible-trigger');if(await config.getAttribute('aria-expanded')==='false')await config.click();
 const [presetDownload]=await Promise.all([page.waitForEvent('download'),page.locator('#downloadPresetButton').click()]);const presetFile=await presetDownload.path();const preset=JSON.parse(fs.readFileSync(presetFile,'utf8'));assert.equal(preset.preset.teachingPreset,'square-trained-partial-50-v5');assert.equal(preset.preset.provenance.training.initialization.inheritedParameters,false);
 await page.locator('#normalModeButton').click();await page.locator('#importPresetInput').setInputFiles(presetFile);await page.waitForFunction(()=>state.teachingPreset==='square-trained-partial-50-v5');
 assert.deepEqual(Array.from(core.flatten(await page.evaluate(()=>state.model))),Array.from(core.flatten(artifact.model)));await page.reload();assert.equal(await page.locator('#trainedVariantSelect').inputValue(),'square-trained-partial-50-v5');
 await page.locator('#trainedVariantSelect').selectOption('square-trained-partial-50-v4');await page.locator('[data-trace-mode="replay"]').click();await ready(0);await seek(12000);assert.equal((await json()).traceId,'partial-50-v4-backprop-v1');
 await page.locator('[data-trace-mode="forward"]').click();await page.reload();assert.equal(await page.locator('#trainedVariantSelect').inputValue(),'square-trained-partial-50-v4');
 assert.deepEqual(errors,[]);console.log('PASS random replay: stages, gradients, delta, bias, 50 outputs, sample search, graph, canvas isolation, fullscreen, zoom, refresh, mobile, old replay.');
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
