const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const core=require('../backprop-core.js'),root=path.resolve(__dirname,'..');
const map=vm.runInNewContext(fs.readFileSync(path.join(root,'backprop-annotations.js'),'utf8')+';SquareReplayAnnotations');
const model=JSON.parse(fs.readFileSync(path.join(root,'partial-features-random-model.json'))).model;
const pixels=core.pixels(469440),forward=core.forward(pixels,model),flat=core.flatten(model);
const value=(html,key)=>{const m=html.match(new RegExp(`data-map-value="${key}" data-value="([^"]*)"`));return m?m[1]===''?null:Number(m[1]):undefined;};
// Every selectable parameter must map to its actual trained position. In particular,
// output biases have no "default" hidden neuron, and gradients must not replace weights.
for(let parameter=0;parameter<core.P;parameter++){
 const description=core.describe(parameter);
 const d={parameter,description,model,pixels,forward,step:18000,networkPhase:'before',networkSource:'sample',sampleIndex:13,batchSize:640,
  losses:[1,2,3,4],parameterInfo:{gradient:-.123,source:'total',sourceName:'本次更新 · 总梯度',totalGradient:-.123,delta:.002}};
 const html=map.render(d,String);
 assert.doesNotMatch(html,/undefined|NaN/);
 assert.equal(value(html,'y'),forward.outputs[0]);assert.equal(value(html,'gradient'),-.123);assert.equal(value(html,'loss'),10);
 if(description.layer===1){
  const j=description.neuron;
  assert.equal(value(html,'h'),forward.activations[1][j]);assert.equal(value(html,'z'),forward.preActivations[0][j]);
  assert.equal(value(html,'b1'),flat[1250+j]);assert.equal(value(html,'w2'),flat[1300+j]);
  if(description.kind==='weight'){assert.equal(value(html,'w1'),flat[parameter]);assert.equal(value(html,'x'),pixels[description.input]);}
  else {assert.equal(value(html,'b'),flat[parameter]);assert.equal(value(html,'x'),undefined);}
 }else if(description.kind==='weight'){
  assert.equal(value(html,'h'),forward.activations[1][description.input]);assert.equal(value(html,'w2'),flat[parameter]);
  assert.equal(value(html,'w1'),undefined);assert.equal(value(html,'x'),undefined);
 }else{
  assert.equal(value(html,'b'),flat[1350]);assert.equal(value(html,'h'),undefined);assert.equal(value(html,'z'),undefined);
 }
 assert.doesNotMatch(html,/class="map-warning"/);
 assert.match(map.render({...d,networkPhase:'after',networkSource:'observation'},String),/class="map-warning"/);
 const initial=map.render({...d,step:0,losses:null,networkPhase:'initial',parameterInfo:{...d.parameterInfo,gradient:null}},v=>v===null?'—':String(v));
 assert.equal(value(initial,'gradient'),null);assert.match(initial,/尚未反向传播/);assert.equal(value(initial,'loss'),undefined);
}
console.log('PASS symbol map: all 1,351 parameter positions, true full-network forward values, bias paths, gradient/weight distinction, observation timing warning, and initialization without invented gradients.');
