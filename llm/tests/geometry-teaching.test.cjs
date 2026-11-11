// node llm/tests/geometry-teaching.test.cjs — public presets, examples and complete comparisons.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const {createSquareGeometryTeaching}=require('../geometry-teaching.js');
const root=path.resolve(__dirname,'..');
const html=fs.readFileSync(path.join(root,'神经网络教学.html'),'utf8');
const scripts=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]);
scripts.forEach(s=>new vm.Script(s));
const app=scripts.at(-1);
const coreContext=vm.createContext({console});
vm.runInContext(fs.readFileSync(path.join(root,'backprop-core.js'),'utf8'),coreContext);
const forward=(pixels,model,activations)=>vm.runInContext('SquareBackpropCore.forward',coreContext)(pixels,model,activations);
const comparisonContext=vm.createContext({state:{inspectorPixel:0},backpropReplay:null,escapeHtml:v=>String(v),colorFor:(v,t)=>`${t}-${v}`});
vm.runInContext(app.slice(app.indexOf('    function comparisonNumber('),app.indexOf('    function showInspectorPixel(')),comparisonContext);
for(const kind of ['single','double']){
  const artifact=JSON.parse(fs.readFileSync(path.join(root,`geometry-${kind}-model.json`),'utf8'));
  const definition=createSquareGeometryTeaching(artifact,`square-trained-geometry-${kind}-v1`,kind);
  const preset=definition.createPreset();
  assert.equal(definition.matchesPreset(preset),true);
  const altered=structuredClone(preset);altered.model.layers[0].weights[0][0]+=.000001;
  assert.equal(definition.matchesPreset(altered),false);
  const threshold=structuredClone(preset);threshold.outputs[0].threshold=0;
  assert.equal(definition.matchesPreset(threshold),false);
  assert.deepEqual(definition.createPreset().model,JSON.parse(JSON.stringify(artifact.model)));
  assert.equal(definition.features.length,kind==='single'?352:119);
  assert.equal(definition.examples.length,6);
  for(let example=0;example<6;example++){
    const pixels=definition.examplePixels(example),f=forward(pixels,preset.model,artifact.activations);
    assert.ok(f.outputs[0]>1,`${kind} complete example ${example}`);
    for(const variant of ['missing','extra'])assert.ok(forward(definition.examplePixels(example,variant),preset.model,artifact.activations).outputs[0]<=1);
    const state={...preset,pixels,compositionExample:example,geometryOpenGroups:[],selectedNeuron:{layerIndex:1,neuronIndex:60}};
    const view=definition.renderView(state,f);
    assert.match(view,/最终得分的完整求和/);
    assert.equal((view.match(/data-geometry-example=/g)||[]).length,6);
    for(let layer=1;layer<preset.model.topology.length;layer++){
      const j=0,weights=preset.model.layers[layer-1].weights[j];
      const inputWeights=weights.map((value,i)=>({index:i,label:`L${layer-1}-${i+1}`,value,inputValue:f.activations[layer-1][i],contribution:value*f.activations[layer-1][i]}));
      comparisonContext.details={layerIndex:layer,inputWeights};
      const rendered=vm.runInContext('renderContributionSection(details)',comparisonContext);
      assert.equal((rendered.match(/data-inspector-pixel=/g)||[]).length,2*weights.length);
      assert.match(rendered,/两侧共用色标/);
      assert.equal((rendered.match(/aria-pressed="true"/g)||[]).length,2);
      assert.ok(Math.abs(inputWeights.reduce((sum,t)=>sum+t.contribution,0)+preset.model.layers[layer-1].biases[j]-f.preActivations[layer-1][j])<1e-10);
    }
  }
  const f=forward(definition.examplePixels(0),preset.model,artifact.activations);
  const state={...preset,pixels:definition.examplePixels(0),compositionExample:0};
  const total=Array.from({length:kind==='single'?5:4},(_,i)=>(definition.renderGroup(state,f,i).match(/data-teaching-neuron=/g)||[]).length).reduce((a,b)=>a+b,0);
  assert.equal(total,definition.features.length);
}
console.log('PASS geometry presets, six examples and perturbations, all feature cards, 25/119/352/5 complete weight-contribution comparisons, synchronized selection and true forward sums');
