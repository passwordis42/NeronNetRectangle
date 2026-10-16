// No dependencies: node llm/tests/partial-features.test.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, '神经网络教学.html'), 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
scripts.forEach((s, i) => new vm.Script(s, { filename: `inline-${i}.js` }));
const context = vm.createContext({ console });
for (const script of scripts.slice(0, -1)) vm.runInContext(script, context);
const app = scripts.at(-1);
// Exercise the exact forward() and activation functions used by the page.
const start = app.indexOf('    function activate(');
const end = app.indexOf('    function trainSample(');
vm.runInContext(`const STEP_THRESHOLD = 1; let state; function activationTypesOf(s) {
  return [...s.hiddenLayers.map(l => l.activation), s.outputActivation];
}\n${app.slice(start, end)}`, context);
vm.runInContext(`
  const definition = SquarePartialFeaturesTeaching;
  const preset = definition.createPreset();
  state = { ...preset, pixels: definition.examplePixels(0), compositionExample: 0,
            selectedNeuron: { layerIndex: 1, neuronIndex: 0 } };
`, context);
const evaluate = s => vm.runInContext(s, context);
assert.equal(evaluate('state.model.topology.join(",")'), '25,50,1');
assert.equal(evaluate('definition.matchesPreset(preset)'), true);
assert.equal(evaluate('SquareShuffledPartialModelData.featureDefinitions.length'), 25);
assert.equal(evaluate('SquareShuffledPartialModelData.featureDefinitions.filter(f => f.kind === "edge").length'), 24);
assert.equal(evaluate('SquareShuffledPartialModelData.verification.errors'), 0);
assert.equal(evaluate('SquareShuffledPartialModelData.verification.inputCount'), 33554432);
assert.equal(evaluate('SquareShuffledPartialModelData.training.trainableParameters'), 1351);

for (let index = 0; index < 6; index++) {
  evaluate(`state.compositionExample = ${index}; state.pixels = definition.examplePixels(${index});`);
  const report = evaluate(`(() => {
    const e = definition.examples[state.compositionExample];
    const f = forward(state.pixels);
    const terms = e.featureIndices.map(i => f.activations[1][i] * state.model.layers[1].weights[0][i]);
    const all = f.activations[1].map((v, i) => v * state.model.layers[1].weights[0][i]);
    const union = [...new Set(e.featureIndices.flatMap(i => SquareShuffledPartialModelData.featureDefinitions.find(feature => feature.neuronIndex === i).on))].sort((a,b) => a-b);
    const view = definition.renderView(state, f);
    return { score: f.outputs[0], terms, sum: all.reduce((a,b)=>a+b,0) + state.model.layers[1].biases[0],
      union: union.join(','), boundary: e.boundary.join(','),
      rows: (view.match(/data-trained-contribution=/g)||[]).length,
      cards: (view.match(/aria-label="查看H1-/g)||[]).length,
      detected: view.includes('当前检测到 4 / 4 条边') };
  })()`);
  assert.ok(report.score > 1);
  assert.ok(report.terms.every(v => v >= .1));
  assert.ok(report.terms.reduce((a,b) => a+b, 0) >= .4);
  assert.ok(Math.abs(report.sum - report.score) < 1e-12);
  assert.equal(report.union, report.boundary);
  assert.equal(report.rows, 50);
  assert.equal(report.cards, 50);
  assert.equal(report.detected, true);
  for (const variant of ['missing', 'extra']) {
    assert.ok(evaluate(`forward(definition.examplePixels(${index}, '${variant}')).outputs[0]`) <= 1);
  }
  assert.equal(evaluate(`(() => { const f=forward(definition.examplePixels(${index}, 'extra')); return definition.examples[${index}].featureIndices.every(i=>f.activations[1][i]>0); })()`), true);
}

// All 30 positives, every single/two-pixel change, and deterministic random inputs.
const total = evaluate(`(() => {
  const positives = new Set(SquareTeaching.candidates.map(c => c.boundary.reduce((mask,p)=>mask|1<<p,0)));
  const inputs = new Set([0, 33554431, SquareShuffledPartialModelData.verification.closestNegative.inputId]);
  for (const mask of positives) {
    inputs.add(mask);
    for (let i=0; i<25; i++) {
      inputs.add(mask ^ (1<<i));
      for(let j=i+1;j<25;j++) inputs.add(mask ^ (1<<i) ^ (1<<j));
    }
  }
  let seed=42;
  for(let k=0;k<10000;k++) { seed=(Math.imul(seed,1664525)+1013904223)>>>0; inputs.add(seed & 33554431); }
  const current=state, original=SquarePartialFeaturesLegacyTeaching.createPreset();
  for(const mask of inputs) {
    const p=Array.from({length:25},(_,i)=>(mask>>>i)&1);
    const score=forward(p).outputs[0];
    if ((score>1)!==positives.has(mask)) throw Error('Incorrect JS inference: '+mask);
    state=original; const before=forward(p).outputs[0]; state=current;
    if(Math.abs(before-score)>1e-12) throw Error('Reordering changed score: '+mask);
  }
  return inputs.size;
})()`);

// Every concept's exact extrema, recomputed independently in JavaScript.
evaluate(`for (const feature of SquareShuffledPartialModelData.featureDefinitions) {
  const i=feature.neuronIndex, w=state.model.layers[0].weights[i];
  const low=w.map(v=>Number(v<0));
  feature.on.forEach(p=>low[p]=1); feature.off.forEach(p=>low[p]=0);
  if (!(forward(low).activations[1][i]>0)) throw Error('Concept positive bound '+i);
  for(const [p,wanted] of [...feature.on.map(p=>[p,1]),...feature.off.map(p=>[p,0])]) {
    const high=w.map(v=>Number(v>0)); high[p]=1-wanted;
    if (!(forward(high).activations[1][i]<0)) throw Error('Concept negative bound '+i);
  }
}`);
assert.equal(evaluate(`(() => { const p=definition.createPreset();p.model.layers[0].weights[0][0]+=.000001; return definition.matchesPreset(p); })()`), false);
assert.equal(evaluate('definition.provenance().compositionExamples.length'), 6);

assert.equal(evaluate('definition.drawOnExampleSelect'), true);
assert.equal(evaluate('definition.ID'), 'square-trained-partial-50-v4');
assert.equal(evaluate('definition.provenance().permutationVerification.passed'), true);
assert.equal(evaluate('definition.matchesPreset(SquarePartialFeaturesLegacyTeaching.createPreset())'), false);
evaluate(`(() => {
  const order = SquareShuffledPartialModelData.neuronOrder;
  const source = SquarePartialFeaturesModelData;
  const model = state.model;
  if(new Set(order.newToOld).size !== 50 || order.newToOld.some((old,i)=>order.oldToNew[old] !== i)) throw Error('Invalid neuron permutation');
  for(let i=0;i<50;i++) {
    const old=order.newToOld[i];
    if(JSON.stringify(model.layers[0].weights[i])!==JSON.stringify(source.model.layers[0].weights[old]) ||
       model.layers[0].biases[i]!==source.model.layers[0].biases[old] ||
       model.layers[1].weights[0][i]!==source.model.layers[1].weights[0][old]) throw Error('Incorrect connection mapping '+i);
    const target=definition.features.find(f=>f.neuronIndex===i), original=source.featureDefinitions.find(f=>f.neuronIndex===old);
    if(Boolean(target)!==Boolean(original) || target && (target.name!==original.name || JSON.stringify(target.on)!==JSON.stringify(original.on) || JSON.stringify(target.off)!==JSON.stringify(original.off))) throw Error('Incorrect feature mapping '+i);
  }
  const indices=new Set(definition.features.map(f=>f.neuronIndex));
  if(![...indices].some(i=>i>=40) || !Array.from({length:10},(_,i)=>i).some(i=>!indices.has(i))) throw Error('Features are still grouped at the front');
})()`);
assert.equal(evaluate('SquareShuffledPartialModelData.training.freeFeatures'), 25);
assert.equal(evaluate('definition.provenance().featurePairs'), undefined);
assert.equal(evaluate('definition.matchesPreset(SquareCompositionTeaching.createPreset())'), false);
assert.equal(evaluate('definition.matchesPreset(SquareAllFeaturesTeaching.createPreset())'), false);
assert.ok(evaluate('definition.renderView(state, forward(state.pixels))').includes('21 个明确特征与 25 个自由学习特征'));

for (const name of ['SquarePartialFeaturesLegacyTeaching', 'SquareTrainedTeaching', 'SquareGuidedTeaching', 'SquareGuidedDoubleTeaching', 'SquareTeaching', 'SquareSingleTeaching']) {
  evaluate(`state={...${name}.createPreset(),pixels:SquareTeaching.pixelsFor(20),teachingExample:20,teachingCandidate:20};`);
  assert.ok(evaluate('forward(state.pixels).outputs[0]') > .5, name);
  assert.equal(evaluate(`${name}.matchesPreset(${name}.createPreset())`), true, name);
  assert.ok(evaluate(`${name}.renderView(state,forward(state.pixels)).length`) > 1000, name);
}
const artifact = JSON.parse(fs.readFileSync(path.join(root, 'partial-features-shuffled-model.json'), 'utf8'));
assert.equal(evaluate('JSON.stringify(SquareShuffledPartialModelData)'), JSON.stringify(artifact));
assert.ok(html.replaceAll('\r', '').includes(fs.readFileSync(path.join(root, 'composition-teaching.js'), 'utf8').replaceAll('\r', '')));
assert.equal(evaluate('JSON.stringify(SquarePartialFeaturesModelData)'), JSON.stringify(JSON.parse(fs.readFileSync(path.join(root, 'partial-features-model.json'), 'utf8'))));
console.log(`PASS: fixed permutation equivalence; inline syntax; ${total} exact-JS classification cases; 25 concept extrema; 6 compositions; all 50 contributions; 6 original modes; preset validation; bundle parity.`);
