// No dependencies: node llm/tests/all-features.test.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, '神经网络教学.html'), 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
scripts.forEach((s, i) => new vm.Script(s, { filename: `inline-${i}.js` }));
const context = vm.createContext({ console });
const run = text => vm.runInContext(text, context);
scripts.slice(0, -1).forEach(s => run(s));
const app = scripts.at(-1);
run(`const STEP_THRESHOLD=1;let state;function activationTypesOf(s){return [...s.hiddenLayers.map(l=>l.activation),s.outputActivation];}
${app.slice(app.indexOf('    function activate('), app.indexOf('    function trainSample('))}
const definition=SquareAllFeaturesTeaching;
state={...definition.createPreset(),pixels:definition.examplePixels(0),compositionExample:0,featurePair:20};`);
assert.equal(run('state.model.topology.join(",")'), '25,73,1');
assert.equal(run('definition.matchesPreset(definition.createPreset())'), true);
assert.equal(run('definition.features.length'), 73);
assert.equal(run('definition.pairs.length'), 30);
assert.equal(run('definition.provenance().training.totalParameters'), 1972);
assert.equal(run('definition.provenance().training.trainableParameters'), 1669);
assert.equal(run('definition.provenance().training.fixedZeroConnections'), 303);
assert.equal(run('definition.provenance().verification.inputCount'), 33554432);
assert.equal(run('definition.provenance().verification.errors'), 0);

run(`for(const f of definition.features) {
  const i=f.neuronIndex,w=state.model.layers[0].weights[i],out=state.model.layers[1].weights[0][i];
  for(let p=0;p<25;p++) {
    if(f.on.includes(p) ? w[p]<=0 : f.off.includes(p) ? w[p]>=0 : w[p]!==0) throw Error('Wrong input mask/sign '+i+':'+p);
  }
  if(f.kind==='defect' ? out>-.05 : out<.05) throw Error('Wrong output sign '+i);
  const match=Array.from({length:25},(_,p)=>Number(f.on.includes(p)));
  const contribution=forward(match).activations[1][i]*out;
  if(f.kind==='defect' ? contribution>=-.01 : contribution<=.01) throw Error('No real matching contribution '+i);
  const low=w.map(v=>Number(v<0));f.on.forEach(p=>low[p]=1);f.off.forEach(p=>low[p]=0);
  if(!(forward(low).activations[1][i]>0)) throw Error('Positive concept bound '+i);
  for(const [p,wanted] of [...f.on.map(p=>[p,1]),...f.off.map(p=>[p,0])]) {
    const high=w.map(v=>Number(v>0));high[p]=1-wanted;
    if(!(forward(high).activations[1][i]<0)) throw Error('Negative concept bound '+i);
  }
  if(!definition.renderFeatureTarget(i,forward(match).activations[1][i]).includes(f.name)) throw Error('Missing feature target '+i);
}`);

run(`for(const pair of definition.pairs) {
  const full=definition.pairPixels(pair.candidateIndex,'complete');
  const missing=definition.pairPixels(pair.candidateIndex,'missing');
  if(full.filter((v,i)=>v!==missing[i]).length!==1 || full[pair.cornerPixel]!==1 || missing[pair.cornerPixel]!==0) throw Error('Pair input');
  const a=forward(full),b=forward(missing),s=pair.skeletonIndex,d=pair.defectIndex;
  if(!(a.outputs[0]>1 && b.outputs[0]<=1 && a.activations[1][s]>0 && b.activations[1][s]>0 && a.activations[1][d]<0 && b.activations[1][d]>0)) throw Error('Wrong pair behavior '+pair.candidateIndex);
  const term=b.activations[1][d]*state.model.layers[1].weights[0][d];
  if(!(term<0 && b.outputs[0]-term>b.outputs[0])) throw Error('Defect must lower score');
}`);

for (let i=0;i<3;i++) {
  run(`state.compositionExample=${i};state.pixels=definition.examplePixels(${i});`);
  const result=run(`(() => {
    const f=forward(state.pixels),indices=definition.examples[state.compositionExample].featureIndices;
    const terms=f.activations[1].map((v,i)=>v*state.model.layers[1].weights[0][i]);
    const edges=indices.map(i=>terms[i]);
    const view=definition.renderView(state,f);
    return {score:f.outputs[0],edges,sum:terms.reduce((a,b)=>a+b,0)+state.model.layers[1].biases[0],
      rows:(view.match(/data-trained-contribution=/g)||[]).length,
      catalog:(view.split('id="allFeaturesCatalog"')[1].split('data-trained-detail="teachingTrainingOpen"')[0].match(/data-teaching-neuron=/g)||[]).length,
      hasOpaqueGroup:view.includes('自由学习特征'),has69:view.includes('其余 69 项贡献')};
  })()`);
  assert.ok(result.score>1 && result.edges.every(v=>v>=.1));
  assert.ok(result.edges.reduce((a,b)=>a+b,0)>=.4);
  assert.ok(Math.abs(result.score-result.sum)<1e-12);
  assert.equal(result.rows,73);
  assert.equal(result.catalog,73);
  assert.equal(result.hasOpaqueGroup,false);
  assert.equal(result.has69,true);
  for(const variant of ['missing','extra']) assert.ok(run(`forward(definition.examplePixels(${i},'${variant}')).outputs[0]`)<=1);
}
const total=run(`(() => {
  const positives=new Set(SquareTeaching.candidates.map(c=>c.boundary.reduce((mask,p)=>mask|(1<<p),0)));
  const inputs=new Set([0,33554431,SquareAllFeaturesModelData.verification.closestNegative.inputId]);
  for(const mask of positives) {
    inputs.add(mask);
    for(let i=0;i<25;i++) {
      inputs.add(mask^(1<<i));
      for(let j=i+1;j<25;j++) inputs.add(mask^(1<<i)^(1<<j));
    }
  }
  for(let height=1;height<=5;height++)for(let width=1;width<=5;width++)for(let row=0;row<=5-height;row++)for(let col=0;col<=5-width;col++) {
    let filled=0,outline=0;
    for(let r=row;r<row+height;r++)for(let c=col;c<col+width;c++) {const bit=1<<(r*5+c);filled|=bit;if(r===row||r===row+height-1||c===col||c===col+width-1)outline|=bit;}
    inputs.add(filled);inputs.add(outline);
  }
  let seed=42;
  for(let i=0;i<10000;i++){seed=(Math.imul(seed,1664525)+1013904223)>>>0;inputs.add(seed&33554431);}
  for(const mask of inputs){const x=Array.from({length:25},(_,p)=>(mask>>>p)&1);if((forward(x).outputs[0]>1)!==positives.has(mask))throw Error('JS classification '+mask);}
  return inputs.size;
})()`);
assert.equal(run(`(() => { const p=definition.createPreset();p.model.layers[0].weights[72][0]+=.000001;return definition.matchesPreset(p); })()`),false);
assert.equal(run('SquareCompositionTeaching.createPreset().model.topology[1]'),50);
assert.equal(run('SquareCompositionTeaching.provenance().featureDefinitions.length'),13);
assert.equal(run('JSON.stringify(SquareAllFeaturesModelData)'),JSON.stringify(JSON.parse(fs.readFileSync(path.join(root,'all-features-model.json'),'utf8'))));
console.log(`PASS: 73 exact concept bounds, input masks, contribution directions and targets; 30 pairs; 3 compositions; ${total} actual-JS classification cases; 73 catalog cards/contribution rows; preset validation and artifact parity.`);
