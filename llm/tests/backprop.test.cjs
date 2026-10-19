// Exact artifacts + every recorded Adam transition + independent gradient checks.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const zlib = require('node:zlib');
const core = require('../backprop-core.js');
const root = path.resolve(__dirname, '..');
const version=process.argv[2]||'v1';
const directory = path.join(root, 'training-trace',version);
const manifest = JSON.parse(fs.readFileSync(path.join(directory,'manifest.json')));
const artifact = JSON.parse(fs.readFileSync(path.join(root,version==='v2'?'partial-features-random-model.json':'partial-features-shuffled-model.json')));
const close = (a,b,tolerance=1e-9) => assert.ok(Math.abs(a-b) <= tolerance, `${a} != ${b}, difference ${a-b}`);
assert.equal(manifest.steps,version==='v2'?18000:12000);
assert.equal(manifest.modelSha256,artifact.modelSha256);
assert.equal(manifest.verification.stepsChecked,manifest.steps);
assert.ok(manifest.verification.maximumGradientError<1e-9);
assert.deepEqual(manifest.neuronOrder,artifact.neuronOrder);
const chosen = new Set([1,100,101,6000,6001,12000,12001,...manifest.stages.slice(1)]);
const saved = new Map();
let previousEnd, transitions=0, maximumError=0;
for (const entry of manifest.chunks) {
  const source = fs.readFileSync(path.join(directory,entry.file),'utf8');
  const match = /^SquareTraceChunks\.receive\((?:"v2",)?(\d+),"([A-Za-z0-9+/=]+)"\);\r?\n$/.exec(source);
  assert.ok(match);assert.equal(Number(match[1]),entry.index);
  const packed = Buffer.from(match[2],'base64');
  assert.equal(crypto.createHash('sha256').update(packed).digest('hex'),entry.sha256);
  const raw=zlib.gunzipSync(packed), chunk=core.readChunk(raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength));
  assert.equal(chunk.firstStep,entry.firstStep);
  if(previousEnd) for(const k of ['p','m','v']) assert.deepEqual(chunk[k],previousEnd[k]);
  if(entry.index===0) {
    const initial=core.reconstruct(chunk,0,manifest.optimizer);
    assert.deepEqual(initial.before,initial.after); assert.equal(initial.ids.length,0);
    saved.set(0,initial);
  }
  const p=chunk.p.slice(),m=chunk.m.slice(),v=chunk.v.slice();
  for(const step of chunk.steps) {
    const oldM=m.slice(),oldV=v.slice();
    const update=core.adam(p,m,v,step.gradient,step.step,manifest.optimizer);
    assert.ok(update.after.every(Number.isFinite));
    if(chosen.has(step.step)) {
      const r={...step,...update,oldM,oldV}; saved.set(step.step,r);
      const details=core.gradients(r.before,r.ids,r.positives,artifact.featureDefinitions,artifact.compositionExamples,0,core.optimizerAt(manifest.optimizer,step.step).lossWeights);
      for(let k=0;k<core.P;k++) close(details.total[k],r.gradient[k],1e-8);
      details.losses.forEach((value,k)=>close(value,r.losses[k],1e-8));
      assert.equal(details.sample.id,r.ids[0]);assert.equal(details.sample.positive,true);
    }
    transitions++;
  }
  for(const [a,b] of [[p,chunk.end.p],[m,chunk.end.m],[v,chunk.end.v]]) for(let i=0;i<core.P;i++) {
    maximumError=Math.max(maximumError,Math.abs(a[i]-b[i]));close(a[i],b[i],1e-10);
  }
  previousEnd=chunk.end;
  assert.throws(()=>core.reconstruct(chunk,-1,manifest.optimizer));
}
assert.equal(transitions,manifest.steps);
assert.deepEqual(Array.from(previousEnd.p,v=>Number(v.toFixed(6))),Array.from(core.flatten(artifact.model)));
// Independent finite differences for W1, b1, W2, b2 and every loss component.
const base=saved.get(1), p=base.before.slice(), j=artifact.compositionExamples[0].featureIndices[0];
p[1250+j]+=2; p[1300+j]=.05; // Activate both robust and contribution constraints.
const ids=base.ids.slice(0,8),positiveCount=8;
const analytical=core.gradients(p,ids,positiveCount,artifact.featureDefinitions,artifact.compositionExamples);
assert.ok(analytical.losses[2]>0 && analytical.losses[3]>0);
for(const parameter of [j*25+6,j*25+7,1250+j,1300+j,1350,36*25+12,1250+36]) {
  const plus=p.slice(),minus=p.slice(),epsilon=1e-6;
  plus[parameter]+=epsilon;minus[parameter]-=epsilon;
  const a=core.gradients(plus,ids,positiveCount,artifact.featureDefinitions,artifact.compositionExamples);
  const b=core.gradients(minus,ids,positiveCount,artifact.featureDefinitions,artifact.compositionExamples);
  for(let component=0;component<4;component++) close((a.losses[component]-b.losses[component])/(2*epsilon),analytical.parts[component][parameter],2e-5);
}
// Single-sample contributions use batch normalization; structural terms are separate.
const one=core.gradients(base.before,base.ids,base.positives,artifact.featureDefinitions,artifact.compositionExamples,0);
for(const k of [506,1250+20,1300+20,1350]) {
  const g=core.sampleGradient(one.sample,k),d=core.describe(k);
  const expected=d.layer===2?one.sample.ds*(d.kind==='bias'?1:one.sample.activations[1][d.input]):one.sample.dzClass[d.neuron]*(d.kind==='bias'?1:one.sample.pixels[d.input]);
  close(g.classification,expected,1e-14);
}
// Adam can update with zero current gradient, and clamps proposals to the bounds.
const ap=new Float64Array(core.P),am=new Float64Array(core.P),av=new Float64Array(core.P),ag=new Float64Array(core.P);
am[0]=.3;av[0]=.01;ap[1]=10;ag[1]=-100;
const update=core.adam(ap,am,av,ag,2,manifest.optimizer);
assert.notEqual(update.after[0],update.before[0]);assert.equal(update.clipped[1],1);assert.equal(update.after[1],10);
// The published observation and six example sums use exactly the shared forward().
for(const example of artifact.compositionExamples) {
  const x=Array.from({length:25},(_,i)=>Number(example.boundary.includes(i))),r=core.forward(x,artifact.model);
  close(r.outputs[0],r.activations[1].reduce((s,v,i)=>s+v*artifact.model.layers[1].weights[0][i],artifact.model.layers[1].biases[0]),1e-12);
  assert.ok(r.outputs[0]>1);
}
assert.equal(new Set(Array.from({length:core.P},(_,i)=>core.describe(i).label)).size,1351);
console.log(`PASS: ${transitions} updates, ${manifest.chunks.length} checksummed chunks, all block boundaries, all stages and 6000/6001, four loss gradients, finite differences, bias/momentum/clipping, exact released model. Maximum JS Adam error ${maximumError}.`);

if(version==='v2') {
  const initial=saved.get(0).before,warm=saved.get(6000).after;
  assert.ok(initial.slice(0,1250).every(v=>v>=-.1&&v<=.1));
  assert.ok(initial.slice(1250,1300).every(v=>v===0));assert.equal(initial[1350],0);
  assert.ok(initial.slice(1300,1350).every(v=>v>=-.1&&v<=.1));
  const guided=new Set(artifact.featureDefinitions.map(f=>f.neuronIndex));
  for(let j=0;j<50;j++)if(!guided.has(j)) for(let i=0;i<25;i++)assert.equal(initial[j*25+i],warm[j*25+i]);
  assert.deepEqual(initial.slice(1300),warm.slice(1300));
  assert.equal(core.optimizerAt(manifest.optimizer,6000).learningRate,.01);
  assert.equal(core.optimizerAt(manifest.optimizer,6001).learningRate,.003);
  const r=saved.get(100),weights=[0,1,1,0];
  const d=core.gradients(r.before,r.ids,r.positives,artifact.featureDefinitions,artifact.compositionExamples,0,weights);
  assert.equal(d.losses[0],0);assert.equal(d.losses[3],0);assert.ok(d.rawLosses[0]>0);
  for(const k of [506,1270,1320,1350]) {
    const plus=r.before.slice(),minus=r.before.slice(),epsilon=1e-6;plus[k]+=epsilon;minus[k]-=epsilon;
    const a=core.gradients(plus,r.ids,r.positives,artifact.featureDefinitions,artifact.compositionExamples,0,weights);
    const b=core.gradients(minus,r.ids,r.positives,artifact.featureDefinitions,artifact.compositionExamples,0,weights);
    close((a.losses.reduce((s,v)=>s+v,0)-b.losses.reduce((s,v)=>s+v,0))/(2*epsilon),d.total[k],2e-5);
  }
  const found=core.findNonzeroSample(r.before,r.ids,r.positives,artifact.featureDefinitions,506,'concept',0);
  assert.ok(found>=0);const sample=core.gradients(r.before,r.ids,r.positives,artifact.featureDefinitions,artifact.compositionExamples,found).sample;
  assert.notEqual(core.sampleGradient(sample,506).concept,0);
  console.log('PASS random initialization, warmup inactive parameters, phase coefficients, learning-rate transition, finite differences and nonzero sample search.');
}
