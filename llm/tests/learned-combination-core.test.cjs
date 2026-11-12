// New H2 units learn only from final classification, even when their count is five.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { execFileSync } = require('node:child_process');
const Core = require('../geometry-backprop-core.js');
const near = (a, b, eps = 2e-8) => assert.ok(Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= eps, `${a} != ${b}`);
const clone = x => JSON.parse(JSON.stringify(x));

function fixture(width) {
  const topology = [25, 3, width, 1], parameterMap = [], layers = [];
  const featureDefinitions = [
    { neuronIndex: 0, name: '亮点 X1', on: [0], off: [] },
    { neuronIndex: 1, name: '两亮一空', on: [1, 2], off: [3] },
    { neuronIndex: 2, name: '左上转角', on: [0, 1, 5], off: [6] }
  ];
  for (let layer = 1; layer < topology.length; layer++) {
    const weights = Array.from({ length: topology[layer] }, () => Array(topology[layer - 1]).fill(0));
    for (let neuron = 0; neuron < topology[layer]; neuron++) for (let input = 0; input < topology[layer - 1]; input++) {
      const feature = featureDefinitions[neuron];
      if (layer === 1 && ![...feature.on, ...feature.off].includes(input)) continue;
      const i = parameterMap.length;
      weights[neuron][input] = Math.sin(i + .37) * .09;
      const positive = layer === 1 && feature.on.includes(input), negative = layer === 1 && feature.off.includes(input);
      parameterMap.push({ index: i, layer, neuron, input, kind: 'weight', lower: positive ? .002 : -128, upper: negative ? -.002 : 128 });
    }
    const biases = Array(topology[layer]).fill(0);
    biases.forEach((_, neuron) => parameterMap.push({ index: parameterMap.length, layer, neuron, input: null, kind: 'bias', lower: -128, upper: 128 }));
    layers.push({ weights, biases });
  }
  const lossDefinitions = [['classification', '分类误差'], ['local', '局部特征监督'], ['geometry', '几何条件监督'], ['boundary', '第一层检测边界'], ['participation', '输出参与约束']].map(([id, name]) => ({ id, name }));
  const lossConfig = { classification: { positive: 1.6, negative: .4 }, local: { coefficient: .2 }, geometry: { coefficient: 0 },
    boundary: { localMargin: 8, geometryEnabled: false, coefficient: 1 }, participation: { coefficient: 0 } };
  const artifact = { learningMode: 'learned-combination', model: { topology, layers }, featureDefinitions, geometryDefinitions: [], parameterMap, training: { lossConfig } };
  const manifest = { id: `fixture-learned-${width}`, presetId: 'square-trained-learned-combination-v1', learningMode: 'learned-combination',
    topology, activations: ['sigmoid', 'sigmoid', 'linear'], parameterMap, lossDefinitions, lossConfig,
    optimizer: { learningRate: .03, lossWeights: [1, 1, 0, 1, 0] }, stages: [0, 1, 3], steps: 3, chunks: [] };
  return { artifact, manifest };
}

for (const width of [5, 8]) {
  const { artifact, manifest } = fixture(width), core = Core.create(manifest, artifact), p = core.flatten(artifact.model);
  const batch = { ids: [0, 99, 7, 469440], labels: [0, 0, 0, 1] }, detail = core.gradients(p, batch, 1, 3);
  assert.equal(core.hasGeometry, false); assert.equal(core.hasGeometryBoundary, false);
  assert.deepEqual(core.lossDefinitions.filter(d => core.lossEnabled(d.id)).map(d => d.id), ['classification', 'local', 'boundary']);
  assert.equal(detail.sample.geometryTargets, null);
  assert.ok(detail.sample.terms.every(t => t.layer === 1 || t.lossIndex === 0), 'No second-layer target, including geometry margin, is generated');
  for (const k of [2, 4]) {
    assert.equal(detail.rawLosses[k], 0); assert.equal(detail.losses[k], 0);
    assert.ok(detail.rawParts[k].every(v => v === 0)); assert.ok(detail.parts[k].every(v => v === 0));
  }
  for (const d of core.parameterMap) {
    const trace = core.parameterTrace(p, batch, 1, d.index);
    near(trace.total, detail.total[d.index]);
    assert.deepEqual(trace.rows[2], []); assert.deepEqual(trace.rows[4], []);
    if (d.layer >= 2) {
      for (const k of [1, 2, 3, 4]) { assert.equal(detail.total[d.index], detail.parts[0][d.index]); assert.equal(detail.parts[k][d.index], 0); assert.deepEqual(trace.rows[k], []); }
      assert.equal(d.lower, -128); assert.equal(d.upper, 128);
    }
    const plus = p.slice(), minus = p.slice(), eps = 1e-5;
    plus[d.index] += eps; minus[d.index] -= eps;
    const lo = core.gradients(minus, batch, 1), hi = core.gradients(plus, batch, 1);
    for (let k = 0; k < 5; k++) near((hi.losses[k] - lo.losses[k]) / (2 * eps), detail.parts[k][d.index], 2e-7);
  }
  // Random second-layer signs survive a zero-gradient update; no old sign projection.
  const positive = core.parameterMap.find(d => d.layer === 2 && d.kind === 'weight' && p[d.index] > 0);
  const negative = core.parameterMap.find(d => d.layer === 2 && d.kind === 'weight' && p[d.index] < 0);
  const q = p.slice(), m = new Float64Array(core.P), v = new Float64Array(core.P);
  core.adam(q, m, v, new Float64Array(core.P), 1);
  assert.equal(q[positive.index], p[positive.index]); assert.equal(q[negative.index], p[negative.index]);
  const source = core.sampleGradient(detail.sample, negative.index, 'boundary');
  assert.equal(source.contribution, 0); assert.equal(source.denominator, null); assert.match(source.note, /只约束第一层/);
  assert.equal(core.findNonzeroSample(p, batch, 1, negative.index, 'geometry'), -1);
  // A short final block is valid; its true last step, not a multiple of 100,
  // determines the endpoint check and the published parameter state.
  const replayParameters = p.slice(), replayM = new Float64Array(core.P), replayV = new Float64Array(core.P);
  const state = () => ({ parameters: Array.from(replayParameters), m: Array.from(replayM), v: Array.from(replayV) });
  let previousEnd;
  for (const [firstStep, count] of [[1, 2], [3, 3]]) {
    const start = state(), updates = [];
    if (previousEnd) assert.deepEqual(start, previousEnd);
    for (let step = firstStep; step < firstStep + count; step++) {
      const gradient = core.gradients(replayParameters, batch, step);
      updates.push({ step, ids: batch.ids, labels: batch.labels, gradient: Array.from(gradient.total), losses: gradient.losses });
      core.adam(replayParameters, replayM, replayV, gradient.total, step);
    }
    const chunk = core.readChunk({ format: 'geometry-trace-v1', firstStep, count, start, updates, end: state() });
    if (firstStep === 1) assert.equal(core.reconstruct(chunk, 0).gradient, null);
    const reconstructed = core.reconstruct(chunk, firstStep + count - 1);
    assert.equal(reconstructed.endpointError, 0);
    assert.deepEqual(Array.from(reconstructed.after), Array.from(replayParameters));
    previousEnd = state();
  }
  const workerFactory = vm.runInNewContext(`(${Core.create.toString()})`), worker = workerFactory(clone(manifest), clone(artifact));
  const isolated = worker.gradients(p, batch, 1, 3);
  isolated.total.forEach((value, i) => near(value, detail.total[i]));
  for (const mutate of [
    mf => mf.lossConfig.geometry.coefficient = .2,
    mf => mf.lossConfig.boundary.geometryEnabled = true,
    mf => mf.lossConfig.participation.coefficient = 1
  ]) {
    const bad = clone(manifest); mutate(bad); assert.throws(() => Core.create(bad, artifact), /不能包含/);
  }

  // Exercise the real replay source selector without a browser or a trace download.
  const host = { innerHTML: '', hidden: true, scrollTop: 0, addEventListener() {} }, tabs = { ...host };
  const storage = new Map([[`nn-square-geometry-replay:${manifest.id}`, JSON.stringify({ source: 'geometry' })]]);
  const sandbox = { GeometryBackpropCore: Core, document: { addEventListener() {} },
    localStorage: { getItem: k => storage.get(k), setItem: (k, v) => storage.set(k, v) }, console };
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../geometry-backprop-teaching.js'), 'utf8') + '\nglobalThis.replayAPI=SquareGeometryBackpropTeaching;', sandbox);
  const replay = sandbox.replayAPI.create({ host, tabs, entries: [{ manifest, artifact }], onChange() {}, onSelect() {} });
  replay.sync({ teachingPreset: manifest.presetId, pixels: Array(25).fill(0) });
  assert.equal(replay.context.source, 'classification', 'A stale disabled loss preference is not restored');
  const toolbar = replay.renderToolbar();
  assert.ok(toolbar.includes('value="classification"') && toolbar.includes('value="local"') && toolbar.includes('value="boundary"'));
  assert.ok(!toolbar.includes('value="geometry"') && !toolbar.includes('value="participation"'));
  assert.match(toolbar, /第二层没有指定的几何目标/);
  replay.setView('source', 'geometry'); assert.equal(replay.context.source, 'classification');
  replay.setView('source', 'boundary'); assert.match(replay.context.sourceName, /第一层局部边界/);
}

// Independent training implementation: no geometry labels are calculated on either side.
if (process.argv.includes('--oracle')) {
  const python = process.env.LEARNED_PYTHON || 'python';
  for (const width of [5, 16]) {
    const code = `import sys,json,numpy as np
sys.path.insert(0,'llm/scripts')
import train_learned_combination as t
d=t.definition(${width})
rng=np.random.default_rng(42);p=rng.uniform(-.1,.1,d['count']);p[[a['index'] for a in d['parameterMap'] if a['kind']=='bias']]=0
ids=np.array([0,99,7,469440,33554431],dtype=np.uint32);labels=np.array([0,0,0,1,0])
loss,g,parts=t.loss_and_gradients(p,ids,labels,d,True)
params=t.base.unpack(p,d);model={'topology':d['topology'],'layers':[{'weights':w.tolist(),'biases':b.tolist()} for w,b in params]}
q=p.copy();m=np.zeros_like(p);v=np.zeros_like(p);t.adam(q,m,v,g,1,.03,d)
print(json.dumps({'manifest':{'learningMode':'learned-combination','topology':d['topology'],'parameterMap':d['parameterMap'],'lossConfig':d['lossConfig'],'optimizer':{'learningRate':.03,'lossWeights':[1,1,0,1,0]}},'artifact':{'learningMode':'learned-combination','model':model,'featureDefinitions':d['features'],'geometryDefinitions':[]},'parameters':p.tolist(),'ids':ids.tolist(),'labels':labels.tolist(),'losses':loss.tolist(),'gradient':g.tolist(),'parts':[x.tolist() for x in parts],'after':q.tolist(),'m':m.tolist(),'v':v.tolist()}))`;
    const oracle = JSON.parse(execFileSync(python, ['-c', code], { cwd: path.resolve(__dirname, '../..'), maxBuffer: 20 * 1024 * 1024 }).toString());
    const core = Core.create(oracle.manifest, oracle.artifact), detail = core.gradients(oracle.parameters, oracle, 1);
    detail.rawLosses.forEach((value, k) => near(value, oracle.losses[k]));
    detail.parts.forEach((part, k) => part.forEach((value, i) => near(value, oracle.parts[k][i])));
    detail.total.forEach((value, i) => near(value, oracle.gradient[i]));
    const p = Float64Array.from(oracle.parameters), m = new Float64Array(core.P), v = new Float64Array(core.P);
    core.adam(p, m, v, oracle.gradient, 1);
    for (const [actual, expected] of [[p, oracle.after], [m, oracle.m], [v, oracle.v]]) actual.forEach((value, i) => near(value, expected[i], 1e-13));
    console.log(`PASS learned H2 Python oracle: width ${width}, ${core.P} parameters, all losses/gradients/Adam`);
  }
}
console.log('PASS learned-combination core: free H2 widths 5/8, finite differences for every parameter, absent geometry and participation supervision, signed dense learning, source choices, worker parity.');
