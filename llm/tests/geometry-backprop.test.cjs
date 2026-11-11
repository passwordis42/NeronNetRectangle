// Numerical tests for the independent sparse geometry replay engine.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const crypto = require('node:crypto');
const zlib = require('node:zlib');
const core = require('../geometry-backprop-core.js');
const close = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b}; difference ${a - b}`);

function fixture() {
  const topology = [2, 2, 2, 1], parameterMap = [], layers = [];
  for (let layer = 1; layer < topology.length; layer++) {
    const weights = [], biases = [];
    for (let neuron = 0; neuron < topology[layer]; neuron++) {
      const row = Array(topology[layer - 1]).fill(0);
      for (let input = 0; input < row.length; input++) {
        if (layer === 1 && neuron === 1 && input === 0) continue;
        row[input] = .07 * (1 + parameterMap.length);
        parameterMap.push({ layer, neuron, input, kind: 'weight', lower: -3, upper: 3 });
      }
      weights.push(row); biases.push(.1 * (neuron - 1));
    }
    for (let neuron = 0; neuron < topology[layer]; neuron++) parameterMap.push({ layer, neuron, input: null, kind: 'bias', lower: -2, upper: 2 });
    layers.push({ weights, biases });
  }
  return { artifact: { model: { topology, layers } }, manifest: { topology, parameterMap, activations: ['sigmoid', 'sigmoid', 'linear'], optimizer: { learningRate: .03, beta1: .9, beta2: .999, epsilon: 1e-8 } } };
}
const { artifact, manifest } = fixture(), engine = core.create(manifest, artifact);
const p = engine.flatten(artifact.model);
assert.equal(engine.P, 14);
assert.deepEqual(engine.model(p), artifact.model);
assert.equal(engine.indexOf(1, 1, 0), -1);
assert.equal(engine.describe(engine.indexOf(3, 0, null, 'bias')).label, '输出 1 偏置');
const f = engine.forward([1, 0], artifact.model), sigmoid = v => 1 / (1 + Math.exp(-v));
for (let k = 0; k < artifact.model.layers.length; k++) {
  for (let j = 0; j < artifact.model.layers[k].weights.length; j++) {
    const layer = artifact.model.layers[k];
    const z = layer.weights[j].reduce((a, w, i) => a + w * f.activations[k][i], layer.biases[j]);
    close(z, f.preActivations[k][j]);
    close(f.activations[k + 1][j], k < 2 ? sigmoid(z) : z);
  }
}
// The factory is genuinely worker-serializable, not dependent on module globals.
const workerFactory = vm.runInNewContext(`(${core.create.toString()})`);
const isolated = workerFactory(JSON.parse(JSON.stringify(manifest)), JSON.parse(JSON.stringify(artifact)));
close(isolated.forward([1, 0], isolated.model(p)).outputs[0], f.outputs[0]);

// Adam: nonzero momentum can update a zero-gradient parameter, and clipping is explicit.
const pp = p.slice(), mm = new Float64Array(engine.P), vv = new Float64Array(engine.P);
const gg = new Float64Array(engine.P); gg[0] = 2;
const first = engine.adam(pp, mm, vv, gg, 1);
close(first.after[0], p[0] - .03 * 2 / (2 + 1e-8));
gg[0] = 0;
const second = engine.adam(pp, mm, vv, gg, 2);
assert.notEqual(second.after[0], second.before[0]);
assert.equal(second.clipped[0], 0);
pp[0] = -2.999; gg[0] = 1e6;
const clipped = engine.adam(pp, mm, vv, gg, 3);
assert.equal(clipped.after[0], -3); assert.equal(clipped.clipped[0], 1); assert.ok(clipped.proposed[0] < -3);

// A self-contained trace reconstructs precisely, including initial random values.
const start = { parameters: Array.from(p), m: Array(engine.P).fill(0), v: Array(engine.P).fill(0) };
const a = p.slice(), m = new Float64Array(engine.P), v = new Float64Array(engine.P), updates = [];
for (let step = 1; step <= 3; step++) {
  const gradient = Array.from({ length: engine.P }, (_, i) => Math.sin(step + i) * .3);
  engine.adam(a, m, v, gradient, step);
  updates.push({ step, gradient, ids: [1, 0, 2], labels: [1, 0, 1], losses: [1, 2, 3, 4, 5] });
}
const rawChunk = { format: 'geometry-trace-v1', firstStep: 1, count: updates.length, start, updates,
  end: { parameters: Array.from(a), m: Array.from(m), v: Array.from(v) } };
const chunk = engine.readChunk(JSON.stringify(rawChunk));
assert.equal(engine.reconstruct(chunk, 0).gradient, null);
assert.deepEqual(Array.from(engine.reconstruct(chunk, 3).after), Array.from(a));
assert.throws(() => engine.reconstruct(chunk, 4), /不包含/);
const corrupt = JSON.parse(JSON.stringify(rawChunk)); corrupt.end.parameters[0] += .01;
assert.throws(() => engine.reconstruct(engine.readChunk(corrupt), 3), /未通过校验/);
const duplicate = JSON.parse(JSON.stringify(manifest)); duplicate.parameterMap.push(duplicate.parameterMap[0]);
assert.throws(() => core.create(duplicate, artifact), /重复/);
const badFixed = JSON.parse(JSON.stringify(artifact)); badFixed.model.layers[0].weights[1][0] = .5;
assert.throws(() => core.create(manifest, badFixed), /固定连接/);
const invalid = JSON.parse(JSON.stringify(rawChunk)); invalid.updates[0].labels = [1];
assert.throws(() => engine.readChunk(invalid), /批次/);

// Finite differences and complete chain provenance on a small three-layer network.
// It has real binary 5x5 inputs but fewer first-layer targets, keeping the test fast.
function gradientFixture() {
  const topology = [25, 2, 5, 1], parameterMap = [], layers = [];
  for (let layer = 1; layer < topology.length; layer++) {
    const weights = Array.from({ length: topology[layer] }, () => Array(topology[layer - 1]).fill(0));
    for (let j = 0; j < weights.length; j++) for (let i = 0; i < weights[j].length; i++) {
      if (layer === 1 && i > 2) continue;
      weights[j][i] = .15 + .01 * parameterMap.length;
      parameterMap.push({ layer, neuron: j, input: i, kind: 'weight', lower: -128, upper: 128 });
    }
    const biases = weights.map((_, j) => -.2 + j * .03);
    for (let j = 0; j < biases.length; j++) parameterMap.push({ layer, neuron: j, input: null, kind: 'bias', lower: -128, upper: 128 });
    layers.push({ weights, biases });
  }
  const artifact = { model: { topology, layers }, featureDefinitions: [
    { neuronIndex: 0, name: '亮点', on: [0], off: [] }, { neuronIndex: 1, name: '两点和空点', on: [0, 1], off: [2] }
  ] };
  const manifest = { topology, parameterMap, optimizer: { learningRate: .01 }, lossConfig: { participation: { minimum: 2 } } };
  return { artifact, manifest };
}
const gf = gradientFixture(), ge = core.create(gf.manifest, gf.artifact), gp = ge.flatten(gf.artifact.model);
const batch = { ids: [0, 99, 7, 469440], labels: [0, 1, 0, 1] }, full = ge.gradients(gp, batch, 1, 1);
const isolatedGrad = workerFactory(JSON.parse(JSON.stringify(gf.manifest)), JSON.parse(JSON.stringify(gf.artifact))).gradients(gp, batch, 1, 1);
isolatedGrad.total.forEach((value, i) => close(value, full.total[i]));
for (let parameter = 0; parameter < ge.P; parameter++) {
  const plus = gp.slice(), minus = gp.slice(), epsilon = 1e-5;
  plus[parameter] += epsilon; minus[parameter] -= epsilon;
  const lo = ge.gradients(minus, batch, 1), hi = ge.gradients(plus, batch, 1);
  for (let k = 0; k < full.parts.length; k++) close((hi.losses[k] - lo.losses[k]) / (2 * epsilon), full.parts[k][parameter], 2e-7);
  const provenance = ge.parameterTrace(gp, batch, 1, parameter);
  close(provenance.total, full.total[parameter]);
  for (let k = 0; k < full.parts.length; k++) {
    close(provenance.parts[k], full.parts[k][parameter]);
    for (const row of provenance.rows[k]) {
      close(row.rawDerivative / row.denominator, row.rawGradient);
      close(row.rawGradient * row.coefficient, row.gradient);
      close(row.chain.at(-1).value, row.gradient);
    }
    const sg = ge.sampleGradient(full.sample, parameter, ge.lossDefinitions[k].id);
    close(sg.contribution, full.sample.parts[k][parameter]);
    if (sg.denominator !== null) close(sg.raw / sg.denominator * sg.coefficient, sg.contribution);
  }
}
// Changing a phase coefficient affects the weighted term, never its raw loss.
gf.manifest.optimizer.phases = [{ firstStep: 1, lastStep: 2, lossWeights: [0, 1, 0, 1, 0] }];
const phased = core.create(gf.manifest, gf.artifact).gradients(gp, batch, 1);
for (const k of [0, 2, 4]) { assert.equal(phased.losses[k], 0); assert.deepEqual(phased.rawLosses[k], full.rawLosses[k]); assert.ok(phased.parts[k].every(x => x === 0)); }

// Optional independent Python trainer comparison, including all five loss gradients.
if (process.argv.includes('--oracle') || process.argv.includes('--oracle-single')) {
  const { execFileSync } = require('node:child_process');
  const code = `import sys,json,numpy as np\nsys.path.insert(0,'llm/scripts')\nimport train_geometry_features as t\nd=t.definition('double')\nrng=np.random.default_rng(42)\np=rng.uniform(-.1,.1,d['count'])\np[[a['index'] for a in d['parameterMap'] if a['kind']=='bias']]=0\nids=np.array([0,99,7,469440,33554431],dtype=np.uint32)\nlabels=np.array([0,1,0,1,0])\nlosses,g,parts=t.loss_and_gradients(p,ids,labels,d,True)\nw=t.unpack(p,d)\nmodel={'topology':d['topology'],'layers':[{'weights':a.tolist(),'biases':b.tolist()} for a,b in w]}\nm=np.zeros_like(p);v=np.zeros_like(p);after=p.copy();t.adam(after,m,v,g,1,.05,d)\nprint(json.dumps({'manifest':{'topology':d['topology'],'parameterMap':d['parameterMap'],'lossConfig':d['lossConfig'],'optimizer':{'learningRate':.05}},'artifact':{'model':model,'featureDefinitions':d['features']},'parameters':p.tolist(),'ids':ids.tolist(),'labels':labels.tolist(),'losses':losses.tolist(),'gradient':g.tolist(),'parts':[a.tolist() for a in parts],'after':after.tolist(),'m':m.tolist(),'v':v.tolist()}))`;
  const oracleCode = process.argv.includes('--oracle-single') ? code.replace("d=t.definition('double')", "d=t.definition('single')") : code;
  const oracle = JSON.parse(execFileSync('python', ['-c', oracleCode], { cwd: path.resolve(__dirname, '../..'), maxBuffer: 20 * 1024 * 1024 }).toString());
  const oe = core.create(oracle.manifest, oracle.artifact), got = oe.gradients(oracle.parameters, oracle, 1);
  got.losses.forEach((x, k) => close(x, oracle.losses[k], 1e-8));
  got.parts.forEach((values, k) => values.forEach((x, i) => close(x, oracle.parts[k][i], 1e-8)));
  got.total.forEach((x, i) => close(x, oracle.gradient[i], 1e-8));
  const pp = Float64Array.from(oracle.parameters), mm = new Float64Array(oe.P), vv = new Float64Array(oe.P);
  oe.adam(pp, mm, vv, oracle.gradient, 1);
  for (const [x, y] of [[pp, oracle.after], [mm, oracle.m], [vv, oracle.v]]) x.forEach((value, i) => close(value, y[i], 1e-13));
  console.log(`Python trainer oracle matched: ${oe.P} sparse parameters, all five loss gradients and Adam state`);
}

// Optional recorded trace audit: node geometry-backprop.test.cjs manifest.json artifact.json
function auditTrace(manifestFile, artifactFile) {
  const mf = JSON.parse(fs.readFileSync(manifestFile)), ar = JSON.parse(fs.readFileSync(artifactFile));
  const e = core.create(mf, ar), dir = path.dirname(manifestFile);
  let previous, count = 0, maximumError = 0, maximumGradientError = 0, gradientsChecked = 0;
  const allGradients = process.argv.includes('--all-gradients');
  const chosen = new Set([1, ...mf.stages]);
  for (const entry of mf.chunks) {
    const source = fs.readFileSync(path.join(dir, entry.file), 'utf8');
    const match = /SquareTraceChunks\.receive\("([^"\n]+)",\s*(\d+),\s*"([A-Za-z0-9+/=]+)"\)/.exec(source);
    assert.ok(match, `Bad trace wrapper: ${entry.file}`);
    assert.equal(match[1], mf.id); assert.equal(+match[2], entry.index);
    const packed = Buffer.from(match[3], 'base64');
    assert.equal(crypto.createHash('sha256').update(packed).digest('hex'), entry.sha256);
    const c = e.readChunk(zlib.gunzipSync(packed).toString('utf8'));
    if (entry.index === 0) {
      assert.equal(mf.initialization.inheritedParameters, false);
      for (const d of e.parameterMap) {
        if (d.kind === 'bias') assert.equal(c.start.parameters[d.index], 0);
        else assert.ok(c.start.parameters[d.index] >= -.1 && c.start.parameters[d.index] <= .1);
      }
      assert.ok(new Set(c.start.parameters).size > e.P / 2, 'Initialization must not be a constant template');
      assert.equal(e.reconstruct(c, 0).gradient, null);
    }
    if (previous) for (const key of ['parameters', 'm', 'v']) assert.deepEqual(c.start[key], previous[key]);
    const parameters = c.start.parameters.slice(), moments = c.start.m.slice(), variances = c.start.v.slice();
    for (const update of c.updates) {
      if (allGradients || chosen.has(update.step) || update.step === c.firstStep || update.step === c.firstStep + c.count - 1) {
        const detail = e.gradients(parameters, update, update.step);
        detail.total.forEach((value, i) => { maximumGradientError = Math.max(maximumGradientError, Math.abs(value - update.gradient[i])); close(value, update.gradient[i], 1e-8); });
        detail.losses.forEach((value, i) => close(value, update.losses[i], 1e-8));
        gradientsChecked++;
      }
      e.adam(parameters, moments, variances, update.gradient, update.step); count++;
    }
    for (const [x, y] of [[parameters, c.end.parameters], [moments, c.end.m], [variances, c.end.v]]) {
      maximumError = Math.max(maximumError, e.compareVectors(x, y, 'trace audit'));
    }
    previous = c.end;
  }
  assert.equal(count, mf.steps);
  assert.deepEqual(Array.from(previous.parameters, x => +x.toFixed(mf.precision || 6)), Array.from(e.flatten(ar.model)));
  assert.equal(mf.verification.stepsChecked, mf.steps);
  console.log(JSON.stringify({ trace: mf.id, updatesChecked: count, gradientsChecked, maximumError, maximumGradientError }));
}
if (process.argv[2] && process.argv[3] && !process.argv[2].startsWith('--')) auditTrace(path.resolve(process.argv[2]), path.resolve(process.argv[3]));
console.log('geometry-backprop core tests passed');
