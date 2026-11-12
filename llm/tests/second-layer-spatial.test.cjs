// node llm/tests/second-layer-spatial.test.cjs
// Spatial explanations must use the same nonlinear forward engine as the page.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const spatial = require('../second-layer-spatial.js');
const forward = require('../backprop-core.js').forward;
const root = path.resolve(__dirname, '..');
const clone = value => structuredClone(value);
const close = (actual, expected, label = 'Numerical identity') => assert.ok(
  Number.isFinite(actual) && Math.abs(actual - expected) <= 2e-10 * Math.max(1, Math.abs(actual), Math.abs(expected)),
  `${label}: ${actual} != ${expected}`
);
function freeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value);
    Object.values(value).forEach(freeze);
  }
  return value;
}
function square(row, column, size) {
  const pixels = Array(25).fill(0);
  for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) {
    if (!r || !c || r === size - 1 || c === size - 1) pixels[(row + r) * 5 + column + c] = 1;
  }
  return pixels;
}

const geometry = JSON.parse(fs.readFileSync(path.join(root, 'geometry-double-model.json'), 'utf8'));
const html = fs.readFileSync(path.join(root, '神经网络教学.html'), 'utf8');
const guidedMatch = html.match(/const SquareGuidedDoubleModelData\s*=\s*(\{[^\r\n]+\});/);
assert.ok(guidedMatch, 'The actual guided two-hidden-layer artifact is embedded in the teaching page');
const guided = JSON.parse(guidedMatch[1]);
const formalModels = [
  { name: 'geometry', model: geometry.model, types: ['sigmoid', 'sigmoid', 'linear'], sourceCount: 119, hiddenCount: 5 },
  { name: 'guided', model: guided.model, types: ['tanh', 'sigmoid', 'linear'], sourceCount: 50, hiddenCount: 30 }
];
let inspected = 0, changedPixels = 0, baselineEvaluations = 0, interventionEvaluations = 0;

function inspect(model, types, input, neuronIndex, label) {
  const beforeModel = JSON.stringify(model), beforeInput = JSON.stringify(input);
  // This is the one baseline forward already performed by the normal page.
  const baseline = forward(input, model, types); baselineEvaluations++;
  const beforeForward = JSON.stringify(baseline);
  freeze(input); freeze(baseline);
  const calls = [];
  const result = spatial.compute(model, baseline, (pixels, suppliedModel) => {
    assert.equal(suppliedModel, model, 'The injected forward receives the actual observed parameter object');
    calls.push(Array.from(pixels));
    return forward(pixels, suppliedModel, types);
  }, neuronIndex);
  assert.ok(result, `${label} has a valid H2 explanation`);
  const binary = input.every(value => value === 0 || value === 1);
  const grayPixels = input.filter(value => value !== 0 && value !== 1).length;
  assert.equal(calls.length, 25 + grayPixels, 'Reuse each known binary endpoint; a gray pixel requires both real endpoints');
  assert.equal(calls.length + 1, 26 + grayPixels, 'Total work is 26 for black/white and 51 for entirely gray input, including the page baseline');
  assert.deepEqual(Array.from(result.input), Array.from(input));
  interventionEvaluations += calls.length;
  for (const pixels of calls) {
    const changed = pixels.flatMap((value, index) => value === input[index] ? [] : [index]);
    assert.ok(changed.length <= 1, 'Every intervention changes at most one pixel and preserves the other 24');
    if (binary) assert.equal(changed.length, 1, 'Binary optimization evaluates exactly the opposite value');
    for (const index of changed) assert.ok(pixels[index] === 0 || pixels[index] === 1, 'Only zero/one endpoints are probed');
  }
  const sources = result.sources;
  assert.equal(sources.length, model.topology[1], 'Keep every H1 source, including zero-weight connections');
  assert.deepEqual(sources.map(source => source.index), Array.from({ length: model.topology[1] }, (_, index) => index));
  sources.forEach((source, index) => {
    assert.deepEqual(Array.from(source.weights), Array.from(model.layers[0].weights[index]), 'Source weight map is its actual full 25-input row');
    assert.equal(source.weights.length, 25); assert.equal(source.pixelContributions.length, 25);
    source.pixelContributions.forEach((value, pixel) => close(value, input[pixel] * model.layers[0].weights[index][pixel], 'Source pixel contribution x × W1'));
    assert.equal(source.activation, baseline.activations[1][index]);
    assert.equal(source.bias, model.layers[0].biases[index]);
    assert.equal(source.preActivation, baseline.preActivations[0][index]);
    close(source.pixelContributions.reduce((sum, value) => sum + value, 0) + source.bias, source.preActivation, 'Each source pixel map plus its own bias gives H1 z');
    assert.equal(source.weight, model.layers[1].weights[neuronIndex][index]);
    close(source.contribution, source.activation * source.weight, 'Actual H1 response × W2');
  });
  close(result.weightedSum, sources.reduce((sum, source) => sum + source.contribution, 0), 'All H1 contributions sum');
  assert.equal(result.bias, model.layers[1].biases[neuronIndex]);
  close(result.weightedSum + result.bias, baseline.preActivations[1][neuronIndex], 'Σq + H2 bias = actual H2 z');
  assert.equal(result.preActivation, baseline.preActivations[1][neuronIndex]);
  assert.equal(result.activation, baseline.activations[2][neuronIndex]);
  assert.equal(result.pixels.length, 25);
  assert.deepEqual(result.pixels.map(pixel => pixel.index), Array.from({ length: 25 }, (_, index) => index));
  result.pixels.forEach((pixel, index) => {
    const zero = Array.from(input), one = Array.from(input); zero[index] = 0; one[index] = 1;
    const f0 = forward(zero, model, types), f1 = forward(one, model, types);
    assert.equal(pixel.inputValue, input[index]);
    assert.equal(pixel.z0, f0.preActivations[1][neuronIndex]);
    assert.equal(pixel.z1, f1.preActivations[1][neuronIndex]);
    assert.equal(pixel.a0, f0.activations[2][neuronIndex]);
    assert.equal(pixel.a1, f1.activations[2][neuronIndex]);
    close(pixel.deltaZ, pixel.z1 - pixel.z0, 'Pixel influence is z(x_i=1) − z(x_i=0)');
    close(pixel.deltaActivation, pixel.a1 - pixel.a0, 'Pixel response influence uses both actual nonlinear responses');
    if (binary) {
      assert.equal(input[index] ? pixel.z1 : pixel.z0, result.preActivation);
      assert.equal(input[index] ? pixel.a1 : pixel.a0, result.activation);
    }
    changedPixels++;
  });
  assert.equal(JSON.stringify(model), beforeModel, 'Spatial inspection never mutates model parameters');
  assert.equal(JSON.stringify(input), beforeInput, 'Spatial inspection never changes the observed drawing');
  assert.equal(JSON.stringify(baseline), beforeForward, 'The supplied baseline remains untouched');
  inspected++;
  return result;
}

for (const artifact of formalModels) {
  assert.deepEqual(artifact.model.topology, [25, artifact.sourceCount, artifact.hiddenCount, 1]);
  freeze(artifact.model);
  const center = square(1, 1, 3), broken = center.slice(), filled = center.slice(), noisy = center.slice();
  broken[6] = 0; filled[12] = 1; noisy[0] = 1;
  const examples = [center, broken, filled, noisy, Array(25).fill(0), Array(25).fill(1), square(0, 0, 5),
    Array.from({ length: 25 }, (_, i) => (i % 7 + 1) / 9)];
  for (const [example, pixels] of examples.entries()) for (let neuron = 0; neuron < artifact.hiddenCount; neuron++) {
    const result = inspect(artifact.model, artifact.types, pixels, neuron, `${artifact.name} input ${example}, H2-${neuron + 1}`);
    if (artifact.name === 'geometry' && example === 0 && neuron === 0) {
      const linearShortcut = result.sources.reduce((sum, source) => sum + source.weight * source.weights[12], 0);
      assert.ok(linearShortcut > 0, 'Counterexample has a positive W2 × W1 center coefficient');
      assert.ok(result.pixels[12].deltaZ < 0, 'Real center insertion lowers the local-connection H2 condition');
      assert.ok(result.pixels[12].deltaActivation < 0, 'The actual sigmoid response agrees with the negative effect');
      assert.notEqual(result.pixels.reduce((sum, pixel) => sum + pixel.deltaZ, 0), result.weightedSum,
        'Conditional pixel differences are not additive source contributions or a score decomposition');
    }
  }
}

// Fully dense, zero-output-weight and zero-input-weight sources all remain in
// canonical order. A negative activation times a negative W2 supports H2.
const fixture = freeze({ topology: [25, 3, 2, 1], layers: [
  { weights: [Array(25).fill(.02), Array(25).fill(0), Array(25).fill(-.03)], biases: [-1, 0, -.5] },
  { weights: [[-2, 0, 3], [0, -0, 0]], biases: [.7, -.25] },
  { weights: [[1, -1]], biases: [.1] }
] });
const fixtureTypes = ['tanh', 'sigmoid', 'linear'];
const denseResult = inspect(fixture, fixtureTypes, Array(25).fill(1), 0, 'Dense/zero fixture');
assert.ok(denseResult.sources[0].activation < 0 && denseResult.sources[0].weight < 0 && denseResult.sources[0].contribution > 0);
assert.equal(denseResult.sources[1].contribution, 0);
assert.ok(denseResult.sources[1].weights.every(value => value === 0));
const zeroResult = inspect(fixture, fixtureTypes, Array(25).fill(.5), 1, 'Zero-connection H2');
assert.ok(zeroResult.sources.every(source => source.contribution === 0));
assert.ok(zeroResult.pixels.every(pixel => pixel.deltaZ === 0 && pixel.deltaActivation === 0));
// A mixed drawing reuses binary endpoints while evaluating both gray endpoints.
const mixed = square(1, 1, 3); mixed[0] = .5;
inspect(fixture, fixtureTypes, mixed, 0, 'Mixed binary/grayscale input');

const baseInput = square(1, 1, 3), baseForward = forward(baseInput, fixture, fixtureTypes);
const evaluate = (pixels, model) => forward(pixels, model, fixtureTypes);
for (const args of [[null, baseForward, evaluate, 0], [fixture, null, evaluate, 0], [fixture, {}, evaluate, 0],
  [fixture, baseForward, null, 0], [fixture, baseForward, undefined, 0], [fixture, baseForward, evaluate, -1],
  [fixture, baseForward, evaluate, .5], [fixture, baseForward, evaluate, 2], [fixture, baseForward, evaluate, NaN],
  [fixture, baseForward, evaluate, '0']]) assert.equal(spatial.compute(...args), null, 'Malformed requests return unavailable, not invented spatial values');

const malformed = [
  model => { model.topology[0] = 24; },
  model => { model.topology[1] = 4; },
  model => { model.topology.pop(); model.layers.pop(); }, // H2 must be hidden, not the output layer.
  model => { model.layers[0].weights[0].pop(); },
  model => { model.layers[1].weights[0].pop(); },
  model => { model.layers[0].weights[0][0] = Infinity; },
  model => { model.layers[1].weights[0][0] = NaN; },
  model => { model.layers[1].biases[0] = undefined; },
  (_, f) => { f.activations[0].pop(); },
  (_, f) => { f.activations[0][0] = NaN; },
  (_, f) => { f.activations[0][0] = -.1; },
  (_, f) => { f.activations[0][0] = 1.1; },
  (_, f) => { f.activations[1].pop(); },
  (_, f) => { f.activations[1][0] = Infinity; },
  (_, f) => { f.activations[2].pop(); },
  (_, f) => { f.preActivations[1][0] = Infinity; },
  (_, f) => { f.preActivations[1][0] += 1; },
  (_, f) => { delete f.preActivations; }
];
for (const mutate of malformed) {
  const model = clone(fixture), f = clone(baseForward); mutate(model, f);
  assert.equal(spatial.compute(model, f, evaluate, 0), null, `Malformed shape or inconsistent H2 sum rejected: ${mutate}`);
}
for (const invalid of [null, {}, { activations: [], preActivations: [] }]) {
  assert.equal(spatial.compute(fixture, baseForward, () => invalid, 0), null, 'Invalid counterfactual forward is not converted to zero');
}
assert.equal(spatial.compute(fixture, baseForward, pixels => {
  const f = evaluate(pixels, fixture); f.preActivations[1][0] = NaN; return f;
}, 0), null);
assert.equal(spatial.compute(fixture, baseForward, pixels => {
  const f = evaluate(pixels, fixture); f.activations[0][0] += .25; return f;
}, 0), null, 'A callback result for another drawing is rejected');
assert.throws(() => spatial.compute(fixture, baseForward, () => { throw Error('counterfactual failed'); }, 0), /counterfactual failed/);

const browser = {};
vm.runInNewContext(fs.readFileSync(path.join(root, 'second-layer-spatial.js'), 'utf8'), browser);
assert.equal(typeof browser.SecondLayerSpatial.compute, 'function', 'UMD export works without Node globals or DOM');
console.log(`PASS second-layer spatial: ${inspected} real/fixture H2 contexts, ${changedPixels} directly checked pixel interventions; ${baselineEvaluations} supplied baselines + ${interventionEvaluations} callback forwards, full dense/zero source identity, nonlinear counterexample, unchanged parameters/input, invalid-data rejection and browser export.`);
