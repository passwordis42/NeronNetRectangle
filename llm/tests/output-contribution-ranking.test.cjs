const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ranking = require('../output-contribution-ranking.js');
const forward = require('../backprop-core.js').forward;
const close = (a, b, epsilon = 1e-10) => assert.ok(Math.abs(a - b) <= epsilon, `${a} != ${b}`);
const clone = value => JSON.parse(JSON.stringify(value));
function freeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const child of Object.values(value)) freeze(child);
  }
  return value;
}
function verifyIdentity(result) {
  close(result.rows.reduce((sum, row) => sum + row.contribution, 0), result.weightedSum);
  close(result.positiveSum + result.negativeSum, result.weightedSum);
  close(result.weightedSum + result.bias, result.preActivation);
  assert.ok(result.positiveSum >= 0 && result.negativeSum <= 0);
  for (let i = 1; i < result.rows.length; i++) {
    const previous = result.rows[i - 1], current = result.rows[i];
    assert.ok(previous.magnitude > current.magnitude || previous.magnitude === current.magnitude && previous.index < current.index);
  }
}

// Mixed signs: a negative hidden response times a negative weight is support.
// Equal absolute contributions retain neuron order, and zero rows remain present.
const model = {
  topology: [5, 5, 2], layers: [
    { weights: Array.from({ length: 5 }, (_, j) => Array.from({ length: 5 }, (_, i) => Number(i === j))), biases: [0, 0, 0, 0, 0] },
    { weights: [[-2, -2, 7, 2, 2], [1, 0, 0, 0, 0]], biases: [.25, -.75] }
  ]
};
const f = forward([-.5, .5, 0, -.5, .5], model, ['linear', 'linear']);
const beforeModel = clone(model), beforeForward = clone(f);
freeze(model); freeze(f);
const result = ranking.compute(model, f);
assert.equal(result.sourceLayer, 1); assert.equal(result.outputLayer, 2); assert.equal(result.outputIndex, 0);
assert.deepEqual(result.rows.map(row => row.index), [0, 1, 3, 4, 2]);
assert.deepEqual(result.rows[0], { index: 0, activation: -.5, weight: -2, contribution: 1, magnitude: 1 });
assert.equal(result.positiveSum, 2); assert.equal(result.negativeSum, -2);
assert.equal(result.weightedSum, 0); assert.equal(result.bias, .25); assert.equal(result.maxMagnitude, 1);
assert.equal(result.rows.at(-1).contribution, 0);
verifyIdentity(result);
const second = ranking.compute(model, f, 1);
assert.equal(second.outputIndex, 1); assert.equal(second.output, -1.25); assert.equal(second.rows.length, 5);
verifyIdentity(second);
assert.deepEqual(model, beforeModel); assert.deepEqual(f, beforeForward);

// Zero hidden layers use input pixels directly; nonlinear output is not the input sum.
const direct = { topology: [2, 1], layers: [{ weights: [[1, -2]], biases: [.5] }] };
const directForward = forward([1, 1], direct, ['sigmoid']);
const directResult = ranking.compute(direct, directForward);
assert.equal(directResult.sourceLayer, 0); assert.equal(directResult.outputLayer, 1);
assert.equal(directResult.preActivation, -.5); close(directResult.output, 1 / (1 + Math.exp(.5)));
assert.notEqual(directResult.output, directResult.preActivation); verifyIdentity(directResult);

// Ranking uses unrounded values even when both values look identical at six decimals.
const precise = { topology: [3, 1], layers: [{ weights: [[1.00000011, 1.00000012, -1.00000012]], biases: [0] }] };
const preciseResult = ranking.compute(precise, forward([1, 1, 1], precise, ['linear']));
assert.equal(ranking.format(preciseResult.rows[0].contribution), '1.000000');
assert.deepEqual(preciseResult.rows.map(row => row.index), [1, 2, 0]);

const zeros = { topology: [2, 1], layers: [{ weights: [[0, -0]], biases: [0] }] };
const zeroResult = ranking.compute(zeros, forward([1, 1], zeros, ['linear']));
assert.deepEqual(zeroResult.rows.map(row => row.index), [0, 1]);
assert.equal(zeroResult.rows.length, 2); assert.equal(zeroResult.maxMagnitude, 0);
assert.equal(Object.is(zeroResult.rows[1].contribution, -0), false);

assert.equal(ranking.format(1.23456789), '1.234568');
assert.equal(ranking.format(-1.23456789), '-1.234568');
assert.equal(ranking.format(1e-6), '0.000001');
assert.equal(ranking.format(1e-7), '1.000000e-7');
assert.equal(ranking.format(-1e-12), '-1.000000e-12');
assert.equal(ranking.format(0), '0.000000'); assert.equal(ranking.format(-0), '0.000000');
assert.equal(ranking.format(-0, true), '0');
assert.equal(ranking.format(1.0000000000000002, true), '1.0000000000000002');
for (const value of [null, undefined, NaN, Infinity, -Infinity, '1']) assert.equal(ranking.format(value), '—');

// Missing, malformed or nonfinite inputs never become fabricated zero contributions.
for (const [m, state, index] of [[null, f, 0], [model, null, 0], [{ layers: [] }, f, 0], [model, {}, 0],
  [model, f, -1], [model, f, .5], [model, f, 2], [model, f, NaN], [model, f, '0']]) assert.equal(ranking.compute(m, state, index), null);
const mutations = [
  (m, state) => { delete state.outputs; },
  (m, state) => { delete state.preActivations; },
  (m, state) => { state.activations.pop(); },
  (m, state) => { state.activations[1].pop(); },
  (m, state) => { state.preActivations[1][0] = NaN; },
  (m, state) => { state.outputs[0] = Infinity; },
  (m, state) => { state.outputs.pop(); },
  (m, state) => { state.activations[1][0] = NaN; },
  m => { m.layers[1].weights[0][0] = Infinity; },
  m => { m.layers[1].biases[0] = undefined; },
  m => { m.layers[1].weights[0].pop(); },
  m => { m.layers[1].weights[0] = []; },
  m => { delete m.layers[1].weights[0][0]; },
  m => { m.topology[1] = 99; }
];
for (const mutate of mutations) {
  const m = clone(model), state = clone(f); mutate(m, state);
  assert.equal(ranking.compute(m, state), null);
}
const overflow = { topology: [2, 1], layers: [{ weights: [[Number.MAX_VALUE, Number.MAX_VALUE]], biases: [0] }] };
const overflowForward = { activations: [[1, 1], [1]], preActivations: [[1]], outputs: [1] };
assert.equal(ranking.compute(overflow, overflowForward), null);
const typed = { topology: [2, 1], layers: [{ weights: [new Float64Array([1, 2])], biases: new Float64Array([0]) }] };
const typedForward = forward([1, 1], typed, ['linear']);
assert.equal(ranking.compute(typed, typedForward).weightedSum, 3);

// The browser-global export works without Node globals or DOM objects.
const context = {};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../output-contribution-ranking.js'), 'utf8'), context);
assert.equal(typeof context.OutputContributionRanking.compute, 'function');
assert.equal(context.OutputContributionRanking.format(-0), '0.000000');

// Use the real shared forward() for both released geometries: 352 and 5 sources.
let actualInputsChecked = 0;
for (const [kind, expectedCount, expectedSource] of [['single', 352, 1], ['double', 5, 2]]) {
  const artifact = JSON.parse(fs.readFileSync(path.join(__dirname, `../geometry-${kind}-model.json`), 'utf8'));
  const original = JSON.stringify(artifact.model); freeze(artifact.model);
  const inputs = [Array(25).fill(0), Array(25).fill(1), Array.from({ length: 25 }, (_, i) => Number(i % 3 === 0))];
  for (let side = 2; side <= 5; side++) for (let row = 0; row <= 5 - side; row++) for (let col = 0; col <= 5 - side; col++) {
    const pixels = Array(25).fill(0);
    for (let r = 0; r < side; r++) for (let c = 0; c < side; c++) if (r === 0 || c === 0 || r === side - 1 || c === side - 1) pixels[(row + r) * 5 + col + c] = 1;
    inputs.push(pixels);
  }
  for (const input of inputs) {
    const actualForward = forward(input, artifact.model, artifact.activations || artifact.model.activations);
    const actual = ranking.compute(artifact.model, actualForward);
    assert.ok(actual); assert.equal(actual.rows.length, expectedCount); assert.equal(actual.sourceLayer, expectedSource);
    actual.rows.forEach(item => {
      assert.equal(item.activation, actualForward.activations.at(-2)[item.index]);
      assert.equal(item.weight, artifact.model.layers.at(-1).weights[0][item.index]);
      assert.equal(item.contribution, item.activation * item.weight);
    });
    verifyIdentity(actual); assert.equal(actual.output, actualForward.outputs[0]);
    assert.equal(actual.preActivation, actualForward.preActivations.at(-1)[0]); actualInputsChecked++;
  }
  assert.equal(JSON.stringify(artifact.model), original);
}
console.log(`output contribution ranking tests passed: ${actualInputsChecked} real-model forwards, complete sign/dimension/precision coverage`);
