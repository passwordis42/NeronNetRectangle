// node llm/tests/learned-combination-teaching.test.cjs
// H2 meanings must remain unassigned, including when a free model has five units.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { createSquareGeometryTeaching } = require('../geometry-teaching.js');
const { forward } = require('../backprop-core.js');
const root = path.resolve(__dirname, '..');
const legacy = JSON.parse(fs.readFileSync(path.join(root, 'geometry-double-model.json'), 'utf8'));
const source = fs.readFileSync(path.join(root, 'geometry-teaching.js'), 'utf8');
const ID = 'square-trained-learned-combination-v1';

function fixture(width) {
  const a = structuredClone(legacy);
  a.presetId = ID;
  a.learningMode = 'learned-combination';
  a.geometryDefinitions = [];
  a.verification = { errors: null }; // Synthetic parameters are not an accepted model.
  a.model.topology = [25, 119, width, 1];
  a.model.layers[1] = {
    weights: Array.from({ length: width }, (_, j) => Array.from({ length: 119 }, (_, i) => ((i + j) % 9 - 4) / 19)),
    biases: Array.from({ length: width }, (_, j) => (j - 2) / 3)
  };
  a.model.layers[2] = {
    weights: [Array.from({ length: width }, (_, j) => j === 0 ? 0 : j % 2 ? -.125 : .1)],
    biases: [-.2]
  };
  return a;
}

function check(a) {
  const definition = createSquareGeometryTeaching(a, ID, '第一层特征引导＋第二层自主组合');
  const preset = definition.createPreset(), width = a.model.topology[2];
  assert.equal(definition.learnedCombination, true);
  assert.equal(definition.geometry, true);
  assert.equal(definition.double, true);
  assert.equal(definition.matchesPreset(preset), true);
  assert.deepEqual(preset.model, a.model);
  assert.equal(definition.provenance().learningMode, 'learned-combination');
  for (let j = 0; j < width; j++) {
    assert.equal(definition.neuronName(2, j), `H2-${j + 1} · 自主组合`);
    for (const response of [0, .49, .5, 1]) {
      const target = definition.renderGeometryTarget(j, response);
      assert.match(target, /没有预设检测目标/);
      assert.match(target, /只根据最终分类误差学习/);
      assert.match(target, /不监督第二层的目标响应/);
      assert.doesNotMatch(target, /满足此条件|不满足此条件|第二层训练目标|转角数量|水平直线数量|垂直直线数量/);
    }
  }
  const changed = structuredClone(preset);
  changed.model.layers[1].weights[0][0] += .000001;
  assert.equal(definition.matchesPreset(changed), false, 'An imported model cannot reuse the identity with different H2 parameters');
  for (const layers of [[], preset.model.layers.slice(0, 1), [...preset.model.layers, preset.model.layers[0]], null,
    preset.model.layers.map((layer, index) => index === 1 ? null : layer)]) {
    const incomplete = structuredClone(preset); incomplete.model.layers = layers;
    assert.equal(definition.matchesPreset(incomplete), false, 'Missing, extra or invalid layers cannot claim a verified teaching model');
  }
  const foreign = structuredClone(preset); foreign.teachingPreset = legacy.presetId;
  assert.equal(definition.matchesPreset(foreign), false);

  for (let example = 0; example < 6; example++) {
    for (const variant of ['complete', 'missing', 'extra']) {
      const pixels = definition.examplePixels(example, variant), before = JSON.stringify(preset.model);
      const f = forward(pixels, preset.model, a.model.activations);
      const state = { ...preset, pixels, compositionExample: example, geometryOpenGroups: [0, 1, 2, 3] };
      const rendered = definition.renderView(state, f);
      assert.equal(JSON.stringify(preset.model), before);
      assert.match(rendered, /第一层特征引导 · 第二层自主组合/);
      assert.match(rendered, new RegExp(`${width} 个自主组合单元`));
      assert.equal((rendered.match(/class="geometry-gate"/g) || []).length, width);
      assert.equal((rendered.match(/data-geometry-example=/g) || []).length, 6);
      assert.doesNotMatch(rendered, /2 · 五项几何条件|没有完整方框或待补角轮廓神经元/);
      assert.match(rendered, /可能偏向局部组合或完整图案/);
      assert.match(rendered, /不指定神经元含义、连接分组或权重正负/);
      assert.match(rendered, new RegExp(`展开全部 ${width} 项输出贡献`));
      const terms = f.activations[2].map((value, j) => value * preset.model.layers[2].weights[0][j]);
      assert.ok(Math.abs(terms.reduce((s, v) => s + v, 0) + preset.model.layers[2].biases[0] - f.outputs[0]) < 1e-12);
      for (const term of terms) {
        const formatted = term !== 0 && Math.abs(term) < .0000005 ? term.toExponential(6) : term.toFixed(6);
        assert.ok(rendered.includes(`贡献 ${formatted}`), 'Every displayed H2 contribution follows the real output connection');
      }
    }
  }

  const context = vm.createContext({ SquareLearnedCombinationModelData: a });
  vm.runInContext(source, context);
  assert.equal(vm.runInContext('SquareLearnedCombinationTeaching.ID', context), ID);
  assert.equal(vm.runInContext('SquareLearnedCombinationTeaching.learnedCombination', context), true);
}

for (const width of [5, 8, 16]) check(fixture(width));
const modelFile = path.join(root, 'learned-combination-model.json');
if (fs.existsSync(modelFile)) check(JSON.parse(fs.readFileSync(modelFile, 'utf8')));

const old = createSquareGeometryTeaching(legacy, legacy.presetId, 'legacy');
assert.equal(old.learnedCombination, false);
assert.match(old.neuronName(2, 1), /转角数量不少于四个/);
assert.match(old.renderGeometryTarget(0, .9), /满足此条件/);
assert.match(old.renderGeometryTarget(0, .1), /不满足此条件/);
console.log('PASS learned-combination teaching: variable 5/8/16 H2 identities, no invented conditions, complete real signed contributions, six examples, model identity, browser export and legacy meanings.');
