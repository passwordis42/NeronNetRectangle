/* Pure, topology-independent geometry replay engine. No DOM or network access. */
function createGeometryBackpropCore(manifest, artifact) {
  'use strict';
  if (!manifest || !artifact || !artifact.model) throw Error('缺少模型或训练清单');
  const topology = Array.from(manifest.topology || artifact.model.topology);
  const activations = Array.from(manifest.activations || artifact.activations || artifact.model.activations ||
    topology.slice(1).map((_, k) => k === topology.length - 2 ? 'linear' : 'sigmoid'));
  const specs = manifest.parameterMap || artifact.parameterMap;
  if (!Array.isArray(specs) || !specs.length) throw Error('训练清单缺少有效参数映射');
  const P = specs.length, lookup = new Map();
  const parameterMap = specs.map((entry, index) => {
    const d = { ...entry, index, input: entry.kind === 'bias' ? null : entry.input };
    if (entry.index !== undefined && entry.index !== index || d.lower !== undefined && d.upper !== undefined && d.lower > d.upper ||
      !Number.isInteger(d.layer) || d.layer < 1 || d.layer >= topology.length ||
      !Number.isInteger(d.neuron) || d.neuron < 0 || d.neuron >= topology[d.layer] ||
      !['weight', 'bias'].includes(d.kind) || d.kind === 'weight' &&
      (!Number.isInteger(d.input) || d.input < 0 || d.input >= topology[d.layer - 1])) throw Error('参数映射维度不正确');
    const key = `${d.layer}:${d.neuron}:${d.kind}:${d.input}`;
    if (lookup.has(key)) throw Error('参数映射重复');
    lookup.set(key, index);
    const from = d.layer === 1 ? `X${d.input + 1}` : `H${d.layer - 1}-${d.input + 1}`;
    const to = d.layer === topology.length - 1 ? `输出 ${d.neuron + 1}` : `H${d.layer}-${d.neuron + 1}`;
    d.label ||= d.kind === 'bias' ? `${to} 偏置` : `${from} → ${to}`;
    return d;
  });
  if (artifact.model.topology.length !== topology.length || artifact.model.topology.some((n, i) => n !== topology[i]) ||
    artifact.model.layers.length !== topology.length - 1) throw Error('训练清单与正式模型拓扑不一致');
  artifact.model.layers.forEach((layer, k) => {
    if (layer.weights.length !== topology[k + 1] || layer.biases.length !== topology[k + 1]) throw Error('正式模型层尺寸无效');
    layer.weights.forEach((weights, j) => {
      if (weights.length !== topology[k]) throw Error('正式模型连接尺寸无效');
      weights.forEach((value, i) => {
        if (!Number.isFinite(value) || !lookup.has(`${k + 1}:${j}:weight:${i}`) && value !== 0) throw Error('固定连接必须严格为零');
      });
      if (!Number.isFinite(layer.biases[j]) || !lookup.has(`${k + 1}:${j}:bias:null`) && layer.biases[j] !== 0) throw Error('未记录的偏置必须严格为零');
    });
  });
  const defaultLosses = [
    ['classification', '分类误差'], ['local', '局部特征监督'], ['geometry', '几何条件监督'],
    ['boundary', '检测边界约束'], ['participation', '参与约束']
  ];
  const lossDefinitions = (manifest.lossDefinitions || manifest.losses || defaultLosses.map(([id, name]) => ({ id, name })))
    .map((loss, index) => ({ ...loss, index }));
  const tolerance = manifest.numericTolerance || { absolute: 1e-8, relative: 1e-10 };
  function checkVector(value, name, size = P) {
    if (!value || value.length !== size || !Array.from(value).every(Number.isFinite)) throw Error(`${name} 维度或数值无效`);
    return Float64Array.from(value);
  }
  function close(a, b) {
    return Math.abs(a - b) <= (tolerance.absolute ?? 1e-8) + (tolerance.relative ?? 1e-10) * Math.max(Math.abs(a), Math.abs(b));
  }
  function compareVectors(a, b, name) {
    let error = 0;
    for (let i = 0; i < P; i++) {
      error = Math.max(error, Math.abs(a[i] - b[i]));
      if (!close(a[i], b[i])) throw Error(`${name} 未通过校验：参数 ${i}，误差 ${a[i] - b[i]}`);
    }
    return error;
  }
  function model(parameters) {
    const p = checkVector(parameters, '参数');
    const net = { topology: topology.slice(), layers: topology.slice(1).map((size, layer) => ({
      weights: Array.from({ length: size }, () => Array(topology[layer]).fill(0)), biases: Array(size).fill(0)
    })) };
    for (const d of parameterMap) {
      if (d.kind === 'bias') net.layers[d.layer - 1].biases[d.neuron] = p[d.index];
      else net.layers[d.layer - 1].weights[d.neuron][d.input] = p[d.index];
    }
    return net;
  }
  function flatten(net) {
    return Float64Array.from(parameterMap, d => d.kind === 'bias' ? net.layers[d.layer - 1].biases[d.neuron] : net.layers[d.layer - 1].weights[d.neuron][d.input]);
  }
  function forward(input, net, types = activations) {
    if (!net) net = artifact.model;
    if (input.length !== net.topology[0]) throw Error('前向输入维度不匹配');
    const values = [Array.from(input)], preActivations = [];
    net.layers.forEach((layer, k) => {
      const previous = values[k], z = layer.weights.map((weights, j) => {
        let sum = layer.biases[j];
        for (let i = 0; i < weights.length; i++) sum += weights[i] * previous[i];
        return sum;
      });
      preActivations.push(z);
      values.push(z.map(v => types[k] === 'sigmoid' ? 1 / (1 + Math.exp(-Math.max(-60, Math.min(60, v))))
        : types[k] === 'tanh' ? Math.tanh(v) : types[k] === 'step' ? Number(v >= 1) : v));
    });
    return { activations: values, preActivations, outputs: values[values.length - 1] };
  }
  function pixels(id) { return Array.from({ length: topology[0] }, (_, i) => (id >>> i) & 1); }
  function describe(index) {
    if (!Number.isInteger(index) || index < 0 || index >= P) throw Error('参数编号超出范围');
    return { ...parameterMap[index] };
  }
  function indexOf(layer, neuron, input, kind = 'weight') { return lookup.get(`${layer}:${neuron}:${kind}:${kind === 'bias' ? null : input}`) ?? -1; }
  function optimizerAt(step) {
    const base = manifest.optimizer || {}, phases = base.phases || manifest.phases || [];
    const phase = phases.find(p => Math.max(1, step) >= p.firstStep && Math.max(1, step) <= p.lastStep);
    const opt = { beta1: .9, beta2: .999, epsilon: 1e-8, ...base, ...(phase || {}) };
    if (opt.betas) [opt.beta1, opt.beta2] = opt.betas;
    opt.lossWeights ||= lossDefinitions.map(d => d.coefficient ?? 1);
    if (!Array.isArray(opt.lossWeights)) opt.lossWeights = lossDefinitions.map(d => opt.lossWeights[d.id] ?? 0);
    return opt;
  }
  function adam(p, m, v, gradient, step, override) {
    const g = checkVector(gradient, '梯度'), opt = { ...optimizerAt(step), ...(override || {}) };
    if (!(step >= 1) || !(opt.learningRate > 0)) throw Error('Adam 更新编号或学习率无效');
    const before = Float64Array.from(p), oldM = Float64Array.from(m), oldV = Float64Array.from(v);
    const proposed = new Float64Array(P), correctedM = new Float64Array(P), correctedV = new Float64Array(P), clipped = new Uint8Array(P);
    for (let i = 0; i < P; i++) {
      m[i] *= opt.beta1; m[i] += (opt.beta1 === .9 ? .1 : 1 - opt.beta1) * g[i];
      v[i] *= opt.beta2; v[i] += (opt.beta2 === .999 ? .001 : 1 - opt.beta2) * g[i] * g[i];
      correctedM[i] = m[i] / (1 - opt.beta1 ** step);
      correctedV[i] = v[i] / (1 - opt.beta2 ** step);
      proposed[i] = p[i] - opt.learningRate * correctedM[i] / (Math.sqrt(correctedV[i]) + opt.epsilon);
      const d = parameterMap[i], lower = d.lower ?? opt.bounds?.[0] ?? -Infinity, upper = d.upper ?? opt.bounds?.[1] ?? Infinity;
      p[i] = Math.max(lower, Math.min(upper, proposed[i]));
      clipped[i] = Number(p[i] !== proposed[i]);
    }
    return { before, after: Float64Array.from(p), oldM, oldV, m: Float64Array.from(m), v: Float64Array.from(v), proposed, correctedM, correctedV, clipped, optimizer: opt };
  }
  function readChunk(input) {
    const value = typeof input === 'string' ? JSON.parse(input) : input;
    if (value?.format !== 'geometry-trace-v1' || !Number.isInteger(value.firstStep) || value.firstStep < 1 ||
      !Array.isArray(value.updates) || value.updates.length !== value.count || value.count < 1) throw Error('训练记录格式无效');
    const state = (s, name) => ({ parameters: checkVector(s?.parameters, `${name}参数`), m: checkVector(s?.m, `${name}一阶矩`), v: checkVector(s?.v, `${name}二阶矩`) });
    const updates = value.updates.map((s, k) => {
      if (s.step !== value.firstStep + k || !Array.isArray(s.ids) || !s.ids.length || !Array.isArray(s.labels) || s.ids.length !== s.labels.length ||
        s.ids.some(id => !Number.isInteger(id) || id < 0 || id >= 2 ** topology[0]) || s.labels.some(x => x !== 0 && x !== 1)) throw Error('训练批次或更新顺序无效');
      return { ...s, ids: Uint32Array.from(s.ids), labels: Uint8Array.from(s.labels), gradient: checkVector(s.gradient, '梯度'), losses: checkVector(s.losses, '损失', lossDefinitions.length) };
    });
    return { ...value, start: state(value.start, '块起点'), end: state(value.end, '块终点'), updates };
  }
  function reconstruct(chunk, step) {
    if (step === 0 && chunk.firstStep === 1) return { step: 0, before: chunk.start.parameters.slice(), after: chunk.start.parameters.slice(),
      oldM: chunk.start.m.slice(), oldV: chunk.start.v.slice(), m: chunk.start.m.slice(), v: chunk.start.v.slice(), gradient: null,
      ids: new Uint32Array(), labels: new Uint8Array(), losses: null, proposed: null, correctedM: null, correctedV: null, clipped: null };
    if (!Number.isInteger(step) || step < chunk.firstStep || step >= chunk.firstStep + chunk.count) throw Error('该数据块不包含所选更新');
    const p = chunk.start.parameters.slice(), m = chunk.start.m.slice(), v = chunk.start.v.slice();
    let result;
    for (const update of chunk.updates) {
      const r = adam(p, m, v, update.gradient, update.step);
      if (update.step === step) { result = { ...update, ...r }; break; }
    }
    if (step === chunk.firstStep + chunk.count - 1) {
      result.endpointError = Math.max(compareVectors(p, chunk.end.parameters, '块终点参数'), compareVectors(m, chunk.end.m, '块终点一阶矩'), compareVectors(v, chunk.end.v, '块终点二阶矩'));
    }
    return result;
  }
  const features = artifact.featureDefinitions || artifact.features || manifest.featureDefinitions || [];
  const config = manifest.lossConfig || artifact.lossConfig || artifact.training?.lossConfig || {};
  const localCoefficient = config.local?.coefficient ?? .2, geometryCoefficient = config.geometry?.coefficient ?? .2;
  const localMargin = config.boundary?.localMargin ?? 8, geometryMargin = config.boundary?.geometryMargin ?? 4;
  const boundaryCoefficient = config.boundary?.coefficient ?? 1, participationCoefficient = config.participation?.coefficient ?? 1;
  const participationMinimum = config.participation?.minimum ?? .02;
  const sourceIndex = id => typeof id === 'number' ? id : lossDefinitions.findIndex(d => d.id === id);
  const layerParameters = topology.slice(1).map((_, k) => parameterMap.filter(d => d.layer === k + 1));
  const hasGeometry = topology.length === 4 && topology[2] === 5, finalLayer = topology.length - 1;
  const formulas = {
    classification: 'L分类 = Σ max(正例下界 − s, 0)² / N（正例）；Σ max(s − 负例上界, 0)² / N（负例）',
    local: `L局部 = ${localCoefficient} × Σ(h − t)² / (N × F)`,
    geometry: `L几何 = ${geometryCoefficient} × Σ(g − t几何)² / (N × G)`,
    boundary: `L边界 = ${boundaryCoefficient} × [Σ局部最坏输入的间隔误差² / F + Σ几何输入和的间隔误差² / (N × G)]`,
    participation: `L参与 = ${participationCoefficient} × Σ min(方向 × 输出权重 − ${participationMinimum}, 0)² / M`
  };
  const normalizations = { classification: 'N 为本次实际训练批次大小', local: 'N × F：样本数乘第一隐藏层目标数',
    geometry: 'N × G：样本数乘第二隐藏层的 5 个几何条件', boundary: '局部最坏输入项除以 F；样本几何边界项除以 N × G',
    participation: 'M 为输出层可训练权重连接数，与批次大小无关' };
  lossDefinitions.forEach(d => { d.formula ||= formulas[d.id] || ''; d.coefficient ??= 1; d.normalization ||= normalizations[d.id] || ''; });
  function activationDerivative(value, layer) {
    return activations[layer - 1] === 'sigmoid' ? value * (1 - value) : activations[layer - 1] === 'tanh' ? 1 - value * value : 1;
  }
  function targetValues(x) {
    return features.map(feature => Number(feature.on.every(i => x[i] === 1) && feature.off.every(i => x[i] === 0)));
  }
  function geometryTargets(x) {
    let points = 0, legal = 0, horizontal = 0, vertical = 0, corners = 0;
    for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) {
      const i = r * 5 + c;
      if (!x[i]) continue;
      points++;
      const u = r > 0 ? x[i - 5] : 0, d = r < 4 ? x[i + 5] : 0;
      const l = c > 0 ? x[i - 1] : 0, rr = c < 4 ? x[i + 1] : 0;
      if (u + d + l + rr !== 2) continue;
      legal++;
      if (l && rr) horizontal++;
      else if (u && d) vertical++;
      else corners++;
    }
    return [Number(legal === points), Number(corners >= 4), Number(corners <= 4), Number(horizontal >= vertical), Number(vertical >= horizontal)];
  }
  function blankDeltas() { return topology.slice(1).map(n => new Float64Array(n)); }
  function backward(deltas, f, net) {
    const g = new Float64Array(P);
    for (let k = finalLayer - 1; k >= 0; k--) {
      const dz = deltas[k], previous = f.activations[k];
      for (const d of layerParameters[k]) {
        const delta = dz[d.neuron];
        if (delta === 0) continue;
        if (d.kind === 'bias') g[d.index] += delta;
        else {
          g[d.index] += delta * previous[d.input];
          if (k) deltas[k - 1][d.input] += delta * net.layers[k].weights[d.neuron][d.input] * activationDerivative(previous[d.input], k);
        }
      }
    }
    return g;
  }
  function addVector(target, source, scale = 1) { for (let i = 0; i < P; i++) target[i] += source[i] * scale; }
  function batchOf(batch) {
    if (!batch?.ids?.length || !batch.labels || batch.ids.length !== batch.labels.length) throw Error('缺少实际训练批次及标签');
    return batch;
  }
  function sampleTerms(f, x, label, n) {
    const terms = [], truth = targetValues(x), geo = hasGeometry ? geometryTargets(x) : null;
    const score = f.outputs[0], target = label ? config.classification?.positive ?? 1.6 : config.classification?.negative ?? .4;
    const error = label ? Math.min(score - target, 0) : Math.max(score - target, 0);
    terms.push({ lossIndex: sourceIndex('classification'), layer: finalLayer, neuron: 0, preActivation: false,
      response: score, target, error, scale: 1, hinge: label ? 'min' : 'max', unaveragedLoss: error * error, derivative: 2 * error, denominator: n, label: '分类间隔误差' });
    features.forEach((feature, j) => {
      const neuron = feature.neuronIndex, response = f.activations[1][neuron], e = response - truth[j];
      terms.push({ lossIndex: sourceIndex('local'), layer: 1, neuron, preActivation: false, response, target: truth[j], error: e,
        scale: localCoefficient, unaveragedLoss: localCoefficient * e * e, derivative: 2 * localCoefficient * e, denominator: n * features.length, label: feature.name || `H1-${neuron + 1} 局部目标` });
    });
    if (geo) geo.forEach((target, neuron) => {
      const response = f.activations[2][neuron], e = response - target;
      terms.push({ lossIndex: sourceIndex('geometry'), layer: 2, neuron, preActivation: false, response, target, error: e,
        scale: geometryCoefficient, unaveragedLoss: geometryCoefficient * e * e, derivative: 2 * geometryCoefficient * e, denominator: n * geo.length, label: `H2-${neuron + 1} 几何目标` });
      const z = f.preActivations[1][neuron], logitTarget = target ? geometryMargin : -geometryMargin;
      const marginError = target ? Math.min(z - logitTarget, 0) : Math.max(z - logitTarget, 0);
      terms.push({ lossIndex: sourceIndex('boundary'), layer: 2, neuron, preActivation: true, response: z, target: logitTarget, binaryTarget: target, error: marginError,
        scale: boundaryCoefficient, hinge: target ? 'min' : 'max', unaveragedLoss: boundaryCoefficient * marginError * marginError, derivative: 2 * boundaryCoefficient * marginError, denominator: n * geo.length, label: `H2-${neuron + 1} 几何检测边界` });
    });
    return { terms, truth, geometryTargets: geo };
  }
  function localBoundaryTerms(net, selectedNeuron) {
    const terms = [], F = features.length;
    for (const feature of features) {
      const neuron = feature.neuronIndex;
      if (selectedNeuron !== undefined && neuron !== selectedNeuron) continue;
      const w = net.layers[0].weights[neuron], b = net.layers[0].biases[neuron];
      const add = (x, matches, bit = null, required = null) => {
        const z = x.reduce((sum, value, i) => sum + value * w[i], b), target = matches ? localMargin : -localMargin;
        const error = matches ? Math.min(z - target, 0) : Math.max(z - target, 0);
        terms.push({ lossIndex: sourceIndex('boundary'), layer: 1, neuron, preActivation: true, pixels: x, response: z, target, error,
          scale: boundaryCoefficient, hinge: matches ? 'min' : 'max', unaveragedLoss: boundaryCoefficient * error * error, derivative: 2 * boundaryCoefficient * error, denominator: F,
          label: `${feature.name || `H1-${neuron + 1}`} · ${matches ? '满足目标的最小输入和' : `X${bit + 1} 违反目标的最大输入和`}`, constraint: 'local-boundary', matches, bit, required });
      };
      const xp = w.map(value => Number(value < 0));
      feature.on.forEach(i => { xp[i] = 1; }); feature.off.forEach(i => { xp[i] = 0; }); add(xp, true);
      for (const [bit, required] of [...feature.on.map(i => [i, 1]), ...feature.off.map(i => [i, 0])]) {
        const xn = w.map(value => Number(value > 0)); xn[bit] = 1 - required; add(xn, false, bit, required);
      }
    }
    return terms;
  }
  function participationTerms(parameters) {
    const outputs = parameterMap.filter(d => d.layer === finalLayer && d.kind === 'weight');
    return outputs.map(d => {
      const sign = d.lower >= 0 ? 1 : d.upper <= 0 ? -1 : d.sign ?? 1;
      const response = parameters[d.index] * sign, error = Math.min(response - participationMinimum, 0);
      return { lossIndex: sourceIndex('participation'), parameter: d.index, label: `${d.label} · 输出参与下界`, constraint: 'participation', sign,
        response, target: participationMinimum, error, unaveragedLoss: participationCoefficient * error * error,
        derivative: 2 * participationCoefficient * error * sign, denominator: outputs.length };
    });
  }
  function gradients(parameters, batch, step, sampleIndex = 0) {
    batchOf(batch);
    const net = model(parameters), K = lossDefinitions.length, n = batch.ids.length;
    const rawParts = Array.from({ length: K }, () => new Float64Array(P)), rawLosses = Array(K).fill(0);
    const weights = optimizerAt(step).lossWeights;
    let sample;
    for (let index = 0; index < n; index++) {
      const id = batch.ids[index], x = pixels(id), label = Number(batch.labels[index]), f = forward(x, net);
      const { terms, truth, geometryTargets: geo } = sampleTerms(f, x, label, n);
      const deltas = Array.from({ length: K }, blankDeltas);
      for (const term of terms) {
        rawLosses[term.lossIndex] += term.unaveragedLoss / term.denominator;
        deltas[term.lossIndex][term.layer - 1][term.neuron] += term.derivative / term.denominator *
          (term.preActivation ? 1 : activationDerivative(f.activations[term.layer][term.neuron], term.layer));
      }
      const sampleParts = deltas.map(d => backward(d, f, net));
      sampleParts.forEach((part, k) => addVector(rawParts[k], part));
      if (index === sampleIndex) sample = { index, id, pixels: x, label, positive: Boolean(label), ...f, targets: truth, geometryTargets: geo,
        terms, parts: sampleParts, lossWeights: weights.slice(), batchSize: n, _model: net, _parameters: Array.from(parameters) };
    }
    const boundary = sourceIndex('boundary');
    for (const term of localBoundaryTerms(net)) {
      rawLosses[boundary] += term.unaveragedLoss / term.denominator;
      for (const d of layerParameters[0]) if (d.neuron === term.neuron) rawParts[boundary][d.index] += term.derivative * (d.kind === 'bias' ? 1 : term.pixels[d.input]) / term.denominator;
    }
    const participation = sourceIndex('participation');
    for (const term of participationTerms(parameters)) {
      rawLosses[participation] += term.unaveragedLoss / term.denominator;
      rawParts[participation][term.parameter] += term.derivative / term.denominator;
    }
    const parts = rawParts.map((part, k) => Float64Array.from(part, value => value * weights[k]));
    const total = new Float64Array(P); parts.forEach(part => addVector(total, part));
    return { parts, rawParts, total, losses: rawLosses.map((value, k) => value * weights[k]), rawLosses, lossWeights: weights, sample };
  }
  function downstream(f, net, d, term) {
    if (term.layer < d.layer || term.layer === d.layer && term.neuron !== d.neuron) return { value: 0, expression: '0（没有依赖路径）' };
    if (term.layer === d.layer) return { value: 1, expression: '1（同一神经元）' };
    let values = new Float64Array(topology[d.layer]); values[d.neuron] = 1;
    let expressions = Array(topology[d.layer]).fill('0'); expressions[d.neuron] = '1';
    for (let layer = d.layer + 1; layer <= term.layer; layer++) {
      const next = new Float64Array(topology[layer]), names = [];
      for (let j = 0; j < next.length; j++) {
        const pieces = [];
        for (let i = 0; i < values.length; i++) {
          const w = net.layers[layer - 1].weights[j][i];
          if (values[i] !== 0 && w !== 0) { next[j] += w * values[i]; pieces.push(`${w} × (${expressions[i]})`); }
        }
        const derivative = layer === term.layer && term.preActivation ? 1 : activationDerivative(f.activations[layer][j], layer);
        next[j] *= derivative;
        names[j] = `(${pieces.join(' + ') || '0'}) × ${derivative}`;
      }
      values = next; expressions = names;
    }
    return { value: values[term.neuron], expression: expressions[term.neuron] };
  }
  function termRow(term, f, net, parameter, coefficient, extra = {}) {
    const d = describe(parameter), input = d.kind === 'bias' ? 1 : f.activations[d.layer - 1][d.input];
    const directPreActivation = term.layer === d.layer && term.preActivation;
    const activationSlope = directPreActivation ? 1 : activationDerivative(f.activations[d.layer][d.neuron], d.layer);
    const following = downstream(f, net, d, term), rawDerivative = term.derivative * following.value * activationSlope * input;
    const rawGradient = rawDerivative / term.denominator, gradient = rawGradient * coefficient;
    const parameterSymbol = d.kind === 'bias' ? `b^(${d.layer})_${d.neuron + 1}` : `w^(${d.layer})_(${d.neuron + 1},${d.input + 1})`;
    const activationSymbol = d.layer === finalLayer ? 's' : `a^(${d.layer})_${d.neuron + 1}`;
    const zSymbol = `z^(${d.layer})_${d.neuron + 1}`;
    const sourceSymbol = term.preActivation ? `z^(${term.layer})_${term.neuron + 1}` : term.layer === finalLayer ? 's' : `a^(${term.layer})_${term.neuron + 1}`;
    const errorFormula = term.hinge ? `e = ${term.hinge}(响应 − 目标, 0)` : 'e = 响应 − 目标';
    const errorSubstitution = term.hinge ? `${term.hinge}(${term.response} − (${term.target}), 0)` : `${term.response} − (${term.target})`;
    const chain = [
      { formula: errorFormula, substitution: errorSubstitution, value: term.error },
      { formula: '本项损失 = c × e² / 归一化分母', substitution: `${term.scale} × (${term.error})² ÷ ${term.denominator}`, value: term.unaveragedLoss / term.denominator },
      { formula: `∂ℓ / ∂${sourceSymbol} = 2 × c × e（尚未归一化）`, substitution: `2 × ${term.scale} × (${term.error})`, value: term.derivative },
      { formula: `∂${sourceSymbol} / ∂${directPreActivation ? zSymbol : activationSymbol}（所有后续路径求和）`, substitution: following.expression, value: following.value },
      { formula: directPreActivation ? `∂${zSymbol} / ∂${zSymbol} = 1；本项不经过激活函数` : `∂${activationSymbol} / ∂${zSymbol}`, substitution: directPreActivation ? '1' : activations[d.layer - 1] === 'sigmoid' ? `${f.activations[d.layer][d.neuron]} × (1 − ${f.activations[d.layer][d.neuron]})` : `${activationSlope}`, value: activationSlope },
      { formula: `∂${zSymbol} / ∂${parameterSymbol} = ${d.kind === 'bias' ? '1' : d.layer === 1 ? `x_${d.input + 1}` : `a^(${d.layer - 1})_${d.input + 1}`}`, substitution: `${input}`, value: input },
      { formula: `∂ℓ / ∂${parameterSymbol}（单项未归一化）`, substitution: `${term.derivative} × ${following.value} × ${activationSlope} × ${input} = ${rawDerivative}`, value: rawDerivative },
      { formula: '对本次总梯度的贡献', substitution: `${rawDerivative} ÷ ${term.denominator} × ${coefficient} = ${gradient}`, value: gradient }
    ];
    return { ...term, ...extra, sourceId: lossDefinitions[term.lossIndex].id, parameterSymbol, sourceSymbol, input, activationDerivative: activationSlope,
      lossDerivative: term.derivative, downstreamDerivative: following.value, rawDerivative, rawGradient, gradient, weightedGradient: gradient,
      coefficient, divisor: term.denominator, loss: term.unaveragedLoss / term.denominator, effectiveLoss: term.unaveragedLoss / term.denominator * coefficient, chain };
  }
  function sampleGradient(sample, parameter, lossId) {
    const k = sourceIndex(lossId);
    if (!sample || k < 0) return null;
    const relevant = sample.terms.filter(t => t.lossIndex === k && t.layer >= describe(parameter).layer &&
      (t.layer !== describe(parameter).layer || t.neuron === describe(parameter).neuron));
    const coefficient = sample.lossWeights[k], denominator = sample.terms.find(t => t.lossIndex === k)?.denominator ?? null;
    const rows = relevant.map(t => termRow(t, sample, sample._model, parameter, coefficient, { index: sample.index, id: sample.id, pixels: sample.pixels }));
    const raw = rows.reduce((sum, row) => sum + row.rawDerivative, 0);
    const contribution = sample.parts[k][parameter] * coefficient;
    return { raw: denominator === null ? null : raw, coefficient, denominator, contribution, rows,
      sourceId: lossDefinitions[k].id, constraintOnly: denominator === null,
      note: lossDefinitions[k].id === 'boundary' ? '这里仅显示本样本的几何边界项；局部最坏输入约束另见完整来源。' : denominator === null ? '本损失不按训练样本定义，请查看约束项来源。' : '',
      chain: rows.length === 1 ? rows[0].chain : [
        { formula: '本样本所有相关项的偏导之和', substitution: `${rows.map(r => r.rawDerivative).join(' + ') || '0'} = ${raw}`, value: raw },
        { formula: '本样本对批次梯度的贡献', substitution: denominator === null ? '非样本约束：没有样本偏导' : `${raw} ÷ ${denominator} × ${coefficient} = ${contribution}`, value: denominator === null ? null : contribution }
      ] };
  }
  function parameterTrace(parameters, batch, step, parameter) {
    batchOf(batch);
    const d = describe(parameter), net = model(parameters), n = batch.ids.length, weights = optimizerAt(step).lossWeights;
    const rows = lossDefinitions.map(() => []);
    for (let index = 0; index < n; index++) {
      const id = batch.ids[index], x = pixels(id), label = Number(batch.labels[index]), f = forward(x, net);
      const { terms } = sampleTerms(f, x, label, n);
      for (const term of terms) {
        if (term.layer < d.layer || term.layer === d.layer && term.neuron !== d.neuron) continue;
        rows[term.lossIndex].push(termRow(term, f, net, parameter, weights[term.lossIndex], { index, id, pixels: x, positive: Boolean(label) }));
      }
    }
    if (d.layer === 1) for (const term of localBoundaryTerms(net, d.neuron)) {
      const f = forward(term.pixels, net);
      rows[term.lossIndex].push(termRow(term, f, net, parameter, weights[term.lossIndex]));
    }
    for (const term of participationTerms(parameters)) if (term.parameter === parameter) {
      const coefficient = weights[term.lossIndex], rawDerivative = term.derivative, rawGradient = rawDerivative / term.denominator, gradient = coefficient * rawGradient;
      rows[term.lossIndex].push({ ...term, coefficient, divisor: term.denominator, rawDerivative, rawGradient, gradient, weightedGradient: gradient,
        loss: term.unaveragedLoss / term.denominator, effectiveLoss: coefficient * term.unaveragedLoss / term.denominator,
        chain: [{ formula: 'e = min(方向 × 权重 − 参与下界, 0)', substitution: `min(${term.sign} × ${parameters[parameter]} − ${participationMinimum}, 0)`, value: term.error },
          { formula: '本项损失 = c × e² / 输出连接数', substitution: `${participationCoefficient} × (${term.error})² ÷ ${term.denominator}`, value: term.unaveragedLoss / term.denominator },
          { formula: '∂ℓ参与 / ∂w = 2 × c × 间隔误差 × 方向', substitution: `${2 * participationCoefficient} × ${term.error} × ${term.sign} = ${rawDerivative}`, value: rawDerivative },
          { formula: '对总梯度的贡献', substitution: `${rawDerivative} ÷ ${term.denominator} × ${coefficient} = ${gradient}`, value: gradient }] });
    }
    const rawGradients = rows.map(items => items.reduce((sum, row) => sum + row.rawGradient, 0));
    const parts = rawGradients.map((value, k) => value * weights[k]);
    return { parameter: d, batchSize: n, lossWeights: weights, rows, rawGradients, parts, total: parts.reduce((sum, value) => sum + value, 0) };
  }
  function findNonzeroSample(parameters, batch, step, parameter, lossId, start = 0) {
    const k = sourceIndex(lossId);
    if (k < 0 || lossDefinitions[k].id === 'participation') return -1;
    for (let offset = 1; offset <= batch.ids.length; offset++) {
      const index = (start + offset) % batch.ids.length;
      const result = gradients(parameters, { ids: [batch.ids[index]], labels: [batch.labels[index]] }, step, 0);
      if (result.sample.parts[k][parameter] * result.lossWeights[k] !== 0) return index;
    }
    return -1;
  }
  return { P, topology, activations, parameterMap, lossDefinitions, model, flatten, forward, pixels, describe, indexOf,
    optimizerAt, adam, readChunk, reconstruct, compareVectors, gradients, sampleGradient, parameterTrace, findNonzeroSample, geometryTargets };
}
const GeometryBackpropCore = { create: createGeometryBackpropCore };
if (typeof module !== 'undefined' && module.exports) module.exports = GeometryBackpropCore;
