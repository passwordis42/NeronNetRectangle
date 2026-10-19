/* Pure numerical engine shared by the page, replay worker and Node tests. */
function createSquareBackpropCore() {
  const P = 1351;
  function forward(input, model, types = ["tanh", "linear"]) {
    const activations = [Array.from(input)], preActivations = [];
    model.layers.forEach((layer, k) => {
      const previous = activations[k];
      const z = layer.weights.map((weights, j) => {
        let sum = layer.biases[j];
        for (let i = 0; i < weights.length; i++) sum += weights[i] * previous[i];
        return sum;
      });
      preActivations.push(z);
      activations.push(z.map(v => types[k] === "tanh" ? Math.tanh(v) : types[k] === "sigmoid" ? 1 / (1 + Math.exp(-Math.max(-60, Math.min(60, v)))) : types[k] === "step" ? Number(v >= 1) : v));
    });
    return { activations, preActivations, outputs: activations.at(-1) };
  }
  function model(p) {
    return { topology: [25, 50, 1], layers: [
      { weights: Array.from({ length: 50 }, (_, j) => Array.from(p.slice(j*25, j*25+25))), biases: Array.from(p.slice(1250, 1300)) },
      { weights: [Array.from(p.slice(1300, 1350))], biases: [p[1350]] }
    ] };
  }
  function flatten(m) { return Float64Array.from([...m.layers[0].weights.flat(), ...m.layers[0].biases, ...m.layers[1].weights[0], ...m.layers[1].biases]); }
  function pixels(id) { return Array.from({ length: 25 }, (_, i) => (id >>> i) & 1); }
  function readChunk(buffer) {
    const d = new DataView(buffer); let at = 0;
    const u32 = () => { const v = d.getUint32(at, true); at += 4; return v; };
    const f64 = () => { const v = d.getFloat64(at, true); at += 8; if (!Number.isFinite(v)) throw Error("训练记录含无效数值"); return v; };
    const vector = () => Float64Array.from({ length: P }, f64);
    if (u32() !== 0x31435254) throw Error("训练记录格式不匹配");
    const firstStep = u32(), count = u32();
    if (u32() !== P || count !== 100 || firstStep < 1 || firstStep > 11901 || firstStep % 100 !== 1) throw Error("训练记录维度不匹配");
    const p = vector(), m = vector(), v = vector(), steps = [];
    for (let k = 0; k < count; k++) {
      const size = u32(), positives = u32();
      if (size < 384 || size > 640 || positives !== 128) throw Error("训练批次大小不匹配");
      const losses = Array.from({ length: 4 }, f64), gradient = vector(), ids = Uint32Array.from({ length: size }, u32);
      if (ids.some(id => id >= 33554432)) throw Error("训练样本超出范围");
      steps.push({ step: firstStep+k, positives, losses, gradient, ids });
    }
    const end = { p: vector(), m: vector(), v: vector() };
    if (at !== buffer.byteLength) throw Error("训练记录长度不匹配");
    return { firstStep, p, m, v, steps, end };
  }
  function adam(p, m, v, g, t, opt) {
    const proposed = new Float64Array(P), correctedM = new Float64Array(P), correctedV = new Float64Array(P);
    const clipped = new Uint8Array(P), before = p.slice();
    for (let k = 0; k < P; k++) {
      // Preserve the trainer's in-place multiply/add order.
      m[k] *= opt.beta1; m[k] += (opt.beta1 === .9 ? .1 : 1-opt.beta1) * g[k];
      v[k] *= opt.beta2; v[k] += (opt.beta2 === .999 ? .001 : 1-opt.beta2) * g[k] * g[k];
      correctedM[k] = m[k] / (1 - opt.beta1 ** t);
      correctedV[k] = v[k] / (1 - opt.beta2 ** t);
      proposed[k] = p[k] - opt.learningRate * correctedM[k] / (Math.sqrt(correctedV[k]) + opt.epsilon);
      p[k] = Math.max(opt.bounds[0], Math.min(opt.bounds[1], proposed[k]));
      clipped[k] = Number(p[k] !== proposed[k]);
    }
    return { before, after: p.slice(), proposed, correctedM, correctedV, clipped, m: m.slice(), v: v.slice() };
  }
  function reconstruct(chunk, step, opt) {
    if (step === 0 && chunk.firstStep === 1) return { step, before: chunk.p.slice(), after: chunk.p.slice(), gradient: new Float64Array(P), m: chunk.m.slice(), v: chunk.v.slice(), ids: new Uint32Array(), positives: 0 };
    if (!Number.isInteger(step) || step < chunk.firstStep || step >= chunk.firstStep+chunk.steps.length) throw Error("该数据块不包含所选更新");
    const p = chunk.p.slice(), m = chunk.m.slice(), v = chunk.v.slice();
    let result;
    for (const item of chunk.steps) {
      const oldM = m.slice(), oldV = v.slice();
      const update = adam(p, m, v, item.gradient, item.step, opt);
      if (item.step === step) { result = { ...item, ...update, oldM, oldV }; break; }
    }
    if (step === chunk.firstStep+99) {
      let max = 0;
      for (let k=0;k<P;k++) max = Math.max(max, Math.abs(result.after[k]-chunk.end.p[k]), Math.abs(result.m[k]-chunk.end.m[k]), Math.abs(result.v[k]-chunk.end.v[k]));
      if (max > 1e-9) throw Error("Adam 重建结果未通过校验");
      // Recorded endpoint, before release rounding; never invent a final state.
      result.after = chunk.end.p.slice();
    }
    return result;
  }
  function gradients(p, ids, positives, features, examples, sampleIndex = 0) {
    const parts = Array.from({ length: 4 }, () => new Float64Array(P));
    const losses = [0,0,0,0], net = model(p), n = ids.length;
    const byNeuron = new Map(features.map(f => [f.neuronIndex, f]));
    let sample;
    for (let s=0;s<n;s++) {
      const x = pixels(ids[s]), f = forward(x, net), h = f.activations[1], score = f.outputs[0];
      const error = s < positives ? Math.min(score-1.6,0) : Math.max(score-.4,0), ds = 2*error/n;
      losses[0] += error*error/n;
      parts[0][1350] += ds;
      const dzClass = [], dzConcept = [], targets = [];
      for (let j=0;j<50;j++) {
        parts[0][1300+j] += ds*h[j];
        const dz = ds*p[1300+j]*(1-h[j]*h[j]);
        dzClass.push(dz); parts[0][1250+j] += dz;
        const feature = byNeuron.get(j);
        const target = feature ? (feature.on.every(i => x[i]===1) && feature.off.every(i => x[i]===0) ? .9 : -.9) : null;
        const local = target === null ? 0 : .4*(h[j]-target)*(1-h[j]*h[j])/n;
        targets.push(target); dzConcept.push(local); parts[1][1250+j] += local;
        if (target !== null) losses[1] += .2*(h[j]-target)**2/n;
        for (let i=0;i<25;i++) { parts[0][j*25+i] += dz*x[i]; parts[1][j*25+i] += local*x[i]; }
      }
      if (s === sampleIndex) sample = { index: s, id: ids[s], positive: s<positives, pixels: x, ...f, error, ds, dzClass, dzConcept, targets };
    }
    for (const feature of features) {
      const j=feature.neuronIndex, w=p.slice(j*25,j*25+25), b=p[1250+j];
      const positive = Array.from(w, v => Number(v<0));
      feature.on.forEach(i => positive[i]=1); feature.off.forEach(i => positive[i]=0);
      const z = x => x.reduce((sum,v,i)=>sum+v*w[i], b);
      const ep=Math.min(z(positive)-.55,0);
      losses[2] += ep*ep; parts[2][1250+j] += 2*ep;
      for(let i=0;i<25;i++) parts[2][j*25+i] += 2*ep*positive[i];
      for(const [bit,required] of [...feature.on.map(i=>[i,1]),...feature.off.map(i=>[i,0])]) {
        const x=Array.from(w,v=>Number(v>0)); x[bit]=1-required;
        const en=Math.max(z(x)+.55,0);
        losses[2] += en*en; parts[2][1250+j] += 2*en;
        for(let i=0;i<25;i++) parts[2][j*25+i] += 2*en*x[i];
      }
    }
    for(const example of examples) {
      const x=Array.from({length:25},(_,i)=>Number(example.boundary.includes(i))), h=forward(x,net).activations[1];
      for(const j of example.featureIndices) {
        const err=Math.min(h[j]*p[1300+j]-.14,0), derivative=8*err;
        losses[3] += 4*err*err;
        parts[3][1300+j] += derivative*h[j];
        const dz=derivative*p[1300+j]*(1-h[j]*h[j]);
        parts[3][1250+j] += dz;
        for(let i=0;i<25;i++) parts[3][j*25+i] += x[i]*dz;
      }
    }
    const total = Float64Array.from({length:P},(_,i)=>parts.reduce((sum,a)=>sum+a[i],0));
    return { parts, losses, total, sample };
  }
  function describe(k) {
    if(k<1250) return { index:k, layer:1, neuron:Math.floor(k/25), input:k%25, kind:"weight", label:`输入 ${k%25+1} → H1-${Math.floor(k/25)+1}` };
    if(k<1300) return { index:k, layer:1, neuron:k-1250, kind:"bias", label:`H1-${k-1249} 偏置` };
    if(k<1350) return { index:k, layer:2, neuron:0, input:k-1300, kind:"weight", label:`H1-${k-1299} → 输出` };
    return { index:k, layer:2, neuron:0, kind:"bias", label:"输出偏置" };
  }
  function sampleGradient(sample, parameter) {
    const d=describe(parameter);
    if(d.layer===2) return { classification: sample.ds * (d.kind==="bias"?1:sample.activations[1][d.input]), concept:0 };
    const x=d.kind==="bias"?1:sample.pixels[d.input];
    return { classification: sample.dzClass[d.neuron]*x, concept: sample.dzConcept[d.neuron]*x };
  }
  return { P, forward, model, flatten, pixels, readChunk, adam, reconstruct, gradients, describe, sampleGradient };
}
const SquareBackpropCore = createSquareBackpropCore();
if (typeof module !== "undefined" && module.exports) module.exports = SquareBackpropCore;
