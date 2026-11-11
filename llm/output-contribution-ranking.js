/* Read-only output attribution. Values always come from one actual forward pass. */
(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.OutputContributionRanking = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  function finiteVector(values, size) {
    if (!(Array.isArray(values) || ArrayBuffer.isView(values)) ||
      !Number.isInteger(values.length) || values.length === 0 ||
      size !== undefined && values.length !== size) return false;
    for (let i = 0; i < values.length; i++) if (!Number.isFinite(values[i])) return false;
    return true;
  }

  function compute(model, forward, outputIndex = 0) {
    if (!model || !forward || !Array.isArray(model.layers) || model.layers.length === 0 ||
      !Number.isInteger(outputIndex) || outputIndex < 0 ||
      !Array.isArray(forward.activations) || forward.activations.length !== model.layers.length + 1 ||
      !Array.isArray(forward.preActivations) || forward.preActivations.length !== model.layers.length ||
      !finiteVector(forward.activations[0])) return null;

    const topology = [forward.activations[0].length];
    for (let layerIndex = 0; layerIndex < model.layers.length; layerIndex++) {
      const layer = model.layers[layerIndex];
      if (!layer || !Array.isArray(layer.weights) || !finiteVector(layer.biases, layer.weights.length)) return null;
      for (const weights of layer.weights) if (!finiteVector(weights, topology[layerIndex])) return null;
      const size = layer.weights.length;
      if (!finiteVector(forward.activations[layerIndex + 1], size) ||
        !finiteVector(forward.preActivations[layerIndex], size)) return null;
      topology.push(size);
    }
    if (model.topology !== undefined && (!finiteVector(model.topology, topology.length) ||
      topology.some((size, i) => size !== model.topology[i]))) return null;

    const outputLayer = model.layers.length, last = model.layers[outputLayer - 1];
    if (outputIndex >= last.weights.length || !finiteVector(forward.outputs, last.weights.length)) return null;
    const activations = forward.activations[outputLayer - 1], weights = last.weights[outputIndex];
    const rows = [];
    let positiveSum = 0, negativeSum = 0, weightedSum = 0, maxMagnitude = 0;
    for (let index = 0; index < weights.length; index++) {
      const activation = activations[index], weight = weights[index];
      const product = activation * weight, contribution = product === 0 ? 0 : product;
      if (!Number.isFinite(contribution)) return null;
      const magnitude = Math.abs(contribution);
      rows.push({ index, activation, weight, contribution, magnitude });
      if (contribution > 0) positiveSum += contribution;
      if (contribution < 0) negativeSum += contribution;
      // Sum in connection order, before sorting or display rounding.
      weightedSum += contribution;
      maxMagnitude = Math.max(maxMagnitude, magnitude);
    }
    if (![positiveSum, negativeSum, weightedSum].every(Number.isFinite)) return null;
    rows.sort((a, b) => b.magnitude - a.magnitude || a.index - b.index);
    return { sourceLayer: outputLayer - 1, outputLayer, outputIndex, rows, positiveSum, negativeSum,
      bias: last.biases[outputIndex], weightedSum,
      preActivation: forward.preActivations[outputLayer - 1][outputIndex],
      output: forward.outputs[outputIndex], maxMagnitude };
  }

  function format(value, full = false) {
    if (!Number.isFinite(value)) return '—';
    const normalized = value === 0 ? 0 : value;
    if (full) return String(normalized);
    return normalized !== 0 && Math.abs(normalized) < 1e-6 ? normalized.toExponential(6) : normalized.toFixed(6);
  }

  return { compute, format };
});
