// Recompute every recorded gradient/loss using the browser's independent JS engine.
// node llm/tests/audit-learned-combination.cjs [manifest.json] [model.json] [verification.json]
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');
const zlib = require('node:zlib');
const { Worker, isMainThread, parentPort, workerData } = require('node:worker_threads');
const Core = require('../geometry-backprop-core.js');
const defaults = [path.resolve(__dirname, '../training-trace/learned-combination-v1/manifest.json'),
  path.resolve(__dirname, '../learned-combination-model.json'), path.resolve(__dirname, '../learned-combination-backprop-verification.json')];
const maximumDifference = (a, b) => {
  assert.equal(a.length, b.length);
  let maximum = 0;
  for (let i = 0; i < a.length; i++) { assert.ok(Number.isFinite(a[i]) && Number.isFinite(b[i])); maximum = Math.max(maximum, Math.abs(a[i] - b[i])); }
  return maximum;
};
function read(entry, manifest, directory, core) {
  const source = fs.readFileSync(path.join(directory, entry.file), 'utf8');
  const match = /SquareTraceChunks\.receive\("([^"\n]+)",\s*(\d+),\s*"([A-Za-z0-9+/=]+)"\)/.exec(source);
  assert.ok(match, `Invalid trace wrapper ${entry.file}`);
  assert.equal(match[1], manifest.id); assert.equal(+match[2], entry.index);
  const packed = Buffer.from(match[3], 'base64');
  assert.equal(packed.byteLength, entry.bytes);
  assert.equal(crypto.createHash('sha256').update(packed).digest('hex'), entry.sha256);
  const chunk = core.readChunk(zlib.gunzipSync(packed).toString('utf8'));
  assert.equal(chunk.firstStep, entry.firstStep); assert.equal(chunk.firstStep + chunk.count - 1, entry.lastStep);
  return chunk;
}
function audit(entry, manifest, artifact, directory) {
  const core = Core.create(manifest, artifact), chunk = read(entry, manifest, directory, core);
  assert.equal(core.learnedCombination, true); assert.equal(core.hasGeometry, false); assert.equal(core.hasGeometryBoundary, false);
  const p = chunk.start.parameters.slice(), m = chunk.start.m.slice(), v = chunk.start.v.slice();
  let maxGradientError = 0, maxLossError = 0, clipCount = 0;
  for (const update of chunk.updates) {
    const details = core.gradients(p, update, update.step);
    maxGradientError = Math.max(maxGradientError, maximumDifference(details.total, update.gradient));
    maxLossError = Math.max(maxLossError, maximumDifference(details.losses, update.losses));
    assert.ok(maxGradientError <= 1e-8, `Gradient mismatch at update ${update.step}: ${maxGradientError}`);
    assert.ok(maxLossError <= 1e-8, `Loss mismatch at update ${update.step}: ${maxLossError}`);
    for (const k of [2, 4]) { assert.equal(details.rawLosses[k], 0); assert.equal(update.losses[k], 0); }
    for (const d of core.parameterMap) if (d.layer >= 2) {
      assert.equal(d.lower, -128); assert.equal(d.upper, 128);
      for (const k of [1, 2, 3, 4]) assert.equal(details.parts[k][d.index], 0, 'Auxiliary target leaked into a free-layer gradient');
      assert.equal(details.total[d.index], details.parts[0][d.index]);
    }
    const result = core.adam(p, m, v, update.gradient, update.step);
    for (const d of core.parameterMap) {
      const expected = Math.max(d.lower, Math.min(d.upper, result.proposed[d.index]));
      assert.equal(p[d.index], expected); assert.equal(result.clipped[d.index], Number(expected !== result.proposed[d.index]));
      clipCount += result.clipped[d.index];
    }
  }
  const maxParameterError = maximumDifference(p, chunk.end.parameters), maxMomentError = maximumDifference(m, chunk.end.m), maxVarianceError = maximumDifference(v, chunk.end.v);
  assert.ok(Math.max(maxParameterError, maxMomentError, maxVarianceError) <= 1e-8);
  return { index: entry.index, firstStep: entry.firstStep, lastStep: entry.lastStep, count: chunk.count,
    start: chunk.start, end: chunk.end, maxGradientError, maxLossError, maxParameterError, maxMomentError, maxVarianceError, clipCount };
}
if (!isMainThread) {
  const { manifest, artifact, directory } = workerData;
  parentPort.on('message', entry => {
    try { parentPort.postMessage({ result: audit(entry, manifest, artifact, directory) }); }
    catch (error) { parentPort.postMessage({ error: error.stack || String(error) }); }
  });
} else {
  (async () => {
    const [manifestFile, artifactFile, outputFile] = defaults.map((fallback, i) => path.resolve(process.argv[i + 2] || fallback));
    const manifest = JSON.parse(fs.readFileSync(manifestFile)), artifact = JSON.parse(fs.readFileSync(artifactFile));
    const core = Core.create(manifest, artifact), directory = path.dirname(manifestFile), started = Date.now();
    assert.equal(manifest.id, 'learned-combination-v1'); assert.equal(manifest.modelSha256, artifact.modelSha256);
    assert.equal(artifact.verification.errors, 0); assert.equal(artifact.verification.inputCount, 2 ** 25);
    assert.equal(manifest.initialization.inheritedParameters, false);
    assert.deepEqual(core.lossDefinitions.filter(d => core.lossEnabled(d.id)).map(d => d.id), ['classification', 'local', 'boundary']);
    const workerCount = Math.min(manifest.chunks.length, Math.max(1, Number(process.env.LEARNED_AUDIT_WORKERS) || Math.min(6, os.availableParallelism?.() || os.cpus().length)));
    const results = Array(manifest.chunks.length), workers = [];
    let next = 0, completed = 0;
    try {
      await new Promise((resolve, reject) => {
        for (let i = 0; i < workerCount; i++) {
          const worker = new Worker(__filename, { workerData: { manifest, artifact, directory } }); workers.push(worker);
          worker.on('error', reject);
          worker.on('message', ({ result, error }) => {
            if (error) { reject(Error(error)); return; }
            results[result.index] = result; completed++;
            if (completed % 5 === 0 || completed === results.length) console.log(`Audited ${completed}/${results.length} chunks (${((Date.now() - started) / 1000).toFixed(1)}s)`);
            if (completed === results.length) resolve(); else if (next < manifest.chunks.length) worker.postMessage(manifest.chunks[next++]);
          });
          worker.postMessage(manifest.chunks[next++]);
        }
      });
    } finally { await Promise.all(workers.map(worker => worker.terminate())); }
    let updates = 0;
    results.forEach((result, index) => {
      assert.equal(result.index, index); assert.equal(result.firstStep, updates + 1); updates += result.count;
      if (index) for (const key of ['parameters', 'm', 'v']) assert.deepEqual(result.start[key], results[index - 1].end[key], 'Exact block continuity');
      else for (const d of core.parameterMap) {
        const initial = result.start.parameters[d.index];
        if (d.kind === 'bias') assert.equal(initial, 0);
        else assert.ok(initial >= -.1 && initial <= .1);
        assert.equal(result.start.m[d.index], 0); assert.equal(result.start.v[d.index], 0);
      }
    });
    assert.equal(updates, manifest.steps);
    assert.deepEqual(Array.from(results.at(-1).end.parameters, x => Number(x.toFixed(6))), Array.from(core.flatten(artifact.model)));
    const maxima = Object.fromEntries(['maxGradientError', 'maxLossError', 'maxParameterError', 'maxMomentError', 'maxVarianceError'].map(key => [key, Math.max(...results.map(result => result[key]))]));
    const verification = { format: 'learned-combination-backprop-verification-v1', id: manifest.id, modelSha256: artifact.modelSha256,
      passed: true, updatesChecked: updates, gradientsRecomputed: updates, lossesRecomputed: updates,
      parametersPerUpdate: core.P, adamParameterUpdatesChecked: updates * core.P, clippingChecks: updates * core.P,
      clippingEvents: results.reduce((sum, result) => sum + result.clipCount, 0), chunksChecked: results.length,
      initializationChecked: true, chunkContinuityExactlyMatch: true, publishedSixDecimalParametersExactlyMatch: true,
      noSecondLayerAuxiliaryGradients: true, ...maxima, absoluteTolerance: 1e-8, seconds: (Date.now() - started) / 1000 };
    fs.writeFileSync(outputFile, JSON.stringify(verification, null, 2) + '\n');
    console.log(JSON.stringify(verification));
  })().catch(error => { console.error(error); process.exitCode = 1; });
}
