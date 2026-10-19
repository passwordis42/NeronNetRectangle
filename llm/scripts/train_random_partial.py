"""Reproduce the random-start curriculum and record every real update.

No inherited parameters or shaped initialization. The fixed hidden permutation
is applied to the model and the entire history together, preserving UI names.
"""
import argparse
import base64
import copy
import gzip
import hashlib
import json
import struct
import time
import train_composition as base
import train_partial_features as partial
from shuffle_partial_features import shuffled_artifact
import numpy as np

DEST = base.ROOT / "training-trace" / "v2"
ARTIFACT = base.ROOT / "partial-features-random-model.json"
ORDER = json.loads((base.ROOT / "partial-features-shuffled-model.json").read_text(encoding="utf-8"))["neuronOrder"]["newToOld"]
PHASES = [
    dict(firstStep=1, lastStep=6000, name="局部特征学习", learningRate=.01, lossWeights=[0, 1, 1, 0]),
    dict(firstStep=6001, lastStep=18000, name="分类与组合联合学习", learningRate=.003, lossWeights=[1, 1, 1, 1]),
]
STAGES = [0, 100, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 18000]


def phase_at(step):
    return PHASES[0 if step <= 6000 else 1]


def flatten(params):
    w, b, v, c = params
    return np.concatenate([w[:, ORDER].T.ravel(), b[ORDER], v[ORDER], c]).astype("<f8")


def unflatten(vector):
    return [vector[:1250].reshape(50, 25).T.copy(), vector[1250:1300].copy(), vector[1300:1350].copy(), vector[1350:].copy()]


class Recorder:
    def __init__(self):
        DEST.mkdir(parents=True, exist_ok=True)
        self.chunks, self.timeline = [], []
        self.checked, self.max_error = 0, 0.
        inverse = {old: new for new, old in enumerate(ORDER)}
        self.features, self.examples = copy.deepcopy((partial.FEATURES, partial.EXAMPLES))
        for f in self.features:
            f["neuronIndex"] = inverse[f["neuronIndex"]]
        for e in self.examples:
            e["featureIndices"] = [inverse[i] for i in e["featureIndices"]]
        self.ex = np.array([[float(p in e["boundary"]) for p in range(25)] for e in self.examples])

    def before_update(self, step, x, params, grads, moments, variances, losses):
        if (step - 1) % 100 == 0:
            self.start = step
            self.buffer = bytearray(struct.pack("<4sIII", b"TRC1", step, 100, 1351))
            for state in (params, moments, variances):
                self.buffer.extend(flatten(state).tobytes())
        ids = (x.astype(np.uint32) * (np.uint32(1) << np.arange(25, dtype=np.uint32))).sum(axis=1).astype("<u4")
        self.buffer.extend(struct.pack("<II4d", len(x), 128, *losses))
        self.buffer.extend(flatten(grads).tobytes())
        self.buffer.extend(ids.tobytes())
        self.timeline.append([step, *losses])

    def after_update(self, step, params, moments, variances):
        if step % 100:
            return
        for state in (params, moments, variances):
            self.buffer.extend(flatten(state).tobytes())
        raw = bytes(self.buffer)
        self.verify_chunk(raw)
        packed = gzip.compress(raw, compresslevel=9, mtime=0)
        index = len(self.chunks)
        name = f"chunk-{index:03d}.js"
        (DEST / name).write_text(f'SquareTraceChunks.receive("v2",{index},"{base64.b64encode(packed).decode()}");\n', encoding="ascii")
        self.chunks.append(dict(index=index, firstStep=self.start, lastStep=step, file=name,
                               sha256=hashlib.sha256(packed).hexdigest(), bytes=len(packed)))

    def verify_chunk(self, raw):
        _, first, count, _ = struct.unpack_from("<4sIII", raw)
        offset = 16
        def vector():
            nonlocal offset
            a = np.frombuffer(raw, dtype="<f8", count=1351, offset=offset).copy()
            offset += 1351 * 8
            return a
        params, moments, variances = vector(), vector(), vector()
        for step in range(first, first + count):
            n, _, *losses = struct.unpack_from("<II4d", raw, offset)
            offset += 40
            recorded = vector()
            ids = np.frombuffer(raw, dtype="<u4", count=n, offset=offset)
            offset += n * 4
            phase = phase_at(step)
            actual_loss, actual = base.loss_and_gradients(unflatten(params), base.pixels(ids), self.features,
                                                         self.examples, self.ex, phase["lossWeights"])
            actual = np.concatenate([actual[0].T.ravel(), actual[1], actual[2], actual[3]])
            error = float(np.max(np.abs(actual - recorded)))
            self.max_error = max(self.max_error, error)
            assert error < 1e-9 and np.allclose(actual_loss, losses, atol=1e-10, rtol=1e-10)
            moments *= .9
            moments += .1 * recorded
            variances *= .999
            variances += .001 * recorded * recorded
            params -= phase["learningRate"] * (moments / (1 - .9**step)) / (np.sqrt(variances / (1 - .999**step)) + 1e-8)
            np.clip(params, -10, 10, out=params)
            self.checked += 1
        assert all(np.array_equal(a, vector()) for a in (params, moments, variances))
        assert offset == len(raw)

    def finish(self, artifact):
        assert self.checked == 18000 and len(self.chunks) == 180
        manifest = dict(version=2, id="partial-50-v5-backprop-v2", dataVersion="v2", dataPath="training-trace/v2/",
            provenance="recorded during deterministic random-initialization training", presetId="square-trained-partial-50-v5",
            modelSha256=artifact["modelSha256"], initialization=artifact["training"]["initialization"], seed=42, dataSeed=20261019,
            steps=18000, parameterCount=1351, chunkSize=100, stages=STAGES, phases=PHASES, precision=6, dtype="float64-le",
            parameterLayout=["W1[50][25]", "b1[50]", "W2[50]", "b2[1]"],
            optimizer=dict(name="Adam", learningRate=.003, beta1=.9, beta2=.999, epsilon=1e-8, bounds=[-10,10], phases=PHASES),
            neuronOrder=artifact["neuronOrder"], chunks=self.chunks, timeline=self.timeline,
            verification=dict(stepsChecked=self.checked, maximumGradientError=self.max_error,
                              finalModelExactMatch=True, exhaustiveErrors=0))
        (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        a = json.loads(ARTIFACT.read_text(encoding="utf-8")); p = base.arrays(a["model"])
        assert base.verify_features(p, a["featureDefinitions"])["passed"]
        assert base.verify_compositions(p, a["featureDefinitions"], a["compositionExamples"])["passed"]
        assert base.exhaustive(p)[0]["errors"] == 0
        return
    features, examples = partial.FEATURES, partial.EXAMPLES
    ex = np.array([[float(p in e["boundary"]) for p in range(25)] for e in examples])
    negids = base.structured_ids(); neg = base.pixels(negids)
    rng = np.random.default_rng(42)
    params = [rng.uniform(-.1, .1, (25, 50)), np.zeros(50), rng.uniform(-.1, .1, 50), np.zeros(1)]
    initial = [p.copy() for p in params]
    moments, variances = [[np.zeros_like(p) for p in params] for _ in range(2)]
    rng = np.random.default_rng(20261019)
    ids = rng.integers(1 << 25, size=131072, dtype=np.uint32)
    ids = ids[~np.isin(ids, base.POS_IDS)]
    check_ids = np.unique(np.concatenate([negids, ids])); check = base.pixels(check_ids)
    hard, hard_ids, history = None, set(), []
    recorder = Recorder(); start = time.time()
    for step in range(1, 18001):
        ids = rng.integers(1 << 25, size=128, dtype=np.uint32)
        ids = ids[~np.isin(ids, base.POS_IDS)]
        parts = [base.POS_X[rng.integers(30, size=128)], neg[rng.integers(len(neg), size=256)], base.pixels(ids)]
        if hard is not None:
            parts.append(hard[rng.integers(len(hard), size=128)])
        x = np.concatenate(parts); phase = phase_at(step)
        losses, grads = base.loss_and_gradients(params, x, features, examples, ex, phase["lossWeights"])
        recorder.before_update(step, x, params, grads, moments, variances, losses)
        base.adam_update(params, grads, moments, variances, step, phase["learningRate"])
        recorder.after_update(step, params, moments, variances)
        if step % 2000 == 0:
            rounded = base.arrays(base.model_of(params))
            _, scores = base.forward(check, rounded)
            indices = np.flatnonzero(scores > .55)
            indices = indices[np.argsort(scores[indices])[-4096:]] if len(indices) else indices
            hard_ids.update(int(v) for v in check_ids[indices])
            hard = base.pixels(sorted(hard_ids)) if hard_ids else None
            print(f"step={step} elapsed={time.time()-start:.1f}s losses={losses}", flush=True)
        if step in (12000, 18000):
            verification, mined = base.exhaustive(rounded)
            hard_ids.update(mined); hard = base.pixels(sorted(hard_ids))
            history.append(dict(step=step, errors=verification["errors"], hardNegativeCount=len(hard_ids)))
    assert verification["errors"] == 0
    model = base.model_of(params)
    canonical = dict(model=model, featureDefinitions=features, compositionExamples=examples,
        modelSha256=hashlib.sha256(json.dumps(model, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
        training=dict(seed=42, dataSeed=20261019, steps=18000, optimizer="Adam", learningRate=.003,
            betas=[.9,.999], epsilon=1e-8, precision=6, parameterBounds=[-10,10], trainableParameters=1351,
            initializedFeatures=0, supervisedFeatures=25, freeFeatures=25, phases=PHASES,
            source="random weights; supervised local-feature curriculum, then joint classification and composition training",
            initialization=dict(method="uniform", bounds=[-.1,.1], weightSeed=42, biases="zero", inheritedParameters=False),
            parameterChangeL2=[float(np.linalg.norm(p-i)) for p,i in zip(params,initial)],
            structuredNegativeExamples=len(negids), screeningNegativeExamples=len(check_ids), history=history, numpyVersion=np.__version__))
    artifact = shuffled_artifact(canonical)
    artifact["neuronOrder"]["sourcePresetId"] = "random-training-canonical-order"
    p = base.arrays(artifact["model"])
    artifact["featureVerification"] = base.verify_features(p, artifact["featureDefinitions"])
    artifact["compositionVerification"] = base.verify_compositions(p, artifact["featureDefinitions"], artifact["compositionExamples"])
    artifact["verification"], _ = base.exhaustive(p)
    assert artifact["featureVerification"]["passed"] and artifact["compositionVerification"]["passed"] and artifact["verification"]["errors"] == 0
    artifact["permutationVerification"] = dict(passed=True, method="same fixed bijection applied to parameters and all recorded updates")
    ARTIFACT.write_text(json.dumps(artifact, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    recorder.finish(artifact)
    print("VERIFIED", artifact["modelSha256"], "recorded updates", recorder.checked, "gradient error", recorder.max_error, flush=True)


if __name__ == "__main__":
    main()
