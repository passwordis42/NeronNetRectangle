"""Reproduce the released model and record every actual Adam update.

Does not overwrite released models/checkpoints. The trace is publishable only
after the complete replay matches the original model and every chunk verifies.
"""
import base64
import gzip
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import train_composition as base
import train_partial_features as partial
import numpy as np

DEST = base.ROOT / "training-trace" / "v1"
ARTIFACT = json.loads((base.ROOT / "partial-features-shuffled-model.json").read_text(encoding="utf-8"))
ORDER = ARTIFACT["neuronOrder"]["newToOld"]
COUNT = 1351


def flatten(params):
    w, b, v, c = params
    return np.concatenate([w[:, ORDER].T.ravel(), b[ORDER], v[ORDER], c]).astype("<f8")


def unflatten(vector):
    return [vector[:1250].reshape(50, 25).T.copy(), vector[1250:1300].copy(),
            vector[1300:1350].copy(), vector[1350:].copy()]


class Recorder:
    def __init__(self):
        DEST.mkdir(parents=True, exist_ok=True)
        self.chunks, self.timeline = [], []
        self.checked = 0
        self.max_error = 0.

    def before_update(self, step, x, params, grads, moments, variances, losses):
        if (step - 1) % 100 == 0:
            self.start = step
            self.buffer = bytearray(struct.pack("<4sIII", b"TRC1", step, 100, COUNT))
            for state in (params, moments, variances):
                self.buffer.extend(flatten(state).tobytes())
        # Targets are explicitly encoded as ordered positive/negative groups.
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
        (DEST / name).write_text(f'SquareTraceChunks.receive({index},"{base64.b64encode(packed).decode()}");\n', encoding="ascii")
        self.chunks.append(dict(index=index, firstStep=self.start, lastStep=step,
            file=name, sha256=hashlib.sha256(packed).hexdigest(), bytes=len(packed)))

    def verify_chunk(self, raw):
        magic, first, count, width = struct.unpack_from("<4sIII", raw)
        assert magic == b"TRC1" and width == COUNT
        offset = 16
        def vector():
            nonlocal offset
            a = np.frombuffer(raw, dtype="<f8", count=COUNT, offset=offset).copy()
            offset += COUNT * 8
            return a
        params, moments, variances = vector(), vector(), vector()
        features, examples = ARTIFACT["featureDefinitions"], ARTIFACT["compositionExamples"]
        ex = np.array([[float(p in e["boundary"]) for p in range(25)] for e in examples])
        for step in range(first, first+count):
            n, positives, *losses = struct.unpack_from("<II4d", raw, offset)
            offset += 40
            recorded = vector()
            ids = np.frombuffer(raw, dtype="<u4", count=n, offset=offset)
            offset += n*4
            assert positives == 128
            actual_loss, actual = base.loss_and_gradients(unflatten(params), base.pixels(ids), features, examples, ex)
            actual = np.concatenate([actual[0].T.ravel(), actual[1], actual[2], actual[3]])
            error = float(np.max(np.abs(actual-recorded)))
            self.max_error = max(error, self.max_error)
            assert error < 1e-9 and np.allclose(actual_loss, losses, atol=1e-10, rtol=1e-10)
            # Match the original in-place operation order exactly.
            moments *= .9
            moments += .1 * recorded
            variances *= .999
            variances += .001 * recorded * recorded
            params -= .003*(moments/(1-.9**step))/(np.sqrt(variances/(1-.999**step))+1e-8)
            np.clip(params, -10, 10, out=params)
            self.checked += 1
        assert all(np.array_equal(a, vector()) for a in (params, moments, variances))
        assert offset == len(raw)

    def finish(self):
        manifest = dict(version=1, id="partial-50-v4-backprop-v1", provenance="deterministic training reproduction",
            presetId="square-trained-partial-50-v4", modelSha256=ARTIFACT["modelSha256"],
            sourceModelSha256=ARTIFACT["training"]["sourceModelSha256"], seed=42, steps=12000,
            parameterCount=COUNT, chunkSize=100, stages=list(range(0,12001,1200)),
            precision=6, dtype="float64-le", parameterLayout=["W1[50][25]", "b1[50]", "W2[50]", "b2[1]"],
            optimizer=dict(name="Adam", learningRate=.003, beta1=.9, beta2=.999, epsilon=1e-8, bounds=[-10,10]),
            neuronOrder=ARTIFACT["neuronOrder"], chunks=self.chunks, timeline=self.timeline,
            verification=dict(stepsChecked=self.checked, maximumGradientError=self.max_error,
                              finalModelExactMatch=True, exhaustiveErrors=0))
        assert self.checked == 12000 and len(self.chunks) == 120
        (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))+"\n", encoding="utf-8")
        print("TRACE VERIFIED", self.checked, "updates; maximum gradient error", self.max_error,
              "; compressed bytes", sum(c["bytes"] for c in self.chunks), flush=True)


def main():
    recorder = Recorder()
    expected = json.loads(partial.ARTIFACT.read_text(encoding="utf-8"))
    recorded = base.train(SimpleNamespace(seed=42, steps=6000, rounds=2, learning_rate=.003, resume=False),
        features=partial.FEATURES, examples=partial.EXAMPLES,
        source=json.loads(partial.SOURCE.read_text(encoding="utf-8")), initialized_features=13,
        artifact_path=None, save_checkpoint=False, recorder=recorder,
        source_description=expected["training"]["source"])
    assert recorded["model"] == expected["model"]
    assert recorded["modelSha256"] == expected["modelSha256"]
    recorder.finish()


if __name__ == "__main__":
    main()
