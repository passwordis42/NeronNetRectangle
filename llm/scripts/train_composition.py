"""Train and verify the 25 -> 50 -> 1 compositional square teaching model.

Requires Python + NumPy. Run from any directory:
  python llm/scripts/train_composition.py
  python llm/scripts/train_composition.py --verify-only

The original embedded guided model is the immutable initialization source.
Only a fully verified, six-decimal artifact is exported. No browser inference
uses the labels, concept masks or example definitions generated here.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "神经网络教学.html"
ARTIFACT = ROOT / "composition-model.json"
CHECKPOINT = ROOT / ".composition-checkpoint.npz"
BITS = np.arange(25, dtype=np.uint32)


def pixels(ids):
    return ((np.asarray(ids, dtype=np.uint32).reshape(-1, 1) >> BITS) & 1).astype(np.float64)


def square(size, row, col):
    top = [row * 5 + col + k for k in range(size)]
    bottom = [i + (size - 1) * 5 for i in top]
    left = [(row + k) * 5 + col for k in range(size)]
    right = [i + size - 1 for i in left]
    return [top, bottom, left, right]


def definitions():
    features = [
        dict(neuronIndex=0, name="横线片段", position="第 2 行，第 2～4 列", on=[6, 7, 8], off=[], kind="edge"),
        dict(neuronIndex=1, name="竖线片段", position="第 2 列，第 2～4 行", on=[6, 11, 16], off=[], kind="edge"),
        dict(neuronIndex=2, name="左上转角", position="第 2～3 行，第 2～3 列", on=[6, 7, 11], off=[12], kind="corner"),
    ]
    examples = []
    for size, row, col, name, candidate in [(3, 1, 1, "居中 3×3", 20), (4, 0, 0, "左上 4×4", 25), (5, 0, 0, "外圈 5×5", 29)]:
        edges = square(size, row, col)
        indices = []
        for side, points in zip(["上边", "下边", "左边", "右边"], edges):
            match = next((f for f in features if f["on"] == points and not f["off"]), None)
            if match is None:
                match = dict(neuronIndex=len(features), name=f"{name} · {side}",
                             position="、".join(f"{p // 5 + 1}行{p % 5 + 1}列" for p in points), on=points, off=[], kind="edge")
                features.append(match)
            indices.append(match["neuronIndex"])
        boundary = sorted(set(sum(edges, [])))
        examples.append(dict(name=name, size=size, row=row, col=col, candidateIndex=candidate,
                             featureIndices=indices, boundary=boundary,
                             missingPixel=edges[0][size // 2], extraPixel=(row + 1) * 5 + col + 1))
    assert len(features) == 13
    return features, examples


FEATURES, EXAMPLES = definitions()
POS_IDS = np.array([sum(1 << p for p in set(sum(square(s, r, c), [])))
                    for s in range(2, 6) for r in range(6 - s) for c in range(6 - s)], dtype=np.uint32)
POS_SET = set(map(int, POS_IDS))
POS_X = pixels(POS_IDS)
EXAMPLE_X = np.array([[float(p in e["boundary"]) for p in range(25)] for e in EXAMPLES])
EDGE_INDICES = [f["neuronIndex"] for f in FEATURES if f["kind"] == "edge"]


def load_source():
    text = HTML.read_text(encoding="utf-8-sig")
    return json.loads(re.search(r"const SquareGuidedModelData = (\{[^\r\n]*\});", text)[1])


def arrays(model):
    a, b = model["layers"]
    return [np.array(a["weights"]).T.copy(), np.array(a["biases"]).copy(),
            np.array(b["weights"][0]).copy(), np.array(b["biases"]).copy()]


def model_of(params):
    w, b, v, c = [np.round(p, 6) for p in params]
    return dict(topology=[25, 50, 1], layers=[dict(weights=w.T.tolist(), biases=b.tolist()),
                                           dict(weights=[v.tolist()], biases=c.tolist())])


def forward(x, params):
    w, b, v, c = params
    h = np.tanh(x @ w + b)
    return h, h @ v + c[0]


def structured_ids():
    ids = set()
    # All one/two-bit perturbations, and three-bit perturbations of positives.
    masks = [sum(1 << p for p in comb) for n in (1, 2, 3) for comb in itertools.combinations(range(25), n)]
    for original in POS_SET:
        ids.update(original ^ mask for mask in masks)
    # Every axis-aligned rectangular outline and filled rectangle.
    for height in range(1, 6):
        for width in range(1, 6):
            for r in range(6 - height):
                for c in range(6 - width):
                    filled = [i * 5 + j for i in range(r, r + height) for j in range(c, c + width)]
                    outline = [p for p in filled if p // 5 in (r, r + height - 1) or p % 5 in (c, c + width - 1)]
                    ids.add(sum(1 << p for p in filled))
                    ids.add(sum(1 << p for p in outline))
    ids.update([0, (1 << 25) - 1])
    return np.array(sorted(ids - POS_SET), dtype=np.uint32)


def worst_inputs(w, feature):
    """Exact extrema for a linear concept detector, including every background.

    Each absent pattern differs in at least one constrained bit. Maximizing
    over the union of those violated-bit sets yields the exact absent maximum.
    """
    on, off = feature["on"], feature["off"]
    positive = (w < 0).astype(float)
    positive[on], positive[off] = 1, 0
    negative = []
    for bit, required in [(p, 1) for p in on] + [(p, 0) for p in off]:
        x = (w > 0).astype(float)
        x[bit] = 1 - required
        negative.append(x)
    return positive, np.array(negative)


def verify_features(params, features=FEATURES):
    w, b, v, c = params
    result = []
    for f in features:
        i = f["neuronIndex"]
        positive, negative = worst_inputs(w[:, i], f)
        low = float(np.tanh(positive @ w[:, i] + b[i]))
        high = float(np.tanh(np.max(negative @ w[:, i] + b[i])))
        result.append(dict(neuronIndex=i, positiveMin=low, negativeMax=high, passed=low > 0 and high < 0))
    return dict(method="exact linear extrema across all satisfying/violating local patterns and all backgrounds",
                passed=all(r["passed"] for r in result), features=result)


def verify_compositions(params, features=FEATURES, examples=EXAMPLES):
    example_x = np.array([[float(p in e["boundary"]) for p in range(25)] for e in examples])
    h, scores = forward(example_x, params)
    results = []
    for k, e in enumerate(examples):
        indices = e["featureIndices"]
        contributions = h[k, indices] * params[2][indices]
        ablated = h[k].copy()
        ablated[indices] = 0
        ablated_score = float(ablated @ params[2] + params[3][0])
        union = sorted(set(p for i in indices for p in features[i]["on"]))
        passed = union == e["boundary"] and np.all(h[k, indices] > 0) and np.all(contributions >= .1) and scores[k] - ablated_score >= .4
        results.append(dict(name=e["name"], featureIndices=indices, contributions=contributions.tolist(),
                            score=float(scores[k]), ablatedScore=ablated_score, scoreDrop=float(scores[k] - ablated_score), passed=bool(passed)))
    return dict(passed=all(r["passed"] for r in results), examples=results)


def exhaustive(params, batch_size=131072):
    """All 33,554,432 inputs; retain the worst hard negatives for another round."""
    start = time.time()
    _, positive_scores = forward(POS_X, params)
    errors = int(np.count_nonzero(positive_scores <= 1))
    max_negative, max_id = -float("inf"), 0
    mined = []
    for begin in range(0, 1 << 25, batch_size):
        ids = np.arange(begin, min(begin + batch_size, 1 << 25), dtype=np.uint32)
        _, scores = forward(pixels(ids), params)
        mask = ~np.isin(ids, POS_IDS)
        negative_ids, negative_scores = ids[mask], scores[mask]
        errors += int(np.count_nonzero(negative_scores > 1))
        arg = int(np.argmax(negative_scores))
        if negative_scores[arg] > max_negative:
            max_negative, max_id = float(negative_scores[arg]), int(negative_ids[arg])
        hard = np.flatnonzero(negative_scores > .55)
        if len(hard):
            hard = hard[np.argsort(negative_scores[hard])[-128:]]
            mined.extend((float(negative_scores[i]), int(negative_ids[i])) for i in hard)
        if begin and begin % (1 << 23) == 0:
            print(f"  exhaustive {begin / (1 << 25):.0%}, errors={errors}, {time.time()-start:.1f}s", flush=True)
    closest = int(np.argmin(positive_scores))
    result = dict(inputCount=1 << 25, positiveCount=30, errors=errors, threshold=1, comparison="gt",
                  closestPositive=dict(inputId=int(POS_IDS[closest]), score=float(positive_scores[closest])),
                  closestNegative=dict(inputId=max_id, score=max_negative))
    hard_ids = sorted(set(i for _, i in sorted(mined, reverse=True)[:4096]))
    print(f"  exhaustive done {time.time()-start:.1f}s: {json.dumps(result)}", flush=True)
    return result, hard_ids


def train(args, *, features=FEATURES, examples=EXAMPLES, source=None, initialized_features=3,
          artifact_path=ARTIFACT, checkpoint_path=CHECKPOINT,
          source_description="initialized from original guided model; new local concepts initialized, then all parameters trained"):
    rng = np.random.default_rng(args.seed)
    source = load_source() if source is None else source
    example_x = np.array([[float(p in e["boundary"]) for p in range(25)] for e in examples])
    edge_indices = [f["neuronIndex"] for f in features if f["kind"] == "edge"]
    params = arrays(source["model"])
    for f in features[initialized_features:]:
        i = f["neuronIndex"]
        params[0][:, i] = rng.normal(0, .005, 25)
        params[0][f["on"], i] = 1.5
        params[0][f["off"], i] = -1.5
        params[1][i] = -1.5 * (len(f["on"]) - .5)
    params[2][edge_indices] = .25
    initial = [p.copy() for p in params]
    moments = [np.zeros_like(p) for p in params]
    variances = [np.zeros_like(p) for p in params]
    total_steps, start_round, history, hard_ids = 0, 0, [], []
    if args.resume and checkpoint_path.exists():
        ck = np.load(checkpoint_path, allow_pickle=False)
        if "sourceHash" in ck and str(ck["sourceHash"]) != source["modelSha256"]:
            raise ValueError("Checkpoint source differs from current initialization model")
        params = [ck[f"p{i}"] for i in range(4)]
        moments = [ck[f"m{i}"] for i in range(4)]
        variances = [ck[f"v{i}"] for i in range(4)]
        total_steps, start_round = int(ck["steps"]), int(ck["round"])
        hard_ids = ck["hard"].tolist()
        history = json.loads(str(ck["history"]))
        rng.bit_generator.state = json.loads(str(ck["rng"]))
    negative_ids = structured_ids()
    neg_x = pixels(negative_ids)
    started = time.time()
    for round_index in range(start_round, args.rounds):
        hard_x = pixels(hard_ids) if hard_ids else None
        for step in range(args.steps):
            pos = POS_X[rng.integers(len(POS_X), size=128)]
            negative = neg_x[rng.integers(len(neg_x), size=256)]
            random_ids = rng.integers(1 << 25, size=128, dtype=np.uint32)
            random_ids = random_ids[~np.isin(random_ids, POS_IDS)]
            parts = [pos, negative, pixels(random_ids)]
            if hard_x is not None:
                parts.append(hard_x[rng.integers(len(hard_x), size=128)])
            x = np.concatenate(parts)
            w, b, v, c = params
            h, scores = forward(x, params)
            ds = np.maximum(scores - .4, 0)
            ds[:128] = np.minimum(scores[:128] - 1.6, 0)
            class_loss = float(np.mean(ds ** 2))
            ds *= 2 / len(x)
            dh = ds[:, None] * v
            # Local concept supervision; all 1,351 parameters remain trainable.
            feature_loss = 0.
            for f in features:
                i = f["neuronIndex"]
                present = np.all(x[:, f["on"]] == 1, axis=1) & np.all(x[:, f["off"]] == 0, axis=1)
                target = np.where(present, .9, -.9)
                error = h[:, i] - target
                dh[:, i] += .4 * error / len(x)
                feature_loss += float(np.mean(error ** 2))
            dz = dh * (1 - h * h)
            grads = [x.T @ dz, dz.sum(axis=0), h.T @ ds, np.array([ds.sum()])]
            # Exact worst-background margin supervision prevents a decorative
            # label whose detector changes meaning on unseen outside pixels.
            robust_loss = 0.
            for f in features:
                i = f["neuronIndex"]
                xp, xn = worst_inputs(w[:, i], f)
                ep = min(float(xp @ w[:, i] + b[i]) - .55, 0.)
                en = np.maximum(xn @ w[:, i] + b[i] + .55, 0.)
                robust_loss += ep ** 2 + float(en @ en)
                grads[0][:, i] += 2 * (ep * xp + en @ xn)
                grads[1][i] += 2 * (ep + en.sum())
            eh, _ = forward(example_x, params)
            for k, example in enumerate(examples):
                indices = example["featureIndices"]
                err = np.minimum(eh[k, indices] * v[indices] - .14, 0)
                derivative = 8 * err
                grads[2][indices] += derivative * eh[k, indices]
                local = derivative * v[indices] * (1 - eh[k, indices] ** 2)
                grads[0][:, indices] += example_x[k, :, None] * local
                grads[1][indices] += local
            total_steps += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                moments[i] *= .9
                moments[i] += .1 * g
                variances[i] *= .999
                variances[i] += .001 * g * g
                p -= args.learning_rate * (moments[i] / (1 - .9 ** total_steps)) / (np.sqrt(variances[i] / (1 - .999 ** total_steps)) + 1e-8)
                np.clip(p, -10, 10, out=p)
            if (step + 1) % 1000 == 0:
                print(f"round={round_index+1} step={step+1} class={class_loss:.6f} concept={feature_loss:.4f} robust={robust_loss:.5f} elapsed={time.time()-started:.1f}s", flush=True)
        rounded = arrays(model_of(params))
        fv, cv = verify_features(rounded, features), verify_compositions(rounded, features, examples)
        print("features", fv["passed"], "composition", cv["passed"], flush=True)
        verification, mined = exhaustive(rounded)
        hard_ids = sorted(set(hard_ids + mined))
        history.append(dict(round=round_index+1, classificationErrors=verification["errors"], featuresPassed=fv["passed"], compositionPassed=cv["passed"], hardNegativeCount=len(hard_ids)))
        np.savez(checkpoint_path, **{f"p{i}": p for i, p in enumerate(params)},
                 **{f"m{i}": p for i, p in enumerate(moments)}, **{f"v{i}": p for i, p in enumerate(variances)},
                 steps=total_steps, round=round_index+1, hard=np.array(hard_ids, dtype=np.uint32),
                 sourceHash=source["modelSha256"], history=json.dumps(history), rng=json.dumps(rng.bit_generator.state))
        if verification["errors"] == 0 and fv["passed"] and cv["passed"]:
            model = model_of(params)
            artifact = dict(model=model, featureDefinitions=features, compositionExamples=examples,
                training=dict(seed=args.seed, steps=total_steps, stepsPerRound=args.steps, optimizer="Adam", learningRate=args.learning_rate,
                    betas=[.9, .999], epsilon=1e-8, precision=6, parameterBounds=[-10, 10], trainableParameters=1351,
                    sourceModelSha256=source["modelSha256"], source=source_description,
                    initializedFeatures=initialized_features, supervisedFeatures=len(features), freeFeatures=50-len(features),
                    parameterChangeL2=[float(np.linalg.norm(p-z)) for p,z in zip(params,initial)],
                    loss="squared classification margin 1.6/0.4 + 0.2 concept MSE + exact concept logit margins +/-0.55 + composition contribution margin 0.14",
                    structuredNegativeExamples=len(negative_ids), rounds=history, numpyVersion=np.__version__),
                verification=verification, featureVerification=fv, compositionVerification=cv,
                modelSha256=hashlib.sha256(json.dumps(model, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest())
            artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"VERIFIED artifact saved to {artifact_path}", flush=True)
            return
    raise SystemExit("Acceptance checks not yet satisfied; checkpoint retained. Increase --rounds and use --resume. No artifact was published.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=.003)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        params = arrays(artifact["model"])
        verification, _ = exhaustive(params)
        assert verification["errors"] == 0
        assert verify_features(params)["passed"] and verify_compositions(params)["passed"]
        print("All acceptance checks passed.")
    else:
        train(args)


if __name__ == "__main__":
    main()
