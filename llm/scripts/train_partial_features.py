"""Train 25 clear local features + 25 free features for six square compositions.

The 25 -> 50 -> 1 topology and all 1,351 trainable parameters are retained.
Only local edges/corner have concept supervision; the remaining 25 neurons
learn freely. No complete-contour or missing-corner templates are introduced.
"""
import argparse
import json
import train_composition as base

ARTIFACT = base.ROOT / "partial-features-model.json"
CHECKPOINT = base.ROOT / ".partial-features-checkpoint.npz"
SOURCE = base.ROOT / "composition-model.json"


def definitions():
    features, examples = base.definitions()
    for size, row, col, name, candidate in [
        (3, 0, 0, "左上 3×3", 16),
        (3, 2, 2, "右下 3×3", 24),
        (4, 1, 1, "右下 4×4", 28),
    ]:
        edges = base.square(size, row, col)
        indices = []
        for side, points in zip(["上边", "下边", "左边", "右边"], edges):
            match = next((f for f in features if f["on"] == points and not f["off"]), None)
            if match is None:
                match = dict(neuronIndex=len(features), name=f"{name} · {side}",
                    position="、".join(f"{p // 5 + 1}行{p % 5 + 1}列" for p in points),
                    on=points, off=[], kind="edge")
                features.append(match)
            indices.append(match["neuronIndex"])
        examples.append(dict(name=name, size=size, row=row, col=col, candidateIndex=candidate,
            featureIndices=indices, boundary=sorted(set(sum(edges, []))),
            missingPixel=edges[0][size // 2], extraPixel=(row + 1)*5+col+1))
    # Keep old feature indices; order the new gallery by size and position.
    examples = [examples[i] for i in (0, 3, 4, 1, 5, 2)]
    assert len(features) == 25 and len(examples) == 6
    return features, examples


FEATURES, EXAMPLES = definitions()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=.003)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        assert artifact["featureDefinitions"] == FEATURES and artifact["compositionExamples"] == EXAMPLES
        params = base.arrays(artifact["model"])
        assert base.verify_features(params, FEATURES)["passed"]
        assert base.verify_compositions(params, FEATURES, EXAMPLES)["passed"]
        verification, _ = base.exhaustive(params)
        assert verification["errors"] == 0
        print("All 25 concepts, 6 compositions and exhaustive classification passed.")
    else:
        source = json.loads(SOURCE.read_text(encoding="utf-8"))
        base.train(args, features=FEATURES, examples=EXAMPLES, source=source, initialized_features=13,
            artifact_path=ARTIFACT, checkpoint_path=CHECKPOINT,
            source_description="initialized from composition-50-v1; 12 additional local edges initialized; all 1351 parameters trained with 25 supervised and 25 free neurons")


if __name__ == "__main__":
    main()
