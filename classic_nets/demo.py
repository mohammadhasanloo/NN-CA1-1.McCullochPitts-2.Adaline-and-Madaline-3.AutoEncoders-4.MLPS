"""Runs each model on each problem and writes the figures and scores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from classic_nets.adaline import Adaline
from classic_nets.datasets import concentric, separable, xor
from classic_nets.figures import boundaries
from classic_nets.madaline import Madaline

PROBLEMS = {"separable": separable, "concentric": concentric, "xor": xor}


def run(docs: Path, results: Path) -> dict:
    summary: dict[str, dict[str, float]] = {}

    for name, generator in PROBLEMS.items():
        x, y = generator(rng=np.random.default_rng(0))
        adaline = Adaline(learning_rate=0.01)
        adaline.fit(x, y, epochs=40)

        panels = [(adaline, x, y, "Adaline")]
        scores = {"adaline": adaline.score(x, y)}
        for vote in ("and", "or"):
            model = Madaline(hidden_units=8, vote=vote, rng=np.random.default_rng(1))
            model.fit(x, y)
            panels.append((model, x, y, f"Madaline, {vote.upper()} vote"))
            scores[f"madaline_{vote}"] = model.score(x, y)

        boundaries(panels, docs / f"{name}.png", suptitle=f"{name} problem")
        summary[name] = {k: round(v, 4) for k, v in scores.items()}

    results.mkdir(parents=True, exist_ok=True)
    (results / "scores.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    summary = run(root / "docs", root / "results")
    header = f"{'problem':<12}{'adaline':>10}{'AND':>10}{'OR':>10}"
    print(header)
    for name, scores in summary.items():
        print(f"{name:<12}{scores['adaline']:>10.2f}{scores['madaline_and']:>10.2f}"
              f"{scores['madaline_or']:>10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
