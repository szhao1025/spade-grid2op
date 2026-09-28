"""Experiment: N-1 contingency analysis of the IEEE 14-bus Grid2Op environment.

For every line, trip it at step 10 and measure how long each operator keeps the grid alive,
under nominal and 5%-tighter thermal limits. The result tells the Environment Designer which
lines are critical (hard for a passive operator) and is written to results/n1_analysis.md.

Usage: python experiments/n1_analysis.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spade_grid2op.evaluate import evaluate  # noqa: E402
from spade_grid2op.scenario import GridScenario, GridScenarioEnv  # noqa: E402


def main():
    env = GridScenarioEnv(GridScenario()).env
    rows = ["| line | substations | passive (limits 1.0) | greedy+hint (1.0) | passive (0.95) | greedy+hint (0.95) |",
            "|---:|:---:|---:|---:|---:|---:|"]
    critical = []
    for line in range(env.n_line):
        cells = []
        for th in (1.0, 0.95):
            class SingleOutage(GridScenario):
                THERMAL_SCALE = th

                def events(self, rng, _line=line):
                    return [(10, _line)]

            ev = evaluate(SingleOutage())
            cells += [f"{ev.survival_nohint:.2f}", f"{ev.survival_hinted:.2f}"]
            if th == 1.0 and ev.survival_nohint < 0.5:
                critical.append(line)
        subs = f"{env.line_or_to_subid[line]}-{env.line_ex_to_subid[line]}"
        rows.append(f"| {line} | {subs} | " + " | ".join(cells) + " |")
        print(rows[-1], flush=True)

    out = Path("results")
    out.mkdir(exist_ok=True)
    text = ("# N-1 contingency analysis (fraction of a 288-step day survived, mean of 2 chronics)\n\n"
            + "\n".join(rows)
            + f"\n\nCritical lines (passive operator survives < 50% after a single trip): {critical}\n")
    (out / "n1_analysis.md").write_text(text)
    print("critical lines:", critical)


if __name__ == "__main__":
    main()
