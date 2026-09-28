"""SPADE-style Environment Designer loop for Grid2Op.

Each iteration:
  1. sample a grounding document from the power-system corpus,
  2. prompt the LLM designer with the document, the environment memory and the current frontier,
  3. execute + validate the generated scenario code (with up to N repair rounds on errors),
  4. estimate difficulty with the hint-based regret proxy (no-hint vs hinted operator),
  5. accept it into memory only if it is valid, feasible, and MORE complex than the frontier.

Usage:
    python run_designer.py --iterations 12
"""
from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from pathlib import Path

from spade_grid2op.corpus import sample_document
from spade_grid2op.designer import LLMClient, SYSTEM_PROMPT, build_prompt, describe_grid, extract_code
from spade_grid2op.evaluate import FEASIBLE_SURVIVAL, evaluate
from spade_grid2op.scenario import GridScenario, GridScenarioEnv, load_scenario_class, validate

MIN_REGRET = 0.15  # the hint must matter, otherwise the scenario is not at the frontier
STEP_MIN, STEP_MAX = 1.5, 6.0  # target complexity increase per accepted scenario (curriculum step)
OVERSHOOT = 3.0  # tolerated overshoot above STEP_MAX before asking the designer to simplify

SEED_SCENARIO = '''class NominalGridScenario(GridScenario):
    NAME = "NominalGridScenario"
    DESCRIPTION = "Normal operating day: nominal demand and limits, no outages."
    LOAD_SCALE = 1.0
    THERMAL_SCALE = 1.0
    MAX_STEPS = 288
    HINT = "Nothing unusual happens today."

    def events(self, rng):
        return []

    def load_multiplier(self, t):
        return 1.0'''


def effective_events(events):
    """Drops repeated trips of a line that is still out (same line within 12 steps counts once)."""
    kept, last = [], {}
    for t, line in sorted(events):
        if line in last and t - last[line] <= 12:
            continue
        kept.append((t, line))
        last[line] = t
    return kept


def complexity(scenario: GridScenario, events) -> float:
    """Interpretable structural complexity of a scenario (independent of the measured difficulty)."""
    events = effective_events(events)
    steps = sorted(t for t, _ in events)
    clustered = sum(1 for a, b in zip(steps, steps[1:]) if b - a <= 12)
    peak = max(scenario.load_multiplier(t) for t in range(0, int(scenario.MAX_STEPS)))
    varying = len({round(scenario.load_multiplier(t), 3) for t in range(0, int(scenario.MAX_STEPS), 6)}) > 1
    return round(
        len(events)
        + 0.5 * len({l for _, l in events})
        + 0.5 * clustered
        + 20 * max(0.0, scenario.LOAD_SCALE - 1.0)
        + 20 * (1.0 - scenario.THERMAL_SCALE)
        + 10 * max(0.0, peak - 1.0)
        + (1.0 if varying else 0.0),
        2,
    )


def summarize_memory(archive) -> str:
    lines = []
    for e in archive:
        m = e["metrics"]
        lines.append(
            f"  - {e['name']}: complexity {e['complexity']}, {e['n_events']} outages, "
            f"LOAD_SCALE {e['load_scale']}, THERMAL_SCALE {e['thermal_scale']} | passive survival "
            f"{m['survival_nohint']:.0%}, skilled survival {m['survival_hinted']:.0%}, regret {m['regret']:.2f}"
        )
    return "\n".join(lines[-6:])  # keep the prompt short for a 7B model


def feedback_for(ev, cx, frontier_cx) -> str:
    msgs = []
    if not ev.feasible:
        msgs.append(f"INFEASIBLE: even the skilled operator survived only {ev.survival_hinted:.0%} "
                    f"(need >= {FEASIBLE_SURVIVAL:.0%}). Reduce brute-force stress (LOAD_SCALE, "
                    "THERMAL_SCALE) or spread the outages slightly.")
    if ev.regret < MIN_REGRET:
        if ev.survival_nohint > 0.9:
            msgs.append(f"TOO EASY: the passive operator survived {ev.survival_nohint:.0%}. "
                        "Add outages that overload neighbouring lines, clustered in time.")
        else:
            msgs.append(f"LOW REGRET ({ev.regret:.2f}): the hint barely helps. Make the outages "
                        "predictable from the HINT so a prepared operator can react.")
    if cx <= frontier_cx:
        msgs.append(f"NOT MORE COMPLEX: complexity {cx} <= frontier {frontier_cx}. Combine more "
                    "interacting stresses than the frontier scenario.")
    return " ".join(msgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--repairs", type=int, default=2, help="repair rounds per iteration on code errors")
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "envs").mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"
    log_path.write_text("")
    llm = LLMClient(model=args.model) if args.model else LLMClient()

    probe = GridScenarioEnv(GridScenario())
    grid_desc = describe_grid(probe.env.line_or_to_subid, probe.env.line_ex_to_subid)
    n_line = probe.env.n_line

    # Iteration 0: the seed (nominal) scenario starts the memory.
    seed_cls = load_scenario_class(SEED_SCENARIO)
    seed = seed_cls()
    seed_val = validate(seed, n_line)
    seed_ev = evaluate(seed)
    archive = [{
        "iteration": 0, "name": seed.NAME, "code": SEED_SCENARIO, "complexity": complexity(seed, seed_val.events),
        "n_events": 0, "load_scale": seed.LOAD_SCALE, "thermal_scale": seed.THERMAL_SCALE,
        "metrics": seed_ev.to_dict(),
    }]
    (out / "envs" / f"00_{seed.NAME}.py").write_text(SEED_SCENARIO + "\n")
    print(f"[0] seed {seed.NAME}: {seed_ev.to_dict()}")

    feedback = None
    for it in range(1, args.iterations + 1):
        frontier = archive[-1]
        doc = sample_document(rng)
        target = (frontier["complexity"] + STEP_MIN, frontier["complexity"] + STEP_MAX)
        prompt = build_prompt(doc, grid_desc, summarize_memory(archive), frontier["code"], feedback, target)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        record = {"iteration": it, "document": doc[0], "status": None}
        t0 = time.time()

        scenario, val, code = None, None, None
        for attempt in range(args.repairs + 1):
            reply = llm.chat(messages)
            code = extract_code(reply)
            error = None
            if code is None:
                error = "No ```python code block found in your answer."
            else:
                try:
                    scenario = load_scenario_class(code)()
                    val = validate(scenario, n_line)
                    if not val.ok:
                        error = "Contract violations: " + "; ".join(val.errors)
                    else:
                        # Curriculum check is cheap (no simulation), so it goes through repair too.
                        # At most one curriculum repair; after that the simulator decides.
                        cx_try = complexity(scenario, val.events)
                        curriculum_repairs = sum("complexity" in r for r in record.get("repairs", []))
                        if curriculum_repairs >= 1:
                            pass
                        elif cx_try > target[1] + OVERSHOOT:
                            error = (f"Your scenario has complexity {cx_try}, far above the target "
                                     f"[{target[0]:.1f}, {target[1]:.1f}]. Remove outages or soften "
                                     "LOAD_SCALE/THERMAL_SCALE/peaks to land inside the target.")
                        elif cx_try <= frontier["complexity"]:
                            error = (f"Your scenario has complexity {cx_try}, not above the frontier "
                                     f"({frontier['complexity']}). Add one more interacting stress.")
                except Exception as e:  # noqa: BLE001 - feed any error back to the designer
                    error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=1)}"
            if error is None:
                break
            record.setdefault("repairs", []).append(error[:500])
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": f"Your code failed: {error}\nReturn the fully corrected class in a ```python block."}]
            scenario = None

        record["llm_seconds"] = round(time.time() - t0, 1)
        if scenario is None:
            record["status"] = "invalid"
            feedback = "Your previous scenario could not be executed: " + record["repairs"][-1][:300]
            print(f"[{it}] {doc[0]}: INVALID after {args.repairs} repairs")
            with log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            continue

        ev = evaluate(scenario)
        cx = complexity(scenario, val.events)
        record.update({"name": scenario.NAME, "complexity": cx, "n_events": len(val.events),
                       "load_scale": scenario.LOAD_SCALE, "thermal_scale": scenario.THERMAL_SCALE,
                       "metrics": ev.to_dict(), "code": code})
        accepted = ev.feasible and ev.regret >= MIN_REGRET and cx > frontier["complexity"]
        record["status"] = "accepted" if accepted else "rejected"
        tag = "ACCEPT" if accepted else "reject"
        print(f"[{it}] {doc[0]}: {tag} {scenario.NAME} complexity={cx} "
              f"(frontier {frontier['complexity']}) passive={ev.survival_nohint:.0%} "
              f"skilled={ev.survival_hinted:.0%} regret={ev.regret:.2f}")
        if accepted:
            archive.append({k: record[k] for k in ("iteration", "name", "code", "complexity", "n_events",
                                                   "load_scale", "thermal_scale", "metrics")})
            header = f"# iteration {it} | grounded on: {doc[0]} | complexity {cx} | {json.dumps(ev.to_dict())}\n"
            (out / "envs" / f"{len(archive) - 1:02d}_{scenario.NAME}.py").write_text(header + code + "\n")
            feedback = None
        else:
            feedback = feedback_for(ev, cx, frontier["complexity"])
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    (out / "archive.json").write_text(json.dumps(archive, indent=1))
    print(f"\nAccepted {len(archive) - 1} new scenarios; frontier complexity "
          f"{archive[0]['complexity']} -> {archive[-1]['complexity']}")


if __name__ == "__main__":
    main()
