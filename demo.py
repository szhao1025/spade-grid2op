"""Interactive walkthrough of the project.

Part 1 replays an accepted scenario and shows, step by step, how a passive operator and the hinted
operator cope with it. Part 2 (--live) runs one Environment Designer iteration with the local LLM.

Usage:
    python demo.py                 # part 1 only (no LLM needed)
    python demo.py --live          # part 1 + one live designer iteration (needs `ollama serve`)
    python demo.py --live --auto   # advance automatically instead of waiting for Enter
"""
import argparse
import random
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from spade_grid2op.evaluate import GreedyHintedAgent, evaluate  # noqa: E402
from spade_grid2op.scenario import GridScenario, GridScenarioEnv, load_scenario_class, validate  # noqa: E402

BOLD, DIM, RED, GREEN, YELLOW, CYAN, RESET = "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[36m", "\033[0m"

AUTO = "--auto" in sys.argv  # advance automatically instead of waiting for Enter


def pause(msg="Press Enter to continue..."):
    if AUTO:
        print(f"\n{DIM}(auto mode: continuing in 3 seconds){RESET}")
        time.sleep(3)
    elif sys.stdin.isatty():
        input(f"\n{DIM}{msg}{RESET}")
    else:
        time.sleep(1.5)


def header(text):
    print(f"\n{BOLD}{CYAN}{'=' * 70}\n{text}\n{'=' * 70}{RESET}")


def clock(step):
    minutes = step * 5
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def bar(rho):
    n = min(int(rho * 20), 30)
    color = GREEN if rho < 0.8 else YELLOW if rho < 1.0 else RED
    return f"{color}{'#' * n}{RESET}{' ' * (30 - n)} {rho:4.0%}"


def replay(scenario, hinted: bool):
    env = GridScenarioEnv(scenario, chronic_id=0, seed=0)
    obs, _ = env.reset()
    agent = GreedyHintedAgent(env.action_space) if hinted else None
    who = (f"{GREEN}Operator B (hint + simulator lookahead){RESET}" if hinted
           else f"{YELLOW}Operator A (passive, never acts){RESET}")
    print(f"\n{BOLD}> {who}{BOLD} starts the shift{RESET}   "
          "(one row per notable moment; bar = loading of the most loaded line)\n")
    actions = 0
    for _ in range(int(scenario.MAX_STEPS)):
        t = env.t
        outs = env.schedule.get(t, [])
        action = agent.act(obs, env) if hinted else env.action_space({})
        acted = hinted and action != env.action_space({})
        obs, _, terminated, truncated, info = env.step(action)
        rho = float(obs.rho.max()) if not terminated else 0.0
        notes = []
        if outs:
            notes.append(f"{RED}OUTAGE: line {', '.join(str(l) for l in outs)} trips{RESET}")
        if acted:
            actions += 1
            d = action.as_dict()
            if "set_line_status" in str(d) or "reconnect" in str(d).lower():
                notes.append(f"{GREEN}ACTION: reconnect line{RESET}")
            else:
                subs = d.get("set_bus_vect", {}).get("modif_subs_id", [])
                notes.append(f"{GREEN}ACTION: reconfigure substation {','.join(map(str, subs))}{RESET}")
        if terminated:
            print(f"  {clock(t)}  {RED}{BOLD}BLACKOUT! The grid survived only "
                  f"{env.t / scenario.MAX_STEPS:.0%} of the day.{RESET}")
            return env.t / scenario.MAX_STEPS
        if notes or t % 36 == 0:
            print(f"  {clock(t)}  {bar(rho)}  {'  '.join(notes)}")
            time.sleep(0.15)
        if truncated:
            break
    print(f"  {clock(env.t)}  {GREEN}{BOLD}Survived the whole day!{RESET}"
          + (f"  ({actions} actions taken)" if hinted else ""))
    return 1.0


def part1():
    header("Part 1: replay a scenario written by the Environment Designer")
    path = Path("results/envs/08_HeatWaveAndMaintenanceScenario.py")
    code = path.read_text()
    print(f"Scenario code written by the LLM in iteration 11, grounded on the 'Heat wave' document ({path}):\n")
    print(DIM + "\n".join(l for l in code.splitlines() if not l.startswith("#")) + RESET)
    scenario = load_scenario_class(code)()
    print(f"\n{BOLD}In plain words:{RESET} demand +{scenario.LOAD_SCALE - 1:.0%}, line limits "
          f"-{1 - scenario.THERMAL_SCALE:.0%}, {len(scenario.events(random.Random(0)))} line outages "
          "during the day, plus an afternoon demand peak.")
    pause()
    replay(scenario, hinted=False)
    pause()
    replay(scenario, hinted=True)
    print(f"\n{BOLD}Takeaway:{RESET} the passive operator collapses early, while the hinted operator survives.")
    print("A large gap (high regret) means the scenario sits at the capability frontier: "
          "solvable with skill, fatal without it.")


def part2():
    from run_designer import complexity, summarize_memory
    from spade_grid2op.corpus import sample_document
    from spade_grid2op.designer import LLMClient, SYSTEM_PROMPT, build_prompt, describe_grid, extract_code
    import json

    header("Part 2: one live Environment Designer iteration")
    archive = json.loads(Path("results/archive.json").read_text())
    frontier = archive[3]  # start from an early frontier so the demo shows a clear step up
    rng = random.Random(int(time.time()))
    doc = sample_document(rng)
    print(f"1. Sampled grounding document: {BOLD}{doc[0]}{RESET}\n   {DIM}{doc[1]}{RESET}")
    print(f"\n2. Current frontier: {BOLD}{frontier['name']}{RESET} (complexity {frontier['complexity']})"
          f"\n   Target for the new scenario: complexity {frontier['complexity'] + 1.5:.1f} to "
          f"{frontier['complexity'] + 6:.1f}")
    pause("Press Enter to send the prompt to the LLM (local Qwen 7B, about 1 minute)...")

    probe = GridScenarioEnv(GridScenario())
    grid = describe_grid(probe.env.line_or_to_subid, probe.env.line_ex_to_subid)
    prompt = build_prompt(doc, grid, summarize_memory(archive[:4]), frontier["code"], None,
                          (frontier["complexity"] + 1.5, frontier["complexity"] + 6))
    print(f"{DIM}   (prompt length: {len(prompt)} characters, structured after SPADE Appendix C.1){RESET}")
    t0 = time.time()
    reply = LLMClient().chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
    code = extract_code(reply)
    print(f"\n3. The LLM answered in {time.time() - t0:.0f} s with this code:\n")
    print(DIM + (code or reply) + RESET)
    pause()

    print("4. Executing and validating the code...")
    try:
        scenario = load_scenario_class(code)()
        val = validate(scenario, probe.env.n_line)
    except Exception as e:  # noqa: BLE001
        print(f"   {RED}FAILED to execute: {e}{RESET}\n   (the real loop feeds this error back for a repair)")
        return
    if not val.ok:
        print(f"   {RED}Contract violations: {'; '.join(val.errors)}{RESET}\n"
              "   (the real loop feeds these back for a repair)")
        return
    cx = complexity(scenario, val.events)
    print(f"   {GREEN}OK{RESET}, complexity = {cx} (frontier: {frontier['complexity']})")

    print("\n5. Running both operators for one day on 2 load/generation time series...")
    ev = evaluate(scenario)
    print(f"   Operator A (passive) mean survival: {ev.survival_nohint:.0%}")
    print(f"   Operator B (hinted)  mean survival: {ev.survival_hinted:.0%}")
    print(f"   regret = {ev.regret:.2f}")
    checks = [
        (ev.feasible, f"Feasible: operator B survives >= 60% (got {ev.survival_hinted:.0%})"),
        (ev.regret >= 0.15, f"At the frontier: the hint matters, regret >= 0.15 (got {ev.regret:.2f})"),
        (cx > frontier["complexity"], f"More complex: {cx} > {frontier['complexity']}"),
    ]
    print()
    for ok, text in checks:
        print(f"   {GREEN + '[x]' if ok else RED + '[ ]'} {text}{RESET}")
    if all(ok for ok, _ in checks):
        print(f"\n   {GREEN}{BOLD}-> ACCEPTED: this scenario becomes the new frontier.{RESET}")
    else:
        print(f"\n   {YELLOW}{BOLD}-> REJECTED: the failed checks are fed back to the LLM for the next iteration.{RESET}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--auto", action="store_true", help="advance automatically (no Enter needed)")
    args = ap.parse_args()
    part1()
    if args.live:
        pause()
        part2()
    print()
