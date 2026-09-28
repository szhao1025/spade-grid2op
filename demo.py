"""Interactive walkthrough of the project.

Part 1 replays an accepted scenario and shows, step by step, how a passive operator and the hinted
operator cope with it. Part 2 (--live) runs one Environment Designer iteration with the local LLM.

Usage:
    python demo.py                 # part 1 only (no LLM needed)
    python demo.py --live          # part 1 + one live designer iteration (needs `ollama serve`)
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


def pause(msg="按回车继续..."):
    if AUTO:
        print(f"\n{DIM}（自动播放，3 秒后继续）{RESET}")
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
    return f"{color}{'█' * n}{RESET}{' ' * (30 - n)} {rho:4.0%}"


def replay(scenario, hinted: bool):
    env = GridScenarioEnv(scenario, chronic_id=0, seed=0)
    obs, _ = env.reset()
    agent = GreedyHintedAgent(env.action_space) if hinted else None
    who = f"{GREEN}调度员 B（有提示 + 模拟器预演）{RESET}" if hinted else f"{YELLOW}调度员 A（什么都不做）{RESET}"
    print(f"\n{BOLD}▶ {who}{BOLD} 开始值班{RESET}   （每行 = 一个值得注意的时刻；条形 = 最拥挤那条线的负载率）\n")
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
            notes.append(f"{RED}⚡ {', '.join(f'{l}号线' for l in outs)} 断开{RESET}")
        if acted:
            actions += 1
            d = action.as_dict()
            if "set_line_status" in str(d) or "reconnect" in str(d).lower():
                notes.append(f"{GREEN}🔧 重新接通线路{RESET}")
            else:
                subs = d.get("set_bus_vect", {}).get("modif_subs_id", [])
                notes.append(f"{GREEN}🔧 调整变电站 {','.join(map(str, subs)) or ''} 的接线{RESET}")
        if terminated:
            print(f"  {clock(t)}  {RED}{BOLD}💥 电网崩溃（大停电）！只撑到了一天的 {env.t / scenario.MAX_STEPS:.0%}{RESET}")
            return env.t / scenario.MAX_STEPS
        if notes or t % 36 == 0:
            print(f"  {clock(t)}  {bar(rho)}  {'  '.join(notes)}")
            time.sleep(0.15)
        if truncated:
            break
    print(f"  {clock(env.t)}  {GREEN}{BOLD}✅ 撑满了一整天！{RESET}" + (f"  （一共出手 {actions} 次）" if hinted else ""))
    return 1.0


def part1():
    header("第 1 部分：重放 AI 出的一道题")
    path = Path("results/envs/08_HeatWaveAndMaintenanceScenario.py")
    code = path.read_text()
    print(f"这是第 11 轮大模型读了「热浪」文档后写的场景代码（{path}）：\n")
    print(DIM + "\n".join(l for l in code.splitlines() if not l.startswith("#")) + RESET)
    scenario = load_scenario_class(code)()
    print(f"\n{BOLD}翻译成人话：{RESET}用电量 +{scenario.LOAD_SCALE - 1:.0%}，线路承受力 −{1 - scenario.THERMAL_SCALE:.0%}，"
          f"一天里 {len(scenario.events(random.Random(0)))} 次断线，下午还有一波用电高峰。")
    pause()
    replay(scenario, hinted=False)
    pause()
    replay(scenario, hinted=True)
    print(f"\n{BOLD}结论：{RESET}没提示的 A 很快就崩了，有提示的 B 能撑下来。")
    print("两者差距大 = 这道题正好卡在「努力就能解、不努力就会挂」的边界上 = SPADE 想要的好题。")


def part2():
    from run_designer import complexity, summarize_memory, SEED_SCENARIO
    from spade_grid2op.corpus import sample_document
    from spade_grid2op.designer import LLMClient, SYSTEM_PROMPT, build_prompt, describe_grid, extract_code
    import json

    header("第 2 部分：现场让大模型出一道新题")
    archive = json.loads(Path("results/archive.json").read_text())
    frontier = archive[3]  # start from an early frontier so the demo shows a clear step up
    rng = random.Random(int(time.time()))
    doc = sample_document(rng)
    print(f"① 抽到的领域文档：{BOLD}{doc[0]}{RESET}\n   {DIM}{doc[1]}{RESET}")
    print(f"\n② 当前题库里最难的题：{BOLD}{frontier['name']}{RESET}（复杂度 {frontier['complexity']}）"
          f"\n   要求大模型出一道复杂度在 {frontier['complexity'] + 1.5:.1f} ~ {frontier['complexity'] + 6:.1f} 之间的新题")
    pause("按回车，把这些发给大模型（本地 Qwen 7B，大约要 1 分钟）...")

    probe = GridScenarioEnv(GridScenario())
    grid = describe_grid(probe.env.line_or_to_subid, probe.env.line_ex_to_subid)
    prompt = build_prompt(doc, grid, summarize_memory(archive[:4]), frontier["code"], None,
                          (frontier["complexity"] + 1.5, frontier["complexity"] + 6))
    print(f"{DIM}   （prompt 一共 {len(prompt)} 个字符，结构仿照 SPADE 论文附录 C.1）{RESET}")
    t0 = time.time()
    reply = LLMClient().chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
    code = extract_code(reply)
    print(f"\n③ 大模型用了 {time.time() - t0:.0f} 秒，写出了这段代码：\n")
    print(DIM + (code or reply) + RESET)
    pause()

    print("④ 检查代码能不能运行、参数有没有越界...")
    try:
        scenario = load_scenario_class(code)()
        val = validate(scenario, probe.env.n_line)
    except Exception as e:  # noqa: BLE001
        print(f"   {RED}✗ 代码运行出错：{e}{RESET}\n   （正式循环里会把报错发回给大模型让它修）")
        return
    if not val.ok:
        print(f"   {RED}✗ 不符合规则：{'; '.join(val.errors)}{RESET}\n   （正式循环里会把问题发回给大模型让它修）")
        return
    cx = complexity(scenario, val.events)
    print(f"   {GREEN}✓ 通过{RESET}，复杂度 = {cx}（题库最难的是 {frontier['complexity']}）")

    print("\n⑤ 让两个调度员各跑一天（2 组不同的用电数据）来评估难度...")
    ev = evaluate(scenario)
    print(f"   调度员 A（不做事）平均存活：{ev.survival_nohint:.0%}")
    print(f"   调度员 B（有提示）平均存活：{ev.survival_hinted:.0%}")
    print(f"   差距 regret = {ev.regret:.2f}")
    checks = [
        (ev.feasible, f"有解：B 至少活 60%（实际 {ev.survival_hinted:.0%}）"),
        (ev.regret >= 0.15, f"有难度：提示真的有用，差距 ≥ 0.15（实际 {ev.regret:.2f}）"),
        (cx > frontier["complexity"], f"更复杂：{cx} > {frontier['complexity']}"),
    ]
    print()
    for ok, text in checks:
        print(f"   {GREEN + '✓' if ok else RED + '✗'} {text}{RESET}")
    if all(ok for ok, _ in checks):
        print(f"\n   {GREEN}{BOLD}→ 收进题库！这道新题成了新的「最难」。{RESET}")
    else:
        print(f"\n   {YELLOW}{BOLD}→ 拒绝，把不合格的原因反馈给大模型，下一轮再试。{RESET}")


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
