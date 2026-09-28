"""Difficulty estimation with a hint-based regret proxy (no policy training).

SPADE rewards the Environment Designer with the gap between the Reasoning Agent's return with and
without a privileged hint. We keep that idea but use two fixed Grid2Op controllers instead of a
trained LLM agent:

* no-hint agent:  DoNothingAgent - keeps the default topology and never reacts.
* hinted agent:   a greedy operator that receives privileged information (the upcoming outage
                  schedule plus simulator lookahead via ``obs.simulate``) and reconnects lines /
                  switches substation topology to keep line loadings low.

regret  = survival(hinted) - survival(no-hint)   (high = the hint matters = frontier environment)
feasible = survival(hinted) >= FEASIBLE_SURVIVAL (an environment nobody can survive teaches nothing)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List

import numpy as np

from .scenario import GridScenario, GridScenarioEnv

FEASIBLE_SURVIVAL = 0.6
EVAL_CHRONICS = (0, 1)  # the two load/generation time series bundled with the test grid


@dataclass
class Evaluation:
    survival_nohint: float
    survival_hinted: float
    regret: float
    feasible: bool
    max_rho_nohint: float
    episodes: int

    def to_dict(self):
        return asdict(self)


class GreedyHintedAgent:
    """Reconnects lines and tries a few topology changes, choosing by simulated max line loading."""

    MARGIN = 0.02  # only deviate from "do nothing" if it lowers the max loading by at least this

    _topo_cache = None  # the 178 unitary topology actions of the 14-bus grid (same for every run)

    def __init__(self, action_space):
        self.action_space = action_space
        if GreedyHintedAgent._topo_cache is None:
            GreedyHintedAgent._topo_cache = action_space.get_all_unitary_topologies_set(action_space)
        self.topo_candidates = GreedyHintedAgent._topo_cache
        self.revert = None

    def _score(self, obs, act, forced_now, forced_soon) -> float:
        """Max line loading after `act`, stress-tested against the hinted outages.

        forced_now: outages the environment applies this very step (combined with our action).
        forced_soon: outages in the next few steps; the action must also survive them.
        """
        worst = 0.0
        for extra in [forced_now] + ([forced_now + forced_soon] if forced_soon else []):
            test = act + self.action_space({"set_line_status": [(l, -1) for l in extra]}) if extra else act
            sim_obs, _, sim_done, _ = obs.simulate(test)
            if sim_done:
                return np.inf
            worst = max(worst, float(sim_obs.rho.max()))
        return worst

    def act(self, obs, env: GridScenarioEnv):
        # The privileged hint: which lines the scenario will trip now and in the next 3 steps.
        forced_now = [l for t, l in env.upcoming_outages(horizon=1)]
        forced_soon = [l for t, l in env.upcoming_outages(horizon=4) if t > env.t]
        score = lambda a: self._score(obs, a, forced_now, forced_soon)  # noqa: E731

        noop = self.action_space({})
        best, best_score = noop, score(noop)

        candidates = []
        # Reconnect any disconnected line that is allowed to be reconnected (and is not about to trip).
        for line in np.where(~obs.line_status & (obs.time_before_cooldown_line == 0))[0]:
            if int(line) not in forced_now:
                candidates.append(self.action_space({"set_line_status": [(int(line), 1)]}))
        # Search topology changes when stressed or when the hint says an outage is imminent.
        if obs.rho.max() > 0.85 or forced_now or forced_soon:
            candidates.extend(self.topo_candidates)
        # Back to the reference topology once the grid is calm again.
        elif not np.all(obs.topo_vect[obs.topo_vect > 0] == 1):
            candidates.append(self.action_space({"set_bus": np.ones(obs.dim_topo, dtype=int)}))

        for act in candidates:
            s = score(act)
            if s < best_score - self.MARGIN:
                best, best_score = act, s
        return best


def _run_episode(scenario: GridScenario, chronic_id: int, hinted: bool) -> tuple[float, float]:
    env = GridScenarioEnv(scenario, chronic_id=chronic_id, seed=chronic_id)
    obs, _ = env.reset()
    agent = GreedyHintedAgent(env.action_space) if hinted else None
    max_rho = 0.0
    horizon = int(scenario.MAX_STEPS)
    for _ in range(horizon):
        action = agent.act(obs, env) if hinted else env.action_space({})
        obs, _, terminated, truncated, _ = env.step(action)
        max_rho = max(max_rho, float(obs.rho.max()))
        if terminated or truncated:
            break
    survived = env.t if not terminated else env.t - 1
    return survived / horizon, max_rho


def evaluate(scenario: GridScenario, chronics: List[int] = EVAL_CHRONICS) -> Evaluation:
    nohint = [_run_episode(scenario, c, hinted=False) for c in chronics]
    hinted = [_run_episode(scenario, c, hinted=True) for c in chronics]
    s_no = float(np.mean([s for s, _ in nohint]))
    s_hi = float(np.mean([s for s, _ in hinted]))
    return Evaluation(
        survival_nohint=round(s_no, 3),
        survival_hinted=round(s_hi, 3),
        regret=round(s_hi - s_no, 3),
        feasible=s_hi >= FEASIBLE_SURVIVAL,
        max_rho_nohint=round(float(np.max([r for _, r in nohint])), 3),
        episodes=len(chronics),
    )
