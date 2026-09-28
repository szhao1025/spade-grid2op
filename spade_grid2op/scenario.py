"""Executable Grid2Op scenarios: the contract the Environment Designer writes code against.

A scenario is a small Python class (written by the LLM) that subclasses ``GridScenario``.
``GridScenarioEnv`` turns it into a Gym-style environment with ``reset()`` / ``step()`` on top
of a real Grid2Op power grid.
"""
from __future__ import annotations

import math
import random
import signal
from dataclasses import dataclass, field
from typing import List, Tuple

import grid2op
from lightsim2grid import LightSimBackend

GRID_NAME = "l2rpn_case14_sandbox"

# Bounds the designer must respect (validated after execution).
LOAD_SCALE_RANGE = (0.8, 1.25)
THERMAL_SCALE_RANGE = (0.85, 1.0)
MAX_EVENTS = 12
MAX_STEPS_RANGE = (48, 288)  # 4 hours to 1 day at 5-minute resolution


class GridScenario:
    """Base class the Environment Designer subclasses.

    Class attributes set the static stress; the two methods add time-dependent,
    randomized structure (this is what makes the scenario *executable code* rather than a config).
    """

    NAME: str = "BaseScenario"
    DESCRIPTION: str = "Nominal grid, no extra stress."
    LOAD_SCALE: float = 1.0      # multiplies every load and generator setpoint
    THERMAL_SCALE: float = 1.0   # multiplies every line's thermal limit (lower = tighter)
    MAX_STEPS: int = 288         # episode length in 5-minute steps
    HINT: str = ""               # privileged hint for a solver (SPADE's h)

    def events(self, rng: random.Random) -> List[Tuple[int, int]]:
        """Forced line outages as (step, line_id) pairs."""
        return []

    def load_multiplier(self, t: int) -> float:
        """Extra time-varying demand multiplier at step t (e.g., an afternoon peak)."""
        return 1.0


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    events: List[Tuple[int, int]] = field(default_factory=list)


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout("scenario code took too long")


_SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in ["abs", "min", "max", "range", "len", "int", "float", "round", "sum", "sorted",
                 "list", "tuple", "dict", "set", "enumerate", "zip", "bool", "str", "any", "all",
                 "isinstance", "ValueError", "super", "__build_class__", "object", "map", "filter"]
}


def load_scenario_class(code: str, timeout_s: int = 5) -> type:
    """Executes LLM-written code in a restricted namespace and returns the GridScenario subclass."""
    namespace = {"__builtins__": _SAFE_BUILTINS, "__name__": "generated_scenario",
                 "GridScenario": GridScenario, "math": math, "random": random,
                 "List": List, "Tuple": Tuple}
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_s)
    try:
        exec(compile(code, "<scenario>", "exec"), namespace)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    classes = [v for v in namespace.values()
               if isinstance(v, type) and issubclass(v, GridScenario) and v is not GridScenario]
    if not classes:
        raise ValueError("no subclass of GridScenario was defined")
    return classes[-1]


def validate(scenario: GridScenario, n_line: int, seed: int = 0) -> ValidationResult:
    """Checks the scenario obeys the contract (types, bounds, line ids, determinism)."""
    errors = []

    def check_range(name, value, lo, hi):
        if not isinstance(value, (int, float)) or not (lo <= value <= hi):
            errors.append(f"{name}={value!r} must be a number in [{lo}, {hi}]")

    check_range("LOAD_SCALE", scenario.LOAD_SCALE, *LOAD_SCALE_RANGE)
    check_range("THERMAL_SCALE", scenario.THERMAL_SCALE, *THERMAL_SCALE_RANGE)
    check_range("MAX_STEPS", scenario.MAX_STEPS, *MAX_STEPS_RANGE)

    events: List[Tuple[int, int]] = []
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(5)
    try:
        events = list(scenario.events(random.Random(seed)))
        again = list(scenario.events(random.Random(seed)))
        if events != again:
            errors.append("events(rng) must be deterministic given the rng (use rng, not global random)")
        for t in range(0, int(scenario.MAX_STEPS), 12):
            m = scenario.load_multiplier(t)
            if not isinstance(m, (int, float)) or not (0.5 <= m <= 1.5):
                errors.append(f"load_multiplier({t})={m!r} must be a number in [0.5, 1.5]")
                break
    except _Timeout as e:
        errors.append(str(e))
    except Exception as e:  # noqa: BLE001 - report any bug in generated code
        errors.append(f"{type(e).__name__} while calling scenario methods: {e}")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

    if len(events) > MAX_EVENTS:
        errors.append(f"{len(events)} events; at most {MAX_EVENTS} allowed")
    for ev in events:
        if (not isinstance(ev, (tuple, list)) or len(ev) != 2
                or not all(isinstance(x, int) for x in ev)):
            errors.append(f"event {ev!r} must be a (step, line_id) pair of ints")
            continue
        t, line = ev
        if not (0 <= line < n_line):
            errors.append(f"event {ev!r}: line_id must be in [0, {n_line - 1}]")
        if not (0 <= t < scenario.MAX_STEPS):
            errors.append(f"event {ev!r}: step must be in [0, MAX_STEPS)")
    return ValidationResult(ok=not errors, errors=errors, events=events)


class GridScenarioEnv:
    """Gym-style wrapper: a Grid2Op environment stressed according to a GridScenario."""

    _shared_env = None  # one Grid2Op instance is reused; creating it is the slow part

    def __init__(self, scenario: GridScenario, chronic_id: int = 0, seed: int = 0):
        if GridScenarioEnv._shared_env is None:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                GridScenarioEnv._shared_env = grid2op.make(GRID_NAME, test=True,
                                                           backend=LightSimBackend())
        self.env = GridScenarioEnv._shared_env
        self.scenario = scenario
        self.chronic_id = chronic_id
        self.seed = seed
        self.t = 0
        self.schedule: dict[int, List[int]] = {}

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        env = self.env
        env.set_id(self.chronic_id)
        env.seed(self.seed)
        obs = env.reset()
        if not hasattr(env, "_spade_base_thermal"):
            env._spade_base_thermal = env.get_thermal_limit().copy()
        env.set_thermal_limit(env._spade_base_thermal * float(self.scenario.THERMAL_SCALE))

        data = env.chronics_handler.real_data.data
        if not hasattr(data, "_spade_orig_load"):
            data._spade_orig_load = data.load_p.copy()
            data._spade_orig_prod = data.prod_p.copy()
        for row in range(data.load_p.shape[0]):
            k = float(self.scenario.LOAD_SCALE) * float(self.scenario.load_multiplier(row))
            data.load_p[row] = data._spade_orig_load[row] * k
            data.prod_p[row] = data._spade_orig_prod[row] * k

        self.schedule = {}
        for t, line in self.scenario.events(random.Random(self.seed)):
            self.schedule.setdefault(int(t), []).append(int(line))
        self.t = 0
        return obs, {"hint": self.scenario.HINT}

    def upcoming_outages(self, horizon: int) -> List[Tuple[int, int]]:
        """Privileged information: scheduled outages in the next `horizon` steps."""
        return [(t, l) for t in range(self.t, self.t + horizon) for l in self.schedule.get(t, [])]

    def step(self, action):
        forced = self.schedule.get(self.t, [])
        if forced:
            action = action + self.env.action_space({"set_line_status": [(l, -1) for l in forced]})
        obs, reward, done, info = self.env.step(action)
        self.t += 1
        truncated = (not done) and self.t >= int(self.scenario.MAX_STEPS)
        terminated = bool(done)  # grid2op "done" before the horizon means blackout (game over)
        return obs, float(reward), terminated, truncated, info
