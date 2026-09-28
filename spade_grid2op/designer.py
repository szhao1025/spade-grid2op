"""The Environment Designer: an LLM that writes executable Grid2Op scenarios.

The prompt follows the structure of SPADE's environment-generation prompt (Appendix C.1):
role, task, grounding document, rules, difficulty requirements, an interface contract, robustness
rules, and a fenced ```python answer. Unlike C.1 (text games graded by \\boxed{} answers), the
environments here are power-grid operating scenarios executed on Grid2Op.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

from .scenario import LOAD_SCALE_RANGE, MAX_EVENTS, MAX_STEPS_RANGE, THERMAL_SCALE_RANGE

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
DEFAULT_MODEL = os.environ.get("SPADE_MODEL", "qwen2.5:7b")

SYSTEM_PROMPT = (
    "You are an expert Python programmer and power-system operations engineer. You design "
    "challenging but survivable operating scenarios for training grid-control agents."
)

INTERFACE_CONTRACT = f'''```python
class DescriptiveNameScenario(GridScenario):   # GridScenario, math and random are already imported
    NAME = "DescriptiveNameScenario"
    DESCRIPTION = "One sentence: what stress this scenario applies and why it is hard."
    LOAD_SCALE = 1.0        # float in [{LOAD_SCALE_RANGE[0]}, {LOAD_SCALE_RANGE[1]}]: scales all demand
    THERMAL_SCALE = 1.0     # float in [{THERMAL_SCALE_RANGE[0]}, {THERMAL_SCALE_RANGE[1]}]: scales line limits (lower = tighter)
    MAX_STEPS = 288         # int in [{MAX_STEPS_RANGE[0]}, {MAX_STEPS_RANGE[1]}], one step = 5 minutes
    HINT = "Privileged advice for an operator, e.g. which lines fail and when."

    def events(self, rng):
        # Return a list of (step, line_id) forced line outages, at most {MAX_EVENTS}.
        # Use ONLY the given rng for randomness (rng.randint, rng.choice, rng.sample).
        return [(30, 3), (32, 5)]

    def load_multiplier(self, t):
        # Extra demand multiplier at step t, a float in [0.5, 1.5]. Must never raise.
        return 1.0
```'''


def describe_grid(line_or_sub, line_ex_sub) -> str:
    rows = [f"  line {i}: substation {o} <-> substation {e}"
            for i, (o, e) in enumerate(zip(line_or_sub, line_ex_sub))]
    return "\n".join(rows)


COMPLEXITY_FORMULA = (
    "complexity = (#outages, repeated trips of a line within 12 steps count once) "
    "+ 0.5*(#distinct lines) + 0.5*(#outages within 12 steps of the previous one) "
    "+ 20*(LOAD_SCALE-1 if >1) + 20*(1-THERMAL_SCALE) + 10*(peak load_multiplier-1 if >1) "
    "+ 1 if load_multiplier varies over time"
)


def build_prompt(document: tuple[str, str], grid_description: str, memory_summary: str,
                 frontier_code: Optional[str], feedback: Optional[str],
                 target: Optional[tuple[float, float]] = None) -> str:
    title, text = document
    parts = [
        "Create a challenging power-grid operating scenario as a Python class that tests an "
        "operator's ability to keep the IEEE 14-bus grid (20 lines, 14 substations) alive.",
        "",
        f"GROUNDING DOCUMENT ({title}) - base your scenario on this real-world situation:",
        text,
        "",
        "GRID LINES (use these ids; lines sharing a substation are in the same area):",
        grid_description,
        "",
        "GRID KNOWLEDGE (from an N-1 experiment on this grid):",
        "- Critical lines - a single trip already makes a passive operator fail: 6, 8, 9, 15, 17, 18, 19.",
        "- Mild lines - a single trip alone is harmless: 0, 1, 2, 4, 5, 10, 12, 13, 14.",
        "- Tightening THERMAL_SCALE from 1.0 to 0.95 turns lines 3, 11 and 16 critical too.",
        "",
        "ENVIRONMENT MEMORY (scenarios already accepted, oldest first, with measured results):",
        memory_summary or "  (empty - this is the first scenario)",
        "",
    ]
    if frontier_code:
        parts += [
            "CURRENT FRONTIER SCENARIO (the hardest accepted so far):",
            f"```python\n{frontier_code}\n```",
            "",
            "YOUR TASK: write a NEW scenario that is MORE COMPLEX than the frontier: combine more "
            "interacting stresses (more or better-timed outages, tighter limits, a demand peak that "
            "coincides with outages) while staying survivable for a skilled operator.",
            "",
        ]
    if target:
        parts += [
            "COMPLEXITY TARGET (progress gradually - a small step beyond the frontier):",
            f"  {COMPLEXITY_FORMULA}",
            f"  Aim for complexity between {target[0]:.1f} and {target[1]:.1f}. Much larger jumps "
            "usually make the grid collapse for everyone and get rejected.",
            "",
        ]
    if feedback:
        parts += ["FEEDBACK ON YOUR PREVIOUS ATTEMPT:", feedback, ""]
    parts += [
        "DIFFICULTY (measured automatically after you answer):",
        "- A passive operator that never acts should FAIL (blackout before MAX_STEPS).",
        "- A skilled operator with your HINT and a simulator should SURVIVE at least 60% of the "
        "episode. Scenarios nobody can survive are rejected as infeasible.",
        "- Outages clustered in time and in the same area are harder than spread-out ones.",
        "- Very high LOAD_SCALE (> 1.15) or very low THERMAL_SCALE (< 0.88) usually makes the "
        "grid collapse immediately for everyone - prefer smart outage timing over brute force.",
        "",
        "RULES:",
        "- Give the class a descriptive, unique name that reflects the grounding document.",
        "- Only use the interface below; do not import anything; no file, network or print calls.",
        "- events() must be deterministic given rng and return (int step, int line_id) tuples.",
        "- load_multiplier() must return a float for every t from 0 to MAX_STEPS.",
        "",
        "INTERFACE CONTRACT:",
        INTERFACE_CONTRACT,
        "",
        "Generate the complete Python code in a ```python block.",
    ]
    return "\n".join(parts)


def extract_code(reply: str) -> Optional[str]:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", reply, flags=re.S)
    if blocks:
        return max(blocks, key=len).strip()
    return reply.strip() if "class " in reply else None


@dataclass
class LLMClient:
    model: str = DEFAULT_MODEL
    temperature: float = 0.8
    url: str = OLLAMA_URL

    def chat(self, messages: List[dict]) -> str:
        body = json.dumps({
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": 8192},
        }).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())["message"]["content"]
