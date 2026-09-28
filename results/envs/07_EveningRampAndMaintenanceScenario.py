# iteration 9 | grounded on: Evening ramp | complexity 24.5 | {"survival_nohint": 0.059, "survival_hinted": 1.0, "regret": 0.941, "feasible": true, "max_rho_nohint": 1.21, "episodes": 2}
class EveningRampAndMaintenanceScenario(GridScenario):
    NAME = "EveningRampAndMaintenanceScenario"
    DESCRIPTION = "This scenario simulates a critical evening demand peak coinciding with multiple maintenance outages, creating a complex challenge for operators to manage thermal stress and maintain grid stability."
    LOAD_SCALE = 1.12      # Slightly increased load to simulate peak demand
    THERMAL_SCALE = 0.93   # Tightened thermal limits due to critical lines and peak load
    MAX_STEPS = 288        # Episode duration
    HINT = "Be cautious with lines 6, 8, 9, 15, and 17, as they are critical during peak load and maintenance events."

    def events(self, rng):
        # Define a sequence of outages that are spread out but include critical lines
        outages = [
            (20, 6),  # Planned maintenance of critical line 6 at step 20
            (50, 8),  # Planned maintenance of critical line 8 at step 50
            (80, 9),  # Planned maintenance of critical line 9 at step 80
            (110, 15), # Planned maintenance of critical line 15 at step 110
            (140, 17), # Planned maintenance of critical line 17 at step 140
            (170, 3),  # Critical line 3 trips at step 170
            (200, 5),  # Critical line 5 trips at step 200
            (230, 15), # Critical line 15 trips again at step 230
            (260, 9),  # Critical line 9 trips again at step 260
            (30, 0),   # Line 0 trips at step 30
            (60, 1),   # Line 1 trips at step 60
            (90, 4)    # Line 4 trips at step 90
        ]
        return outages

    def load_multiplier(self, t):
        # Increase load during the middle of the episode to simulate peak demand
        if 120 <= t < 180:
            return 1.12
        return 1.0
