# iteration 7 | grounded on: Planned maintenance | complexity 24.1 | {"survival_nohint": 0.111, "survival_hinted": 1.0, "regret": 0.889, "feasible": true, "max_rho_nohint": 1.132, "episodes": 2}
class PlannedMaintenanceAndPeakDemandScenario(GridScenario):
    NAME = "PlannedMaintenanceAndPeakDemandScenario"
    DESCRIPTION = "This scenario simulates unexpected peak demand coinciding with planned maintenance outages, increasing thermal stress on critical lines and complicating operator decision-making."
    LOAD_SCALE = 1.1       # Slightly increased load to simulate peak demand
    THERMAL_SCALE = 0.92   # Tightened thermal limits due to critical lines and peak load
    MAX_STEPS = 288        # Episode duration
    HINT = "Be cautious with lines 6, 8, 9, 15, and 17, as they are critical during peak load and maintenance events."

    def events(self, rng):
        # Define a sequence of outages that are spread out but include critical lines
        outages = [
            (30, 6),  # Planned maintenance of critical line 6 at step 30
            (60, 8),  # Planned maintenance of critical line 8 at step 60
            (90, 9),  # Planned maintenance of critical line 9 at step 90
            (120, 15), # Planned maintenance of critical line 15 at step 120
            (150, 17), # Planned maintenance of critical line 17 at step 150
            (180, 3),  # Critical line 3 trips at step 180
            (210, 5),  # Critical line 5 trips at step 210
            (240, 15), # Critical line 15 trips again at step 240
            (270, 9),  # Critical line 9 trips again at step 270
            (100, 0),  # Line 0 trips at step 100
            (130, 1),  # Line 1 trips at step 130
            (160, 4),  # Line 4 trips at step 160
        ]
        return outages

    def load_multiplier(self, t):
        # Increase load during the middle of the episode to simulate peak demand
        if 120 <= t < 180:
            return 1.1
        return 1.0
