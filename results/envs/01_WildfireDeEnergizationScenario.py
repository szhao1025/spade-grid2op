# iteration 1 | grounded on: Wildfire de-energization | complexity 11.7 | {"survival_nohint": 0.17, "survival_hinted": 1.0, "regret": 0.83, "feasible": true, "max_rho_nohint": 1.942, "episodes": 2}
class WildfireDeEnergizationScenario(GridScenario):
    NAME = "WildfireDeEnergizationScenario"
    DESCRIPTION = "Simulates proactive line de-energization due to wildfire risk, causing complex cascading failures and increased demand. Skilled operators must carefully manage load and avoid critical lines from tripping simultaneously."
    LOAD_SCALE = 1.04        # Slightly increased load to simulate peak demand
    THERMAL_SCALE = 0.98     # Slightly tightened line limits due to weather conditions
    MAX_STEPS = 288          # Episode duration
    HINT = "Be cautious with lines 6, 8, 9, and 15, as they are critical during peak load."

    def events(self, rng):
        # Define a sequence of outages that are spread out but include critical lines
        outages = [
            (30, 3),  # Critical line 3 trips at step 30
            (60, 5),  # Critical line 5 trips at step 60
            (90, 9),  # Critical line 9 trips at step 90
            (120, 6), # Critical line 6 trips at step 120
            (150, 8), # Critical line 8 trips at step 150
            (180, 9)  # Critical line 9 trips again at step 180
        ]
        return outages

    def load_multiplier(self, t):
        # Increase load during the middle of the episode to simulate peak demand
        if 120 <= t < 180:
            return 1.1
        return 1.0
