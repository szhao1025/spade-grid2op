# iteration 6 | grounded on: Wildfire de-energization | complexity 21.2 | {"survival_nohint": 0.111, "survival_hinted": 1.0, "regret": 0.889, "feasible": true, "max_rho_nohint": 1.174, "episodes": 2}
class WildfireAndPeakDemandScenario(GridScenario):
    NAME = "WildfireAndPeakDemandScenario"
    DESCRIPTION = "This scenario simulates a combination of critical line outages due to wildfire de-energization and a peak demand period, testing the operator's ability to manage thermal limits and load during stressful conditions."
    LOAD_SCALE = 1.04        # Slightly increased load to simulate peak demand
    THERMAL_SCALE = 0.95     # Tightened thermal limits due to critical lines and peak load
    MAX_STEPS = 288          # Episode duration
    HINT = "Be cautious with lines 3, 5, 8, 9, and 15, as they are critical during peak load and wildfire events."

    def events(self, rng):
        # Define a sequence of outages that are spread out but include critical lines
        outages = [
            (30, 3),  # Critical line 3 trips at step 30
            (60, 5),  # Critical line 5 trips at step 60
            (90, 8),  # Critical line 8 trips at step 90
            (120, 9), # Critical line 9 trips at step 120
            (150, 15),# Critical line 15 trips at step 150
            (180, 6), # Critical line 6 trips at step 180
            (210, 15),# Critical line 15 trips again at step 210
            (240, 3), # Faulty relay trips healthy line 3 at step 240, after real trip 30
            (270, 9), # Faulty relay trips healthy line 9 at step 270, after real trip 120
            (100, 0), # Line 0 trips at step 100
            (130, 1), # Line 1 trips at step 130
            (160, 4), # Line 4 trips at step 160
        ]
        return outages

    def load_multiplier(self, t):
        # Increase load during the middle of the episode to simulate peak demand
        if 120 <= t < 180:
            return 1.04
        return 1.0
