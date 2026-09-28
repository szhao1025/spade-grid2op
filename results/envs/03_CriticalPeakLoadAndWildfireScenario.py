# iteration 3 | grounded on: N-1 contingency | complexity 15.6 | {"survival_nohint": 0.111, "survival_hinted": 1.0, "regret": 0.889, "feasible": true, "max_rho_nohint": 1.284, "episodes": 2}
class CriticalPeakLoadAndWildfireScenario(GridScenario):
    NAME = "CriticalPeakLoadAndWildfireScenario"
    DESCRIPTION = "Simulates a scenario combining critical line outages with peak load and wildfire risks, challenging operators to balance thermal limits and load management."
    LOAD_SCALE = 1.10        # Slightly increased load to simulate peak demand
    THERMAL_SCALE = 0.92     # Tightened thermal limits due to wildfire risks
    MAX_STEPS = 288          # Episode duration
    HINT = "Be cautious with lines 3, 5, 8, 9, and 15, as they are critical during peak load and wildfires."

    def events(self, rng):
        # Define a sequence of outages that are spread out but include critical lines
        outages = [
            (30, 3),  # Critical line 3 trips at step 30
            (60, 5),  # Critical line 5 trips at step 60
            (90, 8),  # Critical line 8 trips at step 90
            (120, 9), # Critical line 9 trips at step 120
            (150, 15),# Critical line 15 trips at step 150
            (180, 6), # Critical line 6 trips at step 180
            (210, 15) # Critical line 15 trips again at step 210
        ]
        return outages

    def load_multiplier(self, t):
        # Increase load during the middle of the episode to simulate peak demand
        if 120 <= t < 180:
            return 1.1
        return 1.0
