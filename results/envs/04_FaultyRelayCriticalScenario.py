# iteration 5 | grounded on: Protection misoperation | complexity 15.9 | {"survival_nohint": 0.111, "survival_hinted": 1.0, "regret": 0.889, "feasible": true, "max_rho_nohint": 1.211, "episodes": 2}
class FaultyRelayCriticalScenario(GridScenario):
    NAME = "FaultyRelayCriticalScenario"
    DESCRIPTION = "This scenario simulates a faulty relay misoperation, causing a healthy line to trip shortly after a real fault, turning a single contingency into a double contingency. The skilled operator must navigate thermal limits and load increases to avoid blackouts."
    LOAD_SCALE = 1.05        # Slightly increased load to simulate peak demand
    THERMAL_SCALE = 0.93     # Tightened thermal limits due to faulty relay
    MAX_STEPS = 288          # Episode duration
    HINT = "Be cautious with lines 3, 5, 8, 9, and 15, as they are critical during peak load and faulty relay events."

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
        ]
        return outages

    def load_multiplier(self, t):
        # Increase load during the middle of the episode to simulate peak demand
        if 120 <= t < 180:
            return 1.05
        return 1.0
