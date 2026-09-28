class NominalGridScenario(GridScenario):
    NAME = "NominalGridScenario"
    DESCRIPTION = "Normal operating day: nominal demand and limits, no outages."
    LOAD_SCALE = 1.0
    THERMAL_SCALE = 1.0
    MAX_STEPS = 288
    HINT = "Nothing unusual happens today."

    def events(self, rng):
        return []

    def load_multiplier(self, t):
        return 1.0
