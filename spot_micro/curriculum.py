"""Performance-gated command progression, checkpointable independently of env count."""
class PerformanceCurriculum:
    def __init__(self, settings):
        self.settings = settings
        self.maximum = float(settings.get("start_m_s", .25))
        self.last_check = 0
        self.episodes = self.survived = self.samples = self.error_sum = 0.
        self.last_error = None
        self.last_survival = None

    def observe(self, episodes, survived, samples, error_sum, transitions):
        self.episodes += episodes
        self.survived += survived
        self.samples += samples
        self.error_sum += error_sum
        if (transitions-self.last_check < self.settings.get("min_transitions",3_000_000)
                or self.episodes < 128 or self.samples < 5000):
            return False
        self.last_error = self.error_sum/self.samples
        self.last_survival = self.survived/self.episodes
        advance = (self.last_error <= self.settings.get("absolute_error_m_s",.10)+.10*self.maximum
                   and self.last_survival >= self.settings.get("survival_fraction",.85))
        if advance:
            self.maximum = min(self.settings.get("max_m_s",2.), self.maximum+self.settings.get("increment_m_s",.10))
        self.last_check = transitions
        self.episodes = self.survived = self.samples = self.error_sum = 0.
        return advance

    def state_dict(self):
        return {k:v for k,v in self.__dict__.items() if k != "settings"}

    def load_state_dict(self, state):
        for key in self.state_dict():
            if key in state:
                setattr(self,key,state[key])
