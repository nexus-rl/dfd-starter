

class FDState(object):
    def __init__(self):
        self.strategy_frames = None
        self.strategy_frames_shape = None
        self.strategy_history = None
        self.strategy_history_shape = None
        self.policy_params = None
        self.epoch = None
        self.cfg = None
        self.experiment_id = None
        self.obs_stats = None

    def cleanup(self):
        del self.policy_params
        del self.strategy_history
        del self.strategy_frames
