import numpy as np
from strategy import StrategyPoint, SparseHistoryManager
from utils import math_helpers


class StrategyHandler(object):
    def __init__(self, policy, strategy_distance_fn, max_history_size=200):
        self.strategy_history_manager = SparseHistoryManager(policy, strategy_distance_fn, max_history_size)
        self.strategy_tensor = np.zeros(0)
        self.policy = policy
        self.zeta = None
        self.max_history_size = max_history_size
        self.strategy_distance_fn = strategy_distance_fn

    def update_from_server_state(self, server_state):
        self.strategy_tensor = np.asarray(server_state.strategy_history, dtype=np.float32).reshape(
            server_state.strategy_history_shape)

        self.zeta = np.asarray(server_state.strategy_frames, dtype=np.float32).reshape(
            server_state.strategy_frames_shape)

    def add_policy(self, policy, obs_stats=None):
        self.strategy_history_manager.submit_policy(policy, obs_stats=obs_stats)

    def set_zeta(self, zeta):
        if zeta is None or len(zeta) == 0:
            return

        self.zeta = zeta
        self.strategy_tensor = self.strategy_history_manager.evaluate_strategies(zeta)

    def compute_novelty(self, policy, obs_stats=None):
        if self.zeta is None or len(self.zeta) == 0 or self.strategy_tensor is None or len(self.strategy_tensor) < 2:
            return 0

        point = StrategyPoint(self.policy, policy.get_trainable_flat())
        strategy = point.evaluate_strategy(self.zeta, obs_stats=obs_stats)

        return math_helpers.compute_strategy_novelty(strategy, self.strategy_tensor, distance_fn=self.strategy_distance_fn)