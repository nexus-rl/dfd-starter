import numpy as np
import os


class AdaptiveOmega(object):
    def __init__(self, default_value=0, improvement_threshold=1.025, reward_history_size=10,
                 min_value=0, max_value=1, steps_to_min=15, steps_to_max=200):

        self.omega = default_value
        self.improvement_threshold = improvement_threshold
        self.reward_history_size = reward_history_size
        self.min_omega = min_value
        self.max_omega = max_value

        self.reward_history = []
        self.steps_to_max = steps_to_max
        self.steps_to_min = steps_to_min

        self.increase = 1 / self.steps_to_max
        self.decrease = 1 / self.steps_to_min

        self.decrease_start = 0
        self.increase_start = 0

    def step(self, theta_reward):
        if theta_reward is None:
            return

        self.advance_reward_history(theta_reward)
        self.adapt_omega(theta_reward)

    def adapt_omega(self, theta_reward):
        if len(self.reward_history) == 0:
            return
        mean_reward = float(np.mean(self.reward_history))

        mean_reward = round(mean_reward, 5)
        theta_reward = round(theta_reward, 5)

        if mean_reward < 0:
            mean_reward /= self.improvement_threshold
        else:
            mean_reward *= self.improvement_threshold

        if theta_reward > mean_reward:
            self.omega = max(self.omega - self.decrease, self.min_omega)
        else:
            self.omega = min(self.omega + self.increase, self.max_omega)

    def advance_reward_history(self, reward):
        self.reward_history.append(reward)
        if len(self.reward_history) > self.reward_history_size:
            self.reward_history.pop(0)