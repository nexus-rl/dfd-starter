from utils import math_helpers
from dsgd import DSGD
import numpy as np


class FiniteDifferences(object):
    def __init__(self, policy, gradient_optimizer, omega, noise_source, noise_std=0.1, batch_size=100, ent_coef=0.0, max_delayed_return=10):
        self.max_delayed_return = max_delayed_return
        self.ent_coef = ent_coef
        self.noise_std = noise_std
        self.policy = policy
        self.gradient_optimizer = gradient_optimizer
        self.noise_source = noise_source
        self.omega = omega

        self.policy_history = [(policy.get_trainable_flat(), 0)]
        self.epoch = 0
        self.discarded_returns = 0
        self.dist_map = {0: 0}
        self.gradient_memory = np.zeros(policy.num_params)

        self.using_dsgd = type(gradient_optimizer) == DSGD

    def step(self, batch, policy_reward, policy_novelty, policy_entropy):
        rewards, novelties, entropies, perturbations = self._process_returns(batch)
        if policy_reward is None:
            policy_reward = 0
            policy_entropy = 0
            policy_novelty = 0
        if len(rewards) == 0:
            return 0

        # print("{:7.4f} | {:7.4f} | {:7.4f} | {:7.4f}".format(
        #     np.mean(novelties), np.std(novelties), np.min(novelties), np.max(novelties)))
        # print("{:7.4f} | {:7.4f} | {:7.4f} | {:7.4f}".format(
        #     np.mean(rewards), np.std(rewards), np.min(rewards), np.max(rewards)))
        # print("{:7.4f} | {:7.4f} | {:7.4f} | {:7.4f}".format(
        #     np.mean(entropies), np.std(entropies), np.min(entropies), np.max(entropies)))

        rewards = np.subtract(rewards, policy_reward)
        # novelties = np.subtract(novelties, policy_novelty)
        # entropies = np.subtract(entropies, policy_entropy)
        rewards = math_helpers.standardize_arr(rewards)
        # novelties = math_helpers.standardize_arr(novelties)
        # entropies = math_helpers.standardize_arr(entropies)
        w = self.omega.omega

        objective_function = rewards #+ entropies*self.ent_coef  # *(1-w) + novelties*w + entropies * self.ent_coef
        np.dot(objective_function, perturbations, out=self.gradient_memory) / len(batch)

        if self.using_dsgd:
            self.gradient_optimizer.adjust_lr(self.omega)

        flat = self.policy.get_trainable_flat()
        self.gradient_optimizer.zero_grad()
        self.policy.set_grad_from_flat(-self.gradient_memory)
        self.gradient_optimizer.step()

        update_size = np.linalg.norm(flat - self.policy.get_trainable_flat())
        self.epoch += 1

        self._build_distance_map()
        self._update_policy_history()
        return update_size

    def _build_distance_map(self):
        flat = self.policy.get_trainable_flat()
        length = len(self.policy_history)
        self.dist_map.clear()
        self.dist_map[self.epoch] = 0
        for i in range(length):
            params, epoch = self.policy_history[i]
            self.dist_map[epoch] = params - flat

    def _update_policy_history(self):
        self.policy_history.append((self.policy.get_trainable_flat(), self.epoch))
        while len(self.policy_history) > self.max_delayed_return:
            _ = self.policy_history.pop(0)

    def _adjust_return(self, ret):
        epoch = ret.epoch
        if epoch not in self.dist_map.keys():
            print("FINITE DIFFERENCE LEARNER RECEIVED RETURN THAT WAS TOO OLD")
            print("RECEIVED EPOCH:", ret.epoch, "ACCEPTABLE EPOCHS:", self.dist_map.keys())
            return False

        decoded_noise = self.noise_source.decode(ret.encoded_noise)
        policy_dist = self.dist_map[epoch]
        lmbda = decoded_noise*self.noise_std + policy_dist
        ret.perturbation = lmbda

        return True

    def _process_returns(self, batch):
        rewards = []
        novelties = []
        entropies = []
        perturbations = []

        for ret in batch:
            if not self._adjust_return(ret):
                self.discarded_returns += 1
                continue

            perturbation = ret.perturbation
            # perturbation = ret.encoded_noise
            norm = np.linalg.norm(perturbation)

            rewards.append(ret.reward)
            novelties.append(ret.novelty)
            entropies.append(ret.entropy)
            perturbations.append(perturbation / (norm * norm))

        return rewards, novelties, entropies, perturbations