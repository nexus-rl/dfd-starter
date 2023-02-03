import numpy as np
from strategy import StrategyPoint
from utils import math_helpers


class SparseHistoryManager(object):
    def __init__(self, policy, strategy_distance_fn, max_history_size):
        self.policy = policy
        self.max_history_size = max_history_size
        self.worst_point_idx = 0
        self.strategy_points = []
        self.strategy_tensor = []
        self.zeta = []
        self.known_dists = {}
        self.strategy_distance_fn = strategy_distance_fn

    def submit_policy(self, policy):
        """
        Function to submit a policy to the history of policies we examine when computing the novelty of a new policy.
        :param policy: The policy to submit.
        :return:
        """

        point = StrategyPoint(self.policy, policy.get_trainable_flat())
        if len(self.strategy_points) >= self.max_history_size and self.zeta is not None and len(self.zeta) > 0:
            return self._replace_point(point)
        else:
            self.strategy_points.append(point)

        return None

    def evaluate_strategies(self, zeta):
        """
        Function to evaluate the strategy of all policies in the history we are tracking on a set of states.
        :param zeta: The states on which strategies will be evaluated.
        :return: A ndarray containing every strategy in our history.
        """

        self.zeta = zeta
        strategy_tensor = []
        strategy_points = self.strategy_points
        for point in strategy_points:
            strategy_tensor.append(point.evaluate_strategy(zeta))

        self._construct_table()
        self.strategy_tensor = np.asarray(strategy_tensor)
        return self.strategy_tensor

    def _construct_table(self):
        """
        Function to construct a table of known distances between pairs of strategies. This is particularly useful for
        computation speed so we don't have to re-compute anything when trying to determine what policy to replace when
        a more novel one is added.
        :return:
        """

        strategy_points = self.strategy_points
        n = len(strategy_points)
        known_dists = {}

        # Compute the distance between every policy and every other policy, saving them in a dictionary as we go.
        for i in range(n):
            a = strategy_points[i]
            for j in range(i + 1, n):
                b = strategy_points[j]
                dist = math_helpers.compute_strategy_distance(a.strategy, b.strategy, distance_fn=self.strategy_distance_fn)
                known_dists[(i, j)] = dist

        self.known_dists.clear()
        self.known_dists = known_dists
        self._update_strategy_point_dists()

    def _replace_point(self, point):
        """
        Function to determine whether a policy should be replaced and to replace one if it should. A policy will only
        replace an older one if its novelty relative to our history is greater than the smallest distance between any
        two strategies that we currently know of. If replacement does happen, the policy with the smallest distance to
        its second-nearest neighbor will be chosen out of the pair that are closest together in our history.
        :param point: StrategyPoint object representing the policy we're trying to add to the history.
        :return:
        """

        strategy = point.evaluate_strategy(self.zeta)
        # Here we compute both the novelty of the policy (distance between it and its nearest neighbor from our history)
        # and the distance between the candidate policy and every other policy in our history.
        novelty, dists = math_helpers.compute_strategy_novelty(strategy, self.strategy_tensor, return_all_dists=True,
                                                               distance_fn=self.strategy_distance_fn)

        # Get the distance between the closest pair in our history.
        idx = self.worst_point_idx
        current_worst = self.strategy_points[idx].closest[1]

        # If that distance is less than the novelty of the candidate policy, we'll replace the least novel of the pair.
        if novelty > current_worst or current_worst == np.inf:
            known_dists = self.known_dists
            self.strategy_points[idx] = point
            self.strategy_tensor[idx] = strategy

            # Here we're replacing all the distances in our known distance table that have to do with the point we are
            # replacing with the distances we computed earlier when checking the novelty of the candidate policy.
            for pair, distance in known_dists.items():
                if idx in pair:
                    other_idx = pair[1 - pair.index(idx)]
                    known_dists[pair] = dists[other_idx]

            self._update_strategy_point_dists()
            return idx

        return -1

    def _update_strategy_point_dists(self):
        """
        Function to tell strategy points about the distances between them and the rest of the points we are tracking.
        This function also serves to keep track of the policy that is currently the least novel thing in our history.
        :return:
        """

        worst_dist = np.inf
        strategy_points = self.strategy_points
        n = len(strategy_points)
        known_dists = self.known_dists

        # Reset distance information in every strategy point and update them with our current known distance table.
        for i in range(n):
            strategy_points[i].reset_dists()
            for key, val in known_dists.items():
                if i in key:
                    strategy_points[i].add_dist(key, val)

        # Seek the least novel policy in our history and save it to a local variable so we don't have to look for it
        # again until we change something.
        for i in range(n):
            closest = strategy_points[i].closest
            if closest[1] < worst_dist:
                worst_idx1 = i

                if closest[0] is None:
                    self.worst_point_idx = i
                    continue

                worst_idx2 = closest[0][1 - closest[0].index(i)]
                worst_dist = closest[1]

                second_closest_1 = strategy_points[worst_idx1].second_closest[1]
                second_closest_2 = strategy_points[worst_idx2].second_closest[1]
                if second_closest_1 < second_closest_2:
                    self.worst_point_idx = worst_idx1
                else:
                    self.worst_point_idx = worst_idx2
