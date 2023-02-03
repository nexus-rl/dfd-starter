import numpy as np
import os


class WelfordRunningStat(object):
    """
    https://www.johndcook.com/blog/skewness_kurtosis/
    """

    def __init__(self, shape):
        self.ones = np.ones(shape=shape, dtype=np.float32)
        self.zeros = np.zeros(shape=shape, dtype=np.float32)

        self.running_mean = np.zeros(shape=shape, dtype=np.float32)
        self.running_variance = np.zeros(shape=shape, dtype=np.float32)

        self.count = 0
        self.shape = shape

    def increment(self, samples, num):
        if num > 1:
            for i in range(num):
                self.update(samples[i])
        else:
            self.update(samples)

    def update(self, sample):
        if type(sample) == dict:
            sample = sample["frame"]
        current_count = self.count
        self.count += 1
        delta = (sample - self.running_mean).reshape(self.running_mean.shape)
        delta_n = (delta / self.count).reshape(self.running_mean.shape)

        self.running_mean += delta_n
        self.running_variance += delta * delta_n * current_count

    def reset(self):
        del self.running_mean
        del self.running_variance

        self.__init__(self.shape)

    @property
    def mean(self):
        if self.count < 2:
            return self.zeros
        return self.running_mean

    @property
    def std(self):
        if self.count < 2:
            return self.ones

        var = self.running_variance / (self.count-1)

        # If variance is zero, set it to 1 to avoid division by zero. Note that dividing an array of identical numbers
        # by 1 will not change its variance at all, and such an array will surely have a mean equal to the repeated
        # value stored inside it, so standardizing that array via (array - self.mean) / self.std will result in the
        # array containing zeroes everywhere.
        var = np.where(var == 0, 1.0, var)
        # var = self.running_variance / (self.count - 1)

        return np.sqrt(var)

    def increment_from_obs_stats_update(self, obs_stats_update):
        n = np.prod(self.shape)
        other_mean = np.asarray(obs_stats_update[:n], dtype=np.float32).reshape(self.running_mean.shape)
        other_var = np.asarray(obs_stats_update[n:-1], dtype=np.float32).reshape(self.running_variance.shape)
        other_count = obs_stats_update[-1]
        if other_count == 0:
            return

        count = self.count + other_count

        mean_delta = other_mean - self.running_mean
        mean_delta_squared = mean_delta * mean_delta

        combined_mean = (self.count * self.running_mean + other_count * other_mean) / count

        combined_variance = self.running_variance + other_var + mean_delta_squared * self.count * other_count / count

        self.running_mean = combined_mean
        self.running_variance = combined_variance
        self.count = count

    def serialize(self):
        return self.running_mean.ravel().tolist() + self.running_variance.ravel().tolist() + [self.count]

    def deserialize(self, other):
        self.reset()
        n = np.prod(self.shape)

        other_mean = other[:n]
        other_var = other[n:-1]
        other_count = other[-1]
        self.running_mean = np.reshape(other_mean, self.shape)
        self.running_variance = np.reshape(other_var, self.shape)
        self.count = other_count

    def save(self, directory):
        full_path = os.path.join(directory, "running_stats.dat")
        with open(full_path, 'w') as f:
            shape = np.shape(self.ones)
            s = ''.join(["{} ".format(x) for x in shape])
            mean = ''.join(["{} ".format(x) for x in self.running_mean.ravel()])
            var = ''.join(["{} ".format(x) for x in self.running_variance.ravel()])
            f.write("{}\n{}\n{}\n{}".format(s, mean, var, self.count))

    def load(self, directory):
        full_path = os.path.join(directory, "running_stats.dat")
        with open(full_path, 'r') as f:
            lines = f.readlines()
            s = lines[0][:-1]
            mean = lines[1][:-1]
            var = lines[2][:-1]
            count = lines[3]

            shape = [int(x) for x in s.split(" ")]
            self.running_mean = np.reshape([float(x) for x in mean.split(" ")], shape)
            self.running_variance = np.reshape([float(x) for x in var.split(" ")], shape)
            self.count = int(count)


def standardize_arr(arr):
    x = np.asarray(arr)
    m = x.mean()
    s = x.std()
    if s == 0:
        return x

    return (x - m)/s


def affine_transform(value, from_min, from_max, to_min, to_max):
    if from_max == from_min or to_max == to_min:
        return to_min

    mapped = (value - from_min) * (to_max - to_min) / (from_max - from_min)
    mapped += to_min

    return mapped


def compute_strategy_novelty(strategy, other_strategies, return_all_dists=False, distance_fn=None):
    if distance_fn is None:
        distance_fn = l2_dist
    dists = distance_fn(strategy, other_strategies)

    if return_all_dists:
        return np.min(dists).item(), dists

    return np.min(dists).item()


def compute_strategy_distance(strategy_a, strategy_b, distance_fn=None):
    if distance_fn is None:
        distance_fn = l2_dist
    dist = distance_fn(strategy_a, strategy_b)

    return dist.item()


def l2_dist(strategy_a, strategy_b):
    diff = strategy_b - strategy_a
    norm = np.linalg.norm(diff, axis=-1)
    dists = norm.mean(axis=-1)
    return dists


def gaussian_bhattacharrya_dist(strategy_a, strategy_b):
    n1 = strategy_a.shape[-1] // 2
    n2 = strategy_b.shape[-1] // 2

    m1 = strategy_a[..., :n1]
    s1 = strategy_a[..., n1:]
    m2 = strategy_b[..., :n2]
    s2 = strategy_b[..., n2:]

    s3 = (s1 + s2) / 2

    s_det1 = s1.prod(axis=-1)
    s_det2 = s2.prod(axis=-1)
    s_det3 = s3.prod(axis=-1)

    a = m1 - m2
    mean_term = np.einsum("ijk->ij", a * a / s3)
    log_term = s_det3 / np.sqrt(s_det1 * s_det2)
    dists = mean_term / 8 + log_term / 4
    dists = dists.mean(axis=-1)
    return dists


def categorical_bhattacharrya_dist(p, q):
    BC = np.sum(np.sqrt(p * q), axis=-1)
    bhattacharrya_distances = -np.log(BC + 1e-12)
    return bhattacharrya_distances.mean(axis=-1)


def gaussian_wasserstein_dist(strategy_a, strategy_b):
    n1 = strategy_a.shape[-1] // 2
    n2 = strategy_b.shape[-1] // 2

    m1 = strategy_a[..., :n1]
    s1 = strategy_a[..., n1:]
    m2 = strategy_b[..., :n2]
    s2 = strategy_b[..., n2:]
    inside_trace = s1 + s2 - 2 * np.sqrt(s1 * s2)
    dists = np.square(np.linalg.norm(m1 - m2, axis=-1)) + inside_trace.sum(axis=-1)
    return dists.mean(axis=-1)


def categorical_tvd(p1, p2):
    diff = np.abs(np.subtract(p1, p2))
    tvd = diff.sum(axis=-1)
    return tvd.mean(axis=-1)

