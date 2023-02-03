import numpy as np


class RNGNoiseSource(object):
    def __init__(self, n_params, random_seed=123):
        self.rng = np.random.default_rng(np.random.SeedSequence(random_seed))
        self.base_state = self.rng.__getstate__()
        self.n_params = n_params

    def sample(self):
        state = "{}".format(self.rng.__getstate__()['state']['state'])
        noise = self.rng.standard_normal(size=self.n_params)
        return state, noise

    def decode(self, state):
        self.base_state['state']['state'] = int(state)
        self.rng.__setstate__(self.base_state)
        return self.rng.standard_normal(size=self.n_params)


class SimpleNoiseSource(object):
    def __init__(self, n_params, random_seed=123):
        self.rng = np.random.RandomState(random_seed)
        self.n_params = n_params

    def sample(self):
        noise = self.rng.randn(self.n_params)
        return noise, noise

    def decode(self, noise):
        return noise


class SharedNoiseTable(object):
    def __init__(self, size, n_params, random_seed=123):
        assert size > n_params, "!ATTEMPTED TO MAKE NOISE TABLE WITH SIZE {} FOR {} PARAMETERS!".format(size, n_params)
        self._rng = np.random.RandomState(random_seed)
        self._table = self._rng.randn(size).astype(np.float32)
        self._n_params = n_params
        self._max_sample_idx = size - n_params

    def sample(self):
        noise_idx = self._rng.randint(0, self._max_sample_idx)
        noise = self._table[noise_idx:noise_idx + self._n_params]
        return "{}".format(noise_idx), noise

    def decode(self, noise_idx):
        noise_idx = int(noise_idx)
        return self._table[noise_idx:noise_idx+self._n_params]
