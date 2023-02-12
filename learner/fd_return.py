import numpy as np
from networking.rpc_misc import client_server_interface_pb2


class FDReturn(object):
    def __init__(self):
        self.epoch = -1
        self.encoded_noise = "-1"
        self.perturbation = None
        self.reward = 0
        self.novelty = 0
        self.entropy = 0
        self.timesteps = 0
        self.is_eval = False
        self.eval_states = []
        self.obs_stats_update = []
        self.worker_id = ""

    def serialize(self):
        return self.worker_id, self.reward, self.novelty, self.entropy, self.timesteps, self.encoded_noise, self.perturbation, \
               self.epoch, self.is_eval, self.eval_states, self.obs_stats_update

    def deserialize(self, other):
        self.worker_id, self.reward, self.novelty, self.entropy, self.timesteps, self.encoded_noise, self.perturbation, self.epoch, \
            self.is_eval, self.eval_states, self.obs_stats_update = other

    def serialize_to_grpc(self):
        return client_server_interface_pb2.Return(
            epoch=self.epoch,
            encoded_noise=self.encoded_noise,
            reward=self.reward,
            novelty=self.novelty,
            entropy=self.entropy,
            timesteps=self.timesteps,
            is_eval=self.is_eval,
            eval_states=np.ravel(self.eval_states).tolist(),
            eval_states_shape=np.shape(self.eval_states),
            obs_stats_update=self.obs_stats_update,
            worker_id=self.worker_id
        )

    def deserialize_from_grpc(self, message):
        m = message
        self.epoch = m.epoch
        self.encoded_noise = m.encoded_noise
        self.perturbation = None
        self.reward = m.reward
        self.novelty = m.novelty
        self.entropy = m.entropy
        self.timesteps = m.timesteps
        self.is_eval = m.is_eval
        self.obs_stats_update = m.obs_stats_update,
        self.worker_id = m.worker_id

        if self.is_eval:
            serialized_states = m.eval_states
            if len(serialized_states) > 0:
                shape = m.eval_states_shape
                self.eval_states = np.asarray(serialized_states).reshape(shape).astype(np.float32)
