import torch.nn as nn
import torch
import numpy as np
from utils import torch_helpers
from functools import partial


class Policy(nn.Module):
    def __init__(self, n_inputs, n_actions, seed=124):
        super().__init__()
        self.model = None
        self.num_params = None
        self.input_shape = n_inputs
        self.output_shape = n_actions
        self.rng = np.random.RandomState(seed)

    def get_action(self, x, deterministic=False):
        raise NotImplementedError

    def get_entropy(self, x):
        raise NotImplementedError

    def get_strategy(self, x):
        raise NotImplementedError

    def forward(self, x):
        if type(x) is not torch.Tensor:
            x = torch.as_tensor(x, dtype=torch.float32).view(-1, self.input_shape)
        return self.model(x)

    def compute_vbn(self, buffer):
        self.train()
        self.forward(buffer)
        self.eval()

    @torch.no_grad()
    def get_trainable_flat(self):
        return nn.utils.parameters_to_vector(self.parameters()).numpy()

    def set_trainable_flat(self, flat):
        flat = torch.as_tensor(flat, dtype=torch.float32)
        nn.utils.vector_to_parameters(flat, self.parameters())

    def serialize(self):
        state_dict = self.state_dict()
        serialized = []
        for key, value in state_dict.items():
            serialized += value.flatten().tolist()
        return serialized

    def deserialize(self, serialized_state_dict):
        state_dict = self.state_dict()
        deserialized = {}
        idx = 0

        for key, value in state_dict.items():
            size = value.numel()
            deserialized[key] = torch.as_tensor(serialized_state_dict[idx:idx+size]).view_as(value)
            idx += size

        self.load_state_dict(deserialized)

    def set_grad_from_flat(self, gradient):
        idx = 0
        for p in self.parameters():
            step = np.prod(p.shape)
            grad = gradient[idx:idx + step]
            grad = torch.as_tensor(grad, dtype=torch.float32).view_as(p)
            p.backward(grad)
            idx += step

    def get_grad_as_flat(self):
        grad = np.zeros(self.num_params)
        idx = 0
        for p in self.parameters():
            step = np.prod(p.shape)
            if p.grad is not None:
                grad[idx:idx + step] = p.grad.view(-1).numpy()
            else:
                grad[idx:idx + step] = np.zeros(step)
            idx += step
        return grad


    def _init_params(self):
        self._normc_init()

    def _normc_init(self):
        model_gain = 1.0
        action_gain = 0.01
        std = model_gain

        layer_count = 0
        for layer in self.model:
            if hasattr(layer, 'weight'):
                layer_count += 1

        layer_num = 0
        for layer in self.model:
            if hasattr(layer, 'weight'):
                if layer_num == layer_count - 1:
                    std = action_gain

                parameter = layer.weight.data
                shape = parameter.shape
                out = self.rng.randn(*shape).astype(np.float32)
                out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))

                new_weight = (torch.as_tensor(out, dtype=torch.float32) - layer.weight.data).reshape_as(
                    layer.weight.data)
                new_bias = -layer.bias.data.reshape_as(layer.bias.data)
                layer.weight.data += new_weight
                layer.bias.data += new_bias

                layer_num += 1

    def _orthogonal_init(self):
        """
        Orthogonal initialization procedure.
        :return:
        """
        model_gain = 1.0
        action_gain = 0.01

        n_trainable_layers = 0
        for layer in self.model:
            trainable = False
            for p in layer.parameters():
                if p.requires_grad:
                    trainable = True
                    break
            if trainable:
                n_trainable_layers += 1

        i = 0
        for layer in self.model:
            trainable = False
            for p in layer.parameters():
                if p.requires_grad:
                    trainable = True
                    break
            if trainable:
                if i < n_trainable_layers - 1:
                    layer.apply(partial(torch_helpers.init_weights_orthogonal, gain=model_gain))
                else:
                    layer.apply(partial(torch_helpers.init_weights_orthogonal, gain=action_gain))
                i += 1

    def reset(self):
        pass

    def _build_model(self):
        raise NotImplementedError
