from policies import DiscretePolicy

import torch
import torch.nn as nn
from torch.nn import functional as F


class ImpalaPolicy(DiscretePolicy):
    def __init__(self, n_inputs, n_actions, seed=124):
        super().__init__(n_inputs, n_actions, seed=seed)

    def compute_vbn(self, buffer):
        buffer = self._get_stacked_obs(buffer)
        self.train()
        self.model(buffer)
        self.eval()

    def forward(self, x):
        return self.model(x)

    def get_entropy(self, x):
        return super().get_entropy(self._get_stacked_obs(x))

    def get_strategy(self, x):
        x = self._get_stacked_obs(x)
        prob_dist, = self.model(x)
        return prob_dist.view(-1, self.output_shape).numpy()

    def reset(self):
        self.model[0].state = self.model[0].initial_state()

    def _build_model(self):
        self.model = nn.Sequential(ImpalaCNN(self.output_shape, use_lstm=True), nn.Softmax(dim=-1))

    def _get_stacked_obs(self, x):
        obs = []
        rews = []
        dones = []

        for arg in x:
            obs.append(arg["frame"])
            rews.append(arg["reward"])
            dones.append(arg["done"])

        return {"frame":torch.cat(obs), "reward":torch.cat(rews), "done":torch.cat(dones)}


# Modified version of the Impala CNN implementation from TorchBeast
# Source: https://github.com/facebookresearch/torchbeast/blob/main/torchbeast/polybeast_learner.py
class ImpalaCNN(nn.Module):
    def __init__(self, num_actions, use_lstm=False):
        super(ImpalaCNN, self).__init__()
        self.num_actions = num_actions
        self.use_lstm = use_lstm

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        input_channels = 3
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(nn.BatchNorm2d(input_channels))
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.BatchNorm2d(input_channels))
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.BatchNorm2d(input_channels))
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        self.fc = nn.Sequential(nn.BatchNorm1d(2048), nn.Linear(2048, 256))

        # FC output size + last reward.
        core_output_size = self.fc[1].out_features + 1

        if use_lstm:
            self.core = nn.LSTM(core_output_size, 256, num_layers=1, batch_first=True)
            core_output_size = 256

        self.policy = nn.Sequential(nn.BatchNorm1d(core_output_size), nn.Linear(core_output_size, self.num_actions))
        self.softmax = nn.Softmax(dim=-1)
        # self.value_estimator = nn.Linear(core_output_size, 1)

        self.state = self.initial_state()

    def initial_state(self, batch_size=1):
        if not self.use_lstm:
            return tuple()

        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2))

    def forward(self, inputs, core_state=None, deterministic=False):
        if core_state is None:
            core_state = self.state
        x = inputs["frame"]
        B, T, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.view(T*B, -1)
        x = F.relu(self.fc(x))

        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for inp, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = (nd.mul(core_state[0]), nd.mul(core_state[1]))

                output, core_state = self.core(inp.unsqueeze(0), core_state)

                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input

        policy_logits = self.policy(core_output)
        # value_estimate = self.value_estimator(core_output)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        # value_estimate = value_estimate.view(T, B)
        self.state = core_state

        return policy_logits
