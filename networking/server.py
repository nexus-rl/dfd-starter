from networking.rpc_misc import client_server_interface_pb2, client_server_interface_pb2_grpc
import numpy as np
import grpc
from concurrent import futures
from learner import FDReturn, FDState
import time


class RPCServer(object):
    MAX_MESSAGE_LENGTH = 1*(1024**3)

    def __init__(self, initial_state):
        self.server_interface = ServerInterface(initial_state)
        self.grpc_server = None

    def update(self, server_state):
        self.server_interface.update(server_state)

    def get_returns_batch(self, batch_size=None, current_epoch=None, max_delayed_return=None):
        return self.server_interface.get_returns_batch(batch_size=batch_size, current_epoch=current_epoch,
                                                       max_delayed_return=max_delayed_return)

    def start(self, max_workers=10, address="localhost", port=50051):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers),
                             options=[('grpc.max_send_message_length', RPCServer.MAX_MESSAGE_LENGTH),
                                      ('grpc.max_receive_message_length', RPCServer.MAX_MESSAGE_LENGTH)],
                             compression=grpc.Compression.Gzip)

        client_server_interface_pb2_grpc.add_CSInterfaceServicer_to_server(
            ServerServicer(self.server_interface), server)

        server.add_insecure_port('{}:{}'.format(address, port))
        server.start()
        self.grpc_server = server

    def stop(self):
        # grace means timeout period?
        self.grpc_server.stop(grace=10)
        self.server_interface.cleanup()


class ServerInterface(object):
    def __init__(self, initial_state):
        self.cfg = None
        self.grpc_cfg = None
        self.epoch = -1
        self.server_state = FDState()
        self.update(initial_state)

        self.waiting_returns = []

    def submit_return(self, ret):
        # print("GOT RETURN")
        # print(ret.epoch)
        # print(ret.encoded_noise)
        # print(ret.reward)
        # print(ret.novelty)
        # print(ret.entropy)
        # print(ret.timesteps)
        # print(ret.is_eval)
        # print(np.shape(ret.eval_states))
        self.waiting_returns.append(ret)

    def get_returns_batch(self, batch_size=None, current_epoch=None, max_delayed_return=None):
        timesteps = 0
        n_delayed = 0
        n_discarded = 0
        n_collected = 0
        rets = []

        # passing bs=None will pull out every waiting return instead of a specific number
        if batch_size is None:
            batch_size = max(len(self.waiting_returns), 1)

        while n_collected < batch_size:
            if len(self.waiting_returns) == 0:
                time.sleep(0.01)
                continue

            ret = self.waiting_returns.pop(-1)
            timesteps += ret.timesteps

            if current_epoch is not None:
                diff = current_epoch - ret.epoch
                if diff > 0:
                    if max_delayed_return is not None and diff > max_delayed_return:
                        n_discarded += 1
                        continue
                    n_delayed += 1

            rets.append(ret)
            if not ret.is_eval:
                n_collected += 1

        return rets, timesteps, n_delayed, n_discarded

    def update(self, server_state):
        self.server_state.epoch = server_state.epoch
        self.server_state.policy_params = server_state.policy_params
        self.server_state.strategy_frames = np.ravel(server_state.strategy_frames).tolist()
        self.server_state.strategy_frames_shape = np.shape(server_state.strategy_frames)
        self.server_state.strategy_history = np.ravel(server_state.strategy_history).tolist()
        self.server_state.strategy_history_shape = np.shape(server_state.strategy_history)
        self.server_state.obs_stats = server_state.obs_stats

        if server_state.experiment_id != self.server_state.experiment_id:
            cfg = server_state.cfg
            self.grpc_cfg = client_server_interface_pb2.Config()
            self.grpc_cfg.params.update(cfg)
            self.cfg = cfg

        self.server_state.experiment_id = server_state.experiment_id

    def cleanup(self):
        self.server_state.cleanup()
        del self.server_state
        del self.waiting_returns
        del self.cfg
        del self.grpc_cfg


class ServerServicer(client_server_interface_pb2_grpc.CSInterfaceServicer):
    def __init__(self, server_interface):
        super().__init__()
        self.null = client_server_interface_pb2.Null()
        self.server_interface = server_interface

    def GetServerState(self, request, context):

        server_interface = self.server_interface
        response = client_server_interface_pb2.ServerState(
            strategy_frames=server_interface.server_state.strategy_frames,
            strategy_frames_shape=server_interface.server_state.strategy_frames_shape,
            strategy_history=server_interface.server_state.strategy_history,
            strategy_history_shape=server_interface.server_state.strategy_history_shape,
            policy_parameters=server_interface.server_state.policy_params,
            epoch=server_interface.server_state.epoch,
            experiment_id=server_interface.server_state.experiment_id,
            obs_stats=server_interface.server_state.obs_stats
        )

        return response

    def GetConfig(self, request, context):
        self.server_interface.grpc_cfg.params["random_seed"] += 1
        print("transmitting config",self.server_interface.grpc_cfg.params["random_seed"])

        # self.server_interface.grpc_cfg["seed"] += 1
        return self.server_interface.grpc_cfg

    def SubmitReturn(self, request, context):
        ret = FDReturn()
        ret.deserialize_from_grpc(request)
        self.server_interface.submit_return(ret)
        return self.null

    def SubmitReturns(self, request, context):
        for serialized_return in request.rets:
            ret = FDReturn()
            ret.deserialize_from_grpc(serialized_return)
            self.server_interface.submit_return(ret)
        return self.null
