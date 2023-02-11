import grpc
from networking.port_resolver import resolve_port
from networking.rpc_misc import client_server_interface_pb2, client_server_interface_pb2_grpc
from google.protobuf import json_format
import numpy as np
from learner import FDState
import time
import traceback
from networking.server import RPCServer


class RPCClient(object):
    OPERATION_SUCCESSFUL_FLAG = 0
    NEW_STATE_FLAG = 1
    NEW_EXPERIMENT_FLAG = 2
    RPC_FAILED_FLAG = 3

    def __init__(self):
        self.comm_pipe = None
        self.channel = None
        self.null = client_server_interface_pb2.Null()
        self.current_state = FDState()

    def connect(self, address="localhost", port=None):
        self.channel = grpc.insecure_channel(resolve_port(address, port),
                                             options=[('grpc.max_send_message_length', RPCServer.MAX_MESSAGE_LENGTH),
                                                      ('grpc.max_receive_message_length', RPCServer.MAX_MESSAGE_LENGTH)],
                                             compression=grpc.Compression.Gzip)

        self.comm_pipe = client_server_interface_pb2_grpc.CSInterfaceStub(self.channel)

    def submit_return(self, ret):
        try:
            self.comm_pipe.SubmitReturn(ret.serialize_to_grpc())
        except:
            print("FAILED TO SEND SINGLE RETURN TO SERVER")
            print(traceback.format_exc())
            time.sleep(1)

    def submit_returns(self, returns):
        try:
            returns_msg = client_server_interface_pb2.ReturnArray(
                rets=[ret.serialize_to_grpc() for ret in returns]
            )
            self.comm_pipe.SubmitReturns(returns_msg)
        except:
            print("FAILED TO SEND RETURNS ARRAY TO SERVER")
            print(traceback.format_exc())
            time.sleep(1)

    def _update_cfg(self):
        try:
            cfg_msg = self.comm_pipe.GetConfig(self.null)
            self.current_state.cfg = json_format.MessageToDict(cfg_msg)["params"]
        except:
            print("FAILED TO RECEIVE CONFIG FROM SERVER")
            print(traceback.format_exc())
            return self.RPC_FAILED_FLAG

        return self.OPERATION_SUCCESSFUL_FLAG

    def _update_state(self, state):
        current_state = self.current_state
        current_state.strategy_frames = np.reshape(state.strategy_frames, state.strategy_frames_shape)
        current_state.strategy_history = np.reshape(state.strategy_history, state.strategy_history_shape)
        current_state.policy_params = np.asarray(state.policy_parameters)
        current_state.obs_stats = state.obs_stats
        current_state.epoch = state.epoch
        current_state.experiment_id = state.experiment_id

    def get_server_state(self):
        try:
            state = self.comm_pipe.GetServerState(self.null)
        except:
            print("FAILED TO RECEIVE STATE FROM SERVER")
            print(traceback.format_exc())
            return self.RPC_FAILED_FLAG

        if self.current_state is None or state.experiment_id != self.current_state.experiment_id:
            self._update_cfg()
            self._update_state(state)
            print("NEW EXPERIMENT ID RECEIVED!")
            return self.NEW_EXPERIMENT_FLAG

        if state.epoch != self.current_state.epoch:
            self._update_state(state)
            return self.NEW_STATE_FLAG

        return self.OPERATION_SUCCESSFUL_FLAG

    def disconnect(self):
        self.current_state.cleanup()
        self.channel.close()
