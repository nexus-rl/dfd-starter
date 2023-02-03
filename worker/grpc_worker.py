import torch
import numpy as np
from networking.server import RPCServer


class GRPCWorker(object):
    def __init__(self, state):
        self.grpc_server = RPCServer(state)

    def collect_returns(self, batch_size=1, current_epoch=None, max_delayed_return=None):
        return self.grpc_server.get_returns_batch(batch_size=batch_size, current_epoch=current_epoch,
                                                  max_delayed_return=max_delayed_return)

    def update(self, state):
        self.grpc_server.update(state)

    def start(self, address, port):
        self.grpc_server.start(address=address, port=port)

    def stop(self):
        self.grpc_server.stop()
