# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import client_server_interface_pb2 as client__server__interface__pb2


class CSInterfaceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetConfig = channel.unary_unary(
                '/CSInterface/GetConfig',
                request_serializer=client__server__interface__pb2.Null.SerializeToString,
                response_deserializer=client__server__interface__pb2.Config.FromString,
                )
        self.GetServerState = channel.unary_unary(
                '/CSInterface/GetServerState',
                request_serializer=client__server__interface__pb2.Null.SerializeToString,
                response_deserializer=client__server__interface__pb2.ServerState.FromString,
                )
        self.SubmitReturn = channel.unary_unary(
                '/CSInterface/SubmitReturn',
                request_serializer=client__server__interface__pb2.Return.SerializeToString,
                response_deserializer=client__server__interface__pb2.Null.FromString,
                )
        self.SubmitReturns = channel.unary_unary(
                '/CSInterface/SubmitReturns',
                request_serializer=client__server__interface__pb2.ReturnArray.SerializeToString,
                response_deserializer=client__server__interface__pb2.Null.FromString,
                )


class CSInterfaceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetConfig(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetServerState(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SubmitReturn(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SubmitReturns(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_CSInterfaceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetConfig': grpc.unary_unary_rpc_method_handler(
                    servicer.GetConfig,
                    request_deserializer=client__server__interface__pb2.Null.FromString,
                    response_serializer=client__server__interface__pb2.Config.SerializeToString,
            ),
            'GetServerState': grpc.unary_unary_rpc_method_handler(
                    servicer.GetServerState,
                    request_deserializer=client__server__interface__pb2.Null.FromString,
                    response_serializer=client__server__interface__pb2.ServerState.SerializeToString,
            ),
            'SubmitReturn': grpc.unary_unary_rpc_method_handler(
                    servicer.SubmitReturn,
                    request_deserializer=client__server__interface__pb2.Return.FromString,
                    response_serializer=client__server__interface__pb2.Null.SerializeToString,
            ),
            'SubmitReturns': grpc.unary_unary_rpc_method_handler(
                    servicer.SubmitReturns,
                    request_deserializer=client__server__interface__pb2.ReturnArray.FromString,
                    response_serializer=client__server__interface__pb2.Null.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'CSInterface', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class CSInterface(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetConfig(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/CSInterface/GetConfig',
            client__server__interface__pb2.Null.SerializeToString,
            client__server__interface__pb2.Config.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetServerState(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/CSInterface/GetServerState',
            client__server__interface__pb2.Null.SerializeToString,
            client__server__interface__pb2.ServerState.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SubmitReturn(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/CSInterface/SubmitReturn',
            client__server__interface__pb2.Return.SerializeToString,
            client__server__interface__pb2.Null.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SubmitReturns(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/CSInterface/SubmitReturns',
            client__server__interface__pb2.ReturnArray.SerializeToString,
            client__server__interface__pb2.Null.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
