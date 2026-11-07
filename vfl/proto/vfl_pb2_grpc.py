# Generated from proto/vfl.proto. Regenerate with: bash scripts/gen_proto.sh
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import vfl_pb2 as vfl__pb2

class CoordinatorStub(object):
    def __init__(self, channel):
        self.Register = channel.unary_unary(
            '/vfl.Coordinator/Register',
            request_serializer=vfl__pb2.RegisterRequest.SerializeToString,
            response_deserializer=vfl__pb2.RegisterReply.FromString)
        self.Heartbeat = channel.unary_unary(
            '/vfl.Coordinator/Heartbeat',
            request_serializer=vfl__pb2.HeartbeatRequest.SerializeToString,
            response_deserializer=vfl__pb2.HeartbeatReply.FromString)
        self.Barrier = channel.unary_unary(
            '/vfl.Coordinator/Barrier',
            request_serializer=vfl__pb2.BarrierRequest.SerializeToString,
            response_deserializer=vfl__pb2.BarrierReply.FromString)
        self.Report = channel.unary_unary(
            '/vfl.Coordinator/Report',
            request_serializer=vfl__pb2.ReportRequest.SerializeToString,
            response_deserializer=vfl__pb2.ReportReply.FromString)

class CoordinatorServicer(object):
    def Register(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')
    def Heartbeat(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')
    def Barrier(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')
    def Report(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

def add_CoordinatorServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'Register': grpc.unary_unary_rpc_method_handler(
            servicer.Register,
            request_deserializer=vfl__pb2.RegisterRequest.FromString,
            response_serializer=vfl__pb2.RegisterReply.SerializeToString),
        'Heartbeat': grpc.unary_unary_rpc_method_handler(
            servicer.Heartbeat,
            request_deserializer=vfl__pb2.HeartbeatRequest.FromString,
            response_serializer=vfl__pb2.HeartbeatReply.SerializeToString),
        'Barrier': grpc.unary_unary_rpc_method_handler(
            servicer.Barrier,
            request_deserializer=vfl__pb2.BarrierRequest.FromString,
            response_serializer=vfl__pb2.BarrierReply.SerializeToString),
        'Report': grpc.unary_unary_rpc_method_handler(
            servicer.Report,
            request_deserializer=vfl__pb2.ReportRequest.FromString,
            response_serializer=vfl__pb2.ReportReply.SerializeToString),
    }
    generic_handler = grpc.method_handlers_generic_handler('vfl.Coordinator', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))

class ActivePartyStub(object):
    def __init__(self, channel):
        self.Align = channel.unary_unary(
            '/vfl.ActiveParty/Align',
            request_serializer=vfl__pb2.AlignRequest.SerializeToString,
            response_deserializer=vfl__pb2.AlignReply.FromString)
        self.Exchange = channel.unary_unary(
            '/vfl.ActiveParty/Exchange',
            request_serializer=vfl__pb2.ExchangeRequest.SerializeToString,
            response_deserializer=vfl__pb2.ExchangeReply.FromString)
        self.Ping = channel.unary_unary(
            '/vfl.ActiveParty/Ping',
            request_serializer=vfl__pb2.PingRequest.SerializeToString,
            response_deserializer=vfl__pb2.PingReply.FromString)

class ActivePartyServicer(object):
    def Align(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')
    def Exchange(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')
    def Ping(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

def add_ActivePartyServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'Align': grpc.unary_unary_rpc_method_handler(
            servicer.Align,
            request_deserializer=vfl__pb2.AlignRequest.FromString,
            response_serializer=vfl__pb2.AlignReply.SerializeToString),
        'Exchange': grpc.unary_unary_rpc_method_handler(
            servicer.Exchange,
            request_deserializer=vfl__pb2.ExchangeRequest.FromString,
            response_serializer=vfl__pb2.ExchangeReply.SerializeToString),
        'Ping': grpc.unary_unary_rpc_method_handler(
            servicer.Ping,
            request_deserializer=vfl__pb2.PingRequest.FromString,
            response_serializer=vfl__pb2.PingReply.SerializeToString),
    }
    generic_handler = grpc.method_handlers_generic_handler('vfl.ActiveParty', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))

class EnrollmentStub(object):
    def __init__(self, channel):
        self.Enroll = channel.unary_unary(
            '/vfl.Enrollment/Enroll',
            request_serializer=vfl__pb2.EnrollRequest.SerializeToString,
            response_deserializer=vfl__pb2.EnrollReply.FromString)
        self.Renew = channel.unary_unary(
            '/vfl.Enrollment/Renew',
            request_serializer=vfl__pb2.RenewRequest.SerializeToString,
            response_deserializer=vfl__pb2.EnrollReply.FromString)

class EnrollmentServicer(object):
    def Enroll(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')
    def Renew(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

def add_EnrollmentServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'Enroll': grpc.unary_unary_rpc_method_handler(
            servicer.Enroll,
            request_deserializer=vfl__pb2.EnrollRequest.FromString,
            response_serializer=vfl__pb2.EnrollReply.SerializeToString),
        'Renew': grpc.unary_unary_rpc_method_handler(
            servicer.Renew,
            request_deserializer=vfl__pb2.RenewRequest.FromString,
            response_serializer=vfl__pb2.EnrollReply.SerializeToString),
    }
    generic_handler = grpc.method_handlers_generic_handler('vfl.Enrollment', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))

