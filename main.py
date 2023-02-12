import os

def compile_protobuf():
    os.system("python -m grpc_tools.protoc -I networking/rpc_misc/proto --python_out=networking/rpc_misc "
              "--grpc_python_out=networking/rpc_misc networking/rpc_misc/proto/client_server_interface.proto")

    modified_file = ""
    file_path = os.path.join("networking","rpc_misc","client_server_interface_pb2_grpc.py")
    with open(file_path, 'r') as f:
        for line in f:
            if "import client_server_interface_pb2" in line:
                modified_file = "{}{}".format(modified_file, "from . import client_server_interface_pb2 as "
                                                             "client__server__interface__pb2")
            else:
                modified_file = "{}{}".format(modified_file, line)

    with open(file_path, 'w') as f:
        f.write(modified_file)


if __name__ == "__main__":
    compile_protobuf()
