#!/usr/bin/env bash
# Regenerate vfl/proto/vfl_pb2*.py from proto/vfl.proto (requires grpcio-tools).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m grpc_tools.protoc -I proto --python_out=vfl/proto --grpc_python_out=vfl/proto proto/vfl.proto
# grpc_tools emits an absolute import; make it package-relative.
sed -i.bak 's/^import vfl_pb2 as vfl__pb2/from . import vfl_pb2 as vfl__pb2/' vfl/proto/vfl_pb2_grpc.py
rm -f vfl/proto/vfl_pb2_grpc.py.bak
echo "regenerated vfl/proto/vfl_pb2.py and vfl_pb2_grpc.py"
