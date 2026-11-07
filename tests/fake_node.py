"""Stand-in for `python -m vfl.cli <role> ...` in the deployment tests (no grpc/torch needed).

It serves /healthz on its --ui-port (as every real node's dashboard does), the coordinator variant also serves
/api/state, and every node starts a CHILD process and records its pid, so the test can check that stopping a
deployment leaves no orphans (the child is only reachable through the node's process group).
"""
import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

role = sys.argv[1]
args = sys.argv[2:]
port = int(args[args.index("--ui-port") + 1])
name = args[args.index("--id") + 1] if "--id" in args else role

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
with open(f"{name}.pids", "w") as fh:
    json.dump({"node": os.getpid(), "child": child.pid}, fh)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = {"ok": True, "role": role}
        if self.path.startswith("/api/state"):
            body = {"role": role, "parties": []}
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


print(f"fake {role} {name} on :{port}", flush=True)
time.sleep(0.3)                                  # look like a node that takes a moment to come up
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
