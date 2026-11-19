# One image for every role (coordinator / active / passive / tools).
#
# Reproducible, supply-chain-pinned build:
#   * the base image is pinned by digest, not by a mutable tag;
#   * every Python package (including transitive ones) is pinned by requirements.lock.
# Both are refreshed together with:  scripts/pin_deps.sh   (then commit Dockerfile + requirements.lock)
FROM python:3.11-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY requirements.txt requirements.lock ./
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu -c requirements.lock \
 && pip install -r requirements.txt -c requirements.lock \
 && pip check

# Unprivileged runtime user. Every role (coordinator / active / passive / setup) runs as uid 10001.
# Volume mount points are created here, owned by that user, so fresh named volumes inherit the
# ownership (Docker copies it into an empty volume on first mount).
#   /var/lib/vfl                    per-container state (certs, data, logs, runs, checkpoints)
#   /var/lib/vfl-setup/pki          setup only: the CA (ca.key) and the full data report
#   /var/lib/vfl-setup/nodes/<n>    setup only: where it hands each node its own material
RUN groupadd --system --gid 10001 vfl \
 && useradd --system --uid 10001 --gid vfl --home-dir /home/vfl --create-home --shell /usr/sbin/nologin vfl \
 && install -d -o vfl -g vfl -m 0700 /var/lib/vfl /var/lib/vfl-setup /var/lib/vfl-setup/pki \
      /var/lib/vfl-setup/nodes /var/lib/vfl-setup/nodes/coordinator /var/lib/vfl-setup/nodes/active \
 && for i in 1 2 3 4 5 6 7 8; do install -d -o vfl -g vfl -m 0700 /var/lib/vfl-setup/nodes/passive-$i; done

# Application code stays root-owned (read-only to the runtime user).
COPY . .

USER 10001:10001
EXPOSE 50050 50051 8080 8081 8082
ENTRYPOINT ["python", "-m", "vfl.cli"]
