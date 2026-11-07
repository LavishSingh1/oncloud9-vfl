# Running on separate machines (same network or over the internet)

## Who connects to whom

```
 passive-1 ──────► coordinator :50050  ◄────── active
 passive-2 ──────►       ▲
     │                   │ (heartbeats, barriers, telemetry: small)
     └──────────────► active :50051       (representations / gradients: every batch)
```

| node | needs INBOUND | makes OUTBOUND to |
|---|---|---|
| coordinator | TCP 50050 from every party | nothing |
| active | TCP 50051 from every passive | coordinator:50050 |
| passive | nothing | coordinator:50050, active:50051 |
| dashboards | 8080 / 8081 / 808x only if you want to view them from another machine | – |

So only **two machines need to be reachable**: the coordinator and the active party. Passive parties
can sit behind NAT, home routers or mobile hotspots — they only dial out.

## Common preparation (every scenario)

1. Install the **same version of this repo** and the requirements on every machine
   (x86 or ARM, e.g. 64-bit Raspberry Pi OS; the code is identical).
2. On ONE machine create the certificates and the data split:
   ```bash
   python -m vfl.cli certs --out certs --passives 2
   python -m vfl.cli prepare-data --dataset breast_cancer --passives 2 --out data
   ```
3. Copy to each machine **only what it owns**:

   | machine | files |
   |---|---|
   | coordinator | `certs/ca.crt`, `certs/coordinator.crt`, `certs/coordinator.key` |
   | active | `certs/ca.crt`, `certs/active.crt`, `certs/active.key`, `data/active.csv` |
   | passive-i | `certs/ca.crt`, `certs/passive-i.crt`, `certs/passive-i.key`, `data/passive-i.csv` |

   Keep `certs/ca.key` offline (it can mint new identities). Use `scp`, a USB stick, etc.
   Certificates are bound to node **names**, not IP addresses, so they keep working if IPs change.
4. Pick the addresses: `COORD` = the coordinator's address and `ACTIVE` = the active party's address,
   as the *other* machines see them (LAN IP, public IP, VPN IP or DNS name).

## The run commands (identical for every scenario — only COORD / ACTIVE change)

```bash
# coordinator machine
python -m vfl.cli coordinator --listen 0.0.0.0:50050 --tls-dir certs

# active machine
python -m vfl.cli active --id active --data data/active.csv --tls-dir certs \
    --coordinator COORD:50050 --listen 0.0.0.0:50051 --advertise ACTIVE:50051

# each passive machine
python -m vfl.cli passive --id passive-1 --data data/passive-1.csv --tls-dir certs \
    --coordinator COORD:50050
```

`--advertise` is the address the *passive parties* will dial to reach the active party, so it must be
reachable from them (never `localhost` on a multi-machine run).

Check reachability before starting (from a passive machine):
```bash
nc -vz COORD 50050      # must say "succeeded"/"open"
nc -vz ACTIVE 50051
```

Start sessions from the coordinator dashboard or `python -m vfl.cli start --url http://COORD:8080 ...`
(see "Dashboards" below for how to reach it).

---

## Doing the preparation from the UI (v2.1)

`python -m vfl.cli ui --token T` on the experimenter's machine:

1. **Data preparation**: generate the dataset and a certificate set for the number of passive parties (add the
   machines' IPs/hostnames as SANs).
2. **Deployment → Multi-machine bundles**: pick the dataset and certificates, then enter the coordinator,
   active and passive hosts. You get `coordinator.zip`, `active.zip` and `passive-N.zip`. Each holds only that
   machine's certificate and key, `ca.crt`, its own data file and manifest, `run-<node>.sh` with the exact
   command, and a README. The CA private key (`ca.key`) is never bundled, and no machine receives another
   party's data or key. The coordinator's command carries a freshly generated `--ui-token`; the UI shows it to you
   once. Bundle downloads need your operator token because each zip holds a private key.
3. Unzip each bundle into `vfl-system/` on its machine and run its `run-<node>.sh`. The panel lists nodes as
   they register with the coordinator.

The commands in the bundles are the ones in this guide, so the scenarios below apply unchanged.

## Scenario A — same network (LAN / Wi-Fi / lab)

Example: coordinator `192.168.1.10`, active `192.168.1.20`, passives `192.168.1.30`, `192.168.1.31`.

```bash
# 192.168.1.10
python -m vfl.cli coordinator --listen 0.0.0.0:50050 --tls-dir certs \
    --ui-host 0.0.0.0 --ui-token CHANGE_ME
# 192.168.1.20
python -m vfl.cli active --data data/active.csv --tls-dir certs --coordinator 192.168.1.10:50050 \
    --listen 0.0.0.0:50051 --advertise 192.168.1.20:50051 \
    --ui-host 0.0.0.0 --public-ui-url http://192.168.1.20:8081
# 192.168.1.30   (on 192.168.1.31 the same with --id passive-2, data/passive-2.csv, its own URL)
python -m vfl.cli passive --id passive-1 --data data/passive-1.csv --tls-dir certs \
    --coordinator 192.168.1.10:50050 --ui-host 0.0.0.0 --public-ui-url http://192.168.1.30:8082
```

Open `http://192.168.1.10:8080/?token=CHANGE_ME` from any machine on the LAN.

Firewall (if enabled):
```bash
# Ubuntu/Debian (ufw), on the coordinator / active respectively
sudo ufw allow from 192.168.1.0/24 to any port 50050 proto tcp
sudo ufw allow from 192.168.1.0/24 to any port 50051 proto tcp
sudo ufw allow from 192.168.1.0/24 to any port 8080:8083 proto tcp   # only for dashboards
```
Windows: allow Python through Windows Defender Firewall on *Private* networks (prompted on first run).
macOS: accept the "allow incoming connections" prompt.

Gotchas: guest / campus Wi-Fi often has **client isolation** (devices cannot reach each other) — use a
phone hotspot, a wired switch, or Scenario B2. Give the coordinator and active machines DHCP
reservations so their IPs don't change between runs.

## Scenario B — over the internet

Pick one of three ways to make COORD and ACTIVE reachable.

### B1. Overlay VPN — recommended for laptops / Raspberry Pis at different homes

Install [Tailscale](https://tailscale.com/download) (or ZeroTier / WireGuard) on every machine and log
in to the same network. Each machine gets a stable private address (e.g. `100.x.y.z`) or name
(`coord-box`), and works through NAT without router changes. Then run **exactly the Scenario A
commands** using those addresses:

```bash
tailscale ip -4            # on each machine, to find its address
python -m vfl.cli active ... --coordinator 100.101.1.10:50050 --advertise 100.101.1.20:50051
```

Traffic is still protected end-to-end by this system's mTLS, on top of the VPN's own encryption.

### B2. Cloud VMs with public IPs (AWS / GCP / Azure)

Typical: coordinator + active on cloud VMs, passive parties anywhere (home laptop, Pi, another cloud).

* Security group / firewall rules on the VMs:
  * coordinator VM: inbound TCP **50050** from the parties' public IPs (or 0.0.0.0/0 — mTLS still
    rejects anyone without a certificate from your CA)
  * active VM: inbound TCP **50051** from the passive parties
  * do **not** open 8080–8083; use an SSH tunnel for dashboards (below)
* Use the VM's **public IP or DNS name** for `--advertise` and `--coordinator`, the private IP only
  for machines inside the same VPC:
  ```bash
  python -m vfl.cli active ... --coordinator 34.93.10.5:50050 --advertise 34.93.10.6:50051
  ```
* Place VMs in one region close to the passive parties (e.g. `ap-south-1` / `asia-south1` for India)
  to keep RTT low; record region and instance types for the NFR-9 cost table.

### B3. Home router port forwarding (no VPN, no cloud)

On the router in front of the coordinator forward TCP 50050 → coordinator's LAN IP, and in front of the
active party TCP 50051 → active's LAN IP. Use the router's public IP (`curl ifconfig.me`) or a dynamic-DNS
name for COORD / ACTIVE. Will **not** work if your ISP uses CGNAT (common on Indian mobile and many
broadband connections: the router's WAN IP starts with 100.64–100.127 or differs from `ifconfig.me`) —
use B1 instead.

---

## Dashboards on other machines

Safest (works everywhere, nothing exposed): keep the default `--ui-host 127.0.0.1` and tunnel:
```bash
ssh -L 8080:localhost:8080 user@COORD      # then open http://localhost:8080 on your laptop
ssh -L 8081:localhost:8081 user@ACTIVE
```
On a trusted LAN/VPN you can instead use `--ui-host 0.0.0.0 --ui-token SECRET` (token is needed for the
start/stop buttons) and `--public-ui-url http://<that-machine>:<port>` so the coordinator's "open ↗"
links point to the right machine. The dashboards have no TLS or user accounts — don't expose them to
the public internet.

## What changes between the scenarios

| | one machine (`run_local.sh`) | same LAN / Wi-Fi | internet (VPN, cloud, port-forward) |
|---|---|---|---|
| RTT per message | ~0.1 ms | 0.3–5 ms (Wi-Fi adds jitter, spikes 20–100 ms) | 10–300 ms, variable |
| bandwidth | memory speed | 100 Mb/s – 1 Gb/s | 5–100 Mb/s, asymmetric upload at home |
| what dominates batch time | compute | compute + small network | **network round trip** |
| `inject_rtt_ms` for benchmarks | use it to emulate WAN | 0 (or emulate WAN on top) | **0** — the delay is real; measure it with `ping` |
| reachability work | none | firewall rules | VPN, security groups or port forwarding; CGNAT problems |
| security exposure | loopback only | other LAN users | anyone on the internet can reach 50050/50051 → mTLS mandatory, never `--insecure` |
| failure modes | rare | Wi-Fi drop-outs, sleeping laptops | packet loss, NAT timeouts, IP changes, slow ARM devices → timeouts/retries |
| heterogeneity | none (same CPU) | real if devices differ (Pi vs laptop) | real CPU **and** network heterogeneity |

Why the network matters so much: training is synchronous — every batch is one round trip from each
passive party to the active party, and the slowest party sets the pace. Roughly:

```
epoch time ≈ (#train batches) × (RTT + transfer + compute at slowest party) + evaluation
#train batches = ceil(n_train / batch_size)
```

With breast_cancer (≈358 training rows): batch 64 → 6 round trips/epoch, negligible even at 100 ms.
With 20k synthetic rows: batch 32 → ~440 round trips → ~44 s/epoch extra at 100 ms RTT; batch 128 → ~110 → ~11 s.

## Settings to change for internet / slow devices

| setting | LAN | internet / Raspberry Pi | why |
|---|---|---|---|
| `batch_size` | 32–64 | 128–256 | fewer round trips per epoch |
| `codec` | fp32 | fp16 | halves payload (measured −48 % app bytes, no AUROC loss) |
| `rpc_deadline_s` | 30 | 60–120 | avoid false timeouts on a lossy link / slow device |
| `max_retries` | 3 | 5 | ride out short drop-outs |
| `slot_timeout_s` | 120 | 300 | active waits longer for the slowest party |
| `barrier_timeout_s` | 300 | 600 | first epoch on a Pi includes PyTorch warm-up |
| alignment | `dh_psi` takes ~10 s locally | `dh_psi` on a Pi can take minutes for 10k+ ids | PSI is CPU-bound modular exponentiation |

Example: `python -m vfl.cli start --url http://localhost:8080 --set batch_size=128 --set codec=fp16 --set rpc_deadline_s=90 --wait`

Also: disable sleep on laptops during runs; keep machines on mains power; a party that stops
heartbeating for >15 s fails the session — resume with `--resume-from <session>` from the last epoch.

## Benchmarking a real multi-machine deployment

Run from a machine that can reach the coordinator dashboard (e.g. through the SSH tunnel):

```bash
ping -c 20 ACTIVE                                  # record the real RTT from each passive machine
python bench/run_benchmarks.py --url http://localhost:8080 --suites e1,e2,eff,def,e5 \
    --rtts 0 --batch-sizes 32,64,128 --data-dir data --out results/lan
python bench/security_test.py --out results/security   # run on one machine (it starts its own local nodes)
python bench/analyze.py results/lan --security results/security/security_report.json
```

`--rtts 0` means "no extra emulated delay" — the measured latency is the real network. You can still
add emulated delay on top (`--rtts 0,50,100`) to extrapolate to worse links. `--data-dir` (with all
CSVs) is only needed on the benchmarking machine for the B0/B1 baselines, which deliberately read every
party's data in one place — experiments only. In remote mode wire bytes are not captured (the local
relay isn't in the path); application and tensor bytes still are. For wire bytes on a real link, put
the relay in front of the active party:

```bash
# on the active machine: active listens on 50052, relay exposes 50051 to the passives
python -m vfl.cli netem --listen 0.0.0.0:50051 --target 127.0.0.1:50052 --stats net.json
python -m vfl.cli active ... --listen 127.0.0.1:50052 --advertise ACTIVE:50051
```

Record in your report: machine types/CPU/RAM, OS, Python/torch versions (already in every
`run_summary.json`), connection type (Ethernet/Wi-Fi/4G/VPN), measured ping RTT, and region for cloud runs.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| party log: `coordinator unreachable … UNAVAILABLE` | wrong COORD address, port 50050 blocked, or coordinator not running — `nc -vz COORD 50050` |
| passive fails in alignment with `UNAVAILABLE` | `--advertise` on the active party is not reachable from the passive (often left as `localhost`) — `nc -vz ACTIVE 50051` |
| `certificate verify failed` / handshake error | machines hold certificates from different CA runs — recopy `ca.crt` and node certs from the same `certs/` folder |
| `PERMISSION_DENIED … certificate identity` | `--id` does not match the certificate name (`passive-2` using `passive-1.crt`) |
| `batch membership digest mismatch` | parties hold different data versions — redistribute CSVs from one `prepare-data` run |
| many `DEADLINE_EXCEEDED … retrying` | slow link or device — raise `rpc_deadline_s`, increase `batch_size` |
| `stopped heartbeating` | machine slept, Wi-Fi dropped, or process killed — fix, restart node, resume session |
| works on LAN, not over internet | CGNAT or router forwarding missing — use Tailscale (B1) |

## Enrollment (PKI mode, v2)

For deployments where distributing certificates by hand is awkward, run a PKI instead of the static CA.

**Set up (once, on a trusted machine):**
```bash
python -m vfl.cli certs --mode pki --domain oncloud9 --out certs    # root + intermediate + node certs
# MOVE certs/root/ca.key OFFLINE afterwards — it is only needed to issue a new intermediate.
```
Copy `ca.crt`, `intermediate.crt`, `intermediate.key` to the coordinator. Give the active party and any
hand-provisioned parties their `<node>.crt`/`.key` as usual, or let parties enrol (below).

**Coordinator with enrollment:**
```bash
python -m vfl.cli coordinator --tls-dir certs --join-mode token \
    --enroll-listen 0.0.0.0:50049 --cert-hours 24 --ui-host 0.0.0.0 --ui-token T
```
Join modes: `static` (enrollment off), `token` (single-use expiring token), `approval` (operator approves
each request on the Identities tab).

**A new passive party enrols (generates its own key):**
```bash
# operator creates a token:
python -m vfl.cli token create --url http://COORD:8080 --token T --role passive
# the party uses it (certificate is installed into its --tls-dir):
python -m vfl.cli passive --id passive-2 --data data/passive-2.csv --coordinator COORD:50050 \
    --tls-dir certs --enroll COORD:50049 --enroll-token vfl1.xxxx --enroll-url http://COORD:8080
```
The token carries the root CA fingerprint, so the party refuses to enrol against the wrong coordinator.
Certificates are short-lived and renewed automatically (new key each time). Revoke a party from the
Identities tab; revoked serials are rejected by every node's gate and pushed to parties in heartbeats.

**Hardening flags** (any node): `--rate-limit-rps`, `--max-peer-concurrency`, `--max-concurrent-rpcs`,
`--ui-users FILE` (dashboard logins; `vfl.cli ui-user add NAME --role operator`), `--ui-tls` (HTTPS
dashboard with the node certificate). Keep dashboards off the public internet regardless.

## Security notes (v2.3)

* **Dashboards through DNS names.** The dashboards refuse Host names they do not know (DNS-rebinding protection).
  IP addresses and `localhost` always work. For names such as a Tailscale MagicDNS name, start the node with
  `VFL_UI_ALLOWED_HOSTS=coord-box,coord-box.tailnet.ts.net`.
* **Enrollment over the network.** `--enroll-url` (fetching the root CA from the dashboard) needs a pin: the join
  token, or `--ca-sha256 <fingerprint printed by certs --mode pki>`. Otherwise pass `--ca-file`.
* **Tokens.** Pass the operator token as `VFL_UI_TOKEN` in the environment rather than `--ui-token`, so it does not
  appear in `ps` on shared machines. The UI bundles already do this.
* **`--insecure`** only listens on loopback. Every multi-machine setup uses mTLS.
* **Certificates from v2.2.** Re-run `python -m vfl.cli certs ...` so static certificates carry their role. Old ones
  keep working, with the role taken from the name, but a custom `--active-id` then needs new certificates.
