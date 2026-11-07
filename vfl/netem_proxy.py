"""Transparent TCP relay for network experiments (no root needed).

  * adds a fixed one-way delay in each direction (RTT = 2 x delay) while preserving order
  * counts on-the-wire bytes (application + gRPC/HTTP2 framing + TLS) for NFR-4
  * optionally captures raw bytes to a file so they can be scanned for plaintext (NFR-7)

Put it between a passive party and the active party:
    python -m vfl.cli netem --listen 0.0.0.0:60051 --target ACTIVE_HOST:50051 --rtt-ms 50
and register the active party with --advertise PROXY_HOST:60051 (mTLS still verifies the
active party's certificate because clients pin the expected node name, not the address).

For kernel-level emulation on Linux you can instead use:
    sudo tc qdisc add dev eth0 root netem delay 25ms     # +25 ms each way on this host
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

log = logging.getLogger("vfl.netem")


class Stats:
    def __init__(self):
        self.up = 0
        self.down = 0
        self.connections = 0
        self.t0 = time.time()

    def as_dict(self):
        return {"bytes_client_to_server": self.up, "bytes_server_to_client": self.down,
                "bytes_total": self.up + self.down, "connections": self.connections, "since": self.t0,
                "updated": time.time()}


async def _pipe(reader, writer, delay_s, counter_attr, stats, capture):
    queue: asyncio.Queue = asyncio.Queue()

    async def sender():
        while True:
            due, chunk = await queue.get()
            if chunk is None:
                break
            wait = due - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            writer.write(chunk)
            await writer.drain()
        try:
            writer.write_eof()
        except Exception:
            pass

    task = asyncio.create_task(sender())
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            setattr(stats, counter_attr, getattr(stats, counter_attr) + len(chunk))
            if capture:
                capture.write(chunk)
            await queue.put((time.monotonic() + delay_s, chunk))
    except (ConnectionResetError, asyncio.IncompleteReadError):
        pass
    await queue.put((0, None))
    await task


async def _serve(listen_host, listen_port, target_host, target_port, delay_s, stats, capture, stats_path):
    async def handle(cr, cw):
        stats.connections += 1
        try:
            sr, sw = await asyncio.open_connection(target_host, target_port)
        except OSError as e:
            log.error("cannot reach target %s:%s: %s", target_host, target_port, e)
            cw.close()
            return
        await asyncio.gather(_pipe(cr, sw, delay_s, "up", stats, capture),
                             _pipe(sr, cw, delay_s, "down", stats, capture))
        for w in (cw, sw):
            try:
                w.close()
            except Exception:
                pass

    server = await asyncio.start_server(handle, listen_host, listen_port)
    log.info("netem relay %s:%d -> %s:%d with %.1f ms one-way delay (RTT +%.1f ms)", listen_host, listen_port,
             target_host, target_port, delay_s * 1000, delay_s * 2000)

    async def dump():
        while True:
            await asyncio.sleep(2)
            if stats_path:
                with open(stats_path, "w") as fh:
                    json.dump(stats.as_dict(), fh)
            if capture:
                capture.flush()

    asyncio.create_task(dump())
    async with server:
        await server.serve_forever()


def run(listen: str, target: str, rtt_ms: float, stats_path: str | None, capture_path: str | None):
    lh, lp = listen.rsplit(":", 1)
    th, tp = target.rsplit(":", 1)
    capture = open(capture_path, "ab") if capture_path else None
    stats = Stats()
    try:
        asyncio.run(_serve(lh, int(lp), th, int(tp), rtt_ms / 2000.0, stats, capture, stats_path))
    except KeyboardInterrupt:
        pass
    finally:
        if capture:
            capture.close()
        if stats_path:
            with open(stats_path, "w") as fh:
                json.dump(stats.as_dict(), fh)
