#!/usr/bin/env python3
# live_stream_eval_partial.py
"""
Live angle + force prediction with a web dashboard, from ONE growing file.

Pipeline
--------
partial_capture.raw  ->  EventWakeAngleModel (frozen)  ->  angle (deg)
                     ->  ForceHead (force_head.pt)     ->  [CH3, CH4]

What it does
------------
1. Follows the growing file
       <session-base>/<...>/partial_capture.raw
   (the newest one under --session-base, or the one given with --file).
   The file is only READ, never modified. When the live stream is switched off
   the infra renames it to capture.raw in the same folder. That is detected,
   the remaining events are processed, and the script then waits for the next
   partial_capture.raw.
2. Each time the file has grown, the bytes written so far are decoded with your
   existing `load_events_from_raw`. The decoded stream is cut into consecutive,
   NON-overlapping packets of --packet-sec (default 0.8 s). A packet is emitted
   only once an event with a later timestamp exists, so it is complete.
3. Each packet is appended to a short rolling buffer and the trained
   EventWakeAngleModel runs on that buffer. The smoothed angle is then passed
   through the force head to get the predicted forces.
4. Serves a dashboard (stdlib HTTP server + Server-Sent Events, no extra pip
   packages): angle readout + force readout, an angle chart (blue) with the
   force chart (orange) below it, and a table of recent packets.

Run (from the folder that has config.py / data.py / model.py / force_model.py):
    python live_stream_eval_partial.py --checkpoint best_angle_model.pt --force-checkpoint force_head.pt
Then open  http://localhost:8000  (or http://<this-machine-ip>:8000).

Useful flags:
    --file PATH           use a fixed partial_capture.raw instead of auto-picking
                          the newest one under --session-base
    --packet-sec 0.8      packet length in seconds (no overlap)
    --skip-existing       ignore data already in the file, only new data
    --poll-ms 400         how often to look for new bytes / re-decode
    --port 8000 --host 0.0.0.0
"""

import argparse
import glob
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DEFAULT_SESSION_BASE = "/home/gupta/Desktop/streaming-infra/received_directory"


# --------------------------------------------------------------------------
# Shared state between the inference loop and the web clients
# --------------------------------------------------------------------------
class Hub:
    def __init__(self, history: int = 600):
        self.lock = threading.Lock()
        self.history = deque(maxlen=history)
        self.meta = {"status": "starting", "processed": 0, "errors": 0}
        self.subscribers = []

    def _broadcast(self, kind, payload):
        with self.lock:
            subs = list(self.subscribers)
        for q in subs:
            try:
                q.put_nowait((kind, payload))
            except queue.Full:
                pass  # slow client; it will catch up on the next event

    def set_meta(self, **kw):
        with self.lock:
            self.meta.update(kw)
        self._broadcast("meta", kw)

    def publish(self, record):
        with self.lock:
            self.history.append(record)
            self.meta["processed"] += 1
            if record.get("error"):
                self.meta["errors"] += 1
            counters = {k: self.meta[k] for k in ("processed", "errors")}
        self._broadcast("meta", counters)
        self._broadcast("packet", record)

    def reset(self, **meta):
        with self.lock:
            self.history.clear()
            self.meta.update(meta, processed=0, errors=0)
            snap_meta = dict(self.meta)
        self._broadcast("reset", snap_meta)

    def subscribe(self):
        q = queue.Queue(maxsize=1000)
        with self.lock:
            self.subscribers.append(q)
            snap = {"meta": dict(self.meta), "history": list(self.history)}
        return q, snap

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def snapshot(self):
        with self.lock:
            return {"meta": dict(self.meta), "history": list(self.history)}


# --------------------------------------------------------------------------
# Session discovery (partial_capture.raw -> capture.raw)
# --------------------------------------------------------------------------
def find_session(args):
    """Return (partial_path, final_path) of the newest session, or None."""
    if args.file:
        p = Path(args.file)
        return p, p.with_name("capture.raw")
    cands = glob.glob(os.path.join(args.session_base, "**", "partial_capture.raw"),
                      recursive=True)
    if not cands:
        return None
    p = Path(max(cands, key=os.path.getmtime))
    return p, p.with_name("capture.raw")


# --------------------------------------------------------------------------
# Snapshot decoding (source file is only read)
# --------------------------------------------------------------------------
def snapshot_decode(mods, cfg, src: Path, size: int, tmp_dir: str):
    """Copy the first `size` bytes of the growing file to a temp .raw file and
    decode it. The writer may be mid-word at `size`, so on a decode error the
    copy is trimmed by 1..7 bytes and retried."""
    tmp = Path(tmp_dir) / "snapshot.raw"
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        remaining = size
        while remaining > 0:
            chunk = fi.read(min(8 << 20, remaining))
            if not chunk:
                break
            fo.write(chunk)
            remaining -= len(chunk)
    n = os.path.getsize(tmp)
    last_err = None
    for trim in range(8):
        if n - trim <= 0:
            break
        if trim:
            os.truncate(tmp, n - trim)
        try:
            return mods.load_events_from_raw(str(tmp), cfg)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise last_err if last_err else RuntimeError("empty snapshot")


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def denormalize(values, config):
    if not config.normalize_angle_target:
        return values
    return values * (config.angle_max - config.angle_min) + config.angle_min


def predict_buffer(torch, DataLoader, mods, model, buf, ev_dtype, h, w, cfg, device, tail):
    """Run the model on the rolling buffer.
    Returns (angle_deg, last_deg, n_samples) or None while warming up."""
    ev = buf[np.argsort(buf[:, 2], kind="stable")].copy()
    ev[:, 2] -= ev[0, 2]  # eval_duration.py recordings start at t=0
    ev = ev.astype(ev_dtype, copy=False)

    rec = mods.Recording("live", 0.0, ev, h, w)  # angle is a dummy label
    try:
        ds = mods.AngleDataset([rec], cfg, name="live")
    except RuntimeError:
        return None  # not enough history yet
    if len(ds) == 0:
        return None

    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))
    preds = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["events"].to(device), batch["event_mask"].to(device))
            preds.append(out["angle"].cpu().numpy())
    pred_deg = denormalize(np.concatenate(preds), cfg)
    return (float(np.median(pred_deg[-tail:])), float(pred_deg[-1]), int(len(pred_deg)))


def load_force_head(torch, path: Path, ForceHead):
    """Load force_head.pt saved by train_force.py. Returns (head, channel_names)
    or (None, default_names) if the file does not exist."""
    if not path.exists():
        print(f"[warn] force checkpoint not found: {path} -> force prediction disabled",
              file=sys.stderr)
        return None, ["CH3", "CH4"]
    ck = torch.load(path, map_location="cpu", weights_only=False)
    head = ForceHead(ck["n_outputs"], ck["hidden"], ck["angle_scale"])
    head.load_state_dict(ck["state_dict"])        # includes the scaler buffers
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    channels = list(ck.get("channels", [f"CH{i + 1}" for i in range(ck["n_outputs"])]))
    return head, channels


# --------------------------------------------------------------------------
# Per-session rolling buffer + one-packet processing
# --------------------------------------------------------------------------
class Stream:
    def __init__(self, args, hub, torch, DataLoader, mods, model, cfg, device,
                 force_head=None):
        self.a, self.hub, self.torch, self.DL = args, hub, torch, DataLoader
        self.mods, self.model, self.cfg, self.device = mods, model, cfg, device
        self.force_head = force_head
        self.buffer_sec = args.buffer_sec or (float(getattr(cfg, "required_history", 1.0)) + 1.0)
        self.buf = None
        self.h = self.w = None
        self.ev_dtype = None

    def predict_force(self, angle_deg):
        """angle (deg) -> list of forces in physical units. The angle is clamped to the
        range the angle model was trained on, so the head never extrapolates."""
        a = float(np.clip(angle_deg, self.cfg.angle_min, self.cfg.angle_max))
        with self.torch.no_grad():
            f = self.force_head(self.torch.tensor([a], dtype=self.torch.float32))
        return [round(float(v), 6) for v in f[0].numpy()]

    def run_packet(self, name, lo, hi, ev_raw, backlog, decode_ms=None, final=False):
        rec = {"packet": name, "wall": time.time(), "angle": None, "angle_last": None,
               "force": None, "n_events": 0, "infer_ms": None, "decode_ms": decode_ms,
               "t_last": None, "note": "", "error": ""}
        try:
            if len(ev_raw):
                ev = ev_raw.astype(np.float64)
                rec["t_last"] = round(float(ev[:, 2].max()), 6)
                rec["n_events"] = int(len(ev))
                self.buf = ev if self.buf is None else np.concatenate([self.buf, ev])
            # stream clock = end of this window (also advances through empty windows)
            if self.buf is not None:
                self.buf = self.buf[self.buf[:, 2] >= hi - self.buffer_sec]

            if len(ev_raw) == 0:
                rec["note"] = f"no events in window [{lo:.3f}, {hi:.3f})"
            elif backlog > self.a.max_backlog:
                rec["note"] = f"catching up ({backlog} packets behind), inference skipped"
            elif self.buf is None or len(self.buf) == 0:
                rec["note"] = "buffer empty"
            else:
                t1 = time.perf_counter()
                res = predict_buffer(self.torch, self.DL, self.mods, self.model, self.buf,
                                     self.ev_dtype, self.h, self.w, self.cfg, self.device,
                                     self.a.smooth)
                if res is not None and self.force_head is not None:
                    rec["force"] = self.predict_force(res[0])
                rec["infer_ms"] = round((time.perf_counter() - t1) * 1000, 1)
                if res is None:
                    rec["note"] = "warming up (buffer shorter than required history)"
                else:
                    rec["angle"] = round(res[0], 3)
                    rec["angle_last"] = round(res[1], 3)
                    rec["note"] = f"{res[2]} windows" + (" (final partial packet)" if final else "")
        except Exception as e:  # noqa: BLE001  keep the stream alive on a bad packet
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"[error] {name}: {rec['error']}", file=sys.stderr)

        self.hub.publish(rec)
        if rec["angle"] is not None:
            fstr = ""
            if rec["force"] is not None:
                fstr = "  force=" + ", ".join(f"{v:+.5f}" for v in rec["force"])
            print(f"{name}  angle={rec['angle']:.2f} deg{fstr}  events={rec['n_events']}  "
                  f"infer={rec['infer_ms']} ms")


# --------------------------------------------------------------------------
# Follow one session until partial_capture.raw turns into capture.raw
# --------------------------------------------------------------------------
def run_session(args, hub, partial: Path, final: Path, tmp_dir, st: Stream, mods, cfg):
    W = args.packet_sec
    hub.reset(status="live", session=partial.parent.name)
    print(f"[follow] {partial}")

    t0 = None          # stream time where packet 0 starts
    k = 0              # index of the next packet to emit
    pos = 0            # number of decoded events already consumed
    last_size = -1
    last_warn = 0.0
    gone_since = None

    while True:
        is_final = (not partial.exists()) and final.exists()
        if not is_final and not partial.exists():
            gone_since = gone_since or time.time()
            if time.time() - gone_since > 5:
                print("[follow] file vanished without capture.raw, ending session")
                return
            time.sleep(0.2)
            continue
        gone_since = None

        src = final if is_final else partial
        try:
            size = src.stat().st_size
        except FileNotFoundError:
            continue  # renamed between exists() and stat(): re-check
        if not is_final and size == last_size:
            time.sleep(args.poll_ms / 1000.0)
            continue
        if size == 0:
            time.sleep(args.poll_ms / 1000.0)
            continue

        try:
            t_dec = time.perf_counter()
            if is_final:
                events, h_, w_ = mods.load_events_from_raw(str(final), cfg)
            else:
                events, h_, w_ = snapshot_decode(mods, cfg, src, size, tmp_dir)
            decode_ms = round((time.perf_counter() - t_dec) * 1000, 1)
        except FileNotFoundError:
            continue
        except Exception as e:  # noqa: BLE001  (header not complete yet, etc.)
            if time.time() - last_warn > 5:
                print(f"[wait] cannot decode {src.name} yet ({size} bytes): "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
                last_warn = time.time()
            time.sleep(args.poll_ms / 1000.0)
            continue

        last_size = size
        events = np.asarray(events)
        if events.size == 0:
            if is_final:
                return
            time.sleep(args.poll_ms / 1000.0)
            continue
        st.h, st.w, st.ev_dtype = h_, w_, events.dtype

        if t0 is None:
            if args.skip_existing and not is_final:
                t0 = float(events[:, 2].max())   # start from "now"
                pos = len(events)
            else:
                t0 = float(events[:, 2].min())

        tail = events[pos:]
        if len(tail) == 0:
            if is_final:
                return
            time.sleep(args.poll_ms / 1000.0)
            continue
        t = tail[:, 2].astype(np.float64)
        if len(t) > 1 and np.any(np.diff(t) < 0):          # defensive: keep time order
            order = np.argsort(t, kind="stable")
            tail, t = tail[order], t[order]
        tmax = float(t[-1])

        # cut consecutive non-overlapping packets; a packet is complete once an
        # event at/after its end exists
        packets, start = [], 0
        while tmax >= t0 + (k + 1) * W:
            lo, hi = t0 + k * W, t0 + (k + 1) * W
            cut = int(np.searchsorted(t, hi, side="left"))
            packets.append((k, lo, hi, tail[start:cut], False))
            start, k = cut, k + 1
        if is_final and start < len(t):                     # last, shorter packet
            lo = t0 + k * W
            packets.append((k, lo, tmax, tail[start:], True))
            start, k = len(t), k + 1
        pos += start

        for i, (idx, lo, hi, ev, fin) in enumerate(packets):
            st.run_packet(f"packet_{idx:06d}", lo, hi, ev, len(packets) - i - 1,
                          decode_ms=decode_ms, final=fin)

        if is_final:
            print(f"[done] {final}")
            hub.set_meta(status="session finished (capture.raw written)")
            return
        time.sleep(args.poll_ms / 1000.0)


def follow(args, hub, torch, DataLoader, mods, model, cfg, device, force_head):
    tmp_dir = args.tmp_dir or ("/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir())
    tmp_dir = tempfile.mkdtemp(prefix="partial_snap_", dir=tmp_dir)
    finished, last_idle = set(), 0.0
    try:
        while True:
            found = find_session(args)
            if found is not None and not found[0].exists():
                finished.discard(str(found[0]))   # allows a new session at the same path (--file)
            if found is None or str(found[0]) in finished or not found[0].exists():
                if time.time() - last_idle > 2:
                    hub.set_meta(status="waiting for partial_capture.raw", session=None)
                    last_idle = time.time()
                time.sleep(0.5)
                continue
            partial, final = found
            st = Stream(args, hub, torch, DataLoader, mods, model, cfg, device, force_head)
            run_session(args, hub, partial, final, tmp_dir, st, mods, cfg)
            finished.add(str(partial))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# Web server (stdlib only): "/" dashboard, "/events" SSE, "/api/state" JSON
# --------------------------------------------------------------------------
def make_handler(hub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, ctype, body: bytes):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _sse(self, kind, payload):
            data = json.dumps(payload, default=float)
            self.wfile.write(f"event: {kind}\ndata: {data}\n\n".encode())
            self.wfile.flush()

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, "text/html; charset=utf-8", DASHBOARD_HTML.encode())
            elif path == "/api/state":
                self._send(200, "application/json",
                           json.dumps(hub.snapshot(), default=float).encode())
            elif path == "/favicon.ico":
                self._send(204, "image/x-icon", b"")
            elif path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                q, snap = hub.subscribe()
                try:
                    self._sse("snapshot", snap)
                    while True:
                        try:
                            kind, payload = q.get(timeout=15)
                        except queue.Empty:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                            continue
                        self._sse(kind, payload)
                except OSError:
                    pass  # client went away
                finally:
                    hub.unsubscribe(q)
            else:
                self._send(404, "text/plain", b"not found")

    return Handler


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", default=str(here),
                    help="folder containing config.py, data.py, model.py, force_model.py "
                         "(default: this file's folder)")
    ap.add_argument("--checkpoint", default="best_angle_model.pt")
    ap.add_argument("--force-checkpoint", default="force_head.pt",
                    help="force head saved by train_force.py (default force_head.pt)")
    ap.add_argument("--session-base", default=DEFAULT_SESSION_BASE,
                    help="folder searched (recursively) for partial_capture.raw")
    ap.add_argument("--file", default=None,
                    help="fixed partial_capture.raw path (disables auto-discovery)")
    ap.add_argument("--packet-sec", type=float, default=0.8,
                    help="packet length in seconds, no overlap (default 0.8)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="ignore data already in the file when the session is first seen")
    ap.add_argument("--buffer-sec", type=float, default=None,
                    help="rolling window length in seconds (default: required_history + 1)")
    ap.add_argument("--smooth", type=int, default=5,
                    help="reported angle = median of the last N window predictions")
    ap.add_argument("--max-backlog", type=int, default=15,
                    help="if more packets than this are waiting, skip inference on the older ones")
    ap.add_argument("--poll-ms", type=float, default=400,
                    help="how often to look for new bytes / re-decode (default 400)")
    ap.add_argument("--tmp-dir", default=None,
                    help="where snapshots are written (default /dev/shm)")
    ap.add_argument("--history", type=int, default=600)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    proj = Path(args.project_dir).resolve()
    if not (proj / "config.py").exists():
        sys.exit(f"config.py not found in {proj}. Put this file next to config.py/data.py/model.py "
                 f"or pass --project-dir.")
    sys.path.insert(0, str(proj))

    import torch
    from torch.utils.data import DataLoader
    from config import Config
    from data import AngleDataset, Recording, load_events_from_raw
    from model import EventWakeAngleModel
    from force_model import ForceHead

    mods = SimpleNamespace(AngleDataset=AngleDataset, Recording=Recording,
                           load_events_from_raw=load_events_from_raw)

    cfg = Config()
    device = torch.device(cfg.device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = proj / ckpt_path
    force_path = Path(args.force_checkpoint)
    if not force_path.is_absolute():
        force_path = proj / force_path
    print(f"Device: {device}   Angle checkpoint: {ckpt_path}   Force checkpoint: {force_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = EventWakeAngleModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    force_head, channels = load_force_head(torch, force_path, ForceHead)

    # warm-up so the first live packet is not slowed by CUDA/cuDNN lazy init
    with torch.no_grad():
        model(torch.zeros(1, cfg.sequence_length, cfg.max_events_per_window, 4, device=device),
              torch.zeros(1, cfg.sequence_length, cfg.max_events_per_window,
                          dtype=torch.bool, device=device))

    hub = Hub(args.history)
    hub.set_meta(status="starting", device=str(device), checkpoint=ckpt_path.name,
                 force_checkpoint=(force_path.name if force_head is not None else None),
                 channels=channels,
                 angle_min=float(cfg.angle_min), angle_max=float(cfg.angle_max))

    server = ThreadingHTTPServer((args.host, args.port), make_handler(hub))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Dashboard: http://localhost:{args.port}  (listening on {args.host})")

    try:
        follow(args, hub, torch, DataLoader, mods, model, cfg, device, force_head)
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


# --------------------------------------------------------------------------
# Dashboard page (self-contained, no CDN, works offline)
# Blue = angle, orange = force
# --------------------------------------------------------------------------
DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wake angle and force, live</title>
<style>
:root{
  --bg:#f3f5f8; --panel:#ffffff; --ink:#14202b; --muted:#5f6d7a; --line:#dde3e9; --lcd:#f5f8fb;
  --angle:#1b6fd1; --angle-soft:#e3eefb;
  --force:#e8710a; --force2:#9c4600; --force-soft:#fdf0e3;
  --warn:#a85a00; --bad:#b3261e; --ok:#1d7a4a;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#0e151c; --panel:#151f29; --ink:#e7edf3; --muted:#94a3b1; --line:#283543; --lcd:#0f1821;
    --angle:#5aaaff; --angle-soft:#16304d;
    --force:#ffa04d; --force2:#ffd2a1; --force-soft:#35250f;
    --warn:#e0a24f; --bad:#ff8a80; --ok:#6fd49b;
  }
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.45 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif;
  font-variant-numeric:tabular-nums}
main{max-width:1180px;margin:0 auto;padding:20px 16px 40px}
header{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:16px;flex-wrap:wrap}
h1{font-size:18px;font-weight:600;margin:0}
.sub{color:var(--muted);font-size:13px}
.pill{display:inline-flex;align-items:center;gap:8px;padding:5px 12px;border:1px solid var(--line);
  border-radius:999px;background:var(--panel);font-size:13px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--muted)}
.pill.live .dot{background:var(--ok)} .pill.idle .dot{background:var(--warn)} .pill.down .dot{background:var(--bad)}

.board{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.top{display:grid;grid-template-columns:minmax(280px,340px) 1fr}
.left{padding:18px;border-right:1px solid var(--line);display:flex;flex-direction:column;gap:14px}
.right{display:flex;flex-direction:column;min-width:0}
@media (max-width:820px){.top{grid-template-columns:1fr}.left{border-right:0;border-bottom:1px solid var(--line)}}

.label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px}
.card{background:var(--lcd);border:1px solid var(--line);border-radius:10px;padding:12px 16px 14px}
.card.angle{border-left:4px solid var(--angle)}
.card.force{border-left:4px solid var(--force)}
.card.angle .label{color:var(--angle)} .card.force .label{color:var(--force)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.big{font-size:58px;font-weight:650;line-height:1;letter-spacing:-1px;color:var(--angle);text-align:right}
.big small{font-size:26px;font-weight:500;color:var(--muted);margin-left:4px}
.frow{display:flex;align-items:baseline;justify-content:space-between;gap:10px;padding:3px 0}
.frow + .frow{border-top:1px dashed var(--line);margin-top:3px;padding-top:8px}
.fname{font-size:13px;font-weight:600;color:var(--muted);min-width:42px}
.fval{font-size:30px;font-weight:650;letter-spacing:-.5px;text-align:right;flex:1}
.fval.c0{color:var(--force)} .fval.c1{color:var(--force2)}
.pktbox{border:1px solid var(--line);border-radius:10px;padding:10px 14px}
.pktname{font-size:15px;font-weight:600;word-break:break-all}

.panel{padding:12px 16px 4px;min-width:0}
.panel + .panel{border-top:1px solid var(--line)}
.charthead{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}
.ctitle{font-size:13px;font-weight:600}
.ctitle.angle{color:var(--angle)} .ctitle.force{color:var(--force)}
.legend{display:flex;gap:14px;font-size:12.5px;color:var(--muted)}
.key i{display:inline-block;width:22px;height:0;border-top:3px solid var(--force);vertical-align:middle;margin-right:6px}
.key i.d{border-top:3px dashed var(--force2)}
canvas{display:block;width:100%}
#chartAngle{height:190px} #chartForce{height:230px}

.stats{display:flex;flex-wrap:wrap;border-top:1px solid var(--line)}
.stat{flex:1 1 150px;padding:12px 16px;border-right:1px solid var(--line)}
.stat:last-child{border-right:0}
.stat span{display:block;color:var(--muted);font-size:12.5px}
.stat b{font-size:17px;font-weight:600}

.wrap{overflow-x:auto;margin-top:16px}
table{width:100%;border-collapse:collapse;background:var(--panel);
  border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:8px 14px;border-bottom:1px solid var(--line);font-size:14px;white-space:nowrap}
th{color:var(--muted);font-weight:500;font-size:13px}
tr:last-child td{border-bottom:0}
td.num,th.num{text-align:right}
td.a{color:var(--angle);font-weight:600} td.f0{color:var(--force);font-weight:600} td.f1{color:var(--force2);font-weight:600}
td.err{color:var(--bad)} td.note{color:var(--muted)}
:focus-visible{outline:2px solid var(--angle);outline-offset:2px}
</style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Wake angle and force, live</h1>
      <div class="sub" id="sub">Waiting for the first packet</div>
    </div>
    <div class="pill idle" id="status"><span class="dot"></span><span id="statusText">Connecting</span></div>
  </header>

  <section class="board">
    <div class="top">
      <div class="left">
        <div class="card angle" role="status" aria-live="polite">
          <div class="label">Predicted angle</div>
          <div class="big mono"><span id="angle">--.--</span><small>&deg;</small></div>
        </div>
        <div class="card force" role="status" aria-live="polite">
          <div class="label">Predicted force</div>
          <div class="frow"><span class="fname" id="fName0">CH3</span><span class="fval c0 mono" id="fVal0">-.-----</span></div>
          <div class="frow"><span class="fname" id="fName1">CH4</span><span class="fval c1 mono" id="fVal1">-.-----</span></div>
        </div>
        <div class="pktbox">
          <div class="label">Packet</div>
          <div class="pktname mono" id="pktName">&ndash;</div>
        </div>
      </div>

      <div class="right">
        <div class="panel">
          <div class="charthead">
            <span class="ctitle angle">Angle (&deg;)</span>
            <span class="sub">vs. last event timestamp of each packet</span>
          </div>
          <canvas id="chartAngle"></canvas>
        </div>
        <div class="panel">
          <div class="charthead">
            <span class="ctitle force">Force</span>
            <div class="legend">
              <span class="key"><i></i><span id="kName0">CH3</span> (left axis)</span>
              <span class="key"><i class="d"></i><span id="kName1">CH4</span> (right axis)</span>
            </div>
          </div>
          <canvas id="chartForce"></canvas>
        </div>
      </div>
    </div>
    <div class="stats">
      <div class="stat"><span>Events in packet</span><b id="sEvents">&ndash;</b></div>
      <div class="stat"><span>Inference time</span><b id="sInfer">&ndash;</b></div>
      <div class="stat"><span>Decode time</span><b id="sDecode">&ndash;</b></div>
      <div class="stat"><span>Packets processed</span><b id="sCount">0</b></div>
      <div class="stat"><span>Errors</span><b id="sErr">0</b></div>
    </div>
  </section>

  <div class="wrap">
    <table>
      <thead><tr>
        <th>Packet</th><th class="num">Angle (&deg;)</th>
        <th class="num" id="thF0">CH3</th><th class="num" id="thF1">CH4</th>
        <th class="num">Events</th><th class="num">Infer (ms)</th><th>Note</th>
      </tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
</main>

<script>
const $ = id => document.getElementById(id);
const MAXPTS = 150, MAXROWS = 14;
let meta = {}, hist = [], lastRecv = 0, connected = false;

const f = (v, d = 1) => v == null ? "–" : Number(v).toFixed(d);
const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const chName = i => (meta.channels && meta.channels[i]) || ["CH3", "CH4"][i];
const forceVal = (p, i) => (p.force && p.force[i] != null) ? p.force[i] : null;

// time label: more decimals when the visible time span is short
function fmtT(v, span) {
  const d = span < 2 ? 3 : span < 20 ? 2 : span < 200 ? 1 : 0;
  return v.toFixed(d);
}

/* Generic line chart. cfg = {
     series: [{get(p), color, dash, axis:"l"|"r"}],
     axes:   {l:{minSpan, dec, color, fallback:[a,b]}, r:{...}},
     xtitle: string | null   (x labels are drawn only when set)
   }  Both charts use the same margins and the same x range, so they line up. */
function drawChart(canvas, pts, cfg) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (!W || !H) return;
  canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  const line = cssVar("--line"), muted = cssVar("--muted");

  const L = 66, R = 66, T = 10, B = cfg.xtitle ? 46 : 10;
  const pw = W - L - R, ph = H - T - B;

  // y range per axis
  const rng = {};
  for (const a of ["l", "r"]) {
    const ax = cfg.axes[a]; if (!ax) continue;
    const vals = [];
    cfg.series.filter(s => s.axis === a).forEach(s => pts.forEach(p => {
      const v = s.get(p); if (v != null) vals.push(v);
    }));
    let mn, mx;
    if (vals.length) { mn = Math.min(...vals); mx = Math.max(...vals); }
    else { mn = ax.fallback[0]; mx = ax.fallback[1]; }
    const pad = (mx - mn) * 0.12; mn -= pad; mx += pad;
    if (mx - mn < ax.minSpan) { const m = (mx + mn) / 2; mn = m - ax.minSpan / 2; mx = m + ax.minSpan / 2; }
    rng[a] = { mn, mx };
  }
  const Y = (a, v) => T + ph * (1 - (v - rng[a].mn) / (rng[a].mx - rng[a].mn));

  // shared x range
  let t0 = pts.length ? pts[0].t_last : 0;
  let t1 = pts.length ? pts[pts.length - 1].t_last : 1;
  if (t1 - t0 < 1e-9) t1 = t0 + 1;
  const span = t1 - t0;
  const X = t => L + pw * (t - t0) / span;

  g.font = "11px ui-sans-serif,system-ui,sans-serif"; g.lineWidth = 1; g.strokeStyle = line;

  // horizontal grid + y labels (grid follows the first axis present)
  const gridAxis = cfg.axes.l ? "l" : "r";
  for (let k = 0; k <= 4; k++) {
    const y = T + ph * (1 - k / 4);
    g.setLineDash([]); g.strokeStyle = line;
    g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke();
    for (const a of ["l", "r"]) {
      const ax = cfg.axes[a]; if (!ax) continue;
      const v = rng[a].mn + (rng[a].mx - rng[a].mn) * k / 4;
      g.fillStyle = ax.color || muted;
      if (a === "l") { g.textAlign = "right"; g.fillText(v.toFixed(ax.dec), L - 7, y + 4); }
      else { g.textAlign = "left"; g.fillText(v.toFixed(ax.dec), W - R + 7, y + 4); }
    }
  }

  // vertical grid (+ labels on the lower chart)
  for (let k = 0; k <= 4; k++) {
    const t = t0 + span * k / 4, x = X(t);
    g.strokeStyle = line; g.setLineDash([2, 4]);
    g.beginPath(); g.moveTo(x, T); g.lineTo(x, T + ph); g.stroke();
    g.setLineDash([]);
    if (cfg.xtitle) {
      g.fillStyle = muted;
      g.textAlign = k === 0 ? "left" : (k === 4 ? "right" : "center");
      g.fillText(fmtT(t, span), x, T + ph + 16);
    }
  }
  if (cfg.xtitle) {
    g.fillStyle = muted; g.textAlign = "center";
    g.fillText(cfg.xtitle, L + pw / 2, H - 6);
  }

  if (!pts.length) {
    g.fillStyle = muted; g.textAlign = "center";
    g.fillText("Waiting for data", L + pw / 2, T + ph / 2);
    return;
  }

  // data lines
  cfg.series.forEach(s => {
    g.strokeStyle = s.color; g.lineWidth = 2; g.lineJoin = "round"; g.setLineDash(s.dash || []);
    let open = false; g.beginPath();
    pts.forEach(p => {
      const v = s.get(p);
      if (v == null) { open = false; return; }
      if (!open) { g.moveTo(X(p.t_last), Y(s.axis, v)); open = true; }
      else g.lineTo(X(p.t_last), Y(s.axis, v));
    });
    g.stroke(); g.setLineDash([]);
    // marker on the newest point that has a value
    for (let i = pts.length - 1; i >= 0; i--) {
      const v = s.get(pts[i]);
      if (v != null) { g.fillStyle = s.color; g.beginPath(); g.arc(X(pts[i].t_last), Y(s.axis, v), 4, 0, 7); g.fill(); break; }
    }
  });
}

function drawCharts() {
  // only packets that carry a timestamp can be placed on the time axis
  const pts = hist.slice(-MAXPTS).filter(p => p.t_last != null);
  const angleC = cssVar("--angle"), f0 = cssVar("--force"), f1 = cssVar("--force2");

  drawChart($("chartAngle"), pts, {
    series: [{ get: p => p.angle, color: angleC, axis: "l" }],
    axes: { l: { minSpan: 4, dec: 1, color: angleC, fallback: [meta.angle_min ?? 0, meta.angle_max ?? 30] } },
    xtitle: null
  });
  drawChart($("chartForce"), pts, {
    series: [
      { get: p => forceVal(p, 0), color: f0, axis: "l" },
      { get: p => forceVal(p, 1), color: f1, axis: "r", dash: [6, 4] }
    ],
    axes: {
      l: { minSpan: 0.002, dec: 4, color: f0, fallback: [-1, 1] },
      r: { minSpan: 0.002, dec: 4, color: f1, fallback: [-1, 1] }
    },
    xtitle: "Time of last event in packet (s)"
  });
}

function td(text, cls) { const e = document.createElement("td"); e.textContent = text; if (cls) e.className = cls; return e; }

function renderRows() {
  const body = $("rows"); body.innerHTML = "";
  hist.slice(-MAXROWS).reverse().forEach(r => {
    const tr = document.createElement("tr");
    tr.appendChild(td(r.packet));
    tr.appendChild(td(f(r.angle, 2), "num a"));
    tr.appendChild(td(f(forceVal(r, 0), 5), "num f0"));
    tr.appendChild(td(f(forceVal(r, 1), 5), "num f1"));
    tr.appendChild(td(r.n_events ? r.n_events.toLocaleString() : "–", "num"));
    tr.appendChild(td(f(r.infer_ms, 1), "num"));
    tr.appendChild(r.error ? td(r.error, "err") : td(r.note || "", "note"));
    body.appendChild(tr);
  });
}

function renderMeta() {
  $("sCount").textContent = (meta.processed ?? 0).toLocaleString();
  $("sErr").textContent = meta.errors ?? 0;
  for (let i = 0; i < 2; i++) {
    const n = chName(i);
    ["fName", "kName", "thF"].forEach(p => $(p + i).textContent = n);
  }
  const parts = [];
  if (meta.session) parts.push(meta.session);
  if (meta.checkpoint) parts.push(meta.checkpoint);
  if (meta.force_checkpoint) parts.push(meta.force_checkpoint);
  if (meta.device) parts.push(meta.device);
  $("sub").textContent = parts.length ? parts.join("  |  ") : "Waiting for the first packet";
}

function render() {
  renderMeta();
  const last = hist[hist.length - 1];
  const lastPred = [...hist].reverse().find(r => r.angle != null);
  $("angle").textContent = lastPred ? lastPred.angle.toFixed(2) : "--.--";
  for (let i = 0; i < 2; i++) {
    const v = lastPred ? forceVal(lastPred, i) : null;
    $("fVal" + i).textContent = v == null ? "-.-----" : v.toFixed(5);
  }
  $("pktName").textContent = last ? last.packet : "–";
  if (last) {
    $("sEvents").textContent = last.n_events ? last.n_events.toLocaleString() : "–";
    $("sInfer").textContent = last.infer_ms != null ? last.infer_ms + " ms" : "–";
    $("sDecode").textContent = last.decode_ms != null ? last.decode_ms + " ms" : "–";
  }
  renderRows(); drawCharts();
}

function updateStatus() {
  const el = $("status"), tx = $("statusText");
  let cls = "live", text = "Live";
  if (!connected) { cls = "down"; text = "Reconnecting"; }
  else if (!hist.length) { cls = "idle"; text = "Waiting for packets"; }
  else if (Date.now() - lastRecv > 5000) { cls = "idle"; text = "No new packets"; }
  el.className = "pill " + cls; tx.textContent = text;
}

const es = new EventSource("/events");
es.onopen = () => { connected = true; updateStatus(); };
es.onerror = () => { connected = false; updateStatus(); };
es.addEventListener("snapshot", e => { const s = JSON.parse(e.data); meta = s.meta; hist = s.history; lastRecv = Date.now(); render(); });
es.addEventListener("meta", e => { Object.assign(meta, JSON.parse(e.data)); renderMeta(); });
es.addEventListener("reset", e => { meta = JSON.parse(e.data); hist = []; render(); });
es.addEventListener("packet", e => { hist.push(JSON.parse(e.data)); if (hist.length > 600) hist.shift(); lastRecv = Date.now(); render(); });
window.addEventListener("resize", drawCharts);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawCharts);
setInterval(updateStatus, 1000);
drawCharts();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()