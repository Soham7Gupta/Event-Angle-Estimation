#!/usr/bin/env python3
# replay_raw_as_partial.py
"""
Simulate the streaming infra for testing stream2.py.

Takes a recorded Prophesee .raw file (EVT2 or EVT3) and writes it into
    <session-base>/replay_test/<timestamp>_<id>/partial_capture.raw
as a GROWING file: the original header first, then the original event data
appended piece by piece at real-time pace. When everything is written, the file
is renamed to capture.raw in the same folder (atomic), exactly like the infra
does when the live stream is switched off.

The data is the original byte stream, unmodified (original timestamps, no
re-emitted state), so capture.raw ends up byte-identical to the source file.

Run (start stream2.py first, in another terminal, then this):
    python replay_raw_as_partial.py
Options:
    --raw FILE         recording to replay
    --speed 2.0        replay twice as fast (1.0 = real time)
    --write-sec 0.2    how often new data is appended (default 0.2 s of recording)
    --ragged           cut the appended chunks at random byte positions (can end
                       in the middle of a word), to test the snapshot trimming
    --max-seconds N    only replay the first N seconds of the recording
    --out-dir DIR      write into this folder instead of a new replay_test session
"""

import argparse
import hashlib
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_RAW = ("/home/gupta/Desktop/EB/data/-5winkel/ebiv_NB_LNDW_-5winkel_60rpm_cam00051568_2026-06-17_13-24-33.raw")
DEFAULT_SESSION_BASE = "/home/gupta/Desktop/streaming-infra/received_directory"

M28 = (1 << 28) - 1


# --------------------------------------------------------------------------
# header handling
# --------------------------------------------------------------------------
def read_header(path):
    """Return the leading ASCII header (all lines starting with '%') as bytes."""
    header = b""
    with open(path, "rb") as fh:
        while True:
            line = fh.readline()
            if not line.startswith(b"%"):
                break
            header += line
    return header


def detect_format(header: bytes):
    txt = header.decode("latin-1").lower()
    if "evt 3.0" in txt or "format evt3" in txt:
        return "evt3"
    if "evt 2.1" in txt or "format evt21" in txt:
        return "evt21"
    if "evt 2.0" in txt or "format evt2" in txt:
        return "evt2"
    return None


# --------------------------------------------------------------------------
# Work out where to cut the data so that each append covers `step_us` of
# recording time. Returns word indices `cuts` (len = n_steps + 1).
# --------------------------------------------------------------------------
def plan_cuts(fmt, w, step_us, max_steps):
    n = len(w)
    ar = np.arange(n)

    if fmt == "evt2":
        # 32-bit words, type = bits 31:28 (0/1 = CD off/on, 8 = TIME_HIGH)
        typ = w >> 28
        last = np.maximum.accumulate(np.where(typ == 8, ar, -1))
        th = np.where(last >= 0, w[np.maximum(last, 0)] & M28, 0).astype(np.int64)
        idx = np.nonzero(typ <= 1)[0]
        if len(idx) == 0:
            sys.exit("no CD events found in the file")
        t = np.maximum.accumulate((th[idx] << 6) | ((w[idx] >> 22) & 0x3F).astype(np.int64))
    else:
        # EVT3: 16-bit words, type = bits 15:12 (0 = ADDR_Y, 6 = TIME_LOW, 8 = TIME_HIGH)
        typ = w >> 12
        is_th = typ == 8
        th_idx = np.nonzero(is_th)[0]
        if len(th_idx) == 0:
            sys.exit("no TIME_HIGH words found in the file")
        hv = (w[th_idx] & 0xFFF).astype(np.int64)
        wraps = np.cumsum(np.diff(hv, prepend=hv[0]) < -2048)   # 24-bit time wrap-arounds
        hv_unw = hv + 4096 * wraps
        rank = np.cumsum(is_th) - 1
        r = np.maximum(rank, 0)
        high_u = np.where(rank >= 0, hv_unw[r], 0)
        last_tl = np.maximum.accumulate(np.where(typ == 6, ar, -1))
        tl = np.where(last_tl >= 0, w[np.maximum(last_tl, 0)] & 0xFFF, 0).astype(np.int64)
        idx = np.nonzero(typ == 0)[0]                            # ADDR_Y words
        if len(idx) == 0:
            sys.exit("no events found in the file")
        t = np.maximum.accumulate(((high_u << 12) | tl)[idx])

    dur = float((t[-1] - t[0]) / 1e6)
    n_steps = int((t[-1] - t[0]) // step_us) + 1
    truncated = bool(max_steps and max_steps < n_steps)
    if truncated:
        n_steps = max_steps
    edges = t[0] + np.arange(n_steps + 1, dtype=np.int64) * step_us
    pos = np.searchsorted(t, edges, side="left")
    cuts = np.where(pos < len(idx), idx[np.minimum(pos, len(idx) - 1)], n)
    cuts[0] = 0
    if not truncated:
        cuts[-1] = n
    return n_steps, cuts, dur, truncated


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=DEFAULT_RAW, help="recorded .raw file to replay")
    ap.add_argument("--session-base", default=DEFAULT_SESSION_BASE,
                    help="folder stream2.py searches for partial_capture.raw")
    ap.add_argument("--out-dir", default=None,
                    help="write into this folder instead of a new replay_test/<session>")
    ap.add_argument("--write-sec", type=float, default=0.2,
                    help="recording time appended per write (default 0.2 s)")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real time")
    ap.add_argument("--ragged", action="store_true",
                    help="cut chunks at random byte positions (may split a word)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="only replay the first N seconds of the recording")
    args = ap.parse_args()

    if args.write_sec <= 0 or args.speed <= 0:
        sys.exit("--write-sec and --speed must be > 0")
    if not Path(args.raw).is_file():
        sys.exit(f"raw file not found: {args.raw}")

    header = read_header(args.raw)
    if not header:
        sys.exit("no '%' header found; is this a Prophesee .raw file?")
    fmt = detect_format(header)
    if fmt is None or fmt == "evt21":
        sys.exit("unsupported raw format (only EVT2 and EVT3 are handled). Header was:\n"
                 + header.decode("latin-1"))

    with open(args.raw, "rb") as fh:
        fh.seek(len(header))
        body = fh.read()
    isz = 4 if fmt == "evt2" else 2
    dtype = "<u4" if fmt == "evt2" else "<u2"
    w = np.frombuffer(body[: len(body) // isz * isz], dtype=dtype)
    if len(w) == 0:
        sys.exit("no event data after the header")

    step_us = int(round(args.write_sec * 1e6))
    max_steps = int(args.max_seconds / args.write_sec) if args.max_seconds else None
    n_steps, cuts, dur, truncated = plan_cuts(fmt, w, step_us, max_steps)

    # word cuts -> byte cuts (optionally ragged)
    bcuts = cuts.astype(np.int64) * isz
    if args.ragged:
        jit = np.random.default_rng(0).integers(0, isz, size=len(bcuts))
        jit[0] = 0
        if not truncated:
            jit[-1] = 0
        bcuts = bcuts + jit
    bcuts = np.minimum(np.maximum.accumulate(bcuts), len(body))
    if not truncated:
        bcuts[-1] = len(body)   # includes any trailing partial word

    print(f"{fmt.upper()}  {len(w):,} words, {dur:.2f} s of data -> {n_steps} writes of "
          f"{args.write_sec:g} s{' (truncated)' if truncated else ''}"
          f"{', ragged byte cuts' if args.ragged else ''}")

    # ---- output folder ----------------------------------------------------
    if args.out_dir:
        sess = Path(args.out_dir)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        sess = Path(args.session_base) / "replay_test" / f"{stamp}_{uuid.uuid4().hex[:8]}"
    sess.mkdir(parents=True, exist_ok=True)
    partial = sess / "partial_capture.raw"
    final = sess / "capture.raw"
    if partial.exists() or final.exists():
        sys.exit(f"{sess} already has partial_capture.raw / capture.raw; use a fresh folder.")
    print(f"Writing to {partial}")

    # ---- paced, growing write ---------------------------------------------
    step_wall = args.write_sec / args.speed
    report_every = max(1, int(round(1.0 / args.write_sec)))
    try:
        with open(partial, "wb") as fh:
            fh.write(header)
            fh.flush()
            start = time.perf_counter()
            for i in range(n_steps):
                # data of step i is "available" at the end of its window;
                # scheduling from `start` avoids cumulative drift
                delay = start + (i + 1) * step_wall - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                fh.write(body[bcuts[i]:bcuts[i + 1]])
                fh.flush()
                if i % report_every == 0:
                    print(f"  {(i + 1) * args.write_sec:7.1f} s recorded   "
                          f"file = {fh.tell():>13,} bytes")
            fh.flush()
            os.fsync(fh.fileno())
    except KeyboardInterrupt:
        print(f"\nStopped. {partial.name} was left in place (not renamed).")
        return

    os.replace(partial, final)   # live stream "switched off"
    print(f"Renamed -> {final}")

    # ---- check that capture.raw equals the source -------------------------
    h_src = hashlib.sha256(header)
    h_src.update(body[: bcuts[-1]])
    h_out = hashlib.sha256(final.read_bytes())
    ok = h_src.digest() == h_out.digest()
    print("Check: capture.raw is " + ("byte-identical to the source" +
          (" (first part only)" if truncated else "") if ok else "DIFFERENT from the source"))


if __name__ == "__main__":
    main()