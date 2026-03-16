"""
Dashboard Server Module — Interval-Based
==========================================
Lightweight HTTP server that runs alongside the staging zone tracker.
Instead of real-time streaming, data is accumulated over a configurable
window (default 30s) and committed as a snapshot. The dashboard displays
the latest snapshot and historical snapshots.

Usage:
    from dashboard_server import DashboardServer

    # In your tracker's main():
    server = DashboardServer(port=5000, interval=30)
    server.start()

    # In your main loop, feed data every frame:
    server.feed({...})   # lightweight — just accumulates internally

    # On shutdown:
    server.stop()

The server handles the timing internally:
  - feed() is called every frame (cheap, just appends to a buffer)
  - Every <interval> seconds, the buffer is averaged into a snapshot
  - The dashboard polls /api/data and gets the latest snapshot + history

Dependencies:
    pip install flask
"""

import json, time, os, threading, statistics
import numpy as np
from flask import Flask, jsonify, send_from_directory


def _sanitize(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


class DashboardServer:
    def __init__(self, port=5000, interval=30, dashboard_dir=None, history_max=200):
        """
        Args:
            port:          HTTP server port
            interval:      Seconds between dashboard updates (user-tunable)
            dashboard_dir: Directory containing dashboard.html
            history_max:   Max snapshots to keep (at 30s interval, 200 = ~100 min)
        """
        self.port = port
        self.interval = interval
        self.dashboard_dir = dashboard_dir or os.path.dirname(os.path.abspath(__file__))
        self.history_max = history_max

        # Thread-safe state
        self._lock = threading.Lock()
        self._buffer = []              # raw frame data accumulated between snapshots
        self._latest_snapshot = {}     # last committed snapshot
        self._history = []             # list of past snapshots
        self._last_commit_time = time.time()
        self._start_time = time.time()
        self._total_feeds = 0
        self._total_snapshots = 0

        # Build Flask app
        self.app = Flask(__name__)
        self._setup_routes()
        self._thread = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, data: dict):
        """
        Feed one frame's worth of data. Called every frame from the tracker.
        This is cheap — it just appends to an internal buffer.
        When the interval elapses, the buffer is averaged into a snapshot.
        """
        now = time.time()
        with self._lock:
            self._buffer.append(_sanitize(data))
            self._total_feeds += 1

            # Check if it's time to commit a snapshot
            elapsed = now - self._last_commit_time
            if elapsed >= self.interval:
                self._commit_snapshot(now)
                self._last_commit_time = now

    def set_interval(self, seconds: float):
        """Change the update interval at runtime."""
        with self._lock:
            self.interval = max(1.0, seconds)
            print(f"[DASHBOARD] Update interval changed to {self.interval}s")

    def start(self):
        """Start the server in a background daemon thread."""
        self._thread = threading.Thread(
            target=self._run_server, daemon=True, name="DashboardServer"
        )
        self._thread.start()
        print(f"[DASHBOARD] Server started at http://localhost:{self.port}")
        print(f"[DASHBOARD] Update interval: {self.interval}s")
        print(f"[DASHBOARD] Open browser:    http://localhost:{self.port}")

    def stop(self):
        """Commit any remaining buffer data and stop."""
        with self._lock:
            if self._buffer:
                self._commit_snapshot(time.time())
        print("[DASHBOARD] Server stopping.")

    # ------------------------------------------------------------------
    # Internal: snapshot logic
    # ------------------------------------------------------------------

    def _commit_snapshot(self, now):
        """Average the buffer into a single snapshot. Called with lock held."""
        if not self._buffer:
            return

        buf = self._buffer
        n = len(buf)

        # --- Numeric fields: compute average ---
        def avg_field(key):
            vals = [d[key] for d in buf if d.get(key) is not None]
            return round(statistics.mean(vals), 2) if vals else None

        def last_field(key):
            for d in reversed(buf):
                if d.get(key) is not None:
                    return d[key]
            return None

        # --- Count fields: average then round ---
        def avg_counts():
            us  = [d.get("counts", {}).get("US", 0) for d in buf]
            uk  = [d.get("counts", {}).get("UK", 0) for d in buf]
            unk = [d.get("counts", {}).get("unknown", 0) for d in buf]
            return {
                "US": round(statistics.mean(us), 1) if us else 0,
                "UK": round(statistics.mean(uk), 1) if uk else 0,
                "unknown": round(statistics.mean(unk), 1) if unk else 0,
            }

        # --- Per-class counts: average ---
        def avg_cls_counts():
            all_keys = set()
            for d in buf:
                all_keys.update(d.get("cls_counts", {}).keys())
            result = {}
            for k in all_keys:
                vals = [d.get("cls_counts", {}).get(k, 0) for d in buf]
                result[k] = round(statistics.mean(vals), 1)
            return result

        # --- Utilization stats ---
        util_vals = [d["utilization"] for d in buf if d.get("utilization") is not None]

        snapshot = {
            "timestamp":            now,
            "window_seconds":       round(now - self._last_commit_time, 1) if self._last_commit_time else self.interval,
            "frames_in_window":     n,

            # Averaged metrics
            "utilization":          round(statistics.mean(util_vals), 2) if util_vals else 0,
            "utilization_min":      round(min(util_vals), 2) if util_vals else 0,
            "utilization_max":      round(max(util_vals), 2) if util_vals else 0,
            "zone_area_m2":         avg_field("zone_area_m2"),
            "occupied_area_m2":     avg_field("occupied_area_m2"),
            "counts":               avg_counts(),
            "cls_counts":           avg_cls_counts(),
            "fps":                  avg_field("fps"),

            # Latest-value fields (don't average these)
            "zone_status":          last_field("zone_status"),
            "detection_quality":    last_field("detection_quality"),
            "markers_detected":     last_field("markers_detected"),
            "pallets_total":        last_field("pallets_total"),
            "pallets_in_zone":      last_field("pallets_in_zone"),
            "pallets_outside":      last_field("pallets_outside"),
            "pallets_filtered":     last_field("pallets_filtered"),
            "confidence_threshold": last_field("confidence_threshold"),
            "zone_locked":          last_field("zone_locked"),
            "ppm":                  last_field("ppm"),
            "edge_lengths_m":       last_field("edge_lengths_m"),
        }

        self._latest_snapshot = snapshot
        self._history.append({
            "t":        now,
            "util":     snapshot["utilization"],
            "util_min": snapshot["utilization_min"],
            "util_max": snapshot["utilization_max"],
            "us":       snapshot["counts"]["US"],
            "uk":       snapshot["counts"]["UK"],
            "unk":      snapshot["counts"]["unknown"],
            "status":   snapshot["zone_status"],
            "frames":   n,
        })
        if len(self._history) > self.history_max:
            self._history.pop(0)

        self._total_snapshots += 1
        self._buffer = []

    # ------------------------------------------------------------------
    # Flask routes
    # ------------------------------------------------------------------

    def _setup_routes(self):
        @self.app.after_request
        def add_cors(response):
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            return response

        @self.app.route("/")
        def serve_dashboard():
            return send_from_directory(self.dashboard_dir, "dashboard.html")

        @self.app.route("/api/data")
        def api_data():
            with self._lock:
                elapsed = time.time() - self._last_commit_time
                next_in = max(0, self.interval - elapsed)
                buffered = len(self._buffer)
                return jsonify({
                    "current":  self._latest_snapshot,
                    "history":  self._history,
                    "server": {
                        "uptime_s":         time.time() - self._start_time,
                        "interval_s":       self.interval,
                        "next_update_in_s": round(next_in, 1),
                        "buffered_frames":  buffered,
                        "total_feeds":      self._total_feeds,
                        "total_snapshots":  self._total_snapshots,
                    }
                })

        @self.app.route("/api/health")
        def api_health():
            with self._lock:
                last_ts = self._latest_snapshot.get("timestamp", 0)
                stale_threshold = self.interval * 3
            return jsonify({
                "status": "ok" if (time.time() - last_ts) < stale_threshold else "stale",
                "last_snapshot_ago_s": round(time.time() - last_ts, 1),
                "interval_s": self.interval,
            })

    def _run_server(self):
        import logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.WARNING)
        self.app.run(host="0.0.0.0", port=self.port, threaded=True)