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

import json, time, os, threading
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
        """Take the last frame's data as the snapshot. Called with lock held."""
        if not self._buffer:
            return

        n = len(self._buffer)
        last = self._buffer[-1]  # the most recent frame at this moment

        snapshot = {
            "timestamp":            now,
            "window_seconds":       round(now - self._last_commit_time, 1) if self._last_commit_time else self.interval,
            "frames_in_window":     n,

            # All values from the last frame at snapshot time
            "utilization":          last.get("utilization", 0),
            "zone_area_m2":         last.get("zone_area_m2"),
            "occupied_area_m2":     last.get("occupied_area_m2"),
            "counts":               last.get("counts", {"US": 0, "UK": 0, "unknown": 0}),
            "cls_counts":           last.get("cls_counts", {}),
            "fps":                  last.get("fps"),
            "zone_status":          last.get("zone_status"),
            "detection_quality":    last.get("detection_quality"),
            "markers_detected":     last.get("markers_detected"),
            "pallets_total":        last.get("pallets_total"),
            "pallets_in_zone":      last.get("pallets_in_zone"),
            "pallets_outside":      last.get("pallets_outside"),
            "pallets_filtered":     last.get("pallets_filtered"),
            "confidence_threshold": last.get("confidence_threshold"),
            "zone_locked":          last.get("zone_locked"),
            "ppm":                  last.get("ppm"),
            "edge_lengths_m":       last.get("edge_lengths_m"),
        }

        self._latest_snapshot = snapshot
        counts = snapshot["counts"]
        self._history.append({
            "t":        now,
            "util":     snapshot["utilization"],
            "us":       counts.get("US", 0),
            "uk":       counts.get("UK", 0),
            "unk":      counts.get("unknown", 0),
            "status":   snapshot["zone_status"],
            "frames":   n,
            "zone_area_m2":     snapshot["zone_area_m2"],
            "occupied_area_m2": snapshot["occupied_area_m2"],
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

        @self.app.route("/api/export")
        def api_export():
            """Export all history as a CSV file download."""
            import io, csv
            from flask import Response
            with self._lock:
                history = list(self._history)

            output = io.StringIO()
            writer = csv.writer(output)
            # Header row
            writer.writerow([
                "timestamp", "datetime", "utilization_%",
                "zone_area_m2", "occupied_area_m2",
                "us_pallets", "uk_pallets", "unknown_pallets",
                "zone_status"
            ])
            # Data rows
            for h in history:
                from datetime import datetime
                ts = h.get("t", 0)
                dt_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else ""
                writer.writerow([
                    round(ts, 2),
                    dt_str,
                    h.get("util", 0),
                    h.get("zone_area_m2", ""),
                    h.get("occupied_area_m2", ""),
                    h.get("us", 0),
                    h.get("uk", 0),
                    h.get("unk", 0),
                    h.get("status", ""),
                ])

            csv_data = output.getvalue()
            output.close()
            filename = f"staging_zone_data_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            return Response(
                csv_data,
                mimetype="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )

    def _run_server(self):
        import logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.WARNING)
        self.app.run(host="0.0.0.0", port=self.port, threaded=True)