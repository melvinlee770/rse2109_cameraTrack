"""
Multi-Zone Dashboard Server
=============================
Single dashboard server that receives data from multiple camera/zone
instances and provides combined + per-zone utilization.

Usage:
    from multi_dashboard_server import MultiDashboardServer

    server = MultiDashboardServer(port=5000, interval=30)
    server.start()

    # From each camera's loop:
    server.feed("A", {...})   # Zone A data
    server.feed("B", {...})   # Zone B data

    server.stop()

Dependencies:
    pip install flask
"""

import json, time, os, threading
import numpy as np
from flask import Flask, jsonify, send_from_directory, Response


def _sanitize(obj):
    """Recursively convert numpy types to native Python types for JSON."""
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


class MultiDashboardServer:
    def __init__(self, port=5000, interval=30, dashboard_dir=None, history_max=200):
        self.port = port
        self.interval = interval
        self.dashboard_dir = dashboard_dir or os.path.dirname(os.path.abspath(__file__))
        self.history_max = history_max

        self._lock = threading.Lock()
        self._start_time = time.time()
        self._total_snapshots = 0

        # Per-zone state
        self._zones = {}  # zone_id -> {buffer, snapshot, history, last_commit, feeds}

        # Snapshot images (latest JPEG per zone)
        self._snapshot_images = {}  # zone_id -> bytes (JPEG)

        # Combined history
        self._combined_history = []

        self.app = Flask(__name__)
        self._setup_routes()
        self._thread = None

    def _get_zone(self, zone_id):
        """Get or create zone state. Must be called with lock held."""
        if zone_id not in self._zones:
            self._zones[zone_id] = {
                "buffer": [],
                "snapshot": {},
                "history": [],
                "last_commit": time.time(),
                "feeds": 0,
            }
        return self._zones[zone_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, zone_id, data):
        """Feed one frame of data from a specific zone."""
        now = time.time()
        with self._lock:
            z = self._get_zone(zone_id)
            z["buffer"].append(_sanitize(data))
            z["feeds"] += 1

            elapsed = now - z["last_commit"]
            if elapsed >= self.interval:
                self._commit_zone_snapshot(zone_id, now)
                z["last_commit"] = now

                # Check if ALL zones have committed at least once — then update combined
                all_have_snapshots = all(
                    zs["snapshot"] for zs in self._zones.values()
                )
                if all_have_snapshots:
                    self._commit_combined(now)

    def start(self):
        self._thread = threading.Thread(
            target=self._run_server, daemon=True, name="MultiDashboardServer"
        )
        self._thread.start()
        print(f"[DASHBOARD] Multi-zone server started at http://localhost:{self.port}")
        print(f"[DASHBOARD] Update interval: {self.interval}s")

    def stop(self):
        with self._lock:
            for zid in list(self._zones.keys()):
                if self._zones[zid]["buffer"]:
                    self._commit_zone_snapshot(zid, time.time())
            if any(z["snapshot"] for z in self._zones.values()):
                self._commit_combined(time.time())
        print("[DASHBOARD] Server stopping.")

    def store_snapshot_image(self, zone_id, jpg_bytes):
        """Store the latest annotated camera frame as JPEG for download."""
        with self._lock:
            self._snapshot_images[zone_id] = jpg_bytes

    # ------------------------------------------------------------------
    # Snapshot logic
    # ------------------------------------------------------------------

    def _commit_zone_snapshot(self, zone_id, now):
        """Commit a snapshot for one zone. Called with lock held."""
        z = self._zones[zone_id]
        if not z["buffer"]:
            return

        last = z["buffer"][-1]
        n = len(z["buffer"])

        snapshot = {
            "timestamp":            now,
            "frames_in_window":     n,
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
            "confidence_threshold": last.get("confidence_threshold"),
            "zone_locked":          last.get("zone_locked"),
            "ppm":                  last.get("ppm"),
            "edge_lengths_m":       last.get("edge_lengths_m"),
            "grid":                 last.get("grid"),
            "grid_labels":          last.get("grid_labels", []),
            "grid_cols":            last.get("grid_cols", 0),
            "grid_rows":            last.get("grid_rows", 0),
            "grid_pallet_info":     last.get("grid_pallet_info", {}),
        }

        z["snapshot"] = snapshot
        counts = snapshot["counts"]
        z["history"].append({
            "t":    now,
            "util": snapshot["utilization"],
            "us":   counts.get("US", 0),
            "uk":   counts.get("UK", 0),
            "unk":  counts.get("unknown", 0),
            "zone_area_m2":     snapshot["zone_area_m2"],
            "occupied_area_m2": snapshot["occupied_area_m2"],
        })
        if len(z["history"]) > self.history_max:
            z["history"].pop(0)
        z["buffer"] = []

    def _commit_combined(self, now):
        """Combine all zone snapshots into one. Called with lock held."""
        total_zone_area = 0
        total_occupied_area = 0
        total_counts = {"US": 0, "UK": 0, "unknown": 0}
        total_cls_counts = {}
        total_pallets = 0
        total_in_zone = 0

        for zid, z in self._zones.items():
            s = z["snapshot"]
            if not s:
                continue
            za = s.get("zone_area_m2") or 0
            oa = s.get("occupied_area_m2") or 0
            total_zone_area += za
            total_occupied_area += oa

            c = s.get("counts", {})
            for k in total_counts:
                total_counts[k] += c.get(k, 0)

            cc = s.get("cls_counts", {})
            for k, v in cc.items():
                total_cls_counts[k] = total_cls_counts.get(k, 0) + v

            total_pallets += s.get("pallets_total") or 0
            total_in_zone += s.get("pallets_in_zone") or 0

        # Combined utilization = total occupied / total zone area
        if total_zone_area > 0:
            combined_util = (total_occupied_area / total_zone_area) * 100.0
        else:
            combined_util = 0

        self._combined_history.append({
            "t":                now,
            "util":             round(combined_util, 2),
            "us":               total_counts["US"],
            "uk":               total_counts["UK"],
            "unk":              total_counts["unknown"],
            "zone_area_m2":     round(total_zone_area, 4),
            "occupied_area_m2": round(total_occupied_area, 4),
        })
        if len(self._combined_history) > self.history_max:
            self._combined_history.pop(0)

        self._total_snapshots += 1

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    def _setup_routes(self):
        @self.app.after_request
        def add_cors(response):
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            return response

        @self.app.route("/")
        def serve_dashboard():
            return send_from_directory(self.dashboard_dir, "multi_dashboard.html")

        @self.app.route("/api/data")
        def api_data():
            with self._lock:
                # Build per-zone data
                zones = {}
                for zid, z in self._zones.items():
                    elapsed = time.time() - z["last_commit"]
                    zones[zid] = {
                        "current":  z["snapshot"],
                        "history":  z["history"],
                        "next_update_in_s": round(max(0, self.interval - elapsed), 1),
                        "buffered_frames":  len(z["buffer"]),
                        "total_feeds":      z["feeds"],
                    }

                # Build combined current
                combined_current = {}
                if self._combined_history:
                    ch = self._combined_history[-1]
                    combined_current = {
                        "timestamp":        ch["t"],
                        "utilization":      ch["util"],
                        "zone_area_m2":     ch["zone_area_m2"],
                        "occupied_area_m2": ch["occupied_area_m2"],
                        "counts": {"US": ch["us"], "UK": ch["uk"], "unknown": ch["unk"]},
                    }

                return jsonify({
                    "combined": {
                        "current":  combined_current,
                        "history":  self._combined_history,
                    },
                    "zones": zones,
                    "server": {
                        "uptime_s":         time.time() - self._start_time,
                        "interval_s":       self.interval,
                        "total_snapshots":  self._total_snapshots,
                        "zone_count":       len(self._zones),
                    }
                })

        @self.app.route("/api/export")
        def api_export():
            import io, csv
            from datetime import datetime
            with self._lock:
                combined = list(self._combined_history)
                zone_histories = {zid: list(z["history"]) for zid, z in self._zones.items()}

            output = io.StringIO()
            writer = csv.writer(output)

            # Combined section
            writer.writerow(["=== COMBINED ==="])
            writer.writerow(["timestamp", "datetime", "utilization_%",
                             "zone_area_m2", "occupied_area_m2",
                             "us_pallets", "uk_pallets", "unknown_pallets"])
            for h in combined:
                ts = h.get("t", 0)
                dt_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else ""
                writer.writerow([round(ts,2), dt_str, h.get("util",0),
                                 h.get("zone_area_m2",""), h.get("occupied_area_m2",""),
                                 h.get("us",0), h.get("uk",0), h.get("unk",0)])

            # Per-zone sections
            for zid in sorted(zone_histories.keys()):
                writer.writerow([])
                writer.writerow([f"=== ZONE {zid} ==="])
                writer.writerow(["timestamp", "datetime", "utilization_%",
                                 "zone_area_m2", "occupied_area_m2",
                                 "us_pallets", "uk_pallets", "unknown_pallets"])
                for h in zone_histories[zid]:
                    ts = h.get("t", 0)
                    dt_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else ""
                    writer.writerow([round(ts,2), dt_str, h.get("util",0),
                                     h.get("zone_area_m2",""), h.get("occupied_area_m2",""),
                                     h.get("us",0), h.get("uk",0), h.get("unk",0)])

            csv_data = output.getvalue(); output.close()
            fname = f"staging_zone_multi_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            return Response(csv_data, mimetype="text/csv",
                            headers={"Content-Disposition": f"attachment; filename={fname}"})

        @self.app.route("/api/health")
        def api_health():
            with self._lock:
                zone_status = {}
                for zid, z in self._zones.items():
                    last_ts = z["snapshot"].get("timestamp", 0)
                    zone_status[zid] = {
                        "status": "ok" if (time.time() - last_ts) < self.interval * 3 else "stale",
                        "last_snapshot_ago_s": round(time.time() - last_ts, 1),
                    }
            return jsonify({"zones": zone_status, "interval_s": self.interval})

        @self.app.route("/api/snapshot/<zone_id>")
        def api_snapshot(zone_id):
            """Download the latest annotated camera frame as JPEG."""
            import time as _time
            with self._lock:
                jpg = self._snapshot_images.get(zone_id)
            if jpg is None:
                return Response("No snapshot available", status=404)
            fname = f"zone_{zone_id}_{_time.strftime('%Y%m%d_%H%M%S')}.jpg"
            return Response(jpg, mimetype="image/jpeg",
                            headers={"Content-Disposition": f"attachment; filename={fname}"})

    def _run_server(self):
        import logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.WARNING)
        self.app.run(host="0.0.0.0", port=self.port, threaded=True)