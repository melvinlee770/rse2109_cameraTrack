"""
Staging Zone ArUco Tracker v7 - CALIBRATED
============================================
Same as custom_track_v7.py but loads camera calibration data
to undistort frames before processing, giving more accurate
real-world area measurements.

Run calibrate_camera.py first to generate camera_calibration.npz

Usage:
    python custom_track_v7_calibrated.py
    python custom_track_v7_calibrated.py --camera 1
    python custom_track_v7_calibrated.py --calib-file my_calibration.npz
    python custom_track_v7_calibrated.py --no-calib   # run without calibration
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import time
import argparse
import os
import sys


# =============================================================================
# Configuration
# =============================================================================

# ArUco tag physical size in CENTIMETERS
TAG_SIZE_CM = 2.4

# ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Tune detection for better reliability
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 53
parameters.adaptiveThreshWinSizeStep = 4
parameters.minMarkerPerimeterRate = 0.01

detector = aruco.ArucoDetector(aruco_dict, parameters)


# =============================================================================
# Calibration Loader
# =============================================================================

def load_calibration(calib_file):
    """
    Load camera calibration data from .npz file.

    Returns:
        camera_matrix, dist_coeffs, reprojection_error
        or (None, None, None) if file not found
    """
    if not os.path.exists(calib_file):
        return None, None, None

    data = np.load(calib_file)
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
    error = data['reprojection_error'][0]

    return camera_matrix, dist_coeffs, error


def setup_undistortion(camera_matrix, dist_coeffs, frame_width, frame_height):
    """
    Pre-compute undistortion maps for fast per-frame remapping.

    Returns:
        map1, map2: remapping arrays for cv2.remap()
        new_camera_matrix: optimized camera matrix after undistortion
        roi: valid pixel region after undistortion
    """
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs,
        (frame_width, frame_height), 1,
        (frame_width, frame_height))

    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None,
        new_camera_matrix,
        (frame_width, frame_height),
        cv2.CV_32FC1)

    return map1, map2, new_camera_matrix, roi


# =============================================================================
# Helper Functions (same as v7)
# =============================================================================

def order_points(pts):
    """Sorts 4 coordinates into: [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def get_marker_centers(corners):
    """Get the center point of each detected marker."""
    return np.array([np.mean(c[0], axis=0) for c in corners])


def get_marker_inner_corners(corners):
    """
    Get the inner corner of each marker — the corner closest to the
    centroid of all markers (zone center). Orientation-independent.
    """
    all_centers = np.array([np.mean(c[0], axis=0) for c in corners])
    centroid = np.mean(all_centers, axis=0)

    inner_points = []
    for marker_corners in corners:
        pts = marker_corners[0]
        distances = np.linalg.norm(pts - centroid, axis=1)
        closest_idx = np.argmin(distances)
        inner_points.append(pts[closest_idx])

    return np.array(inner_points, dtype=np.float32)


def get_pixels_per_meter(corners, tag_size_cm):
    """Calculate pixels-per-meter from detected ArUco marker edges."""
    edge_lengths_px = []
    for marker_corners in corners:
        pts = marker_corners[0]
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            edge_lengths_px.append(np.linalg.norm(p2 - p1))

    if len(edge_lengths_px) == 0:
        return None

    avg_edge_px = np.mean(edge_lengths_px)
    pixels_per_cm = avg_edge_px / tag_size_cm
    return pixels_per_cm * 100.0


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calibrated Staging Zone Tracker")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device ID (default: 0)")
    parser.add_argument("--calib-file", type=str, default="camera_calibration.npz",
                        help="Path to calibration file (default: camera_calibration.npz)")
    parser.add_argument("--no-calib", action="store_true",
                        help="Run without calibration (same as v7)")
    parser.add_argument("--tag-size", type=float, default=TAG_SIZE_CM,
                        help=f"ArUco tag size in cm (default: {TAG_SIZE_CM})")
    args = parser.parse_args()

    tag_size = args.tag_size

    # --- Load Calibration ---
    use_calibration = False
    map1, map2 = None, None
    calib_error = 0

    if not args.no_calib:
        camera_matrix, dist_coeffs, calib_error = load_calibration(args.calib_file)
        if camera_matrix is not None:
            use_calibration = True
            print(f"[INFO] Calibration loaded: {args.calib_file}")
            print(f"[INFO] Reprojection error: {calib_error:.4f} px")
        else:
            print(f"[WARNING] No calibration file found: {args.calib_file}")
            print(f"[WARNING] Running WITHOUT calibration (less accurate)")
            print(f"[TIP]    Run calibrate_camera.py --calibrate first")
    else:
        print(f"[INFO] Running without calibration (--no-calib)")

    # --- Open Camera ---
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Pre-compute undistortion maps after first frame
    undistort_initialized = False

    # Persistent state
    last_valid_pts = None
    last_ppm = None
    last_real_area = None
    last_pixel_area = 0

    # Reference mode
    ref_mode = "CENTER"

    # FPS tracking
    fps_time = time.time()
    frame_count = 0
    fps = 0.0

    calib_label = "CALIBRATED" if use_calibration else "UNCALIBRATED"
    print(f"[INFO] Mode: {calib_label}")
    print(f"[INFO] ArUco tag size: {tag_size} cm")
    print(f"[INFO] Controls:")
    print(f"         'q' = quit")
    print(f"         'c' = toggle CENTER / CORNER mode")
    print(f"         's' = save snapshot")
    print(f"         'r' = reset zone memory")
    print(f"         'u' = toggle undistortion on/off")

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        # --- Initialize undistortion maps on first frame ---
        if use_calibration and not undistort_initialized:
            h, w = raw_frame.shape[:2]
            map1, map2, new_cam_matrix, roi = setup_undistortion(
                camera_matrix, dist_coeffs, w, h)
            undistort_initialized = True
            print(f"[INFO] Undistortion maps computed for {w}x{h}")

        # --- Apply undistortion ---
        if use_calibration and map1 is not None:
            frame = cv2.remap(raw_frame, map1, map2, cv2.INTER_LINEAR)
        else:
            frame = raw_frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        n_markers = len(ids) if ids is not None else 0
        status = "SEARCHING"

        # --- DETECTION ---
        if ids is not None and len(ids) >= 4:
            if ref_mode == "CENTER":
                ref_points = get_marker_centers(corners)
            else:
                ref_points = get_marker_inner_corners(corners)

            ordered_pts = order_points(ref_points)
            last_valid_pts = ordered_pts
            last_pixel_area = cv2.contourArea(ordered_pts)

            ppm = get_pixels_per_meter(corners, tag_size)
            if ppm and ppm > 0:
                last_ppm = ppm
                last_real_area = last_pixel_area / (ppm ** 2)

            status = "DETECTED"
        elif last_valid_pts is not None:
            status = "MEMORY"

        # --- DRAW OVERLAY ---
        if last_valid_pts is not None:
            pts_draw = last_valid_pts.astype(int).reshape((-1, 1, 2))

            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts_draw], (0, 255, 0))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            cv2.polylines(frame, [pts_draw], isClosed=True,
                          color=(0, 255, 0), thickness=2)

            labels = ["TL", "TR", "BR", "BL"]
            for pt, label in zip(last_valid_pts, labels):
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(frame, label, (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            if last_ppm and last_ppm > 0:
                for i in range(4):
                    p1 = last_valid_pts[i]
                    p2 = last_valid_pts[(i + 1) % 4]
                    mid = ((p1 + p2) / 2).astype(int)
                    dist_m = np.linalg.norm(p2 - p1) / last_ppm
                    cv2.putText(frame, f"{dist_m:.2f}m",
                                (mid[0] - 25, mid[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 2)

        # --- INFO PANEL ---
        h, w = frame.shape[:2]
        panel_x, panel_y = 10, h - 220
        panel_w = 420

        cv2.rectangle(frame, (panel_x, panel_y),
                      (panel_x + panel_w, h - 10), (0, 0, 0), -1)

        border_color = ((0, 255, 0) if status == "DETECTED"
                        else (0, 165, 255) if status == "MEMORY"
                        else (0, 0, 255))
        cv2.rectangle(frame, (panel_x, panel_y),
                      (panel_x + panel_w, h - 10), border_color, 2)

        y = panel_y + 25
        line_h = 28

        # Status
        cv2.putText(frame, f"STAGING ZONE: {status}", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_color, 2)
        y += line_h

        # Calibration status
        if use_calibration:
            cv2.putText(frame, f"CALIBRATED (err: {calib_error:.3f}px)",
                        (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "UNCALIBRATED (press 'u' info)",
                        (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        y += line_h

        # Reference mode
        mode_color = (255, 200, 0) if ref_mode == "CENTER" else (0, 200, 255)
        cv2.putText(frame, f"Ref: {ref_mode}  (press 'c' to toggle)",
                    (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 2)
        y += line_h

        # Area
        if last_real_area is not None:
            cv2.putText(frame, f"Area: {last_real_area:.4f} m^2",
                        (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y += line_h

        if last_pixel_area > 0:
            cv2.putText(frame, f"Pixel Area: {int(last_pixel_area):,} px^2",
                        (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += line_h

        if last_ppm and last_ppm > 0:
            cv2.putText(frame,
                        f"Scale: {last_ppm:.1f} px/m  (tag={tag_size}cm)",
                        (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            y += line_h

        cv2.putText(frame, f"Markers: {n_markers} visible",
                    (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # ArUco outlines
        if ids is not None and len(corners) > 0:
            aruco.drawDetectedMarkers(frame, corners, ids)

        # FPS
        frame_count += 1
        elapsed = time.time() - fps_time
        if elapsed > 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Staging Zone Monitor", frame)

        # --- KEY HANDLING ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            ref_mode = "CORNER" if ref_mode == "CENTER" else "CENTER"
            last_valid_pts = None
            last_real_area = None
            last_ppm = None
            last_pixel_area = 0
            print(f"[INFO] Switched to {ref_mode} mode")
        elif key == ord('u'):
            # Toggle undistortion
            if camera_matrix is not None:
                use_calibration = not use_calibration
                calib_label = "ON" if use_calibration else "OFF"
                last_valid_pts = None
                last_real_area = None
                last_ppm = None
                last_pixel_area = 0
                print(f"[INFO] Undistortion: {calib_label}")
            else:
                print("[INFO] No calibration data loaded. Run calibrate_camera.py first.")
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"snapshot_{ts}.png"
            cv2.imwrite(fname, frame)
            print(f"[INFO] Saved: {fname}")
        elif key == ord('r'):
            last_valid_pts = None
            last_ppm = None
            last_real_area = None
            last_pixel_area = 0
            print("[INFO] Zone memory reset")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()