"""
Staging Zone Tracker + YOLO Pallet Detection
==============================================
Combines ArUco staging zone tracking with YOLOv8 pallet detection.
Filters out false positive detections on ArUco markers.

Usage:
    python custom_track_v10.py
    python custom_track_v10.py --camera 1
    python custom_track_v10.py --model best_v4.pt
    python custom_track_v10.py --calib-file camera_calibration.npz
    python custom_track_v10.py --no-calib
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import time
import argparse
import os
import sys
from ultralytics import YOLO


# =============================================================================
# Configuration
# =============================================================================

TAG_SIZE_CM = 2.4
MODEL_PATH = "best_v4.pt"
CONFIDENCE_THRESHOLD = 0.5

# ArUco setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_params = aruco.DetectorParameters()
aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 53
aruco_params.adaptiveThreshWinSizeStep = 4
aruco_params.minMarkerPerimeterRate = 0.01
aruco_detector = aruco.ArucoDetector(aruco_dict, aruco_params)


# =============================================================================
# Calibration
# =============================================================================

def load_calibration(calib_file):
    if not os.path.exists(calib_file):
        return None, None, None
    data = np.load(calib_file)
    return data['camera_matrix'], data['dist_coeffs'], data['reprojection_error'][0]


def setup_undistortion(camera_matrix, dist_coeffs, w, h):
    new_cam, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 0, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_cam, (w, h), cv2.CV_32FC1)
    return map1, map2, new_cam, roi


# =============================================================================
# Zone Helpers
# =============================================================================

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def get_marker_centers(corners):
    return np.array([np.mean(c[0], axis=0) for c in corners])


def get_marker_inner_corners(corners):
    all_centers = np.array([np.mean(c[0], axis=0) for c in corners])
    centroid = np.mean(all_centers, axis=0)
    inner_points = []
    for mc in corners:
        pts = mc[0]
        distances = np.linalg.norm(pts - centroid, axis=1)
        inner_points.append(pts[np.argmin(distances)])
    return np.array(inner_points, dtype=np.float32)


def get_pixels_per_meter(corners, tag_size_cm):
    edge_lengths = []
    for mc in corners:
        pts = mc[0]
        for i in range(4):
            edge_lengths.append(np.linalg.norm(pts[(i+1) % 4] - pts[i]))
    if not edge_lengths:
        return None
    return (np.mean(edge_lengths) / tag_size_cm) * 100.0


# =============================================================================
# ArUco False Positive Filter
# =============================================================================

def get_aruco_regions(corners, margin=15):
    """
    Get bounding boxes around each detected ArUco marker,
    expanded by a margin. Used to filter YOLO false positives.

    Returns list of (x1, y1, x2, y2) bounding boxes.
    """
    regions = []
    for mc in corners:
        pts = mc[0]
        x_min = int(pts[:, 0].min()) - margin
        y_min = int(pts[:, 1].min()) - margin
        x_max = int(pts[:, 0].max()) + margin
        y_max = int(pts[:, 1].max()) + margin
        regions.append((x_min, y_min, x_max, y_max))
    return regions


def box_overlap_ratio(box, region):
    """
    Calculate how much of a YOLO detection box overlaps with an ArUco region.
    Returns overlap ratio (0.0 to 1.0) relative to the YOLO box area.
    """
    x1 = max(box[0], region[0])
    y1 = max(box[1], region[1])
    x2 = min(box[2], region[2])
    y2 = min(box[3], region[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    overlap_area = (x2 - x1) * (y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])

    if box_area <= 0:
        return 0.0

    return overlap_area / box_area


def filter_yolo_detections(results, aruco_regions, min_area_px=500,
                           overlap_threshold=0.3, conf_threshold=0.5):
    """
    Filter YOLO detections to remove false positives caused by ArUco markers.

    Filters applied:
    1. Confidence threshold — reject low-confidence detections
    2. Overlap filter — reject detections that overlap significantly with ArUco markers
    3. Size filter — reject detections that are too small (likely marker-sized)

    Returns:
        filtered_boxes: list of (x1, y1, x2, y2, confidence, class_id, class_name)
    """
    filtered = []

    if results[0].boxes is None or len(results[0].boxes) == 0:
        return filtered

    boxes = results[0].boxes

    for i in range(len(boxes)):
        conf = float(boxes.conf[i])
        cls_id = int(boxes.cls[i])
        cls_name = results[0].names[cls_id]
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()

        # Filter 1: Confidence
        if conf < conf_threshold:
            continue

        # Filter 2: Size — reject tiny detections (likely ArUco tags)
        box_area = (x2 - x1) * (y2 - y1)
        if box_area < min_area_px:
            continue

        # Filter 3: Overlap with ArUco markers
        is_aruco_overlap = False
        for region in aruco_regions:
            overlap = box_overlap_ratio((x1, y1, x2, y2), region)
            if overlap > overlap_threshold:
                is_aruco_overlap = True
                break

        if is_aruco_overlap:
            continue

        filtered.append((int(x1), int(y1), int(x2), int(y2),
                          conf, cls_id, cls_name))

    return filtered


def is_inside_zone(box, zone_pts):
    """Check if the center of a detection box is inside the staging zone."""
    if zone_pts is None:
        return False
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    result = cv2.pointPolygonTest(
        zone_pts.astype(np.float32).reshape((-1, 1, 2)),
        (float(cx), float(cy)), False)
    return result >= 0


# =============================================================================
# Drawing
# =============================================================================

def draw_detections(frame, detections, zone_pts):
    """Draw filtered YOLO detections on the frame."""
    in_zone_count = 0
    out_zone_count = 0

    for (x1, y1, x2, y2, conf, cls_id, cls_name) in detections:
        inside = is_inside_zone((x1, y1, x2, y2), zone_pts)

        if inside:
            color = (0, 165, 255)  # Orange — inside staging zone
            in_zone_count += 1
        else:
            color = (255, 0, 0)    # Blue — outside staging zone
            out_zone_count += 1

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"{cls_name} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 8),
                      (x1 + label_size[0] + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # "IN ZONE" / "OUTSIDE" tag
        zone_label = "IN ZONE" if inside else "OUTSIDE"
        cv2.putText(frame, zone_label, (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    return in_zone_count, out_zone_count


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Staging Zone + YOLO Pallet Tracker")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Path to YOLO model (default: best_v4.pt)")
    parser.add_argument("--calib-file", type=str, default="camera_calibration.npz")
    parser.add_argument("--no-calib", action="store_true")
    parser.add_argument("--tag-size", type=float, default=TAG_SIZE_CM)
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD,
                        help="YOLO confidence threshold (default: 0.5)")
    args = parser.parse_args()

    # --- Load YOLO model ---
    if not os.path.exists(args.model):
        print(f"[ERROR] YOLO model not found: {args.model}")
        sys.exit(1)

    print(f"[INFO] Loading YOLO model: {args.model}")
    model = YOLO(args.model)
    print(f"[INFO] Model classes: {model.names}")

    # --- Load Calibration ---
    use_calibration = False
    map1, map2 = None, None
    camera_matrix, dist_coeffs = None, None
    calib_error = 0

    if not args.no_calib:
        camera_matrix, dist_coeffs, calib_error = load_calibration(args.calib_file)
        if camera_matrix is not None:
            use_calibration = True
            print(f"[INFO] Calibration loaded (error: {calib_error:.4f}px)")
        else:
            print(f"[WARNING] No calibration file. Running uncalibrated.")
    else:
        print(f"[INFO] Running without calibration")

    # --- Open Camera ---
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    undistort_initialized = False

    # Zone state
    last_valid_pts = None
    last_ppm = None
    last_real_area = None
    last_pixel_area = 0
    ref_mode = "CENTER"

    # FPS
    fps_time = time.time()
    frame_count = 0
    fps = 0.0

    print(f"[INFO] Tag size: {args.tag_size}cm | Conf threshold: {args.conf}")
    print(f"[INFO] Controls:")
    print(f"         'q' = quit")
    print(f"         'c' = toggle CENTER / CORNER ref mode")
    print(f"         'r' = reset zone | 's' = snapshot")
    print(f"         'u' = toggle undistortion")
    print(f"         '+'/'-' = adjust confidence threshold")

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        # --- Undistortion ---
        if use_calibration and not undistort_initialized:
            h, w = raw_frame.shape[:2]
            map1, map2, _, _ = setup_undistortion(
                camera_matrix, dist_coeffs, w, h)
            undistort_initialized = True

        if use_calibration and map1 is not None:
            frame = cv2.remap(raw_frame, map1, map2, cv2.INTER_LINEAR)
        else:
            frame = raw_frame.copy()

        # =====================================================================
        # STEP 1: Detect ArUco markers (for zone + false positive filtering)
        # =====================================================================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_corners, aruco_ids, _ = aruco_detector.detectMarkers(gray)

        n_markers = len(aruco_ids) if aruco_ids is not None else 0
        status = "SEARCHING"

        # Get ArUco bounding regions (used to filter YOLO false positives)
        aruco_regions = get_aruco_regions(aruco_corners) if aruco_corners else []

        # --- Update staging zone ---
        if aruco_ids is not None and len(aruco_ids) >= 4:
            if ref_mode == "CENTER":
                ref_points = get_marker_centers(aruco_corners)
            else:
                ref_points = get_marker_inner_corners(aruco_corners)

            ordered_pts = order_points(ref_points)
            last_valid_pts = ordered_pts
            last_pixel_area = cv2.contourArea(ordered_pts)

            ppm = get_pixels_per_meter(aruco_corners, args.tag_size)
            if ppm and ppm > 0:
                last_ppm = ppm
                last_real_area = last_pixel_area / (ppm ** 2)

            status = "DETECTED"
        elif last_valid_pts is not None:
            status = "MEMORY"

        # =====================================================================
        # STEP 2: Run YOLO detection
        # =====================================================================
        results = model(frame, verbose=False, conf=args.conf)

        # =====================================================================
        # STEP 3: Filter out false positives on ArUco markers
        # =====================================================================
        filtered_detections = filter_yolo_detections(
            results,
            aruco_regions,
            min_area_px=500,            # reject tiny detections
            overlap_threshold=0.3,       # reject if 30%+ overlap with ArUco
            conf_threshold=args.conf
        )

        # =====================================================================
        # DRAW: Zone overlay
        # =====================================================================
        if last_valid_pts is not None:
            pts_draw = last_valid_pts.astype(int).reshape((-1, 1, 2))

            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts_draw], (0, 255, 0))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            cv2.polylines(frame, [pts_draw], isClosed=True,
                          color=(0, 255, 0), thickness=2)

            # Corner labels
            for pt, label in zip(last_valid_pts, ["TL", "TR", "BR", "BL"]):
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(frame, label, (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

            # Edge lengths
            if last_ppm and last_ppm > 0:
                for i in range(4):
                    p1 = last_valid_pts[i]
                    p2 = last_valid_pts[(i + 1) % 4]
                    mid = ((p1 + p2) / 2).astype(int)
                    dist_m = np.linalg.norm(p2 - p1) / last_ppm
                    cv2.putText(frame, f"{dist_m:.2f}m",
                                (mid[0] - 25, mid[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 255, 255), 2)

        # =====================================================================
        # DRAW: YOLO detections (filtered)
        # =====================================================================
        in_zone, out_zone = draw_detections(frame, filtered_detections, last_valid_pts)

        # Draw ArUco marker outlines
        if aruco_ids is not None and len(aruco_corners) > 0:
            aruco.drawDetectedMarkers(frame, aruco_corners, aruco_ids)

        # =====================================================================
        # INFO PANEL
        # =====================================================================
        h, w = frame.shape[:2]
        panel_x, panel_y = 10, h - 250
        panel_w = 420

        cv2.rectangle(frame, (panel_x, panel_y),
                      (panel_x + panel_w, h - 10), (0, 0, 0), -1)

        border_color = ((0, 255, 0) if status == "DETECTED"
                        else (0, 165, 255) if status == "MEMORY"
                        else (0, 0, 255))
        cv2.rectangle(frame, (panel_x, panel_y),
                      (panel_x + panel_w, h - 10), border_color, 2)

        y = panel_y + 22
        lh = 25

        cv2.putText(frame, f"STAGING ZONE: {status}", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, border_color, 2)
        y += lh

        # Calibration
        if use_calibration:
            cv2.putText(frame, f"CALIBRATED (err: {calib_error:.3f}px)",
                        (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "UNCALIBRATED",
                        (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        y += lh

        # Ref mode
        mode_color = (255, 200, 0) if ref_mode == "CENTER" else (0, 200, 255)
        cv2.putText(frame, f"Ref: {ref_mode}  ('c' toggle)",
                    (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, mode_color, 1)
        y += lh

        # Area
        if last_real_area is not None:
            cv2.putText(frame, f"Zone Area: {last_real_area:.4f} m^2",
                        (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y += lh

        # YOLO detections info
        total_det = len(filtered_detections)
        raw_det = len(results[0].boxes) if results[0].boxes is not None else 0
        cv2.putText(frame,
                    f"Pallets: {total_det} detected ({raw_det - total_det} filtered)",
                    (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += lh

        cv2.putText(frame,
                    f"  In zone: {in_zone}  |  Outside: {out_zone}",
                    (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        y += lh

        # Scale + markers
        if last_ppm and last_ppm > 0:
            cv2.putText(frame,
                        f"Scale: {last_ppm:.0f} px/m | Markers: {n_markers}",
                        (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y += lh

        cv2.putText(frame,
                    f"Conf: {args.conf:.2f} ('+'/'-' adjust)",
                    (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # FPS
        frame_count += 1
        elapsed = time.time() - fps_time
        if elapsed > 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Staging Zone + Pallet Detection", frame)

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
            print(f"[INFO] Ref mode: {ref_mode}")
        elif key == ord('u'):
            if camera_matrix is not None:
                use_calibration = not use_calibration
                last_valid_pts = None
                last_real_area = None
                last_ppm = None
                last_pixel_area = 0
                print(f"[INFO] Undistortion: {'ON' if use_calibration else 'OFF'}")
            else:
                print("[INFO] No calibration data available")
        elif key == ord('+') or key == ord('='):
            args.conf = min(0.95, args.conf + 0.05)
            print(f"[INFO] Confidence: {args.conf:.2f}")
        elif key == ord('-'):
            args.conf = max(0.05, args.conf - 0.05)
            print(f"[INFO] Confidence: {args.conf:.2f}")
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"snapshot_{ts}.png", frame)
            print(f"[INFO] Saved: snapshot_{ts}.png")
        elif key == ord('r'):
            last_valid_pts = None
            last_ppm = None
            last_real_area = None
            last_pixel_area = 0
            print("[INFO] Zone reset")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()