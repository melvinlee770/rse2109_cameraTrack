"""
Staging Zone Tracker v11 - Zone Utilization Monitor
=====================================================
Combines:
  - ArUco staging zone detection & area calculation
  - Camera calibration (undistortion)
  - YOLOv8 pallet detection with ArUco false positive filtering
  - ** Staging zone UTILIZATION % calculation **

Utilization = (area occupied by pallets inside zone) / (total zone area) × 100%

Uses a pixel-mask approach for accurate calculation:
  1. Create mask of the staging zone polygon
  2. Create mask of all pallet bounding boxes inside the zone
  3. Utilization = (pallet mask AND zone mask) / (zone mask) × 100%
  This automatically handles overlapping pallets (no double counting).

Usage:
    python custom_track_v11.py --model best_v4.pt
    python custom_track_v11.py --model best_v4.pt --camera 1
    python custom_track_v11.py --model best_v4.pt --no-calib
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
    # alpha=0: crop to valid region (no fisheye)
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
    """Get expanded bounding boxes around each ArUco marker."""
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
    """Overlap ratio of YOLO box with an ArUco region."""
    x1 = max(box[0], region[0])
    y1 = max(box[1], region[1])
    x2 = min(box[2], region[2])
    y2 = min(box[3], region[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    overlap_area = (x2 - x1) * (y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    return overlap_area / box_area if box_area > 0 else 0.0


def filter_yolo_detections(results, aruco_regions, min_area_px=500,
                           overlap_threshold=0.3, conf_threshold=0.5):
    """
    Filter YOLO detections:
    1. Confidence threshold
    2. Size filter (reject tiny = likely ArUco)
    3. Overlap filter (reject if overlaps ArUco marker)
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

        if conf < conf_threshold:
            continue

        box_area = (x2 - x1) * (y2 - y1)
        if box_area < min_area_px:
            continue

        is_aruco = False
        for region in aruco_regions:
            if box_overlap_ratio((x1, y1, x2, y2), region) > overlap_threshold:
                is_aruco = True
                break
        if is_aruco:
            continue

        filtered.append((int(x1), int(y1), int(x2), int(y2),
                          conf, cls_id, cls_name))
    return filtered


def is_inside_zone(box, zone_pts):
    """Check if center of detection box is inside the zone polygon."""
    if zone_pts is None:
        return False
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    result = cv2.pointPolygonTest(
        zone_pts.astype(np.float32).reshape((-1, 1, 2)),
        (float(cx), float(cy)), False)
    return result >= 0


# =============================================================================
# UTILIZATION CALCULATION
# =============================================================================

def calculate_utilization(frame_shape, zone_pts, detections):
    """
    Calculate staging zone utilization percentage using pixel masks.

    How it works:
    1. Create a binary mask of the staging zone polygon
    2. Create a binary mask of all pallet bounding boxes that are INSIDE the zone
    3. AND the two masks → pixels where pallets overlap with zone
    4. Utilization = (overlapping pixels) / (total zone pixels) × 100%

    This method:
    - Handles overlapping pallets correctly (no double counting)
    - Only counts the portion of pallet boxes that are actually inside the zone
    - Works regardless of camera angle or zone shape

    Args:
        frame_shape: (height, width) of the frame
        zone_pts: ordered zone corner points (4, 2)
        detections: list of filtered YOLO detections

    Returns:
        utilization_pct: float (0.0 to 100.0)
        zone_area_px: total zone area in pixels
        occupied_area_px: occupied area in pixels
        n_pallets_in_zone: number of pallets inside the zone
    """
    h, w = frame_shape[:2]

    # Step 1: Create zone mask
    zone_mask = np.zeros((h, w), dtype=np.uint8)
    zone_poly = zone_pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(zone_mask, [zone_poly], 255)

    zone_area_px = cv2.countNonZero(zone_mask)
    if zone_area_px == 0:
        return 0.0, 0, 0, 0

    # Step 2: Create pallet mask (only pallets inside the zone)
    pallet_mask = np.zeros((h, w), dtype=np.uint8)
    n_pallets_in_zone = 0

    for (x1, y1, x2, y2, conf, cls_id, cls_name) in detections:
        if is_inside_zone((x1, y1, x2, y2), zone_pts):
            # Clamp to frame bounds
            x1c = max(0, x1)
            y1c = max(0, y1)
            x2c = min(w, x2)
            y2c = min(h, y2)
            cv2.rectangle(pallet_mask, (x1c, y1c), (x2c, y2c), 255, -1)
            n_pallets_in_zone += 1

    # Step 3: AND masks — only count pallet area that's inside the zone
    occupied_mask = cv2.bitwise_and(pallet_mask, zone_mask)
    occupied_area_px = cv2.countNonZero(occupied_mask)

    # Step 4: Calculate percentage
    utilization_pct = (occupied_area_px / zone_area_px) * 100.0

    return utilization_pct, zone_area_px, occupied_area_px, n_pallets_in_zone


def calculate_real_occupied_area(occupied_area_px, ppm):
    """Convert occupied pixel area to real-world m²."""
    if ppm and ppm > 0:
        return occupied_area_px / (ppm ** 2)
    return None


# =============================================================================
# Visualization
# =============================================================================

def get_utilization_color(pct):
    """
    Color gradient based on utilization:
      0-30%  = Green  (low usage, lots of space)
      30-70% = Yellow/Orange (moderate usage)
      70-100% = Red (high usage, nearly full)
    """
    if pct < 30:
        return (0, 255, 0)       # Green
    elif pct < 50:
        return (0, 255, 255)     # Yellow
    elif pct < 70:
        return (0, 165, 255)     # Orange
    elif pct < 90:
        return (0, 0, 255)       # Red
    else:
        return (0, 0, 200)       # Dark red


def draw_utilization_bar(frame, pct, x, y, bar_width=300, bar_height=25):
    """Draw a visual utilization bar on the frame."""
    color = get_utilization_color(pct)

    # Background bar
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height),
                  (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height),
                  (150, 150, 150), 1)

    # Filled portion
    fill_width = int(bar_width * min(pct, 100.0) / 100.0)
    if fill_width > 0:
        cv2.rectangle(frame, (x, y), (x + fill_width, y + bar_height),
                      color, -1)

    # Percentage text on bar
    text = f"{pct:.1f}%"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    text_x = x + (bar_width - text_size[0]) // 2
    text_y = y + (bar_height + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_detections(frame, detections, zone_pts):
    """Draw filtered YOLO detections with in/out zone labels."""
    in_zone = 0
    out_zone = 0

    for (x1, y1, x2, y2, conf, cls_id, cls_name) in detections:
        inside = is_inside_zone((x1, y1, x2, y2), zone_pts)

        if inside:
            color = (0, 165, 255)  # Orange
            in_zone += 1
        else:
            color = (255, 0, 0)    # Blue
            out_zone += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name} {conf:.2f}"
        lbl_sz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - lbl_sz[1] - 8),
                      (x1 + lbl_sz[0] + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        zone_label = "IN ZONE" if inside else "OUTSIDE"
        cv2.putText(frame, zone_label, (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    return in_zone, out_zone


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Staging Zone Utilization Monitor")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--calib-file", type=str, default="camera_calibration.npz")
    parser.add_argument("--no-calib", action="store_true")
    parser.add_argument("--tag-size", type=float, default=TAG_SIZE_CM)
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    # --- Load YOLO ---
    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        sys.exit(1)
    print(f"[INFO] Loading YOLO: {args.model}")
    model = YOLO(args.model)
    print(f"[INFO] Classes: {model.names}")

    # --- Calibration ---
    use_calibration = False
    map1, map2 = None, None
    camera_matrix, dist_coeffs = None, None
    calib_error = 0

    if not args.no_calib:
        camera_matrix, dist_coeffs, calib_error = load_calibration(args.calib_file)
        if camera_matrix is not None:
            use_calibration = True
            print(f"[INFO] Calibration loaded (err: {calib_error:.4f}px)")
        else:
            print(f"[WARNING] No calibration file. Running uncalibrated.")

    # --- Camera ---
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

    # Utilization state (smoothed)
    utilization_pct = 0.0
    utilization_smooth = 0.0  # Exponential moving average for stable display
    SMOOTH_ALPHA = 0.3        # 0.0 = very smooth, 1.0 = instant

    # FPS
    fps_time = time.time()
    frame_count = 0
    fps = 0.0

    print(f"[INFO] Tag: {args.tag_size}cm | Conf: {args.conf}")
    print(f"[INFO] Controls:")
    print(f"  'q'=quit  'c'=ref mode  'r'=reset  's'=snapshot")
    print(f"  'u'=undistort  '+'/'-'=confidence")

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

        # =================================================================
        # STEP 1: ArUco detection
        # =================================================================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_corners, aruco_ids, _ = aruco_detector.detectMarkers(gray)

        n_markers = len(aruco_ids) if aruco_ids is not None else 0
        aruco_regions = get_aruco_regions(aruco_corners) if aruco_corners else []
        status = "SEARCHING"

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

        # =================================================================
        # STEP 2: YOLO pallet detection
        # =================================================================
        results = model(frame, verbose=False, conf=args.conf)

        filtered_detections = filter_yolo_detections(
            results, aruco_regions,
            min_area_px=500,
            overlap_threshold=0.3,
            conf_threshold=args.conf)

        # =================================================================
        # STEP 3: Calculate utilization
        # =================================================================
        n_pallets_in_zone = 0
        occupied_area_px = 0
        occupied_area_m2 = None

        if last_valid_pts is not None:
            utilization_pct, zone_px, occupied_area_px, n_pallets_in_zone = \
                calculate_utilization(frame.shape, last_valid_pts,
                                     filtered_detections)

            # Smooth the percentage for stable display
            utilization_smooth = (SMOOTH_ALPHA * utilization_pct +
                                  (1 - SMOOTH_ALPHA) * utilization_smooth)

            # Real-world occupied area
            occupied_area_m2 = calculate_real_occupied_area(
                occupied_area_px, last_ppm)

        # =================================================================
        # DRAW: Zone overlay
        # =================================================================
        if last_valid_pts is not None:
            pts_draw = last_valid_pts.astype(int).reshape((-1, 1, 2))

            # Zone fill — color changes with utilization
            util_color = get_utilization_color(utilization_smooth)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts_draw], util_color)
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

        # =================================================================
        # DRAW: Pallet detections
        # =================================================================
        in_zone, out_zone = draw_detections(
            frame, filtered_detections, last_valid_pts)

        # ArUco outlines
        if aruco_ids is not None and len(aruco_corners) > 0:
            aruco.drawDetectedMarkers(frame, aruco_corners, aruco_ids)

        # =================================================================
        # UTILIZATION DISPLAY (top-center, prominent)
        # =================================================================
        h, w = frame.shape[:2]

        if last_valid_pts is not None:
            # Large utilization percentage display
            util_display_color = get_utilization_color(utilization_smooth)
            util_text = f"UTILIZATION: {utilization_smooth:.1f}%"

            # Background box (top center)
            text_sz, _ = cv2.getTextSize(util_text, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.9, 2)
            box_w = text_sz[0] + 40
            box_x = (w - box_w) // 2
            cv2.rectangle(frame, (box_x, 5), (box_x + box_w, 75),
                          (0, 0, 0), -1)
            cv2.rectangle(frame, (box_x, 5), (box_x + box_w, 75),
                          util_display_color, 2)

            cv2.putText(frame, util_text,
                        (box_x + 20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, util_display_color, 2)

            # Utilization bar
            draw_utilization_bar(frame, utilization_smooth,
                                 box_x + 10, 48, box_w - 20, 20)

        # =================================================================
        # INFO PANEL (bottom-left)
        # =================================================================
        panel_x, panel_y = 10, h - 280
        panel_w = 420

        cv2.rectangle(frame, (panel_x, panel_y),
                      (panel_x + panel_w, h - 10), (0, 0, 0), -1)

        border_color = ((0, 255, 0) if status == "DETECTED"
                        else (0, 165, 255) if status == "MEMORY"
                        else (0, 0, 255))
        cv2.rectangle(frame, (panel_x, panel_y),
                      (panel_x + panel_w, h - 10), border_color, 2)

        y = panel_y + 22
        lh = 24

        # Status
        cv2.putText(frame, f"STAGING ZONE: {status}",
                    (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, border_color, 2)
        y += lh

        # Calibration
        if use_calibration:
            cv2.putText(frame, f"CALIBRATED (err: {calib_error:.3f}px)",
                        (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "UNCALIBRATED",
                        (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
        y += lh

        # Ref mode
        mode_color = (255, 200, 0) if ref_mode == "CENTER" else (0, 200, 255)
        cv2.putText(frame, f"Ref: {ref_mode}  ('c' toggle)",
                    (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, mode_color, 1)
        y += lh

        # Zone area
        if last_real_area is not None:
            cv2.putText(frame, f"Zone Area: {last_real_area:.4f} m^2",
                        (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            y += lh

        # Occupied area
        if occupied_area_m2 is not None:
            cv2.putText(frame,
                        f"Occupied: {occupied_area_m2:.4f} m^2",
                        (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
            y += lh

        # Pallet counts
        total_det = len(filtered_detections)
        raw_det = len(results[0].boxes) if results[0].boxes is not None else 0
        filtered_out = raw_det - total_det

        cv2.putText(frame,
                    f"Pallets: {total_det} ({filtered_out} filtered)",
                    (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        y += lh

        cv2.putText(frame,
                    f"  In zone: {in_zone}  |  Outside: {out_zone}",
                    (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        y += lh

        # Scale + markers
        if last_ppm and last_ppm > 0:
            cv2.putText(frame,
                        f"Scale: {last_ppm:.0f} px/m | Markers: {n_markers}",
                        (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
            y += lh

        cv2.putText(frame, f"Conf: {args.conf:.2f} ('+'/'-')",
                    (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        # FPS (top-right)
        frame_count += 1
        elapsed = time.time() - fps_time
        if elapsed > 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Staging Zone Utilization Monitor", frame)

        # --- KEYS ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            ref_mode = "CORNER" if ref_mode == "CENTER" else "CENTER"
            last_valid_pts = None
            last_real_area = None
            last_ppm = None
            last_pixel_area = 0
            utilization_smooth = 0.0
            print(f"[INFO] Ref mode: {ref_mode}")
        elif key == ord('u'):
            if camera_matrix is not None:
                use_calibration = not use_calibration
                last_valid_pts = None
                last_real_area = None
                last_ppm = None
                last_pixel_area = 0
                utilization_smooth = 0.0
                print(f"[INFO] Undistortion: {'ON' if use_calibration else 'OFF'}")
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
            utilization_smooth = 0.0
            print("[INFO] Zone reset")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()