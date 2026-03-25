"""
Camera Calibration Tool for ArUco Staging Zone Tracker
=======================================================
Run this BEFORE custom_track_v7.py to calibrate your camera.
Calibration corrects lens distortion, improving area accuracy.

3-Step Process:
    Step 1: Generate a printable ChArUco calibration board
    Step 2: Capture calibration images (show board to camera at different angles)
    Step 3: Compute & save calibration data

Usage:
    # Step 1: Generate the calibration board (print this on A4 paper)
    python calibrate_camera.py --generate-board

    # Step 2 + 3: Run calibration with live camera
    python calibrate_camera.py --calibrate

    # Optional: specify camera ID
    python calibrate_camera.py --calibrate --camera 1

    # Optional: verify calibration by viewing undistorted feed
    python calibrate_camera.py --verify
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import argparse
import os
import sys
import time
import glob


# =============================================================================
# Configuration
# =============================================================================

# ChArUco board parameters
CHARUCO_ROWS = 5        # Number of chessboard squares vertically
CHARUCO_COLS = 5        # Number of chessboard squares horizontally
SQUARE_LENGTH = 0.026   # Chessboard square size in meters (2.6 cm)
MARKER_LENGTH = 0.025   # ArUco marker size in meters (2.6 cm)

# Print settings
PRINT_DPI = 300         # Printer resolution (dots per inch)

# ArUco dictionary (use a DIFFERENT dict from your staging markers to avoid conflicts)
CHARUCO_DICT = aruco.DICT_4X4_50

# Output files
CALIBRATION_FILE = "camera_calibration.npz"
BOARD_IMAGE_FILE = "charuco_board.png"
CALIB_IMAGES_DIR = "calibration_images"

# Minimum number of calibration images required
MIN_IMAGES = 12


# =============================================================================
# Step 1: Generate Printable ChArUco Board
# =============================================================================

def generate_board():
    """Generate a printable ChArUco calibration board image.
    
    The board is sized so that the printed squares match SQUARE_LENGTH exactly
    when printed at PRINT_DPI. The board is centered on an A4 page.
    """
    aruco_dict = aruco.getPredefinedDictionary(CHARUCO_DICT)

    board = aruco.CharucoBoard(
        (CHARUCO_COLS, CHARUCO_ROWS),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        aruco_dict
    )

    # --- Compute pixel dimensions so printed size matches SQUARE_LENGTH ---
    # Conversion: meters -> inches -> pixels at PRINT_DPI
    meters_to_inches = 39.3701
    square_px = int(SQUARE_LENGTH * meters_to_inches * PRINT_DPI)

    # Ensure minimum pixel size per square so ArUco bits are resolvable.
    # DICT_4X4_50 markers need at least 6x6 px (4 data + 2 border bits).
    # We want a comfortable margin, so enforce at least 30 px per square.
    MIN_SQUARE_PX = 30
    if square_px < MIN_SQUARE_PX:
        print(f"[WARNING] At {PRINT_DPI} DPI, a {SQUARE_LENGTH*1000:.1f} mm square "
              f"is only {square_px} px — too small to encode ArUco bits reliably.")
        print(f"[INFO]    Upscaling to {MIN_SQUARE_PX} px per square for readability.")
        print(f"[INFO]    The printed squares will be LARGER than {SQUARE_LENGTH*1000:.1f} mm.")
        print(f"[INFO]    Measure the printed square and update SQUARE_LENGTH accordingly.\n")
        square_px = MIN_SQUARE_PX

    board_width_px = CHARUCO_COLS * square_px
    board_height_px = CHARUCO_ROWS * square_px

    # Generate the board image at the exact computed size (no extra margin here,
    # we will center it on an A4 canvas ourselves).
    board_image = board.generateImage(
        (board_width_px, board_height_px),
        marginSize=0,
        borderBits=1
    )

    # --- Center on an A4 canvas (300 DPI = 2480 x 3508 px) ---
    a4_w = int(8.27 * PRINT_DPI)   # 210 mm
    a4_h = int(11.69 * PRINT_DPI)  # 297 mm

    if board_width_px > a4_w or board_height_px > a4_h:
        print(f"[WARNING] Board ({board_width_px}x{board_height_px} px) exceeds "
              f"A4 page ({a4_w}x{a4_h} px) at {PRINT_DPI} DPI.")
        print(f"[INFO]    Reduce CHARUCO_COLS/ROWS or SQUARE_LENGTH, or increase DPI.\n")
        # Still save it, just won't fit on one sheet
        canvas = board_image
    else:
        canvas = np.full((a4_h, a4_w), 255, dtype=np.uint8)
        y_offset = (a4_h - board_height_px) // 2
        x_offset = (a4_w - board_width_px) // 2
        canvas[y_offset:y_offset + board_height_px,
               x_offset:x_offset + board_width_px] = board_image

    cv2.imwrite(BOARD_IMAGE_FILE, canvas)

    actual_square_mm = square_px / (PRINT_DPI * meters_to_inches) * 1000

    print(f"\n{'='*60}")
    print(f"  CHARUCO CALIBRATION BOARD GENERATED")
    print(f"{'='*60}")
    print(f"  Saved to       : {BOARD_IMAGE_FILE}")
    print(f"  Board layout   : {CHARUCO_COLS} x {CHARUCO_ROWS} squares")
    print(f"  Square size    : {SQUARE_LENGTH*1000:.1f} mm  ({square_px} px at {PRINT_DPI} DPI)")
    print(f"  Marker size    : {MARKER_LENGTH*1000:.1f} mm")
    print(f"  ArUco dict     : DICT_4X4_50")
    print(f"  Board pixels   : {board_width_px} x {board_height_px}")
    print(f"  Canvas (A4)    : {a4_w} x {a4_h}")
    print(f"")
    print(f"  INSTRUCTIONS:")
    print(f"  1. Print this image on A4 paper at ACTUAL SIZE (no scaling)")
    print(f"  2. Stick it flat onto cardboard or a rigid surface")
    print(f"  3. Measure one black square with a ruler — it should be")
    print(f"     {actual_square_mm:.1f} mm. If not, update SQUARE_LENGTH to match.")
    print(f"  4. Then run: python calibrate_camera.py --calibrate")
    print(f"{'='*60}\n")

    return canvas


# =============================================================================
# Step 2 + 3: Capture Images & Calibrate
# =============================================================================

def run_calibration(camera_id=0):
    """
    Interactive calibration process:
    - Shows live camera feed
    - User presses SPACE to capture calibration frames
    - Need at least MIN_IMAGES good frames from different angles
    - Then computes and saves calibration parameters
    """
    aruco_dict = aruco.getPredefinedDictionary(CHARUCO_DICT)
    board = aruco.CharucoBoard(
        (CHARUCO_COLS, CHARUCO_ROWS),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        aruco_dict
    )

    charuco_detector = aruco.CharucoDetector(board)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_id}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Create directory for saving calibration images
    os.makedirs(CALIB_IMAGES_DIR, exist_ok=True)

    # Storage for calibration data
    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None
    captured_count = 0

    print(f"\n{'='*60}")
    print(f"  CAMERA CALIBRATION - CAPTURE MODE")
    print(f"{'='*60}")
    print(f"  Hold the printed ChArUco board in front of the camera.")
    print(f"  Move it to different positions and angles.")
    print(f"")
    print(f"  TIPS for best calibration:")
    print(f"  - Cover all areas of the frame (center, edges, corners)")
    print(f"  - Tilt the board at various angles (15-45 degrees)")
    print(f"  - Move closer and farther from the camera")
    print(f"  - Keep the board steady when capturing")
    print(f"  - Need at least {MIN_IMAGES} good captures")
    print(f"")
    print(f"  Controls:")
    print(f"    SPACE = capture frame")
    print(f"    'c'   = run calibration (after enough captures)")
    print(f"    'q'   = quit without saving")
    print(f"{'='*60}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if image_size is None:
            image_size = (frame.shape[1], frame.shape[0])

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ChArUco board
        charuco_corners, charuco_ids, marker_corners, marker_ids = \
            charuco_detector.detectBoard(gray)

        n_markers = len(marker_ids) if marker_ids is not None else 0
        n_charuco = len(charuco_ids) if charuco_ids is not None else 0

        # Draw detected markers
        if marker_ids is not None and len(marker_corners) > 0:
            aruco.drawDetectedMarkers(display, marker_corners, marker_ids)

        # Draw detected ChArUco corners
        if charuco_ids is not None and len(charuco_corners) > 0:
            aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)

        # Quality indicator
        if n_charuco >= 6:
            quality_color = (0, 255, 0)
            quality_text = "GOOD - Press SPACE to capture"
        elif n_charuco >= 3:
            quality_color = (0, 165, 255)
            quality_text = "FAIR - Try to show more of the board"
        else:
            quality_color = (0, 0, 255)
            quality_text = "NO BOARD DETECTED"

        # Info panel
        cv2.rectangle(display, (10, 10), (550, 130), (0, 0, 0), -1)
        cv2.rectangle(display, (10, 10), (550, 130), quality_color, 2)

        cv2.putText(display, f"CALIBRATION CAPTURE ({captured_count}/{MIN_IMAGES})",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f"Markers: {n_markers}  |  Corners: {n_charuco}",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, quality_text,
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, quality_color, 2)

        if captured_count >= MIN_IMAGES:
            cv2.putText(display, "Press 'c' to compute calibration!",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow("Camera Calibration", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # SPACE - capture
            if charuco_ids is not None and n_charuco >= 6:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                captured_count += 1

                # Save the image
                img_path = os.path.join(CALIB_IMAGES_DIR,
                                        f"calib_{captured_count:03d}.png")
                cv2.imwrite(img_path, frame)

                print(f"[CAPTURE {captured_count}] {n_charuco} corners detected"
                      f" - saved to {img_path}")

                # Flash effect
                flash = np.full_like(display, 255)
                cv2.imshow("Camera Calibration", flash)
                cv2.waitKey(100)
            else:
                print("[SKIP] Not enough corners detected. Show more of the board.")

        elif key == ord('c'):  # Compute calibration
            if captured_count >= MIN_IMAGES:
                break
            else:
                print(f"[INFO] Need at least {MIN_IMAGES} captures."
                      f" Have {captured_count}.")

        elif key == ord('q'):
            print("[INFO] Calibration cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    # --- Compute Calibration ---
    print(f"\n[INFO] Computing calibration from {captured_count} images...")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        all_charuco_corners,
        all_charuco_ids,
        board,
        image_size,
        None,
        None
    )

    if ret:
        # Calculate reprojection error
        total_error = 0
        for i in range(len(all_charuco_corners)):
            corners2, _ = cv2.projectPoints(
                board.getChessboardCorners()[all_charuco_ids[i].flatten()],
                rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(all_charuco_corners[i], corners2, cv2.NORM_L2)
            total_error += error
        mean_error = total_error / len(all_charuco_corners)

        # Save calibration
        np.savez(CALIBRATION_FILE,
                 camera_matrix=camera_matrix,
                 dist_coeffs=dist_coeffs,
                 image_size=np.array(image_size),
                 reprojection_error=np.array([mean_error]),
                 n_images=np.array([captured_count]))

        print(f"\n{'='*60}")
        print(f"  CALIBRATION COMPLETE")
        print(f"{'='*60}")
        print(f"  Saved to          : {CALIBRATION_FILE}")
        print(f"  Images used       : {captured_count}")
        print(f"  Reprojection error: {mean_error:.4f} px")
        print(f"  (Lower is better. Under 0.5 is excellent, under 1.0 is good)")
        print(f"")
        print(f"  Camera Matrix:")
        print(f"    fx = {camera_matrix[0,0]:.2f}")
        print(f"    fy = {camera_matrix[1,1]:.2f}")
        print(f"    cx = {camera_matrix[0,2]:.2f}")
        print(f"    cy = {camera_matrix[1,2]:.2f}")
        print(f"")
        print(f"  Distortion Coefficients:")
        print(f"    {dist_coeffs.flatten()}")
        print(f"")
        print(f"  Next step: run custom_track_v7_calibrated.py")
        print(f"{'='*60}\n")
    else:
        print("[ERROR] Calibration failed. Try capturing more diverse angles.")


# =============================================================================
# Verify: View Undistorted Feed
# =============================================================================

def verify_calibration(camera_id=0):
    """Show side-by-side comparison of raw vs undistorted camera feed."""
    if not os.path.exists(CALIBRATION_FILE):
        print(f"[ERROR] No calibration file found: {CALIBRATION_FILE}")
        print(f"[INFO]  Run: python calibrate_camera.py --calibrate")
        sys.exit(1)

    # Load calibration
    data = np.load(CALIBRATION_FILE)
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
    mean_error = data['reprojection_error'][0]

    print(f"[INFO] Loaded calibration (error: {mean_error:.4f} px)")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_id}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Pre-compute undistortion maps for speed
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1)

    print("[INFO] Showing RAW (left) vs UNDISTORTED (right)")
    print("[INFO] Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Undistort
        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        # Crop to valid region
        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            undistorted_cropped = undistorted[y:y+rh, x:x+rw]
            undistorted_cropped = cv2.resize(undistorted_cropped, (w, h))
        else:
            undistorted_cropped = undistorted

        # Labels
        cv2.putText(frame, "RAW", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(undistorted_cropped, "UNDISTORTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Draw grid lines to visualize distortion correction
        for i in range(0, w, 80):
            cv2.line(frame, (i, 0), (i, h), (50, 50, 50), 1)
            cv2.line(undistorted_cropped, (i, 0), (i, h), (50, 50, 50), 1)
        for i in range(0, h, 80):
            cv2.line(frame, (0, i), (w, i), (50, 50, 50), 1)
            cv2.line(undistorted_cropped, (0, i), (w, i), (50, 50, 50), 1)

        # Side by side
        combined = np.hstack([
            cv2.resize(frame, (w // 2, h // 2)),
            cv2.resize(undistorted_cropped, (w // 2, h // 2))
        ])

        cv2.imshow("Calibration Verification (RAW | UNDISTORTED)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Camera Calibration for ArUco Staging Zone Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. python calibrate_camera.py --generate-board
     → Print the board image on paper, stick to flat surface

  2. python calibrate_camera.py --calibrate
     → Show board to camera at different angles, press SPACE to capture

  3. python calibrate_camera.py --verify
     → Check the undistortion quality

  4. Run custom_track_v7_calibrated.py (uses the calibration data)
        """
    )

    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--generate-board", action="store_true",
                         help="Generate printable ChArUco calibration board")
    actions.add_argument("--calibrate", action="store_true",
                         help="Run interactive calibration capture")
    actions.add_argument("--verify", action="store_true",
                         help="View raw vs undistorted feed")

    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device ID (default: 0)")

    args = parser.parse_args()

    if args.generate_board:
        generate_board()
    elif args.calibrate:
        run_calibration(args.camera)
    elif args.verify:
        verify_calibration(args.camera)


if __name__ == "__main__":
    main()