import cv2
import numpy as np
import math
from collections import deque

# 1. Setup for Multi-Camera Stability
area_history = deque(maxlen=30) # Increased to 30 for extra stability with two feeds

# Define your 2 camera indices (usually 0 and 1)
cam_indices = [0, 1]
caps = [cv2.VideoCapture(idx) for idx in cam_indices]

# ArUco Setup
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

print("🎥 Multi-Camera Sync Started! Press 'q' to exit.")

while True:
    all_centers = []
    frames = []
    ref_width_px = None

    # 2. Capture and Process each camera
    for i, cap in enumerate(caps):
        success, frame = cap.read()
        if not success:
            continue
        
        # Detect markers for this specific camera
        try:
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            corners, ids, rejected = detector.detectMarkers(frame)
        except AttributeError:
            corners, ids, rejected = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Extract centers and add to the GLOBAL list
            for corner_set in corners:
                pts = corner_set[0]
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                
                # IMPORTANT: If cameras are side-by-side, you'd add an offset here
                # For testing, we just collect all points found by both
                all_centers.append([cx, cy])

            # Use the first tag found by Cam 1 as the scale reference
            if ref_width_px is None and len(corners) > 0:
                ref_pts = corners[0][0]
                ref_width_px = math.dist(ref_pts[0], ref_pts[1])

        frames.append(frame)

    # 3. Calculate Global Area if we see at least 3 tags across BOTH cameras
    if len(all_centers) >= 3 and ref_width_px:
        centers_array = np.array(all_centers, dtype=np.int32)
        hull = cv2.convexHull(centers_array)
        
        # Calculate Area
        REAL_MARKER_WIDTH_MM = 25.0 
        pixels_per_mm = ref_width_px / REAL_MARKER_WIDTH_MM
        
        raw_area_px = cv2.contourArea(hull)
        current_area_cm2 = (raw_area_px / (pixels_per_mm**2)) / 100.0
        
        area_history.append(current_area_cm2)
        smooth_area = sum(area_history) / len(area_history)

        # Draw the result on the first frame for display
        cv2.putText(frames[0], f"Global Area: {smooth_area:.1f} cm sq", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

    # 4. Show both camera feeds
    for i, f in enumerate(frames):
        cv2.imshow(f"Camera {i}", f)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()