import cv2
import cv2.aruco as aruco
import numpy as np

# --- Configuration ---
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# If you know the physical width/height of the zone (in meters), enter them here:
REAL_WIDTH = 5.0  
REAL_HEIGHT = 3.0 

def order_points(pts):
    """Sorts coordinates: [top-left, top-right, bottom-right, bottom-left]"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)] # Bottom-right has largest sum
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-right has smallest difference
    rect[3] = pts[np.argmax(diff)] # Bottom-left has largest difference
    return rect

cap = cv2.VideoCapture(0)

# To keep the area stable, we store the last valid coordinates
last_valid_pts = None

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None and len(ids) >= 4:
        # 1. Get centers of all detected markers
        centers = np.array([np.mean(c[0], axis=0) for c in corners])
        
        # 2. Sort them to define the polygon correctly
        ordered_pts = order_points(centers)
        last_valid_pts = ordered_pts # Update "memory"
    
    if last_valid_pts is not None:
        # 3. Draw the Staging Area Polygon
        pts = last_valid_pts.astype(int).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # 4. Calculate Area (Pixels)
        pixel_area = cv2.contourArea(last_valid_pts)
        
        # 5. Calculate Real World Area (Simplified approximation)
        # We calculate the average pixel distance of the sides to get a scale
        dist_top = np.linalg.norm(last_valid_pts[0] - last_valid_pts[1])
        dist_side = np.linalg.norm(last_valid_pts[0] - last_valid_pts[3])
        
        ppm_x = dist_top / REAL_WIDTH   # Pixels Per Meter (X)
        ppm_y = dist_side / REAL_HEIGHT # Pixels Per Meter (Y)
        
        real_area = pixel_area / (ppm_x * ppm_y)

        # 6. Display Data
        cv2.putText(frame, f"Area: {real_area:.2f} m2", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Pixels: {int(pixel_area)} px", (50, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow('Enhanced Staging Monitor', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()