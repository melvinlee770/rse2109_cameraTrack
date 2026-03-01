import cv2
import math

# 1. Setup the Webcam (0 is usually your built-in laptop camera)
cap = cv2.VideoCapture(0)

# 2. Setup OpenCV's built-in ArUco Detector using the correct 4x4 Dictionary!
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

print("🎥 Live ArUco Tracking Started! Press 'q' on your keyboard to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. Detect the markers in the current video frame
    try:
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(frame)
    except AttributeError:
        # Fallback for older OpenCV versions
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

    # 4. If it finds at least 2 tags, calculate the geometry!
    if ids is not None and len(corners) >= 2:
        
        # Draw the exact outlines of the tags
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        centers = []
        
        for corner_set in corners:
            # Get the 4 corners: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
            pts = corner_set[0]
            tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]
            
            # Calculate exact center by averaging all 4 corners
            cx = int((tl[0] + tr[0] + br[0] + bl[0]) / 4)
            cy = int((tl[1] + tr[1] + br[1] + bl[1]) / 4)
            centers.append((cx, cy))
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # --------------------------------------------------
        # REAL-WORLD CONVERSION MATH
        # --------------------------------------------------
        # Grab the top-left and top-right corners of the FIRST tag detected
        ref_tl = corners[0][0][0]
        ref_tr = corners[0][0][1]
        
        # Calculate the EXACT pixel width of the tag's top edge
        ref_width_px = math.sqrt((ref_tr[0] - ref_tl[0])**2 + (ref_tr[1] - ref_tl[1])**2)
        
        # Prevent math errors if the tag glitches and shows 0 width
        if ref_width_px > 0:
            # Assuming the first tag it looks at is your 25mm tag
            REAL_WIDTH_MM = 25.0 
            pixels_per_mm = ref_width_px / REAL_WIDTH_MM

            # Calculate distance between the first two tags
            pt1 = centers[0]
            pt2 = centers[1]
            
            distance_px = math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
            distance_mm = distance_px / pixels_per_mm
            
            # Draw the line and text
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            
            text_x = int((pt1[0] + pt2[0]) / 2) + 20
            text_y = int((pt1[1] + pt2[1]) / 2) 
            cv2.putText(frame, f"Dist: {distance_mm:.1f} mm", (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        # --------------------------------------------------

    # 5. Show the live video feed
    cv2.imshow("Live OpenCV ArUco Geometry", frame)

    # Press 'q' to quit the video loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up when done
cap.release()
cv2.destroyAllWindows()