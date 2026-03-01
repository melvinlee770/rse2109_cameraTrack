import cv2
import math

image_path = 'IMG_4108.JPG' 
frame = cv2.imread(image_path)

if frame is None:
    print(f"\n❌ ERROR: Could not find '{image_path}'!")
else:
    # 1. Setup OpenCV's built-in ArUco Detector
    # (Looking at your photo, they appear to be 5x5 or 4x4 grids. We will try 5x5 first!)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    # This try/except handles both older and newer versions of OpenCV automatically
    try:
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(frame)
    except AttributeError:
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

    # 2. If it found tags, let's do the exact math
    if ids is not None and len(corners) >= 2:
        
        # Draw the exact outlines of the tags (Notice how they hug the black edges perfectly!)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        centers = []
        
        for corner_set in corners:
            # corner_set[0] contains the 4 corners: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
            pts = corner_set[0]
            tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]
            
            # Calculate exact center by averaging all 4 corners
            cx = int((tl[0] + tr[0] + br[0] + bl[0]) / 4)
            cy = int((tl[1] + tr[1] + br[1] + bl[1]) / 4)
            centers.append((cx, cy))
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # 3. REAL-WORLD CONVERSION MATH (Using exact corners)
        # Grab the top-left and top-right corners of the FIRST tag detected
        ref_tl = corners[0][0][0]
        ref_tr = corners[0][0][1]
        
        # Calculate the EXACT pixel width of the tag's edge (Rotation immune!)
        ref_width_px = math.sqrt((ref_tr[0] - ref_tl[0])**2 + (ref_tr[1] - ref_tl[1])**2)
        
        # Assuming the first tag it looks at is your 25mm tag
        REAL_WIDTH_MM = 25.0 
        pixels_per_mm = ref_width_px / REAL_WIDTH_MM

        # 4. Calculate distance between the first two tags
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
                    
    else:
        print("\n⚠️ Not enough ArUco tags detected. You might need to change the DICTIONARY on Line 13!")

    # Display the result
    cv2.namedWindow("OpenCV Exact Geometry", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("OpenCV Exact Geometry", 800, 600)
    cv2.imshow("OpenCV Exact Geometry", frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()