import cv2
import math
from ultralytics import YOLO

# 1. Load your custom model
model = YOLO('best_v2.pt')

image_path = 'IMG_4108.JPG' 
frame = cv2.imread(image_path)

if frame is None:
    print(f"\n❌ ERROR: Could not find '{image_path}'!")
else:
    results = model(frame, conf=0.6)
    annotated_frame = results[0].plot()

    boxes = results[0].boxes.xyxy.cpu().numpy() 
    centers = []

    for box in boxes:
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        centers.append((cx, cy))
        cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

    # 4. Calculate distance if at least 2 tags are found
    if len(centers) >= 2:
        pt1 = centers[0]
        pt2 = centers[1]
        
        # Calculate pixel distance (the green line)
        distance_px = math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        cv2.line(annotated_frame, pt1, pt2, (0, 255, 0), 2)
        
        # ==========================================
        # REAL-WORLD CONVERSION MATH
        # ==========================================
        # Grab the bounding box of the FIRST tag detected (assuming it's your 25mm tag)
        ref_box = boxes[0]
        ref_width_px = ref_box[2] - ref_box[0] # x2 - x1
        
        # We know this tag is 25mm in real life
        REAL_WIDTH_MM = 25.0 
        
        # Calculate how many pixels equal 1 millimeter
        pixels_per_mm = ref_width_px / REAL_WIDTH_MM
        
        # Convert the distance of the green line into millimeters!
        distance_mm = distance_px / pixels_per_mm
        # ==========================================
        
        # Write the distance text (Moved it to the left slightly so it's easier to read)
        text_x = int((pt1[0] + pt2[0]) / 2) + 20 
        text_y = int((pt1[1] + pt2[1]) / 2) 
        
        # Now it prints "mm" instead of "px"!
        cv2.putText(annotated_frame, f"Dist: {distance_mm:.1f} mm", (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
    cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO", 800, 600)
    cv2.imshow("YOLO", annotated_frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()