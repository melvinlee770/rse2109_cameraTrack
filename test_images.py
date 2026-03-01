import cv2
import math
from ultralytics import YOLO

# 1. Load your custom model
model = YOLO('best_v2.pt')

# 2. Provide the path to your test picture
image_path = 'IMG_4108.JPG' 
frame = cv2.imread(image_path)

# Check if the image loaded successfully
if frame is None:
    print(f"\n❌ ERROR: Could not find '{image_path}'!")
    print("Make sure the picture is in the exact same folder as this script, and the spelling is exactly right.\n")
else:
    # 3. Run inference
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
        
        distance = math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        cv2.line(annotated_frame, pt1, pt2, (0, 255, 0), 2)
        
        text_x = int((pt1[0] + pt2[0]) / 2)
        text_y = int((pt1[1] + pt2[1]) / 2) - 15
        cv2.putText(annotated_frame, f"Dist: {int(distance)} px", (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    # 5. Display the result (Properly indented outside the 'if' block!)
    cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO", 800, 600)
    cv2.imshow("YOLO", annotated_frame)

    # Freeze the screen until you press a key
    cv2.waitKey(0)
    cv2.destroyAllWindows()