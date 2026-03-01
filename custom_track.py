import cv2
from ultralytics import YOLO

# Load your custom model
model = YOLO('best_v2.pt')

# Open webcam (0 is usually the integrated laptop camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Run YOLOv11 inference
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the result
        cv2.imshow("YOLO ArUco Test", annotated_frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()