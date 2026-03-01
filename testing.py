from ultralytics import YOLO
import cv2

# 'yolov8n.pt' is the Nano model—the fastest one for CPU usage
model = YOLO('yolov8n.pt') 

# Initialize the laptop webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # imgsz=320 makes the detection much faster on a CPU
        # stream=True is more memory efficient for video
        results = model(frame, imgsz=320, conf=0.4, stream=True)

        for r in results:
            annotated_frame = r.plot()
            cv2.imshow("YOLOv8 CPU Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()