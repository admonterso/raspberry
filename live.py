from ultralytics import YOLO
import cv2

print("Model Loading")
model = YOLO("models/1lariV4_n.pt")
print("Model Loaded")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting live prediction. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLO prediction on the frame
    results = model(frame)

    # Draw boxes with confidence > 0.8
    for box in results[0].boxes:
        conf = box.conf.item()
        if conf > 0.8:
            cls_id = int(box.cls)
            class_name = results[0].names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("YOLOv8 Live", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
