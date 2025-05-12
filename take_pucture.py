from ultralytics import YOLO
import cv2
print("Model Loading")
# Load a pre-trained model (e.g., yolov8n, yolov8s, yolov8m, etc.)
model = YOLO("models/1lariV4_n.pt")  # 'n' is nano version for speed
print("Model Loaded")
# Load an image

print("input IMG")
image_path = "imgs/coins4.jpg"
results = model(image_path)
print("IMG sent")
# Save results


results[0].save(filename="predicted4.jpg")

# Optional: Get prediction details
# for result in results:
#     for box in result.boxes:
#         print(f"Class: {result.names[int(box.cls)]}, Confidence: {box.conf.item()}, Box: {box.xyxy[0].tolist()}")
