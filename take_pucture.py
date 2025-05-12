from ultralytics import YOLO

print("Model Loading")
model = YOLO("models/1lariV4_n.pt")  # Replace path if needed
print("Model Loaded")

image_path = "imgs/coins4.jpg"

for run in range(2):
    print(f"\n--- Run {run + 1} ---")
    results = model(image_path)
    print("IMG processed")

    # Save the result image with unique name
    out_filename = f"predicted4_run{run + 1}.jpg"
    results[0].save(filename=out_filename)
    print(f"Saved: {out_filename}")

    # Print prediction details
    print("\nPrediction Results:")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        label = results[0].names[cls_id]
        print(f"Class: {label}, Confidence: {conf:.2f}, Box: {xyxy}")
