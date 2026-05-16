import glob
from pathlib import Path
from ultralytics import YOLO

# File hiện tại: PRJ/scripts/inference/inference.py
current_file = Path(__file__).resolve()

# Thư mục scripts/
scripts_dir = current_file.parents[1]

# Data
extract_dir = scripts_dir / "download_deps" / "pcb_dataset"

# Model
model_path = scripts_dir / "download_deps" / "models" / "best.pt"

# Output predict
output_dir = scripts_dir / "runs_pcb"

print("CURRENT FILE:", current_file)
print("SCRIPTS DIR:", scripts_dir)
print("DATA DIR:", extract_dir)
print("MODEL PATH:", model_path)
print("MODEL EXISTS:", model_path.exists())
print("DATA EXISTS:", extract_dir.exists())

if not model_path.exists():
    raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

if not extract_dir.exists():
    raise FileNotFoundError(f"Không tìm thấy dataset: {extract_dir}")

# Lấy ảnh test
image_paths = []
for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
    image_paths.extend(glob.glob(str(extract_dir / "**" / ext), recursive=True))

image_paths = sorted(image_paths)

print("TOTAL IMAGES:", len(image_paths))

if len(image_paths) == 0:
    raise FileNotFoundError(f"Không tìm thấy ảnh trong thư mục: {extract_dir}")

# Load model
model = YOLO(str(model_path))

# Predict 10 ảnh đầu
for image_path in image_paths[:10]:
    print("\nIMAGE:", image_path)

    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=0.05,
        iou=0.45,
        save=True,
        project=str(output_dir),
        name="predict",
        exist_ok=True
    )

    result = results[0]

    print("boxes:", len(result.boxes))

    for box in result.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        xyxy = box.xyxy[0].tolist()

        print(model.names[cls_id], conf, xyxy)

print("\nDONE")
print("Kết quả predict được lưu tại:", output_dir / "predict")