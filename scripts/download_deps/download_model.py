from huggingface_hub import hf_hub_download
from pathlib import Path

print("Starting download model")

# Thư mục project hiện tại
project_dir = Path(__file__).resolve().parent

# Thư mục lưu model riêng
model_dir = project_dir / "models"
model_dir.mkdir(parents=True, exist_ok=True)

model_path = hf_hub_download(
    repo_id="keremberke/yolov8n-pcb-defect-segmentation",
    filename="best.pt",
    local_dir=model_dir
)

print("MODEL:", model_path)