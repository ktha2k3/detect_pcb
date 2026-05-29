from huggingface_hub import hf_hub_download
from pathlib import Path
import os
import zipfile

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
if ENV_PATH.exists():
    with ENV_PATH.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

# Thư mục riêng trong project
project_dir = Path(__file__).resolve().parent
download_dir = project_dir / "downloads"
extract_dir = project_dir / "pcb_dataset"

download_dir.mkdir(parents=True, exist_ok=True)
extract_dir.mkdir(parents=True, exist_ok=True)

zip_path = hf_hub_download(
    repo_id="keremberke/pcb-defect-segmentation",
    filename="data/test.zip",
    repo_type="dataset",
    local_dir=download_dir
)

print("DATA ZIP:", zip_path)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_dir)

print("EXTRACTED TO:", extract_dir)