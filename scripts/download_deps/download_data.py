from huggingface_hub import hf_hub_download
from pathlib import Path
import zipfile

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