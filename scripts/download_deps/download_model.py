from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

from huggingface_hub import hf_hub_download


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EXTRA_MODELS = {
    "sammy_yolov8n_pcb_detect": (
        "https://raw.githubusercontent.com/Sammy970/PCB-Defect-detection-using-YOLOv8/main/train/weights/best.pt",
        MODEL_DIR / "sammy_yolov8n_pcb" / "best.pt",
    ),
    "kadiri_yolov8_pcb_detect": (
        "https://raw.githubusercontent.com/KADIRI-SANDEYYA/PCB-DEFECT-DETECTION/main/best.pt",
        MODEL_DIR / "kadiri_yolov8_pcb" / "best.pt",
    ),
}


def download_default_model() -> Path:
    print("Downloading default model from Hugging Face...")
    model_path = hf_hub_download(
        repo_id="keremberke/yolov8n-pcb-defect-segmentation",
        filename="best.pt",
        local_dir=MODEL_DIR,
    )
    print("Saved:", model_path)
    return Path(model_path)


def download_extra(model_key: str) -> Path:
    if model_key not in EXTRA_MODELS:
        raise ValueError(f"Unsupported model key: {model_key}")
    url, target = EXTRA_MODELS[model_key]
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_key}...")
    urlretrieve(url, str(target))
    print("Saved:", target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PCB models.")
    parser.add_argument("--skip-default", action="store_true", help="Skip default Hugging Face model download.")
    parser.add_argument(
        "--extra-model",
        choices=["all"] + sorted(EXTRA_MODELS.keys()),
        default=None,
        help="Download additional model(s) from GitHub.",
    )
    args = parser.parse_args()

    if not args.skip_default:
        download_default_model()

    if args.extra_model == "all":
        for key in sorted(EXTRA_MODELS.keys()):
            download_extra(key)
    elif args.extra_model:
        download_extra(args.extra_model)


if __name__ == "__main__":
    main()
