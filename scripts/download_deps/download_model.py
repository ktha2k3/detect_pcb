from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SOURCES = {
    "current_pcb_yolov8n_seg": "keremberke/yolov8n-pcb-defect-segmentation",
    "pcb_yolov8s_seg": "keremberke/yolov8s-pcb-defect-segmentation",
    "pcb_yolov8m_seg": "keremberke/yolov8m-pcb-defect-segmentation",
}


def download_model(model_key: str) -> Path:
    if model_key not in MODEL_SOURCES:
        raise ValueError(f"Unsupported model key: {model_key}")
    repo_id = MODEL_SOURCES[model_key]
    print(f"Downloading {model_key} from {repo_id}...")
    downloaded_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="best.pt",
            local_dir=MODEL_DIR,
        )
    )
    target = MODEL_DIR / f"{model_key}.pt"
    target.write_bytes(downloaded_path.read_bytes())
    print("Saved:", target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PCB models.")
    parser.add_argument(
        "--model",
        choices=["all"] + sorted(MODEL_SOURCES.keys()),
        default="all",
        help="Download one model key or all models.",
    )
    args = parser.parse_args()

    if args.model == "all":
        for key in sorted(MODEL_SOURCES.keys()):
            download_model(key)
        return
    download_model(args.model)


if __name__ == "__main__":
    main()
