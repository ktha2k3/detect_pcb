from __future__ import annotations

import argparse
import glob
from functools import lru_cache
from pathlib import Path

from ultralytics import YOLO


current_file = Path(__file__).resolve()
scripts_dir = current_file.parents[1]
extract_dir = scripts_dir / "download_deps" / "pcb_dataset"
model_path = scripts_dir / "download_deps" / "models" / "best.pt"
output_dir = scripts_dir / "runs_pcb"

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
MODEL_REGISTRY = {
    "current_pcb_yolov8n_seg": scripts_dir / "download_deps" / "models" / "best.pt",
    "sammy_yolov8n_pcb_detect": scripts_dir / "download_deps" / "models" / "sammy_yolov8n_pcb" / "best.pt",
    "kadiri_yolov8_pcb_detect": scripts_dir / "download_deps" / "models" / "kadiri_yolov8_pcb" / "best.pt",
}


def find_image_paths(root_dir: Path) -> list[Path]:
    image_paths: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(Path(path) for path in glob.glob(str(root_dir / "**" / ext), recursive=True))
    return sorted(image_paths)


def resolve_model_path(model_key_or_path: str | None = None) -> Path:
    if not model_key_or_path:
        return model_path
    if model_key_or_path in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_key_or_path]
    return Path(model_key_or_path).expanduser().resolve()


@lru_cache(maxsize=8)
def load_model(model_key_or_path: str | None = None) -> YOLO:
    selected_model_path = resolve_model_path(model_key_or_path)
    if not selected_model_path.exists():
        raise FileNotFoundError(f"Could not find model: {selected_model_path}")
    return YOLO(str(selected_model_path))


def predict_one(
    source,
    *,
    model: YOLO | None = None,
    imgsz: int = 640,
    conf: float = 0.05,
    iou: float = 0.45,
    save: bool = False,
    project: Path = output_dir,
    name: str = "predict",
):
    model = model or load_model()
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        save=save,
        project=str(project),
        name=name,
        exist_ok=True,
        verbose=False,
    )

    result = results[0]
    detections = []

    for box in result.boxes:
        cls_id = int(box.cls.item() if hasattr(box.cls, "item") else box.cls)
        confidence = float(box.conf.item() if hasattr(box.conf, "item") else box.conf)
        xyxy = [float(value) for value in box.xyxy[0].tolist()]
        detections.append(
            {
                "class_id": cls_id,
                "class_name": model.names[cls_id],
                "confidence": confidence,
                "xyxy": xyxy,
            }
        )

    return result, detections


def predict_many(
    sources,
    *,
    model: YOLO | None = None,
    imgsz: int = 640,
    conf: float = 0.05,
    iou: float = 0.45,
    save: bool = False,
    project: Path = output_dir,
    name: str = "predict",
):
    model = model or load_model()
    return [
        predict_one(
            source,
            model=model,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            save=save,
            project=project,
            name=name,
        )
        for source in sources
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PCB inference on test images.")
    parser.add_argument(
        "--model",
        default="current_pcb_yolov8n_seg",
        help="Model key from registry or direct path to .pt file",
    )
    parser.add_argument("--input-dir", type=Path, default=extract_dir, help="Directory containing test images")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of images to process")
    parser.add_argument("--save", action="store_true", help="Save annotated outputs to runs_pcb/")
    parser.add_argument("--project", type=Path, default=output_dir, help="Output project directory")
    parser.add_argument("--name", default="predict", help="Run name under the output project")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    args = parser.parse_args()

    print("CURRENT FILE:", current_file)
    print("SCRIPTS DIR:", scripts_dir)
    print("DATA DIR:", args.input_dir)
    selected_model_path = resolve_model_path(args.model)
    print("MODEL:", args.model)
    print("MODEL PATH:", selected_model_path)
    print("MODEL EXISTS:", selected_model_path.exists())
    print("DATA EXISTS:", args.input_dir.exists())

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Could not find dataset: {args.input_dir}")

    image_paths = find_image_paths(args.input_dir)
    print("TOTAL IMAGES:", len(image_paths))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"Could not find images in folder: {args.input_dir}")

    model = load_model(args.model)
    for image_path in image_paths[: args.limit]:
        print("\nIMAGE:", image_path)
        result, detections = predict_one(
            str(image_path),
            model=model,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            save=args.save,
            project=args.project,
            name=args.name,
        )

        print("boxes:", len(result.boxes))
        for detection in detections:
            print(detection["class_name"], detection["confidence"], detection["xyxy"])

    print("\nDONE")
    print("Prediction output folder:", args.project / args.name)


if __name__ == "__main__":
    main()
