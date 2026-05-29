Mình tìm được **2 model thay thế có weight `.pt` sẵn** để bạn thử trong repo `detect_pcb`.

## 1. `Sammy970/PCB-Defect-detection-using-YOLOv8`

Repo này ghi rõ là **YOLOv8 custom model trained on PCB-Defect-Detection data**. File training config cho thấy model gốc là `yolov8n.pt`, task là `detect`, train 100 epochs trên dataset `PCB-Dataset-Defect-1`.  

Weight có sẵn tại:

```text
train/weights/best.pt
```

Mình kiểm tra được file `best.pt` tồn tại trong repo. 

Dùng để download:

```bash
mkdir -p scripts/download_deps/models/sammy_yolov8n_pcb

wget -O scripts/download_deps/models/sammy_yolov8n_pcb/best.pt \
https://raw.githubusercontent.com/Sammy970/PCB-Defect-detection-using-YOLOv8/main/train/weights/best.pt
```

Tên registry nên đặt:

```python
"sammy_yolov8n_pcb_detect": "scripts/download_deps/models/sammy_yolov8n_pcb/best.pt"
```

---

## 2. `KADIRI-SANDEYYA/PCB-DEFECT-DETECTION`

Repo này là hệ thống **PCB Defect Detection using YOLOv8**, có Streamlit app và load trực tiếp `YOLO("best.pt")`. README ghi model được train bằng YOLOv8 để detect lỗi PCB.  

Repo cũng công bố metric: Precision `96.0%`, Recall `93.9%`, mAP@50 `0.97`. 

App load model như sau:

```python
model = YOLO("best.pt")
```



Weight `best.pt` cũng có sẵn ở root repo. 

Download:

```bash
mkdir -p scripts/download_deps/models/kadiri_yolov8_pcb

wget -O scripts/download_deps/models/kadiri_yolov8_pcb/best.pt \
https://raw.githubusercontent.com/KADIRI-SANDEYYA/PCB-DEFECT-DETECTION/main/best.pt
```

Tên registry nên đặt:

```python
"kadiri_yolov8_pcb_detect": "scripts/download_deps/models/kadiri_yolov8_pcb/best.pt"
```

---

## Lưu ý quan trọng

Model hiện tại trong repo của bạn là **segmentation model** từ Hugging Face `keremberke/yolov8n-pcb-defect-segmentation`, còn 2 model trên là **detection model**, tức là trả về bounding box, không trả về mask segmentation.

Vì vậy khi so sánh:

```text
current model: segmentation + box
sammy model: detection box
kadiri model: detection box
```

Nên so bằng các metric detection trước: `precision`, `recall`, `mAP50`, `mAP50-95`, `inference_ms`.

## Gợi ý thêm vào registry

```python
MODEL_REGISTRY = {
    "current_pcb_yolov8n_seg": "scripts/download_deps/models/best.pt",

    "sammy_yolov8n_pcb_detect": "scripts/download_deps/models/sammy_yolov8n_pcb/best.pt",

    "kadiri_yolov8_pcb_detect": "scripts/download_deps/models/kadiri_yolov8_pcb/best.pt",
}
```

Chạy test:

```bash
python scripts/inference/inference.py \
  --model scripts/download_deps/models/sammy_yolov8n_pcb/best.pt \
  --input-dir scripts/download_deps/pcb_dataset \
  --save

python scripts/inference/inference.py \
  --model scripts/download_deps/models/kadiri_yolov8_pcb/best.pt \
  --input-dir scripts/download_deps/pcb_dataset \
  --save
```

Hai model này là lựa chọn thực dụng nhất để làm “model thay thế” nhanh vì đều dùng YOLOv8/Ultralytics và có `.pt` sẵn.
