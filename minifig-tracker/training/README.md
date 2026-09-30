# Training the minifig YOLO model

This is the "no cheating" pipeline: Roboflow is only used to store and export the
annotated dataset. The actual model training happens locally in `train.py` via
`ultralytics`, not Roboflow's hosted AutoML.

## 1. Capture images

```
cd training
pip install -r requirements.txt
python capture_images.py
```

Press **space** to save a frame, **q** to quit. Aim for **150-300 images** of the
green minifig: vary distance, angle, background, and lighting so the model
generalizes. Images are saved to `training/raw_images/`.

## 2. Annotate in Roboflow (manual, in your browser)

1. Create a Roboflow account/project (Object Detection).
2. Upload everything in `training/raw_images/`.
3. Draw a bounding box around the minifig in each image, labeled `minifig`
   (single class).
4. Generate a dataset version (default train/val/test split is fine to start).
5. Export it — pick the **YOLOv8** format. Copy your **API key** and the
   **workspace / project / version number** shown in the export code snippet.

## 3. Download the annotated dataset

Create `training/.env` (git-ignored, never commit this):

```
ROBOFLOW_API_KEY=your_key_here
ROBOFLOW_WORKSPACE=your_workspace
ROBOFLOW_PROJECT=your_project
ROBOFLOW_VERSION=1
```

Then:

```
python download_dataset.py
```

This pulls the exported, annotated dataset into `training/dataset/`.

## 4. Train locally

```
python train.py
```

This runs a real local training job (`ultralytics` `YOLO('yolov8n.pt').train(...)`)
against `training/dataset/data.yaml`. Training on CPU works but is slow — if you
have an NVIDIA GPU it'll be used automatically; otherwise consider a free Colab
GPU runtime and copying `best.pt` back down.

Best weights land at `training/runs/detect/train/weights/best.pt` — copy that file
to `laptop/models/minifig_yolo.pt` for `part2_yolo_track.py` to use.
