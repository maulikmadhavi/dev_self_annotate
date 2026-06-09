# Self-Annotate

Open-vocabulary **auto-annotation** to bootstrap image-labelling. Point it at a
folder of images, type free-text keywords (the open vocabulary), and let
state-of-the-art vision models pre-label everything — then correct the results
in a human-in-the-loop review step and export a training-ready dataset.

The goal is to turn "label 10,000 images from scratch" into "review and fix what
the model already proposed".

Built as a small [Gradio](https://www.gradio.app/) UI over
[Ultralytics](https://docs.ultralytics.com/) open-vocabulary models:

| Model | Task | Vocabulary API |
| --- | --- | --- |
| `yolov8x-worldv2` | open-vocabulary **detection** (boxes) | `set_classes(names)` |
| `yoloe-26x-seg` | open-vocabulary **segmentation** (boxes + masks) | `set_classes(names, get_text_pe(names))` |

Both accept arbitrary text classes, so you can label anything describable in
words without any task-specific training.

## Current capabilities

- **Open-vocabulary detection & segmentation** — free-text keywords as classes,
  no fine-tuning required; switch models from the UI.
- **Three workflows, three tabs:**
  - **Single image** — upload one image, pick model + confidence threshold, get
    an annotated preview and a detection table. Good for trying out a vocabulary.
  - **Batch** — point at a server-side folder (best for large sets) and/or
    upload images; pre-labels the whole set and writes a **YOLO-format dataset**.
    Streams live progress and a per-image summary.
  - **Review** — human-in-the-loop correction. Step through each image with the
    predicted boxes shown as **editable** boxes (drag/resize, double-click to
    relabel, `Del` to remove, draw to add), then save corrections back to the
    label files.
- **Training-ready export** (batch) under the output dir (default
  `annotations_out/`):
  - `labels/*.txt` — normalized YOLO labels (bbox for YOLO-World, polygons for
    YOLOE segmentation); class id follows the keyword order
  - `classes.txt`, `data.yaml` — ready to feed back into Ultralytics training
  - `predictions.csv` — flat, human-readable table of every detection
  - optional `annotated/` preview images
  - `labels_bundle.zip` — a small, downloadable bundle of all of the above
- **GPU-aware** — runs on CUDA when available, CPU otherwise; models are cached
  and kept warm across a batch run.

## Usage

### Install & run

**With pixi (Windows / NVIDIA, recommended):**

```powershell
pixi run python app.py
```

**With pip (Linux / any CUDA host):**

```bash
pip install -r requirements.txt
python app.py
```

The UI binds to `0.0.0.0:7860` so it is reachable when deployed on a remote GPU
server (override with `APP_HOST` / `APP_PORT`). Model weights download on first
use into the working directory.

### Typical bootstrapping loop

1. **Batch** tab → set the image directory, type keywords
   (e.g. `person, forklift, pallet`), pick a model and threshold → **Run batch**.
   This writes a YOLO dataset to `annotations_out/`.
2. **Review** tab → point at the same images folder + `annotations_out/` → step
   through, fix the boxes, **Save** (or **Save + Next**). Corrections are written
   straight back to the YOLO labels.
3. Use `data.yaml` + `labels/` to train or fine-tune a model on the corrected
   set, then feed a better model back into step 1.

## Plans / To-do

Roughly in priority order — contributions and reordering welcome:

- [ ] **COCO JSON export** alongside the YOLO format (and Pascal VOC XML).
- [ ] **Polygon-level review** — the current Review tab edits bounding boxes
  only, so reviewing a segmentation set collapses masks to boxes. Add an
  editable mask/polygon component to correct segmentation properly.
- [ ] **Per-class thresholds** and `iou` / `max_det` controls (e.g.
  `person:0.4, scratch:0.15`) instead of a single global confidence slider.
- [ ] **SAHI-style tiling / slicing** for large, high-resolution images so small
  objects aren't missed at native scale.
- [ ] **Active-learning triage** — surface low-confidence / zero-detection
  images first so review time is spent where the model is unsure.
- [ ] **Configurable preprocessing** — image resize / letterbox options exposed
  in the UI.
- [ ] **Session persistence** — track which images have been reviewed; resume,
  re-export, and report progress.
- [ ] **Class synonym merging** — map several keywords (e.g. `car, automobile`)
  onto one output class.
- [ ] **Video / frame sampling** input for labelling extracted frames.
- [ ] **Concurrency safety** — guard the shared model cache so multiple
  simultaneous users don't race on device placement.

## Environments

Developed across a Raspberry Pi (authoring, CPU), a Windows RTX 5070 box
(Blackwell GPU, managed by pixi with CUDA 12.8 wheels), and a separate GPU
server for deployment. See [CLAUDE.md](CLAUDE.md) for environment-specific
setup notes and gotchas.
