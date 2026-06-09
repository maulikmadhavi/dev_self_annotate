# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Gradio UI for open-vocabulary auto-annotation of images, built on Ultralytics.
Upload an image (or a folder of them), type free-text keywords (the open
vocabulary), pick a model and a confidence threshold, and get back annotated
images plus a training-ready label set.

**Agenda:** build an interactive **MLOps cycle** that begins with
open-vocabulary object detection. Rather than labelling from scratch, the tool
pre-labels images from free-text keywords, lets a human correct the results, and
exports a dataset that can train/fine-tune a model — which then feeds back in to
improve the next round of auto-labelling.

`gradio_app.py` is the Gradio UI — the whole app today; `Readme.md` holds the
project intent and roadmap. A Flask front end (`flask_app.py`) is planned as an
alternative entry point over the same model/annotation functions.

## The annotation cycle

```
keywords → batch auto-label → human review/correct → export YOLO dataset → (train) → repeat
```

The three UI tabs cover the first stages of this loop:

- **Single image** — try a vocabulary on one image.
- **Batch** — auto-label a whole folder into a YOLO-format dataset.
- **Review** — human-in-the-loop correction of those labels.

## Running

```bash
pip install -r requirements.txt
python gradio_app.py  # binds 0.0.0.0:7860 by default; override with APP_HOST/APP_PORT
```

The env is also managed by **pixi** (`pixi.toml`, pinned CUDA wheels for a
reproducible GPU install):

```bash
pixi run python gradio_app.py
```

`gradio_app.py` launches on `0.0.0.0` so the UI is reachable when deployed remotely.
Model weights download on first use into the working directory (gitignored).
Keep the pinned wheels in `pixi.toml` as-is — a plain `pip install torch` can
pull a CUDA build that won't run on newer GPUs.

## Code notes

The two models take their text vocabulary through different APIs — see
`_set_vocabulary()` in `gradio_app.py`:
- **yolov8x-worldv2** (YOLO-World, detection): `model.set_classes(names)`
- **yoloe-26x-seg** (YOLOE, segmentation): needs prompt embeddings,
  `model.set_classes(names, model.get_text_pe(names))`. First use also triggers
  an automatic install of the `clip` package.

Models are cached in `_model_cache` and inference runs on `DEVICE`
(`cuda` if available, else `cpu`). **Device pinning matters:** `set_classes`
runs the CLIP text encoder and tokenizes on CPU, but the first `predict()` moves
the cached model (CLIP encoder included) to GPU — so a *rerun* would feed CPU
tokens to a GPU encoder and crash with `index_select ... cuda:0 and cpu`.
`_set_vocabulary()` therefore pins the model to CPU before `set_classes`, and
`annotate()` passes `device=DEVICE` to `predict()` to move it back. Don't remove
the `model.to("cpu")` — it's what keeps reruns working.

`annotate()` is the single-image inference entry point used by the UI; it returns
`(annotated_image, detection_rows, status_text)` and surfaces errors as the
status string rather than raising, so failures show in the UI.

The UI has three tabs (`build_ui()`): **Single image** (`annotate`), **Batch**
(`batch_annotate`) for bootstrapping a label set over many images, and **Review**
for human-in-the-loop correction of that set.

`batch_annotate()` is a generator that streams progress to the UI. It takes a
server-side directory (the practical path for large sets) and/or uploaded files,
sets the vocabulary **once**, then predicts over every image — keeping the model
warm and avoiding the per-call `set_classes` device dance. It writes a
YOLO-format dataset under the output dir (default `annotations_out/`, gitignored):
`labels/*.txt` (normalized bbox for YOLO-World, polygons for YOLOE seg — the
class id matches the vocabulary order), `classes.txt`, `data.yaml`,
`predictions.csv`, optional `annotated/` previews, and a downloadable
`labels_bundle.zip`. Image types scanned are in `IMAGE_EXTS`.

The **Review** tab (`review_load`/`review_nav`/`review_save`) is the
human-in-the-loop step. It points at the images folder + a batch `output_dir`,
loads each image's YOLO label as editable boxes via the `gradio_image_annotation`
`image_annotator` component, and writes corrections back to the `.txt`.
`_yolo_to_boxes`/`_boxes_to_yolo` convert between normalized YOLO and
absolute-pixel boxes; the annotator only does **boxes**, so saving a reviewed
segmentation set collapses polygons to their bounding box. Per-class colors come
from `PALETTE`.
