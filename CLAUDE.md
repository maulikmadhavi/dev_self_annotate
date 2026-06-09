# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Gradio UI for open-vocabulary auto-annotation of images, built on Ultralytics.
Upload an image, type free-text keywords (the open vocabulary), pick a model and
a confidence threshold, and get back an annotated image plus a table of
detections. Goal: bootstrap image annotation with state-of-the-art
open-vocabulary detection/recognition/segmentation models.

`app.py` is the whole application; `Readme.md` holds the original project intent.

## Environments

- **Raspberry Pi (linux/aarch64, CPU-only)** — authoring only; quick smoke
  tests, not real model testing. (The original `.venv` here is the Pi's ARM
  venv and will not run on Windows.)
- **Windows box (win32, NVIDIA RTX 5070 / Blackwell)** — a real GPU test target,
  managed by **pixi** (`pixi.toml`); see the Windows section below. Verified
  working end-to-end (both models, GPU inference, Gradio HTTP 200).
- **A separate GPU server** is where the app is also deployed/tested. Keep the
  code portable: don't bake in environment-specific assumptions.

## Running

On the GPU server (and any normal CUDA/x86 host):

```bash
pip install -r requirements.txt
python app.py        # binds 0.0.0.0:7860 by default; override with APP_HOST/APP_PORT
```

`app.py` launches on `0.0.0.0` so the UI is reachable remotely. Model weights
download on first use into the working directory (gitignored).

On this Pi, use the project venv (created with `--system-site-packages` so it
reuses the system PyTorch), and mind the gotchas below:

```bash
.venv/bin/python app.py
```

## Pi-only environment gotchas

These apply to the local CPU box only; the GPU server uses the plain
`requirements.txt` install above.

- **`/tmp` is a 1.9G tmpfs.** pip builds there by default and runs out of space.
  Install with `TMPDIR=/home/maulik/self_annotate/.venv/tmp` pointing at disk.
- **Do not let pip upgrade torch/torchvision on the Pi.** The default
  linux-aarch64 wheels for torch ≥2.12 are CUDA builds (`+cu130`) whose compiled
  ops (e.g. `torchvision::nms`) SIGILL ("Illegal instruction") on the Pi CPU. The
  working CPU stack is system **torch 2.11.0+cpu** plus **torchvision
  0.26.0+cpu** from `https://download.pytorch.org/whl/cpu` (ABI must match torch
  exactly — a piwheels torchvision fails to register `torchvision::nms`).

## Windows GPU box (pixi)

The Windows env is managed by **pixi** (`pixi.toml`): Python 3.12, torch
2.7.1+cu128 / torchvision 0.22.1+cu128 (from the cu128 index), gradio,
ultralytics. Run it with:

```powershell
pixi run python app.py
```

The old `.venv` (Pi/ARM) and `.venv-win` (a manual pre-pixi attempt) are
obsolete and gitignored — don't use them.

- **Blackwell needs cu128.** The RTX 5070 is sm_120; a plain `pip install torch`
  can hand you a CUDA build that won't run on it. pixi.toml pins the cu128 wheels
  explicitly via the PyTorch cu128 index — keep it that way.
- **The parent `D:\dev_repos\.git` is a broken WSL symlink** (reparse tag
  `0xa000001d`). Ultralytics walks parent dirs for a `.git` and crashes on
  import with `OSError WinError 1920`. Fix: this project has its own real `.git`
  (`git init`) so the walk stops here. Don't delete the parent symlink.

## Code notes

The two models take their text vocabulary through different APIs — see
`_set_vocabulary()` in `app.py`:
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
`image_annotator` component (a pixi dep), and writes corrections back to the
`.txt`. `_yolo_to_boxes`/`_boxes_to_yolo` convert between normalized YOLO and
absolute-pixel boxes; the annotator only does **boxes**, so saving a reviewed
segmentation set collapses polygons to their bounding box. Per-class colors come
from `PALETTE`.
