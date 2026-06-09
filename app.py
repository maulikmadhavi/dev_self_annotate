"""Self-annotate: open-vocabulary auto-annotation UI.

A minimal Gradio front end over Ultralytics open-vocabulary models. Give it an
image plus free-text keywords (the open vocabulary), pick a model and a
confidence threshold, and it returns an annotated image plus a table of
detections. Both supported models accept arbitrary text classes:

  - yolov8x-worldv2  : YOLO-World open-vocabulary *detection* (boxes)
  - yoloe-26x-seg    : YOLOE open-vocabulary *segmentation* (boxes + masks)

Weights are downloaded on first use by Ultralytics into the working directory.
"""

import csv
import os
import zipfile
from pathlib import Path

import gradio as gr
import torch
from gradio_image_annotation import image_annotator
from PIL import Image
from ultralytics import YOLO

# Run on GPU when available; falls back to CPU on the Pi/authoring box.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Image file types the batch scanner picks up (matched case-insensitively).
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Per-class box colors for the Review tab, cycled by class index.
PALETTE = [
    (255, 56, 56), (56, 168, 255), (56, 255, 120), (255, 178, 29),
    (190, 56, 255), (255, 56, 196), (56, 255, 235), (148, 120, 84),
]

# Label shown in the UI -> weights file Ultralytics will load/download.
MODELS = {
    "yolov8x-worldv2 (detection)": "yolov8x-worldv2.pt",
    "yoloe-26x-seg (segmentation)": "yoloe-26x-seg.pt",
}

# Cache loaded models so switching back and forth doesn't reload weights.
_model_cache: dict[str, YOLO] = {}


def _get_model(weights: str) -> YOLO:
    if weights not in _model_cache:
        _model_cache[weights] = YOLO(weights)
    return _model_cache[weights]


def _parse_keywords(text: str) -> list[str]:
    """Split the keyword box on commas and newlines into a clean class list."""
    raw = text.replace("\n", ",").split(",")
    seen, names = set(), []
    for token in raw:
        kw = token.strip()
        if kw and kw.lower() not in seen:
            seen.add(kw.lower())
            names.append(kw)
    return names


def _set_vocabulary(model: YOLO, weights: str, names: list[str]) -> None:
    """Apply the text vocabulary; the two model families have different APIs.

    set_classes runs the CLIP text encoder, tokenizing on CPU. Models are cached,
    so a prior predict() may have left this one on GPU — then CPU tokens hit a GPU
    encoder and it crashes with a device mismatch. Pin the model to CPU first;
    predict() moves it to DEVICE afterward (the same path a fresh model takes).
    """
    model.to("cpu")
    if "yoloe" in weights:
        # YOLOE needs text prompt embeddings.
        model.set_classes(names, model.get_text_pe(names))
    else:
        # YOLO-World takes the class names directly.
        model.set_classes(names)


def annotate(image, keywords, threshold, model_label):
    if image is None:
        return None, [], "Upload an image first."

    names = _parse_keywords(keywords or "")
    if not names:
        return None, [], "Enter at least one keyword (comma- or newline-separated)."

    weights = MODELS[model_label]
    try:
        model = _get_model(weights)
        _set_vocabulary(model, weights, names)
        result = model.predict(image, conf=float(threshold), device=DEVICE, verbose=False)[0]
    except Exception as exc:  # surface load/inference errors in the UI
        return None, [], f"Error: {exc}"

    # result.plot() returns a BGR numpy array; flip to RGB for display.
    annotated = Image.fromarray(result.plot()[:, :, ::-1])

    rows = []
    class_names = result.names
    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = (round(v, 1) for v in box.xyxy[0].tolist())
        rows.append([class_names[cls_id], round(conf, 3), x1, y1, x2, y2])
    rows.sort(key=lambda r: r[1], reverse=True)

    summary = f"{len(rows)} detection(s) for vocabulary: {', '.join(names)}"
    return annotated, rows, summary


# --------------------------------------------------------------------------- #
# Batch mode: bootstrap labels for a whole folder of images.
# --------------------------------------------------------------------------- #

def _collect_paths(input_dir: str, uploaded, recursive: bool) -> list[str]:
    """Gather image paths from uploaded files and/or a server-side directory."""
    paths: list[str] = []
    for f in uploaded or []:
        paths.append(f if isinstance(f, str) else f.name)
    if input_dir and input_dir.strip():
        base = Path(input_dir.strip())
        files = base.rglob("*") if recursive else base.iterdir()
        paths += [str(p) for p in files if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(set(paths))


def _yolo_lines(result) -> tuple[list[str], int]:
    """YOLO label lines for one result.

    Segmentation models (YOLOE) emit normalized polygons; detection models
    (YOLO-World) emit normalized cx/cy/w/h boxes. The class id Ultralytics
    assigns matches the order of the vocabulary passed to set_classes, which is
    exactly the index we want in the label file.
    """
    lines: list[str] = []
    masks = result.masks
    if masks is not None and len(masks) == len(result.boxes):
        for box, poly in zip(result.boxes, masks.xyn):
            cid = int(box.cls.item())
            coords = " ".join(f"{v:.6f}" for v in poly.reshape(-1))
            lines.append(f"{cid} {coords}")
    else:
        for box in result.boxes:
            cid = int(box.cls.item())
            cx, cy, w, h = box.xywhn[0].tolist()
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines, len(result.boxes)


def _write_dataset_meta(out: Path, names: list[str]) -> None:
    """Write classes.txt and a YOLO data.yaml describing the vocabulary."""
    (out / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    (out / "data.yaml").write_text(
        "# Generated by self_annotate batch mode.\n"
        "# YOLO dataset layout: put your images under an `images/` folder with\n"
        "# this `labels/` folder as its sibling (matching base filenames).\n"
        f"path: {out.resolve().as_posix()}\n"
        "train: images\n"
        "val: images\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )


def _zip_bundle(out: Path, labels_dir: Path) -> Path:
    """Bundle the label set + metadata into a small downloadable zip."""
    zip_path = out / "labels_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for meta in ("classes.txt", "data.yaml", "predictions.csv"):
            if (out / meta).exists():
                zf.write(out / meta, meta)
        for txt in sorted(labels_dir.glob("*.txt")):
            zf.write(txt, f"labels/{txt.name}")
    return zip_path


def batch_annotate(input_dir, uploaded, keywords, threshold, model_label,
                   output_dir, save_previews, recursive):
    """Run a model over many images and write a YOLO-format label set.

    Yields (status_markdown, summary_rows, zip_path) so the UI shows live
    progress; the final yield carries the downloadable bundle.
    """
    names = _parse_keywords(keywords or "")
    if not names:
        yield "Enter at least one keyword (comma- or newline-separated).", [], None
        return

    paths = _collect_paths(input_dir, uploaded, recursive)
    if not paths:
        yield "No images found — check the directory path or upload files.", [], None
        return

    out = Path((output_dir or "").strip() or "annotations_out")
    labels_dir = out / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    prev_dir = out / "annotated"
    if save_previews:
        prev_dir.mkdir(parents=True, exist_ok=True)

    weights = MODELS[model_label]
    model = _get_model(weights)
    try:
        _set_vocabulary(model, weights, names)  # once; model stays warm below
    except Exception as exc:
        yield f"Error setting vocabulary: {exc}", [], None
        return

    total = len(paths)
    summary_rows: list[list] = []
    csv_rows: list[list] = []
    n_dets = 0
    step = max(1, total // 50)  # ~50 progress updates regardless of set size

    for i, p in enumerate(paths, 1):
        name = Path(p).name
        try:
            result = model.predict(p, conf=float(threshold), device=DEVICE, verbose=False)[0]
        except Exception as exc:
            summary_rows.append([name, -1, f"error: {exc}"])
            if i % step == 0 or i == total:
                yield f"Processing {i}/{total}… {n_dets} detections so far.", summary_rows, None
            continue

        lines, n = _yolo_lines(result)
        (labels_dir / f"{Path(p).stem}.txt").write_text("\n".join(lines), encoding="utf-8")

        class_names = result.names
        for box in result.boxes:
            cid = int(box.cls.item())
            conf = round(float(box.conf.item()), 3)
            x1, y1, x2, y2 = (round(v, 1) for v in box.xyxy[0].tolist())
            csv_rows.append([name, class_names[cid], conf, x1, y1, x2, y2])

        if save_previews:
            Image.fromarray(result.plot()[:, :, ::-1]).save(prev_dir / f"{Path(p).stem}.jpg")

        n_dets += n
        summary_rows.append([name, n, "ok"])
        if i % step == 0 or i == total:
            yield f"Processing {i}/{total}… {n_dets} detections so far.", summary_rows, None

    _write_dataset_meta(out, names)
    with open(out / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "label", "confidence", "x1", "y1", "x2", "y2"])
        writer.writerows(csv_rows)
    zip_path = _zip_bundle(out, labels_dir)

    failed = sum(1 for r in summary_rows if r[1] == -1)
    done = (
        f"Done — {total} image(s), {n_dets} detection(s)"
        + (f", {failed} failed" if failed else "")
        + f". Labels in `{labels_dir.resolve()}`."
    )
    yield done, summary_rows, str(zip_path)


# --------------------------------------------------------------------------- #
# Review mode: human-in-the-loop correction of the bootstrapped labels.
#
# Loads each image with its YOLO labels as editable boxes (drag/resize/relabel/
# delete/add), and writes the corrected boxes back to the label .txt on Save.
# Edits are saved in YOLO *bbox* format — reviewing a segmentation set therefore
# collapses polygons to their bounding box.
# --------------------------------------------------------------------------- #

def _read_classes(out: Path) -> list[str]:
    cls_file = out / "classes.txt"
    if not cls_file.exists():
        return []
    return [c for c in cls_file.read_text(encoding="utf-8").splitlines() if c.strip()]


def _yolo_to_boxes(label_path: Path, names: list[str], w: int, h: int) -> list[dict]:
    """Convert a YOLO label file to absolute-pixel boxes for the annotator.

    Handles both bbox (cx cy w h) and polygon (x1 y1 x2 y2 …) lines; polygons
    are shown as their enclosing box.
    """
    boxes: list[dict] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        coords = [float(v) for v in parts[1:]]
        if len(coords) == 4:
            cx, cy, bw, bh = coords
            xmin, xmax = (cx - bw / 2) * w, (cx + bw / 2) * w
            ymin, ymax = (cy - bh / 2) * h, (cy + bh / 2) * h
        else:
            xs, ys = coords[0::2], coords[1::2]
            xmin, xmax = min(xs) * w, max(xs) * w
            ymin, ymax = min(ys) * h, max(ys) * h
        boxes.append({
            "xmin": int(max(0, xmin)), "ymin": int(max(0, ymin)),
            "xmax": int(min(w, xmax)), "ymax": int(min(h, ymax)),
            "label": names[cid] if 0 <= cid < len(names) else str(cid),
            "color": PALETTE[cid % len(PALETTE)],
        })
    return boxes


def _boxes_to_yolo(boxes: list[dict], names: list[str], w: int, h: int) -> list[str]:
    lines: list[str] = []
    for b in boxes or []:
        cid = names.index(b["label"]) if b.get("label") in names else 0
        xmin, xmax = sorted((b["xmin"], b["xmax"]))
        ymin, ymax = sorted((b["ymin"], b["ymax"]))
        cx, cy = (xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h
        bw, bh = (xmax - xmin) / w, (ymax - ymin) / h
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def _review_value(state: dict) -> dict:
    """Annotator payload (image + boxes) for the current index in state."""
    path = Path(state["paths"][state["idx"]])
    img = Image.open(path).convert("RGB")
    label_path = Path(state["output_dir"]) / "labels" / f"{path.stem}.txt"
    return {"image": img, "boxes": _yolo_to_boxes(label_path, state["names"], *img.size)}


def _review_status(state: dict, note: str = "") -> str:
    path = Path(state["paths"][state["idx"]])
    head = f"**{state['idx'] + 1} / {len(state['paths'])}** — `{path.name}`"
    return f"{head} · {note}" if note else head


def review_load(images_dir, output_dir):
    """Start a review session over images_dir against labels in output_dir."""
    out = Path((output_dir or "").strip() or "annotations_out")
    names = _read_classes(out)
    if not names:
        return None, None, f"No `classes.txt` in `{out}` — run a batch first."
    paths = _collect_paths(images_dir, None, recursive=False)
    if not paths:
        return None, None, "No images found in that directory."
    state = {"paths": paths, "idx": 0, "names": names, "output_dir": str(out)}
    return _review_value(state), state, _review_status(state)


def review_nav(state, delta):
    if not state:
        return None, state, "Load a directory first."
    state["idx"] = max(0, min(state["idx"] + delta, len(state["paths"]) - 1))
    return _review_value(state), state, _review_status(state)


def review_save(annotation, state, go_next: bool):
    """Write the edited boxes back to the current image's YOLO label file."""
    if not state:
        return None, state, "Load a directory first."
    path = Path(state["paths"][state["idx"]])
    w, h = Image.open(path).size
    boxes = (annotation or {}).get("boxes", [])
    label_path = Path(state["output_dir"]) / "labels" / f"{path.stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(_boxes_to_yolo(boxes, state["names"], w, h)), encoding="utf-8")
    note = f"saved {len(boxes)} box(es) → {label_path.name}"
    if go_next and state["idx"] < len(state["paths"]) - 1:
        state["idx"] += 1
        return _review_value(state), state, _review_status(state, note)
    return _review_value(state), state, _review_status(state, note)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Self-Annotate") as demo:
        gr.Markdown("# Self-Annotate\nOpen-vocabulary auto-annotation.")
        model_choices = list(MODELS.keys())

        with gr.Tab("Single image"):
            with gr.Row():
                with gr.Column():
                    image_in = gr.Image(type="pil", label="Image")
                    keywords_in = gr.Textbox(
                        label="Keywords (open vocabulary)",
                        placeholder="person, dog, traffic light",
                        lines=2,
                    )
                    model_in = gr.Dropdown(
                        choices=model_choices, value=model_choices[0], label="Model",
                    )
                    threshold_in = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.25, step=0.01,
                        label="Confidence threshold",
                    )
                    run_btn = gr.Button("Detect", variant="primary")
                with gr.Column():
                    image_out = gr.Image(type="pil", label="Detected items")
                    status_out = gr.Markdown()
                    table_out = gr.Dataframe(
                        headers=["label", "confidence", "x1", "y1", "x2", "y2"],
                        label="Detections",
                        wrap=True,
                    )
            run_btn.click(
                annotate,
                inputs=[image_in, keywords_in, threshold_in, model_in],
                outputs=[image_out, table_out, status_out],
            )

        with gr.Tab("Batch"):
            gr.Markdown(
                "Bootstrap YOLO-format labels for a whole folder. Point it at a "
                "directory on the server (best for large sets) or upload images. "
                "Writes `labels/`, `classes.txt`, `data.yaml` and `predictions.csv`."
            )
            with gr.Row():
                with gr.Column():
                    batch_dir_in = gr.Textbox(
                        label="Image directory (on the server)",
                        placeholder="/data/images_to_label",
                    )
                    batch_files_in = gr.File(
                        label="…or upload images", file_count="multiple",
                        file_types=["image"],
                    )
                    batch_keywords_in = gr.Textbox(
                        label="Keywords (open vocabulary)",
                        placeholder="person, dog, traffic light",
                        lines=2,
                    )
                    batch_model_in = gr.Dropdown(
                        choices=model_choices, value=model_choices[0], label="Model",
                    )
                    batch_threshold_in = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.25, step=0.01,
                        label="Confidence threshold",
                    )
                    batch_out_in = gr.Textbox(
                        label="Output directory", value="annotations_out",
                    )
                    with gr.Row():
                        batch_recursive_in = gr.Checkbox(
                            label="Scan subfolders", value=False,
                        )
                        batch_preview_in = gr.Checkbox(
                            label="Save annotated previews", value=False,
                        )
                    batch_btn = gr.Button("Run batch", variant="primary")
                with gr.Column():
                    batch_status_out = gr.Markdown()
                    batch_zip_out = gr.File(label="Download labels (.zip)")
                    batch_table_out = gr.Dataframe(
                        headers=["image", "detections", "status"],
                        label="Per-image summary",
                        wrap=True,
                    )
            batch_btn.click(
                batch_annotate,
                inputs=[
                    batch_dir_in, batch_files_in, batch_keywords_in,
                    batch_threshold_in, batch_model_in, batch_out_in,
                    batch_preview_in, batch_recursive_in,
                ],
                outputs=[batch_status_out, batch_table_out, batch_zip_out],
            )

        with gr.Tab("Review"):
            gr.Markdown(
                "Human-in-the-loop correction. Point at the **images** folder and "
                "the **output** folder from a batch run, then step through and fix "
                "boxes: drag/resize, double-click to relabel, `Del` to remove, draw "
                "to add. **Save** writes the corrected boxes back to the YOLO "
                "label file (in bbox format)."
            )
            review_state = gr.State()
            with gr.Row():
                review_imgs_in = gr.Textbox(
                    label="Image directory", placeholder="/data/images_to_label",
                )
                review_out_in = gr.Textbox(
                    label="Output directory (labels/ + classes.txt)",
                    value="annotations_out",
                )
                review_load_btn = gr.Button("Load", variant="primary")
            review_status_out = gr.Markdown()
            review_annotator = image_annotator(
                label="Annotation", use_default_label=True,
            )
            with gr.Row():
                review_prev_btn = gr.Button("← Prev")
                review_save_btn = gr.Button("Save", variant="primary")
                review_savenext_btn = gr.Button("Save + Next →", variant="primary")
                review_next_btn = gr.Button("Next →")

            review_load_btn.click(
                review_load,
                inputs=[review_imgs_in, review_out_in],
                outputs=[review_annotator, review_state, review_status_out],
            )
            review_prev_btn.click(
                lambda s: review_nav(s, -1), inputs=[review_state],
                outputs=[review_annotator, review_state, review_status_out],
            )
            review_next_btn.click(
                lambda s: review_nav(s, 1), inputs=[review_state],
                outputs=[review_annotator, review_state, review_status_out],
            )
            review_save_btn.click(
                lambda a, s: review_save(a, s, False),
                inputs=[review_annotator, review_state],
                outputs=[review_annotator, review_state, review_status_out],
            )
            review_savenext_btn.click(
                lambda a, s: review_save(a, s, True),
                inputs=[review_annotator, review_state],
                outputs=[review_annotator, review_state, review_status_out],
            )
    return demo


if __name__ == "__main__":
    # Bind to all interfaces so the UI is reachable when deployed on a remote
    # (GPU) server; override host/port via env if needed.
    build_ui().launch(
        server_name=os.environ.get("APP_HOST", "0.0.0.0"),
        server_port=int(os.environ.get("APP_PORT", "7860")),
    )
