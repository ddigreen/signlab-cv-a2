"""Reproduce the three-arm detector workflow with a privately supplied data ZIP.

The training body follows the standalone training notebook's ordered Python cells.
Only source code is included. Generated data previews and run outputs are written
under the caller's private work directory.
"""
from pathlib import Path, PurePosixPath
from collections import Counter, defaultdict
from importlib import metadata
import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

MODEL_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = MODEL_ROOT.parent
DEFAULT_CONFIG = MODEL_ROOT / "configs/expanded-experiment.json"
EXPORT_ARMS = ("original", "expanded", "expanded_synthetic")


def show_table(value):
    print(value.to_string())


def clear_docstrings(message):
    if "doc_string" in message.DESCRIPTOR.fields_by_name:
        message.ClearField("doc_string")
    for field, value in message.ListFields():
        if field.message_type is not None:
            values = value if field.is_repeated else [value]
            for child in values:
                clear_docstrings(child)


def export_checkpoint(checkpoint, output, experiment, record=None, arm=None):
    """Export without writing generated files beside the editable checkpoint."""
    import onnx
    from ultralytics import YOLO

    checkpoint = Path(checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    arm = arm or experiment["selected_arm"]
    validation_selected_arm = experiment.get("validation_selected_arm", experiment["selected_arm"])
    is_validation_selected = arm == validation_selected_arm
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="signlab-export-") as temporary:
        staged_checkpoint = Path(temporary) / "model.pt"
        shutil.copyfile(checkpoint, staged_checkpoint)
        model = YOLO(str(staged_checkpoint))
        exported = Path(model.export(format="onnx", imgsz=320, batch=1, opset=17,
                                     simplify=False, dynamic=False, nms=False, device="cpu"))
        graph = onnx.load(exported)
        old_properties = {entry.key: entry.value for entry in graph.metadata_props}
        keep = {"stride", "task", "batch", "imgsz", "names", "channels", "end2end", "version", "license", "docs"}
        properties = {key: value for key, value in old_properties.items() if key in keep}
        properties.update({
            "description": "SignLab traffic-sign detector",
            "arm": arm,
            "validation_selected_arm": validation_selected_arm,
            "is_validation_selected": str(is_validation_selected).lower(),
            "dataset_sha256": experiment["dataset_archive_sha256"],
        })
        if is_validation_selected:
            properties["selected_arm"] = arm
        onnx.helper.set_model_props(graph, properties)
        clear_docstrings(graph)
        onnx.checker.check_model(graph)
        inputs = [{"name": value.name, "shape": [d.dim_value for d in value.type.tensor_type.shape.dim]}
                  for value in graph.graph.input]
        outputs = [{"name": value.name, "shape": [d.dim_value for d in value.type.tensor_type.shape.dim]}
                   for value in graph.graph.output]
        if inputs[0]["shape"] != [1, 3, 320, 320] or outputs[0]["shape"] != [1, 8, 2100]:
            raise ValueError("The checkpoint must export the four-class static YOLO11n graph.")
        temporary_output = output.with_name(output.name + ".tmp")
        onnx.save(graph, temporary_output)
        temporary_output.replace(output)
    details = {
        "arm": arm,
        "validation_selected_arm": validation_selected_arm,
        "is_validation_selected": is_validation_selected,
        "onnx_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "opset": 17, "input_shape": [1, 3, 320, 320], "nms_in_graph": False,
        "preprocess": "RGB letterbox, pad114, scale1/255, NCHW",
        "inputs": inputs, "outputs": outputs,
    }
    if is_validation_selected:
        details["selected_arm"] = arm
    if record:
        record = Path(record).expanduser().resolve()
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(details, indent=2))
    return details


def train_experiment(args):

    # Configuration
    DATA_ARCHIVE = str(args.data_archive)  # ZIP path. Leave empty to search beside the notebook or upload in Colab.
    WORK_DIR = args.work_dir
    EXPERIMENT_NAME = args.experiment_name or f"yolo11n_expanded_seed42_{args.epochs}epochs"
    EPOCHS = args.epochs
    DEVICE = args.device  # CUDA GPU 0 when available, then Apple MPS, then CPU.
    RESUME_EXISTING = not args.no_resume
    BASE_WEIGHTS_PATH = str(args.weights) if args.weights else ""  # Optional override. Otherwise use the weights inside the data ZIP.

    WORK_DIR = Path(WORK_DIR).expanduser().resolve()
    if not isinstance(EPOCHS, int) or EPOCHS < 1:
        raise ValueError("EPOCHS must be a positive integer.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", EXPERIMENT_NAME):
        raise ValueError("Use a simple experiment folder name without slashes.")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR = WORK_DIR / "runs" / EXPERIMENT_NAME
    CLASSES = ["give_way", "main_road", "parking", "pedestrian_crossing"]
    ARMS = ["original", "expanded", "expanded_synthetic"]
    ARM_TRAIN_FILES = {arm: f"train_{arm}.txt" for arm in ARMS}

    def sha256_file(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def save_json(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(path)

    print("Work directory", WORK_DIR)

    # Runtime environment
    if metadata.version("ultralytics") != "8.4.190":
        raise RuntimeError("Install ultralytics==8.4.190 for this workflow.")
    os.environ["YOLO_CONFIG_DIR"] = str(WORK_DIR / "ultralytics_settings")
    os.environ.setdefault("MPLCONFIGDIR", str(WORK_DIR / "matplotlib_config"))
    os.environ.setdefault("XDG_CACHE_HOME", str(WORK_DIR / "cache"))
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["COMET_MODE"] = "DISABLED"
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from PIL import Image
    from ultralytics import YOLO, settings

    settings.update({key: False for key in (
        "sync", "clearml", "comet", "dvc", "mlflow", "raytune", "tensorboard", "wandb"
    )})
    if DEVICE == "auto":
        DEVICE = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if DEVICE == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "device": str(DEVICE),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": {name: metadata.version(name) for name in (
            "torch", "torchvision", "ultralytics", "onnx", "numpy", "pandas", "matplotlib", "PyYAML"
        )},
    }
    print(json.dumps(environment, indent=2))

    # Safe extraction and immutable file inventory
    archive_path = Path(DATA_ARCHIVE).expanduser().resolve()
    if archive_path is None or not archive_path.is_file() or not zipfile.is_zipfile(archive_path):
        raise FileNotFoundError("Place the training-data ZIP beside the notebook or set DATA_ARCHIVE.")
    archive_sha256 = sha256_file(archive_path)
    DATA_ROOT = WORK_DIR / "datasets" / archive_sha256[:16]

    def package_path(value):
        path = PurePosixPath(value)
        if not value or path.is_absolute() or ".." in path.parts or chr(92) in value or path.as_posix() != value:
            raise ValueError(f"Invalid relative dataset path: {value!r}")
        resolved = (DATA_ROOT / path).resolve()
        if not resolved.is_relative_to(DATA_ROOT):
            raise ValueError(f"Dataset path escapes its folder: {value}")
        return resolved

    if DATA_ROOT.exists():
        marker = DATA_ROOT / ".archive_sha256"
        if not marker.is_file() or marker.read_text().strip() != archive_sha256:
            raise RuntimeError(f"The cached dataset has an unexpected archive marker: {DATA_ROOT}")
    else:
        DATA_ROOT.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="extract-", dir=DATA_ROOT.parent) as temporary:
            stage = Path(temporary) / "dataset"
            stage.mkdir()
            with zipfile.ZipFile(archive_path) as archive:
                members = archive.infolist()
                if sum(member.file_size for member in members) > 4 * 1024**3:
                    raise ValueError("The expanded archive exceeds the expected 4 GiB size limit.")
                seen = set()
                for member in members:
                    relative = member.filename.rstrip("/")
                    package_path(relative)
                    if relative in seen or stat.S_ISLNK(member.external_attr >> 16):
                        raise ValueError(f"Duplicate entry or symlink in ZIP: {relative}")
                    seen.add(relative)
                    destination = stage / relative
                    if member.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member) as source, destination.open("wb") as target:
                            shutil.copyfileobj(source, target)
            (stage / ".archive_sha256").write_text(archive_sha256 + "\n")
            stage.rename(DATA_ROOT)
    with zipfile.ZipFile(archive_path) as archive:
        inventory_bytes = archive.read("manifest_sha256.csv")
        immutable_files = {member.filename for member in archive.infolist() if not member.is_dir()}
    if package_path("manifest_sha256.csv").read_bytes() != inventory_bytes:
        raise ValueError("The cached integrity manifest differs from the source ZIP.")
    inventory = list(csv.DictReader(io.StringIO(inventory_bytes.decode("utf-8-sig"))))
    recorded_files = [row["path"] for row in inventory]
    if len(recorded_files) != len(set(recorded_files)) or set(recorded_files) != immutable_files - {"manifest_sha256.csv"}:
        raise ValueError("The integrity manifest must list every other packaged file exactly once.")
    for row in inventory:
        path = package_path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"].lower():
            raise ValueError(f"Packaged file is missing or altered: {row['path']}")
    print("Dataset archive SHA-256", archive_sha256)
    print("Dataset folder", DATA_ROOT)
    print("Verified packaged file hashes", len(inventory))

    # Data and bounding-box validation
    def read_csv(relative):
        with package_path(relative).open(newline="", encoding="utf-8-sig") as stream:
            return list(csv.DictReader(stream))

    def read_image_list(filename):
        values = package_path(filename).read_text().splitlines()
        paths = []
        for value in values:
            value = value.strip()
            if value.startswith("./"):
                value = value[2:]
            if value:
                path = package_path(value)
                if not path.is_file() or not path.is_relative_to(DATA_ROOT / "images"):
                    raise FileNotFoundError(f"Missing dataset image: {value}")
                paths.append(path)
        if not paths or len(paths) != len(set(paths)):
            raise ValueError(f"Empty or duplicated image list: {filename}")
        return paths

    train_paths = {arm: read_image_list(name) for arm, name in ARM_TRAIN_FILES.items()}
    val_paths = read_image_list("val.txt")
    test_paths = read_image_list("test.txt")
    expected_counts = {"original": 353, "expanded": 453, "expanded_synthetic": 581}
    if {arm: len(paths) for arm, paths in train_paths.items()} != expected_counts:
        raise ValueError("The three training lists do not match the expected 353/453/581 images.")
    if len(val_paths) != 120 or len(test_paths) != 96 or set(val_paths) & set(test_paths):
        raise ValueError("Expected disjoint 120-image validation and 96-image test lists.")
    if not set(train_paths["original"]) < set(train_paths["expanded"]) < set(train_paths["expanded_synthetic"]):
        raise ValueError("The training arms must contain the same base images plus their stated additions.")
    if set(train_paths["expanded_synthetic"]) & (set(val_paths) | set(test_paths)):
        raise ValueError("A training image also appears in validation or test.")

    original_manifest = read_csv("metadata/original_manifest.csv")
    addon_manifest = read_csv("metadata/addon_manifest.csv")
    synthetic_manifest = read_csv("metadata/synthetic_manifest.csv")
    if (len(original_manifest), len(addon_manifest), len(synthetic_manifest)) != (569, 100, 128):
        raise ValueError("Dataset manifests do not contain the expected 569/100/128 images.")
    expected_original = {package_path(row["image_path"]) for row in original_manifest}
    if expected_original != set(train_paths["original"]) | set(val_paths) | set(test_paths):
        raise ValueError("The original manifest disagrees with the image lists.")
    original_splits = {"train": set(train_paths["original"]), "val": set(val_paths), "test": set(test_paths)}
    for row in original_manifest:
        if row["split"] not in original_splits or package_path(row["image_path"]) not in original_splits[row["split"]]:
            raise ValueError("The original manifest split differs from its image list.")
    expected_addon = {package_path(row["image_path"]) for row in addon_manifest}
    expected_synthetic = {package_path(row["image_path"]) for row in synthetic_manifest}
    if expected_addon != set(train_paths["expanded"]) - set(train_paths["original"]):
        raise ValueError("The expanded arm does not add exactly the 100 annotated video frames.")
    if expected_synthetic != set(train_paths["expanded_synthetic"]) - set(train_paths["expanded"]):
        raise ValueError("The synthetic arm does not add exactly the 128 composites.")
    if any(row["split"] != "train" for row in addon_manifest + synthetic_manifest):
        raise ValueError("Added real and synthetic frames must remain in training.")
    for field in ("source_video", "group_id"):
        group_splits = defaultdict(set)
        for row in addon_manifest:
            if not row[field].strip():
                raise ValueError(f"Missing addon grouping field: {field}")
            group_splits[row[field]].add(row["split"])
        if any(splits != {"train"} for splits in group_splits.values()):
            raise ValueError(f"An addon {field} occurs outside training.")

    all_images = sorted(set(train_paths["expanded_synthetic"]) | set(val_paths) | set(test_paths))
    boxes_by_image = {}
    image_hashes = {}
    for image_path in all_images:
        relative = image_path.relative_to(DATA_ROOT / "images")
        label_path = DATA_ROOT / "labels" / relative.with_suffix(".txt")
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing YOLO box labels: {label_path}")
        image_hashes[image_path] = sha256_file(image_path)
        with Image.open(image_path) as image:
            image.verify()
        boxes = []
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            values = [float(value) for value in line.split()]
            if len(values) != 5 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"Expected five finite YOLO detection values: {label_path}")
            class_id, x, y, width, height = values
            if class_id != int(class_id) or int(class_id) not in range(len(CLASSES)):
                raise ValueError(f"Unknown class ID: {label_path}")
            if min(width, height) <= 0 or min(x - width/2, y - height/2) < -0.000001 or max(x + width/2, y + height/2) > 1.000001:
                raise ValueError(f"Box extends outside its image: {label_path}")
            boxes.append((int(class_id), x, y, width, height))
        if not boxes or len(boxes) != len(set(boxes)):
            raise ValueError(f"Missing or duplicate boxes: {label_path}")
        boxes_by_image[image_path] = boxes
    hash_checks = 0
    for manifest in (original_manifest, addon_manifest, synthetic_manifest):
        for row in manifest:
            if row.get("sha256"):
                if image_hashes[package_path(row["image_path"])] != row["sha256"].lower():
                    raise ValueError(f"Image SHA-256 differs from its manifest: {row['image_path']}")
                hash_checks += 1

    counts = []
    for name, paths in [*train_paths.items(), ("validation", val_paths), ("test", test_paths)]:
        instances = Counter(box[0] for image in paths for box in boxes_by_image[image])
        counts.append({"set": name, "images": len(paths), "boxes": sum(instances.values()),
                       **{label: instances[index] for index, label in enumerate(CLASSES)}})
    counts_table = pd.DataFrame(counts).set_index("set")
    show_table(counts_table)
    addon_box_count = sum(len(boxes_by_image[image]) for image in expected_addon)
    primary_counts = Counter(row["class_name"] for row in addon_manifest)
    instance_counts = Counter(box[0] for image in expected_addon for box in boxes_by_image[image])
    show_table(pd.DataFrame([
        {"class": label, "primary_labeled_frames": primary_counts[label],
         "annotated_instances": instance_counts[index]}
        for index, label in enumerate(CLASSES)
    ]).set_index("class"))
    print("Added frames", len(expected_addon), "with", addon_box_count, "boxes")
    print("Added multi-object frames", sum(len(boxes_by_image[image]) > 1 for image in expected_addon))
    print("Provisional addon groups", len({row["group_id"] for row in addon_manifest}))
    print("Image hashes verified against manifests", hash_checks)
    annotation_provenance = json.loads(package_path("annotation_provenance.json").read_text())
    print("Annotation provenance")
    provenance_fields = ("annotation_method", "reviewed_addon_image_count", "reviewed_addon_box_count",
                         "box_convention", "independent_human_review")
    print(json.dumps({key: annotation_provenance.get(key) for key in provenance_fields}, indent=2, ensure_ascii=False))

    # Portable runtime configurations and experiment identity
    train_settings = {
        "epochs": EPOCHS, "imgsz": 320, "batch": 8, "workers": 0,
        "seed": 42, "deterministic": True, "freeze": 10,
        "optimizer": "AdamW", "lr0": 0.001, "lrf": 0.1, "weight_decay": 0.0005,
        "patience": 8, "fliplr": 0.0, "flipud": 0.0, "degrees": 10.0,
        "hsv_h": 0.01, "hsv_s": 0.3, "hsv_v": 0.3, "mosaic": 0.5,
        "close_mosaic": min(5, EPOCHS), "mixup": 0.0, "copy_paste": 0.0,
        "amp": False, "cache": False, "plots": True, "save": True, "verbose": True,
    }
    WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
    WEIGHTS_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
    experiment_spec = {
        "dataset_archive_sha256": archive_sha256,
        "architecture": "YOLO11n", "pretrained_sha256": WEIGHTS_SHA256,
        "ultralytics_version": metadata.version("ultralytics"),
        "torch_version": metadata.version("torch"),
        "torchvision_version": metadata.version("torchvision"),
        "arms": ARMS, "classes": CLASSES, "training_settings": train_settings,
        "validation_selection": "highest mAP50-95 on validation; arm order breaks ties",
        "test_used_for_selection": False,
    }
    specification_path = EXPERIMENT_DIR / "experiment.json"
    if specification_path.exists():
        if json.loads(specification_path.read_text()) != experiment_spec:
            raise RuntimeError("This experiment name has different data or settings. Choose a new EXPERIMENT_NAME.")
        if not RESUME_EXISTING:
            raise FileExistsError("Choose a fresh EXPERIMENT_NAME or enable RESUME_EXISTING.")
    elif EXPERIMENT_DIR.exists() and any(EXPERIMENT_DIR.iterdir()):
        raise FileExistsError("The experiment folder has unrecognized contents. Choose a fresh name.")
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(specification_path, experiment_spec)
    save_json(EXPERIMENT_DIR / "environment.json", environment)
    save_json(EXPERIMENT_DIR / "annotation_provenance.json", annotation_provenance)
    counts_table.to_csv(EXPERIMENT_DIR / "dataset_counts.csv")
    save_json(EXPERIMENT_DIR / "data_audit.json", {
        "dataset_sha256": archive_sha256, "images": len(all_images),
        "verified_packaged_files": len(inventory),
        "hashes_checked_against_manifest": hash_checks,
        "added_real_frames": len(expected_addon), "added_real_boxes": addon_box_count,
        "addon_groups": len({row["group_id"] for row in addon_manifest}),
        "all_new_real_and_synthetic_frames_in_train": True,
        "inherited_video_and_physical_sign_dependence": True,
    })
    runtime = EXPERIMENT_DIR / "runtime"
    runtime.mkdir(exist_ok=True)
    for name, paths in [*train_paths.items(), ("val", val_paths), ("test", test_paths)]:
        (runtime / f"{name}.txt").write_text("\n".join(str(path) for path in paths) + "\n")
    runtime_configs = {}
    for arm in ARMS:
        configuration = {"path": str(DATA_ROOT), "train": str(runtime / f"{arm}.txt"),
                         "val": str(runtime / "val.txt"), "test": str(runtime / "test.txt"),
                         "names": dict(enumerate(CLASSES))}
        path = runtime / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(configuration, sort_keys=False))
        runtime_configs[arm] = path
    print("Experiment folder", EXPERIMENT_DIR)
    print(json.dumps(train_settings, indent=2))

    # Private ground-truth preview
    colors = ["#ef4444", "#facc15", "#60a5fa", "#4ade80"]
    multi_object = sorted(expected_addon, key=lambda path: (-len(boxes_by_image[path]), str(path)))
    examples = []
    used_videos = set()
    source_lookup = {package_path(row["image_path"]): row["source_video"] for row in addon_manifest}
    for path in multi_object:
        video = source_lookup[path]
        if video not in used_videos:
            examples.append(path)
            used_videos.add(video)
        if len(examples) == 6:
            break
    figure, axes = plt.subplots(2, 3, figsize=(13, 10))
    for axis, path in zip(axes.flat, examples):
        image = Image.open(path).convert("RGB")
        width, height = image.size
        axis.imshow(image)
        for class_id, x, y, box_width, box_height in boxes_by_image[path]:
            left, top = (x-box_width/2)*width, (y-box_height/2)*height
            axis.add_patch(plt.Rectangle((left, top), box_width*width, box_height*height,
                                         fill=False, edgecolor=colors[class_id], linewidth=2))
            axis.text(left, top, str(class_id), fontsize=9, va="bottom", color="black",
                      bbox={"facecolor": colors[class_id], "alpha": 0.9, "pad": 1})
        axis.set_title(f"{source_lookup[path]} | {len(boxes_by_image[path])} labeled signs")
        axis.axis("off")
    figure.suptitle("Ground-truth examples from the added TRAIN frames")
    figure.text(0.5, 0.01, "0 give way   |   1 main road   |   2 parking   |   3 pedestrian crossing",
                ha="center", fontsize=11)
    figure.tight_layout(rect=(0, 0.03, 1, 0.97))
    figure.savefig(EXPERIMENT_DIR / "ground_truth_preview.png", dpi=170)
    plt.close(figure)

    if args.checks_only:
        print("Dataset checks complete.")
        return

    # Train the three arms or resume matching runs
    def clear_device_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def history_for(arm):
        history = pd.read_csv(EXPERIMENT_DIR / "training" / arm / "results.csv")
        history.columns = history.columns.str.strip()
        return history

    if BASE_WEIGHTS_PATH:
        weights_path = Path(BASE_WEIGHTS_PATH).expanduser().resolve()
    else:
        weights_path = package_path("pretrained/yolo11n.pt")
    if not weights_path.is_file() or sha256_file(weights_path) != WEIGHTS_SHA256:
        raise ValueError("BASE_WEIGHTS_PATH must contain the recorded official yolo11n.pt weights.")
    save_json(EXPERIMENT_DIR / "pretrained_provenance.json", {
        "source": WEIGHTS_URL, "sha256": WEIGHTS_SHA256, "bytes": weights_path.stat().st_size,
    })
    for arm in ARMS:
        arm_dir = EXPERIMENT_DIR / "training" / arm
        completed_path = arm_dir / "completed.json"
        best_path = arm_dir / "weights" / "best.pt"
        last_path = arm_dir / "weights" / "last.pt"
        if completed_path.exists():
            completed = json.loads(completed_path.read_text())
            if not best_path.exists() or not (arm_dir / "results.csv").exists() or sha256_file(best_path) != completed["best_weights_sha256"]:
                raise RuntimeError(f"Completed checkpoint is missing or changed for {arm}.")
            print("Reusing completed training", arm)
            continue
        model = None
        if last_path.exists():
            checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
            saved_epoch = int(checkpoint.get("epoch", -1))
            stopper = checkpoint.get("stopper") or {}
            stopped_early = "best_epoch" in stopper and train_settings["patience"] > 0 and (
                saved_epoch + 1 - int(stopper["best_epoch"]) >= train_settings["patience"]
            )
            finished = saved_epoch == -1 or saved_epoch >= EPOCHS-1 or stopped_early
            unfinished = not finished and saved_epoch >= 0 and checkpoint.get("optimizer") is not None
            del checkpoint
            if unfinished:
                print("Resuming", arm)
                model = YOLO(str(last_path))
                model.train(resume=True, data=str(runtime_configs[arm]), device=DEVICE,
                            workers=0, save_dir=str(arm_dir))
            elif not finished or not best_path.exists() or not (arm_dir / "results.csv").exists():
                raise RuntimeError(f"Incomplete training artifacts for {arm}. Choose a new experiment folder.")
        elif arm_dir.exists() and any(arm_dir.iterdir()):
            raise RuntimeError(f"Training stopped before saving {arm}/weights/last.pt. Choose a new experiment name.")
        else:
            print("Training", arm, "from identical COCO weights")
            model = YOLO(str(weights_path))
            model.train(data=str(runtime_configs[arm]), device=DEVICE, pretrained=True,
                        project=str(EXPERIMENT_DIR / "training"), name=arm,
                        exist_ok=True, **train_settings)
        if not best_path.exists():
            raise RuntimeError(f"Training did not save best.pt for {arm}.")
        history = history_for(arm)
        save_json(completed_path, {
            "arm": arm, "epochs_recorded": len(history),
            "best_training_validation_map50_95": float(history["metrics/mAP50-95(B)"].max()),
            "best_weights_sha256": sha256_file(best_path),
            "selection_split": "val", "test_used": False,
        })
        del model
        clear_device_memory()


    # Select checkpoints and arm from validation
    def evaluate_checkpoint(arm, split):
        model = YOLO(str(EXPERIMENT_DIR / "training" / arm / "weights" / "best.pt"))
        result = model.val(data=str(runtime_configs[arm]), split=split,
                           imgsz=320, batch=8, device=DEVICE, workers=0,
                           conf=0.001, iou=0.7, rect=False, plots=True, save_json=True,
                           project=str(EXPERIMENT_DIR / "evaluation"),
                           name=f"{arm}_{split}", exist_ok=True, verbose=True)
        metrics = {
            "arm": arm, "split": split, "images": len(val_paths if split == "val" else test_paths),
            "mAP50": float(result.box.map50), "mAP50_95": float(result.box.map),
            "precision": float(result.box.mp), "recall": float(result.box.mr),
            "model_inference_ms_per_image": float(result.speed["inference"]),
        }
        save_json(EXPERIMENT_DIR / "evaluation" / f"{arm}_{split}_metrics.json", {
            **metrics, "class_order": CLASSES, "per_class_map50_95": result.box.maps.tolist(),
            "speed_ms": {name: float(value) for name, value in result.speed.items()},
        })
        del model, result
        clear_device_memory()
        return metrics

    validation_metrics = [evaluate_checkpoint(arm, "val") for arm in ARMS]
    selected_arm = max(ARMS, key=lambda arm: next(
        row["mAP50_95"] for row in validation_metrics if row["arm"] == arm
    ))
    selection = {
        "selected_arm": selected_arm, "criterion": "highest validation mAP50-95",
        "tie_break_order": ARMS, "test_used_for_selection": False,
        "dataset_archive_sha256": archive_sha256, "validation_metrics": validation_metrics,
        "best_checkpoint_sha256": {
            arm: sha256_file(EXPERIMENT_DIR / "training" / arm / "weights" / "best.pt") for arm in ARMS
        },
    }
    save_json(EXPERIMENT_DIR / "selection.json", selection)
    pd.DataFrame(validation_metrics).to_csv(EXPERIMENT_DIR / "validation_metrics.csv", index=False)
    show_table(pd.DataFrame(validation_metrics).set_index("arm"))
    print("Selected by validation", selected_arm)


    # Evaluate the fixed held-out test after selection
    selection = json.loads((EXPERIMENT_DIR / "selection.json").read_text())
    if selection["dataset_archive_sha256"] != archive_sha256 or selection["test_used_for_selection"]:
        raise RuntimeError("The saved selection does not match this validation-only experiment.")
    for arm in ARMS:
        checkpoint = EXPERIMENT_DIR / "training" / arm / "weights" / "best.pt"
        if sha256_file(checkpoint) != selection["best_checkpoint_sha256"][arm]:
            raise RuntimeError(f"The checkpoint for {arm} changed after validation selection.")
    test_metrics = [evaluate_checkpoint(arm, "test") for arm in ARMS]
    test_table = pd.DataFrame(test_metrics).set_index("arm")
    test_table["delta_mAP50_95_vs_original"] = test_table["mAP50_95"] - test_table.loc["original", "mAP50_95"]
    test_table.to_csv(EXPERIMENT_DIR / "test_metrics.csv")
    save_json(EXPERIMENT_DIR / "test_metrics.json", {
        "selected_arm": selection["selected_arm"], "rows": test_metrics,
        "selection_criterion": selection["criterion"], "test_used_for_selection": False,
        "scope": "Original dominant-target test annotations, inherited source dependence",
        "repeats": "One training seed per arm. No estimate of seed variation.",
    })
    show_table(test_table)


    # Save measured training and comparison plots
    figures_dir = EXPERIMENT_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for arm in ARMS:
        history = history_for(arm)
        for axis, column, title in zip(axes, ["train/box_loss", "metrics/mAP50(B)", "metrics/mAP50-95(B)"],
                                        ["Training box loss", "Validation mAP50", "Validation mAP50-95"]):
            axis.plot(history["epoch"], history[column], marker=".", label=arm)
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(figures_dir / "training_curves.png", dpi=170)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    x = np.arange(len(ARMS))
    validation = {row["arm"]: row["mAP50_95"] for row in validation_metrics}
    test = {row["arm"]: row["mAP50_95"] for row in test_metrics}
    axis.bar(x-0.18, [validation[arm] for arm in ARMS], width=0.36, label="Validation")
    axis.bar(x+0.18, [test[arm] for arm in ARMS], width=0.36, label="Original held-out test")
    axis.set_xticks(x, ARMS)
    axis.set_ylim(0, 1)
    axis.set_ylabel("mAP50-95")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures_dir / "validation_and_test.png", dpi=170)
    plt.close(figure)


    # Save private qualitative test predictions
    selected_arm = selection["selected_arm"]
    model = YOLO(str(EXPERIMENT_DIR / "training" / selected_arm / "weights" / "best.pt"))
    preview_paths = [test_paths[index] for index in np.linspace(0, len(test_paths)-1, 6, dtype=int)]
    results = model.predict(source=[str(path) for path in preview_paths], imgsz=320, device=DEVICE,
                            conf=0.25, iou=0.45, save=True,
                            project=str(EXPERIMENT_DIR / "prediction_preview"),
                            name="original_test", exist_ok=True, verbose=False)
    figure, axes = plt.subplots(2, 3, figsize=(13, 9))
    for axis, result, path in zip(axes.flat, results, preview_paths):
        axis.imshow(result.plot()[..., ::-1])
        axis.set_title(path.name, fontsize=8)
        axis.axis("off")
    figure.suptitle(f"Selected arm {selected_arm} | original test predictions at confidence 0.25")
    figure.tight_layout()
    figure.savefig(EXPERIMENT_DIR / "figures" / "test_predictions.png", dpi=170)
    plt.close(figure)
    del model, results
    clear_device_memory()


    # Export the validation-selected arm, then archive the private run outputs.
    selected_arm = selection["selected_arm"]
    selected_checkpoint = EXPERIMENT_DIR / "training" / selected_arm / "weights" / "best.pt"
    export_dir = EXPERIMENT_DIR / "deployment"
    export_dir.mkdir(exist_ok=True)
    export_checkpoint(selected_checkpoint, export_dir / "selected_yolo11n.onnx",
                      {"selected_arm": selected_arm, "dataset_archive_sha256": archive_sha256},
                      record=export_dir / "export.json")
    shutil.copyfile(selected_checkpoint, export_dir / "selected_yolo11n.pt")

    lock = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True)
    (EXPERIMENT_DIR / "requirements-lock.txt").write_text(lock.stdout)
    result_zip = EXPERIMENT_DIR.with_name(EXPERIMENT_DIR.name + ".zip")
    temporary_zip = result_zip.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(EXPERIMENT_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, Path(EXPERIMENT_DIR.name) / path.relative_to(EXPERIMENT_DIR))
    temporary_zip.replace(result_zip)
    print("Results ZIP", result_zip)
    print("SHA-256", sha256_file(result_zip))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Run data audit, all three arms, validation/test, plots and export.")
    train.add_argument("--data-archive", type=Path, required=True)
    train.add_argument("--work-dir", type=Path, required=True, help="Private directory for datasets and run outputs.")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--experiment-name")
    train.add_argument("--device", default="auto")
    train.add_argument("--weights", type=Path, help="Override the official initialization bundled with the private ZIP.")
    train.add_argument("--no-resume", action="store_true")
    train.add_argument("--checks-only", action="store_true", help="Validate the ZIP, labels and splits and save a private preview.")
    export = commands.add_parser("export", help="Re-export an experiment arm without training data.")
    export.add_argument("--arm", choices=EXPORT_ARMS,
                        help="Arm identity. Defaults to original, or infers it from a named checkpoint folder.")
    export.add_argument("--checkpoint", type=Path,
                        help="Override artifacts/detection/<arm>/weights/best.pt. Custom paths require --arm.")
    export.add_argument("--output", type=Path,
                        help="Override the arm-specific ONNX destination under models/.")
    export.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    export.add_argument("--record", type=Path, help="Optional output export-provenance JSON.")
    args = parser.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    if args.command == "train":
        train_experiment(args)
    else:
        checkpoint_arm = None
        if args.checkpoint and args.checkpoint.parent.name == "weights":
            folder_arm = args.checkpoint.parent.parent.name
            if folder_arm in EXPORT_ARMS:
                checkpoint_arm = folder_arm
        if args.arm and checkpoint_arm and args.arm != checkpoint_arm:
            parser.error("--arm disagrees with the arm in the checkpoint folder.")
        if args.checkpoint and not checkpoint_arm and not args.arm:
            parser.error("Provide --arm for a checkpoint outside a named arm/weights folder.")
        arm = args.arm or checkpoint_arm or "original"
        checkpoint = args.checkpoint or MODEL_ROOT / "artifacts/detection" / arm / "weights/best.pt"
        filename = "detector.onnx" if arm == "original" else f"detector-{arm.replace('_', '-')}.onnx"
        output = args.output or PUBLIC_ROOT / "models" / filename
        with tempfile.TemporaryDirectory(prefix="signlab-export-cache-") as cache:
            os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(cache) / "ultralytics"))
            os.environ.setdefault("MPLCONFIGDIR", str(Path(cache) / "matplotlib"))
            os.environ.setdefault("XDG_CACHE_HOME", str(Path(cache) / "cache"))
            experiment = json.loads(args.config.read_text())
            export_checkpoint(checkpoint, output, experiment, args.record, arm=arm)


if __name__ == "__main__":
    main()
