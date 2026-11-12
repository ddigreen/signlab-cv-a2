"""Fine-tune SignLab on a reviewed video dataset and evaluate fixed splits."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time


CLASS_NAMES = ["give_way", "main_road", "parking", "pedestrian_crossing"]


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def checkpoint_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    data = args.data_root.resolve()
    output = args.output.resolve()
    initial = args.checkpoint.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(output / "settings")
    os.environ["MPLCONFIGDIR"] = str(output / "mpl")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ["YOLO_AUTOINSTALL"] = "false"

    import torch
    import ultralytics
    from ultralytics import YOLO
    import yaml

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable.")
    ultralytics.settings.update({"sync": False})
    torch.set_num_threads(4)
    cfg = yaml.safe_load((data / "data.yaml").read_text())
    names = cfg["names"]
    assert [names[i] for i in range(4)] == CLASS_NAMES
    assert cfg["train"] == "train_video_refresh.txt"
    cfg["path"] = str(data)
    yaml_file = output / "runtime_data.yaml"
    yaml_file.write_text(yaml.safe_dump(cfg, sort_keys=False))
    for split in ("train", "val", "test"):
        paths = (data / cfg[split]).read_text().splitlines()
        assert paths and len(paths) == len(set(paths)) and all((data / p).is_file() for p in paths)
    train_config = {
        "epochs": args.epochs, "imgsz": 320, "batch": args.batch, "workers": 0,
        "optimizer": "AdamW", "lr0": .0004, "lrf": .3, "warmup_epochs": 0.0,
        "freeze": 10, "fliplr": 0.0, "flipud": 0.0, "degrees": 5.0,
        "translate": .08, "scale": .25, "hsv_h": .005, "hsv_s": .2, "hsv_v": .18,
        "mosaic": 0.0, "mixup": 0.0, "close_mosaic": 0, "seed": 42,
        "deterministic": False, "amp": False, "patience": 0, "plots": False,
        "cache": False, "val": True, "save": True,
    }
    started = time.monotonic()
    model = YOLO(str(initial))
    model.train(
        data=str(yaml_file), device=args.device, project=str(output),
        name="fine_tune", exist_ok=False, **train_config,
    )
    train_seconds = time.monotonic() - started
    best = output / "fine_tune" / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    checkpoints = {"expanded_synthetic": initial, "video_refresh": best}

    def evaluate(arm, split):
        result = YOLO(str(checkpoints[arm])).val(
            data=str(yaml_file), split=split, imgsz=320, batch=8, rect=False,
            conf=.001, iou=.70, device="cpu", workers=0, plots=False, verbose=False,
            project=str(output / "eval"), name=f"{arm}_{split}", exist_ok=True,
        )
        return {
            "arm": arm, "split": split, "mAP50": float(result.box.map50),
            "mAP50_95": float(result.box.map), "precision": float(result.box.mp),
            "recall": float(result.box.mr), "per_class_mAP50_95": [float(x) for x in result.box.maps],
            "inference_ms_per_image": float(result.speed["inference"]),
        }

    validation = [evaluate(arm, "val") for arm in checkpoints]
    selected = max(validation, key=lambda item: item["mAP50_95"])["arm"]
    selection = {
        "selected_arm": selected, "criterion": "Uniform validation mAP@0.50:0.95",
        "validation_metrics": validation, "test_used_for_selection": False,
        "sha256_by_arm": {arm: checkpoint_hash(path) for arm, path in checkpoints.items()},
    }
    write_json(output / "selection_before_test.json", selection)
    test = [evaluate(arm, "test") for arm in checkpoints]
    write_json(output / "results.json", {
        **selection, "test_metrics": test, "config": train_config,
        "training_seconds": round(train_seconds, 2),
        "training_device": args.device, "evaluation_device": "cpu",
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "validation_images": 120, "test_images": 96,
    })
    print("Training and uniform evaluation complete:", output / "results.json", flush=True)


if __name__ == "__main__":
    main()
