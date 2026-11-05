# SignLab model source

This folder accompanies the browser source in the SignLab repository. It contains training and export code, the selected editable YOLO11n checkpoint, the official starting checkpoint, and the checkpoint inputs used to export the two MobileNetV3 classifiers. It also contains portable training arguments, the YOLO11 architecture configuration, and the recorded validation selection.

## Export the supplied models

From this folder, use Python 3.12 and install `requirements-export.txt` in a virtual environment. `requirements-lock.txt` records the original experiment environment. Run:

```sh
python src/train_detector.py export
python src/run_experiments.py export
```

The scripts write `web/models/` within this folder. Detector export uses CPU, image size 320, batch size 1 and ONNX opset 17 with static dimensions. NMS runs in the browser. The classifier exports use the supplied synthetic example when the original manifest is unavailable. Export checks the analytical Grad-CAM implementation against PyTorch gradients.

The selected detector checkpoint uses relative filesystem paths and has its original repository metadata cleared. `checkpoint_review.json` records hashes, metadata changes, and the tensor equality check.

## Training inputs

Training and evaluation use local inputs consisting of images, reviewed labels, split lists, and a synthetic manifest. The coursework street photographs, videos, original identifiers, annotations, and per-image evaluation logs remain in the private coursework package. Recorded scores and selection in the app come from the original experiment. Re-exporting the included checkpoints is independent of these private inputs.

The original training workflow reads `data/original/manifest.csv` with `path`, `label`, `split`, and `clip` columns and matching images. The annotation script reads optional `bright_triangle_clips` and `parking_merge_exempt_clip` overrides from `data/private/annotation_clip_rules.json`. For synthesis, `data/private/cutouts.json` maps each class to its local training image `file` and `polygon` coordinates. Generate backgrounds with `src/generate_maskgen.py` and `requirements-generation.txt` in a separate environment, then run `src/build_synthetic.py`. The pinned upstream revision is recorded in `vendor/VENDOR.json`.

The reviewed detector data uses `data/detection/data.yaml`, `train.txt`, `val.txt`, `test.txt`, image and label directories, and `annotation_provenance.json`. The original fixed-dataset run expects 569 reviewed frames. Run `python src/train_detector.py prepare` on the local data to generate runtime configurations, then use `train-real`, `train-maskgen`, or `evaluate` as needed.

SignLab application source and the Ultralytics detector are distributed under AGPL-3.0 as described in the repository's root LICENSE. Third-party components retain the licenses in `licenses/` and `vendor/1d-tokenizer/LICENSE`.
