# SignLab model source

This folder contains source and editable checkpoint inputs for the browser models. The current detector was selected from the completed 353/453/581-image experiment. The two MobileNetV3 classifiers retain the earlier pilot workflow and measurements.

## Re-export the current detector

From this folder, create a Python 3.12 virtual environment and install `requirements-export.txt`. This CPU export environment is recorded separately from the measured CUDA training environment.

```sh
python -m pip install -r requirements-export.txt
python src/train_expanded_detector.py export
```

The exporter loads `artifacts/detection/original/weights/best.pt` and writes `../models/detector.onnx`. It uses a temporary copy of the checkpoint, CPU, static 1×3×320×320 input, batch size 1 and ONNX opset 17. The raw 1×8×2100 output requires box decoding and per-class NMS in the browser. Preprocessing is RGB letterboxing with pad value 114 and division by 255. Exported metadata uses portable model information, and graph docstrings are cleared.

For a separate verification output, specify its destination and optional provenance record.

```sh
python src/train_expanded_detector.py export \
  --output /path/to/export-check/detector.onnx \
  --record /path/to/export-check/export.json
```

The source checkpoint has portable metadata. `checkpoint_review.json` records the original and sanitized checkpoint hashes and equality of all 499 stored tensors. Different PyTorch/export environments can change serialized ONNX bytes, so verification should compare topology, tensors and numerical outputs. The shipped ONNX retains the completed training run's export.

## Reproduce the three-arm experiment

The completed run used Python 3.13.16 on Linux x86_64 and a Tesla T4, with PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130 and Ultralytics 8.4.190. `requirements-expanded.txt` pins the measured direct dependencies and relevant runtime dependencies. It uses the official CUDA 13.0 wheel index. The existing `requirements-export.txt`, `requirements-lock.txt` and generation requirements describe their separate export or pilot environments. [Official PyTorch CUDA wheels](https://download.pytorch.org/whl/cu130/).

Install the CUDA requirements in a suitable Linux environment. Place the private `Assignment2_YOLO_Training_Data.zip` and the output directory outside this public repository. The ZIP includes the original images, 100 added frames with 169 reviewed boxes, 128 labeled MaskGen-background composites, official initialization weights, immutable hashes and metadata.

```sh
python -m pip install -r requirements-expanded.txt
python src/train_expanded_detector.py train \
  --data-archive /path/to/private/Assignment2_YOLO_Training_Data.zip \
  --work-dir /path/to/private/training-work \
  --epochs 20 --device 0
```

`--checks-only` validates safe ZIP extraction, all packaged hashes, image/box consistency, 353/453/581 nested training lists and the fixed 120/96 validation/test split. It also saves a private ground-truth preview. A normal run trains all three arms, selects each best checkpoint using training validation mAP50–95, evaluates the three checkpoints uniformly on validation, saves the selected arm, and then evaluates the test set. It saves curves, private predictions, selected ONNX and a results ZIP. The CLI follows the source cells of the standalone training notebook and has no notebook or original project dependency.

Each arm uses seed 42, batch size 8, AdamW learning rate 0.001, freeze 10, disabled flips and up to 20 epochs with patience 8. The shared parameters are recorded in `configs/expanded-experiment.json`. Matching runs resume `last.pt` or reuse completed training. For changed data, software or settings, choose a fresh `--experiment-name`. Fixed seeds do not guarantee identical results across devices and software builds.

Validation selected `original` with mAP50–95 0.9051286. Test mAP50–95 was 0.8622392 for original, 0.8441348 for expanded and 0.8469105 for expanded plus synthetic. All three test mAP50 values were 0.995. The additions did not improve this inherited test score. Every new frame is training data. Repeated physical signs, source overlap and partial original annotations limit generalization. The synthetic comparison also changes sample count, compositing and optimizer steps. [Recorded experiment](configs/expanded-experiment.json), [validation selection](artifacts/detection/deployment_selection.json).

## Historical classifiers and generation

For the two MobileNetV3 classifiers, use the CPU export environment and `python src/run_experiments.py export`. Their outputs are written under this folder's `web/models/` and can be copied to the repository's `models/` folder after verification. Analytical Grad-CAM is checked against PyTorch gradients. Their earlier classifier trials retained the A1 head and use the original pilot metrics.

`src/train_detector.py` preserves the historical two-arm detector workflow. Its exporter belongs to that earlier experiment. Use `src/train_expanded_detector.py export` for the currently deployed detector.

`src/generate_maskgen.py`, `src/build_synthetic.py`, generation requirements and the pinned vendor revision preserve the earlier offline background/compositing workflow. The packaged composites use actual MaskGen and TA-TiTok backgrounds plus real training sign cutouts. Original photographs, videos, extracted frames, annotations, generated private previews and per-image evaluation outputs stay in the private coursework package and private run directory.

SignLab application source and the Ultralytics detector are distributed under AGPL-3.0. Third-party components retain the licenses in `licenses/` and `vendor/1d-tokenizer/LICENSE`.
