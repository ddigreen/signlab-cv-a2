# SignLab

SignLab detects four traffic-sign classes with YOLO11n and classifies images with MobileNetV3. The classifier shows a Grad-CAM overlay. Both modes measure browser inference latency and report the browser and platform.

Run the app from this directory with `python3 -m http.server 8000` and open [localhost:8000](http://localhost:8000). Select a synthetic example, choose a local file, or enable the camera. Camera access needs browser permission and a secure context such as localhost or HTTPS.

The deployed app is available at [SignLab on GitHub Pages](https://ddigreen.github.io/signlab-cv-a2/). GitHub Pages serves this directory directly. Keep the assets in their folders so models and the bundled ONNX Runtime resolve under the repository path. The `.nojekyll` file enables static publishing. [GitHub Pages documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages).

## Current detector experiment

The deployed detector comes from a completed three-arm experiment on a Tesla T4. Every arm started from the same COCO-pretrained YOLO11n weights and trained for 20 epochs with seed 42, 320-pixel inputs, batch size 8, AdamW and the first 10 modules frozen. All arms share the original 120 validation and 96 test images.

| Arm | Training images | Validation mAP50–95 | Test mAP50–95 |
| --- | ---: | ---: | ---: |
| Original | 353 | 0.9051 | 0.8622 |
| Expanded | 453 | 0.8931 | 0.8441 |
| Expanded + synthetic | 581 | 0.9026 | 0.8469 |

Validation selected **original** before test evaluation. Every arm obtained test mAP50 of 0.995. The 100 added frames contain 169 boxes and belong to training. The synthetic arm adds 128 composites using MaskGen backgrounds. These additions did not improve the original test mAP50–95 in this run. One seed, repeated signs and source dependence, and dominant-target original annotations limit interpretation. The test provides an evaluation under the inherited protocol. New-site and exhaustive multi-object evaluation remain future work. [Recorded experiment](model-source/configs/expanded-experiment.json).

## Classifier pilot and examples

The two MobileNetV3 classifiers and their Grad-CAM measurements come from the earlier 569-image pilot. That rerun achieved 94/96 clean predictions and macro-F1 0.9816. All nine new classifier arm/seed trials selected epoch zero, preserving the starting A1 classifier. These classifier measurements describe a separate task and experiment from the current detector.

The public examples and replay contain labeled synthetic composites. The Experiments tab contains aggregate held-out measurements. Camera and file inference runs on the device in the browser. The private coursework package holds original street photographs, videos, extracted frames, annotations and per-image logs.

## Source and license

[Model source and export instructions](model-source/README.md) cover the current three-arm detector CLI, checkpoint export, recorded CUDA environment and historical classifier source. The editable selected detector and official initialization checkpoint are included. Full training uses the user's private data ZIP and a private output directory. The source distribution contains generic code and aggregate records.

SignLab application code and the bundled Ultralytics detector are distributed under AGPL-3.0. Third-party components retain their licenses and notices. [License](LICENSE), [third-party notices](THIRD_PARTY_NOTICES.md).
