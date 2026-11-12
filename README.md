# SignLab

SignLab detects four traffic-sign classes with YOLO11n and classifies images with MobileNetV3. The classifier shows a Grad-CAM overlay. Both modes measure browser inference latency and report the browser and platform.

Run the app from this directory with `python3 -m http.server 8000` and open [localhost:8000](http://localhost:8000). Select a synthetic example, choose a local file, or enable the camera. Camera access needs browser permission and a secure context such as localhost or HTTPS.

The detector model selector appears above the frame. The camera opens as soon as you grant permission and shows a live preview while the detector loads. The first run may download about 10 MB for the model and 12 MB for the browser engine. The result card distinguishes download, setup, detections and completed frames with no detections. Digital zoom crops the center of the camera or file input; keep the sign in the middle. After no detections in a photo at 1×, auto zoom tries the center at 2× and 4×. It shows the full photo if all three attempts are empty. Camera and video zoom remain manual. Recorded metrics stay fixed.

The deployed app is available at [SignLab on GitHub Pages](https://ddigreen.github.io/signlab-cv-a2/). GitHub Pages serves this directory directly. Keep the assets in their folders so models and the bundled ONNX Runtime resolve under the repository path. The `.nojekyll` file enables static publishing. [GitHub Pages documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages).

## November video refresh

We screened 414 candidates from 16 new phone videos and added 126 reviewed frames with 134 sign boxes to training. The new package has 923 images: 579 real and 128 synthetic in training, 120 real in validation, and 96 real in test. The original validation and test sets are unchanged. Video refresh fine-tunes Expanded + synthetic for four epochs on an Apple M1.

| Checkpoint | Validation mAP50–95 | Clean test mAP50 | Clean test mAP50–95 |
| --- | ---: | ---: | ---: |
| Expanded + synthetic | 90.40% | 99.50% | 84.67% |
| Video refresh | 89.88% | 99.50% | 87.52% |

Expanded + synthetic remains the default because it scored higher on validation before the test comparison. [Try Video refresh](https://ddigreen.github.io/signlab-cv-a2/?detector=video_refresh) to use the new checkpoint. Both rows use the same CPU evaluation protocol. Test scores describe the original held-out images; the added videos only supplied training frames.

## Original detector experiment

The earlier three-arm YOLO11n experiment ran on a Tesla T4. **Original**, its validation winner, remains available in the selector or through a direct link. [Try Original](https://ddigreen.github.io/signlab-cv-a2/?detector=original).

Every arm started from the same COCO-pretrained YOLO11n weights and trained for 20 epochs with seed 42, 320-pixel inputs, batch size 8, AdamW and the first 10 modules frozen. All arms share the original 120 validation and 96 test images.

| Arm | Training images | Validation mAP50–95 | Test mAP50–95 |
| --- | ---: | ---: | ---: |
| Original | 353 | 0.9051 | 0.8622 |
| Expanded | 453 | 0.8931 | 0.8441 |
| Expanded + synthetic | 581 | 0.9026 | 0.8469 |

Validation selected **original** before test evaluation. Every arm obtained test mAP50 of 0.995. The 100 added frames contain 169 boxes and belong to training. The synthetic arm adds 128 composites using MaskGen backgrounds. These additions did not improve the original test mAP50–95 in this run. One seed, repeated signs and source dependence, and dominant-target original annotations limit interpretation. The test provides an evaluation under the inherited protocol. New-site and exhaustive multi-object evaluation remain future work. [Recorded experiment](model-source/configs/expanded-experiment.json).

The earlier recorded demonstration shows Original. The current app offers Original, Expanded + synthetic, and Video refresh; the metrics card identifies the study for the active checkpoint. Independent evaluation of the added locations requires separate held-out data.

## Classifier pilot and examples

The two MobileNetV3 classifiers and their Grad-CAM measurements come from the earlier 569-image pilot. That rerun achieved 94/96 clean predictions and macro-F1 0.9816. All nine new classifier arm/seed trials selected epoch zero, preserving the starting A1 classifier. These classifier measurements describe a separate task and experiment from the current detector.

The public examples and replay contain labeled synthetic composites. The Experiments tab contains aggregate held-out measurements. Camera and file inference runs on the device in the browser. The private coursework package holds original street photographs, videos, extracted frames, annotations and per-image logs.

## Source and license

[Model source and export instructions](model-source/README.md) cover both detector experiments and the historical classifier. The editable checkpoints and official initialization checkpoint are included. Full training uses the user's private data ZIP and a private output directory. The source distribution contains generic code and aggregate records.

SignLab application code and the bundled Ultralytics detector are distributed under AGPL-3.0. Third-party components retain their licenses and notices. [License](LICENSE), [third-party notices](THIRD_PARTY_NOTICES.md).
