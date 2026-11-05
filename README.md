# SignLab

SignLab detects four traffic-sign classes with YOLO11n and classifies images with MobileNetV3. The classifier shows a Grad-CAM overlay. Both modes measure browser inference latency and report the browser and platform.

Run the app from this directory with `python3 -m http.server 8000` and open [localhost:8000](http://localhost:8000). Select an example, choose a local file, or enable the camera. The camera needs browser permission and a secure context such as localhost or HTTPS.

The deployed app is available at https://ddigreen.github.io/signlab-cv-a2/.

GitHub Pages serves this directory directly as static files. Keep the assets in their current folders so models and the bundled ONNX Runtime resolve under the repository path. The `.nojekyll` file enables direct static publishing. [GitHub Pages documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages).

The four sample inputs and replay use synthetic sign composites. The Experiments tab shows measurements from the separate held-out evaluation. Camera and file inference runs on the device in the browser.

See the bundled notices for the software, models, and generated examples. [Third-party notices](THIRD_PARTY_NOTICES.md).

## Source and license

SignLab application code and the bundled Ultralytics detector are distributed under AGPL-3.0. The license text is in [LICENSE](LICENSE). Third-party components retain their licenses and notices.

[Model source and export instructions](model-source/README.md) include training and export code, configs, and the editable checkpoints used for the shipped models. Re-exporting the models works with the supplied synthetic example. Training requires locally provided images and labels.

The public sample images and replay contain labeled synthetic composites. Original street photos, raw videos, extracted frames, private annotations, and coursework downloads are kept in the separate private coursework package.
