# 6-DOF Pose Estimation — Automated Benchmarking Pipeline

Camera calibration, 6-DOF pose estimation, and an automated benchmarking/export
pipeline for perception models. Built with Python, OpenCV, PyTorch, ONNX
Runtime, and MLflow; containerized for reproducible headless execution.

## What it does

1. **Camera calibration & pose estimation** (`calibration_engine.py`)
   Checkerboard-based intrinsic calibration (`cv2.calibrateCamera`) and 6-DOF
   pose recovery via `solvePnP` (rotation matrix + translation vector from
   3D-2D point correspondences), with reprojection-error reporting.

2. **Stress testing** (`stress_augmentor.py`)
   Augmentation suite (brightness/contrast extremes, solar-glare-style glare,
   sensor noise, motion blur) for probing model robustness under harsh
   lighting and simulated sensor noise. Falls back to a pure-NumPy
   implementation when `albumentations` isn't available.

3. **Benchmarking harness** (`benchmarking_harness.py`)
   ADD-S (average closest-point distance for symmetric objects), SO(3)
   geodesic rotation error, translation L2 error, and latency percentiles
   (P50/P99), with optional MLflow experiment logging.

4. **ONNX export & inference** (`onnx_optimizer.py`)
   PyTorch → ONNX export with dynamic batch axes, and an ONNX Runtime
   inference engine that prefers CUDA and falls back to CPU automatically.

5. **End-to-end demo** (`run_benchmark.py`)
   Wires all of the above together: builds a synthetic dataset, benchmarks
   a PyTorch model, exports it to ONNX, benchmarks the ONNX Runtime version,
   and reports the latency delta between the two backends.

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` installs plain `onnxruntime` (CPU). On a CUDA machine,
swap in `onnxruntime-gpu` instead.

## Usage

Run the test suite:

```bash
pytest -v tests/test_pipeline.py
```

Run the end-to-end benchmark demo:

```bash
python run_benchmark.py
```

This generates a synthetic dataset and prints a latency/accuracy comparison
between the PyTorch and ONNX Runtime backends. The model weights are
untrained, so the accuracy numbers (ADD-S, rotation/translation error) are
smoke-test values, not real pose-estimation accuracy — the latency
comparison is a genuine measurement on whatever machine it runs on.

## Docker

```bash
docker build -t pose-pipeline .
docker run --rm pose-pipeline
```

Runs the test suite headlessly inside the container (see `Dockerfile`).

## CI

`.github/workflows/benchmark.yml` runs the test suite and the end-to-end
benchmark demo on every push/PR to `main`/`develop`.

## Layout

```
calibration_engine.py     # intrinsic calibration + solvePnP
stress_augmentor.py       # lighting/noise/blur stress augmentations
benchmarking_harness.py   # ADD-S, rotation/translation error, MLflow logging
onnx_optimizer.py         # PyTorch -> ONNX export + ONNX Runtime inference
run_benchmark.py          # end-to-end demo tying the pipeline together
tests/test_pipeline.py    # math/accuracy test suite
Dockerfile                # headless reproducible container
.github/workflows/        # CI
```
