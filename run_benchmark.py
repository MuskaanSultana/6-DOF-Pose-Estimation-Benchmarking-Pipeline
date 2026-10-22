"""
End-to-end pipeline demo: synthetic dataset -> PyTorch baseline -> ONNX export
-> benchmark both backends with MLflow logging -> report latency delta.

Run: python run_benchmark.py
"""
import time
import numpy as np
import cv2
import torch

from onnx_optimizer import DummyPoseModel, export_pytorch_to_onnx, ONNXInferenceEngine
from benchmarking_harness import MLflowBenchmarker
from stress_augmentor import StressTestAugmentor

IMG_SIZE = 64
NUM_SAMPLES = 150
WARMUP_ITERS = 10
MESH_POINTS = 200


def build_synthetic_dataset(n=NUM_SAMPLES):
    dataset = []
    mesh = np.random.uniform(-1, 1, (MESH_POINTS, 3)).astype(np.float32)
    for _ in range(n):
        img = np.random.randint(0, 255, (3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        img = StressTestAugmentor.apply_numpy_fallback(img.transpose(1, 2, 0), glare=True, noise_std=10.0)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0

        rvec_gt = (np.random.randn(3) * 0.3).astype(np.float32)
        R_gt, _ = cv2.Rodrigues(rvec_gt)
        t_gt = np.random.uniform(-50, 50, (3, 1)).astype(np.float32)

        dataset.append({"img": img, "gt_R": R_gt.astype(np.float32), "gt_t": t_gt, "mesh": mesh})
    return dataset


class TorchModelRunner:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict(self, img_np):
        with torch.no_grad():
            x = torch.from_numpy(img_np).unsqueeze(0)
            out = self.model(x).squeeze(0).numpy()
        return _decode_pose(out)


class ONNXModelRunner:
    def __init__(self, engine):
        self.engine = engine

    def predict(self, img_np):
        out = self.engine.predict(img_np[None, ...].astype(np.float32))[0]
        return _decode_pose(out)


def _decode_pose(out):
    rvec, tvec = out[:3].astype(np.float32), out[3:].astype(np.float32)
    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float32), tvec.reshape(3, 1)


def main():
    print(f"Building synthetic dataset ({NUM_SAMPLES} samples, {IMG_SIZE}x{IMG_SIZE})...")
    dataset = build_synthetic_dataset()

    model = DummyPoseModel()
    bench = MLflowBenchmarker(experiment_name="6dof_pose_demo")

    print("\n[1/3] Benchmarking PyTorch baseline (CPU)...")
    pt_runner = TorchModelRunner(model)
    for _ in range(WARMUP_ITERS):
        pt_runner.predict(dataset[0]["img"])
    summary_pt = bench.run_benchmark(pt_runner, dataset, device="pytorch_cpu")

    print("[2/3] Exporting PyTorch model to ONNX...")
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    export_pytorch_to_onnx(model, dummy_input, "model.onnx")
    engine = ONNXInferenceEngine("model.onnx", preferred_provider="CUDAExecutionProvider")
    active_provider = engine.session.get_providers()[0]

    print(f"[3/3] Benchmarking ONNX Runtime ({active_provider})...")
    onnx_runner = ONNXModelRunner(engine)
    for _ in range(WARMUP_ITERS):
        onnx_runner.predict(dataset[0]["img"])
    summary_onnx = bench.run_benchmark(onnx_runner, dataset, device=active_provider)

    reduction_pct = (
        (summary_pt["p99_latency_ms"] - summary_onnx["p99_latency_ms"]) / summary_pt["p99_latency_ms"] * 100.0
    )

    print("\n===== Benchmark Summary =====")
    print(f"{'Metric':<22}{'PyTorch':>15}{'ONNX Runtime':>18}")
    print(f"{'P50 latency (ms)':<22}{summary_pt['p50_latency_ms']:>15.3f}{summary_onnx['p50_latency_ms']:>18.3f}")
    print(f"{'P99 latency (ms)':<22}{summary_pt['p99_latency_ms']:>15.3f}{summary_onnx['p99_latency_ms']:>18.3f}")
    print(f"{'Mean ADD-S':<22}{summary_pt['mean_adds_mm']:>15.3f}{summary_onnx['mean_adds_mm']:>18.3f}")
    print(f"{'Mean rot err (deg)':<22}{summary_pt['mean_rot_error_deg']:>15.3f}{summary_onnx['mean_rot_error_deg']:>18.3f}")
    print(f"\nP99 latency change (ONNX vs PyTorch): {reduction_pct:+.1f}%")
    print("\nNote: model weights are untrained/random; accuracy metrics (ADD-S, rotation/translation")
    print("error) are smoke-test values, not real pose-estimation accuracy. Latency numbers are real")
    print("measurements of this machine's PyTorch vs ONNX Runtime inference path.")


if __name__ == "__main__":
    main()
