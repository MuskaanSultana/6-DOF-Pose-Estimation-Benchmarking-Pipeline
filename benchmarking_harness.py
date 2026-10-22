"""
Automated Benchmarking Harness & MLflow Metric Tracking Module.
Computes ADD-S (Average Distance for Symmetric Objects), SO(3) Geodesic Rotation Errors,
and Translation L2 norms across device backends.
"""
from typing import Dict, List, Any, Optional
import time
import numpy as np

try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def compute_adds_metric(
    pred_R: np.ndarray,
    pred_t: np.ndarray,
    gt_R: np.ndarray,
    gt_t: np.ndarray,
    object_mesh_points: np.ndarray
) -> float:
    """
    Computes ADD-S (Average Distance for Symmetric Objects):
    ADD-S = (1 / m) * sum_{x1 in M} min_{x2 in M} || (R_gt * x2 + t_gt) - (R_pred * x1 + t_pred) ||_2

    Args:
        pred_R: Predicted 3x3 rotation matrix
        pred_t: Predicted 3x1 translation vector
        gt_R: Ground truth 3x3 rotation matrix
        gt_t: Ground truth 3x1 translation vector
        object_mesh_points: (N, 3) 3D model vertices

    Returns:
        float: Mean closest-point distance in millimeters / coordinate units.
    """
    pred_points = (np.dot(pred_R, object_mesh_points.T) + pred_t.reshape(3, 1)).T
    gt_points = (np.dot(gt_R, object_mesh_points.T) + gt_t.reshape(3, 1)).T

    if SCIPY_AVAILABLE:
        # Fast O(N log N) k-d tree query
        tree = cKDTree(gt_points)
        distances, _ = tree.query(pred_points, k=1)
        return float(np.mean(distances))
    else:
        # Fallback pairwise distance
        distances = []
        for p_p in pred_points:
            min_d = np.min(np.linalg.norm(gt_points - p_p, axis=1))
            distances.append(min_d)
        return float(np.mean(distances))


def compute_rotation_geodesic_error(pred_R: np.ndarray, gt_R: np.ndarray) -> float:
    """
    Computes geodesic distance error on SO(3) in degrees:
    theta = arccos((trace(R_pred * R_gt^T) - 1) / 2)
    """
    R_diff = np.dot(pred_R, gt_R.T)
    tr = np.trace(R_diff)
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


class MLflowBenchmarker:
    """Benchmarking runner for logging performance metrics and distributions to MLflow."""

    def __init__(self, experiment_name: str = "6DOF_Pose_Estimation_Benchmarking"):
        self.experiment_name = experiment_name
        self.mlflow_available = False
        try:
            import mlflow
            self.mlflow = mlflow
            self.mlflow.set_experiment(experiment_name)
            self.mlflow_available = True
        except ImportError:
            self.mlflow = None

    def run_benchmark(
        self,
        model_runner: Any,
        test_dataset: List[Dict[str, Any]],
        device: str = "cpu"
    ) -> Dict[str, float]:
        """
        Executes benchmark over a dataset and reports metrics.
        """
        adds_errors = []
        rot_errors = []
        trans_errors = []
        latencies = []

        for sample in test_dataset:
            img = sample["img"]
            gt_R = sample["gt_R"]
            gt_t = sample["gt_t"]
            mesh_points = sample["mesh"]

            start_time = time.perf_counter()
            pred_R, pred_t = model_runner.predict(img)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            latencies.append(latency_ms)

            # Metric 1: ADD-S
            adds = compute_adds_metric(pred_R, pred_t, gt_R, gt_t, mesh_points)
            adds_errors.append(adds)

            # Metric 2: Translation Error (L2 Norm)
            t_err = float(np.linalg.norm(gt_t.flatten() - pred_t.flatten()))
            trans_errors.append(t_err)

            # Metric 3: Rotation Error (Geodesic Distance in SO(3))
            r_err = compute_rotation_geodesic_error(pred_R, gt_R)
            rot_errors.append(r_err)

        p99_latency = float(np.percentile(latencies, 99))
        p50_latency = float(np.percentile(latencies, 50))
        mean_adds = float(np.mean(adds_errors))
        mean_rot_err = float(np.mean(rot_errors))
        mean_trans_err = float(np.mean(trans_errors))

        summary = {
            "mean_adds_mm": mean_adds,
            "mean_rot_error_deg": mean_rot_err,
            "mean_trans_error_mm": mean_trans_err,
            "p50_latency_ms": p50_latency,
            "p99_latency_ms": p99_latency,
            "device": device
        }

        if self.mlflow_available:
            with self.mlflow.start_run():
                self.mlflow.log_param("device", device)
                self.mlflow.log_param("dataset_size", len(test_dataset))
                for k, v in summary.items():
                    if isinstance(v, (int, float)):
                        self.mlflow.log_metric(k, v)

        return summary
