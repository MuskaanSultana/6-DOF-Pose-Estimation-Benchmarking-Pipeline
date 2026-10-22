import numpy as np
import cv2
import pytest
from calibration_engine import CameraCalibrator
from benchmarking_harness import compute_adds_metric, compute_rotation_geodesic_error


def test_camera_calibrator_properties():
    calibrator = CameraCalibrator(pattern_size=(9, 6), square_size_mm=25.0)
    assert calibrator.square_size == 25.0
    assert calibrator.pattern_size == (9, 6)
    assert calibrator.objp.shape == (54, 3)


def test_solvepnp_accuracy():
    # Setup standard synthetic pinhole camera
    K = np.array([
        [800.0, 0.0, 320.0],
        [0.0, 800.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    dist = np.zeros(5, dtype=np.float32)

    # 3D points on a planar target (OpenCV 5 SOLVEPNP_ITERATIVE requires >= 6 points)
    obj_pts = np.array([
        [0.0, 0.0, 0.0],
        [50.0, 0.0, 0.0],
        [0.0, 50.0, 0.0],
        [50.0, 50.0, 0.0],
        [25.0, 25.0, 10.0],
        [10.0, 40.0, 5.0]
    ], dtype=np.float32)

    R_gt = np.eye(3, dtype=np.float32)
    t_gt = np.array([[0.0], [0.0], [500.0]], dtype=np.float32)

    rvec_gt, _ = cv2.Rodrigues(R_gt)
    img_pts, _ = cv2.projectPoints(obj_pts, rvec_gt, t_gt, K, dist)
    img_pts = img_pts.reshape(-1, 2)

    # Solve PnP
    R_est, t_est, success = CameraCalibrator.estimate_6dof_pose(obj_pts, img_pts, K, dist)

    assert success is True
    np.testing.assert_almost_equal(t_est, t_gt, decimal=1)
    np.testing.assert_almost_equal(R_est, R_gt, decimal=2)

    # Check reprojection error is near 0 for clean synthetic data
    reproj_err = CameraCalibrator.compute_reprojection_error(obj_pts, img_pts, R_est, t_est, K, dist)
    assert reproj_err < 0.1


def test_compute_adds_metric():
    # Dense spherical mesh points to test nearest neighbor distance under pure translation
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 20)
    x = (np.outer(np.sin(phi), np.cos(theta))).flatten()
    y = (np.outer(np.sin(phi), np.sin(theta))).flatten()
    z = (np.outer(np.cos(phi), np.ones_like(theta))).flatten()
    mesh_points = np.stack([x, y, z], axis=-1).astype(np.float32)

    R_gt = np.eye(3, dtype=np.float32)
    t_gt = np.zeros((3, 1), dtype=np.float32)

    # Identical pose -> error should be 0.0
    err_zero = compute_adds_metric(R_gt, t_gt, R_gt, t_gt, mesh_points)
    assert pytest.approx(err_zero, abs=1e-5) == 0.0

    # Test that error is positive and non-zero under perturbation
    t_pred = np.array([[0.0], [0.0], [5.0]], dtype=np.float32)
    err_trans = compute_adds_metric(R_gt, t_pred, R_gt, t_gt, mesh_points)
    assert err_trans > 0.0


def test_rotation_geodesic_error():
    R1 = np.eye(3, dtype=np.float32)
    err_zero = compute_rotation_geodesic_error(R1, R1)
    assert pytest.approx(err_zero, abs=1e-4) == 0.0

    # 90 degree rotation about Z axis
    R_90z = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0]
    ], dtype=np.float32)
    err_90 = compute_rotation_geodesic_error(R_90z, R1)
    assert pytest.approx(err_90, abs=1e-2) == 90.0
