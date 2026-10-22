"""
Camera Calibration & 6-DOF Perspective-n-Point (solvePnP) Pose Estimation Module.
"""
from typing import List, Tuple, Optional
import numpy as np
import cv2


class CameraCalibrator:
    """Handles intrinsic camera calibration and 6-DOF pose estimation."""

    def __init__(self, pattern_size: Tuple[int, int] = (9, 6), square_size_mm: float = 25.0):
        self.pattern_size = pattern_size
        self.square_size = square_size_mm
        # Prepare 3D object points in physical board space (Z = 0)
        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = (
            np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size_mm
        )

    def calibrate(self, calibration_images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calibrates camera using checkerboard images.
        
        Returns:
            K: Intrinsic camera matrix (3x3)
            dist: Distortion coefficients (5x1 or 8x1)
            mean_reprojection_error: Overall RMSE reprojection error in pixels
        """
        obj_points = []
        img_points = []
        img_shape = None

        for img in calibration_images:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            img_shape = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)

            if ret:
                obj_points.append(self.objp)
                corners_subpix = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                img_points.append(corners_subpix)

        if not obj_points:
            raise ValueError("No chessboard corners detected in the provided images.")

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_shape, None, None
        )

        # Compute Mean Reprojection Error
        total_error = 0.0
        total_points = 0
        for i in range(len(obj_points)):
            img_points_reproj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
            error = cv2.norm(img_points[i], img_points_reproj, cv2.NORM_L2)
            total_error += error ** 2
            total_points += len(obj_points[i])

        mean_reprojection_error = float(np.sqrt(total_error / total_points))
        return K, dist, mean_reprojection_error

    @staticmethod
    def estimate_6dof_pose(
        object_points_3d: np.ndarray,
        image_points_2d: np.ndarray,
        K: np.ndarray,
        dist: Optional[np.ndarray] = None,
        method: int = cv2.SOLVEPNP_ITERATIVE
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Estimates 6-DOF Rotation matrix (3x3) and Translation vector (3x1) via solvePnP.

        Args:
            object_points_3d: Coordinates in 3D object frame (N, 3)
            image_points_2d: Corresponding 2D pixel coordinates (N, 2)
            K: Intrinsic matrix (3x3)
            dist: Distortion coefficients
            method: OpenCV solvePnP flag

        Returns:
            R_mat: 3x3 Rotation matrix in SO(3)
            tvec: 3x1 Translation vector
            success: Boolean status flag
        """
        if dist is None:
            dist = np.zeros(5, dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            object_points_3d.astype(np.float32),
            image_points_2d.astype(np.float32),
            K.astype(np.float32),
            dist.astype(np.float32),
            flags=method
        )

        if not success:
            return np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32), False

        R_mat, _ = cv2.Rodrigues(rvec)
        return R_mat.astype(np.float32), tvec.astype(np.float32), True

    @staticmethod
    def compute_reprojection_error(
        object_points_3d: np.ndarray,
        image_points_2d: np.ndarray,
        R_mat: np.ndarray,
        tvec: np.ndarray,
        K: np.ndarray,
        dist: Optional[np.ndarray] = None
    ) -> float:
        """Calculates RMSE reprojection error for an estimated pose."""
        if dist is None:
            dist = np.zeros(5, dtype=np.float32)

        rvec, _ = cv2.Rodrigues(R_mat)
        projected, _ = cv2.projectPoints(
            object_points_3d.astype(np.float32),
            rvec,
            tvec.astype(np.float32),
            K.astype(np.float32),
            dist.astype(np.float32)
        )
        projected = projected.reshape(-1, 2)
        err = np.linalg.norm(image_points_2d.reshape(-1, 2) - projected, axis=1)
        return float(np.mean(err))
