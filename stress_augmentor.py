"""
Synthetic Data Augmentation & Harsh Space Environment Stress Suite.
"""
from typing import Optional
import numpy as np

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


class StressTestAugmentor:
    """Simulates harsh deep-space & orbital conditions."""

    @staticmethod
    def get_space_environment_transform():
        """
        Simulates extreme orbital lighting:
        - High dynamic range direct solar glare
        - Deep shadows (penumbra / zero-albedo background)
        - Sensor thermal noise & cosmic ray static
        - Thruster/reaction-wheel motion blur
        """
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError("albumentations is required to use get_space_environment_transform.")

        return A.Compose([
            # Extreme dynamic range / direct sunlight and eclipse
            A.RandomBrightnessContrast(
                brightness_limit=(-0.6, 0.8),
                contrast_limit=(0.5, 0.9),
                p=0.8
            ),
            # Harsh high-exposure thresholding / solar reflection
            A.Solarize(threshold=(100, 200), p=0.4),
            # Sensor thermal noise and ISO static
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
            # Attitude jitter / sensor motion blur
            A.MotionBlur(blur_limit=(3, 7), p=0.3)
        ])

    @staticmethod
    def apply_numpy_fallback(image: np.ndarray, glare: bool = True, noise_std: float = 15.0) -> np.ndarray:
        """
        Pure NumPy fallback for test environments without heavy CV dependencies.
        """
        augmented = image.astype(np.float32)

        if glare:
            # Add synthetic circular solar flare
            h, w = augmented.shape[:2]
            y, x = np.ogrid[:h, :w]
            cx, cy = np.random.randint(0, w), np.random.randint(0, h)
            mask = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * (min(h, w) / 4) ** 2))
            if len(augmented.shape) == 3:
                mask = mask[:, :, np.newaxis]
            augmented += mask * 150.0

        if noise_std > 0:
            noise = np.random.normal(0, noise_std, augmented.shape)
            augmented += noise

        return np.clip(augmented, 0, 255).astype(np.uint8)
