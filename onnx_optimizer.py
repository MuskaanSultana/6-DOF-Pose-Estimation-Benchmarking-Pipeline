"""
ONNX Model Export, Optimization, and Inference Execution Engine.
Supports CUDA, TensorRT, and CPU Execution Providers.
"""
from typing import Optional, List
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


class DummyPoseModel(nn.Module if TORCH_AVAILABLE else object):
    """Reference lightweight convolutional backbone for 6-DOF direct regression."""
    def __init__(self):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DummyPoseModel.")
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        # Outputs: 6-DOF (3 for rotation vector / representation + 3 for translation)
        self.fc = nn.Linear(16 * 4 * 4, 6)

    def forward(self, x):
        feats = self.features(x)
        feats = feats.view(feats.size(0), -1)
        return self.fc(feats)


def export_pytorch_to_onnx(
    pytorch_model,
    dummy_input,
    output_onnx_path: str,
    opset_version: int = 14
):
    """
    Exports PyTorch model to ONNX with dynamic batch dimensions.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to export ONNX models.")

    pytorch_model.eval()
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['output_pose'],
        dynamic_axes={
            'input_image': {0: 'batch_size'},
            'output_pose': {0: 'batch_size'}
        },
        dynamo=False
    )


class ONNXInferenceEngine:
    """ONNX Runtime Inference Engine with provider selection."""

    def __init__(
        self,
        onnx_path: str,
        preferred_provider: str = "CUDAExecutionProvider"
    ):
        if not ORT_AVAILABLE:
            raise ImportError("onnxruntime is required for ONNXInferenceEngine.")

        available_providers = ort.get_available_providers()
        selected_providers = []
        if preferred_provider in available_providers:
            selected_providers.append(preferred_provider)
        selected_providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(onnx_path, providers=selected_providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_tensor_np: np.ndarray) -> np.ndarray:
        """
        Runs forward inference. Expects input shaped (B, C, H, W).
        """
        if len(input_tensor_np.shape) == 3:
            input_tensor_np = np.expand_dims(input_tensor_np, axis=0)

        outputs = self.session.run([self.output_name], {self.input_name: input_tensor_np})
        return outputs[0]
