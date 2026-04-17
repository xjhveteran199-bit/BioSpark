"""Grad-CAM (Gradient-weighted Class Activation Mapping) for 1D biosignal CNNs.

Hooks into the last convolutional layer of PyTorch models, computes gradients
of the target class w.r.t. the feature maps, and produces a heatmap showing
which signal regions the CNN attends to for its prediction.

References:
    Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization"
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional


class GradCAM1D:
    """Grad-CAM for 1D convolutional neural networks.

    Attaches forward/backward hooks to the target convolutional layer,
    captures feature maps and their gradients, then computes the
    gradient-weighted activation map.
    """

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.model.eval()

        # Auto-detect the last Conv1d layer if not specified
        if target_layer is None:
            target_layer = self._find_last_conv(model)
        if target_layer is None:
            raise ValueError("No Conv1d layer found in model.")

        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _find_last_conv(self, model: torch.nn.Module) -> Optional[torch.nn.Module]:
        """Walk module tree and return the last Conv1d layer."""
        last_conv = None
        for module in model.modules():
            if isinstance(module, torch.nn.Conv1d):
                last_conv = module
        return last_conv

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> dict:
        """Compute Grad-CAM heatmap for a single input.

        Args:
            input_tensor: Shape (1, C, L) — single sample, C channels, L time steps.
            target_class: Class index to explain. If None, uses the predicted class.

        Returns:
            Dict with keys:
                heatmap: np.ndarray of shape (L,) — normalized 0..1 attention weights
                predicted_class: int
                confidence: float
                probabilities: np.ndarray of shape (n_classes,)
                feature_map_size: int — spatial size of the last conv feature map
        """
        self.model.eval()

        # Enable gradient computation for this forward pass
        input_tensor = input_tensor.clone().requires_grad_(True)

        # Forward
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1).detach().numpy()[0]
        pred_class = int(np.argmax(probs))

        if target_class is None:
            target_class = pred_class

        # Backward for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM computation
        # gradients: (1, n_filters, spatial)
        # activations: (1, n_filters, spatial)
        gradients = self._gradients[0]    # (n_filters, spatial)
        activations = self._activations[0]  # (n_filters, spatial)

        # Global average pooling of gradients → per-filter weights
        weights = gradients.mean(dim=1)  # (n_filters,)

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1], dtype=torch.float32)  # (spatial,)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU — only keep positive contributions
        cam = F.relu(cam)

        # Upsample to input signal length
        input_length = input_tensor.shape[-1]
        cam_np = cam.numpy()

        if len(cam_np) < input_length:
            # Linear interpolation to match input length
            from numpy import interp
            x_cam = np.linspace(0, input_length - 1, len(cam_np))
            x_full = np.arange(input_length)
            heatmap = interp(x_full, x_cam, cam_np)
        else:
            heatmap = cam_np[:input_length]

        # Normalize to 0..1
        hm_min, hm_max = heatmap.min(), heatmap.max()
        if hm_max - hm_min > 1e-8:
            heatmap = (heatmap - hm_min) / (hm_max - hm_min)
        else:
            heatmap = np.zeros_like(heatmap)

        return {
            "heatmap": heatmap,
            "predicted_class": pred_class,
            "target_class": target_class,
            "confidence": float(probs[pred_class]),
            "probabilities": probs,
            "feature_map_size": int(activations.shape[1]),
        }

    def cleanup(self):
        """Remove hooks from the model."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def compute_gradcam_for_segments(
    model: torch.nn.Module,
    segments: list[np.ndarray],
    in_channels: int = 1,
    target_class: Optional[int] = None,
    max_segments: int = 50,
) -> list[dict]:
    """Run Grad-CAM on multiple segments.

    Args:
        model: PyTorch model with Conv1d layers.
        segments: List of numpy arrays, each shape (time,) or (channels, time).
        in_channels: Number of input channels for the model.
        target_class: Class to explain (None = predicted class per segment).
        max_segments: Cap on number of segments to process.

    Returns:
        List of dicts, each containing heatmap, prediction info, and signal data.
    """
    gradcam = GradCAM1D(model)
    results = []

    for i, seg in enumerate(segments[:max_segments]):
        # Prepare input tensor
        if in_channels > 1 and seg.ndim == 2:
            x = torch.FloatTensor(seg).unsqueeze(0)  # (1, channels, time)
        else:
            x = torch.FloatTensor(seg).unsqueeze(0).unsqueeze(0)  # (1, 1, time)

        result = gradcam.generate(x, target_class=target_class)
        result["segment_idx"] = i
        result["signal"] = seg.flatten() if seg.ndim == 1 else seg[0]  # first channel for display
        result["signal"] = result["signal"].tolist()
        result["heatmap"] = result["heatmap"].tolist()
        result["probabilities"] = result["probabilities"].tolist()
        results.append(result)

    gradcam.cleanup()
    return results
