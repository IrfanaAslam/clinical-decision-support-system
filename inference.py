# inference.py
import torch
import torch.nn.functional as F
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

CLASS_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
SEVERITY_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#c0392b"]


@torch.inference_mode()
def predict_image(model, image_tensor, device):
    logits = model(image_tensor.to(device))
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    return pred_idx, confidence, probs


def get_gradcam_visualization(model, image_tensor, original_image, target_layer_name=None):
    """
    Returns superimposed Grad-CAM image (numpy array 0-255)
    """
    if target_layer_name is None:
        # Try to find a reasonable target layer automatically
        target_layers = [model.backbone.norm_pre] if hasattr(model.backbone, "norm_pre") else \
                        [model.backbone.stages[-1]]   # fallback
    else:
        # You would need to implement proper layer access
        target_layers = [model.backbone.norm_pre]

    try:
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=image_tensor, targets=None)
        cam_image = show_cam_on_image(
            original_image.astype(np.float32) / 255.0,
            grayscale_cam[0],
            use_rgb=True,
            colormap=cv2.COLORMAP_JET
        )
        return (cam_image * 255).astype(np.uint8)
    except Exception as e:
        print(f"Grad-CAM failed: {str(e)}")
        return None