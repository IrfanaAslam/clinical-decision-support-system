# model.py
import torch
import torch.nn as nn
import timm
self.backbone = timm.create_model(...)
class DRSeverityModel(nn.Module):
    """
    Advanced model using modern vision backbone (Swin / ConvNeXt / EfficientNetV2 / etc.)
    """
    def __init__(self, 
                 model_name="swin_large_patch4_window12_384",
                 num_classes=5,
                 pretrained=True,
                 dropout_rate=0.35):
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,           # remove classifier
            global_pool="avg"        # or "max" / ""
        )
        
        in_features = self.backbone.num_features
        
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def load_model(checkpoint_path="best_model.pth", device="cuda"):
    model = DRSeverityModel()
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Handle possible DataParallel / module. prefix
    if "module." in next(iter(state_dict.items()))[0]:
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)  # allow partial loading
    model.to(device)
    model.eval()
    return model