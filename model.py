import torch
import torchvision

def create_effnetb0_model(num_classes: int, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)
    
    for param in model.features.parameters():
        param.requires_grad = False
    
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(
            in_features=1280,
            out_features=num_classes,
            bias=True
        )
    ).to(device)
    
    return model, weights 