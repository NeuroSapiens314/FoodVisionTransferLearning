import torch
from pathlib import Path
import data_setup
import model
import utils
import argparse

def main():
    parser = argparse.ArgumentParser(description="Make predictions on images using trained model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    data_path = Path("data")
    train_dir = data_path / "train"
    
    class_names = [d.name for d in train_dir.iterdir() if d.is_dir()]
    
    model_instance, weights = model.create_effnetb0_model(
        num_classes=len(class_names),
        device=device
    )
    
    model_instance.load_state_dict(torch.load(args.model_path, map_location=device))
    
    transform = data_setup.get_transforms()
    
    utils.pred_and_plot(
        model=model_instance,
        image_path=args.image_path,
        class_names=class_names,
        transform=transform,
        device=device
    )

if __name__ == "__main__":
    main() 