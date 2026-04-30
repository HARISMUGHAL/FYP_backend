import os
import sys
import yaml
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.models.fruit_model import build_model
from ml.utils.helpers import load_class_mapping

def predict_image(image_path, config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    try:
        class_names = load_class_mapping()
    except FileNotFoundError:
        print("Class mapping not found. Please run scripts/train.py first.")
        return None
        
    model_path = config['model']['save_path']
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please run scripts/train.py first.")
        return None
        
    model = build_model(
        num_classes=len(class_names),
        pretrained=False,
        dropout_rate=config['model']['dropout_rate']
    )
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    
    img_size = config['dataset']['img_size']
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
        
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze().tolist()
        
    if not isinstance(probabilities, list):
        probabilities = [probabilities]
        
    probs_with_classes = [(class_names[i], prob) for i, prob in enumerate(probabilities)]
    probs_with_classes.sort(key=lambda x: x[1], reverse=True)
    
    top_label, top_prob = probs_with_classes[0]
    top_3 = [{"label": label, "prob": round(prob, 4)} for label, prob in probs_with_classes[:3]]
    
    result = {
        "label": top_label,
        "confidence": round(top_prob, 4),
        "top_3": top_3
    }
    
    return result

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test.py <image_path>")
        return
        
    image_path = sys.argv[1]
    result = predict_image(image_path)
    if result:
        import json
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
