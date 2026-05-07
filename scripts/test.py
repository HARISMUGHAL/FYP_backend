import os
import sys
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.models.fruit_model import build_model
from ml.utils.helpers import load_class_mapping, load_yaml

def predict_image(image_path, model, transform, class_names):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
        
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze().tolist()
        
    # Safely handle single vs multiple class outputs
    if not isinstance(probabilities, list):
        probabilities = [probabilities]
        
    probs_with_classes = [(class_names[i], prob) for i, prob in enumerate(probabilities)]
    probs_with_classes.sort(key=lambda x: x[1], reverse=True)
    
    top_label, top_prob = probs_with_classes[0]
    
    return {
        "label": top_label,
        "confidence": top_prob
    }

def main():
    test_dir = "test_images"
    if not os.path.exists(test_dir):
        print(f"Directory {test_dir} not found.")
        return
        
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}")
        return
        
    config = load_yaml(config_path)
    
    try:
        class_names = load_class_mapping()
    except FileNotFoundError:
        print("Class mapping not found. Please run scripts/train.py first.")
        return
        
    model_path = config.get('model', {}).get('save_path', "ml/models/saved/best_model.pth")
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please run scripts/train.py first.")
        return
        
    model = build_model(
        num_classes=len(class_names),
        pretrained=False,
        dropout_rate=config.get('model', {}).get('dropout_rate', 0.3)
    )
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    
    img_size = config.get('dataset', {}).get('img_size', 224)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        print(f"No images found in {test_dir}")
        return

    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        result = predict_image(img_path, model, transform, class_names)
        
        if result is None:
            continue
            
        label = result.get("label", "unknown")
        confidence = result.get("confidence", 0.0)
        
        # Parse label (e.g., apple_A -> fruit=apple, grade=A)
        parts = label.split("_")
        fruit = parts[0]
        
        grade = None
        if len(parts) > 1:
            grade = parts[1]
            
        status = "KNOWN"
        
        # Unknown handling
        if confidence < 0.65:
            fruit = "unknown"
            grade = None
            status = "UNKNOWN"
            
        print("-" * 34)
        print(f"Image: {img_file}")
        print(f"Fruit: {fruit}")
        print(f"Grade: {grade}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Status: {status}")
        print("-" * 34)

if __name__ == "__main__":
    main()
