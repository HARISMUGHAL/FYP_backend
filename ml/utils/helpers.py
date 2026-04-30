import os
import json

def save_class_mapping(class_names, save_path="ml/models/saved/class_mapping.json"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(class_names, f, indent=4)

def load_class_mapping(load_path="ml/models/saved/class_mapping.json"):
    with open(load_path, 'r') as f:
        class_names = json.load(f)
    return class_names
