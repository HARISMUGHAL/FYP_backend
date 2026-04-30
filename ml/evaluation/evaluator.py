import torch
import numpy as np

def evaluate_model(model, val_loader, class_names):
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    
    num_classes = len(class_names)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            total_correct += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)
            
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
                
    acc = total_correct / total_samples
    print(f"Overall Accuracy: {acc:.4f}")
    
    print("\nPer-class Accuracy:")
    for i in range(num_classes):
        class_total = confusion_matrix[i, :].sum()
        if class_total > 0:
            class_acc = confusion_matrix[i, i] / class_total
            print(f"{class_names[i]}: {class_acc:.4f}")
        else:
            print(f"{class_names[i]}: No samples in validation set.")
            
    print("\nConfusion Matrix:")
    print(confusion_matrix)
    
    return acc, confusion_matrix
