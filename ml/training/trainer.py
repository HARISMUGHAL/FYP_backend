import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy

def compute_class_weights(train_loader, num_classes):
    class_counts = [0] * num_classes
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1
            
    total = sum(class_counts)
    class_weights = []
    for count in class_counts:
        weight = total / (num_classes * count) if count > 0 else 0.0
        class_weights.append(weight)
        
    return torch.tensor(class_weights, dtype=torch.float)

def train_model(model, train_loader, val_loader, config, class_names):
    device = torch.device("cpu")
    model.to(device)
    
    epochs = config['training']['epochs']
    lr = config['training']['lr']
    patience = config['training']['patience']
    save_path = config['model']['save_path']
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    class_weights = compute_class_weights(train_loader, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
        train_loss = running_loss / total_samples
        train_acc = running_corrects.double() / total_samples
        
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_samples += inputs.size(0)
                
        val_loss = val_loss / val_samples
        val_acc = val_corrects.double() / val_samples
        
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
            print("Saved new best model.")
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break
            
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model
