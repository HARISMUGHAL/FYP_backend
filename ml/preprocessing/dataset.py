import os
import glob
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

class FruitDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = []
        self._scan_dataset()
        
    def _scan_dataset(self):
        classes_set = set()
        for fruit_name in os.listdir(self.data_dir):
            fruit_path = os.path.join(self.data_dir, fruit_name)
            if not os.path.isdir(fruit_path) or fruit_name.lower() == "real":
                continue
            for grade in os.listdir(fruit_path):
                grade_path = os.path.join(fruit_path, grade)
                if not os.path.isdir(grade_path):
                    continue
                label_name = f"{fruit_name}_{grade}"
                classes_set.add(label_name)
                
                # Find images
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
                    for img_path in glob.glob(os.path.join(grade_path, ext)):
                        self.image_paths.append(img_path)
                        self.labels.append(label_name)
                        
        self.class_names = sorted(list(classes_set))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_name = self.labels[idx]
        label_idx = self.class_to_idx[label_name]
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        return image, label_idx

class DatasetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        real_idx = self.subset.indices[index]
        img_path = self.subset.dataset.image_paths[real_idx]
        label_name = self.subset.dataset.labels[real_idx]
        label_idx = self.subset.dataset.class_to_idx[label_name]
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label_idx
        
    def __len__(self):
        return len(self.subset)

def get_dataloaders(data_dir, img_size=224, batch_size=32, val_split=0.2, num_workers=0):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = FruitDataset(data_dir, transform=None)
    
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_wrapper = DatasetWrapper(train_dataset, transform=train_transform)
    val_wrapper = DatasetWrapper(val_dataset, transform=val_transform)
    
    train_loader = DataLoader(train_wrapper, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_wrapper, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, full_dataset.class_names
