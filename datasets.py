import os
from PIL import Image
from torch.utils.data import Dataset

class ImageSet(Dataset):
    def __init__(self, dir, transform=None):
        self.paths = [
            os.path.join(root, name)
            for root, dirs, files in os.walk(dir)
            for name in files
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        image = self.transform(image)
        return image