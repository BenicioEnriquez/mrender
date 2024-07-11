import os
import glob
from PIL import Image
from torch.utils.data import Dataset

class ImageSet(Dataset):
    def __init__(self, dir, transform=None):
        self.paths = [f for f in glob.glob(os.path.join(dir, '**/*'), recursive=True) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        image = self.transform(image)
        return image