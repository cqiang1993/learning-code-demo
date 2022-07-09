import torchvision
from torchvision import datasets, models, transforms
import os
from PIL import Image
from torch.utils.data import Dataset,DataLoader


class MyDataset(Dataset):
    def __init__(self, annotations, img_dir, mode=None, target_transform=None):
        super().__init__()
        self.img_labels = annotations
        self.img_dir = img_dir
        if mode == 'train':
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(p=.25),
                transforms.RandomVerticalFlip(p=.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
        elif mode == 'test':
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
        elif mode == 'val':
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
        self.transform = preprocess
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir[idx])
        with Image.open(img_path) as im:
            image = im
            label = self.img_labels.iloc[idx]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label
