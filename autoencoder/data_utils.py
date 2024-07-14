import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageNet


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class ImageNetDataModule:
    def __init__(
        self,
        root,
        image_size,
        random_horizonal_flip=False,
        batch_size=1,
        num_workers=0,
        **kwargs
    ):
        train_transform = T.Compose(
            [
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.RandomHorizontalFlip() if random_horizonal_flip else nn.Identity(),
                T.ToTensor(),
                T.Lambda(lambda x: x * 2 - 1),
            ]
        )
        val_transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2 - 1),
        
        ])
        self.train_ds = ImageNet(root=root, transform=train_transform)
        self.val_ds = ImageNet(root=root, split="val", transform=val_transform)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_train_dataloader(self):
        dl = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        dl = cycle(dl)
        return dl
    
    def get_val_dataloader(self):
        dl = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
        return dl