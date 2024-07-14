import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10, ImageNet, ImageFolder
from PIL import Image
from pathlib import Path

import torch.nn as nn
import torchvision.transforms as T


AVAILABLE_DATASETS = ["mnist", "cifar10", "imagenet"]


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class RetDictDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        return {"img": img, "label": label}


class ExperimentDataModule:
    def __init__(
        self,
        dataset_name,
        root,
        image_size,
        random_horizonal_flip=False,
        batch_size=1,
        num_workers=0,
        **kwargs,
    ):
        self.dataset_name = dataset_name
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        ds_class = self.get_dataset()
        transform = T.Compose(
            [
                T.Resize(image_size),
                T.RandomHorizontalFlip() if random_horizonal_flip else nn.Identity(),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )

        val_transform = T.Compose(
            [T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor()]
        )

        if ds_class is ImageNet:
            self.train_ds = ds_class(root, transform=transform)
            self.val_ds = ds_class(root, split="val", transform=val_transform)
        else:
            self.train_ds = ds_class(
                root, train=True, download=True, transform=transform
            )
            self.val_ds = ds_class(
                root, train=False, download=True, transform=val_transform
            )
        self.train_ds = RetDictDataset(self.train_ds)
        self.val_ds = RetDictDataset(self.val_ds)

    def get_dataset(self):
        assert (
            self.dataset_name in AVAILABLE_DATASETS
        ), f"Dataset {self.dataset_name} not available. Available datasets: {AVAILABLE_DATASETS}"
        if self.dataset_name == "mnist":
            return MNIST
        elif self.dataset_name == "cifar10":
            return CIFAR10
        elif self.dataset_name == "imagenet":
            return ImageNet

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
            pin_memory=True,
        )
        dl = cycle(dl)
        return dl
class LatentImageNetDataset(Dataset):
    def __init__(self, root):
        self.imagenet_ds = ImageNet(root, split="train")
        imgs = self.imagenet_ds.imgs
        latent_paths = [Path(img[0].replace("train", "latent_train")) for img in imgs]
        latent_paths = [str(p.parent / (p.stem + ".pth")) for p in latent_paths]
        self.ds = [(img[0], l_path, img[1]) for img, l_path in zip(imgs, latent_paths)]
        self.transform = T.Compose([T.Resize(256), T.CenterCrop(256), T.ToTensor()])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        path, latent_path, label = self.ds[idx]
        # img = self.transform(Image.open(path))
        # if img.shape[0] == 1:
        #     img = img.repeat(3, 1, 1)
        # elif img.shape[0] == 4:
        #     img = img[:3]
        latent = torch.load(latent_path)
        return {"img": latent, "label": label}


class LatentImageFolderDataset(Dataset):
    def __init__(self, root):
        self.image_ds = ImageFolder(root)
        imgs = self.image_ds.imgs
        latent_paths = [Path(img[0].replace("train", "latent_train")) for img in imgs]
        latent_paths = [str(p.parent / (p.stem + ".pth")) for p in latent_paths]
        self.ds = [(img[0], l_path, img[1]) for img, l_path in zip(imgs, latent_paths)]
        self.transform = T.Compose([T.Resize(256), T.CenterCrop(256), T.ToTensor()])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        path, latent_path, label = self.ds[idx]
        # img = self.transform(Image.open(path))
        # if img.shape[0] == 1:
        #     img = img.repeat(3, 1, 1)
        # elif img.shape[0] == 4:
        #     img = img[:3]
        latent = torch.load(latent_path)
        return {"img": latent, "label": label}


class LatentImageDataModule:
    def __init__(self, root, dataset_name, batch_size=1, num_workers=0, **kwargs):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        if dataset_name == "imagenet":
            self.ds = LatentImageNetDataset(root)
        elif dataset_name == "imagefolder":
            self.ds = LatentImageFolderDataset(root)

    def get_dataloader(self):
        dl = DataLoader(
            self.ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        dl = cycle(dl)
        return dl
