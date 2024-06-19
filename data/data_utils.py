import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, ImageNet


AVAILABLE_DATASETS = ["mnist", "cifar10", "imagenet"]


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


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
        if ds_class is ImageNet:
            self.ds = ds_class(root, transform=transform)
        else:
            self.ds = ds_class(root, train=True, download=True, transform=transform)

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
