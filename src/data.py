from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import numpy as np
import os

# our imagenet has those bad files.
def get_bad_files(data_dir):
    # known bad files.
    if "train" in data_dir:
        return {
            os.path.join(data_dir, "n02233338/n02233338_20142.JPEG"),
            os.path.join(data_dir, "n02233338/n02233338_3078.JPEG"),
            os.path.join(data_dir, "n02233338/n02233338_5273.JPEG"),
            os.path.join(data_dir, "n03792782/n03792782_15679.JPEG"),
            os.path.join(data_dir, "n03792782/n03792782_23966.JPEG"),
        }
    return set()


def get_dataloader(data_dir,
                   batch_size,
                   indices_path=None,
                   shuffle=True,
                   num_workers=4,
                   return_dataset=False, transform=None):

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    dataset = ImageFolder(
        root=data_dir,
        transform=transform
    )

    bad_files = get_bad_files(data_dir)

    if len(bad_files) > 0:
        # normalize paths to avoid mismatch
        bad_files = {os.path.abspath(p) for p in bad_files}

        filtered_samples = [
            (p, l)
            for (p, l) in dataset.samples
            if os.path.abspath(p) not in bad_files
        ]

        removed = len(dataset.samples) - len(filtered_samples)

        dataset.samples = filtered_samples
        dataset.imgs = filtered_samples

        print(f"Filtered out {removed} corrupted images")

    if indices_path is not None:
        indices = np.load(indices_path)
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    if return_dataset:
        return loader, dataset

    return loader