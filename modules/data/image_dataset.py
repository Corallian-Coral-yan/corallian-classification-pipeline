# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

import os
import pandas as pd
from pathlib import PureWindowsPath, PurePosixPath
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, train=False, transform=None, target_transform=None, random_state=1, verbose=False):
        self.annotations_file = annotations_file
        self.img_dir = img_dir
        self.verbose = verbose
        self.transform = transform
        self.target_transform = target_transform
        self.le = LabelEncoder()

        raw_labels = pd.read_csv(annotations_file)

        # Temporarily handle incorrectly sized crops by dropping them
        raw_labels = raw_labels.drop(
            raw_labels[(raw_labels["width"] != 500) | (raw_labels["height"] != 500)].index
        )

        # label encode categorical labels
        self.le.fit(raw_labels["annotation"])
        raw_labels["annotation"] = self.le.transform(raw_labels["annotation"])

        # paths in annotations file are encoded in windows format
        # change them to posix if necessary
        if os.name != "nt":
            raw_labels["filepath"] = raw_labels["filepath"].apply(
                lambda path: str(PurePosixPath(*PureWindowsPath(path).parts))
            )

        if train:
            self.img_labels = raw_labels.sample(frac=0.8,random_state=random_state)
        else:
            self.img_labels = raw_labels.drop(
                raw_labels.sample(frac=0.8,random_state=random_state).index
            )

        self._print(f"Annotations File: {annotations_file}")
        self._print(f"Image Base Directory: {img_dir}")
        self._print(f"Train={train}, Transform={transform}, Target Transform={target_transform}, Random State={random_state}")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def _print(self, *args, **kwargs):
        if self.verbose == True:
            print(*args, **kwargs)
        