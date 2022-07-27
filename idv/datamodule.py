from typing import Callable, Optional, List, Tuple
from pathlib import Path
import random

import pytorch_lightning as pl
import torch
from torchvision import transforms as T
import pandas as pd
import numpy as np

try:
    # Internal data loader with caching.
    # Dataset source code is inside this directory.
    from ciip_dataloader.chestxray14 import ChestXray14
    from ciip_dataloader import irma
    from ciip_dataloader import rsna_bone_age, chexpert, mura, imagenet
    ROOT = Path(__file__).parent.parent
    CHESTXRAY14 = ChestXray14(cache_dir="/space/wollek")
    TRAIN_SPLIT = ROOT / "data/train.csv"
    VAL_SPLIT = ROOT / "data/valid.csv"

except ImportError:
    print("CIIP DATALOADER NOT FOUND")


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, load_image: Callable, transforms: Optional[Callable] = None):
        self.df = df
        self.load_image = load_image
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        target = self.df.iloc[index]
        img = self.load_image(target.Path)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index


class Transforms():

    def __init__(self, labels: List[str], size=512, **kwargs):
        imagenet_mean = [0.485, 0.456, 0.406, ]
        imagenet_std = [0.229, 0.224, 0.225, ]
        self.labels = labels

        normalize = T.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        self.transform = T.Compose([
            T.Resize(256),
            T.TenCrop(224),
            T.Lambda(lambda crops: torch.stack(
                [T.ToTensor()(crop) for crop in crops])),
            T.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))])

    def __call__(self, input, target):
        return self.transform(input), self.target_transform(target)

    def target_transform(self, target):
        return target[self.labels].astype('float32').values


class DataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("DataModule")
        parser.add_argument("--num_workers", type=int, default=32)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--pin_memory", default=False, action="store_true")
        parser.add_argument("--size", type=int, default=512)
        parser.add_argument("--sample", default=None, type=int)
        parser.add_argument("--imagenet", default=False, action="store_true")
        parser.add_argument("--ood_nofinding",
                            default=False, action="store_true")
        parser.add_argument("--outlier_exposure",
                            default=False, action="store_true")
        parser.add_argument("--exposure_dataset", default="irma")
        parser.add_argument("--irma_length",
                            default=False, action="store_true")
        parser.add_argument("--cxr14_length",
                            default=False, action="store_true")
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        """Pass the ArgParser's args to the constructor."""
        return pl.utilities.argparse.from_argparse_args(cls, args, **kwargs)

    def __init__(self, batch_size=8, num_workers=32, pin_memory=True, size=512,
                 sample=None, original_labels=False,
                 imagenet=False, ood_nofinding=False, outlier_exposure=False,
                 exposure_dataset="irma", irma_length=False, cxr14_length=False, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.size = size
        self.sample = sample
        self.imagenet = imagenet
        self.ood_nofinding = ood_nofinding
        self.outlier_exposure = outlier_exposure
        self.exposure_dataset = exposure_dataset
        self.irma_length = irma_length
        self.cxr14_length = cxr14_length

    def setup(self, stage=None):
        CHESTXRAY14.load()
        self.data = CHESTXRAY14
        CHESTXRAY14.read_original_labels()
        self.df_train = CHESTXRAY14.df_train
        self.df_val = CHESTXRAY14.df_val
        self.df_test = CHESTXRAY14.df_test

        if "No Finding" in self.data.labels:
            self.data.labels.remove("No Finding")

        if self.sample is not None:
            self.df_train = self.df_train.sample(self.sample)
            self.df_val = self.df_val.sample(self.sample)

        if self.ood_nofinding:
            if "No Finding" not in self.df_train:
                self.df_train.loc[:, "No Finding"] = self.df_train.loc[:, self.data.labels].max(
                    axis=1).apply(lambda x: 1 if x == 0 else 0) - 1
            if "No Finding" not in self.df_val:
                self.df_val.loc[:, "No Finding"] = self.df_val.loc[:, self.data.labels].max(
                    axis=1).apply(lambda x: 1 if x == 0 else 0)
            assert "No Finding" in self.df_train
            assert "No Finding" in self.df_val
            self.df_train.loc[:, "InsideDistribution"] = self.df_train.loc[:, "No Finding"].apply(
                lambda x: 1 if x == 0 else 0)
            self.df_val.loc[:, "InsideDistribution"] = self.df_val.loc[:, "No Finding"].apply(
                lambda x: 1 if x == 0 else 0)

            class id_data():
                def __init__(self, labels):
                    self.labels = labels

            self.id_data = id_data(self.data.labels.copy())
            self.data.labels.append("InsideDistribution")

        if self.outlier_exposure:
            self.load_irma()
            irma_train_length = len(self.irma_train_df)
            irma_val_length = len(self.irma_val_df)
            if self.exposure_dataset == "irma":
                self.load_irma()
                self.df_train = pd.concat([self.df_train, self.irma_train_df])
                self.df_val = pd.concat([self.df_val, self.irma_val_df])
            elif self.exposure_dataset == "boneage":
                self.load_boneage()
                if self.irma_length:
                    self.boneage_train_df = self.boneage_train_df.sample(irma_train_length, random_state=0)
                    self.boneage_val_df = self.boneage_val_df.sample(irma_val_length, random_state=0)
                print(len(self.boneage_train_df), len(self.irma_train_df))
                self.df_train = pd.concat([self.df_train, self.boneage_train_df])
                self.df_val = pd.concat([self.df_val, self.boneage_val_df])
            elif self.exposure_dataset == "mura":
                self.load_mura()
                if self.irma_length:
                    self.mura_train_df = self.mura_train_df.sample(irma_train_length, random_state=0)
                    self.mura_val_df = self.mura_val_df.sample(irma_val_length, random_state=0)
                print(len(self.mura_train_df), len(self.irma_train_df))
                self.df_train = pd.concat([self.df_train, self.mura_train_df])
                self.df_val = pd.concat([self.df_val, self.mura_val_df])
            elif self.exposure_dataset == "imagenet":
                self.load_imagenet()
                if self.irma_length:
                    self.imagenet_train_df = self.imagenet_train_df.sample(irma_train_length, random_state=0)
                    self.imagenet_val_df = self.imagenet_val_df.sample(irma_val_length, random_state=0)
                elif self.cxr14_length:
                    self.imagenet_train_df = self.imagenet_train_df.sample(len(self.df_train), random_state=0)
                    self.imagenet_val_df = self.imagenet_val_df.sample(len(self.df_val), random_state=0)
                print(len(self.imagenet_train_df))
                self.df_train = pd.concat([self.df_train, self.imagenet_train_df])
                self.df_val = pd.concat([self.df_val, self.imagenet_val_df])
            elif self.exposure_dataset == "imagenet_boneage":
                self.load_imagenet()
                self.load_boneage()
                if self.cxr14_length:
                    self.imagenet_train_df = self.imagenet_train_df.sample(len(self.df_train) - len(self.boneage_train_df), random_state=0)
                    self.imagenet_val_df = self.imagenet_val_df.sample(len(self.df_val) - len(self.boneage_val_df), random_state=0)
                print(len(self.imagenet_train_df), len(self.irma_train_df))
                print("CXR Size (train, val)", len(self.df_train), len(self.df_val))
                self.df_train = pd.concat([self.df_train, self.imagenet_train_df, self.boneage_train_df])
                self.df_val = pd.concat([self.df_val, self.imagenet_val_df, self.boneage_val_df])
                print("Total Size (train, val)", len(self.df_train), len(self.df_val))
            elif self.exposure_dataset == "imagenet_irma":
                self.load_imagenet()
                if self.irma_length:
                    self.imagenet_train_df = self.imagenet_train_df.sample(irma_train_length // 2, random_state=0)
                    self.imagenet_val_df = self.imagenet_val_df.sample(irma_val_length // 2, random_state=0)
                    self.irma_train_df = self.irma_train_df.sample(irma_train_length - irma_train_length // 2, random_state=0)
                    self.irma_val_df = self.irma_val_df.sample(irma_val_length - irma_val_length // 2, random_state=0)
                elif self.cxr14_length:
                    self.imagenet_train_df = self.imagenet_train_df.sample(len(self.df_train) - irma_train_length, random_state=0)
                    self.imagenet_val_df = self.imagenet_val_df.sample(len(self.df_val) - irma_val_length, random_state=0)
                print(len(self.imagenet_train_df), len(self.irma_train_df))
                self.df_train = pd.concat([self.df_train, self.imagenet_train_df, self.irma_train_df])
                self.df_val = pd.concat([self.df_val, self.imagenet_val_df, self.irma_val_df])
            elif self.exposure_dataset == "all":
                self.load_irma()
                self.load_boneage()
                self.load_mura()
                self.load_imagenet()
                self.df_train = pd.concat([self.df_train, self.imagenet_train_df, self.irma_train_df, self.boneage_train_df, self.mura_train_df])
            else:
                raise ValueError(f"{self.exposure_dataset} is not available, only [irma, boneage, mura, imagenet_irma, all]")

    def load_irma(self):
        self.irma = irma.Irma(cache_dir="/space/wollek", chest=False)
        self.irma.load()
        self.irma.df.loc[:, self.data.labels] = np.zeros((len(self.irma.df),
                                                          len(self.data.labels)))
        train = self.irma.df.sample(frac=0.5, random_state=0)
        val = train.sample(frac=0.2, random_state=0)
        self.irma_train_df = train.drop(val.index)
        self.irma_val_df = val
        self.irma_test_df = self.irma.df.drop(train.index)

    def load_boneage(self):
        self.load_irma()
        self.boneage = rsna_bone_age.RSNA_Bone_Age(cache_dir="/space/wollek")
        self.boneage.load()
        self.boneage.df.loc[:, self.data.labels] = np.zeros((len(self.boneage.df),
                                                             len(self.data.labels)))
        test = self.boneage.df.sample(len(self.irma_test_df), random_state=0)
        train = self.boneage.df.drop(test.index)
        val = train.sample(len(self.irma_val_df), random_state=0)
        self.boneage_test_df = test
        self.boneage_val_df = val
        self.boneage_train_df = train.drop(val.index)

    def load_mura(self):
        self.load_irma()
        self.mura = mura.Mura(cache_dir="/space/wollek")
        self.mura.load()
        self.mura.df.loc[:, self.data.labels] = np.zeros((len(self.mura.df),
                                                          len(self.data.labels)))
        test = self.mura.df.sample(len(self.irma_test_df), random_state=0)
        train = self.mura.df.drop(test.index)
        val = train.sample(len(self.irma_val_df), random_state=0)
        self.mura_test_df = test
        self.mura_val_df = val
        self.mura_train_df = train.drop(val.index)

    def load_chexpert(self, small=True):
        self.chexpert = chexpert.Chexpert(cache_dir="/space/wollek", version="small" if small else "original")
        self.chexpert.load()
        self.chexpert.df = self.chexpert.df_train.loc[self.chexpert.df_train['Frontal/Lateral'] == "Frontal"]

        missing_labels = [label for label in self.data.labels
                          if label not in self.chexpert.labels]
        self.chexpert.df.loc[:, missing_labels] = np.zeros((len(self.chexpert.df),
                                                            len(missing_labels)))
        # This data set is only used for testing.
        self.chexpert_test_df = self.chexpert.df.sample(len(self.df_test), random_state=0)

    def load_imagenet(self):
        self.load_irma()
        self.ImageNet = imagenet.ImageNet(cache_dir="/space/wollek")
        self.ImageNet.load()
        self.ImageNet.df_train.loc[:, self.data.labels] = np.zeros((len(self.ImageNet.df_train),
                                                                    len(self.data.labels)))
        train = self.ImageNet.df_train.sample(frac=0.5, random_state=0)
        val = train.sample(frac=0.2, random_state=0)
        self.imagenet_train_df = train.drop(val.index)
        self.imagenet_val_df = val
        self.imagenet_test_df = self.ImageNet.df_train.drop(train.index).sample(len(self.irma_test_df), random_state=0)

    def train_dataloader(self):
        self.train_transforms = Transforms(
            labels=self.data.labels, size=self.size)

        if self.imagenet:
            self.train_dataset = imagenet.Dataset(
                self.df_train, self.data, transforms=self.train_transforms)
        else:
            self.train_dataset = Dataset(
                self.df_train, self.data.load_image, self.train_transforms)

            if self.ood_nofinding:
                self.train_dataset.id_data = self.id_data

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self, df=None):
        self.val_transforms = Transforms(labels=self.data.labels, size=self.size, )
        if df is not None:
            df_val = df
        else:
            df_val = self.df_val
        if self.imagenet:
            self.val_dataset = imagenet.Dataset(
                self.df_val, self.data, transforms=self.val_transforms)
        else:
            self.val_dataset = Dataset(
                df_val, self.data.load_image, self.val_transforms)
            if self.ood_nofinding:
                self.val_dataset.id_data = self.id_data
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self, df=None, exclude_no_finding=False, additional_df=None):
        self.test_transforms = Transforms(labels=self.data.labels, size=self.size)
        if df is not None:
            df_test = df
        else:
            df_test = self.df_test
        if exclude_no_finding:
            df_test = df_test.loc[df_test[self.data.labels].sum(axis=1) > 0]
            print("No finding images: ", len(df_test.loc[df_test[self.data.labels].sum(axis=1) == 0]))
        if additional_df is not None:
            df_test = pd.concat([df_test, additional_df])
        self.test_dataset = Dataset(
            df_test, self.data.load_image, self.test_transforms)
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
