from pathlib import Path

import pandas as pd
from PIL import Image

class ChestXray14():
    """This is the ChestX-ray 14 data set.

    See: https://arxiv.org/abs/1705.02315,
    https://www.kaggle.com/nih-chest-xrays/data
    """

    labels = [
        "Cardiomegaly",
        "Emphysema",
        "Effusion",
        "No Finding",
        "Hernia",
        "Infiltration",
        "Mass",
        "Nodule",
        "Atelectasis",
        "Pneumothorax",
        "Pleural_Thickening",
        "Pneumonia",
        "Fibrosis",
        "Edema",
        "Consolidation",
    ]

    def __init__(self, root, *args, **kwargs):
        self.data_dir = Path(root)

    def load(self):
        self.train_labels_path = self.data_dir / "Data_Entry_2017.csv"
        df = pd.read_csv(self.train_labels_path)
        # Add disease labels as columns instead of string-based "Disease1|Disease2".
        for label in self.labels:
            df.loc[:, label] = self._get_disease_label(df, label)
        df.drop("Finding Labels", axis=1, inplace=True)
        df.loc[:, "Path"] = df["Image Index"].apply(self._get_image_path)
        self.df = df

    def load_image(self, path: str) -> Image:
        """Cache and load an image."""
        return Image.open(path).convert("RGB")

    def _get_disease_label(self, df: pd.DataFrame, disease: str) -> pd.Series:
        return df["Finding Labels"].apply(lambda label: 1 if disease in label else 0)

    def _get_image_path(self, path: str) -> str:
        path = Path(path)
        major, minor = list(map(int, path.stem.split("_")))
        if major < 1336: dir_number = "01"
        elif major < 3923 and minor < 14: dir_number = "02"
        elif major < 6585 and minor < 7: dir_number = "03"
        elif major < 9232 and minor < 4: dir_number = "04"
        elif major < 11558 and minor < 8: dir_number = "05"
        elif major < 13774 and minor < 26: dir_number = "06"
        elif major < 16051: dir_number = "07"
        elif major < 18387 and minor < 35: dir_number = "08"
        elif major < 20945 and minor < 50: dir_number = "09"
        elif major < 24718: dir_number = "10"
        elif major < 28173 and minor < 3: dir_number = "11"
        else: dir_number = "12"
        return self.data_dir / f"images_0{dir_number}/images" / path


def read_split(dataset: ChestXray14, path: str, split: str) -> str:
    """Use the split in https://stanfordmedicine.app.box.com/s/b3gk9qnanzrdocqge0pbuh07mreu5x7y/folder/49785298071"""
    df = pd.read_csv(path)
    df.drop(['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia',
   'Infiltration', 'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax',
   'Pleural-Thickening', 'Pneumonia', 'Fibrosis', 'Edema',
   'Consolidation'], axis=1, inplace=True)
    df.loc[:, "Path"] = df["Path"].apply(lambda x: x.split("/")[-1])
    df.loc[:, "Split"] = split
    df = pd.merge(df, dataset.df, left_on="Path", right_on="Image Index", how="left", suffixes=("_x", ""))
    df.drop("Path_x", inplace=True, axis=1)
    return df
