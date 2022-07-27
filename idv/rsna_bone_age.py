from pathlib import Path

from PIL import Image
import pandas as pd

from ciip_dataloader import Dataset

class RSNA_Bone_Age(Dataset):
    """This is the RSNA Bone Age Data set.

    The data set contains hand X-ray images of children.
    The Kaggle task was to predict their age.
    See: https://www.kaggle.com/kmader/rsna-bone-age
    """

    labels = ["boneage", "Male"]

    def __init__(self, root=None, cache_dir=None, verbose=False):
        super().__init__(root, cache_dir=cache_dir, verbose=verbose)
        self.data_dir = self.root / "nonchest-xray-data/rsna-bone-age"

    def load(self):
        self.train_labels_path = self._cache(self.data_dir / "boneage-training-dataset.csv")
        self.test_labels_path = self._cache(self.data_dir / "boneage-test-dataset.csv")
        self.train_image_dir = self.data_dir / "boneage-training-dataset/boneage-training-dataset"
        self.test_image_dir = self.data_dir / "boneage-test-dataset/boneage-test-dataset"
        self._read_data()

    def _get_image_path(self, id, train=True):
        if train:
            return self.train_image_dir / f"{id}.png"
        return self.test_image_dir / f"{id}.png"

    def _read_data(self):
        df_train = pd.read_csv(self.train_labels_path)
        df_train.loc[:, "Male"] = df_train["male"].apply(
            lambda sex: 0 if sex == "male" else 1)
        df_train.loc[:, "Path"] = df_train["id"].apply(self._get_image_path)
        self.df_train = df_train

        df_test = pd.read_csv(self.test_labels_path)
        df_test.loc[:, "Male"] = df_test["Sex"].apply(
            lambda sex: 0 if sex == "M" else 1)
        df_test.loc[:, "Path"] = df_test["Case ID"].apply(
            lambda id: self._get_image_path(id, train=False))

        self.df_test = df_test

        self.df = pd.concat([df_train, df_test], ignore_index=True, join="inner")

    def load_image(self, path: str) -> Image:
        """Cache and load an image."""
        return Image.open(self._cache(path)).convert("RGB")
