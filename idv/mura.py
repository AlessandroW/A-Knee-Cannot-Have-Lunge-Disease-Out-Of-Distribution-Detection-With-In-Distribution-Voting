from PIL import Image
import pandas as pd

from ciip_dataloader import Dataset


class Mura(Dataset):
    """This is the Mura Data set.

    The data set contains upper body bone x-rays.
    The original task was to predict signs of abnormality.
    See: https://stanfordmlgroup.github.io/competitions/mura/
    """

    labels = ["boneage", "Male"]

    def __init__(self, root=None, cache_dir=None, verbose=False):
        super().__init__(root, cache_dir=cache_dir, verbose=verbose)
        self.data_dir = self.root / "nonchest-xray-data/MURA"

    def load(self):
        self.train_labels_path = self._cache(self.data_dir / "MURA-v1.1/train_labeled_studies.csv")
        self.test_labels_path = self._cache(self.data_dir / "MURA-v1.1/valid_labeled_studies.csv")

        self.train_images_path = self._cache(self.data_dir / "MURA-v1.1/train_image_paths.csv")
        self.test_images_path = self._cache(self.data_dir / "MURA-v1.1/valid_image_paths.csv")

        self._read_data()

    def _get_image_path(self, path):
        return self.data_dir / path

    def _merge_labels_and_paths(self, train=True):
        """Merge the image and label csvs.

        The one file contains all image filepaths.
        The other file study directories and labels.
        """
        if train:
            labels_path = self.train_labels_path
            images_path = self.train_images_path
        else:
            labels_path = self.test_labels_path
            images_path = self.test_images_path
        df = pd.read_csv(images_path, header=0, names=["Path"])
        df.loc[:, "Path"] = df.Path.apply(self._get_image_path)
        df.loc[:, "Directory"] = df.Path.apply(lambda path: str(path.parent))
        df_labels = pd.read_csv(labels_path, header=0, names=["Directory", "Label"])
        df_labels.loc[:, "Directory"] = df_labels.Directory.apply(
            lambda path: str(self._get_image_path(path)))
        return df.merge(df_labels, on="Directory")

    def _read_data(self):
        self.df_train = self._merge_labels_and_paths(train=True)
        self.df_test = self._merge_labels_and_paths(train=False)
        self.df = pd.concat([self.df_train, self.df_test], ignore_index=True, join="inner")

    def load_image(self, path: str) -> Image:
        """Cache and load an image."""
        return Image.open(self._cache(path)).convert("RGB")
