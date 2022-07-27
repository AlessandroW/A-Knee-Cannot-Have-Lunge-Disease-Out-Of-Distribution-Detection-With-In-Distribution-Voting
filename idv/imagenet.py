from pathlib import Path

import pandas as pd
from PIL import Image

from ciip_dataloader import Dataset


class ImageNet(Dataset):
    """This is the ImageNet data set.

    See: https://www.kaggle.com/c/imagenet-object-localization-challenge/overview
    """

    def __init__(self, root=None, cache_dir=None):
        """Abstraction of the ImageNet data set.

        Parameters
        ----------
        root : Optional(str/Path)
            Path to the share-all directory.

        cache_dir : Optional(str/Path)
            Path to the cache directory.
        """
        super().__init__(root, cache_dir=cache_dir)
        self.name = "ImageNet"
        self.data_dir = self.root / self.name

    def load(self):
        self._specify_paths()
        self._read_data()
        self.categories = self.df_mapping.Description.values

    def _specify_paths(self):
        """Set the paths to the relevant files and folders."""
        self.path_train_labels = self.data_dir / "LOC_train_solution.csv"
        self.path_val_labels = self.data_dir / "LOC_val_solution.csv"
        self.image_dir = self.data_dir / "ILSVRC/Data/CLS-LOC"

        """The mapping between the 1000 synset id and their descriptions.
        For example, Line 1 says n01440764 tench, Tinca tinca means this is class 1,
        has a synset id of n01440764, and it contains the fish tench"""
        self.path_synset = self.data_dir / "LOC_synset_mapping.txt"

    def _read_data(self):
        """Load the annotations into memory."""
        self.df_mapping = pd.read_csv(
            self._cache(self.path_synset),
            sep="(?<=n[0-9]{8}) ",
            names=["Label", "Description"],
            engine="python",
        )
        self.df_train = self._read_annotations(self.path_train_labels)
        self.df_val = self._read_annotations(self.path_val_labels)

    def _read_annotations(self, path):
        """Load and parse the training or validation annotations."""
        path = self._cache(path)
        df = pd.read_csv(path)
        df.loc[:, "PredictionString"] = df.loc[:, "PredictionString"].apply(
            self._parse_prediction_string
        )
        df["Path"] = df.loc[:, "ImageId"].apply(self._get_image_path)
        return df

    def _parse_prediction_string(self, prediction_string: str):
        """Parse prediction string (labels+locations).

        Returns
        -------
        List[Dict[Union[str, int]]]
            List of labels and locations (x_min, y_min, x_max, y_max)
        """
        elements = prediction_string.split()
        return [
            {
                "label": elements[index],
                "x_min": elements[index + 1],
                "y_min": elements[index + 2],
                "x_max": elements[index + 3],
                "y_max": elements[index + 4],
            }
            for index in range(0, len(elements), 5)
        ]

    def _get_image_path(self, image_id: str) -> str:
        """Return the path for an image id."""
        if "val" in image_id:
            return str(self.image_dir / "val" / f"{image_id}.JPEG")
        elif "test" in image_id:
            return str(self.image_dir / "test" / f"{image_id}.JPEG")
        folder = image_id.split("_")[0]
        return str(self.image_dir / "train" / folder / f"{image_id}.JPEG")

    def load_image(self, path):
        """Load an image."""
        return Image.open(self._cache(path)).convert("RGB")

    def get_label_description(self, label=None, index=None):
        """Return the description of the synset label id or the class index"""
        if label is not None:
            return self.df_mapping.loc[self.df_mapping.Label == label, "Description"]
        elif index is not None:
            return self.df_mapping.iloc[index].Description
        raise ValueError("You need to specify either the synset label id or the class")
