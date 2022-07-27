from pathlib import Path

import yaml
import torch

from idv.model import Model
from idv import datamodule

ROOT = Path(__file__).parent.parent


class Experiment:
    def __init__(
        self,
        model=str,
        hparams=None,
    ):
        self.root = ROOT / "models"
        self.load_hparams(hparams=hparams)
        self.load_model(model=model)

    def load_hparams(self, hparams=None):
        with open(self.root / hparams) as f:
            self.hparams = yaml.load(
                f, Loader=yaml.FullLoader
            )  # use full loader to disable unsafe warning

    def load_model(self, model):
        self.checkpoint = torch.load(self.root / model, map_location=torch.device("cpu"))
        self.load_data()
        self.model = Model(**self.hparams)
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.eval()

    def load_data(self):
        if not hasattr(self, "data"):
            self.data = datamodule.DataModule(**self.hparams)
            self.data.setup(stage=None)
