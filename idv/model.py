from pathlib import Path

import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics.functional import auroc

from idv import densenet


class Model(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=None)
        parser.add_argument("--optim", type=str, default="adam")
        parser.add_argument("--scheduler", type=str, default=None)
        parser.add_argument("--patience", type=int, default=10)
        parser.add_argument("--pos_weight", type=int, default=None, nargs="*")
        parser.add_argument("--pretrained", default=False, action="store_true")
        parser.add_argument("--dropout", default=0, type=float)
        parser.add_argument("--ood", default=False, action="store_true")
        parser.add_argument("--ood_weight", default=0.5, type=float)
        parser.add_argument("--model_id", default=None, type=str)
        parser.add_argument("--model_checkpoint", default=None, type=str)
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        """Pass the ArgParser's args to the constructor."""
        result = pl.utilities.argparse.from_argparse_args(cls, args, **kwargs)
        # HACK: Log all Hyperparameters
        params = vars(args)
        params.update(**kwargs)
        result._hparams_initial = params
        return result

    def __init__(
        self,
        learning_rate=1e-3,
        weight_decay=None,
        optim="adam",
        scheduler=None,
        pos_weight=None,
        patience=10,
        pretrained=False,
        dropout=0,
        ood=False,
        ood_weight=0.5,
        model_id=None,
        model_checkpoint=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = 14
        if pos_weight is not None:
            pos_weight = torch.Tensor(pos_weight)
        if ood:
            self.loss = CombinedLoss(pos_weight, ood_weight)
        else:
            self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self._load_model()

    def _load_model(self):

        return_features = (
            self._hparams_initial["model_id"] is not None
            and self._hparams_initial["model_checkpoint"] is not None
        )
        model = densenet.densenet121(
            pretrained=self.hparams.pretrained,
            drop_rate=self.hparams.dropout,
            return_features=return_features,
        )
        kernel_count = model.classifier.in_features
        if return_features:
            # Add an OOD Detection Head to a pre-trained model
            root = (
                Path("/space/wollek/virtualenvs/idv/.guild/runs")
                / self._hparams_initial["model_id"]
                / "experiments/run/version_0/checkpoints"
            )
            checkpoint = torch.load(
                root / self._hparams_initial["model_checkpoint"],
                map_location=torch.device("cpu"),
            )
            renamed_state_dict = {
                key[len("model.") :]: value
                for key, value in checkpoint["state_dict"].items()
            }
            model.classifier = torch.nn.Linear(kernel_count, self.num_classes)
            model.load_state_dict(renamed_state_dict)
            self.model = PostOODClassifier(model)

        else:
            self.model = model
            self.model.classifier = torch.nn.Linear(kernel_count, self.num_classes)

    def forward(self, images):
        if True: #self._hparams_initial["option"] == "tencrop":
            # 10 crops per image
            batch_size, n_crops, channels, height, width = images.size()
            all_images = images.view(-1, channels, height, width)
            all_y_hat = self.model(all_images)
            y_hat = all_y_hat.view(batch_size, n_crops, -1).mean(1)
        else:
            y_hat = self.model(images)
        return y_hat

    def _step(self, batch, batch_idx):
        images, y, indices = batch
        inside_distribution_mask = y[:, -1] == 1
        if self._hparams_initial["option"] == "tencrop":
            # 10 crops per image
            batch_size, n_crops, channels, height, width = images.size()
            all_images = images.view(-1, channels, height, width)
            all_y_hat, all_y_ood = self.model(all_images)
            y_pred = all_y_hat.view(batch_size, n_crops, -1).mean(1)
            ood_pred = all_y_ood.view(batch_size, n_crops, -1).mean(1)
        else:
            y_pred, ood_pred = self.model(images)
        loss = self.loss(
            y[inside_distribution_mask][:, : self.num_classes],
            y_pred[inside_distribution_mask],
            y[:, -1].unsqueeze(-1),
            ood_pred,
        )
        probs = torch.sigmoid(y_pred[:, : self.num_classes]).detach()
        all_probs = torch.sigmoid(y_pred).detach()
        ood_probs = torch.sigmoid(ood_pred).detach()
        return {
            "loss": loss,
            "probs": probs,
            "labels": y[:, : self.num_classes],
            "logits": y_pred.detach(),
            "y": y,
            "all_probs": all_probs,
            "ood_probs": ood_probs,
            "indices": indices,
        }

    def training_step(self, batch, batch_idx):
        if self.hparams.ood:
            return self._step(batch, batch_idx)
        images, y, indices = batch
        y_hat = self(images)
        loss = self.loss(y_hat, y)
        probs = torch.sigmoid(y_hat).detach()
        return {"loss": loss, "probs": probs, "labels": y, "logits": y_hat.detach()}

    def training_step_end(self, outputs):
        self.log(
            "loss/train", outputs["loss"], prog_bar=True, on_step=True, on_epoch=True
        )
        return outputs

    def training_epoch_end(self, outputs):
        probs = torch.vstack([o["probs"] for o in outputs])
        labels = torch.vstack([o["labels"] for o in outputs])
        for index, label in enumerate(self.trainer.datamodule.data.labels):
            if label == "InsideDistribution":
                continue
            try:
                self.log(
                    f"{label}/train",
                    auroc(probs[:, index], labels[:, index].int()),
                    on_epoch=True,
                )
            except ValueError:
                print(
                    f"Could not calculate the AUC for {label} with {labels[index].sum().item()}"
                )
        if self.hparams.ood:
            self.evaluate_ood("train", outputs)

    def validation_step(self, batch, batch_idx):
        if self.hparams.ood:
            return self._step(batch, batch_idx)
        images, y, indices = batch
        y_hat = self(images)
        loss = self.loss(y_hat, y)
        probs = torch.sigmoid(y_hat).detach()
        return {
            "loss": loss,
            "probs": probs,
            "labels": y,
            "indices": indices,
            "logits": y_hat.detach(),
        }

    def evaluate_ood(self, stage, outputs):
        try:
            probs = torch.vstack([o["probs"] for o in outputs])
            all_probs = torch.vstack([o["all_probs"] for o in outputs])
            ood_probs = torch.vstack([o["ood_probs"] for o in outputs])
            labels = torch.vstack([o["labels"] for o in outputs])
            indices = torch.hstack([o["indices"] for o in outputs])
            y = torch.vstack([o["y"] for o in outputs])
            try:
                self.log(
                    f"OOD/{stage}",
                    auroc(ood_probs[:, 0], y[:, -1].int()),
                    on_epoch=True,
                )
            except ValueError as e:
                print(
                    f"Could not calculate the AUC for OOD with {labels[-1].sum().item()}"
                )
                print(e)
        except IndexError:
            import pdb

            pdb.set_trace()

    def validation_step_end(self, outputs):
        self.log(
            "loss/val", outputs["loss"], prog_bar=True, on_step=True, on_epoch=True
        )
        return outputs

    def validation_epoch_end(self, outputs):
        probs = torch.vstack([o["probs"] for o in outputs])
        labels = torch.vstack([o["labels"] for o in outputs])
        indices = torch.hstack([o["indices"] for o in outputs])
        logits = torch.vstack([o["logits"] for o in outputs])
        for index, label in enumerate(self.trainer.datamodule.data.labels):
            if label == "InsideDistribution":
                continue
            try:
                self.log(
                    f"{label}/val",
                    auroc(probs[:, index], labels[:, index].int()),
                    on_epoch=True,
                )
            except ValueError as e:
                print(
                    f"Could not calculate the AUC for {label} with {labels[index].sum().item()}"
                )
            except IndexError:
                import pdb

                pdb.set_trace()

        if self.hparams.ood:
            self.evaluate_ood("val", outputs)
            all_probs = torch.vstack([o["all_probs"] for o in outputs])
            ood_probs = torch.vstack([o["ood_probs"] for o in outputs])
            y = torch.vstack([o["y"] for o in outputs])
            self.log_prediction(
                probs.cpu(),
                labels.cpu(),
                indices.cpu(),
                ood_probs.cpu(),
                y.cpu(),
                all_probs.cpu(),
                logits.cpu(),
            )
        else:
            self.log_prediction(
                probs.cpu(), labels.cpu(), indices.cpu(), logits=logits.cpu()
            )

    def log_prediction(
        self,
        probs,
        labels,
        indices,
        ood_probs=None,
        y=None,
        all_probs=None,
        logits=None,
        filename=None,
    ):

        checkpoint = f"epoch={self.current_epoch}-step={self.global_step}"
        checkpoint_dir = Path(self.logger.log_dir) / "checkpoints" / checkpoint

        Path.mkdir(checkpoint_dir, parents=True, exist_ok=True)
        filename = checkpoint_dir / ("predictions.pt" if filename is None else filename)

        print(f"Saving predictions in {filename}")
        if not filename.exists():
            torch.save(
                {
                    "predictions": probs,
                    "labels": labels,
                    "indices": indices,
                    "logits": logits,
                    "ood_probs": ood_probs,
                    "y": y,
                    "all_probs": all_probs,
                },
                filename,
            )

    def test_step(self, batch, batch_idx):
        images, y, indices = batch
        y_hat = self(images)
        loss = self.loss(y_hat, y)
        probs = torch.sigmoid(y_hat).detach()
        return {
            "loss": loss,
            "probs": probs,
            "labels": y,
            "logits": y_hat.detach(),
            "indices": indices,
        }

    def test_step_end(self, outputs):
        self.log(
            "loss/test", outputs["loss"], prog_bar=True, on_step=True, on_epoch=True
        )
        return outputs

    def test_epoch_end(self, outputs):
        probs = torch.vstack([o["probs"] for o in outputs]).cpu()
        labels = torch.vstack([o["labels"] for o in outputs]).cpu()
        logits = torch.vstack([o["logits"] for o in outputs]).cpu()
        indices = torch.hstack([o["indices"] for o in outputs]).cpu()
        for index, label in enumerate(self.trainer.datamodule.data.labels):
            try:
                self.log(
                    f"{label}/test",
                    auroc(probs[:, index], labels[:, index].int()),
                    on_epoch=True,
                )
            except ValueError:
                continue
        self.log_prediction(
            probs,
            labels,
            indices,
            logits=logits,
            filename=f"test.pt",
        )

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        if self.hparams.optim == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
            )

        if self.hparams.scheduler is not None:
            if self.hparams.scheduler == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, "min", verbose=True, patience=self.hparams.patience
                )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "loss/val_epoch"},
            }
        return optimizer


class CombinedLoss(torch.nn.Module):
    def __init__(self, pos_weight=None, ood_weight=0.5):
        super().__init__()
        self.class_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.ood_loss = torch.nn.BCEWithLogitsLoss()
        self.ood_weight = ood_weight

    def forward(self, y_true, y_pred, ood_true, ood_pred):
        # If a batch contains only OOD samples then the class loss would be NaN otherwise.
        class_loss = self.class_loss(y_pred, y_true) if y_true.nelement() != 0 else 0
        ood_loss = self.ood_loss(ood_pred, ood_true)
        return (1 - self.ood_weight) * class_loss + self.ood_weight * ood_loss
