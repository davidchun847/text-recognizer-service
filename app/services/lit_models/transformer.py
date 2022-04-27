import torch
import torch.nn as nn

try:
    import wandb
except ModuleNotFoundError:
    pass


from services.metrics import CharacterErrorRate
from .base import BaseLitModel
from utils import str_utils


class TransformerLitModel(BaseLitModel):  # pylint: disable=too-many-ancestors
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.

    The module must take x, y as inputs, and have a special predict() method.
    """

    def __init__(self, model, args=None):
        super().__init__(model, args)

        self.mapping = self.model.data_config["mapping"]
        inverse_mapping = {val: ind for ind, val in enumerate(self.mapping)}

        start_token = inverse_mapping[str_utils.get_special_char(str_utils.ESpecialChar.start)]
        end_token = inverse_mapping[str_utils.get_special_char(str_utils.ESpecialChar.end)]
        padding_token = inverse_mapping[str_utils.get_special_char(str_utils.ESpecialChar.pad)]

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=padding_token)

        ignore_tokens = [start_token, end_token, padding_token]
        self.val_cer = CharacterErrorRate(ignore_tokens)
        self.test_cer = CharacterErrorRate(ignore_tokens)

    def forward(self, x):
        return self.model.predict(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self.model(x, y[:, :-1])
        loss = self.loss_fn(logits, y[:, 1:])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self.model(x, y[:, :-1])
        loss = self.loss_fn(logits, y[:, 1:])
        self.log("val_loss", loss, prog_bar=True)

        pred = self.model.predict(x)
        pred_str = "".join(self.mapping[_] for _ in pred[0].tolist() if _ != 3)
        try:
            self.logger.experiment.log({"val_pred_examples": [wandb.Image(x[0], caption=pred_str)]})
        except AttributeError:
            pass
        self.val_cer(pred, y)
        self.log("val_cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)
        o_dict = {"val_loss": loss, "val_cer": self.val_cer}
        return o_dict

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        pred = self.model.predict(x)
        pred_str = "".join(self.mapping[_] for _ in pred[0].tolist() if _ != 3)
        try:
            self.logger.experiment.log({"test_pred_examples": [wandb.Image(x[0], caption=pred_str)]})
        except AttributeError:
            pass
        self.test_cer(pred, y)
        self.log("test_cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)
        o_dict = {
            "test_cer": self.test_cer,
        }
        return o_dict
