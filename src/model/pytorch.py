import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger


class MultiClassClassifier(pl.LightningModule):
    def __init__(self, seed, input_dim=4, hidden_dim=5, learning_rate=1e-3, max_learning_rate=0.1, total_steps=100):
        super().__init__()
        self.save_hyperparameters()
        pl.seed_everything(seed)
        self.layer_1 = torch.nn.Linear(self.hparams.input_dim, self.hparams.hidden_dim)
        self.layer_2 = torch.nn.Linear(self.hparams.hidden_dim, 3)

    def forward(self, x):
        x = self.layer_1(x)
        x = torch.relu(x)
        x = self.layer_2(x)
        x = torch.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = OneCycleLR(optimizer, self.hparams.max_learning_rate, self.hparams.total_steps)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        self.log("test_loss", loss)

    def predict(self, x):
        self.eval()
        logits = self.forward(self.df_to_tensor(x))
        self.train()
        return torch.argmax(logits, dim=1).detach().numpy()

    def df_to_tensor(self, df, target_col=None, format=np.float32):
        if target_col is not None:
            feature_cols = [col for col in df.columns if col != target_col]
            tensor = TensorDataset(
                torch.tensor(df[feature_cols].values.astype(format)),
                torch.tensor(df[target_col].values),
            )
        else:
            tensor = torch.tensor(df.values.astype(format))
        return tensor

    def tensor_to_loader(self, tensor, batch_size, num_workers, shuffle=True):
        loader = DataLoader(dataset=tensor, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        return loader

    def setup_trainer(self, experiment, run_id, max_epochs):
        mlf_logger = MLFlowLogger(experiment_name=experiment)
        mlf_logger._run_id = run_id
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[EarlyStopping(monitor="val_loss"), LearningRateMonitor(logging_interval="epoch")],
            logger=mlf_logger,
        )
        return trainer
