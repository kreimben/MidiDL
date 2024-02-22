import lightning as L
import torch
from torch import nn


class MusicGeneratorModel(L.LightningModule):
    def __init__(self, hidden_size1: int, hidden_size2: int, learning_rate: float = 1e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self.generator = nn.Sequential(
            nn.Linear(81, hidden_size1),  # input dim is 81 notes on piano!
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(512, 81),  # output 81 notes on piano!
            nn.Tanh()  # Tanh activation to output values between -1 and 1
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        z, targets = batch
        generated_music = self(z)
        loss = self.criterion(generated_music, targets)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.learning_rate)
        return optimizer
