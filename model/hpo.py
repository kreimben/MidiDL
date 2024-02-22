import lightning as L
import optuna
import torch
from optuna.integration import PyTorchLightningPruningCallback
from torch.utils.data import DataLoader

from midi2vec.train import load_trained_node
from model.dataset import Node2vecDataset
from model.lightning_module import MusicGeneratorModel


def objective(trial):
    # Define the hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    hidden_dim1 = trial.suggest_int("hidden_dim", 32, 512)
    hidden_dim2 = trial.suggest_int("hidden_dim", 32, 512)

    # Instantiate the model with the suggested hyperparameters
    model = MusicGeneratorModel(hidden_size1=hidden_dim1, hidden_size2=hidden_dim2, learning_rate=learning_rate)

    # Define PyTorch Lightning trainer
    trainer = L.Trainer(
        max_epochs=10,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="loss")]
    )

    _, G = load_trained_node()

    # Convert your trained node2vec embedding to a dataset
    node2vec_dataset = Node2vecDataset(G)  # replace `labels` with your actual labels
    dataloader = DataLoader(node2vec_dataset, batch_size=64, shuffle=True)

    # Train the model
    trainer.fit(model, dataloader)

    return trainer.callback_metrics["train_loss"].item()


def hpo():
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
