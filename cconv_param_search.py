import argparse
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# import model and lightning moule
from src.model.continuous_conv.cconv_lightning_graph import ConCov
from src.model.continuous_conv.cconv_lightning_graph import LitModel
# import dataloader
from src.dataset import dataloader as DB

PERCENT_VALID_EXAMPLES = 0.1
EQUATION = "wave"
EPOCHS = 8
BATCH_SIZE = 32
LOOKBACK = 8
FEATURES_IN = 1
COORDINATE_DIM = 2
KNN_NUM = 10

def objective(trial: optuna.trial.Trial) -> float:
    # Optimize hidden cconv layers, hidden size of mlp, hidden layers of mlp
    hidden_layer = trial.suggest_int("hidden_layer", 1, 10)
    hidden_size_mlp = trial.suggest_int("hidden_size_mlp", 32, 256, log=True)
    hidden_layer_mlp = trial.suggest_int("hidden_layer_mlp", 1, 3)

    net = ConCov(k=KNN_NUM,
             hidden_mlp=hidden_size_mlp,
             input_size=LOOKBACK*FEATURES_IN,
             output_size=FEATURES_IN,
             hidden_layers=hidden_layer,
             hidden_size=LOOKBACK*FEATURES_IN,
             hl_mlp=hidden_layer_mlp)
    model = LitModel(net)
    dynabench = DB.DynaBenchDataModule(batch_size=BATCH_SIZE,
                                   equation=EQUATION, base_path="data",
                                   structure="graph",
                                   lookback=LOOKBACK,
                                   num_workers=8)

    trainer = pl.Trainer(
        logger=True,
        limit_val_batches=PERCENT_VALID_EXAMPLES,
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=1
        #callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
    )

    hyperparameters = dict(hidden_layer=hidden_layer, hidden_size_mlp=hidden_size_mlp, hidden_layer_mlp=hidden_layer_mlp)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=dynabench)

    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCONV param trial")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    parser.add_argument(
        '--equation',
        dest='eq',
        action='store',
        default="wave",
        help='specify equation for dynabench'
    )
    args = parser.parse_args()

    EQUATION = args.eq
    if EQUATION == "gas_dynamics":
        # specific changes for gas_dynamics
        FEATURES_IN = 4
        LOOKBACK = 4
    if EQUATION == "brusselator":
        # specific changes for brusselator
        FEATURES_IN = 2

    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)

    study.optimize(objective, n_trials=100, timeout=6000)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))