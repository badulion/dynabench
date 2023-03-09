import argparse
import optuna
import logging
import sys
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# import model and lightning moule
from src.model.continuous_conv.cconv_lightning_graph import ConCov
from src.model.continuous_conv.cconv_lightning_graph import LitModel
# import dataloader
from src.dataset import dataloader as DB

# Hyperparameters
PERCENT_VALID_EXAMPLES = 0.1
EQUATION = "wave"
EPOCHS = 5
LOOKBACK = 8
FEATURES_IN = 1
COORDINATE_DIM = 2
KNN_NUM = 10
BATCH_SIZE = 16

class NumParameterPruner(optuna.pruners.BasePruner):
    def __init__(self, min: int, max: int):
        self.min = min
        self.max = max
    
    def prune(self, study, trial) -> bool:
        net_size = trial.user_attrs["net_size"]
        prune=False if self.min <= net_size <= self.max else True
        return prune


def objective(trial: optuna.trial.Trial) -> float:
    # Optimize hidden cconv layers, hidden size of mlp, hidden layers of mlp
    hidden_layer = trial.suggest_int("hidden_layer", 1, 6)
    hidden_size_mlp = trial.suggest_int("hidden_size_mlp", 32, 256, log=True)
    hidden_layer_mlp = trial.suggest_int("hidden_layer_mlp", 1, 2)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    hidden_size_divident = trial.suggest_categorical("hidden_size_divident", [1, 2, 4])
    tanh_mlp = trial.suggest_int("tanh_mlp", 0, 1)

    net = ConCov(k=KNN_NUM,
             hidden_mlp=hidden_size_mlp,
             input_size=LOOKBACK*FEATURES_IN,
             output_size=FEATURES_IN,
             hidden_layers=hidden_layer,
             hidden_size= (LOOKBACK*FEATURES_IN)//hidden_size_divident,
             hl_mlp=hidden_layer_mlp,
             tanh_mlp=tanh_mlp)
    trial.set_user_attr("net_size", sum(p.numel() for p in net.parameters() if p.requires_grad))
    if trial.should_prune():
        raise optuna.TrialPruned()
    model = LitModel(net, weight_decay=weight_decay, learning_rate=learning_rate)
    dynabench = DB.DynaBenchDataModule(batch_size=BATCH_SIZE,
                                   equation=EQUATION, base_path="data",
                                   structure="graph",
                                   lookback=LOOKBACK,
                                   num_workers=8)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="outputs/study/tb_logs_"+EQUATION)
    csv_logger = pl_loggers.CSVLogger(save_dir="outputs/study/csv_logs_"+EQUATION)
    trainer = pl.Trainer(
        logger=[tb_logger, csv_logger],
        limit_val_batches=PERCENT_VALID_EXAMPLES,
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=1
    )

    hyperparameters = dict(hidden_layer=hidden_layer,
                           hidden_size_mlp=hidden_size_mlp,
                           hidden_layer_mlp=hidden_layer_mlp,
                           weight_decay=weight_decay,
                           learning_rate=learning_rate,
                           hidden_size_divident=hidden_size_divident
                           )
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=dynabench)

    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCONV param trial")
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
    if EQUATION == "brusselator":
        # specific changes for brusselator
        FEATURES_IN = 2
    
    pruner = NumParameterPruner(50000, 150000)
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "Continuous-Convolution-Hyperparater-search"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        direction="minimize",
        storage=storage_name,
        study_name=study_name,
        pruner=pruner
        )

    study.optimize(objective, n_trials=1000)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))