import io
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
import torch

from src.utils import get_logger
from src.utils.mongo_db import MongoDatabase
from src.utils.utils import get_callback

log = get_logger(__name__)


# Convert time representation
def to_hms(time):
    h = int(time // 3600)
    m = int((time - h * 3600) // 60)
    s = int(time - h * 3600 - m * 60)
    if h > 0:
        line = "{}h{:02d}m".format(h, m)
    else:
        line = "{}m{:02d}s".format(m, s)

    return line


class DBLogger(pl.Callback):
    def __init__(self, mongodb_connection_string: str):
        super().__init__()
        self.train_epoch_start_time = None

        self.db = MongoDatabase(mongodb_connection_string)

        self.best_AUROC = 0.0
        self.bestAUROCEpoch = 0

    def on_init_start(self, trainer: "pl.Trainer") -> None:
        self.train_start_time = time.time()

        dateTimeObj = datetime.now()
        dateStr = dateTimeObj.strftime("%Y_%m_%d_%H_%M_%S_%f")
        self.experiment_uid = f"{dateStr}"

        self.db.setup_experiment_entry(
            checkpoint_path=get_callback(trainer.callbacks, "ModelCheckpoint").dirpath,
            experiment_uid=self.experiment_uid,
            sanity_check=trainer.sanity_checking,
        )

    def store_hparams(self, trainer, hparams: dict):
        experiments_db = self.db.connect(sanity_check=trainer.sanity_checking)
        self.db.insert_dict(db=experiments_db, d={"config": hparams}, _id=self.experiment_uid)

    def on_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_epoch_start_time = time.time()

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if any(k.startswith("val") for k in trainer.logged_metrics):
            experiments_db = self.db.connect(sanity_check=trainer.sanity_checking)
            # add epoch results to db
            epoch_results = experiments_db.find_one({"_id": self.experiment_uid})["epoch_results"]
            epoch_results[f"{trainer.current_epoch}"] = {
                "train_results": {
                    k.replace("train.", ""): v.cpu().item() if type(v) == torch.Tensor else v
                    for k, v in trainer.logged_metrics.items()
                    if k.startswith("train")
                },
                "eval_results": {
                    k.replace("val.", ""): v.cpu().item() if type(v) == torch.Tensor else v
                    for k, v in trainer.logged_metrics.items()
                    if k.startswith("val")
                },
                "epoch_time": time.time() - self.train_epoch_start_time,
            }

            self.db.insert_dict(
                db=experiments_db, d={"epoch_results": epoch_results}, _id=self.experiment_uid
            )

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_start_time = time.time()

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        total_time_taken = time.time() - self.train_start_time
        print("Execution finished in: {}".format(to_hms(total_time_taken)))

        experiments_db = self.db.connect(sanity_check=trainer.sanity_checking)
        # add final results to db
        self.db.insert_dict(
            db=experiments_db,
            d={
                "experiment_duration": total_time_taken,
                "learning_rates": get_callback(trainer.callbacks, "LearningRateMonitor").lrs,
            },
            _id=self.experiment_uid,
        )

        trained_model_path = get_callback(trainer.callbacks, "ModelCheckpoint").best_model_path
        pl_module.trained_model_path = trained_model_path
        self.db.insert_dict(
            db=experiments_db,
            d={"trained_model_path": trained_model_path},
            _id=self.experiment_uid,
        )

        if not trainer.fast_dev_run:
            self.db.save_checkpoint_to_db(_id=self.experiment_uid, checkpoint_fn=Path(trained_model_path))

    def __repr__(self):
        return "DBLogger"
