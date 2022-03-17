import os
import tempfile
from pathlib import Path

import dotenv
from hydra import compose, initialize

from src import utils
from src.testing_pipeline import test
from src.training_pipeline import train
from src.utils.mongo_db import MongoDatabase

dotenv.load_dotenv(override=True)


def load_db_checkpoint():
    initialize(config_path="../../configs/", job_name="test_app")
    train_config = compose(config_name="train",
                           overrides=["trainer.max_epochs=1", "hydra.sweep.subdir=1"],
                           return_hydra_config=True)

    # Applies optional utilities
    utils.extras(train_config)

    # Train model
    train(train_config)

    mongo_db = MongoDatabase(os.getenv('MONGODB_CONNECTION_STR'))
    last_experiment_uid = mongo_db.get_last_experiment_uid()

    with tempfile.TemporaryDirectory() as tmpdirname:
        checkpoint_fn = Path(tmpdirname) / 'last_checkpoint.ckpt'
        mongo_db.load_checkpoint_from_db(checkpoint_fn, last_experiment_uid)

        test_config = compose(config_name="test",
                              overrides=[f"ckpt_path={checkpoint_fn}", "hydra.sweep.subdir=1"],
                              return_hydra_config=True)

        # Applies optional utilities
        utils.extras(test_config)

        # Evaluate model
        test(test_config)


if __name__ == '__main__':
    load_db_checkpoint()
