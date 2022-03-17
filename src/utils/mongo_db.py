import io
from pathlib import Path

import gridfs
from pymongo import MongoClient

from src.utils import get_logger

log = get_logger(__name__)


class MongoDatabase:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    def connect(self, sanity_check=False):
        client = MongoClient(self.connection_string)
        db = client.acc
        return db.dl_template_sanity_checks if sanity_check else db.dl_template

    def connect_with_gridfs(self):
        client = MongoClient(self.connection_string)
        db = client.acc
        return gridfs.GridFS(db)

    @staticmethod
    def insert_dict(db, d: dict, _id: str) -> None:
        db.find_one_and_update({"_id": _id}, {"$set": d})

    def setup_experiment_entry(
            self, checkpoint_path: str, experiment_uid: str, sanity_check: bool
    ) -> None:
        """Create an entry in the database for the experiment."""
        experiments_db = self.connect(sanity_check)
        experiments_db.insert_one(
            {
                "_id": experiment_uid,
                "checkpoints_path": checkpoint_path,
                "epoch_results": {},
            }
        )

    def save_checkpoint_to_db(self, _id: str, checkpoint_fn: Path):
        """
        Inspired from https://medium.com/naukri-engineering/way-to-store-large-deep-learning-models-in-production-ready-environments-d8a4c66cc04c
        There is probably a better way to store Tensors in MongoDB.
        """
        if checkpoint_fn.exists():
            fs = self.connect_with_gridfs()

            with io.FileIO(str(checkpoint_fn), "r") as fileObject:
                log.info(f"Saving checkpoint to db: {checkpoint_fn}")
                fs.put(fileObject, filename=str(checkpoint_fn), _id=_id)

    def get_last_experiment_uid(self):
        db = self.connect()
        return db.find_one({}, sort=[('_id', -1)])['_id']

    def get_last_checkpoint(self):
        fs = self.connect_with_gridfs()
        return fs.find_one({}, sort=[('_id', -1)])

    def load_checkpoint_from_db(self, out_fn: Path, uid: str):
        fs = self.connect_with_gridfs()

        with open(out_fn, 'wb') as fileobject:
            fileobject.write(fs.get(uid).read())
