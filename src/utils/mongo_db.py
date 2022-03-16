import gridfs
from pymongo import MongoClient


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
