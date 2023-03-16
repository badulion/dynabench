from pytorch_lightning import LightningDataModule
from .dataset_base import DynaBenchBase
from .dataset_graph import DynaBenchGraph
from torch.utils.data import DataLoader as DataLoaderBase
from torch_geometric.loader import DataLoader as DataLoaderGraph
from .datapipe import create_datapipes

class DynaBenchDataModule(LightningDataModule):
    def __init__(self,
                name="dyna-benchmark",
                equation="gas_dynamics",
                support="cloud",
                num_points="high",
                base_path="data",
                structure="torch",
                lookback=1,
                rollout=1,
                k=10,
                batch_size=16,
                num_workers=0,
                *args,
                **kwargs) -> None:
        super().__init__()

        self.name=name
        self.equation=equation
        self.support=support
        self.num_points=num_points
        self.base_path=base_path
        self.lookback=lookback
        self.rollout=rollout
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.structure=structure
        self.k=k

        self.Dataloader = DataLoaderGraph if self.structure == "graph" else DataLoaderBase


    def setup(self, stage: str):
        self.train = create_datapipes(
            base_path=self.base_path,
            split="train",
            equation=self.equation,
            support=self.support, 
            num_points=self.num_points,
            lookback=self.lookback, 
            rollout=1,
            as_graph=(self.structure == "graph"),
            k = self.k
        )
        self.val = create_datapipes(
            base_path=self.base_path,
            split="val",
            equation=self.equation,
            support=self.support, 
            num_points=self.num_points,
            lookback=self.lookback, 
            rollout=1,
            as_graph=(self.structure == "graph"),
            k = self.k
        )
        self.test = create_datapipes(
            base_path=self.base_path,
            split="test",
            equation=self.equation,
            support=self.support, 
            num_points=self.num_points,
            lookback=self.lookback, 
            rollout=self.rollout,
            as_graph=(self.structure == "graph"),
            k = self.k
        )


    def train_dataloader(self):
        return self.Dataloader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return self.Dataloader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        return self.Dataloader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)

    