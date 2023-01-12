from pytorch_lightning import LightningDataModule
from src.dataset.dataset_base import DynaBenchBase
from src.dataset.dataset_graph import DynaBenchGraph
from torch.utils.data import DataLoader as DataLoaderBase
from torch_geometric.loader import DataLoader as DataLoaderGraph



class DynaBenchDataModule(LightningDataModule):
    def __init__(self,
                name="dyna-benchmark",
                equation="gas_dynamics",
                task="forecast",
                support="high",
                base_path="data",
                structure="points",
                lookback=1,
                rollout=1,
                test_ratio=0.1,
                val_ratio=0.1,
                k=10,
                batch_size=16,
                num_workers=16,
                *args,
                **kwargs) -> None:
        super().__init__()
        self.name=name
        self.equation=equation
        self.task=task
        self.support=support
        self.base_path=base_path
        self.lookback=lookback
        self.rollout=rollout
        self.test_ratio=test_ratio
        self.val_ratio=val_ratio
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.structure=structure
        self.k=k

        self.Dataset = DynaBenchGraph if structure == "graph" else DynaBenchBase
        self.Dataloader = DataLoaderGraph if structure == "graph" else DataLoaderBase


    def setup(self, stage: str):
        self.train = self.Dataset(name=self.name, mode="train", equation=self.equation, support=self.support, task=self.task, base_path=self.base_path, lookback=self.lookback, rollout=1, test_ratio=self.test_ratio, val_ratio=self.val_ratio, k=self.k)
        self.val = self.Dataset(name=self.name, mode="val", equation=self.equation, support=self.support, task=self.task, base_path=self.base_path, lookback=self.lookback, rollout=self.rollout, test_ratio=self.test_ratio, val_ratio=self.val_ratio, k=self.k)
        self.test = self.Dataset(name=self.name, mode="test", equation=self.equation, support=self.support, task=self.task, base_path=self.base_path, lookback=self.lookback, rollout=self.rollout, test_ratio=self.test_ratio, val_ratio=self.val_ratio, k=self.k)


    def train_dataloader(self):
        return self.Dataloader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return self.Dataloader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return self.Dataloader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)

    